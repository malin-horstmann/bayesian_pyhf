import numpy as np
import matplotlib.pyplot as plt
import json
import pymc as pm

import pyhf
pyhf.set_backend('jax')

from jax import grad, jit, vmap, value_and_grad, random
import jax
import jax.numpy as jnp
import pytensor
from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph import Apply, Op
# from contextlib import contextmanager

def prepare_priors(model, unconstr_dict):
    """
    Preparing priors for model preparation
    Input: 
        - pyhf model
        - dictionary of all unconstrained parameters 
    Output: 
        - dictionary with all parameters (unconstrained, normal, poisson)
    """

    unconstr_mu, unconstr_sigma, norm_mu, norm_sigma, poiss_pars = [], [], [], [], []
    norm_poiss_dict = {}
    
    ## Add normal priors to dictionary
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal):
            a, b  = [], []
            for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                a.append(model.config.auxdata[int(i)])
                b.append(model.constraint_model.constraints_gaussian.sigmas[int(i)])
            norm_poiss_dict[k] = {'type': 'normal', 'input': [a, b]}

    ## Add poisson priors to dictionary
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
            a = []
            for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                a.append(model.config.auxdata[int(i)]**3)
            norm_poiss_dict[k] = {'type': 'poisson', 'input': [a, a]}

    return {**unconstr_dict, **norm_poiss_dict}

def get_target(model):
    """
    Ordering vector for the parameters
    Input: 
        - pyhf model
    Output: 
        - index vector
    """

    target = []
    unconstr_idx, norm_idx, poiss_idx = [], [], []
    norm_poiss_dict = {}

    for k, v in model.config.par_map.items():

        if isinstance(v['paramset'], pyhf.parameters.unconstrained):
            unconstr_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.unconstrained)
                ])
            pass

        if isinstance(v['paramset'], pyhf.parameters.paramsets.constrained_by_normal):
            norm_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal)
                ])
            pass

        if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
            poiss_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson)
                ])
            pass

    for i in [unconstr_idx, norm_idx, poiss_idx]:
        i = np.array(i)
        if i.size != 0:
            target.append(i)
    target = np.concatenate(target)

    return target



def prepare_model(model, observations, priors, precision):
    """
    Preparing model for sampling
    Input: 
        - pyhf model
        - observarions
        - dictionary of priors
        - model precision
    Output: 
        - dictinonary of the model with keys 'model', 'obs', 'priors', 'precision'
    """

    model_dict = {}
    model_dict['model'] = model
    model_dict['obs'] = observations
    model_dict['priors'] = priors
    model_dict['precision'] = precision

    return model_dict

def priors2pymc(prepared_model):
    input1, input2 = [], []
    unconstr_pars, norm_pars, poiss_pars = [], [], []
    norm_mu, norm_sigma = [], []
    poiss_alpha, poiss_beta = [], []
    model = prepared_model['model']
    obs = prepared_model['obs']
    prior_dict = prepared_model['priors']
    precision = prepared_model['precision']

    with pm.Model():
        
        for key in prior_dict.keys():
            sub_dict = prior_dict[key]

        ## Unconstrained
            if sub_dict['type'] == 'unconstrained':
                input1.append(sub_dict['input'][0])
                input2.append(sub_dict['input'][1])
                
            pass

        ## Normal and Poisson constraints            
            if sub_dict['type'] == 'normal':
                norm_mu.append(sub_dict['input'][0])
                norm_sigma.append(sub_dict['input'][1])
            
            if sub_dict['type'] == 'poisson':
                poiss_alpha.append(sub_dict['input'][0])
                poiss_beta.append(sub_dict['input'][1])

        if np.array(input1, dtype=object).size != 0:
            unconstr_pars.extend(pm.Normal('Unconstrained', mu=list(np.concatenate(input1)), sigma=list(np.concatenate(input2))))

        if np.array(norm_mu, dtype=object).size != 0:
            norm_pars.extend(pm.Normal('Normals', mu=list(np.concatenate(norm_mu)), sigma=list(np.concatenate(norm_sigma))))

        if np.array(poiss_alpha, dtype=object).size != 0:
            poiss_pars.extend(pm.Gamma('Gammas', alpha=list(np.concatenate(poiss_alpha)), beta=list(np.concatenate(poiss_beta))))

        pars = []
        for i in [unconstr_pars, norm_pars, poiss_pars]:
            i = np.array(i)
            if i.size != 0:
                pars.append(i)
        pars = np.concatenate(pars)
        target = get_target(model)
        final = pt.as_tensor_variable(pars[target.argsort()].tolist())

        return final