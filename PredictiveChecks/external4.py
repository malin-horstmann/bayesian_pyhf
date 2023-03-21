"""
left to fix: 
    - 
"""

import numpy as np
import json
import pytensor 
import pymc as pm
import jax
import jax.numpy as jnp
import pyhf
pyhf.set_backend('jax')

from jax import grad, jit, vmap, value_and_grad, random
from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph import Apply, Op
from contextlib import contextmanager

class ExpDataClass(pt.Op):
    """
    Input: 
        - name
        - func (model.expected_actualdata())
    Output: 
        - Object that can take tensor_variables as input and returns the value of func
    """
    itypes = [pt.dvector]  
    otypes = [pt.dvector]  

    def __init__(self, name, func):
        ## Add inputs as class attributes
        self.func = func
        self.name = name

    def perform(self, node, inputs, outputs):
        ## Method that is used when calling the Op
        (theta,) = inputs  # Contains my variables

        ## Calling input function (in our case the model.expected_actualdata)
        result = self.func(theta)

        ## Output values of model.expected_actualdata
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)


def prepare_priors(model, unconstr_dict):
    """
    Preparing priors for model preparation
    Input: 
        - pyhf model
        - dictionary of all unconstrained parameters 
    Output: 
        - dictionary with all parameters (unconstrained, normal, poisson)
    """
    target, indices = [], []
    unconstr_mu, unconstr_sigma, norm_mu, norm_sigma, poiss_pars = [], [], [], [], []
    unconstr_idx, norm_idx, poiss_idx = [], [], []

    unconstr_input = [1, 0.1]

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
    ## Stitching order
    target, indices = [], []
    unconstr_mu, unconstr_sigma, norm_mu, norm_sigma, poiss_pars = [], [], [], [], []
    unconstr_idx, norm_idx, poiss_idx = [], [], []

    unconstr_input = [1, 0.1]

    norm_poiss_dict = {}

    for k, v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.unconstrained):
            unconstr_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.unconstrained)
                ])
            break

    for k, v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.paramsets.constrained_by_normal):
            norm_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal)
                ])
            break

    for k, v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
            poiss_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson)
                ])
            break

    for i in [unconstr_idx, norm_idx, poiss_idx]:
        i = np.array(i)
        if i.size != 0:
            indices.append(i)
    target = np.concatenate(indices)

    return target



def prepare_model(model, observations, priors, precision):
    """
    Preparing model for sampling
    Input: 
        - 
    Output: 
        - 
    """

    model_dict = {}
    model_dict['model'] = model
    model_dict['obs'] = observations
    model_dict['priors'] = priors
    model_dict['precision'] = precision

    return model_dict


def sampling(prepared_model):
    """
    Sampling
    Input: 
        - 
    Output: 
        - 
    """
    unconstr_pars, norm_pars, poiss_pars = [], [], []
    norm_mu, norm_sigma = [], []
    poiss_alpha, poiss_beta = [], []
    model = prepared_model['model']
    obs = prepared_model['obs']
    prior_dict = prepared_model['priors']
    with pm.Model():
        
        ## Unconstrained
        for key in prior_dict.keys():
            sub_dict = prior_dict[key]

            if sub_dict['type'] == 'unconstrained':
                unconstr_pars.extend(pm.Normal('Unconstrained', mu=sub_dict['input'][0], sigma=sub_dict['input'][1]))
            break

        ## Normal and Poisson constraints
        for key in prior_dict.keys():
            sub_dict = prior_dict[key]
            
            if sub_dict['type'] == 'normal':
                norm_mu.append(sub_dict['input'][0])
                norm_sigma.append(sub_dict['input'][1])
            
            if sub_dict['type'] == 'poisson':
                poiss_alpha.append(sub_dict['input'][0])
                poiss_beta.append(sub_dict['input'][1])

        if np.array(norm_mu).size != 0:
            norm_pars.extend(pm.Normal('Normals', mu=list(np.concatenate(norm_mu)), sigma=list(np.concatenate(norm_sigma))))

        if np.array(poiss_alpha).size != 0:
            poiss_pars.extend(pm.Gamma('Gammas', alpha=list(np.concatenate(poiss_alpha)), beta=list(np.concatenate(poiss_beta))))

        pars_list = [unconstr_pars, norm_pars, poiss_pars]
        pars = []
        for i in pars_list:
            i = np.array(i)
            if i.size != 0:
                pars.append(i)
        pars = np.concatenate(pars)
        target = external4.get_target(model)
        final = pt.as_tensor_variable(pars[target.argsort()].tolist())
        
        mainOp = external4.ExpDataClass('mainOp', jax.jit(model.expected_actualdata))

        main = pm.Normal('main', mu=mainOp(final), observed=obs)
        post_data = pm.sample(500)
        post_pred = pm.sample_posterior_predictive(post_data)
        prior_pred = pm.sample_prior_predictive(500)

        return post_data, post_pred, prior_pred