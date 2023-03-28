"""
left to fix: 
    - 
"""

import numpy as np
import matplotlib.pyplot as plt
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


def sampling(prepared_model, n_samples):
    """
    Sampling
    Input: 
        - the prepared model dictionary
    Output: 
        - post_data, post_pred, prior_pred
    """
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
                unconstr_pars.extend(pm.Normal('Unconstrained', mu=sub_dict['input'][0], sigma=sub_dict['input'][1]))
            pass

        ## Normal and Poisson constraints            
            if sub_dict['type'] == 'normal':
                norm_mu.append(sub_dict['input'][0])
                norm_sigma.append(sub_dict['input'][1])
            
            if sub_dict['type'] == 'poisson':
                poiss_alpha.append(sub_dict['input'][0])
                poiss_beta.append(sub_dict['input'][1])

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
        
        mainOp = ExpDataClass('mainOp', jax.jit(model.expected_actualdata))

        main = pm.Normal('main', mu=mainOp(final), sigma=precision, observed=obs)
        post_data = pm.sample(n_samples)
        post_pred = pm.sample_posterior_predictive(post_data)
        prior_pred = pm.sample_prior_predictive(n_samples)

        return post_data, post_pred, prior_pred


def plot_ppc(model, plot_name, obs, post_pred, prior_pred):
    nBins = len(model.expected_actualdata(model.config.suggested_init()))
    plt.step(np.linspace(0,nBins-1,nBins),prior_pred.prior_predictive.main[0].T, alpha = 0.01, c = 'steelblue', where = 'mid');
    plt.vlines(np.arange(nBins),*np.quantile(prior_pred.prior_predictive.main[0],[.15,.85],axis=0), colors = 'steelblue', label='prior');

        ## Posterior predictive
    plt.step(np.linspace(0,nBins-1,nBins),post_pred.posterior_predictive.main[0].T, alpha = 0.01, c = 'orange', where = 'mid');
    plt.vlines(np.arange(nBins),*np.quantile(post_pred.posterior_predictive.main[0],[.15,.85],axis=0), colors = 'orange', label='posterior')

        ## Observations
    plt.scatter(np.arange(nBins), obs, c = 'k',s=12, zorder = 999, label = "data")
    plt.legend(loc='upper left')

    plt.title(f'Predictive checks for {plot_name}')

    plt.savefig(f'{plot_name}')
    plt.show()

    return 

