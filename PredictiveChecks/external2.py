"""
left to fix: 
    - super ugly u, n, p choice
    - individial choice for possibly multiple unconstrained parameters
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


def prepare_priors(model, unconstr_input):
    """
    Function that prepares the auxiliary data such that one can build priors for the bayesian inference from it.
    Input: 
        - pyhf-model, [[mu for all unconstr parameters], [sigma for all unconstr parameters]]
    Output: 
        - Target vector
        - list of mu's and sigma's for the normal priors (unconst + constr. by normal) 
        - list of a sigle value for the gamma prior
    """
    ## Stitching order
    unconstr_mu, unconstr_sigma, norm_mu, norm_sigma, poiss_pars = [], [], [], [], []
    unconstr_idx, norm_idx, poiss_idx = [], [], []

    a, b = list(model.config.par_map.values()), []
    [b.append('u') for v in a if isinstance(v['paramset'], pyhf.parameters.unconstrained)]  
    [b.append('n') for v in a if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal)]
    [b.append('p') for v in a if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson)]

    if any(a == 'u' for a in b):
        unconstr_idx = np.concatenate([
            np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.unconstrained)
            ])
    if any(a == 'n' for a in b):
            norm_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal)
            ])

    if any(a == 'p' for a in b):
            poiss_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson)
            ])

    ## This is still wonky
    if any(a == 'u' for a in b):
        target = np.array(np.concatenate([unconstr_idx]))
    if any(a == 'n' for a in b):
        target = np.array(np.concatenate([norm_idx]))
    if any(a == 'p' for a in b):
        target = np.array(np.concatenate([poiss_idx]))

    if any(a == 'u' for a in b) and any(a == 'n' for a in b):
        target = np.array(np.concatenate([unconstr_idx, norm_idx]))
    if any(a == 'u' for a in b) and any(a == 'p' for a in b):
        target = np.array(np.concatenate([unconstr_idx, poiss_idx]))
    if any(a == 'n' for a in b) and any(a == 'p' for a in b):
        target = np.array(np.concatenate([norm_idx, poiss_idx]))

    if any(a == 'u' for a in b) and any(a == 'n' for a in b) and any(a == 'p' for a in b):    
        target = np.array(np.concatenate([unconstr_idx, norm_idx, poiss_idx]))


    ## Unconstrained
    unconstr_mu.append(unconstr_input[0])
    unconstr_sigma.append(unconstr_input[1])

    ## Normals
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal):
            for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                norm_mu.append(model.config.auxdata[int(i)])
                norm_sigma.append(model.constraint_model.constraints_gaussian.sigmas[int(i)])

    ## Poissons
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
            for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                poiss_pars.append(model.config.auxdata[int(i)]**3)
                    

    return [target, unconstr_mu, unconstr_sigma, norm_mu, norm_sigma, poiss_pars]


class BayesianInference:
    """
    Class for Bayesian inference.
    Input: 
        - pyhf-model
        - observations
        - the prepared parameters 
        - the number of samples
    Output: 
        - posterior sampling, posterior predictive
        - prior predictive
    """

    def posterior_sampling(model, obs, prepared_priors, n_samples):
        a = list(model.config.par_map.values())
        b = []
        [b.append('u') for v in a if isinstance(v['paramset'], pyhf.parameters.unconstrained)]  
        [b.append('n') for v in a if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal)]
        [b.append('p') for v in a if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson)]

        with pm.Model():
            unconstr_pars, norm_pars, poiss_pars = [], [], []
            if any(a == 'u' for a in b):
                unconstr_pars.extend(pm.Normal(f'{model.config.poi_name}', mu=prepared_priors[1][0], sigma=prepared_priors[2][0]))
            if any(a == 'n' for a in b):
                norm_pars.extend(pm.Normal('Normals', mu=prepared_priors[3], sigma=prepared_priors[4]))
            if any(a == 'p' for a in b):
                poiss_pars.extend(pm.Gamma('Gamma', alpha=prepared_priors[5], beta=prepared_priors[5]))
            
            pars = np.concatenate([unconstr_pars, norm_pars, poiss_pars])

            final = pt.as_tensor_variable(pars[prepared_priors[0].argsort()].tolist())

            mainOp = ExpDataClass('mainOp', jax.jit(model.expected_actualdata))

            main = pm.Normal('main', mu=mainOp(final), observed=obs)
            post_data = pm.sample(n_samples)
            post_pred = pm.sample_posterior_predictive(post_data)
            prior_pred = pm.sample_prior_predictive(n_samples)

        return post_data, post_pred

    def prior_sampling(model, obs, prepared_priors, n_samples):
        a = list(model.config.par_map.values())
        b = []
        [b.append('u') for v in a if isinstance(v['paramset'], pyhf.parameters.unconstrained)]  
        [b.append('n') for v in a if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal)]
        [b.append('p') for v in a if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson)]

        with pm.Model():
            unconstr_pars, norm_pars, poiss_pars = [], [], []
            if any(a == 'u' for a in b):
                unconstr_pars.extend(pm.Normal(f'{model.config.poi_name}', mu=prepared_priors[1][0], sigma=prepared_priors[2][0]))
            if any(a == 'n' for a in b):
                norm_pars.extend(pm.Normal('Normals', mu=prepared_priors[3], sigma=prepared_priors[4]))
            if any(a == 'p' for a in b):
                poiss_pars.extend(pm.Gamma('Gamma', alpha=prepared_priors[5], beta=prepared_priors[5]))
            
            pars = np.concatenate([unconstr_pars, norm_pars, poiss_pars])
            print(type(pars))
            print(len(pars))

            final = pt.as_tensor_variable(pars[prepared_priors[0].argsort()].tolist())

            mainOp = ExpDataClass('mainOp', jax.jit(model.expected_actualdata))

            main = pm.Normal('main', mu=mainOp(final), observed=obs)
            prior_pred = pm.sample_prior_predictive(n_samples)

        return prior_pred

