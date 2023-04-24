import numpy as np
from random import randint
import matplotlib.pyplot as plt
import corner
import json

import pytensor
from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph import Apply, Op
from pytensor.tensor.type import TensorType

import jax
from jax import grad, jit, vmap, value_and_grad, random
import jax.numpy as jnp

import pyhf
pyhf.set_backend('jax')
# pyhf.set_backend('numpy')

import pymc as pm
import arviz as az

from pyhf_pymc import prepare_inference
from pyhf_pymc import make_op



def sampling(prepared_model, draws, chains, step_method, tune):
    '''
    
    '''
    obs = prepared_model['obs']
    model = prepared_model['model']
    expData_op = make_op.make_op(model)

    with pm.Model() as m:
        pars = prepare_inference.priors2pymc(prepared_model)

        Expected_Data = pm.Poisson("Expected_Data", mu=expData_op(pars), observed=obs)
        
        step1 = pm.Metropolis()
        
        if step_method == 'Metropolis':
            thinning = 1
            post_data = pm.sample(chains=chains, draws=thinning*draws, step=step1)
            post_pred = pm.sample_posterior_predictive(post_data)
            prior_pred = pm.sample_prior_predictive(draws)

        if step_method == 'NUTS_with_advi':
            thinning = 1
            post_data = pm.sample(chains=chains, draws=thinning*draws, init='advi', tune=tune)
            post_pred = pm.sample_posterior_predictive(post_data)
            prior_pred = pm.sample_prior_predictive(draws)

        if step_method == 'NUTS_with_jitter':
            thinning = 1
            post_data = pm.sample(chains=chains, draws=thinning*draws, tune=tune)
            post_pred = pm.sample_posterior_predictive(post_data)
            prior_pred = pm.sample_prior_predictive(draws)

        # How should I return this?
        # post_data.posterior = post_data.posterior.thin(thinning)

        return post_data, post_pred, prior_pred

