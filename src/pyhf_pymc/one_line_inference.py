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

def make_op(model):
    '''
    
    '''
    @jax.jit
    def processed_expData(parameters):
        return model.expected_actualdata(parameters)
    jitted_processed_expData = jax.jit(processed_expData)

    @jax.jit
    def vjp_expData(pars, tang_vec):
        _, back = jax.vjp(processed_expData, pars)
        return back(tang_vec)[0]
    jitted_vjp_expData = jax.jit(vjp_expData)

    class VJPOp(Op):
        '''
        
        '''
        itypes = [pt.dvector,pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, tangent_vector) = inputs
            results = jitted_vjp_expData(parameters, tangent_vector)

            outputs[0][0] = np.asarray(results)

    vjp_op = VJPOp()

    class ExpDataOp(Op):
        '''
        
        '''
        itypes = [pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, ) = inputs
            results = jitted_processed_expData(parameters)

            outputs[0][0] = np.asarray(results)

        def grad(self, inputs, output_gradients):
            (parameters,) = inputs
            (tangent_vector,) = output_gradients
            return [vjp_op(parameters, tangent_vector)]

        
    expData_op = ExpDataOp()    

    return expData_op

def sampling(prepared_model, expData_op, draws, chains, step_method, tune):
    '''
    
    '''
    obs = prepared_model['obs']

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

