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


class sampling:

    def __init__(self, model):
        self.model = model
        self.inner = self.ExpDataOp()
    
    @jax.jit
    def processed_expData(self, parameters):
        return model.expected_actualdata(parameters)

    jitted_processed_expData = jax.jit(processed_expData)

    # Gradient list (dn_bins/dx_1, ..., dn_bins/dx_nPars)
    @jax.jit
    def vjp_expData(self, pars, tang_vec):
        _, back = jax.vjp(processed_expData, pars)
        return back(tang_vec)[0]
    
    class ExpDataOp(Op):

        itypes = [pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, ) = inputs
            results = jitted_processed_expData(parameters)

            # if len(outputs) == 1:
            #         outputs[0][0] = np.asarray(results)
            #         return
            # for i, r in enumerate(results):
            #         outputs[i][0] = np.asarray(r)
            outputs[0][0] = np.asarray(results)

        # def grad(self, inputs, output_gradients):
        #     (parameters,) = inputs
        #     (tangent_vector,) = output_gradients
        #     return [vjp_op(parameters, tangent_vector)]

    expData_op = ExpDataOp()

    def actual_sampling(self, prepared_model):
        obs = prepared_model['obs']
        prior_dict = prepared_model['priors']
        precision = prepared_model['precision']

           

        with pm.Model() as m:
            pars = prepare_inference.priors2pymc(prepared_model)

            # Expected_Data = pm.Poisson("Expected_Data", mu=expData_op(pars), observed=obs)
            Expected_Data = pm.Normal("Expected_Data", mu=expData_op(pars), sigma = precision, observed=obs)
            
            step1 = pm.Metropolis()
            
            post_data = pm.sample(draws=1000, chains = 1, cores=4, step=step1, )
            post_pred = pm.sample_posterior_predictive(post_data)
            prior_pred = pm.sample_prior_predictive(100)

        return post_data, post_pred, prior_pred