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



def make_ops(model):
    class ExpDataOp(Op):

        itypes = [pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, ) = inputs

            jitted_processed_expData = jax.jit(model.expected_actualdata)
            results = jitted_processed_expData(parameters)

            if len(outputs) == 1:
                    outputs[0][0] = np.asarray(results)
                    return
            for i, r in enumerate(results):
                    outputs[i][0] = np.asarray(r)

        def grad(self, inputs, output_gradients):
            (parameters,) = inputs
            (tangent_vector,) = output_gradients
            return [vjp_op(parameters, tangent_vector)]
        
    expData_op = ExpDataOp()

    return expData_op




############################################################
############################################################

# # Jax expected data
# @jax.jit
# def processed_expData(model, parameters):
#     return model.expected_actualdata(parameters)

# jitted_processed_expData = jax.jit(processed_expData)

# # Gradient list (dn_bins/dx_1, ..., dn_bins/dx_nPars)
# @jax.jit
# def vjp_expData(pars, tang_vec):
#     _, back = jax.vjp(processed_expData, pars)
#     return back(tang_vec)[0]

# jitted_vjp_expData = jax.jit(vjp_expData)

# class VJPOp(Op):
#     '''
    
#     '''
#     itypes = [pt.dvector,pt.dvector]  
#     otypes = [pt.dvector]

#     def perform(self, node, inputs, outputs):
#         (parameters, tangent_vector) = inputs
#         results = jitted_vjp_expData(parameters, tangent_vector)

#         # if not isinstance(results, (list, tuple)):
#         #         results = (results,)
                
#         # for i, r in enumerate(results):
#         #     outputs[i][0] = np.asarray(r)
#         outputs[0][0] = np.asarray(results)

# vjp_op = VJPOp()

# class ExpDataOp(Op):
#     '''
    
#     '''
#     itypes = [pt.dvector]  
#     otypes = [pt.dvector]

#     def perform(self, node, inputs, outputs):
#         (parameters, ) = inputs
#         results = jitted_processed_expData(parameters)

#         # if len(outputs) == 1:
#         #         outputs[0][0] = np.asarray(results)
#         #         return
#         # for i, r in enumerate(results):
#         #         outputs[i][0] = np.asarray(r)
#         outputs[0][0] = np.asarray(results)

#     def grad(self, inputs, output_gradients):
#         (parameters,) = inputs
#         (tangent_vector,) = output_gradients
#         return [vjp_op(parameters, tangent_vector)]
        
# expData_op = ExpDataOp()

# def sampling(prepared_model, n_samples, n_chains, sampling_method):
#     '''
    
#     '''

#     model = prepared_model['model']
#     obs = prepared_model['obs']
#     prior_dict = prepared_model['priors']
#     precision = prepared_model['precision']

# jitted_vjp_expData = jax.jit(vjp_expData)

# with pm.Model() as m1:
#     pars = pm.Deterministic('pars', prepare_inference.priors2pymc(prepared_model))
#     ExpData_Det = pm.Deterministic('ExpData_Det', expData_op(pars))

#     # ExpData = pm.Poisson("ExpData", mu=ExpData_Det, observed=obs)
#     ExpData = pm.Normal("ExpData", mu=ExpData_Det, sigma = precision, observed=obs)
    
#     step1 = pm.Metropolis()
#     step2 = pm.NUTS()
#     step3 = pm.HamiltonianMC()
    
#     if sampling_method == 'Metropolis':
#         post_data = pm.sample(n_samples, chains = n_chains, cores=4, step=step1)
#     if sampling_method == 'NUTS_jitter':
#         post_data = pm.sample(n_samples, chains = n_chains, cores=4)
#     if sampling_method == 'NUTS_advi':
#         post_data = pm.sample(n_samples, chains = n_chains, cores=4, init='advi')

#     post_pred = pm.sample_posterior_predictive(post_data)
#     prior_pred = pm.sample_prior_predictive(n_samples)

#     return post_data, post_pred, prior_pred

def plot_ppc(model, plot_name, obs, post_pred, prior_pred):
    nBins = len(model.expected_actualdata(model.config.suggested_init()))
    plt.step(np.linspace(0,nBins-1,nBins),prior_pred.prior_predictive.ExpData[0].T, alpha = 0.01, c = 'steelblue', where = 'mid');
    plt.vlines(np.arange(nBins),*np.quantile(prior_pred.prior_predictive.ExpData[0],[.15,.85],axis=0), colors = 'steelblue', label='prior');

    # Posterior predictive
    plt.step(np.linspace(0,nBins-1,nBins),post_pred.posterior_predictive.ExpData[0].T, alpha = 0.01, c = 'orange', where = 'mid');
    plt.vlines(np.arange(nBins),*np.quantile(post_pred.posterior_predictive.ExpData[0],[.15,.85],axis=0), colors = 'orange', label='posterior')

    # Observations
    plt.scatter(np.arange(nBins), obs, c = 'k',s=12, zorder = 999, label = "data")
    plt.legend(loc='upper left')

    plt.title(f'Predictive checks for {plot_name}')

    plt.savefig(f'{plot_name}')
    plt.show()
