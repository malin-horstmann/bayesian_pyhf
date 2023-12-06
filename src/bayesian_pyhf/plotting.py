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
# pyhf.set_backend('jax')
# pyhf.set_backend('numpy')

import pymc as pm
import arviz as az

from bayesian_pyhf import prepare_inference
from bayesian_pyhf import make_op
from bayesian_pyhf import infer

blue = '#7CA1CC' # '#A8B6CC'
rosa = '#E57A77'

def prior_posterior_predictives(model, data, post_pred, prior_pred, bin_steps=1):
    '''
    
    '''
    nBins = len(model.expected_actualdata(model.config.suggested_init()))

    # Build means
    prior_means = []
    post_means = []
    for i in range(nBins):
        prior_means.append(prior_pred.prior_predictive.Expected_Data[0].T[i].mean())
        post_means.append(post_pred.posterior_predictive.Expected_Data[0].T[i].mean())

    # Plot means
    plt.scatter(np.linspace(0,nBins-1,nBins), prior_means, color=rosa, label='Prior Predictive')
    plt.scatter(np.linspace(0,nBins-1,nBins), post_means, color=blue, label='Posterior Predictive')

    # Plot samples
    for i in range(nBins):
        plt.scatter(np.full(len(prior_pred.prior_predictive.Expected_Data[0].T[i]), i), prior_pred.prior_predictive.Expected_Data[0].T[i], alpha=0.051, color=rosa, linewidths=0)
        plt.scatter(np.full(len(post_pred.posterior_predictive.Expected_Data[0].T[i]), i), post_pred.posterior_predictive.Expected_Data[0].T[i], alpha=0.051, color=blue, linewidths=0)

    # Plot data
    plt.scatter(np.arange(nBins), data, marker='P', c = 'k',s=50, zorder = 999, label = "Data")

    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, nBins, bin_steps))
    plt.xlabel('Bins')
    plt.ylabel('Events')


def calibration(model, prior_pred, prior_dict):
    '''
    
    '''
    # Sampling
    expData_op = make_op.make_op(model)
    prior_Normals, prior_Unconstrained, prior_data = np.concatenate(prior_pred.prior.Normals[0]), np.concatenate(prior_pred.prior.Gammas[0]), np.array(prior_pred.prior_predictive.Expected_Data[0])

    def posterior_from_prior(prior_data):
            with pm.Model() as m:
                    pars = prepare_inference.priors2pymc(model, prior_dict)
                    Expected_Data = pm.Poisson("Expected_Data", mu=expData_op(pars), observed=prior_data)
                    
                    step1 = pm.Metropolis()
                    post_data = pm.sample(1, chains=1, step=step1)
                    post_pred = pm.sample_posterior_predictive(post_data)

            return np.concatenate(post_data.posterior.Normals[0]), np.concatenate(post_data.posterior.Gammas[0]), np.array(post_pred.posterior_predictive.Expected_Data[0][0])
                
    post_Normals, post_Gammas, post_data = [], [], []
    for p_d in prior_data:
        a, b, c = posterior_from_prior(p_d)
        post_Normals.append(a[0])
        post_Gammas.append(b[0])
        post_data.append(c[0])

    # Plot Normals
    plt.hist(prior_Normals, 40, alpha = 0.5, color=rosa, linewidth=2, label='Prior', edgecolor=rosa)
    _, bins, _ = plt.hist(prior_Normals, bins=40, histtype='step', color=rosa, alpha=0.000001)
    plt.hist(post_Normals, bins=bins, alpha = 0.5, color=blue, linewidth=2, label='Posterior', edgecolor=blue)
    plt.xlabel('Background')

    plt.legend()

    plt.show()
        
    # Plot Unconstrained 
    plt.hist(prior_Gammas, 40, alpha = 0.5, color=rosa, linewidth=2, label='Prior', edgecolor=rosa)
    _, bins, _ = plt.hist(prior_Gammas, bins=40, histtype='step', color=rosa, alpha=0.000001)
    plt.hist(post_Gammas, bins=bins, alpha = 0.5, color=blue, linewidth=2, label='Posterior', edgecolor=blue)
    plt.xlabel('Signal Strenth')

    plt.legend()

    plt.show()

    return post_Normals, post_Unconstrained, post_data


def plot_autocorrelation(model, unconstrained_priors, data):
    '''
    
    '''
    with infer.model(model, unconstrained_priors, data):
        step = pm.Metropolis()
        post_data_MH = pm.sample(100, chains = 1, step=step)

    with infer.model(model, unconstrained_priors, data):
        step = pm.Metropolis()
        post_data_MH_thinned = pm.sample(1200, chains = 1, step=step)

    with infer.model(model, unconstrained_priors, data):
        post_data_NUTS = pm.sample(100, chains = 1)

    with infer.model(model, unconstrained_priors, data):
        post_data_NUTS_thinned = pm.sample(600, chains = 1)

    thinned_MH = post_data_MH_thinned.posterior.thin(12)
    thinned_NUTS = post_data_NUTS_thinned.posterior.thin(6)

    # Metropolis
    post_Normals_MH = np.concatenate(np.array(post_data_MH.posterior.Normals[0]))
    post_Unconstrained_MH = np.concatenate(np.array(post_data_MH.posterior.Unconstrained[0]))

    post_Normals_MH_thinned = np.concatenate(np.array(thinned_MH.Normals[0]))
    post_Unconstrained_MH_thinned = np.concatenate(np.array(thinned_MH.Unconstrained[0]))

    # NUTS
    post_Normals_NUTS = np.concatenate(np.array(post_data_NUTS.posterior.Normals[0]))
    post_Unconstrained_NUTS = np.concatenate(np.array(post_data_NUTS.posterior.Unconstrained[0]))

    post_Normals_NUTS_thinned = np.concatenate(np.array(thinned_NUTS.Normals[0]))
    post_Unconstrained_NUTS_thinned = np.concatenate(np.array(thinned_NUTS.Unconstrained[0]))

    fig = plt.bar(np.linspace(0, 100, 100), az.autocorr(post_Normals_MH), width=0.5, alpha=0.8, color=blue, label='Metropolis-Hastings')
    plt.bar(np.linspace(0, 100, 100), az.autocorr(post_Normals_NUTS), width=0.5, alpha=0.8, color=rosa, label='HMC')
    plt.fill_between(np.linspace(0, 100, 100), -0.2, 0.2, color='grey', alpha=0.2, zorder=0, linewidth=0)
    plt.xlabel('Draws')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.title('Background')
    plt.show()


    plt.bar(np.linspace(0, 100, 100), az.autocorr(post_Normals_MH_thinned), width=0.5, alpha=0.8, color=blue, label='Metropolis-Hastings, thin: 12')
    plt.bar(np.linspace(0, 100, 100), az.autocorr(post_Normals_NUTS_thinned), width=0.5, alpha=0.8, color=rosa, label='HMC, thin: 6')
    plt.fill_between(np.linspace(0, 100, 100), -0.2, 0.2, color='grey', alpha=0.2, zorder=0, linewidth=0)
    plt.xlabel('Draws')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.title('Background, thinned chains');