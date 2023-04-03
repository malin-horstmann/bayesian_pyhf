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
from contextlib import contextmanager

# import sys
# sys.path.insert(1, '/Users/malinhorstmann/Documents/pyhf_pymc/src')
import prepare_inference

####

class ExpDataClass(pt.Op):
    """
    Input: 
        - name
        - func (model.expected_actualdata())
    Output: 
        - instantiates object that can take tensor_variables as input and returns the value of func
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

        final = prepare_inference.priors2pymc(prepared_model)
        
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