### Importing modules
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import json
import time
import pytensor 
import pymc as pm
import arviz as az
import jax
import jax.numpy as jnp
import pyhf
import scipy.stats as sps
pyhf.set_backend('jax')

from jax import grad, jit, vmap, value_and_grad, random
from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph import Apply, Op

### Class that creates the model Op which is used for model.expected_actualdata
class ExpDataClass(pt.Op):
    itypes = [pt.dvector]  # Expects a vector of parameter values
    otypes = [pt.dvector]  # Outputs a vector of values (the model.expected_actualdata)

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

### Building the sampling parameters
def build_parameters(model):
    
    ## Index order
    unconstr_idx, norm_idx, poiss_idx = [], [], []
        # Unconstrained
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.unconstrained):
            unconstr_idx = np.concatenate([np.arange(v['slice'].start,v['slice'].stop)])

    # Normal
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal):
            norm_idx = np.concatenate([np.arange(v['slice'].start,v['slice'].stop)])
    
    # Poisson
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
            poiss_idx = np.concatenate([np.arange(v['slice'].start,v['slice'].stop)])
        
    target = np.array(np.concatenate([unconstr_idx, norm_idx, poiss_idx]))

        ## Parameters
            # Unconstrained
    with pm.Model():
        unconstr_pars, norm_pars, poiss_pars = [], [], []
        unconstr_pars.extend(pm.Normal(f'{model.config.poi_name}', mu=[1], sigma=[1]))

            # Normal constraint (histosys, lumi, staterror)
        mu, sigma = [], []
        for k,v in model.config.par_map.items():
            if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal):
                for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                    mu.append(model.config.auxdata[int(i)])
                    sigma.append(model.constraint_model.constraints_gaussian.sigmas[int(i)])
            norm_pars.extend(pm.Normal('Normals', mu=mu, sigma=sigma))
        
            # Poisson constraint (shapesys)
        # alpha, beta = [], []
        # for k,v in model.config.par_map.items():
        #     if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
        #         for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
        #             alpha.append(model.config.auxdata[int(i)]**3)
        #         beta = alpha
        # poiss_pars.extend(pm.Gamma('Gammas', alpha=alpha, beta=beta))

        pars = np.concatenate([unconstr_pars, norm_pars])
        final = pt.as_tensor_variable(pars[target.argsort()].tolist())

    return final

### Creating the inference class
class InferenceModel:
    
    ## Model building
    def build_model(input_file):
        with open(input_file) as serialized:
            spec = json.load(serialized)
        workspace = pyhf.Workspace(spec)
        model = workspace.model()
        obs = workspace.data(model, include_auxdata=False)
        nBins = len(model.expected_actualdata(model.config.suggested_init()))

        return [model, obs, nBins, input_file]

    ## Posterior sampling and predictions (input is a Inference.build_model() object)
    def posterior_sampling(model, n_samples):
        mainOp = ExpDataClass('mainOp', jax.jit(model[0].expected_actualdata))

        with pm.Model():
            main = pm.Normal('main', mu=mainOp(build_parameters(model[0])), observed=model[1])
            post_data = pm.sample(n_samples)
            post_pred = pm.sample_posterior_predictive(post_data)

        return post_data, post_pred
    
    ## Prior predictions (input is a Inference.build_model() object)
    def prior_predictive(model, n_samples):
        mainOp = ExpDataClass('mainOp', jax.jit(model[0].expected_actualdata))

        with pm.Model():
            main = pm.Normal('main', mu=mainOp(build_parameters(model[0])), observed=model[1])
            prior_data = pm.sample_prior_predictive(500)

        return prior_data

    ## Plotting the result
    def plot_ppc(model, post_pred, prior_data):
        nBins = model[2]
        obs = model[1]

            ## Prior predictive
        plt.step(np.linspace(0,nBins-1,nBins),prior_data.prior_predictive.main[0].T, alpha = 0.01, c = 'steelblue', where = 'mid');
        plt.vlines(np.arange(nBins),*np.quantile(prior_data.prior_predictive.main[0],[.15,.85],axis=0), colors = 'steelblue', label='prior');

            ## Posterior predictive
        plt.step(np.linspace(0,nBins-1,nBins),post_pred.posterior_predictive.main[0].T, alpha = 0.01, c = 'orange', where = 'mid');
        plt.vlines(np.arange(nBins),*np.quantile(post_pred.posterior_predictive.main[0],[.15,.85],axis=0), colors = 'orange', label='posterior')

            ## Observations
        plt.scatter(np.arange(nBins), obs, c = 'k',s=12, zorder = 999, label = "data")
        plt.legend(loc='upper left')

        plt.title(f'Predictive checks for {model[3]}')

        plt.show()
    
    
            



    
    

