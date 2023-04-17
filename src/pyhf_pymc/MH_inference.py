import jax
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from pytensor import tensor as pt

from pyhf_pymc import prepare_inference


# VJP Op with gradients
class VJPOp(Op):

    itypes = [pt.dvector,pt.dvector]  
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        (parameters, tangent_vector) = inputs
        results = jitted_vjp_expData(parameters, tangent_vector)

        # if not isinstance(results, (list, tuple)):
        #         results = (results,)
                
        # for i, r in enumerate(results):
        #     outputs[i][0] = np.asarray(r)
        outputs[0][0] = np.asarray(results)

vjp_op = VJPOp()

# Op with gradient method
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

    def grad(self, inputs, output_gradients):
        (parameters,) = inputs
        (tangent_vector,) = output_gradients
        return [vjp_op(parameters, tangent_vector)]

        
expData_op = ExpDataOp()

def sampling(prepared_model, n_samples):
    """
    Sampling
    Input:
        - the prepared model dictionary
    Output:
        - post_data, post_pred, prior_pred
    """
    model = prepared_model['model']
    obs = prepared_model['obs']
    prior_dict = prepared_model['priors']
    precision = prepared_model['precision']

    with pm.Model():

        final = prepare_inference.priors2pymc(prepared_model)

        mu = ExpDataOp('mainOp', jax.jit(model.expected_actualdata))

        ExpData = pm.Normal('ExpData', mu=mu(final), sigma=precision, observed=obs)
        post_data = pm.sample(n_samples)
        post_pred = pm.sample_posterior_predictive(post_data)
        prior_pred = pm.sample_prior_predictive(n_samples)

        return post_data, post_pred, prior_pred


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
