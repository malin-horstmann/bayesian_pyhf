import numpy as np
import math
import pyhf
import pymc as pm
from pytensor import tensor as pt


def get_gammaPostHyperpars(aux_obs, ur_alpha, ur_beta):
    '''
    Returns the (normal) posterior hyperparameters according to the rules of conjufate priors for normal models and ur-priors.
    Args:
        - aux_mu (array): mean hyperparameter of the data-generating model
        - aux_sigma (array): sigma hyperparameter of the data-generating model
        - aux_obs (array): auxiliary data
        - ur_mu (array): mean hyperparameter of the ur-prior distributuion
        - ur_sigma (array): sigma hyperparameter of the ur-prior distributuion
    Returns:
        - alpha (array): alpha hyperparameter
        - beta (array): beta hyperparameter
    '''
    alpha = ur_alpha + aux_obs
    beta = aux_obs*(ur_beta + 1)

    return alpha, beta

def get_normalPostHyperpars(aux_mu, aux_sigma, aux_obs, ur_mu, ur_sigma):
    '''
    Returns the (normal) posterior hyperparameters according to the rules of conjufate priors for normal models and ur-priors.
    Args:
        - aux_mu (array): mean hyperparameter of the data-generating model
        - aux_sigma (array)): sigma hyperparameter of the data-generating model
        - aux_obs (array): auxiliary data
        - ur_mu (array): mean hyperparameter of the ur-prior distributuion
        - ur_sigma (array): sigma hyperparameter of the ur-prior distributuion
    Returns:
        - mu (array): hyperparameter mean
        - sigma (array): hyperparameter sigma
    '''
    
    var = (aux_sigma**2 * ur_sigma**2) / (aux_sigma**2 + ur_sigma**2)

    mu = var * ((ur_mu)/(ur_sigma**2) + (aux_obs)/(aux_sigma**2)) 

    return np.array(mu), np.array(np.sqrt(var))

def get_target(model):
    """
    Ordering list for the parameters.

    Args:
        - model: pyhf model.
    Returns:
        - target (list): Specifies the position index for each parameter.
    """

    target = []
    unconstr_idx, norm_idx, poiss_idx = [], [], []

    for k, v in model.config.par_map.items():

        if isinstance(v['paramset'], pyhf.parameters.unconstrained):
            unconstr_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.unconstrained)
                ])

        if isinstance(v['paramset'], pyhf.parameters.paramsets.constrained_by_normal):
            norm_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal)
                ])

        if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
            poiss_idx = np.concatenate([
                np.arange(v['slice'].start,v['slice'].stop) for k,v in model.config.par_map.items() if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson)
                ])

    for i in [unconstr_idx, norm_idx, poiss_idx]:
        i = np.array(i)
        if i.size != 0:
            target.append(i)
    target = np.concatenate(target)

    return target