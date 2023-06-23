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

import pymc as pm
import arviz as az

from pyhf_pymc import utils
from pyhf_pymc import make_op

from contextlib import contextmanager

def build_priorDict_combined(model, unconstr_priors):
    """
    Combined! I.e. Ur-Priors

    Args:
        - model:  pyhf model.
        - unconstr_priors (dictionary): Dictionary of unconstrained parameters of the form:
            unconstr_priors = {
                'mu_2': {'type': 'HalfNormal_Unconstrained', 'sigma': [.1]},
                'mu': {'type': 'Gamma_Unconstrained', 'alpha': [5.], 'beta': [1.]}
            }
    Returns:
        - prior_dict (dictionary): Dictionary of of all parameter priors. Next to the 'name'- and 'type'-keys, the following keys for the constrained
          parameters depend on the distribution type: Normal ('mu', 'sigma'), HalfNormal ('mu'), Gamma ('alpha_beta')
    """ 

    # Turn partition indices to ints
    partition_indices = []
    for array in model.constraint_model.viewer_aux.selected_viewer._partition_indices:
        array = [int(x) for x in array]
        partition_indices.append(array)

    prior_dict = {}
    sigma_counter = 0

    ur_mu, ur_sigma = utils.set_UrHyperPars_Normal()
    ur_alpha, ur_beta = utils.set_UrHyperPars_Gamma()

    for key, specs in model.config.par_map.items():

        if isinstance(specs['paramset'], pyhf.parameters.constrained_by_normal):
            prior_dict[key] = {}
            prior_dict[key]['type'] = 'Normal'
            # prior_dict[key]['mu'] = np.array(model.config.auxdata)[partition_indices[model.config.auxdata_order.index(key)]]
            prior_dict[key]['mu'] = np.full(len(np.array(model.config.auxdata)[partition_indices[model.config.auxdata_order.index(key)]]), ur_mu)
            
            sigma = []
            for i in partition_indices[model.config.auxdata_order.index(key)]:
                sigma.append(model.constraint_model.constraints_gaussian.sigmas[sigma_counter])
                sigma_counter += 1
            prior_dict[key]['sigma'] = np.full(len(sigma), ur_sigma)
    
        if isinstance(specs['paramset'], pyhf.parameters.constrained_by_poisson):
            prior_dict[key] = {}
            prior_dict[key]['type'] = 'Gamma'
            prior_dict[key]['alpha'] = np.full(len(partition_indices[model.config.auxdata_order.index(key)]), ur_alpha)
            prior_dict[key]['beta'] = np.full(len(partition_indices[model.config.auxdata_order.index(key)]), ur_beta)
        
        if key in unconstr_priors.keys():
            prior_dict[key] = unconstr_priors[key]

    # Test
    try:
        assert prior_dict.keys() == model.config.par_map.keys()
    except:
        raise ValueError('Order of parameters is different from model.config.par_map.')

    return prior_dict


def priors2pymc_combined(model, prior_dict):
    """
    Creates a pytensor pdf for each parameter. Alike pdfs are combined.

    Args:
        - model: pyhf model.
        - prior_dict (dictionary): Dictionary with all parameter priors.
    Returns:
        - final (list): pt.tensor distribution for each group of parameter types (Normal, Gamma, HalfNormal).
    """
    with pm.Model():
        # Constrained
        Normal_mu = [specs['mu'] for _, specs in prior_dict.items() if specs['type'] == 'Normal']
        Normal_sigma = [specs['sigma'] for _, specs in prior_dict.items() if specs['type'] == 'Normal']

        Gamma_alpha = [specs['alpha'] for _, specs in prior_dict.items() if specs['type'] == 'Gamma']
        Gamma_beta = [specs['beta'] for _, specs in prior_dict.items() if specs['type'] == 'Gamma']

        # Unconstrained
        Beta_Unconstr_alpha = [specs['alpha'] for _, specs in prior_dict.items() if specs['type'] == 'Beta_Unconstrained']
        Beta_Unconstr_beta = [specs['beta'] for _, specs in prior_dict.items() if specs['type'] == 'Beta_Unconstrained']

        Cauchy_Unconstr_alpha = [specs['alpha'] for _, specs in prior_dict.items() if specs['type'] == 'Cauchy_Unconstrained']
        Cauchy_Unconstr_beta = [specs['beta'] for _, specs in prior_dict.items() if specs['type'] == 'Cauchy_Unconstrained']

        ExGaussian_Unconstr_mu = [specs['mu'] for _, specs in prior_dict.items() if specs['type'] == 'ExGaussian_Unconstrained']
        ExGaussian_Unconstr_sigma = [specs['sigma'] for _, specs in prior_dict.items() if specs['type'] == 'ExGaussian_Unconstrained']
        ExGaussian_Unconstr_nu = [specs['nu'] for _, specs in prior_dict.items() if specs['type'] == 'ExGaussian_Unconstrained']

        Exponential_Unconstr_lam = [specs['lam'] for _, specs in prior_dict.items() if specs['type'] == 'Exponential_Unconstrained']

        Gamma_Unconstr_alpha = [specs['alpha'] for _, specs in prior_dict.items() if specs['type'] == 'Gamma_Unconstrained']
        Gamma_Unconstr_beta = [specs['beta'] for _, specs in prior_dict.items() if specs['type'] == 'Gamma_Unconstrained']
        
        HalfNormal_Unconstr_sigma = [specs['sigma'] for _, specs in prior_dict.items() if specs['type'] == 'HalfNormal_Unconstrained']

        InverseGamma_Unconstr_alpha = [specs['alpha'] for _, specs in prior_dict.items() if specs['type'] == 'InverseGamma_Unconstrained']
        InverseGamma_Unconstr_beta = [specs['beta'] for _, specs in prior_dict.items() if specs['type'] == 'InverseGamma_Unconstrained']
        
        Laplace_Unconstr_mu = [specs['mu'] for _, specs in prior_dict.items() if specs['type'] == 'Laplace_Unconstrained']
        Laplace_Unconstr_b = [specs['b'] for _, specs in prior_dict.items() if specs['type'] == 'Laplace_Unconstrained']

        LogNormal_Unconstr_mu = [specs['mu'] for _, specs in prior_dict.items() if specs['type'] == 'LogNormal_Unconstrained']
        LogNormal_Unconstr_sigma = [specs['sigma'] for _, specs in prior_dict.items() if specs['type'] == 'LogNormal_Unconstrained']

        Logistic_Unconstr_mu = [specs['mu'] for _, specs in prior_dict.items() if specs['type'] == 'Logistic_Unconstrained']
        Logistic_Unconstr_scale = [specs['scale'] for _, specs in prior_dict.items() if specs['type'] == 'Logistic_Unconstrained']

        Normal_Unconstr_mu = [specs['mu'] for _, specs in prior_dict.items() if specs['type'] == 'Normal_Unconstrained']
        Normal_Unconstr_sigma = [specs['sigma'] for _, specs in prior_dict.items() if specs['type'] == 'Normal_Unconstrained']

        Uniform_Unconstr_lower = [specs['lower'] for _, specs in prior_dict.items() if specs['type'] == 'Uniform_Unconstrained']
        Uniform_Unconstr_upper = [specs['upper'] for _, specs in prior_dict.items() if specs['type'] == 'Uniform_Unconstrained']


        # Building the PyMC distributions
        pars_combined = []
        if len(Beta_Unconstr_alpha) != 0:
            pymc_Unconstr_Betas = pm.Beta('Unconstrained_Betas', alpha=np.concatenate(Beta_Unconstr_alpha), beta=np.concatenate(Beta_Unconstr_beta))
            pars_combined.extend(pymc_Unconstr_Betas)
        if len(Cauchy_Unconstr_alpha) != 0:
            pymc_Unconstr_Cauchys = pm.Cauchy('Unconstrained_Cauchys', alpha=np.concatenate(Cauchy_Unconstr_alpha), beta=np.concatenate(Cauchy_Unconstr_beta))
            pars_combined.extend(pymc_Unconstr_Cauchys)
        if len(Exponential_Unconstr_lam) != 0:
            pymc_Unconstr_Exponentials = pm.Exponential('Unconstrained_Exponentials', lam=np.concatenate(Exponentials_Unconstr_lam))
            pars_combined.extend(pymc_Unconstr_Exponentials)
        if len(ExGaussian_Unconstr_sigma) != 0:
            pymc_Unconstr_ExGaussians = pm.ExGaussian('Unconstrained_ExGaussians', mu=np.concatenate(ExGaussian_Unconstr_mu), sigma=np.concatenate(ExGaussian_Unconstr_sigma), nu=np.concatenate(ExGaussian_Unconstr_nu))
            pars_combined.extend(pymc_Unconstr_ExGaussians)
        if len(Gamma_Unconstr_alpha) != 0:
            pymc_Unconstr_Gammas = pm.Gamma('Unconstrained_Gammas', alpha=np.concatenate(Gamma_Unconstr_alpha), beta=np.concatenate(Gamma_Unconstr_beta))
            pars_combined.extend(pymc_Unconstr_Gammas)
        if len(HalfNormal_Unconstr_sigma) != 0:
            pymc_Unconstr_HalfNormals = pm.HalfNormal('Unconstrained_HalfNormals', sigma=np.concatenate(HalfNormal_Unconstr_sigma))
            pars_combined.extend(pymc_Unconstr_HalfNormals)
        if len(InverseGamma_Unconstr_alpha) != 0:
            pymc_Unconstr_InverseGammas = pm.InverseGamma('Unconstrained_InverseGammas', alpha=np.concatenate(InverseGamma_Unconstr_alpha), beta=np.concatenate(InverseGamma_Unconstr_beta))
            pars_combined.extend(pymc_Unconstr_InverseGammas)
        if len(Laplace_Unconstr_mu) != 0:
            pymc_Unconstr_Laplaces = pm.Laplace('Unconstrained_Laplaces', mu=np.concatenate(Laplace_Unconstr_mu), b=np.concatenate(Laplace_Unconstr_b))
            pars_combined.extend(pymc_Unconstr_Laplaces)
        if len(LogNormal_Unconstr_sigma) != 0:
            pymc_Unconstr_LogNormals = pm.LogNormal('Unconstrained_LogNormals', mu=np.concatenate(LogNormal_Unconstr_mu), sigma=np.concatenate(LogNormal_Unconstr_sigma))
            pars_combined.extend(pymc_Unconstr_LogNormals)
        if len(Logistic_Unconstr_scale) != 0:
            pymc_Unconstr_Logistics = pm.Logistic('Unconstrained_Logstics', mu=np.concatenate(Logistic_Unconstr_mu), s=np.concatenate(Logistic_Unconstr_scale))
            pars_combined.extend(pymc_Unconstr_Logistics)
        if len(Normal_Unconstr_sigma) != 0:
            pymc_Unconstr_Normals = pm.Normal('Unconstrained_Normals', mu=np.concatenate(Normal_Unconstr_mu), sigma=np.concatenate(Normal_Unconstr_sigma))
            pars_combined.extend(pymc_Unconstr_Normals)
        if len(Uniform_Unconstr_upper) != 0:
            pymc_Unconstr_Uniforms = pm.Uniform('Unconstrained_Uniforms', upper=np.concatenate(Uniform_Unconstr_upper), lower=np.concatenate(Uniform_Unconstr_lower))
            pars_combined.extend(pymc_Unconstr_Uniforms)

        # Unconstrained
        if len(Normal_mu) != 0:
            pymc_Normals = pm.Normal('Normals', mu=np.concatenate(Normal_mu), sigma=np.concatenate(Normal_sigma))
            pars_combined.extend(pymc_Normals)
        if len(Gamma_alpha) != 0:  
            pymc_Gammas = pm.Gamma('Gammas', alpha=np.concatenate(Gamma_alpha), beta=np.concatenate(Gamma_beta))
            pars_combined.extend(pymc_Gammas)

        # Test
        try:
            assert len(pars_combined) == len(model.config.suggested_init())
        except:
            raise ValueError('Number of parameters is incorrect.')

        target = utils.get_target(model)
        pars_combined = pt.as_tensor_variable(np.array(pars_combined, dtype=object)[target.argsort()].tolist())

    return pars_combined

