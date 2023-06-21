import numpy as np
import pyhf
import pymc as pm
from pytensor import tensor as pt

def build_priorDict_conjugate(model, unconstr_priors):
    """
    Builds a combined dictionary of constrained parameters (from the model definition) and 
    unconstrained parameters (have to be submitted by hand).

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

    # Turn partiotion indices to ints
    partition_indices = []
    for array in model.constraint_model.viewer_aux.selected_viewer._partition_indices:
        array = [int(x) for x in array]
        partition_indices.append(array)

    prior_dict = {}
    sigma_counter = 0

    for key, specs in model.config.par_map.items():

        if isinstance(specs['paramset'], pyhf.parameters.constrained_by_normal):
            prior_dict[key] = {}
            prior_dict[key]['type'] = 'Normal'
            prior_dict[key]['mu'] = np.array(model.config.auxdata)[partition_indices[model.config.auxdata_order.index(key)]]
            
            sigma = []
            for i in partition_indices[model.config.auxdata_order.index(key)]:
                sigma.append(model.constraint_model.constraints_gaussian.sigmas[sigma_counter])
                sigma_counter += 1
            prior_dict[key]['sigma'] = sigma
    
        if isinstance(specs['paramset'], pyhf.parameters.constrained_by_poisson):
            prior_dict[key] = {}
            prior_dict[key]['type'] = 'Gamma'
            prior_dict[key]['alpha_beta'] = (np.array(model.config.auxdata)[partition_indices[model.config.auxdata_order.index(key)]])**3
        
        if key in unconstr_priors.keys():
            prior_dict[key] = unconstr_priors[key]

    # Test
    try:
        assert prior_dict.keys() == model.config.par_map.keys()
    except:
        raise ValueError('Order of parameters is different from model.config.par_map.')

    return prior_dict


def priors2pymc(model, prior_dict):
    """
    Creates a unique pytensor pdf for each parameter. 

    Args:
        - model: pyhf model.
        - prior_dict (dictionary): Dictionary with all parameter priors.
    Returns:
        - final (list): pytensor.tensor distributions for each parameter.
    """

    pars_combined = []

    with pm.Model():
        for name, specs in  prior_dict.items():

            # Unconstrained
            if specs['type'] == 'Beta_Unconstrained':
                pars_combined.extend(pm.Beta(name, alpha=specs['alpha'], beta=specs['beta']))
            
            if specs['type'] == 'Cauchy_Unconstrained':
                pars_combined.extend(pm.Beta(name, alpha=specs['alpha'], beta=specs['beta']))

            if specs['type'] == 'ExGaussian_Unconstrained':
                pars_combined.extend(pm.ExGaussian(name, mu=specs['mu'], sigma=specs['sigma'], nu=specs['nu']))

            if specs['type'] == 'Exponential_Unconstrained':
                pars_combined.extend(pm.Exponential(name, lam=specs['lam']))

            if specs['type'] == 'Gamma_Unconstrained':
                pars_combined.extend(pm.Gamma(name, alpha=specs['alpha'], beta=specs['beta']))
            
            if specs['type'] == 'HalfNormal_Unconstrained':
                pars_combined.extend(pm.HalfNormal(name, sigma=specs['sigma'])) 

            if specs['type'] == 'InverseGamma_Unconstrained':
                pars_combined.extend(pm.InverseGamma(name, alpha=specs['alpha'], beta=specs['beta']))

            if specs['type'] == 'Laplace_Unconstrained':
                pars_combined.extend(pm.Laplace(name, mu=specs['mu'], b=specs['b']))

            if specs['type'] == 'Logistic_Unconstrained':
                pars_combined.extend(pm.Logistic(name, mu=specs['mu'], s=specs['scale']))

            if specs['type'] == 'LogNormal_Unconstrained':
                pars_combined.extend(pm.LogNormal(name, mu=specs['mu'], sigma=specs['sigma']))

            if specs['type'] == 'Normal_Unconstrained':
                pars_combined.extend(pm.Normal(name, mu=specs['mu'], sigma=specs['sigma']))
                            
            if specs['type'] == 'Uniform_Unconstrained':
                pars_combined.extend(pm.Uniform(name, lower=specs['lower'], upper=specs['upper']))
            
            # Constrained
            if specs['type'] == 'Normal':
                pars_combined.extend(pm.Normal(name, mu=specs['mu'], sigma=specs['sigma']))
            
            if specs['type'] == 'Gamma':
                pars_combined.extend(pm.Gamma(name, alpha=specs['alpha_beta'], beta=specs['alpha_beta']))

    
    # Test
    try:
        assert len(pars_combined) == len(model.config.suggested_init())
    except:
        raise ValueError('Number of parameters is incorrect.')

    pars_combined = pt.as_tensor_variable(pars_combined)
       
    return pars_combined


