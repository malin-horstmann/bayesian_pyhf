import numpy as np
import pyhf
import pymc as pm
from pytensor import tensor as pt

def build_priorDict(model, unconstr_priors):
    """
    Builds a combined dictionary of constrained parameters (from the model definition) and 
    unconstrained parameters (have to be submitted by hand).

    Args:
        - model:  pyhf model.
        - unconstr_priors (dictionary): Dictionary of unconstrained parameters of the form:
            unconstr_priors = {
                'mu_poisson': {'type': 'unconstrained_poisson', 'input': [[5.], [1.]]}
                'mu_halfnormal': {'type': 'unconstrained_halfnormal', 'input': [[0.1]]}
                } 
    Returns:
        - prior_dict (dictionary): Dictionary of of all parameter priors.
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
            prior_dict[key]['alpha_beta'] = np.array(model.config.auxdata)[partition_indices[model.config.auxdata_order.index(key)]]**3
        
        if key in unconstr_priors.keys():
            prior_dict[key] = unconstr_priors[key]

    return prior_dict


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
    # norm_poiss_dict = {}

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



def priors2pymc(model, prior_dict):
    """
    Creates a pytensor object of the parameters for sampling with pyhf

    Args:
        - model: pyhf model.
        - prior_dict (dictionary): Dictionary with all parameter priors.
    Returns:
        - final (list): pt.tensor distribution for each parameter.
    """

    pars_combined = []

    with pm.Model():
        for name, specs in  prior_dict.items():

            if specs['type'] == 'HalfNormal':
                pars_combined.extend(pm.HalfNormal(name, sigma=specs['sigma']))   
            
            if specs['type'] == 'Normal':
                pars_combined.extend(pm.Normal(name, mu=specs['mu'], sigma=specs['sigma']))
            
            if specs['type'] == 'Gamma':
                pars_combined.extend(pm.Gamma(name, alpha=specs['alpha_beta'], beta=specs['alpha_beta']))
            
        return pt.as_tensor_variable(pars_combined)

def priors2pymc_combined(model, prior_dict):
    """
    Creates a pytensor object of the parameters for sampling with pyhf

    Args:
        - model: pyhf model.
        - prior_dict (dictionary): Dictionary with all parameter priors.
    Returns:
        - final (list): pt.tensor distribution for each group of parameter types (Normal, Gamma, HalfNormal).
    """
    
    return model