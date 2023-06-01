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
    constrained_priors = {}

    ## Add Normal priors
    sigma_counter = 0
    for k,v in model.config.par_map.items():
            if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal):

                mu, sigma  = [], []
                for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                    mu.append(model.config.auxdata[int(i)])
                    sigma.append(model.constraint_model.constraints_gaussian.sigmas[sigma_counter])
                    sigma_counter += 1

                constrained_priors[k] = {'type': 'normal', 'input': [mu, sigma]}

    ## Add Poisson priors
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
            alpha = []
            for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                alpha.append(model.config.auxdata[int(i)]**3)
                beta = alpha
            constrained_priors[k] = {'type': 'poisson', 'input': [alpha, beta]}


    prior_dict = {**unconstr_priors, **constrained_priors}

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
    Returns:
        - target (list): Specifies the position index for each parameter.
    """
    unconstr_halfnorm = []
    unconstr_poiss_alpha, unconstr_poiss_beta = [], []
    unconstr_pars_HN, unconstr_pars_Poiss, norm_pars, poiss_pars = [], [], [], []
    norm_mu, norm_sigma = [], []
    poiss_alpha, poiss_beta = [], []
    # model = prepared_model['model']
    # obs = prepared_model['obs']
    # prior_dict = prepared_model['priors']

    with pm.Model():

        for key in prior_dict.keys():
            sub_dict = prior_dict[key]

        ## Unconstrained
            if sub_dict['type'] == 'unconstrained_halfnormal':
                unconstr_halfnorm.append(sub_dict['input'][0])
            if sub_dict['type'] == 'unconstrained_poisson':
                unconstr_poiss_alpha.append(sub_dict['input'][0])
                unconstr_poiss_beta.append(sub_dict['input'][1])

        ## Normal and Poisson constraints
            if sub_dict['type'] == 'normal':
                norm_mu.append(sub_dict['input'][0])
                norm_sigma.append(sub_dict['input'][1])

            if sub_dict['type'] == 'poisson':
                poiss_alpha.append(sub_dict['input'][0])
                poiss_beta.append(sub_dict['input'][1])

        ##
        if np.array(unconstr_halfnorm, dtype=object).size != 0:
            unconstr_pars_HN.extend(pm.HalfNormal('Unconstrained_HalfNormal', sigma=list(np.concatenate(unconstr_halfnorm))))
        if np.array(unconstr_halfnorm, dtype=object).size != 0:
            unconstr_pars_Poiss.extend(pm.Gamma('Unconstrained_Gamma', alpha=list(np.concatenate(unconstr_poiss_alpha)), beta=list(np.concatenate(unconstr_poiss_beta))))

        if np.array(norm_mu, dtype=object).size != 0:
            norm_pars.extend(pm.Normal('Normals', mu=list(np.concatenate(norm_mu)), sigma=list(np.concatenate(norm_sigma))))

        if np.array(poiss_alpha, dtype=object).size != 0:
            poiss_pars.extend(pm.Gamma('Gammas', alpha=list(np.concatenate(poiss_alpha)), beta=list(np.concatenate(poiss_beta))))

        pars = []
        for i in [unconstr_pars_HN, unconstr_pars_Poiss, norm_pars, poiss_pars]:
            i = np.array(i)
            if i.size != 0:
                pars.append(i)
        pars = np.concatenate(pars)
        target = get_target(model)
        final = pt.as_tensor_variable(pars[target.argsort()].tolist())

        return final