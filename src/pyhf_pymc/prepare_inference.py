import numpy as np
import pyhf
import pymc as pm
from pytensor import tensor as pt


def prepare_priors(model, unconstr_dict):
    """

    """

    norm_poiss_dict = {}
    ii = 0
    for k,v in model.config.par_map.items():
            if isinstance(v['paramset'], pyhf.parameters.constrained_by_normal):
                a, b  = [], []

                for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                    a.append(model.config.auxdata[int(i)])
                    b.append(model.constraint_model.constraints_gaussian.sigmas[ii])
                    ii = ii + 1
                norm_poiss_dict[k] = {'type': 'normal', 'input': [a, b]}

    ## Add poisson priors to dictionary
    for k,v in model.config.par_map.items():
        if isinstance(v['paramset'], pyhf.parameters.constrained_by_poisson):
            a = []
            for i in model.constraint_model.viewer_aux.selected_viewer._partition_indices[model.config.auxdata_order.index(k)]:
                a.append(model.config.auxdata[int(i)]**3)
            norm_poiss_dict[k] = {'type': 'poisson', 'input': [a, a]}

    return {**unconstr_dict, **norm_poiss_dict}

def get_target(model):
    """
    Ordering vector for the parameters
    Input:
        - pyhf model
    Output:
        - index vector
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



def prepare_model(model, observations, prior_dict):
    """
    Preparing model for sampling
    Input:
        - pyhf model
        - observarions
        - dictionary of priors
    Output:
        - dictinonary of the model with keys 'model', 'obs', 'priors'
    """

    model_dict = {}
    model_dict['model'] = model
    model_dict['obs'] = observations
    model_dict['priors'] = prior_dict

    return model_dict

def priors2pymc(prepared_model):
    """

    """
    unconstr_norm1, unconstr_norm2 = [], []
    unconstr_poiss1, unconstr_poiss2 = [], []
    unconstr_pars, norm_pars, poiss_pars = [], [], []
    norm_mu, norm_sigma = [], []
    poiss_alpha, poiss_beta = [], []
    model = prepared_model['model']
    obs = prepared_model['obs']
    prior_dict = prepared_model['priors']

    with pm.Model():

        for key in prior_dict.keys():
            sub_dict = prior_dict[key]

        ## Unconstrained
            if sub_dict['type'] == 'unconstrained_normal':
                unconstr_norm1.append(sub_dict['input'][0])
                unconstr_norm2.append(sub_dict['input'][1])
            if sub_dict['type'] == 'unconstrained_poisson':
                unconstr_poiss1.append(sub_dict['input'][0])
                unconstr_poiss2.append(sub_dict['input'][1])

        ## Normal and Poisson constraints
            if sub_dict['type'] == 'normal':
                norm_mu.append(sub_dict['input'][0])
                norm_sigma.append(sub_dict['input'][1])

            if sub_dict['type'] == 'poisson':
                poiss_alpha.append(sub_dict['input'][0])
                poiss_beta.append(sub_dict['input'][1])

        if np.array(unconstr_poiss1, dtype=object).size != 0:
            unconstr_pars.extend(pm.Gamma('Unconstrained', alpha=list(np.concatenate(unconstr_poiss1)), beta=list(np.concatenate(unconstr_poiss2))))
        
        if np.array(unconstr_norm1, dtype=object).size != 0:
            unconstr_pars.extend(pm.Normal('Unconstrained', mu=list(np.concatenate(unconstr_norm1)), sigma=list(np.concatenate(unconstr_norm2))))

        if np.array(norm_mu, dtype=object).size != 0:
            norm_pars.extend(pm.Normal('Normals', mu=list(np.concatenate(norm_mu)), sigma=list(np.concatenate(norm_sigma))))

        if np.array(poiss_alpha, dtype=object).size != 0:
            poiss_pars.extend(pm.Gamma('Gammas', alpha=list(np.concatenate(poiss_alpha)), beta=list(np.concatenate(poiss_beta))))

        pars = []
        for i in [unconstr_pars, norm_pars, poiss_pars]:
            i = np.array(i)
            if i.size != 0:
                pars.append(i)
        pars = np.concatenate(pars)
        target = get_target(model)
        final = pt.as_tensor_variable(pars[target.argsort()].tolist())

        return final