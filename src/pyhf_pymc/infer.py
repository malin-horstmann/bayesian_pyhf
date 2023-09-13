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

from pyhf_pymc import prepare_inference
from pyhf_pymc import prepare_inference_combined
from pyhf_pymc import make_op

from contextlib import contextmanager

@contextmanager
def model_conjugate(stat_model, unconstrained_priors, data, ur_hyperparameters = None):
    '''
    Builds a context with the pyhf model set up as data-generating model. The priors for the constrained parameters 
    have already been updated using conjugate priors.

    Args:
        - stat_model: pyhf model.
        - unconstrained_priors (dictionary): Dictionary of all unconstrained priors.
        - data (list or array): Observations used for the inference step.
    Returns:
        - model_combined (context): Context in which PyMC methods can be used.
    '''
    priorDict_conjugate = prepare_inference.build_priorDict_conjugate(stat_model, unconstrained_priors, ur_hyperparameters)
    expData_op_Act = make_op.makeOp_Act(stat_model)

    with pm.Model():
        pars_conjugate = prepare_inference.priors2pymc(stat_model, priorDict_conjugate)
        
        # Expected_Data = pm.Poisson("Expected_Data", mu=expData_op_Act(pars_conjugate), observed=data)
        Expected_Data = pm.Normal("Expected_Data", mu=expData_op_Act(pars_conjugate), observed=data)
        yield
        
    return model_conjugate

@contextmanager
def model_combined(stat_model, unconstrained_priors, data, auxdata):
    '''
    Builds a context with the pyhf model set up as data-generating model. The priors for the constrained parameters are 
    their ur-priors.

    Args:
        - stat_model: pyhf model.
        - unconstrained_priors (dictionary): Dictionary of all unconstrained priors.
        - data (list or array): Observations used for the inference step.
        - auxdata (list or array): Auxiliary observations used for the inference step
    Returns:
        - model_combined (context): Context in which PyMC methods can be used.
    '''
    priorDict_combined = prepare_inference_combined.build_priorDict_combined(stat_model, unconstrained_priors)

    expData_op_Act = make_op.makeOp_Act(stat_model)
    expData_op_Aux = make_op.makeOp_Aux(stat_model)

    with pm.Model():
        pars_combined = prepare_inference_combined.priors2pymc_combined(stat_model, priorDict_combined)

        Expected_ActData = pm.Poisson("Expected_ActData", mu=expData_op_Act(pars_combined), observed=data)
        # Normal constraints
        # Expected_AuxData = pm.Normal("Expected_AuxData", mu=expData_op_Aux(pars_combined), observed=auxdata)
        # Poisson constraints
        Expected_AuxData = pm.Poisson("Expected_AuxData", mu=expData_op_Aux(pars_combined), observed=auxdata)
        aa = pm.Poisson('Mult', mu=pars_combined, observed=auxdata)
        yield
        
    return model_combined


