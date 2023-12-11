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

from bayesian_pyhf import prepare_inference
from bayesian_pyhf import make_op

from contextlib import contextmanager

@contextmanager
def model(stat_model, unconstrained_priors, data, ur_hyperparameters = None):
    '''
    Builds a context with the pyhf model set up as data-generating model. The priors for the constrained parameters 
    have already been updated using conjugate priors.

    Args:
        - stat_model: pyhf model.
        - unconstrained_priors (dictionary): Dictionary of all unconstrained priors.
        - data (list or array): Observations used for the inference step.
    Returns:
        - model (context): Context in which PyMC methods can be used.
    '''
    priorDict = prepare_inference.build_priorDict(stat_model, unconstrained_priors, ur_hyperparameters)
    expData_op_Act = make_op.makeOp_Act(stat_model)

    with pm.Model():
        pars = prepare_inference.priors2pymc(stat_model, priorDict)
        
        Expected_Data = pm.Poisson("Expected_Data", mu=expData_op_Act(pars), observed=data)
        # Expected_Data = pm.Normal("Expected_Data", mu=expData_op_Act(pars), observed=data)
        yield
        
    return model