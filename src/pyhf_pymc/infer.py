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
pyhf.set_backend('jax')
# pyhf.set_backend('numpy')

import pymc as pm
import arviz as az

from pyhf_pymc import prepare_inference
from pyhf_pymc import make_op

from contextlib import contextmanager

@contextmanager
def model_conjugate(stat_model, unconstrained_priors, data):
    '''
    
    '''
    priorDict_conjugate = prepare_inference.build_priorDict_conjugate(stat_model, unconstrained_priors)
    expData_op_Act = make_op.makeOp_Act(stat_model)

    with pm.Model():
        pars_conjugate = prepare_inference.priors2pymc_combined(stat_model, priorDict_conjugate)
        Expected_Data = pm.Poisson("Expected_Data", mu=expData_op_Act(pars_conjugate), observed=data)
        # Expected_Data = pm.Normal("Expected_Data", mu=expData_op_Act(pars_conjugate), observed=data)
        yield
        
    return model_conjugate

@contextmanager
def model_combined(stat_model, unconstrained_priors, data, auxdata):
    '''
    
    '''
    priorDict_combined = prepare_inference.build_priorDict_combined(stat_model, unconstrained_priors)

    expData_op_Act = make_op.makeOp_Act(stat_model)
    expData_op_Aux = make_op.makeOp_Aux(stat_model)

    with pm.Model():
        pars_combined = prepare_inference.priors2pymc(stat_model, priorDict_combined)

        Expected_ActData = pm.Poisson("Expected_ActData", mu=expData_op_Act(pars_combined), observed=data)
        Expected_AuxData = pm.Normal("Expected_AuxData", mu=expData_op_Aux(pars_combined), observed=auxdata)
        yield
        
    return model_combined


@contextmanager
def model_combined_1Op(stat_model, unconstrained_priors, data):
    '''
    
    '''
    priorDict_combined = prepare_inference.build_priorDict_combined(stat_model, unconstrained_priors)

    expData_op = make_op.makeOp_ActAndAux(stat_model)

    with pm.Model():
        pars_combined = prepare_inference.priors2pymc(stat_model, priorDict_combined)

        Expected_ActData = pm.Normal("Expected_Data", mu=expData_op(pars_combined), observed=data)
        yield
        
    return model_combined_1Op

