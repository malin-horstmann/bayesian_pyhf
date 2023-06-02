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
def model(stat_model, unconstrained_priors, data):
    '''
    
    '''
    prior_dict = prepare_inference.build_priorDict(stat_model, unconstrained_priors)
    expData_op = make_op.make_op(stat_model)

    with pm.Model():
        pars = prepare_inference.priors2pymc_combined(stat_model, prior_dict)
        Expected_Data = pm.Poisson("Expected_Data", mu=expData_op(pars), observed=data)
        yield
        
    return model