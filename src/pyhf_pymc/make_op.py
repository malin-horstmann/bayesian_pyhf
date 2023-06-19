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

def makeOp_Act(model):
    '''
    Wrapping pyhf's model.expected_actualdata for PyMC.

    Args:
        - model: pyhf model.
    Returns:
        - expData_op (class): Wrapper class for model.expected_actualdata.
    '''
   
    def processed_expData(parameters):
        return model.expected_actualdata(parameters)

    class ExpDataOp(Op):
        '''
        
        '''
        itypes = [pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, ) = inputs
            results = processed_expData(parameters)

            outputs[0][0] = np.asarray(results)
       
    expData_op = ExpDataOp()    

    return expData_op
    