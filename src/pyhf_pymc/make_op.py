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

    @jax.jit
    def processed_expData(parameters):
        return model.expected_actualdata(parameters)
    jitted_processed_expData = jax.jit(processed_expData)

    @jax.jit
    def vjp_expData(pars, tang_vec):
        _, back = jax.vjp(processed_expData, pars)
        return back(tang_vec)[0]
    jitted_vjp_expData = jax.jit(vjp_expData)

    class VJPOp(Op):
        '''
        
        '''
        itypes = [pt.dvector,pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, tangent_vector) = inputs
            results = jitted_vjp_expData(parameters, tangent_vector)

            outputs[0][0] = np.asarray(results)

    vjp_op = VJPOp()

    class ExpDataOp(Op):
        '''
        
        '''
        itypes = [pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, ) = inputs
            results = jitted_processed_expData(parameters)

            outputs[0][0] = np.asarray(results)

        def grad(self, inputs, output_gradients):
            (parameters,) = inputs
            (tangent_vector,) = output_gradients
            return [vjp_op(parameters, tangent_vector)]
       
    expData_op = ExpDataOp()    

    return expData_op
    

def makeOp_Aux(model):
    '''
    Wrapping pyhf's model.expected_auxdata for PyMC (i.e. including auxiliary data).

    Args:
        - model: pyhf model.
    Returns:
        - expData_op (class): Wrapper class for model.expected_data.
    '''

    @jax.jit
    def processed_expData(parameters):
        return model.expected_auxdata(parameters)
    jitted_processed_expData = jax.jit(processed_expData)

    @jax.jit
    def vjp_expData(pars, tang_vec):
        _, back = jax.vjp(processed_expData, pars)
        return back(tang_vec)[0]
    jitted_vjp_expData = jax.jit(vjp_expData)

    class VJPOp(Op):
        '''
        
        '''
        itypes = [pt.dvector,pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, tangent_vector) = inputs
            results = jitted_vjp_expData(parameters, tangent_vector)

            outputs[0][0] = np.asarray(results)

    vjp_op = VJPOp()

    class ExpDataOp(Op):
        '''
        
        '''
        itypes = [pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, ) = inputs
            results = jitted_processed_expData(parameters)

            outputs[0][0] = np.asarray(results)

        def grad(self, inputs, output_gradients):
            (parameters,) = inputs
            (tangent_vector,) = output_gradients
            return [vjp_op(parameters, tangent_vector)]

    expData_op = ExpDataOp()    

    return expData_op

def makeOp_ActAndAux(model):
    '''
    Wrapping pyhf's model.expected_auxdata for PyMC (i.e. including auxiliary data).

    Args:
        - model: pyhf model.
    Returns:
        - expData_op (class): Wrapper class for model.expected_data.
    '''

    @jax.jit
    def processed_expData(parameters):
        return model.expected_data(parameters)
    jitted_processed_expData = jax.jit(processed_expData)

    @jax.jit
    def vjp_expData(pars, tang_vec):
        _, back = jax.vjp(processed_expData, pars)
        return back(tang_vec)[0]
    jitted_vjp_expData = jax.jit(vjp_expData)

    class VJPOp(Op):
        '''
        
        '''
        itypes = [pt.dvector,pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, tangent_vector) = inputs
            results = jitted_vjp_expData(parameters, tangent_vector)

            outputs[0][0] = np.asarray(results)

    vjp_op = VJPOp()

    class ExpDataOp(Op):
        '''
        
        '''
        itypes = [pt.dvector]  
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (parameters, ) = inputs
            results = jitted_processed_expData(parameters)

            outputs[0][0] = np.asarray(results)

        def grad(self, inputs, output_gradients):
            (parameters,) = inputs
            (tangent_vector,) = output_gradients
            return [vjp_op(parameters, tangent_vector)]

    expData_op = ExpDataOp()    

    return expData_op