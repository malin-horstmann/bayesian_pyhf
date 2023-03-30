import numpy as np
import matplotlib.pyplot as plt
import json
import pymc as pm

import pyhf
pyhf.set_backend('jax')

from jax import grad, jit, vmap, value_and_grad, random
import jax
import jax.numpy as jnp

import pytensor
from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph import Apply, Op

import aesara
import aesara.tensor as at
from aesara.graph.op import Op
from aesara.link.jax.dispatch import jax_funcify

from contextlib import contextmanager

import sys
sys.path.insert(1, '/Users/malinhorstmann/Documents/pyhf_pymc/src')
from pyhf_pymc import prepare_inference

####

def make_op(func, itypes, otypes):
    """

    """
    @jax.jit
    def vjp_func(fwd_inputs, vector):
        _,back = jax.vjp(func,fwd_inputs)
        return back(vector)

    class JaxVJPOp(Op):
        __props__ = ("jax_vjp_func",)

        def __init__(self):
            self.jax_vjp_func = vjp_func
            self.itypes = itypes + otypes
            self.otypes = itypes
            super().__init__()

        def perform(self, node, inputs, outputs):

            results = self.jax_vjp_func(*(jnp.asarray(x) for x in inputs))

            if not isinstance(results, (list, tuple)):
                results = (results,)

            for i, r in enumerate(results):
                outputs[i][0] = np.asarray(r)


    jax_grad_op = JaxVJPOp()
                
    @jax_funcify.register(JaxVJPOp)
    def jax_funcify_JaxGradOp(op):
        return op.jax_vjp_func

    @jax.jit
    def fwd_func(fwd_inputs):
        return func(fwd_inputs)
    
    class JaxOp(Op):
        __props__ = ("fwd_func",)

        def __init__(self):
            self.fwd_func = fwd_func
            self.itypes = itypes
            self.otypes = otypes
            super().__init__()

        def perform(self, node, inputs, outputs):
            results = self.fwd_func(*(jnp.asarray(x) for x in inputs))
            if len(outputs) == 1:
                outputs[0][0] = np.asarray(results)
                return
            for i, r in enumerate(results):
                outputs[i][0] = np.asarray(r)

        def grad(self, inputs, vectors):
            return [jax_grad_op(inputs[0], vectors[0])]

    @jax_funcify.register(JaxOp)
    def jax_funcify_JaxOp(op):
        return op.fwd_func

    jax_op = JaxOp()
    
    return jax_op, jax_grad_op

