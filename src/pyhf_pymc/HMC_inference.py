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

from contextlib import contextmanager

import sys
sys.path.insert(1, '/Users/malinhorstmann/Documents/pyhf_pymc/src')
from pyhf_pymc import prepare_inference

####

