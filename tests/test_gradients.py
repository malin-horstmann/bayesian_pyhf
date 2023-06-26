import pytest

import pyhf
pyhf.set_backend('jax')
import pytensor
from pytensor import tensor as pt

from pyhf_pymc import make_op

import numpy as np

@pytest.fixture
def model():
    return pyhf.simplemodels.uncorrelated_background(signal=[10, 20], bkg=[100, 200], bkg_uncertainty=[2, 2])

@pytest.fixture
def Op_Act(model):
    return make_op.makeOp_Act(model)

@pytest.fixture
def vjp(model, Op_Act):
    nPars, nBins = len(model.config.suggested_init()), len(model.expected_actualdata(model.config.suggested_init()))

    return Op_Act.grad(inputs=(pt.as_tensor_variable(np.zeros(nPars)),), output_gradients=(pt.as_tensor_variable(np.ones(nBins)),))[0].eval()

class TestGradients:
    def test_gradients_SimpleModel(self, model, Op_Act, vjp):
        nPars = len(model.config.suggested_init())

        pytensor.gradient.verify_grad(Op_Act, pt=(np.arange(nPars, dtype="float64"),), rng=np.random.default_rng())
        assert (vjp == np.array([ 30., 100., 200.])).all()