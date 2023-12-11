import pytest

import pyhf
import pytensor
from pytensor import tensor as pt

from bayesian_pyhf import make_op

import numpy as np

@pytest.fixture()
def model():
    model = pyhf.Model(
        {'channels': [{'name': 'singlechannel',
        'samples': [
        {'name': 'signal',
        'data': [10, 20, 10],
        'modifiers': [
            {'name': 'mu', 'type': 'normfactor', 'data': None},
            {'name': 'my_shapefactor', "type": 'shapefactor', 'data': None} ]},
        {'name': 'background',
        'data': [120, 110, 100],
        'modifiers': [
            # Normalisation Uncertainty / Normal
            {'name': 'normSys', "type": "normsys", "data": {"hi": 0.95, "lo": 1.05}},
            # Staterror / Normal
            {"name": "my_staterror","type": "staterror","data": [10., 1., 0.1],},
            # Lumi / Normal
            {'name': 'lumi', 'type': 'lumi', 'data': None},
            # Correlated / Normal
            {'name': 'corr_bkg', 'type': 'histosys','data': {'hi_data': [225, 210, 200] , 'lo_data': [100, 95, 90]}},
            # Uncorrelated / Poisson
            {'name': 'uncorr_bkg', 'type': 'shapesys','data': [40, 51, 62]},]}
        ]}],
        "parameters": [
                {
                    "name": "lumi", "auxdata": [1.0], "sigmas": [0.017], "bounds": [[0.915, 1.085]], "inits": [1.0],
                }],
            }
        )

    return model

@pytest.fixture
def Op_Act(model):
    return make_op.makeOp_Act(model)

@pytest.fixture
def Op_Aux(model):
    return make_op.makeOp_Aux(model)


class TestOps():
    def test_ActOp(self, model, Op_Act):
        assert (model.expected_actualdata(model.config.suggested_init()) == Op_Act(pt.as_tensor_variable(model.config.suggested_init())).eval()).all()

    def test_AuxOp(self, model, Op_Aux):
        assert (model.expected_auxdata(model.config.suggested_init()) == Op_Aux(pt.as_tensor_variable(model.config.suggested_init())).eval()).all()