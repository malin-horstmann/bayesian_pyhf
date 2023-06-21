import pytest

import pyhf
import pytensor
from pytensor import tensor as pt

from pyhf_pymc import make_op
from pyhf_pymc import prepare_inference
from pyhf_pymc import prepare_inference_combined

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
def priorDict_conjugate(model):
    unconstr_priors = {'my_shapefactor': {'type': 'HalfNormal_Unconstrained', 'sigma': [.1]}, 
    'mu': {'type': 'Gamma_Unconstrained', 'alpha': [5.], 'beta': [1.]}} 

    return prepare_inference.build_priorDict_conjugate(model, unconstr_priors)

@pytest.fixture
def priorDict_combined(model):
    unconstr_priors = {'my_shapefactor': {'type': 'HalfNormal_Unconstrained', 'sigma': [.1]}, 
    'mu': {'type': 'Gamma_Unconstrained', 'alpha': [5.], 'beta': [1.]}} 

    return prepare_inference_combined.build_priorDict_combined(model, unconstr_priors)

class TestPriorDicts:
    def test_PriorDict_conjugate(self, model, priorDict_conjugate):
        assert list(priorDict_conjugate.keys()) == list(model.config.par_map.keys())
    
    def test_PriorDict_combined(self, model, priorDict_combined):
        assert list(priorDict_combined.keys()) == list(model.config.par_map.keys())