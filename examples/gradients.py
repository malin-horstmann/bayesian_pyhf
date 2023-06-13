from pyhf_pymc import make_op

import pyhf
import pytensor
import pymc as pm
from pytensor import tensor as pt



# Model
model = pyhf.Model(
    {'channels': [{'name': 'singlechannel',
    'samples': [
    {'name': 'signal',
     'data': [10, 20, 10],
     'modifiers': [
        {'name': 'mu', 'type': 'normfactor', 'data': None}
        ]},
    {'name': 'background',
     'data': [120, 110, 100],
     'modifiers': [
        # Correlated / Normal
        {'name': 'corr_bkg', 'type': 'histosys','data': {'hi_data': [125, 110, 100] , 'lo_data': [100, 95, 90]}},
        # Uncorrelated / Poisson
        {'name': 'uncorr_bkg', 'type': 'shapesys','data': [40, 51, 62]},
         ]}]}]}
)

# Make Op

expData_op = make_op.makeOp_Act(model)

print(expData_op.grad(pt.as_tensor_variable(model.config.suggested_init())).eval())