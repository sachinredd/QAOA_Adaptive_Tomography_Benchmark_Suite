"""Prespecified physical-channel distribution for the generalisation experiment.

Rates and angular frequencies use the generator's existing dt=1 units.
These are benchmark design choices, not fitted device parameters.
"""
import numpy as np
from .study_protocol import stable_seed

RANGES = {
    'amplitude_damping': {'down': [.02, .40]},
    'thermal_relaxation': {'down': [.04, .30], 'up_down_ratio': [.10, .80]},
    'dephasing': {'z': [.01, .25]},
    'coherent_rotation': {'omega_norm': [.10, .80], 'axis': 'uniform sphere'},
    'driven_dissipation': {'omega_norm': [.10, .80], 'axis': 'uniform sphere',
                           'down': [.02, .25], 'up': [.005, .15], 'z': [.005, .15]},
    'depolarizing': {'a': [.02, .25], 'rates': '[a, a, a/2]'},
}

def broad_parameters(family, instance, strength, seed):
    if family not in RANGES:
        raise ValueError('Unknown family')
    rng = np.random.default_rng(stable_seed(seed, family, instance, 'broad_v1'))
    bounds = RANGES[family]
    def draw(name): return float(rng.uniform(*bounds[name]))
    omega = np.zeros(3); rates = np.zeros(3)
    if 'omega_norm' in bounds:
        axis = rng.normal(size=3)
        while np.linalg.norm(axis) == 0: axis = rng.normal(size=3)
        omega = axis / np.linalg.norm(axis) * draw('omega_norm')
    if family == 'depolarizing':
        a = draw('a'); rates = np.array([a, a, a/2])
    else:
        for i,name in enumerate(['down','up','z']):
            if name in bounds: rates[i] = draw(name)
        if family == 'thermal_relaxation': rates[1] = rates[0] * draw('up_down_ratio')
    return dict(n_qubits=1, dt=1., zz=0., omega=(omega*strength).tolist(), rates=(rates*strength).tolist())
