"""Activation Function backends. ReLU is the JJAP baseline instance.

Custom backends require an explicit name, voltage transfer and derivative.
Registering a calculation does not supply or validate a physical circuit.
"""
import numpy as np

class ReLU:
    name='relu'
    validation_status='source-forward-parity-and-numerical-tests; transistor simulation pending'
    def evaluate(self,u,cfg,offset):
        mask=u>cfg.v0-offset
        return np.where(mask,u,cfg.v0),mask.astype(float)

BACKENDS={'relu':ReLU()}

def register_activation(backend):
    if not isinstance(backend.name,str) or not backend.name or not callable(backend.evaluate):
        raise ValueError('backend requires name and evaluate(u,cfg,offset)')
    if backend.name in BACKENDS: raise ValueError('cannot overwrite an existing activation backend')
    BACKENDS[backend.name]=backend

def get_activation(name):
    if name not in BACKENDS: raise ValueError(f'activation backend {name!r} is not registered')
    return BACKENDS[name]
