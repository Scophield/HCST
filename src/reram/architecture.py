"""User-facing hardware boundaries: Synapse Array and Neuron Circuit."""
from dataclasses import dataclass
import numpy as np
from .activation import get_activation

@dataclass
class SynapseArray:
    """ReRAM Crossbar Array; normalized differential conductances."""
    p: np.ndarray
    n: np.ndarray

    def response(self,x):
        return [(w.sum(axis=0),x@w) for w in (self.p,self.n)]

class NeuronCircuit:
    """IV-Converter + Subtractor + Activation Function + Voltage Follower.

The linear final JJAP readout bypasses Activation Function and Follower.
"""
    def __init__(self,cfg,offsets,load,hidden):
        self.cfg,self.offsets,self.load,self.hidden=cfg,offsets,load,hidden
        self.activation=get_activation(cfg.activation)
        self.ivc_floor=cfg.mac_min if hidden else cfg.output_mac_min

    def forward(self,response):
        c,o=self.cfg,self.offsets
        bl,raw,den=[],[],[]
        for (s,mac),vos in zip(response,[o.i_p,o.i_n]):
            d=self.load*(c.gi+1)+s
            b=((c.ref_i+c.gi*(c.v0+vos))*(self.load+s)-c.gi*mac)/d
            raw.append(b); den.append(d); bl.append(np.maximum(b,self.ivc_floor))
        u=(bl[1]-bl[0]+c.v0+2*o.subtractor+2*c.ref_s/c.gs)*c.gs/(c.gs+2)
        activated,derivative=self.activation.evaluate(u,c,o.activation) if self.hidden else (u,np.ones_like(u))
        if activated.shape!=u.shape or derivative.shape!=u.shape or not np.all(np.isfinite(activated)) or not np.all(np.isfinite(derivative)):
            raise ValueError('activation backend must return finite voltage and derivative arrays matching u')
        y=(activated+o.follower+c.ref_f/c.gf)*c.gf/(c.gf+1) if self.hidden else u
        return y,dict(iv_p=bl[0],iv_n=bl[1],subtractor=u,output=y), (raw,den,derivative)
