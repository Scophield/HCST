"""Finite-gain voltage equations traced to mnist_hcst_tensor.py forward methods.

No MOS dynamics, RC parasitics, 1T1R access devices, SC compensation or timing.
The last layer bypasses activation and follower exactly as the legacy source.
"""
from dataclasses import dataclass
import numpy as np
from .mapping import project_signed,validate_pair
from .architecture import SynapseArray,NeuronCircuit

@dataclass
class Offsets:
    i_p: np.ndarray
    i_n: np.ndarray
    subtractor: np.ndarray
    activation: np.ndarray
    follower: np.ndarray

    @classmethod
    def sample(cls,n,rng,sigma):
        return cls(*(rng.normal(0,sigma,n) for _ in range(5)))

class Layer:
    def __init__(self,p,n,load,offsets,cfg,hidden):
        p,n=validate_pair(p,n,cfg)
        self.array=SynapseArray(p,n)
        self.load,self.offsets,self.cfg,self.hidden=load,offsets,cfg,hidden
        if load<=0: raise ValueError("load must be positive")
        for name,v in vars(offsets).items():
            if np.asarray(v).shape!=(p.shape[1],) or not np.all(np.isfinite(v)):
                raise ValueError(f"invalid {name} offset shape/value")
        self.neuron=NeuronCircuit(cfg,offsets,load,hidden)

    @property
    def p(self): return self.array.p
    @p.setter
    def p(self,value): self.array.p=value
    @property
    def n(self): return self.array.n
    @n.setter
    def n(self,value): self.array.n=value

    def forward(self,x,cache=False):
        c,o=self.cfg,self.offsets
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=self.p.shape[0] or not np.all(np.isfinite(x)):
            raise ValueError("input shape or values invalid")
        y,trace,(raw,den,derivative)=self.neuron.forward(self.array.response(x))
        if cache: self.cache=(x,raw,den,derivative)
        return y,trace

    def backward(self,dy):
        c,o=self.cfg,self.offsets
        x,raw,den,active=self.cache
        du=dy*active*c.gf/(c.gf+1) if self.hidden else dy
        grads=[]; dx=np.zeros_like(x)
        for w,vos,b,d,sign in zip([self.p,self.n],[o.i_p,o.i_n],raw,den,[-1,1]):
            db=du*c.gs/(c.gs+2)*sign*(b>self.neuron.ivc_floor)
            # Exact chain rule; includes follower and IVC clipping derivatives.
            a=c.ref_i+c.gi*(c.v0+vos)
            dw=np.sum(db*(a-b)/d,axis=0)[None,:]-c.gi*x.T@(db/d)
            dx+=-c.gi*(db/d)@w.T
            grads.append(dw)
        return dx,grads

class Network:
    def __init__(self,cfg,pairs=None,offsets=None):
        self.cfg=cfg.validate()
        rng=np.random.default_rng(cfg.seed)
        pairs=pairs or [project_signed(.1*rng.standard_normal((a,b)),cfg) for a,b in zip(cfg.widths[:-1],cfg.widths[1:])]
        offsets=offsets or [Offsets.sample(n,rng,cfg.offset_std_v) for n in cfg.widths[1:]]
        if len(pairs)!=len(cfg.loads) or len(offsets)!=len(pairs): raise ValueError("wrong layer count")
        self.layers=[]
        for i,((p,n),o,g) in enumerate(zip(pairs,offsets,cfg.loads)):
            if p.shape!=(cfg.widths[i],cfg.widths[i+1]): raise ValueError("wrong layer dimensions")
            self.layers.append(Layer(p,n,g,o,cfg,i<len(pairs)-1))

    def forward(self,x,cache=False):
        traces=[]
        for layer in self.layers:
            x,t=layer.forward(x,cache); traces.append(t)
        return x,traces

    def train_batch(self,x,target):
        """Experimental exact-forward-gradient optimizer, not original HCST training.

        Gradients describe the smooth forward regions before signed hard projection.
        At zero signed weight, the negative-branch derivative is the chosen policy.
        """
        y,_=self.forward(x,True)
        dy=(y-target)/len(x)
        loss=float(np.mean(np.sum((y-target)**2,axis=1))/2)
        grads=[]
        for layer in reversed(self.layers):
            dy,g=layer.backward(dy); grads.append(g)
        for layer,(gp,gn) in zip(self.layers,reversed(grads)):
            # Signed re-projection avoids order-dependent legacy sign-transfer.
            grad=np.where(layer.p>0,gp,-gn)
            layer.p,layer.n=project_signed(layer.p-layer.n-self.cfg.learning_rate*grad,self.cfg)
        return loss

    def save(self,path):
        arrays={}
        for i,l in enumerate(self.layers):
            arrays[f'p{i}'],arrays[f'n{i}']=l.p,l.n
            for k,v in vars(l.offsets).items(): arrays[f'{k}{i}']=v
        np.savez_compressed(path,**arrays)

    @classmethod
    def load(cls,path,cfg):
        with np.load(path,allow_pickle=False) as a:
            pairs=[(a[f'p{i}'],a[f'n{i}']) for i in range(len(cfg.loads))]
            offsets=[Offsets(*(a[f'{k}{i}'] for k in Offsets.__dataclass_fields__)) for i in range(len(cfg.loads))]
        return cls(cfg,pairs,offsets)
