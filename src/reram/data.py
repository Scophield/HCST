"""Read existing official MNIST IDX files; never download implicitly."""
import gzip,hashlib,struct
from pathlib import Path
import numpy as np

def idx(path):
    p=Path(path)
    if not p.exists() and Path(str(p)+'.gz').exists(): p=Path(str(p)+'.gz')
    b=gzip.open(p,'rb').read() if p.suffix=='.gz' else p.read_bytes()
    if len(b)<4 or b[:3]!=b'\0\0\x08': raise ValueError("expected uint8 IDX")
    dims=b[3]
    if dims not in (1,3): raise ValueError("unsupported IDX dimensions")
    shape=struct.unpack('>'+('I'*dims),b[4:4+4*dims])
    a=np.frombuffer(b[4+4*dims:],dtype=np.uint8)
    if a.size!=np.prod(shape): raise ValueError("IDX size mismatch")
    return a.reshape(shape),dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def mnist(root,split,limit,cfg):
    root=Path(root)
    if (root/'MNIST'/'raw').is_dir(): root=root/'MNIST'/'raw'
    prefix='train' if split=='train' else 't10k'
    x,mx=idx(root/(prefix+'-images-idx3-ubyte'))
    y,my=idx(root/(prefix+'-labels-idx1-ubyte'))
    if x.shape[1:]!=(28,28) or len(x)!=len(y) or np.any(y>9): raise ValueError("invalid MNIST")
    if limit>len(x): raise ValueError("limit exceeds split size")
    return cfg.v0+cfg.input_span*x[:limit].reshape(limit,784)/255.,y[:limit], [mx,my]

def targets(y,cfg):
    t=np.full((len(y),cfg.widths[-1]),cfg.label_low_v)
    t[np.arange(len(y)),y]=cfg.label_high_v
    return t
