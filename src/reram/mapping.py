"""Dimensionless legacy conductance to physical siemens; zero means open circuit."""
import numpy as np

def project_signed(w, cfg):
    w=np.asarray(w,dtype=np.float64)
    if not np.all(np.isfinite(w)): raise ValueError("nonfinite weights")
    a=np.minimum(np.abs(w),cfg.g_max)
    a=np.where(a<cfg.g_min,0,a)
    return np.maximum(np.sign(w)*a,0),np.maximum(-np.sign(w)*a,0)

def validate_pair(p,n,cfg):
    p,n=np.asarray(p,dtype=np.float64),np.asarray(n,dtype=np.float64)
    if p.ndim!=2 or p.shape!=n.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(n)):
        raise ValueError("conductance pairs must be finite, equal 2D shapes")
    if np.any(p<0) or np.any(n<0): raise ValueError("negative conductance")
    # Legacy checkpoints can contain values below g_min. Preserve them for audit.
    return p,n

def resistance(g,cfg):
    if g<0 or not np.isfinite(g): raise ValueError("invalid conductance")
    return None if g==0 else 1/(g*cfg.conductance_scale_s)
