import numpy as np
import pytest
from reram.config import Config
from reram.model import Network
from reram.activation import register_activation,BACKENDS
from reram.netlist import generate

def solve_neuron_by_kcl(x,p,n,load,o,j,c,hidden):
    # Unknowns: BL+, BL-, IVC+, IVC-, subtractor+, subtractor-, U.
    a=np.zeros((7,7)); b=np.zeros(7)
    a[0,[0,2]]=[load+p.sum(),-load]; b[0]=x@p
    a[1,[1,3]]=[load+n.sum(),-load]; b[1]=x@n
    a[2,[0,2]]=[c.gi,1]; b[2]=c.ref_i+c.gi*(c.v0+o.i_p[j])
    a[3,[1,3]]=[c.gi,1]; b[3]=c.ref_i+c.gi*(c.v0+o.i_n[j])
    a[4,[3,4]]=[-1,2]; b[4]=c.v0
    a[5,[2,5,6]]=[-1,2,-1]
    a[6,[4,5,6]]=[-c.gs,c.gs,1]; b[6]=c.ref_s+c.gs*o.subtractor[j]
    result=np.linalg.solve(a,b)
    floor=c.mac_min if hidden else c.output_mac_min
    for row,col in [(2,2),(3,3)]:
        if result[col]<floor:
            a[row]=0; a[row,col]=1; b[row]=floor
    result=np.linalg.solve(a,b)
    u=result[6]
    if not hidden: return u
    act=c.v0 if u<=c.v0-o.activation[j] else u
    # Follower nodal equation: y=VrefF+GF*(act+VosF-y).
    return np.linalg.solve(np.array([[1+c.gf]]),np.array([c.ref_f+c.gf*(act+o.follower[j])]))[0]

def test_full_network_matches_independent_dc_nodal_solver():
    c=Config(widths=(4,5,3,2),loads=(1,.3333333333333333,.16666666666666666))
    net=Network(c)
    x=c.v0+c.input_span*np.random.default_rng(4).random((7,4))
    actual,_=net.forward(x)
    reference=x.copy()
    for l in net.layers:
        reference=np.array([[solve_neuron_by_kcl(row,l.p[:,j],l.n[:,j],l.load,l.offsets,j,c,l.hidden)
                             for j in range(l.p.shape[1])] for row in reference])
    np.testing.assert_allclose(actual,reference,atol=4e-13,rtol=0)

def test_custom_activation_is_explicit_and_spice_rejects_it(tmp_path):
    class IdentityTest:
        name='identity-test-only'
        validation_status='test calculation only; no circuit backend'
        def evaluate(self,u,cfg,offset): return u,np.ones_like(u)
    register_activation(IdentityTest())
    try:
        c=Config(activation=IdentityTest.name); net=Network(c)
        assert np.isfinite(net.forward(np.full((2,4),.6))[0]).all()
        with pytest.raises(ValueError): generate(net,np.ones(4),tmp_path/'custom.sp')
    finally: BACKENDS.pop(IdentityTest.name)

def test_output_ivc_floor_is_distinct():
    c=Config(); net=Network(c)
    assert net.layers[0].neuron.ivc_floor==.05
    assert net.layers[-1].neuron.ivc_floor==1e-9
