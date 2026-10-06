import copy
import numpy as np
import pytest
from reram.config import Config
from reram.mapping import project_signed,resistance
from reram.model import Network,Layer,Offsets
from reram.netlist import generate
from reram.data import idx
from reram.hspice import compare

def test_mapping_bounds_zero_and_units():
    c=Config(); p,n=project_signed(np.array([[-2,-.005,0,.01,2]]),c)
    np.testing.assert_equal(p,[[0,0,0,.01,1]])
    np.testing.assert_equal(n,[[1,0,0,0,0]])
    assert resistance(0,c) is None
    assert resistance(1,c)==pytest.approx(100000)
    assert resistance(c.loads[1],c)==pytest.approx(300000)

def test_finite_gain_kcl_equation():
    c=Config(offset_std_v=0); n=Network(c); l=n.layers[0]
    x=np.array([[.56,.6,.67,.75]])
    _,trace=l.forward(x)
    # Independently substitute IVC output into nodal KCL and amplifier equation.
    for w,offset,b in [(l.p,l.offsets.i_p,trace['iv_p']),(l.n,l.offsets.i_n,trace['iv_n'])]:
        node=(x@w+l.load*b)/(w.sum(axis=0)+l.load)
        solved=np.maximum(c.mac_min,c.ref_i+c.gi*(c.v0+offset-node))
        np.testing.assert_allclose(b,solved,atol=2e-13)

@pytest.mark.parametrize('hidden',[True,False])
def test_gradient_matches_finite_difference(hidden):
    c=Config(); rng=np.random.default_rng(8)
    p=np.array([[.12,.21],[.19,.03],[.1,.08]])
    n=np.array([[.03,.08],[.11,.15],[.14,.05]])
    l=Layer(p,n,1,Offsets.sample(2,rng,.001),c,hidden)
    x=np.array([[.57,.62,.72],[.83,.66,.52]])
    dy=np.array([[.3,-.2],[.5,.7]])
    l.forward(x,True); dx,g=l.backward(dy)
    eps=1e-6
    def objective(): return float(np.sum(l.forward(x)[0]*dy))
    for w,analytic in [(l.p,g[0]),(l.n,g[1])]:
        for ix in np.ndindex(w.shape):
            orig=w[ix]; w[ix]=orig+eps; a=objective(); w[ix]=orig-eps; b=objective(); w[ix]=orig
            assert analytic[ix]==pytest.approx((a-b)/(2*eps),abs=2e-8)
    for ix in np.ndindex(x.shape):
        orig=x[ix]; x[ix]=orig+eps; a=objective(); x[ix]=orig-eps; b=objective(); x[ix]=orig
        assert dx[ix]==pytest.approx((a-b)/(2*eps),abs=2e-8)

def test_clipping_and_activation_gradient():
    c=Config(); o=Offsets(*(np.zeros(1) for _ in range(5)))
    l=Layer(np.array([[1.]]),np.array([[0.]]),.01,o,c,True)
    l.forward(np.array([[.85]]),True)
    assert l.cache[1][0][0,0]<c.mac_min
    _,g=l.backward(np.ones((1,1)))
    assert g[0][0,0]==0

def test_final_layer_is_linear_and_serialization(tmp_path):
    c=Config(); n=Network(c); x=np.array([[.55,.65,.75,.85]])
    y,t=n.forward(x)
    np.testing.assert_equal(y,t[-1]['subtractor'])
    n.save(tmp_path/'a.npz'); loaded=Network.load(tmp_path/'a.npz',c)
    np.testing.assert_equal(loaded.forward(x)[0],y)

def test_deck_mapping_topology_and_external_models(tmp_path):
    c=Config(); n=Network(c); p=tmp_path/'test.sp'
    outputs=generate(n,np.array([.55,.65,.75,.85]),p)
    s=p.read_text(); assert len(outputs)==3
    assert 'Synapse Array' in s and 'Neuron Circuit' in s
    assert 'E_l0_j0_fol' in s and 'E_l1_j0_fol' not in s
    assert '.measure tran out_1_2' in s
    with pytest.raises(ValueError): generate(n,np.ones(4),p,'transistor')
    header=tmp_path/'header'; header.write_text('.subckt AMPP2 f g out vn vp vh ref\n.ends\n.subckt RELUOUT g out vin vn vh vref\n.ends\n')
    model=tmp_path/'model'; model.write_text('* TEST ONLY NOT A DEVICE MODEL')
    generate(n,np.ones(4),p,'transistor',header,model)
    s=p.read_text(); assert 'X_l0_j0_fol f 0 l0_j0_out l0_j0_out l0_j0_vpf vdd vref AMPP2' in s

def test_idx_rejects_truncated_data(tmp_path):
    p=tmp_path/'idx'; p.write_bytes(b'\0\0\x08\x01\0\0\0\x03\x01\x02')
    with pytest.raises(ValueError): idx(p)

def test_listing_comparison_fails_closed(tmp_path):
    p=tmp_path/'a.lis'; p.write_text('out_0_0 = 5.2e-1\n')
    assert compare(p,{'out_0_0':.52})['passed']
    assert not compare(p,{'out_0_0':.54})['passed']
    with pytest.raises(ValueError): compare(p,{'out_0_1':.5})
    p.write_text('out_0_0 = failed')
    with pytest.raises(ValueError): compare(p,{'out_0_0':.5})

def test_validation_rejects_bad_configs():
    with pytest.raises(ValueError): Config(loads=(1,)).validate()
    with pytest.raises(ValueError): Config(gi=float('nan')).validate()

def test_historical_profiles_are_distinct():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    current=Config.read(root/'configs/legacy_evaluation.json')
    old=Config.read(root/'configs/legacy_fw_emulator.json')
    assert current.gi==163 and old.gi==1369
    assert current.loads==(1,1/3,1/6) and old.loads==(1,1,1)
    assert current.output_mac_min==old.output_mac_min==1e-9

def test_reproducible_end_to_end(tmp_path):
    from reram.experiment import run
    c=Config(epochs=2,train_limit=16,test_limit=8,batch_size=8)
    a,b=run(c,tmp_path/'a'),run(c,tmp_path/'b')
    assert a['training_backend']=='experimental-analytic'
    assert 'not equivalent' in a['training_support']
    for key in ['untrained_chip','offline_zero_offsets','offline_deployed_to_chip','hcst_same_chip']:
        assert a[key]==b[key]
    na=Network.load(tmp_path/'a'/'hcst.npz',c); nb=Network.load(tmp_path/'b'/'hcst.npz',c)
    for la,lb in zip(na.layers,nb.layers): np.testing.assert_equal(la.p,lb.p)
    with pytest.raises(FileExistsError): run(c,tmp_path/'a')
