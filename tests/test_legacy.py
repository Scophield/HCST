import os
import numpy as np
import pytest
from reram.config import Config
from reram.model import Network
from reram.legacy import forward_parity

@pytest.mark.skipif(not os.environ.get('RERAM_LEGACY_SOURCE'),reason='external original source not supplied')
def test_external_original_forward():
    c=Config(widths=(4,5,3,2),loads=(1,1/3,1/6))
    x=c.v0+c.input_span*np.random.default_rng(c.seed).random((17,4))
    assert forward_parity(os.environ['RERAM_LEGACY_SOURCE'],Network(c),x)['passed']

@pytest.mark.skipif(not os.environ.get('RERAM_LEGACY_SOURCE'),reason='external original source not supplied')
def test_external_training_adapter_one_batch():
    from reram.training import LegacyTrainer
    c=Config(widths=(4,5,3,2),loads=(1,1/3,1/6))
    n=Network(c); trainer=LegacyTrainer(n,os.environ['RERAM_LEGACY_SOURCE'])
    loss=trainer.train_batch(np.full((4,4),.65),np.full((4,2),.55))
    assert np.isfinite(loss)
    assert np.isfinite(n.forward(np.full((4,4),.65))[0]).all()

@pytest.mark.skipif(not os.environ.get('RERAM_LEGACY_SOURCE'),reason='external original source not supplied')
def test_original_output_local_floor_override_under_clipping():
    c=Config(widths=(4,5,3,2),loads=(1,1/3,1/6))
    n=Network(c); n.layers[-1].offsets.i_p.fill(-1)
    x=np.full((3,4),.65)
    _,trace=n.forward(x)
    np.testing.assert_equal(trace[-1]['iv_p'],np.full((3,2),1e-9))
    assert forward_parity(os.environ['RERAM_LEGACY_SOURCE'],n,x)['passed']
