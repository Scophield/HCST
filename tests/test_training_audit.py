import importlib.util,os
from pathlib import Path
import pytest

def audit_module():
    p=Path(__file__).resolve().parents[1]/'tools'/'audit_training.py'
    spec=importlib.util.spec_from_file_location('audit_training',p); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def test_complete_network_gradient_in_smooth_region():
    m=audit_module(); _,n,x,t=m.smooth_fixture(); result=m.numeric_gradient_report(n,x,t)
    assert result['passed']
    for l in result['layers']:
        assert l['ivc_boundary_margin_v']>1e-4
        if l['activation_boundary_margin_v'] is not None: assert l['activation_boundary_margin_v']>1e-4

@pytest.mark.skipif(not os.environ.get('RERAM_LEGACY_SOURCE'),reason='external original source required')
def test_original_orchestration_matches_adapter_exactly():
    result=audit_module().audit(os.environ['RERAM_LEGACY_SOURCE'])
    assert result['passed']
    assert all(r['maximum_conductance_update_difference']==0 for r in result['trajectory'])

def test_old_analytic_alias_reports_experimental_support(tmp_path):
    from reram.experiment import run
    from reram.config import Config
    result=run(Config(epochs=1,train_limit=8,test_limit=4,batch_size=4),tmp_path/'run',training_backend='analytic')
    assert result['training_backend']=='experimental-analytic'
    assert 'not equivalent' in result['training_support']
