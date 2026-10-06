import subprocess,json
import pytest
from reram.hspice import run

def test_missing_simulator_is_explicit(tmp_path,monkeypatch):
    monkeypatch.delenv('HSPICE_EXE',raising=False)
    with pytest.raises(RuntimeError,match='unset'): run(tmp_path/'deck.sp',tmp_path/'out')

def test_timeout_records_failure_and_never_claims_validation(tmp_path,monkeypatch):
    def timeout(*a,**kw): raise subprocess.TimeoutExpired('test-stub',1)
    monkeypatch.setattr(subprocess,'run',timeout)
    with pytest.raises(RuntimeError,match='timeout'): run(tmp_path/'deck.sp',tmp_path/'out','test-stub',1)
    status=json.loads((tmp_path/'out'/'status.json').read_text())
    assert status['verified'] is False and status['failure']=='TimeoutExpired'
