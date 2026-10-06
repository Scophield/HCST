"""Optional commercial simulator adapter. No credentials or device models included."""
import os,re,subprocess,json,math
from pathlib import Path

def run(deck,out,executable=None,timeout=120):
    executable=executable or os.environ.get('HSPICE_EXE')
    if not executable: raise RuntimeError('HSPICE_EXE is unset; provide an existing authorized HSPICE executable or local wrapper')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    deck=Path(deck).resolve()
    try:
        result=subprocess.run([executable,'-i',str(deck),'-o',str(out.resolve()/'simulation')],
                              capture_output=True,text=True,timeout=timeout,shell=False)
    except (subprocess.TimeoutExpired,OSError) as exc:
        (out/'status.json').write_text(json.dumps(dict(verified=False,failure=type(exc).__name__)),encoding='utf-8')
        raise RuntimeError('simulator launch/timeout failure; inspect status.json') from exc
    (out/'stdout.log').write_text(result.stdout,encoding='utf-8')
    (out/'stderr.log').write_text(result.stderr,encoding='utf-8')
    lis=out/'simulation.lis'
    text=lis.read_text(encoding='utf-8',errors='replace') if lis.exists() else ''
    failed=bool(re.search(r'\b(error|aborted|failed)\b',text,re.I))
    status=dict(returncode=result.returncode,listing_present=lis.exists(),listing_reports_failure=failed,
                verified=False,reason='simulation alone requires numerical comparison')
    (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if result.returncode or failed or not text: raise RuntimeError('HSPICE did not produce a successful listing; inspect local logs')
    return status

def parse_listing(path):
    text=Path(path).read_text(encoding='utf-8',errors='replace')
    if re.search(r'\b(aborted|failed|error)\b',text,re.I): raise ValueError('listing reports simulation failure')
    values={}
    for key,value in re.findall(r'\b((?:ivp|ivn|sub|out)_\d+_\d+)\s*=\s*([^\s]+)',text,re.I):
        value=value.lower().replace('d','e')
        try: v=float(value)
        except ValueError:
            m=re.fullmatch(r'([+-]?[\d.]+)(meg|[tgkmunpf])',value)
            if not m: raise ValueError(f'unsupported measure format for {key}')
            v=float(m[1])*dict(t=1e12,g=1e9,meg=1e6,k=1e3,m=1e-3,u=1e-6,n=1e-9,p=1e-12,f=1e-15)[m[2]]
        if not math.isfinite(v): raise ValueError('nonfinite HSPICE result')
        key=key.lower()
        if key in values and values[key]!=v: raise ValueError('conflicting measurement records')
        values[key]=v
    if not values: raise ValueError('no expected measurements in listing')
    return values

def compare(listing,expected,tolerance=.001):
    if not math.isfinite(tolerance) or tolerance<0: raise ValueError('invalid tolerance')
    actual=parse_listing(listing)
    missing=set(expected)-set(actual)
    if missing: raise ValueError(f'missing measurements: {sorted(missing)}')
    errors={k:abs(actual[k]-v) for k,v in expected.items()}
    return dict(measurements=len(errors),max_absolute_error_v=max(errors.values()),
                tolerance_v=tolerance,passed=all(e<=tolerance for e in errors.values()),errors_v=errors)
