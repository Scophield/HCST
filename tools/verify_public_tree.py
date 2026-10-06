"""Scan the distributable source tree and local README links; never print secrets."""
import argparse,json,re
from pathlib import Path

EXCLUDE={'.git','.venv','build','dist','runs','data','private','__pycache__','.pytest_cache'}
FORBIDDEN={'.pt','.npz','.pm','.mdl','.skw','.sqlite','.pem','.key','.pdf','.docx','.xlsx','.lis','.mt0','.tr0'}

def verify(root):
    findings=[]; inventory=[]
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if any(part in EXCLUDE or part.endswith('.egg-info') for part in rel.parts) or not p.is_file(): continue
        inventory.append(rel.as_posix())
        if p.suffix.lower() in FORBIDDEN: findings.append({'file':rel.as_posix(),'reason':'excluded research/private artifact'})
        if p.suffix.lower() not in {'.py','.json','.md','.toml','.yml','.cff','.txt',''}: continue
        text=p.read_text(encoding='utf-8',errors='replace')
        patterns={
            'private absolute path':r'(?i)(?:[a-z]:[\\/](?:users|onedrive|gsc|codex)|/home/user\d+)',
            'credential assignment':r'(?i)(?:password|passwd|api[_-]?key|token)\s*[:=]\s*[\x22\x27][^\x22\x27\n]{6,}',
            'private key':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
            'credential URL':r'https?://[^\s/@:]+:[^\s/@]+@',
            'private IP address':r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b',
        }
        # Do not scan this scanner's own literal pattern definitions as findings.
        if rel.as_posix()!='tools/verify_public_tree.py':
            for reason,pattern in patterns.items():
                if re.search(pattern,text): findings.append({'file':rel.as_posix(),'reason':reason})
        if p.suffix=='.md':
            for link in re.findall(r'\]\(([^\s)]+)\)',text):
                if link.startswith(('http://','https://','#','mailto:')): continue
                target=(p.parent/link.split('#')[0]).resolve()
                if not target.exists(): findings.append({'file':rel.as_posix(),'reason':'broken local link','target':link})
    return dict(passed=not findings,files=len(inventory),findings=findings,inventory=inventory,
                scope='content patterns and local Markdown targets; copyright/third-party review remains separate')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',default='.'); parser.add_argument('--out'); args=parser.parse_args()
    result=verify(Path(args.root))
    if args.out: Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result['passed'] else 1)
