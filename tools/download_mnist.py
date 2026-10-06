"""Explicit opt-in download from the PyTorch MNIST mirror; original files retained."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

FILES={
 'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873',
 'train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432',
 't10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3',
 't10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); args=p.parse_args()
    root=Path(args.out); root.mkdir(parents=True,exist_ok=False)
    manifest=[]
    for name,expected in FILES.items():
        url='https://ossci-datasets.s3.amazonaws.com/mnist/'+name
        with urllib.request.urlopen(url,timeout=60) as response: b=response.read(64*1024*1024)
        if hashlib.md5(b).hexdigest()!=expected: raise ValueError(f'checksum mismatch: {name}')
        (root/name).write_bytes(b)
        manifest.append(dict(file=name,url=url,md5=expected,sha256=hashlib.sha256(b).hexdigest()))
    (root/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__': main()
