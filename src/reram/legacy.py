"""External source audit and forward verification; never executes legacy main."""
import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .model import Network,Offsets

def checkpoint_manifest(root,family):
    root=Path(root); prefix='ref_output' if family=='ref' else 'output'
    names=[f'{prefix}_{s}_weight_{branch}.pt' for s in ['middlelayer1','middlelayer2','outputlayer'] for branch in ['p','n']]
    names+=['vosi1p.pt','vosi1n.pt','voss1.pt','vosr1.pt','vosf1.pt','vosi2p.pt','vosi2n.pt','voss2.pt','vosr2.pt','vosf2.pt','vos11p.pt','vos11n.pt','vos22.pt']
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}

def audit(root):
    root=Path(root)
    files=[]
    for p in sorted(root.iterdir()):
        if p.is_file():
            record=dict(name=p.name,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                        redistribution='pending-owner-or-third-party-review')
            if p.suffix=='.py':
                try:
                    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
                    record['classes']=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]
                    record['functions']=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                except (SyntaxError,UnicodeError): record['parse']='failed'
            files.append(record)
    return dict(source_alias='external-training_tensor',files=files,raw_source_copied=False)

def forward_parity(root,network,x,source_file='mnist_hcst_tensor.py',tolerance=2e-6):
    import torch
    source=Path(root)/source_file
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    c=network.cfg; tx=torch.tensor(x,dtype=torch.float32)
    errors=[]
    for i,l in enumerate(network.layers):
        classname=['MiddleLayer1','MiddleLayer2','OutputLayer'][i]
        if len(network.layers)!=3: raise ValueError('legacy MNIST parity requires three layers')
        fn=next(n for n in classes[classname].body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        # Execute only the already inspected forward function, with CPU globals.
        module=ast.Module(body=[fn],type_ignores=[]); ast.fix_missing_locations(module)
        env=dict(torch=torch,device=torch.device('cpu'))
        for k,v in dict(v0=c.v0,GI=c.gi,GS=c.gs,GF=c.gf,VrefI=c.ref_i,VrefS=c.ref_s,VrefF=c.ref_f,
                        gL=c.loads[0],gL1=c.loads[0],gL2=c.loads[1],gL3=c.loads[2],mac_min=c.mac_min).items():
            env[k]=torch.tensor(v,dtype=torch.float32)
        names=[('vosi1p_tensor','vosi1n_tensor','voss1_tensor','vosr1_tensor','vosf1_tensor'),
               ('vosi2p_tensor','vosi2n_tensor','voss2_tensor','vosr2_tensor','vosf2_tensor'),
               ('vos11p_tensor','vos11n_tensor','vos22_tensor','unused_r','unused_f')][i]
        for name,v in zip(names,vars(l.offsets).values()): env[name]=torch.tensor(v,dtype=torch.float32)
        exec(compile(module,str(source.name),'exec'),env)
        obj=SimpleNamespace(w_p=torch.tensor(l.p,dtype=torch.float32),w_n=torch.tensor(l.n,dtype=torch.float32),device=torch.device('cpu'))
        env['forward'](obj,tx)
        y,_=l.forward(tx.numpy())
        errors.append(float(np.max(np.abs(y-obj.y.numpy())))); tx=obj.y
    rc,_=network.forward(x)
    end_error=float(np.max(np.abs(rc-tx.numpy())))
    mismatches=int(np.sum(rc.argmax(axis=1)!=tx.numpy().argmax(axis=1)))
    return dict(source_file=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                samples=len(x),max_layer_absolute_error_v=errors,end_to_end_absolute_error_v=end_error,
                classification_mismatches=mismatches,tolerance_v=tolerance,
                passed=max(errors+[end_error])<tolerance and mismatches==0,
                scope='forward-only float32 legacy vs float64 RC; not original training equivalence')

def import_checkpoints(root,cfg,trust_local=False,family='ref'):
    import torch,inspect
    root=Path(root)
    safe='weights_only' in inspect.signature(torch.load).parameters
    if not safe and not trust_local:
        raise RuntimeError('installed torch lacks weights_only; explicit --trust-local is required for known local original tensors')
    def read(name):
        kwargs=dict(map_location='cpu')
        if safe: kwargs['weights_only']=True
        value=torch.load(root/name,**kwargs)
        if not isinstance(value,torch.Tensor): raise ValueError('expected standalone tensor')
        return value.detach().numpy().astype(np.float64)
    stems=['middlelayer1','middlelayer2','outputlayer']
    if family not in ['ref','output']: raise ValueError('unknown checkpoint family')
    prefix='ref_output' if family=='ref' else 'output'
    pairs=[(read(f'{prefix}_{s}_weight_p.pt'),read(f'{prefix}_{s}_weight_n.pt')) for s in stems]
    names=[['vosi1p.pt','vosi1n.pt','voss1.pt','vosr1.pt','vosf1.pt'],
           ['vosi2p.pt','vosi2n.pt','voss2.pt','vosr2.pt','vosf2.pt'],
           ['vos11p.pt','vos11n.pt','vos22.pt']]
    offsets=[Offsets(*(read(n) for n in names[i])) for i in range(2)]
    offsets.append(Offsets(*(read(n) for n in names[2]),np.zeros(cfg.widths[-1]),np.zeros(cfg.widths[-1])))
    return Network(cfg,pairs,offsets)
