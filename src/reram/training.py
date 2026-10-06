"""Optional exact local-source training adapter; all I/O/main code excluded."""
import ast,hashlib
from pathlib import Path
import numpy as np

class LegacyTrainer:
    def __init__(self,network,source_root):
        import torch
        self.torch,self.network=torch,network
        c=network.cfg
        if len(network.layers)!=3 or c.activation!='relu' or c.g_min!=.01 or c.g_max!=1:
            raise ValueError('legacy trainer requires 3 layers, ReLU and source conductance bounds .01..1')
        source=Path(source_root)/'mnist_hcst_tensor.py'
        self.sha256=hashlib.sha256(source.read_bytes()).hexdigest()
        tree=ast.parse(source.read_text(encoding='utf-8-sig'))
        names=['BaseLayer','MiddleLayer1','MiddleLayer2','OutputLayer']
        classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
        if len(classes)!=4: raise ValueError('expected original layer classes missing')
        module=ast.Module(body=classes,type_ignores=[]); ast.fix_missing_locations(module)
        env=dict(torch=torch,device=torch.device('cpu'),wb_width=.1)
        for k,v in dict(v0=c.v0,GI=c.gi,GS=c.gs,GF=c.gf,VrefI=c.ref_i,VrefS=c.ref_s,VrefF=c.ref_f,
                        gL1=c.loads[0],gL2=c.loads[1],gL3=c.loads[2],mac_min=c.mac_min).items():
            env[k]=torch.tensor(v,dtype=torch.float32)
        offset_names=[['vosi1p_tensor','vosi1n_tensor','voss1_tensor','vosr1_tensor','vosf1_tensor'],
                      ['vosi2p_tensor','vosi2n_tensor','voss2_tensor','vosr2_tensor','vosf2_tensor'],
                      ['vos11p_tensor','vos11n_tensor','vos22_tensor','unused_r','unused_f']]
        for l,ns in zip(network.layers,offset_names):
            for name,v in zip(ns,vars(l.offsets).values()): env[name]=torch.tensor(v,dtype=torch.float32)
        exec(compile(module,source.name,'exec'),env)
        self.env=env
        self.layers=[]
        for i,l in enumerate(network.layers):
            obj=env[names[i+1]](l.p.shape[0],l.p.shape[1],torch.device('cpu'))
            obj.w_p=torch.tensor(l.p,dtype=torch.float32); obj.w_n=torch.tensor(l.n,dtype=torch.float32)
            self.layers.append(obj)

    def train_batch(self,x,target):
        torch=self.torch
        z=torch.tensor(x,dtype=torch.float32)
        for l in self.layers: l.forward(z); z=l.y
        t=torch.tensor(target,dtype=torch.float32)
        # Original OutputLayer.backward references the module-level minibatch t.
        self.env['t']=t
        loss=float(torch.mean(torch.sum((z-t)**2,dim=1)/2))
        self.layers[-1].backward(t)
        for i in reversed(range(len(self.layers)-1)): self.layers[i].backward(self.layers[i+1].grad_x)
        for local,adapted in zip(self.layers,self.network.layers):
            local.update(torch.tensor(self.network.cfg.learning_rate,dtype=torch.float32))
            adapted.p=local.w_p.detach().numpy().astype(np.float64)
            adapted.n=local.w_n.detach().numpy().astype(np.float64)
        return loss
