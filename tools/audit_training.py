"""Numerical gradient and original-function vs adapter audit; external source only.

No original main, downloads, checkpoint deserialization, or source writes.
"""
import argparse,ast,copy,hashlib,json
from pathlib import Path
import numpy as np
from reram.config import Config
from reram.model import Network,Offsets
from reram.training import LegacyTrainer

def smooth_fixture():
    c=Config(widths=(3,4,3,2),loads=(1,1/3,1/6),offset_std_v=.0001)
    rng=np.random.default_rng(c.seed)
    pairs=[]
    for a,b in zip(c.widths[:-1],c.widths[1:]):
        w=rng.uniform(.08,.14,(a,b)); w[0,0]=-.025
        pairs.append((np.maximum(w,0),np.maximum(-w,0)))
    offsets=[Offsets.sample(b,rng,c.offset_std_v) for b in c.widths[1:]]
    n=Network(c,pairs,offsets)
    x=c.v0+c.input_span*rng.uniform(.2,.8,(5,c.widths[0]))
    t=np.full((5,2),.55); t[np.arange(5),np.arange(5)%2]=.85
    return c,n,x,t

def numeric_gradient_report(network,x,target):
    y,traces=network.forward(x,True)
    dy=(y-target)/len(x); grads=[]
    for l in reversed(network.layers): dy,g=l.backward(dy); grads.append(g)
    grads=list(reversed(grads))
    eps=1e-6; layer_reports=[]
    def loss():
        z,_=network.forward(x)
        return float(np.mean(np.sum((z-target)**2,axis=1))/2)
    for li,(l,(gp,gn)) in enumerate(zip(network.layers,grads)):
        w=l.p-l.n; analytic=np.where(w>0,gp,-gn); numerical=np.zeros_like(w)
        for ix in np.ndindex(w.shape):
            orig=w[ix]
            for sign in [1,-1]:
                value=orig+sign*eps; l.p[ix]=max(value,0); l.n[ix]=max(-value,0)
                if sign==1: plus=loss()
                else: minus=loss()
            l.p[ix]=max(orig,0); l.n[ix]=max(-orig,0)
            numerical[ix]=(plus-minus)/(2*eps)
        err=float(np.max(np.abs(analytic-numerical)))
        margin=float(min(np.min(np.abs(l.cache[1][0]-l.neuron.ivc_floor)),np.min(np.abs(l.cache[1][1]-l.neuron.ivc_floor))))
        activation_margin=float(np.min(np.abs(traces[li]['subtractor']-(network.cfg.v0-l.offsets.activation)))) if l.hidden else None
        layer_reports.append(dict(layer=li,checked_signed_weights=w.size,max_absolute_gradient_error=err,
                                  ivc_boundary_margin_v=margin,activation_boundary_margin_v=activation_margin))
    return dict(method='central differences of complete mean batch loss; fixed complementary conductance signs',
                epsilon=eps,layers=layer_reports,passed=all(r['max_absolute_gradient_error']<2e-8 for r in layer_reports))

class OriginalFunctionReference:
    """Independent reference invokes the original orchestration functions verbatim."""
    def __init__(self,network,source):
        import torch
        self.torch=torch; self.cfg=c=network.cfg
        tree=ast.parse(Path(source).read_text(encoding='utf-8-sig'))
        cls_names=['BaseLayer','MiddleLayer1','MiddleLayer2','OutputLayer']
        fn_names=['forward_propagation','backpropagation','uppdate_wb']
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in cls_names or isinstance(n,ast.FunctionDef) and n.name in fn_names]
        if len(nodes)!=7: raise ValueError('expected original classes/functions unavailable')
        module=ast.Module(body=nodes,type_ignores=[]); ast.fix_missing_locations(module)
        env={'torch':torch,'device':torch.device('cpu'),'wb_width':.1}
        scalars={'v0':c.v0,'GI':c.gi,'GS':c.gs,'GF':c.gf,'VrefI':c.ref_i,'VrefS':c.ref_s,'VrefF':c.ref_f,
                 'gL1':c.loads[0],'gL2':c.loads[1],'gL3':c.loads[2],'mac_min':c.mac_min,'eta':c.learning_rate}
        env.update({k:torch.tensor(v,dtype=torch.float32) for k,v in scalars.items()})
        ns=[['vosi1p_tensor','vosi1n_tensor','voss1_tensor','vosr1_tensor','vosf1_tensor'],
            ['vosi2p_tensor','vosi2n_tensor','voss2_tensor','vosr2_tensor','vosf2_tensor'],
            ['vos11p_tensor','vos11n_tensor','vos22_tensor','unused_r','unused_f']]
        for l,names in zip(network.layers,ns):
            for name,value in zip(names,vars(l.offsets).values()): env[name]=torch.tensor(value,dtype=torch.float32)
        exec(compile(module,Path(source).name,'exec'),env)
        self.layers=[]
        for i,(l,name) in enumerate(zip(network.layers,['middle_layer_1','middle_layer_2','output_layer'])):
            obj=env[cls_names[i+1]](l.p.shape[0],l.p.shape[1],torch.device('cpu'))
            obj.w_p=torch.tensor(l.p,dtype=torch.float32); obj.w_n=torch.tensor(l.n,dtype=torch.float32)
            env[name]=obj; self.layers.append(obj)
        self.env=env

    def step(self,x,target):
        torch=self.torch; self.env['t']=torch.tensor(target,dtype=torch.float32)
        self.env['forward_propagation'](torch.tensor(x,dtype=torch.float32))
        z=self.layers[-1].y
        loss=float(torch.mean(torch.sum((z-self.env['t'])**2,dim=1)/2))
        self.env['backpropagation'](self.env['t']); self.env['uppdate_wb']()
        return loss

def audit(source_root):
    c,initial,x,t=smooth_fixture()
    numeric=numeric_gradient_report(copy.deepcopy(initial),x,t)
    reference=OriginalFunctionReference(copy.deepcopy(initial),Path(source_root)/'mnist_hcst_tensor.py')
    adapted=copy.deepcopy(initial); trainer=LegacyTrainer(adapted,source_root)
    analytic=copy.deepcopy(initial); trajectory=[]
    for step in range(10):
        ref_loss=reference.step(x,t); adapter_loss=trainer.train_batch(x,t); analytic_loss=analytic.train_batch(x,t)
        errors=[]
        for r,a in zip(reference.layers,adapted.layers):
            errors.extend([float(np.max(np.abs(r.w_p.detach().numpy()-a.p))),float(np.max(np.abs(r.w_n.detach().numpy()-a.n)))])
        trajectory.append(dict(step=step+1,reference_loss=ref_loss,adapter_loss=adapter_loss,experimental_analytic_loss=analytic_loss,
                               maximum_conductance_update_difference=max(errors),reference_adapter_loss_difference=abs(ref_loss-adapter_loss)))
    # Compare original hand gradients against exact SUM-loss gradients explicitly.
    gradient_reference=OriginalFunctionReference(copy.deepcopy(initial),Path(source_root)/'mnist_hcst_tensor.py')
    gradient_reference.step(x,t)
    differentiable=copy.deepcopy(initial); y,_=differentiable.forward(x,True)
    dy=y-t; exact=[]
    for l in reversed(differentiable.layers): dy,g=l.backward(dy); exact.append(g)
    diagnostics=[]
    for i,(r,gs) in enumerate(zip(gradient_reference.layers,reversed(exact))):
        original=np.concatenate([r.grad_w_p.numpy().ravel(),r.grad_w_n.numpy().ravel()])
        computed=np.concatenate([gs[0].ravel(),gs[1].ravel()])
        diagnostics.append(dict(layer=i,original_sum_gradient_l2=float(np.linalg.norm(original)),
                                exact_sum_gradient_l2=float(np.linalg.norm(computed)),
                                original_to_exact_norm_ratio=float(np.linalg.norm(original)/np.linalg.norm(computed)),
                                relative_l2_difference=float(np.linalg.norm(original-computed)/np.linalg.norm(computed))))
    passed=numeric['passed'] and all(r['maximum_conductance_update_difference']==0 and r['reference_adapter_loss_difference']==0 for r in trajectory)
    return dict(passed=passed,seed=c.seed,config=vars(c),source_sha256=trainer.sha256,
                input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),target_sha256=hashlib.sha256(t.tobytes()).hexdigest(),
                initial_array_sha256={f'layer{i}_{k}':hashlib.sha256(v.tobytes()).hexdigest() for i,l in enumerate(initial.layers) for k,v in [('p',l.p),('n',l.n),*vars(l.offsets).items()]},
                numerical_gradient=numeric,trajectory=trajectory,original_vs_exact_sum_gradient=diagnostics,
                conclusion='legacy adapter is exact for tested original functions/float32 updates; experimental analytic is a different optimizer, not paper-equivalent')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--out',required=True); args=p.parse_args()
    if Path(args.out).exists(): raise FileExistsError(args.out)
    result=audit(args.source); Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result['passed'] else 1)
