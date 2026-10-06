import copy,hashlib,json,platform,sys,time,os
from pathlib import Path
import numpy as np
from .model import Network
from .data import mnist,targets
from .netlist import generate

def measure(network,x,y):
    predictions=[]; outputs=[]
    for i in range(0,len(x),128):
        z,_=network.forward(x[i:i+128]); outputs.append(z); predictions.extend(z.argmax(axis=1).tolist())
    z=np.concatenate(outputs)
    confusion=np.zeros((network.cfg.widths[-1],network.cfg.widths[-1]),dtype=int)
    for a,b in zip(y,predictions): confusion[int(a),b]+=1
    return dict(samples=len(x),correct=int(np.sum(np.asarray(predictions)==y)),
                accuracy=float(np.mean(np.asarray(predictions)==y)),min_output_v=float(z.min()),
                max_output_v=float(z.max()),confusion=confusion.tolist())

def environment():
    package=Path(__file__).parent
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob('*.py'))}
    return dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
                framework='0.1.0',backend='numpy-float64-cpu',thread_limits={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},source_sha256=source_hashes)

def run(cfg,out,data=None,training_backend='experimental-analytic',source=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    cfg.write(out/'config.json')
    (out/'environment.json').write_text(json.dumps(environment(),indent=2),encoding='utf-8')
    start=time.monotonic(); manifest=[]
    if data:
        if cfg.widths[0]!=784 or cfg.widths[-1]!=10: raise ValueError('MNIST requires 784 inputs and 10 outputs')
        x,y,m=mnist(data,'train',cfg.train_limit,cfg); manifest+=m
        xt,yt,m=mnist(data,'test',cfg.test_limit,cfg); manifest+=m
        dataset='MNIST-first-N-independent-official-train-and-test-splits'
    else:
        rng=np.random.default_rng(cfg.seed+1)
        x=cfg.v0+cfg.input_span*rng.random((cfg.train_limit,cfg.widths[0]))
        xt=cfg.v0+cfg.input_span*rng.random((cfg.test_limit,cfg.widths[0]))
        teacher=rng.standard_normal((cfg.widths[0],cfg.widths[-1]))
        y=((x-cfg.v0)@teacher).argmax(axis=1); yt=((xt-cfg.v0)@teacher).argmax(axis=1)
        dataset='synthetic-plumbing-demo-no-paper-or-MNIST-accuracy-claim'
    chip=Network(cfg); hcst=copy.deepcopy(chip)
    idealcfg=copy.deepcopy(cfg); idealcfg.offset_std_v=0
    offline=Network(idealcfg)
    training_backend={'analytic':'experimental-analytic','legacy':'legacy-source'}.get(training_backend,training_backend)
    if training_backend not in ['experimental-analytic','legacy-source']: raise ValueError('unknown training backend')
    trainer_off,trainer_h=offline,hcst
    source_hash=None
    if training_backend=='legacy-source':
        if not source: raise ValueError('external original source required for legacy training')
        from .training import LegacyTrainer
        trainer_off,trainer_h=LegacyTrainer(offline,source),LegacyTrainer(hcst,source)
        source_hash=trainer_h.sha256
    rng=np.random.default_rng(cfg.seed+2)
    history=[]
    for epoch in range(cfg.epochs):
        order=rng.permutation(len(x)); loss_off=[]; loss_h=[]
        for st in range(0,len(x),cfg.batch_size):
            ix=order[st:st+cfg.batch_size]; t=targets(y[ix],cfg)
            loss_off.append(trainer_off.train_batch(x[ix],t)); loss_h.append(trainer_h.train_batch(x[ix],t))
        history.append(dict(epoch=epoch+1,offline_loss=float(np.mean(loss_off)),hcst_loss=float(np.mean(loss_h))))
        # Durable progress without claiming completion before final evaluation.
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(history[-1])+'\n')
    deployed=copy.deepcopy(chip)
    for target,trained in zip(deployed.layers,offline.layers): target.p,target.n=trained.p.copy(),trained.n.copy()
    results=dict(provenance=cfg.provenance,dataset=dataset,
                 untrained_chip=measure(chip,xt,yt),offline_zero_offsets=measure(offline,xt,yt),
                 offline_deployed_to_chip=measure(deployed,xt,yt),hcst_same_chip=measure(hcst,xt,yt),
                 elapsed_seconds=time.monotonic()-start,hspice_verified=False,
                 paper_reproduced=False,training_backend=training_backend,training_source_sha256=source_hash,
                 training_support=('experimental computational trainer; not equivalent to original HCST optimization' if training_backend=='experimental-analytic'
                                   else 'source-compatible class updates; publication schedule and chip provenance unverified'),
                 training=('exact-analytic-gradient signed SGD, not legacy hand-gradient update' if training_backend=='experimental-analytic'
                           else 'external original class backward/update on CPU; controlled RC data and batching'))
    for name,net in [('offline',offline),('hcst',hcst),('deployed',deployed)]: net.save(out/f'{name}.npz')
    np.savez_compressed(out/'sample.npz',x=xt[:1],y=yt[:1])
    z,traces=hcst.forward(xt[:1]); expected={}
    for i,t in enumerate(traces):
        for stage,key in [('ivp','iv_p'),('ivn','iv_n'),('sub','subtractor'),('out','output')]:
            for j,v in enumerate(t[key][0]): expected[f'{stage}_{i}_{j}']=float(v)
    generate(hcst,xt[0],out/'sample_behavioral.sp')
    for name,obj in [('results.json',results),('history.json',history),('data_manifest.json',manifest),('expected_measures.json',expected)]:
        (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'run.log').write_text('\n'.join(json.dumps(row) for row in history)+'\n'+json.dumps(results)+'\n',encoding='utf-8')
    filehashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'artifact_hashes.json').write_text(json.dumps(filehashes,indent=2),encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,paper_reproduced=False,hspice_verified=False)),encoding='utf-8')
    return results
