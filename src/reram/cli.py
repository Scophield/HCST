import argparse,json,sys,hashlib
from pathlib import Path
from .config import Config
from .model import Network
from .experiment import run,measure,environment
from .data import mnist
from .netlist import generate
from . import legacy,hspice

def main(argv=None):
    parser=argparse.ArgumentParser(description='HCST: source-informed analog circuit computation; not a paper-reproduction claim')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run',help='resource-limited computation; not paper retraining'); p.add_argument('--config',required=True); p.add_argument('--out',required=True); p.add_argument('--data'); p.add_argument('--training-backend',choices=['experimental-analytic','legacy-source','analytic','legacy'],default='experimental-analytic',help='default experimental-analytic differs from original HCST optimization; legacy-source preserves external original class updates; analytic/legacy are older RC aliases'); p.add_argument('--source')
    p=sub.add_parser('audit'); p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('parity'); p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('import-legacy'); p.add_argument('--source',required=True); p.add_argument('--config',required=True); p.add_argument('--out',required=True); p.add_argument('--trust-local',action='store_true'); p.add_argument('--family',choices=['ref','output'],default='ref')
    p=sub.add_parser('evaluate'); p.add_argument('--config',required=True); p.add_argument('--weights',required=True); p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('netlist'); p.add_argument('--config',required=True); p.add_argument('--weights',required=True); p.add_argument('--sample',required=True); p.add_argument('--out',required=True); p.add_argument('--mode',choices=['behavioral','transistor'],default='behavioral'); p.add_argument('--header'); p.add_argument('--model')
    p=sub.add_parser('hspice'); p.add_argument('--deck',required=True); p.add_argument('--out',required=True); p.add_argument('--timeout',type=int,default=120)
    p=sub.add_parser('compare'); p.add_argument('--listing',required=True); p.add_argument('--expected',required=True); p.add_argument('--out',required=True); p.add_argument('--tolerance',type=float,default=.001)
    args=parser.parse_args(argv)
    if Path(args.out).exists(): parser.error('output exists; choose a new path to preserve evidence')
    try:
        if args.command=='run': result=run(Config.read(args.config),args.out,args.data,args.training_backend,args.source)
        elif args.command=='audit': result=legacy.audit(args.source)
        elif args.command=='parity':
            import numpy as np
            cfg=Config(widths=(4,5,3,2),loads=(1,1/3,1/6))
            x=cfg.v0+cfg.input_span*np.random.default_rng(cfg.seed).random((17,4))
            result=legacy.forward_parity(args.source,Network(cfg),x)
        elif args.command=='import-legacy':
            cfg=Config.read(args.config); net=legacy.import_checkpoints(args.source,cfg,args.trust_local,args.family)
            out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
            net.save(out/'legacy.npz'); cfg.write(out/'config.json')
            import torch,inspect
            result=dict(imported=True,weights_safe_pickle_available='weights_only' in inspect.signature(torch.load).parameters,
                        checkpoint_family=args.family,source_file_sha256=legacy.checkpoint_manifest(args.source,args.family),
                        provenance='local-original-tensors-paper-version-unverified')
            (out/'import.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        elif args.command=='evaluate':
            cfg=Config.read(args.config); net=Network.load(args.weights,cfg)
            x,y,m=mnist(args.data,'test',cfg.test_limit,cfg)
            result=dict(metrics=measure(net,x,y),data_manifest=m,environment=environment(),paper_reproduced=False,
                        config=vars(cfg),weights_sha256=hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
                        config_sha256=hashlib.sha256(json.dumps(vars(cfg),sort_keys=True).encode()).hexdigest())
        elif args.command=='netlist':
            import numpy as np
            cfg=Config.read(args.config); net=Network.load(args.weights,cfg)
            with np.load(args.sample,allow_pickle=False) as a: x=a['x'][0]
            generate(net,x,args.out,args.mode,args.header,args.model)
            result=dict(generated=True,mode=args.mode,simulated=False)
        elif args.command=='hspice': result=hspice.run(args.deck,args.out,timeout=args.timeout)
        elif args.command=='compare':
            result=hspice.compare(args.listing,json.loads(Path(args.expected).read_text()),args.tolerance)
        if args.command in ['audit','parity','evaluate','compare']:
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2,allow_nan=False))
        if result.get('passed') is False: return 1
        return 0
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr); return 2

if __name__=='__main__': sys.exit(main())
