"""Resistive Synapse Array and Neuron Circuit DC validation decks.

Behavioral mode uses finite-gain dependent sources. Transistor mode references
external AMPP2/RELUOUT definitions and an authorized device model file.
"""
from pathlib import Path
import numpy as np
from .mapping import resistance

def generate(network,x,path,mode='behavioral',header=None,model=None):
    x=np.asarray(x,dtype=float)
    if x.shape!=(network.cfg.widths[0],) or not np.all(np.isfinite(x)): raise ValueError("one finite sample required")
    if mode not in ['behavioral','transistor']: raise ValueError("invalid deck mode")
    if mode=='transistor':
        if not header or not model: raise ValueError("external authorized header and model required")
        for p in (header,model):
            if not Path(p).is_file() or any(c in str(p) for c in ['"','\n','\r']): raise ValueError("invalid external model path")
        h=Path(header).read_text(encoding='utf-8',errors='replace').lower()
        if '.subckt ampp2 ' not in h or '.subckt reluout ' not in h: raise ValueError("header must define AMPP2 and RELUOUT")
    c=network.cfg
    if c.activation!='relu': raise ValueError('custom Activation Function has no implemented SPICE backend')
    lines=['Fully Analog ReRAM - static computational validation, not paper replication',
           f'* mode={mode}; normalized conductance unit={c.conductance_scale_s:.12g} S',
           '* ReRAM Crossbar Array = Synapse Array; IVC/Subtractor/Activation/Follower = Neuron Circuit',
           '.option post=2',f'VREF vref 0 {c.v0:.17g}','VDD vdd 0 1','VF f 0 0']
    if mode=='transistor':
        lines += [f'.include "{Path(model).resolve().as_posix()}"',f'.include "{Path(header).resolve().as_posix()}"']
    previous=[f'in_{i}' for i in range(len(x))]
    lines += [f'VIN_{i} {node} 0 {v:.17g}' for i,(node,v) in enumerate(zip(previous,x))]
    for li,l in enumerate(network.layers):
        lines.append(f'* Layer {li}: Synapse Array -> Neuron Circuit')
        output=[]
        for j in range(l.p.shape[1]):
            tag=f'l{li}_j{j}'; o=l.offsets
            for branch,w,vos in [('p',l.p,o.i_p[j]),('n',l.n,o.i_n[j])]:
                bl,iv=f'{tag}_bl{branch}',f'{tag}_iv{branch}'
                for i,g in enumerate(w[:,j]):
                    r=resistance(g,c)
                    if r is not None: lines.append(f'R_{tag}_{branch}_{i} {previous[i]} {bl} {r:.17g}')
                lines.append(f'R_{tag}_load{branch} {bl} {iv} {1/(l.load*c.conductance_scale_s):.17g}')
                if mode=='behavioral':
                    lines.append(f"E_{tag}_iv{branch} {iv} 0 VOL='max({l.neuron.ivc_floor:.17g},{c.ref_i:.17g}+{c.gi:.17g}*({c.v0+vos:.17g}-v({bl})))'")
                else:
                    vp=f'{tag}_vp{branch}'
                    lines += [f'V_{tag}_osi{branch} {vp} vref {vos:.17g}',f'X_{tag}_iv{branch} f 0 {iv} {bl} {vp} vdd vref AMPP2']
            sp,sn,u=f'{tag}_sp',f'{tag}_sn',f'{tag}_u'
            r=1/c.conductance_scale_s
            lines += [f'R_{tag}_a {tag}_ivp {sn} {r:.17g}',f'R_{tag}_b {tag}_ivn {sp} {r:.17g}',
                      f'R_{tag}_c {sn} {u} {r:.17g}',f'R_{tag}_d {sp} vref {r:.17g}']
            if mode=='behavioral':
                lines.append(f"E_{tag}_sub {u} 0 VOL='{c.ref_s:.17g}+{c.gs:.17g}*(v({sp})-v({sn})+{o.subtractor[j]:.17g})'")
            else:
                lines += [f'V_{tag}_oss {tag}_vps {sp} {o.subtractor[j]:.17g}',f'X_{tag}_sub f 0 {u} {sn} {tag}_vps vdd vref AMPP2']
            out=f'{tag}_out'
            if l.hidden:
                act=f'{tag}_act'
                if mode=='behavioral':
                    lines += [f"E_{tag}_act {act} 0 VOL='v({u})>{c.v0-o.activation[j]:.17g} ? v({u}) : {c.v0:.17g}'",
                              f"E_{tag}_fol {out} 0 VOL='{c.ref_f:.17g}+{c.gf:.17g}*(v({act})+{o.follower[j]:.17g}-v({out}))'"]
                else:
                    lines += [f'V_{tag}_osr {tag}_vnr {u} {o.activation[j]:.17g}',
                              f'X_{tag}_cmp f 0 {tag}_ctrl {tag}_vnr vref vdd vref AMPP2',
                              f'X_{tag}_act 0 {act} {tag}_ctrl {u} vdd vref RELUOUT',
                              f'V_{tag}_osf {tag}_vpf {act} {o.follower[j]:.17g}',
                              f'X_{tag}_fol f 0 {out} {out} {tag}_vpf vdd vref AMPP2']
            else:
                # Linear final layer preserves legacy readout topology.
                out=u
            output.append(out)
            for stage,node in [('ivp',f'{tag}_ivp'),('ivn',f'{tag}_ivn'),('sub',u),('out',out)]:
                lines.append(f'.measure tran {stage}_{li}_{j} FIND v({node}) AT=1n')
        previous=output
    lines += ['.op','.tran 10p 1n','.end']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return previous
