from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

@dataclass
class Config:
    seed: int = 20261006
    widths: tuple = (4, 5, 3)
    loads: tuple = (1.0, 1 / 3)
    gi: float = 163.0
    gs: float = 163.0
    gf: float = 163.0
    ref_i: float = .4616
    ref_s: float = .4616
    ref_f: float = .4616
    v0: float = .5
    input_span: float = .35
    mac_min: float = .05
    output_mac_min: float = 1e-9
    conductance_scale_s: float = 1e-5
    g_min: float = .01
    g_max: float = 1.0
    offset_std_v: float = .015902
    learning_rate: float = .002
    epochs: int = 10
    batch_size: int = 32
    train_limit: int = 256
    test_limit: int = 128
    label_low_v: float = .55
    label_high_v: float = .85
    provenance: str = "new-2026-validation-not-paper-reproduction"
    activation: str = "relu"

    def validate(self):
        vals=asdict(self)
        for k,v in vals.items():
            if isinstance(v,(float,int)) and not math.isfinite(v):
                raise ValueError(f"{k} must be finite")
        if len(self.widths)<2 or any(not isinstance(w,int) or w<1 for w in self.widths):
            raise ValueError("widths must be positive integers")
        if len(self.loads)!=len(self.widths)-1 or any(g<=0 for g in self.loads):
            raise ValueError("one positive load per layer is required")
        if min(self.gi,self.gs,self.gf,self.conductance_scale_s,self.learning_rate)<=0:
            raise ValueError("gains, physical scale and learning rate must be positive")
        if not 0 <= self.g_min < self.g_max or self.offset_std_v<0:
            raise ValueError("invalid conductance bounds or offset sigma")
        if any(not isinstance(v,int) or v<1 for v in [self.epochs,self.batch_size,self.train_limit,self.test_limit]):
            raise ValueError("run sizes must be positive integers")
        if any(not math.isfinite(g) for g in self.loads):
            raise ValueError("loads must be finite")
        return self

    @classmethod
    def read(cls,path):
        raw=json.loads(Path(path).read_text(encoding="utf-8"))
        unknown=set(raw)-set(cls.__dataclass_fields__)
        if unknown: raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for k in ['widths','loads']:
            if k in raw: raw[k]=tuple(raw[k])
        return cls(**raw).validate()

    def write(self,path):
        Path(path).write_text(json.dumps(asdict(self),indent=2)+"\n",encoding="utf-8")
