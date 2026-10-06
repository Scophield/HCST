# Architecture and model contract

The `SynapseArray` class is the **ReRAM Crossbar Array**, represented by positive and negative conductance matrices. `NeuronCircuit` contains **IV-Converter + Subtractor**, **Activation Function**, and **Voltage Follower**. The JJAP instance of Activation Function is voltage-domain ReLU. Final-layer readout is linear, as in the original code; hidden layers keep the complete chain.

## Units and signs

The original weight files are dimensionless normalized conductances, not arbitrary digital logits. `Gphysical = g * conductance_scale_s`; the base generator uses `R = 100000 / g`, so the default scale is `1e-5 S`. With `g_min=.01`, `g_max=1`, the nonzero physical resistance range is 10 MOhm to 100 kOhm. Zero means no resistor is emitted. Do not substitute a small resistance or finite leakage for zero without changing the model contract.

For an input pixel `a` in `[0,1]`, `Vin = .5 + .35*a` volts. Targets are .55 V for other classes and .85 V for the correct class. Classification is the largest final output voltage, without softmax or digital bias. Weights use `[input, output]` orientation. Positive weights connect to the positive IVC branch, whose inverted response is subtracted from the negative branch.

## Source-derived finite-gain equations

For either conductance branch, `s = sum_i(g_i)`, `q = sum_i(Vin_i*g_i)`, `d = gL*(GI+1)+s`:

```text
Braw = [(VrefI + GI*(V0 + VosI))*(gL+s) - GI*q] / d
B = max(Braw, IVC_floor)
U = (Bn - Bp + V0 + 2*VosS + 2*VrefS/GS) * GS/(GS+2)
A = V0 if U <= V0 - VosA else U       # baseline ReLU
Yhidden = (A + VosF + VrefF/GF) * GF/(GF+1)
Yfinal = U
```

The hidden-layer IVC floor is .05 V; the output-layer source **locally overrides it to 1e-9 V**. This is a regression-tested distinction. Finite gains and five independent per-neuron offsets are retained. The baseline RC samples offsets once for the chip; they are not resampled during training. Existing checkpoint offsets are loaded exactly. The offset sampling distribution in the 2026 demo is a specified synthetic distribution, not a fabricated-chip measurement.

## Activation Function API

`register_activation(backend)` takes an object with `name` and `evaluate(u,cfg,offset) -> (voltage, derivative)`. Arrays must be finite and match `u`. Existing backend names cannot be overwritten. Numerical registration is explicit and local to the Python process. Only `relu` has a SPICE generation backend. Generation for another activation raises an error. ReLU's validation state is source/CPU/nodal verification; actual transistor simulation remains pending.

## Training contracts

The default `experimental-analytic` backend computes exact derivatives through IVC clipping, subtractor, Activation Function and follower. It uses mean minibatch loss and a single signed projection onto complementary conductance pairs. It explicitly does not claim equivalence to original HCST optimization. Gradients are checked in smooth forward regions; projection is discontinuous at conductance thresholds. At zero signed weight, the negative-branch derivative is a specified update policy.

The external `legacy-source` backend extracts the original classes from `mnist_hcst_tensor.py`, executes no top-level I/O, uses CPU tensors, preserves summed hand gradients, and retains the original sequential sign transfer and clipping update. Its output backward method references a module-global `t`; the adapter explicitly binds the current target minibatch. Bounds .01..1, three layers and ReLU are required for this backend.

The legacy source's hidden weight gradients and `grad_x` factors are not identical to exact analytic derivatives of the clipped forward calculation. They also use sum rather than mean gradients. These are retained as historical semantics, not silently fixed. The two trainers are distinct experiments, and using the same numerical learning rate does not make their effective updates equal. Both use the RC's seeded initial conductances and controlled data limits; neither reconstructs undocumented original training provenance.

## Circuit boundary

The behavioral SPICE deck includes physical resistors, load resistors, four subtractor resistors, finite-gain dependent sources, explicit offsets, ReLU transfer and follower feedback. The independent test solves the nodal equations of these circuits rather than repeating the model's closed-form expression.

The external transistor deck uses original AMPP2/RELUOUT pin order, a 1 V supply, original zero gate-bias signal, differential IVC feedback and four subtractor resistors. All device models and subcircuit definitions are external and must be compatible. A static 1 ns measurement is an operating-point check, not a valid timing claim. Transistor rail limits differ from the unconstrained numerical subtractor/follower. No upper-rail clipping is added to hide this discrepancy.

The model cannot predict ReRAM switching, device nonlinear I-V, parasitic RC, 1T1R access resistance, thermal noise, retention, switched-capacitor sampling/compensation, pipeline timing, power or area. Those source variants are preserved externally and listed in the audit; they are not replaced by a pretend equivalent backend.
