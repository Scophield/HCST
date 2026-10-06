# Circuit-aware update methodology

Source: Gao and Ohsawa, [JJAP](https://doi.org/10.35848/1347-4065/ad1895), Sections 4.2-4.3, Eqs. (1)-(14), and Section 5. Author-held galley printed pages 5-8 contain these equations. The notation below makes branch indices explicit and uses compact algebra independently derived from the forward equations. It is implementation guidance, not a claim that this repository's experimental optimizer equals the published or historical training program.

## State and notation

Layer l maps input i to neuron j through nonnegative conductances g_ji^+ and g_ji^-. Let x_i=V_i^(l-1), b_j^+=Vmac_j^+, b_j^-=Vmac_j^-, u_j=Vsub_j, a_j=Vrelu_j, and y_j=V_j^(l). The effective signed connection uses the positive-minus-negative pair, but finite-gain branch behavior depends on both conductances separately. V0 is half the supply. GI, GS and GF are finite gains. oI_j^+/-, oS_j, oR_j and oF_j denote the fixed input offsets of the corresponding amplifiers. vI0, vS0 and vF0 denote their zero-differential-input output voltages, not additional input offsets. gL is the IVC load conductance, eta the learning rate and t_j the output target voltage.

## One training iteration

1. **Keep a consistent chip model and weight snapshot.** Characterize or supply gains and offsets as in Section 5/Figs. 7-10. In the fixed-Vos case, hold them fixed while weights change. Use the same pre-update conductances for the forward pass and backward pass.
2. **Forward each layer.** Solve the branch KCL and amplifier relation (Eqs. 1-2), then subtraction (Eq. 3), Activation Function (Eq. 4, ReLU instance) and follower (Eq. 5). Cache each layer's input and intermediate voltages. The output layer omits ReLU/follower.
3. **Compute output loss and error.** Eq. (6) gives C=1/2 sum_j (u_j^L-t_j)^2; Eqs. (7)-(8) define delta_j^l=dC/du_j^l and delta_j^L=u_j^L-t_j. This is the per-pattern expression. Batch accumulation, normalization and schedule must be specified separately rather than silently inferred.
4. **Backpropagate to each hidden layer.** Apply Eqs. (9)-(12): multiply the next-layer unit errors by the next layer's IVC/subtractor input sensitivity, the current follower sensitivity GF/(GF+1), and the current ReLU mask. Keep these sensitivities in the same physical voltage coordinates.
5. **Update the corresponding branch conductances.** For each (l,j,i,+/-), compute its own gradient from the local subtractor sensitivity and delta_j^l, then apply Eq. (13)'s negative-gradient rule. Update both branches from the pre-update state; do not substitute one ideal signed-weight gradient without an explicit mapping.
6. **Apply device bounds.** Following the paragraph after Eq. (14), values above gmax become gmax; values below gmin become zero, including negative values. A value exactly at gmin is retained. This discontinuous projection is not differentiation through a programming pulse.
7. **Repeat and evaluate consistently.** HCST repeats software updates, deploys the final conductance pairs, then evaluates them with the same chip model/offsets. It does not set Vos to zero after training.

## Compact forward equations

For either branch define s_j=sum_i g_ji, q_j=sum_i x_i*g_ji, D_j=(GI+1)*gL+s_j. Independently solving Eqs. (1)-(2) gives:

```text
N_j = (gL+s_j)*vI0 + GI*((V0+oI_j)*(gL+s_j)-q_j)
b_j = N_j/D_j
alpha = GS/(GS+2)
u_j = alpha*(b_j^- - b_j^+ + V0 + 2*oS_j + 2*vS0/GS)
a_j = u_j if u_j > V0-oR_j else V0
y_j = (GF*(a_j+oF_j)+vF0)/(GF+1)
```

These are the unclipped mathematical relations. Repository IVC floors are source-code additions and are not part of the paper Eqs. (1)-(5). Physical supply limits and switching transients require a circuit model. At an activation boundary or a projection threshold, a smooth derivative check does not define a physical update protocol.

## Backward and differential-conductance update

With next-layer branch denominators D_p^+/- and alpha_next, Eq. (10)'s input sensitivity is:

```text
K_pj = GI_next*alpha_next*(g_pj^+/D_p^+ - g_pj^-/D_p^-)
mask_j = 1 if u_j > V0-oR_j else 0
delta_j^l = mask_j * GF/(GF+1) * sum_p(delta_p^(l+1)*K_pj)
```

For each current branch, independent differentiation of the compact forward form above gives the following consistent quotient-rule expression (a re-derivation from Eqs. 1-3, corresponding to the role of Eq. 14):

```text
db_j^+/- / dg_ji^+/- = ((vI0+GI*(V0+oI_j^+/- - x_i))*D_j^+/- - N_j^+/-)/(D_j^+/-)^2
J_ji^+ = -alpha * db_j^+ / dg_ji^+
J_ji^- = +alpha * db_j^- / dg_ji^-
Delta g_ji^+/- = -eta * delta_j^l * J_ji^+/-
g_new = g_old + Delta g
g_new = gmax if g_new > gmax else (0 if g_new < gmin else g_new)
```

**Source transcription cautions:** Eq. (13) in the inspected manuscript/galley mixes i and j in its error index; the chain-rule-consistent local error is delta_j for output neuron j. The displayed Eq. (14)'s final summation appears to omit an input-voltage factor needed by Eq. (1). The compact derivative above is explicitly a re-derivation, not a verbatim transcription or an unannounced correction of the published equation. These ambiguities and the historical program's update semantics should be resolved with the authors before claiming algorithm identity. The existing default optimizer uses additional signed-pair projection and batch conventions; it is not the branch-update recipe above or an original-HCST equivalent.

## What transfers to an on-chip design

The circuit-dependent sensitivities explain which signals and parameters a designer must preserve at every update. A fully on-chip design would need to realize forward values, errors and gradients in hardware, and translate each requested Delta g^+/- into a device programming operation. An in-situ design can obtain forward outputs from hardware while computing updates in software. HCST computes both passes and updates in software, then programs final pairs; Section 2 compares these placements and their endurance/circuit costs.

The equations supply important methodology and guidance for on-chip implementations. They do not provide a complete pulse schedule, conductance-to-pulse calibration, read/verify controller, endurance strategy or fabricated demonstration. ReRAM switching nonlinearity/asymmetry, saturation and drift need their own verified programming loop. No such implementation is claimed here.
