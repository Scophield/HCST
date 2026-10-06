# Repository validation - separate from paper results

## Standalone computational checks

Tests cover conductance bounds/physical units, zero-as-open mapping, hidden/final IVC limits, independent KCL and end-to-end nodal solution, finite-difference gradients, custom Activation Function registration and SPICE rejection, safe NPZ handling, IDX corruption, netlist connectivity, missing/failed circuit measurements, timeout behavior, determinism and explicit trainer support levels.

The optional four original-source tests are skipped in a normal public install because the original scripts are not distributed. Local pre-publication reference verification included those tests with an authorized external copy. Their successful execution does not mean the external sources are part of the MIT release.

## Training semantics

The fixed three-layer smooth fixture in [training_semantics_audit.json](../evidence/training_semantics_audit.json) checks 30 signed weights against central differences of the complete mean batch loss, epsilon 1e-6. Maximum absolute error is about 2.11e-11. This validates smooth forward derivatives before hard conductance projection.

An independently loaded reference calls the original `forward_propagation`, `backpropagation`, and `uppdate_wb` functions. Same initial weights, fixed offsets, five-sample batch, target, precision and learning rate produce **zero adapter/reference update and loss differences** for ten steps. The exact-source adapter is therefore compatible with that tested path.

The original hand gradient and the exact derivative are distinct. After aligning both to **summed loss**, their gradient norm ratios are about 18.1105, 6.03682 and 1.0000003 across the three layers. Source-specific inverse-load and follower factors account for material hidden-layer differences. Mean versus sum and sequential sign transfer add further optimization differences. The default experimental trainer is intentionally not described as source-equivalent.

## New bounded MNIST experiment

Both runs used the same current-source 784x256x128x10 profile, seed 20261006, fixed synthetic chip offsets, 2,048 first training samples and 1,024 first independent official test samples, 40 epochs, batch size 32, learning rate .002. Pixel preprocessing, labels and argmax readout were unchanged. No accuracy-targeted tuning was performed. These subsets are not a full benchmark or multi-chip uncertainty estimate.

| Backend | Zero-offset offline | Offline deployed | HCST / nonideal-model training | Time |
|---|---:|---:|---:|---:|
| Original-source adapter | 658/1024 | 117/1024 | 458/1024 | 178.204 s |
| Experimental analytic | 98/1024 | 132/1024 | 98/1024 | 115.343 s |

See the corresponding [source-adapter results](../evidence/mnist_medium_legacy_results.json), [history](../evidence/mnist_medium_legacy_history.json), [experimental results](../evidence/mnist_medium_experimental_results.json) and [history](../evidence/mnist_medium_experimental_history.json). The JSON records originate from the pre-publication computational candidate and retain their original provenance labels. No weights, images, local path or connection information is included.

The source-adapter run shows recovery for its particular fixed model, but does not reach or reproduce a matched JJAP result. The experimental run does not show accuracy recovery. Small smoke outcomes cannot support a general claim that HCST is ineffective, and comparing the two algorithms at a shared numerical learning rate is not a controlled isolation of their differences.

## Circuit limits

The independent nodal solver agrees with the small complete model at 4e-13 V tolerance. This is a resistor/finite-gain calculation, not MOS dynamics. Transistor generation has structural tests; there is no verified HSPICE result in this release. The public evidence cannot establish rail behavior, dynamic accuracy, retention, power, latency, area or fabricated-chip operation.
