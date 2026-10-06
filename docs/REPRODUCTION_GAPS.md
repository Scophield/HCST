# What is still needed for a matched JJAP reproduction

The repository already supplies a standalone computational chain and checked equations. Publication-level matching still requires:

- Final published forward/backward equations and per-op-amp-type gains/intercepts/readout limits; resolve the differing source profiles and follower parameters.
- Publication-era load conductance and normalization profile, including the generator/emulator differences.
- Original initialization, RNG implementation/state/seeds, data split/order/shuffle, batch size, target voltages, optimizer reduction, learning-rate schedule, update rules and stopping history.
- Matching per-chip offset values and distribution/measurement provenance, chip count, conductance/checkpoint hashes, and original result files.
- Authorized model/subcircuit versions, HSPICE version/options, stimulus/precharge clocks, measurement times and classification rules.
- Fresh circuit comparison and multi-chip/seed uncertainty rather than relying on one checkpoint or resource-limited run.

The inspected current source's 60,000-sample/1,000-epoch defaults are not established as the publication-era schedule. No full retraining or matching HSPICE run is claimed. The paper's reported metrics should be cited as paper results, never relabeled as results reproduced by this code.

Useful next extensions are a single-neuron authorized HSPICE comparison, followed by a small array and then a matched MNIST setup. Custom Activation Function, device, RC and compensation backends need their own equations, tests, model provenance and circuit validation. A numerical plugin alone is not a transistor implementation.
