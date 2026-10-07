# v0.1.0 - research preview

This is a research preview of the version 0.1.0 implementation, not a stable software guarantee or a complete reproduction of the JJAP experiment. The tag follows the existing package and software citation version. A GitHub prerelease, when created, should set `prerelease=true` for this tag.

## Included

- Installable NumPy voltage-domain model of differential ReRAM Synapse Arrays and finite-gain Neuron Circuits, with fixed input offsets and layer load parameters.
- Standalone seeded CPU demonstration, conductance mapping, behavioral netlist generation, logs and numerical verification.
- Explicit experimental analytic training and optional original-source compatibility paths. They are not equivalent optimizers; the latter requires separately authorized historical scripts.
- Paper references, original project illustrations, credited author-reused JJAP Fig. 13 and separated validation evidence.

## Limits

- This release does not establish paper-level accuracy reproduction, fabricated-chip operation or fully on-chip learning.
- No new HSPICE simulation is verified. HSPICE and authorized transistor/device models remain optional external dependencies.
- No dataset, historical model weights, original experimental scripts, private server configuration or credentials are included.
- Fixed-offset behavior does not establish general compensation for time-varying noise, retention, drift or programming nonlinearities.

MIT applies to the newly authored software and original project artwork. The original-paper figure has separate rights recorded in `docs/FIGURE_RIGHTS.md` and is excluded from MIT. Automatic source archives contain only tracked reviewed public files; no private experiment directory is attached.

## Run and verify

```sh
python -m pip install '.[test]'
reram run --config configs/smoke.json --out runs/preview-smoke
python -m pytest -q
```

See the README for data acquisition and optional circuit validation. Consult the GitHub Actions run for the exact tag commit; a source archive is not a substitute for recorded validation.
