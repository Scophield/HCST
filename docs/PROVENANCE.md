# Provenance of this public implementation

The primary reference is Gao and Ohsawa's JJAP article, DOI [10.35848/1347-4065/ad1895](https://doi.org/10.35848/1347-4065/ad1895). Its MNIST architecture is 784x256x128x10. SSDM 2023, available as [arXiv:2609.04259](https://arxiv.org/abs/2609.04259), supplies the differential Synapse Array and Neuron Circuit context; its original dataset experiment was IRIS. The 2026 arXiv upload date is not the conference experiment date.

This repository is **new implementation code**, not a renamed dump of the historical directory. It separates array response, finite-gain neuron equations, gradients, mapping, reproducible experiments, SPICE generation and analysis. No original script/header/model/checkpoint/figure/manuscript is copied or licensed here. Optional compatibility adapters read separately authorized originals at runtime.

| Public module | Origin of implementation | Research/source relationship |
|---|---|---|
| `config.py`, `data.py`, `cli.py`, `experiment.py` | New parameter/data/logging/CLI code | Reproduces explicitly recorded voltage preprocessing and controlled experiments |
| `architecture.py`, `model.py`, `mapping.py` | New mathematical implementation and class structure | Source-informed finite-gain equations and differential conductance convention; independently tested |
| `activation.py` | New registration/validation interface | ReLU transfer is the baseline instance; other numerical transfers are explicit extensions |
| `netlist.py`, `hspice.py` | New topology renderer and process/measurement interface | Functional resistor/neuron connectivity; external macro/device bodies are never included |
| `training.py`, `legacy.py` | New AST/reference/import adapters | Selected original classes are only loaded from caller-supplied paths; no class bodies are embedded |
| Tests and `tools/audit_training.py` | New test fixtures/reference loaders | Check KCL, gradients, update compatibility and operational failure handling |
| `tools/download_mnist.py` | New simple explicit downloader | Public archive filenames/checksums; libraries and dataset not vendored |
| Documentation and diagram | Newly written text and Mermaid diagram | Method attribution, evidence and limits; no paper figure copied |

The inspected current training source uses gains 163, intercepts .4616 and loads 1,1/3,1/6. The older emulator uses gains 1369, intercepts .4859 and all loads 1. They are not interchangeable. The final IVC floor is locally overridden to 1e-9 V while hidden layers use .05 V. These details are captured in configs and regression tests; they are not declared the definitive publication-era profile.

The inspected historical scripts carried an `@author: Gsc` note but no explicit redistribution license was found beside them. That is insufficient to relicense those files or assert sole ownership of all assets. The public repository therefore contains only the new software and aggregate validation described above. The method is credited to its paper authors, and the MIT scope is limited by [NOTICE](../NOTICE.md).

Historical checkpoint evaluations, exact chip identities, original raw logs, device models, local host information and unpublished extensions are not included. New aggregate computational evidence is explicitly labeled 2026 validation and binds source/config/data/seed identities without carrying trained weights or images. It is not proof of paper reproduction.
