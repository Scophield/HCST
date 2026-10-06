# Scope of the MIT license

The MIT license applies to the newly authored implementation, tests, tooling and documentation committed in this repository, published with the repository owner's authorization. It does not apply to external research scripts, checkpoints, datasets, semiconductor models, commercial programs, publications, or collaborator materials.

The historical experiment scripts inspected for compatibility carry an `@author: Gsc` note, but no explicit redistribution license was found alongside the inspected files. No historical script, operational-amplifier subcircuit body, PDK, ReRAM device model, paper PDF/figure, original weight file, or unpublished manuscript is copied into this repository. Algorithmic equations and circuit interfaces are implemented in newly written code and attributed to the HCST research below. Optional external-source adapters read an authorized copy supplied by the user at runtime; this project's license grants no rights over that copy.

The HCST method is attributed to Shuchao Gao and Takashi Ohsawa, particularly the JJAP article DOI [10.35848/1347-4065/ad1895](https://doi.org/10.35848/1347-4065/ad1895). This repository is a computational implementation and educational framework, not a claim that the original publication artifact, chip, or full experiment has been reproduced.

NumPy, pytest, PyTorch, setuptools and wheel are dependencies, not vendored source. Their own licenses govern those packages. The optional MNIST downloader uses the public PyTorch mirror and verifies known archive checksums; this repository does not bundle MNIST or grant dataset redistribution rights. HSPICE is an optional separately licensed commercial tool. Device/subcircuit models must be obtained separately under applicable permissions. No license, server configuration or credential is included.

Original JJAP Fig. 13 in docs/figures/jjap-fig13.png is excluded from MIT. Copyright 2024 The Japan Society of Applied Physics; reused under original-author rights with credit and DOI. See docs/FIGURE_RIGHTS.md.
