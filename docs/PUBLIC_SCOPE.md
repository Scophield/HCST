# Public file and licensing review

MIT was selected for the newly authored software under the owner's delegated decision to prepare and publish the project. The owner did not individually specify MIT or approve ownership of every historical research asset. The license is not extended to those assets.

The public file scope is:

- New `src/reram` implementation, configuration, data reader, experiment orchestration, new topology renderer and measurement interface.
- New computational/operational tests and test-only dummy circuit headers.
- New optional AST adapters and audit loaders. Their source contains no embedded historical class/function bodies; those are read from a caller-authorized external directory at runtime.
- Newly authored README, Mermaid architecture diagram, method/validation/provenance notes, citation metadata, MIT license and Notice.
- Explicit configs and aggregate new 2026 computational evidence. Historical checkpoint accuracy reports, checkpoints, neuron subcircuit bodies, data, private environments, and unpublished extensions are excluded.
- A minimal CPU CI workflow using dependencies through package installation; no secret or deployment permission is needed.

The per-module source/structure relationship is documented in [PROVENANCE](PROVENANCE.md). The implementation uses mathematical reformulations and a newly separated class structure, rather than copying original script text into differently named files. There is no transplanted historical header, author comment, training loop or device-model body in the distributable files. Algorithms, functional interfaces and public research attribution are distinguished from the copyright and redistribution status of external source files.

The private pre-publication originals and review packages are retained outside this repository and are not part of its Git history. Optional package dependencies keep their own licenses. See [NOTICE](../NOTICE.md) for the precise license boundary and [REPRODUCTION_GAPS](REPRODUCTION_GAPS.md) for unfinished matching work.
