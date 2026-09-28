# Documentation · from CFD inputs to refined fields

[Start with the project overview](../README.md) for the fidelity-lift objective,
real Sod gallery, installation, and workflow selection.

## Reproduce a packaged workflow

| Guide | First action | Evidence boundary |
| --- | --- | --- |
| [Sod](../problems/1d_sod/README.md) | Predict with the supplied MLP | Six packaged pressure cases; density gallery is reproducible |
| [Lax](../problems/1d_lax/README.md) | Predict with the primary MLP | Residual branch needs separate weights |
| [Woodward–Colella](../problems/1d_woodward_colella/README.md) | Predict case 55 from supplied arrays | Original CFD batch generator is absent |
| [2D Config 3](../problems/2d_riemann/README.md) | Evaluate the software check | Bootstrap, two epochs, training-array evaluation |

Each guide describes grids, stencil inputs, transforms, artifact locations,
and optional regeneration/training separately from checkpoint inference.

## Understand the figures

- [Sod gallery](assets/promo/README.md): checkpoint, data sources, density
  arrays, error-profile interpretation, and memory-gated regeneration.
- [Upload manifest](UPLOAD_MANIFEST.md): package provenance and LFS retrieval.
- [Training-tokenization diagram](assets/nnlci-training-tokenization.png):
  conceptual input/label construction, not a numerical benchmark.
- [LF/HF concept image](assets/nnlci-lf-to-hf.png): illustrative refinement
  objective, not output from the packaged checkpoint.
- [Common utilities](../common/README.md): reusable components and compatibility
  considerations for new experiments.

## Historical material

[The intro draft](README_INTRO_DRAFT.md) is superseded.
[Legacy getting started](getting_started_legacy.md) and the
[legacy README](legacy/README_legacy.md) preserve historical paths and setup
assumptions. They are archival references, not the current installation guide.
