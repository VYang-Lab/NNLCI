# Common utilities · building blocks for new experiments

[Project overview](../README.md) · [Adapt NNLCI](../README.md#adapting-nnlci)

Reuse these small components when designing a new CFD-informed fidelity-lift
workflow. The packaged problem scripts remain self-contained so that loading
existing checkpoints does not depend on a refactor of their numerical path.

| Module | Reusable pieces | Contract to preserve |
| --- | --- | --- |
| [data.py](data.py) | Z-score, min/max scaling, linear 1D interpolation, local windows | Fit transforms on training cases; preserve feature order, grid convention, and boundary handling |
| [models.py](models.py) | Configurable MLP and residual MLP | Match layer widths, activation, state-dictionary names, and prediction target |
| [metrics.py](metrics.py) | Relative L1 and L2 error | Specify reduction axes and variables; compare to an LF baseline on the same coordinates |

These utilities supply no data, checkpoints, solver adapters, or automatic
train/test split. The generic model classes are not drop-in checkpoint
loaders for every problem's custom architecture.

For a new workflow, first reproduce its alignment and stencil construction
on a small field. Verify transforms and feature ordering against the intended
model, then hold out whole physical cases for evaluation. Many local samples
from one flow field do not establish generalization to new conditions.

Use the maintained problem guides for checkpoint reproduction:
[Sod](../problems/1d_sod/README.md), [Lax](../problems/1d_lax/README.md),
[Woodward–Colella](../problems/1d_woodward_colella/README.md), and
[2D Riemann](../problems/2d_riemann/README.md).
