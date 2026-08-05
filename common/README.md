# Common utilities

Problem-independent building blocks:

- `data.py`: Z-score and min/max transforms, 1D interpolation, local windows.
- `models.py`: configurable MLP and residual MLP components.
- `metrics.py`: relative L1 and L2 errors.

The validated problem scripts remain self-contained so that the numerical
results do not depend on an untested refactor. New experiments can import these
utilities to reduce duplication.
