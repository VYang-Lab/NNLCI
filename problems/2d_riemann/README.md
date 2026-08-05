# 2D Riemann problems

The CFD solver implements characteristic-wise WENO-5 for the 2D Euler
equations in CUDA Fortran. The machine-learning path reconstructs an
800-resolution field from aligned 100- and 200-resolution solutions.

```bash
python3 problems/2d_riemann/src/cfd/generate_configs.py
python3 problems/2d_riemann/src/preprocessing/prepare_config3_bootstrap.py
python3 problems/2d_riemann/src/preprocessing/preprocess.py
export NNLCI_EPOCHS=500
python3 problems/2d_riemann/src/training/train.py
python3 problems/2d_riemann/src/prediction/evaluate.py
```

Compile the CFD solver on Linux/WSL with NVIDIA HPC SDK:

```bash
nvfortran -O3 -cuda problems/2d_riemann/src/cfd/weno_gpu_solver.f90 \
  -o problems/2d_riemann/src/cfd/bin/weno_gpu_solver
```

The provided Config 3 bootstrap is intended for software validation, not as a
scientific training dataset.
