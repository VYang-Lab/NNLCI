# 1D Sod shock tube

Pipeline: C++ Euler CFD → 50/100/800-point data → dual-fidelity three-point
stencils → GELU MLP → predictions and metrics.

```bash
# Run from the repository root inside WSL
python3 problems/1d_sod/src/cfd/generate_data.py

# Training and prediction
python3 problems/1d_sod/src/training/train.py
python3 problems/1d_sod/src/prediction/predict.py
```

- Raw CFD data: `data/raw/`
- Trained weights: `models/`
- Figures and metrics: `results/`
