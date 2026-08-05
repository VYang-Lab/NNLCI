# 1D Woodward–Colella blast wave

The repository contains precomputed 100/200/1600-point arrays. The original
CFD batch generator is not included. The local CLI runs the maintained training
and prediction scripts:

```bash
python3 problems/1d_woodward_colella/src/run.py train --epochs 10000 --model best_model_800.pth
python3 problems/1d_woodward_colella/src/run.py predict --model best_model_800.pth
```

- Supplied arrays: `data/raw/`
- Current and legacy weights: `models/`
- Prediction arrays and figures: `results/`
