# 1D Lax shock tube

The primary MLP uses dual-fidelity five-point stencils. A residual-learning
ResNet branch is retained as a separate training/prediction path.

```bash
python3 problems/1d_lax/src/cfd/generate_data.py
python3 problems/1d_lax/src/training/train_mlp.py
python3 problems/1d_lax/src/prediction/predict_mlp.py
python3 problems/1d_lax/src/training/train_resnet.py
python3 problems/1d_lax/src/prediction/predict_resnet.py
```
