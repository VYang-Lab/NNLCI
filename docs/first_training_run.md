# Your first training and prediction run

[Overview](../README.md) · [Installation](../README.md#installation) · [Sod details](../problems/1d_sod/README.md)

Start with **Sod on CPU**. It has supplied CFD data, a trained checkpoint, and
a complete training path. The 2D animation explains the method; this smaller
1D example teaches the runnable workflow. Run these Bash commands from the
repository root after installation and `git lfs pull`.

## 1. Inspect the training data

Each physical case has three files in `problems/1d_sod/data/raw/`:

| Role | Example | Used for |
| --- | --- | --- |
| LF1 | `train_P1.00_50.dat` | Coarse input, 50 points |
| LF2 | `train_P1.00_100.dat` | Second input, 100 points |
| HF | `train_P1.00_800.dat` | Training targets, 800 points |

All three must describe the same pressure case, variables and final time.
Columns are `x den vel pres`. Inspect a supplied file and check the assembled
training tensors before starting a run:

```bash
head -n 4 problems/1d_sod/data/raw/train_P1.00_50.dat
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python - <<'PY'
import importlib.util
from pathlib import Path
path = Path("problems/1d_sod/src/training/train.py")
spec = importlib.util.spec_from_file_location("sod_training", path)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
x, y, width = training.load_sod_data(training.DATA_DIR)
assert tuple(x.shape) == (7182, 18) and tuple(y.shape) == (7182, 3)
assert x.isfinite().all() and y.isfinite().all()
print("Ready:", tuple(x.shape), "->", tuple(y.shape))
PY
```

For the packaged nine training cases, this gives 9 × 798 interior points.
Preprocessing happens inside `train.py`: interpolate LF1/LF2 to 800 target
coordinates, fit per-variable z-score statistics on training LF1, and extract
two three-point stencils. Each row has **18 inputs** and **3 HF targets**.
The 2D method instead has 72 inputs and four outputs; don't mix these contracts.

## 2. Train a one-epoch software check

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NNLCI_EPOCHS=1 NNLCI_MODEL_FILENAME=smoke_model_first_sod.pth \
python problems/1d_sod/src/training/train.py
```

Expect `models/smoke_model_first_sod.pth` under `problems/1d_sod/` and a
completed training loop. This separate filename preserves the supplied
checkpoint. **One epoch checks the pipeline, not reconstruction quality.**

## 3. Predict with the checkpoint you just trained

Use the **same model filename** in prediction. Without it, prediction loads
the supplied trained checkpoint instead.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg \
NNLCI_MODEL_FILENAME=smoke_model_first_sod.pth \
NNLCI_RESULTS_DIR=.smoke/first-sod/predictions \
python problems/1d_sod/src/prediction/predict.py
```

Expect six PNGs (`P1.25.png` through the six packaged cases, including
`P6.00.png`) in `.smoke/first-sod/predictions/`, plus per-case errors in the
terminal. `NNLCI_RESULTS_DIR` is relative to your working directory when given
as a relative path. The model lives under the problem's `models/` directory.

To compare against the **supplied checkpoint**, run it into a second folder:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg \
NNLCI_MODEL_FILENAME=best_model_sod_final.pth \
NNLCI_RESULTS_DIR=.smoke/supplied-sod/predictions \
python problems/1d_sod/src/prediction/predict.py
```

Compare density, velocity and pressure against HF and the LF1 baseline.
Inspect shock position, transition width and overshoot as well as errors.
The first/last output points are copied from aligned LF1, not predicted.

## 4. Train beyond the software check

After the short run works, start a fresh longer run with a separate name:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NNLCI_EPOCHS=4000 NNLCI_MODEL_FILENAME=my_sod.pth \
python problems/1d_sod/src/training/train.py
```

Use `NNLCI_MODEL_FILENAME=my_sod.pth` for its prediction command. Runtime
depends on your CPU. The default recipe starts from scratch, saves periodically
and at completion, and does not select a checkpoint on a validation split.
The historical `best_model` filename does not imply validation-based selection.

## 5. Generate data or bring a new case

The supplied data is enough for steps 1–4. To regenerate it with the repository
CFD solver, install `g++` and run:

```bash
python problems/1d_sod/src/cfd/generate_data.py
```

This overwrites the named raw files with 45 solves: nine training and six
prediction cases at three resolutions. LF uses the generator's first-order
scheme; HF uses its third-order scheme. Edit its `doe_plan` to change pressure
cases, preserving the synchronized triples and filename format. Update the
step-1 sample-count assertion if you change the number of training cases.

For a scored new prediction case, supply `pred_P<pressure>_50.dat`,
`pred_P<pressure>_100.dat`, and `pred_P<pressure>_800.dat`, with the expected
grid sizes and columns. Keep the `train_*` files used by the checkpoint so
prediction rebuilds the same normalization. Hold out **whole physical cases**
when evaluating generalization; random neighboring points are not independent
cases.

The packaged predictor is an **evaluation script**: it enumerates cases from
their HF reference files and loads HF for plots/errors. HF values do not enter
the network. An LF-only deployment entry point would need to enumerate LF
pairs and persist/reload the training transforms; that entry point is not
provided here. Never fit transforms on the new case's HF.

## If a run fails

| Symptom | Check |
| --- | --- |
| A `.dat` or `.pth` contains `version https://git-lfs` | Run `git lfs pull`; these are pointers, not data |
| Missing training files | Retrieve all three resolutions; retain training LF1 files for normalization |
| Model not found | Match `NNLCI_MODEL_FILENAME` to the file under `problems/1d_sod/models/` |
| No prediction cases | Retrieve the six `pred_*` triples, including HF evaluation references |
| Unexpectedly poor smoke-test plots | One epoch is a software check; compare the supplied checkpoint separately |

Continue with the [four workflow guides](../README.md#choose-a-workflow).
Woodward–Colella starts from supplied arrays; its original generator is absent.
The packaged 2D data is HF-subsampled bootstrap data, and its default evaluator
uses training arrays. The independent LF simulations in the promo are kept
separately and do not turn that bootstrap into a validated training ensemble.
