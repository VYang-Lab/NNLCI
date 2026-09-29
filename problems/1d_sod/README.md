# Sod · refine a coarse shock-tube solution

[Project overview](../../README.md) · [Installation](../../README.md#installation) · [Choose a workflow](../../README.md#choose-a-workflow)

Lift two inexpensive Euler solutions toward an 800-point reference with the
packaged GELU MLP. This is the entry workflow for reproducing the repository's
CFD-informed fidelity-lift gallery.

**First time training?** Follow the [hands-on walkthrough](../../docs/first_training_run.md)
for data checks, a separate smoke checkpoint, prediction with your new weights,
and comparison with the supplied model.

## Data contract

| Stage | Packaged recipe |
| --- | --- |
| LF inputs | 50 and 100 points, linearly aligned to 800 target coordinates |
| Features | Two three-point stencils × density, velocity, pressure = 18 |
| Output | Density, velocity, pressure at each interior target point |
| Transforms | Z-score statistics rebuilt from aligned training LF1 fields |
| Boundaries | First and last prediction points copied from aligned LF1 |

## Run the packaged checkpoint

After installation and `git lfs pull`, run from the repository root:

```bash
MPLBACKEND=Agg python problems/1d_sod/src/prediction/predict.py
```

Reads [raw fields](data/raw/) and
[best_model_sod_final.pth](models/best_model_sod_final.pth), prints per-case
errors, and writes [prediction figures](results/predictions/). No training or
CFD generation is needed. Training files are needed to rebuild normalization;
prediction HF fields supply evaluation references, not network inputs.

## Inspect the result

![Packaged Sod P1.25 density comparison](../../docs/assets/promo/sod_P1.25_fidelity_lift_triptych.png)

The displayed LF curve is the 50-point input aligned to the target grid; the
network uses both LF levels. See the [gallery guide](../../docs/assets/promo/README.md)
for error profiles, the six-case sweep, saved arrays, and regeneration.
The [original P1.25 figure](results/predictions/P1.25.png) also shows velocity
and pressure; [publication figures](results/publication/) are retained separately.

Prediction cases are distinct from generator training pressures. This alone
does not establish performance on different solvers, boundary conditions, or
flow families. Gallery figures report no runtime or FLOP speedup.

## Optional rebuild from CFD

The following steps generate data and train; they are unnecessary for the
checkpoint demo. Use a separate model filename for smoke tests.

The CFD script compiles `solver.cpp`, varies the left pressure, and generates
45 synchronized solutions: 9 training cases and 6 prediction cases, each on
50-, 100-, and 800-point grids.

### Step 1.1: Generate CFD Data

```bash
python problems/1d_sod/src/cfd/generate_data.py
```

- **Reads:** `problems/1d_sod/src/cfd/solver.cpp`
- **Generates:** the executable under `src/cfd/bin/` and files such as
  `train_P1.00_50.dat` and `pred_P1.25_800.dat` under
  `problems/1d_sod/data/raw/`

Optional data-quality check:

```bash
python problems/1d_sod/src/preprocessing/check_data_quality.py
```

### Step 1.2: Train

Preprocessing is performed inside the training script. The 50- and 100-point
solutions are interpolated to the target coordinates, dual three-point
stencils are extracted, and an 18-feature MLP is trained.

```bash
python problems/1d_sod/src/training/train.py
```

- **Reads:** `data/raw/train_P*_50.dat`, `train_P*_100.dat`, and
  `train_P*_800.dat`
- **Generates:** `problems/1d_sod/models/best_model_sod_final.pth`
- **Default:** 4000 epochs. For a quick smoke test, run
  `NNLCI_EPOCHS=1 NNLCI_MODEL_FILENAME=smoke_model_first_sod.pth python problems/1d_sod/src/training/train.py`.
  Pass that same `NNLCI_MODEL_FILENAME` to prediction to evaluate the new weights.

### Step 1.3: Predict and Visualize

```bash
python problems/1d_sod/src/prediction/predict.py
```

- **Reads:** prediction `.dat` files and `models/best_model_sod_final.pth`
- **Generates:** prediction figures under
  `problems/1d_sod/results/predictions/`
- **Custom outputs:** set `NNLCI_RESULTS_DIR=.smoke/first-sod/predictions` to
  preserve supplied figures. Relative paths resolve from the working directory.
