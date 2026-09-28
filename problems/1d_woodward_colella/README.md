# Woodward–Colella · refine blast-wave profiles

[Project overview](../../README.md) · [Installation](../../README.md#installation) · [Choose a workflow](../../README.md#choose-a-workflow)

Start from the supplied arrays to reconstruct density, velocity, and pressure.
**The original CFD batch generator is not included.** This package supports
array-based reproduction, not regeneration of the original CFD campaign.

## Data contract

| Stage | Packaged recipe |
| --- | --- |
| LF inputs | 100 and 200 points, linearly aligned to target coordinates |
| HF target | Every second point of a 1600-point field → 800 points |
| Features | Two three-point stencils × three variables = 18 |
| Transforms | Training-derived shift, log10, then min/max scaling |
| Model | `best_model_800.pth`, 18-input MLP |

## Run the packaged checkpoint

After [installation](../../README.md#installation) and LFS retrieval, run from
the repository root:

```bash
MPLBACKEND=Agg python problems/1d_woodward_colella/src/run.py predict --model best_model_800.pth
```

Reads [supplied arrays](data/raw/) and [weights](models/best_model_800.pth),
prints case-55 errors, and writes `results/prediction_case_55.png`.
The default path selects case 55 from `training_dataset_*.npy`; the training
recipe selects indices 0, 10, …, 390. Thus this case is outside that recipe's
training subset, although its file is named “training”. This is not a claim
that every supplied checkpoint shares the same split.

The predictor rebuilds transforms using the selected training arrays,
including training HF. Preserve those arrays for reproduction. A deployment
should persist the fitted transforms and use them unchanged on new LF inputs.

## Representative result

![Existing Woodward–Colella case-55 density, velocity, and pressure comparison](results/prediction_case_55.png)

This packaged figure compares LF input, prediction, and the reference.
[Additional results](results/) and [test_metrics.csv](results/test_metrics.csv)
are historical artifacts from other evaluation paths; the default case-55
command does not regenerate that CSV or every stored array.

## Optional training from supplied arrays

Multiple legacy checkpoints and prediction variants remain in [models](models/)
and [source](src/). Match each checkpoint to its architecture and transforms;
the default entry point below uses the 800-point recipe.

The repository includes precomputed 100-, 200-, and 1600-point NumPy arrays.
The original CFD batch generator is not included, so this workflow begins
with the supplied arrays. The 1600-point solution is subsampled to form the
800-point target.

### Step 3.1: Check the Supplied Data

The training script reads these files from
`problems/1d_woodward_colella/data/raw/`:

- `training_dataset_100.npy`
- `training_dataset_200.npy`
- `training_dataset_hf.npy`

### Step 3.2: Train

```bash
python problems/1d_woodward_colella/src/run.py train --epochs 10000 --model best_model_800.pth
```

Preprocessing and dual three-point stencil extraction occur inside the
training code. The command generates
`problems/1d_woodward_colella/models/best_model_800.pth`. Use `--epochs 1`
and a different `--model` filename for a smoke test so the supplied checkpoint
is not overwritten.

### Step 3.3: Predict

```bash
python problems/1d_woodward_colella/src/run.py predict --model best_model_800.pth
```

The prediction command reads the supplied arrays and checkpoint, reports
relative errors, and writes the case-55 figure under
`problems/1d_woodward_colella/results/`. A representative result figure is
shown in the [Woodward-Colella problem README](#representative-result).
