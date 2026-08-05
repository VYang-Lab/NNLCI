# NNLCI: Neural Networks with Local Converging Inputs

NNLCI reconstructs high-fidelity solutions of conservation laws from local
stencils sampled from two lower-fidelity solutions. This repository contains
reproducible workflows for three 1D Euler problems and one 2D Riemann problem.

| Problem | Fidelity levels | NNLCI input | Prediction target |
| --- | --- | --- | --- |
| [1D Sod](problems/1d_sod/README.md) | 50 / 100 / 800 points | 18 features | density, velocity, pressure |
| [1D Lax](problems/1d_lax/README.md) | 50 / 100 / 800 points | 30 features | density, velocity, pressure |
| [1D Woodward–Colella](problems/1d_woodward_colella/README.md) | 100 / 200 / 1600 points; 800-point target | 18 features | density, velocity, pressure |
| [2D Riemann, Config 3](problems/2d_riemann/README.md) | 100 / 200 / 800 grids | 72 features | density, x/y velocity, pressure |

All commands below are run from the repository root. Linux or Ubuntu under
WSL 2 is recommended because the CFD build scripts use `g++` and
`nvfortran`.

## 0. Install drivers and environment

The 1D workflows need Python and a C++ compiler. The supplied 2D
CUDA-Fortran CFD solver additionally needs an NVIDIA GPU, NVIDIA driver, and
NVIDIA HPC SDK. The Python prediction scripts can also run on CPU, although
training is considerably faster on a CUDA-capable GPU.

### 0.1 NVIDIA GPU driver

- **Windows/WSL 2:** install a current Game Ready or Studio driver from the
  [NVIDIA driver download page](https://www.nvidia.com/download/index.aspx),
  restart Windows, and enable/install WSL 2 with Ubuntu.
- **Native Linux:** install the driver recommended for the Linux distribution,
  reboot, and verify it with:

```bash
nvidia-smi
```

Do not install a second Linux display driver inside WSL; WSL uses the Windows
host driver.

### 0.2 C++ compiler for the 1D CFD solvers

On Ubuntu or WSL Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential python3 python3-venv python3-pip git git-lfs
g++ --version
```

### 0.3 NVIDIA HPC SDK for the 2D CFD solver

`problems/2d_riemann/src/cfd/weno_gpu_solver.f90` uses CUDA Fortran and must
be compiled with `nvfortran`, which is supplied by the
[NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk). Install the SDK for
your Linux distribution, then add its compiler directory to `PATH`. The exact
version directory depends on the installed release; for example:

```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/<version>/compilers/bin:$PATH
export MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/<version>/compilers/man:$MANPATH
nvfortran --version
```

Add the two `export` lines, with `<version>` replaced, to `~/.bashrc` if they
should persist. The HPC SDK is optional when using the included 2D raw or
bootstrap data without regenerating CFD solutions.

### 0.4 Python environment

Create one environment for all four workflows:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For GPU training, first install the PyTorch build matching the system from the
[official PyTorch selector](https://pytorch.org/get-started/locally/). For
example, if that selector recommends CUDA 12.1:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU-only execution, use:

```bash
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 0.5 Large files after cloning

The repository routes `.npy`, `.npz`, `.pth`, `.dat`, and `.pdf` files through
Git LFS. After cloning, retrieve them with:

```bash
git lfs install
git lfs pull
```

## 1. Run the 1D Sod problem

The CFD script compiles `solver.cpp`, varies the left pressure, and generates
45 synchronized solutions: 9 training cases and 6 prediction cases, each on
50-, 100-, and 800-point grids.

### Step 1.1: Generate CFD data

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
  `NNLCI_EPOCHS=1 NNLCI_MODEL_FILENAME=smoke_sod.pth python problems/1d_sod/src/training/train.py`.

### Step 1.3: Predict and visualize

```bash
python problems/1d_sod/src/prediction/predict.py
```

- **Reads:** prediction `.dat` files and `models/best_model_sod_final.pth`
- **Generates:** prediction figures and reported errors under
  `problems/1d_sod/results/predictions/`

## 2. Run the 1D Lax problem

The Lax workflow has a primary MLP branch and an optional residual-learning
ResNet branch. The CFD generator creates 48 files: 11 training cases and 5
prediction cases at three resolutions.

### Step 2.1: Generate and verify CFD data

```bash
python problems/1d_lax/src/cfd/generate_data.py
python problems/1d_lax/src/preprocessing/verify_data.py
```

- **Reads:** `problems/1d_lax/src/cfd/solver.cpp`
- **Generates:** the compiled solver under `src/cfd/bin/` and
  `train_P*_50.dat`, `train_P*_100.dat`, `train_P*_800.dat`, plus the
  corresponding `pred_P*` files under `problems/1d_lax/data/raw/`

### Step 2.2: Train and run the primary MLP

```bash
python problems/1d_lax/src/training/train_mlp.py
python problems/1d_lax/src/prediction/predict_mlp.py
```

- **Training reads:** synchronized training `.dat` files at 50, 100, and 800
  points; preprocessing and dual five-point stencil extraction occur in the
  training script
- **Training generates:** `problems/1d_lax/models/best_model_lax_final.pth`
- **Prediction reads:** prediction `.dat` files and the trained checkpoint
- **Prediction generates:** figures and metrics under
  `problems/1d_lax/results/mlp/`

For a one-epoch check, prefix the training command with
`NNLCI_EPOCHS=1 NNLCI_MODEL_FILENAME=smoke_lax.pth`.

### Step 2.3: Optional residual-learning branch

```bash
python problems/1d_lax/src/training/train_resnet.py
python problems/1d_lax/src/prediction/predict_resnet.py
```

This branch generates/reads `models/best_model_lax_resnet.pth` and writes its
results to `problems/1d_lax/results/resnet/`.

## 3. Run the 1D Woodward–Colella problem

The repository includes precomputed 100-, 200-, and 1600-point NumPy arrays.
The original CFD batch generator is not included, so this workflow begins
with the supplied arrays. The 1600-point solution is subsampled to form the
800-point target.

### Step 3.1: Check the supplied data

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
relative errors, and writes prediction arrays/figures under
`problems/1d_woodward_colella/results/`.

## 4. Run the 2D Riemann problem (Config 3)

The 2D CFD code is a CUDA-Fortran characteristic-wise WENO-5 solver for the
Euler equations. The NNLCI pipeline aligns 100- and 200-resolution fields to
an 800-resolution field, applies shock-aware sampling, extracts dual local
stencils, and trains a deep MLP.

The standard Config 3 initial states generated by the repository are:

| Quadrant | Density | x velocity | y velocity | Pressure |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 1.5000 | 0.000 | 0.000 | 1.500 |
| Q2 | 0.5323 | 1.206 | 0.000 | 0.300 |
| Q3 | 0.1380 | 1.206 | 1.206 | 0.029 |
| Q4 | 0.5323 | 0.000 | 1.206 | 0.300 |

### Step 4.1: Generate the Fortran configuration

```bash
python problems/2d_riemann/src/cfd/generate_configs.py
```

- **Generates:** `problems/2d_riemann/configs/input_Config3_std.nml`
  (and other standard configurations supported by the script)

### Step 4.2: Compile and run the CFD solver

```bash
mkdir -p problems/2d_riemann/src/cfd/bin
nvfortran -O3 -cuda problems/2d_riemann/src/cfd/weno_gpu_solver.f90 \
  -o problems/2d_riemann/src/cfd/bin/weno_gpu_solver
python problems/2d_riemann/src/cfd/run_solver.py
```

- **Reads:** the Config 3 `.nml` file
- **Generates:** `problems/2d_riemann/data/raw/flow_standard_config3.dat`

For scientific training, solve each independently perturbed case at 100, 200,
and 800 resolutions and arrange it as:

```text
problems/2d_riemann/data/bootstrap/DATA_TRAIN/Config3/<case_name>/
├── flow_100x100.dat
├── flow_200x200.dat
└── flow_800x800.dat
```

Varying the initial pressure, density, or velocity by values such as ±5% and
±10% expands the training solution space.

### Step 4.3: Build the included software-validation dataset

To reproduce the code path using the supplied standard 801-by-801 node field:

```bash
python problems/2d_riemann/src/preprocessing/prepare_config3_bootstrap.py
```

This writes one aligned case under `data/bootstrap/DATA_TRAIN/Config3/` by
subsampling the same high-resolution solution. It validates the software
pipeline only; it is **not** a substitute for independently solved
multi-resolution CFD data in a scientific experiment.

### Step 4.4: Preprocess and sample stencils

```bash
python problems/2d_riemann/src/preprocessing/preprocess.py
```

The script:

1. reads the 100-, 200-, and 800-resolution Tecplot fields;
2. bilinearly aligns the two low-fidelity fields to the target grid;
3. prioritizes high-gradient shock regions with smart sampling;
4. extracts the dual spatial stencil and creates a 72-feature input; and
5. scales physical variables to the range used by the `Tanh` network.

It generates:

```text
problems/2d_riemann/data/processed/NNLCI_Data/Config3_Stencils/
├── train_input_config3.npy
├── train_target_config3.npy
└── x_scaler_config3.npz
```

### Step 4.5: Train the deep MLP

```bash
NNLCI_EPOCHS=500 NNLCI_NUM_WORKERS=8 \
python problems/2d_riemann/src/training/train.py
```

- **Reads:** `train_input_config3.npy` and `train_target_config3.npy` using
  memory mapping
- **Generates:**
  `problems/2d_riemann/models/NNLCI_Models/nnlci_config3_deep_latest.pth`
- **Defaults:** batch size 65536, Adam learning rate `1e-4`, and 500 epochs

For a quick check, use `NNLCI_EPOCHS=1 NNLCI_NUM_WORKERS=0`. Training uses
CUDA automatically when available.

### Step 4.6: Evaluate

```bash
NNLCI_NUM_WORKERS=0 python problems/2d_riemann/src/prediction/evaluate.py
```

The evaluator reads the processed arrays and Config 3 checkpoint, prints MAE,
MSE, relative L1/L2 error, PSNR, and SSIM when the sample shape permits. It
generates
`problems/2d_riemann/results/NNLCI_Output/predictions_config3.npy`.

## 5. Execution summary

```text
1D Sod/Lax:
CFD generation → internal interpolation/stencil preprocessing → training → prediction

1D Woodward–Colella:
supplied arrays → internal interpolation/stencil preprocessing → training → prediction

2D Riemann Config 3:
configuration → CFD at three resolutions → preprocessing/smart sampling
→ train_input_config3.npy + train_target_config3.npy
→ training → nnlci_config3_deep_latest.pth → evaluation
```

Runnable notebooks containing the same end-to-end flow are available at:

- [1D Sod, Lax, and Woodward–Colella](docs/notebooks/NNLCI_1D_End_to_End.ipynb)
- [2D Riemann Config 3](docs/notebooks/NNLCI_Config3_End_to_End.ipynb)

## 6. Further NNLCI extensions

NNLCI is modular: the supplied WENO solvers can be replaced with another CFD
code, including OpenFOAM, Ansys Fluent, STAR-CCM+, an in-house solver, or a
solver for another system of conservation laws.

### 6.1 Prepare synchronized multi-fidelity data

NNLCI is supervised. Each physical case must provide solutions at matching
times and physical coordinates:

1. **HF:** the high-fidelity ground truth and prediction target;
2. **LF1:** the first coarse-grid input; and
3. **LF2:** the second coarse-grid input.

Use the same variables, units, domain, boundary conditions, output time, and
variable ordering at all three fidelities. Create many training cases by
perturbing initial/boundary conditions (for example ±5% or ±10%), changing
geometry, or varying physical parameters. Reserve unseen cases for validation
and testing; do not split neighboring points from the same field across the
training and test sets if case-level generalization is the objective.

### 6.2 Align coordinates and extract local stencils

The NNLCI mapping is local. Interpolate LF1 and LF2 onto HF target coordinates,
then gather a neighborhood from both low-fidelity fields for every selected HF
point. In the supplied 2D Euler workflow, one sample is a 72-dimensional dual-LF
input and a 4-dimensional HF center target. The implementation is in
`problems/2d_riemann/src/preprocessing/preprocess.py`; reusable helpers are
under `common/`.

When adapting the method:

- choose a stencil large enough to capture the local wave structure while
  keeping the model input compact;
- use consistent boundary padding and flattening order;
- fit normalization parameters on training cases only and reuse those exact
  parameters for validation and prediction; and
- scale variables consistently. The provided 2D network uses a `[-1, 1]`
  scaling compatible with its `Tanh` activations.

### 6.3 Use physics-aware sampling

Dense high-resolution fields can exceed system memory and contain many
redundant smooth points. The supplied 2D preprocessor computes a gradient mask
and samples shock/high-gradient and smooth regions at different rates. Adjust
the shock and smooth sampling ratios for the new flow regime. Also inspect rare
features—contacts, shear layers, reacting fronts, wall layers, or geometric
singularities—because a density-gradient mask alone may not capture all
important physics.

### 6.4 Adapt the model and outputs

Change the network input width when the stencil, spatial dimension, or number
of physical variables changes, and change the output width to match the HF
target variables. Keep training and prediction definitions synchronized. For
large datasets, retain memory-mapped loading and tune batch size and worker
count for the available RAM and GPU memory.

### 6.5 Validate a new extension

Before a long training run:

1. verify every CFD field is finite and physically admissible;
2. plot LF1, LF2, and HF fields on common coordinates;
3. run preprocessing on one case and inspect array shapes and scaling;
4. overfit a very small sample to test the model/data connection;
5. run a one-epoch end-to-end smoke test; and
6. report per-variable and global relative errors together with visual shock
   location and conservation checks.

The Config 3 bootstrap in this repository is useful for steps 3–5, but a
scientific accuracy claim requires independently computed multi-resolution
solutions and held-out physical cases.

## Repository layout

```text
NNLCI/
├── problems/
│   ├── 1d_sod/
│   ├── 1d_lax/
│   ├── 1d_woodward_colella/
│   └── 2d_riemann/
├── common/       reusable models, data processing, and error metrics
├── docs/         notebooks and legacy documentation
├── tests/        lightweight smoke-test support
├── README.md
├── requirements.txt
└── LICENSE
```

Each problem keeps source code, data, models, and results separate. Python
paths are resolved from the script location, so the documented commands can be
run from the repository root without editing working directories.

## Citation

If you use NNLCI in research, please cite the relevant papers. Parts I and II
introduce the 1D and 2D NNLCI methods; the AIAA Journal paper demonstrates an
extension to unstructured-grid CFD.

```bibtex
@article{huang2023nnlci_part1,
  title={Neural networks with local converging inputs (NNLCI) for solving conservation laws, Part I: 1D problems},
  author={Huang, Haoxiang and Yang, Vigor and Liu, Yingjie},
  journal={Communications in Computational Physics},
  volume={34},
  number={2},
  pages={290--317},
  year={2023},
  doi={10.4208/cicp.OA-2022-0285}
}

@article{huang2023nnlci_part2,
  title={Neural networks with local converging inputs (NNLCI) for solving conservation laws, Part II: 2D problems},
  author={Huang, Haoxiang and Yang, Vigor and Liu, Yingjie},
  journal={Communications in Computational Physics},
  volume={34},
  number={4},
  pages={907--933},
  year={2023},
  doi={10.4208/cicp.OA-2023-0026}
}

@article{ding2024nnlci_unstructured,
  title={Neural network with local converging input for unstructured-grid computational fluid dynamics},
  author={Ding, Weiming and Huang, Haoxiang and Lee, Tzu-Jung and Liu, Yingjie and Yang, Vigor},
  journal={AIAA Journal},
  volume={62},
  number={8},
  pages={3155--3166},
  year={2024},
  doi={10.2514/1.J063885}
}
```

## License

See [LICENSE](LICENSE).
