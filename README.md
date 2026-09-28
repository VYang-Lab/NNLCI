# NNLCI: Neural Networks with Local Converging Inputs

> Lift low-cost CFD solutions toward high-fidelity flow fields.

**NNLCI is a CFD-informed solution-refinement framework.** It learns to map
coarse, low-fidelity (LF) solutions to high-fidelity-like predictions of
density, velocity, and pressure. The goal is **fidelity lifting**: recover
physical structure that inexpensive CFD under-resolves, using high-fidelity
(HF) solutions as supervision during training.

This repository brings that idea into four Euler workflows: **Sod, Lax,
Woodward–Colella, and 2D Riemann Config 3**, with source, data, model
checkpoints, and evaluation scripts.

[Repository results](#results-from-this-repository) ·
[Why fidelity lifting?](#why-fidelity-lifting) · [How it works](#how-it-works) ·
[Choose a workflow](#choose-a-workflow) · [Installation](#installation) ·
[Run the demos](#run-the-demos) · [Adapt NNLCI](#adapting-nnlci) ·
[Repository map](#repository-layout) · [Citation](#citation)

## Results from this repository

These density fields come from the packaged Sod data and
`best_model_sod_final.pth`, evaluated on CPU with the same preprocessing and
inference logic as the repository's prediction script. This is **fidelity
lift from the packaged demo, not a surrogate speedup claim**.

![Sod P1.25 density: LF1, NNLCI prediction, and HF reference on a shared density scale](docs/assets/promo/sod_P1.25_fidelity_lift_triptych.png)

*Sod P1.25 · checkpoint: `best_model_sod_final.pth` · LF | NNLCI | HF.
The displayed LF baseline is LF1 (50 points), interpolated to the 800-point
target grid. NNLCI uses both LF1 and LF2 (100 points); HF is the 800-point
reference. The three panels share the same density limits.*

**Read the result:** compare wave positions and transition widths against HF;
inspect the [overlay and signed density error](docs/assets/promo/sod_P1.25_density_overlay.png)
to see where agreement holds and where differences remain. A visual match in
one variable is not evidence of conservation, stability, or generalization.

![Six packaged Sod prediction cases, comparing LF1, NNLCI, and HF density](docs/assets/promo/sod_pred_density_sweep.png)

*All six shipped prediction cases, with shared axes. P1.25–P4.25 lie inside
the generator's training pressure range (1.0–5.0); P5.50 and P6.00 lie beyond
it. These are case comparisons, not an aggregate accuracy or speed benchmark.*

[Gallery and regeneration guide](docs/assets/promo/README.md) ·
[Saved density arrays](docs/assets/promo/arrays/) ·
[Permanent static comparison strip](docs/assets/promo/sod_P1.25_fidelity_lift_strip.png)

> **Evidence boundary.** The Sod gallery uses a packaged trained checkpoint.
> Concept diagrams below are illustrative. The 2D Config 3 package is a
> **two-epoch bootstrap software check**, and its default evaluator reads
> **training arrays**, not held-out cases.

## Why fidelity lifting?

<p align="center">
  <img src="docs/assets/nnlci-lf-to-hf.png"
       alt="Low-fidelity and high-fidelity field examples illustrating the refinement objective"
       width="100%">
  <br>
  <em>The objective: lift a coarse physical representation toward a more resolved
  field. This LF/HF illustration is not a benchmark of the packaged checkpoint.</em>
</p>

A coarse CFD solution already carries physical information: wave locations,
jumps, and relationships among flow variables. It can also smear shocks or
miss finer structure. NNLCI uses this imperfect solution as the starting
point for a learned refinement.

The central question is **how much HF structure can be reconstructed from
inexpensive LF solutions?** Local neighborhoods across fidelity levels give
the model evidence about how a solution changes with refinement. Near shocks
and strong gradients, that evidence includes the shape and position of a
transition, rather than just its value at one point.

Reconstruction quality comes first. Reusing a trained mapping may reduce the
cost of repeated HF simulations, but a computational-savings claim must
account for LF CFD, preprocessing, and neural refinement together. This
landing page makes no measured speedup claim.

## How it works

The included workflows use **two LF fields, LF1 and LF2**, organized around
the same target coordinates. “Converging inputs” refers to related numerical
solutions at successive fidelity levels: their local differences expose
refinement trends for the network to learn. It does not guarantee that every
point improves monotonically or that a prediction equals the HF solution.

1. **Solve related cases at multiple fidelities.** Keep physical conditions,
   variables, and output time consistent across LF and HF fields.
2. **Align and collect local inputs.** Interpolate LF1 and LF2 to target
   coordinates and gather a stencil around each point of interest.
3. **Learn the lift.** Concatenate the LF stencil values into an input vector;
   use the HF value at the same point as the training label.
4. **Refine a new LF solution.** Apply the trained model with its training
   transforms to LF stencils and assemble point predictions into a field.

<p align="center">
  <img src="docs/assets/nnlci-training-tokenization.png"
       alt="Two local low-fidelity neighborhoods feed a regressor; the high-fidelity value supplies the training loss"
       width="100%">
  <br>
  <em>One training sample: LF neighborhoods form the input; HF supplies the
  supervisor. The mesh drawing illustrates the concept; the packaged Euler
  demos use structured grids.</em>
</p>

**No new-case HF truth enters the model's inference inputs.** HF references
are used for training and evaluation. The supplied prediction scripts are
reproduction/evaluation tools: they load reference fields to score and plot
results. Woodward–Colella also rebuilds training transforms from the supplied
training arrays, including training HF data. A standalone deployment should
save those fixed transforms and use LF data and available coordinates for
new cases.

Locality and converging inputs are the mechanism serving the refinement
objective. Each field supplies many point-centered training samples, but
those samples do not replace independent physical cases for testing
generalization. The broader idea can accommodate different input designs;
the recipes below document the specific implementations shipped here.

For a closer look at point alignment, see the
[annotated local-stencil diagram](docs/assets/nnlci-local-stencil-explained.png).

## Choose a workflow

| Problem | LF → target data | Model input | Predicted variables |
| --- | --- | --- | --- |
| [1D Sod](#1d-sod) | 50 / 100 → 800 points | 18 features; two three-point stencils | Density, velocity, pressure |
| [1D Lax](#1d-lax) | 50 / 100 → 800 points | 30 features; two five-point stencils (primary MLP) | Density, velocity, pressure |
| [1D Woodward–Colella](#1d-woodward-colella) | 100 / 200 → 800-point target subsampled from 1600 points | 18 features; two three-point stencils | Density, velocity, pressure |
| [2D Riemann Config 3](#2d-riemann-config-3) | 100 / 200 → 800 resolution in each direction | 72 features; two dilated 3 × 3 stencils | Density, x/y velocity, pressure |

**What is included, and what it establishes:**

- **Sod and Lax:** CFD generators, training/prediction paths, and supplied
  trained checkpoints. Lax also includes an optional residual-learning branch.
- **Woodward–Colella:** supplied arrays and checkpoints; the original CFD
  batch generator is not included. See its
  [existing prediction figure and metrics](problems/1d_woodward_colella/README.md#representative-result).
- **2D Config 3:** a CUDA-Fortran solver and a bootstrap dataset formed by
  subsampling one HF solution. **This bootstrap validates software, not
  scientific multi-resolution reconstruction.** The packaged model is a
  **two-epoch pipeline-validation checkpoint**, not a publication-quality
  scientific model. The default evaluator reads training arrays, so its
  metrics are not held-out accuracy results.

See the [upload manifest](docs/UPLOAD_MANIFEST.md) for package provenance.
Scientific validation requires independently solved multi-resolution cases
and held-out physical conditions.

For provenance and figure interpretation, see the [documentation index](docs/README.md).

## Installation

Run all commands from the repository root. Linux or Ubuntu under WSL 2 fits
the supplied CFD build scripts. Python prediction can run on CPU; regenerating
Sod/Lax data needs `g++`. Regenerating 2D CFD additionally needs an NVIDIA GPU,
a working driver, and NVIDIA HPC SDK (`nvfortran`). The SDK is unnecessary
when using the supplied 2D data.

<details>
<summary>Ubuntu / WSL system packages</summary>

```bash
sudo apt update
sudo apt install -y build-essential python3 python3-venv python3-pip git git-lfs
g++ --version
```

</details>

Create a Python environment and install the repository dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Retrieve the supplied data and weights after cloning. Arrays, weights, CFD
fields, and PDFs use Git LFS:

```bash
git lfs install
git lfs pull
```

<details>
<summary>Optional GPU setup and CUDA-Fortran compiler</summary>

Use the NVIDIA driver appropriate for your system. WSL uses the Windows host
driver; do not install a second Linux display driver inside WSL. Check access:

```bash
nvidia-smi
```

For GPU training, install a compatible CUDA-enabled PyTorch build in the same
environment. Compiler setup is separate: the supplied 2D solver requires
`nvfortran` from NVIDIA HPC SDK. Add your installed SDK compiler directory to
`PATH` (replace `YOUR_VERSION` with the actual directory name):

```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/YOUR_VERSION/compilers/bin:$PATH
nvfortran --version
```

</details>

Check the Python environment:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run the demos

To inspect a supplied 1D checkpoint after installation and LFS retrieval,
start with Sod prediction:

```bash
python problems/1d_sod/src/prediction/predict.py
```

This uses supplied raw data and weights, writes figures to
`problems/1d_sod/results/predictions/` and prints metrics; it does not retrain the model.
Full recipes live in the problem guides below. Training commands can overwrite
named checkpoints; use separate smoke-test filenames when checking a training path.

### 1D Sod

**Start here for a packaged fidelity-lift example.** Predict density, velocity,
and pressure with the supplied MLP. The [Sod guide](problems/1d_sod/README.md)
contains the data contract, gallery, and optional CFD/training recipe.

### 1D Lax

Use the [Lax guide](problems/1d_lax/README.md) for primary MLP inference and
its five-point stencils. The optional residual branch requires its own
weights; its default checkpoint is not packaged.

### 1D Woodward-Colella

The [Woodward–Colella guide](problems/1d_woodward_colella/README.md) starts
from supplied arrays and explains the shifted-log transforms, case-55
figure, and unavailable original CFD batch generator.

### 2D Riemann Config 3

The [2D guide](problems/2d_riemann/README.md) separates packaged software
validation from the independent CFD data needed for scientific evaluation.
The supplied two-epoch bootstrap model and training-array evaluator establish
pipeline operation only.

## Adapting NNLCI

Start with the data contract: synchronized physical cases, consistent units,
variables, boundary conditions, and output times. A different solver or mesh
requires corresponding alignment and stencil code; this package does not
provide ready-made adapters for other CFD tools.

- **Organize the lift.** Choose LF sources and a trusted HF target. Keep
  stencil size, flattening order, boundary handling, and normalization
  consistent between training and prediction.
- **Separate training from deployment.** Fit transforms on training cases and
  persist them. HF labels and HF-based sample selection belong to training
  and evaluation; prediction of a new case must use available LF information.
- **Match the model to the inputs.** Adjust input/output widths for the number
  of variables, LF levels, and stencil points. The objective remains solution
  refinement when the architecture changes.
- **Validate physical cases.** Hold out entire cases when testing
  generalization; neighboring points from one field are not independent test
  cases. Inspect shocks, contacts, and smooth regions alongside per-variable
  errors, and compare with the LF baseline.
- **Check the pipeline before scaling up.** Inspect aligned fields, finite
  values, array shapes, and transforms; run a one-epoch smoke test. Any cost
  comparison should include LF generation and the full refinement path.

## Repository Layout

```text
NNLCI/
|-- problems/
|   |-- 1d_sod/
|   |-- 1d_lax/
|   |-- 1d_woodward_colella/
|   `-- 2d_riemann/
|-- common/       optional reusable models, data processing, and metrics
|-- docs/         gallery guide, package manifest, assets, and archived docs
|-- scripts/      repository maintenance and asset-preparation scripts
|-- CITATION.cff
|-- README.md
|-- requirements.txt
`-- LICENSE
```

Each problem keeps source code, data, models, and results separate. The
[common utilities](common/README.md) are available for new experiments; the
problem scripts remain self-contained. Historical recipes under `docs/legacy/`
and `docs/getting_started_legacy.md` retain old paths; use the commands above.

Python paths are resolved from the script location, so the documented commands can be
run from the repository root without editing working directories.

## Software Authors

Jinbo Xiao, Weiming Ding, Li-Chian Yao, Yingjie Liu, Haoxiang Huang,
Vigor Yang, I-Hsuan She, and Haotian Sun.

## Citation

For software attribution, use [CITATION.cff](CITATION.cff). If you use NNLCI
in research, please also cite the relevant papers. Parts I and II introduce
the 1D and 2D NNLCI methods; the AIAA Journal paper demonstrates an extension
to unstructured-grid CFD.

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

Released under the [MIT License](LICENSE).
