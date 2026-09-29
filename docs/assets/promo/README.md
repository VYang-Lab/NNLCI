# Visual gallery · independent CFD, workflow, checkpoint evidence

[Project overview](../../../README.md) · [Figure tools](../../figure_tools/README.md) ·
[Documentation](../../README.md)

## 2D fields and the motion open

![Independent LF CFD and packaged HF density, shared physical crop and scale](config3_independent_resolution.png)

**LF1 and LF2 are separate CFD simulations.** They start from the repository's
standard Config 3 quadrant states, using 100 and 200 intervals per axis
(101 × 101 and 201 × 201 nodes), γ = 1.4, dt = 0.0001, t = 0.25 and transmissive
boundaries. Both use a serial CPU adapter of the repository's CUDA-Fortran
characteristic WENO5-JS / RK3 kernels. No HF file enters either solve.
[Fields, configurations, logs and hashes](../../../problems/2d_riemann/data/promo_cfd/)
are separate from the unchanged bootstrap/training data.

The **HF target** is the existing 801 × 801
[`flow_standard_config3.dat`](../../../problems/2d_riemann/data/raw/flow_standard_config3.dat),
with nominal setup from the repository's standard configuration. Its original
execution log is unavailable, so exact historical run settings are not independently
verified. It was neither regenerated nor produced by NNLCI in this pass.
The CPU adapter preserves numerical expressions and changes execution/storage;
GPU/CPU parity could not be tested without NVIDIA hardware. This is one CFD case
for the figures, not a new training ensemble or held-out NNLCI validation.

### Two animations

The landing page separates a **12-second visual comparison** from a
**42-second method lesson**. The lesson uses a Graphviz neural-network layout
and OpenSCAD water-drop meshes, composed with SVG/Python on CPU. Both animations
retain consistent colors, typography and real CFD sources. No GPU is needed.
[Current render provenance](followup_provenance.json) records input,
renderer and output hashes, memory readings, versions and exact durations.

The [opening comparison](nnlci_comparison_poster.png) has only three columns:
LF1 with its nine-point patch, LF2 with its nine-point patch, and the **HF
output target**. The latter is a reconstruction objective, not a measured
NNLCI prediction. The density views share x/y = 0.26–0.52 and one global linear
scale. They use 160 common contour levels interpolated from native CFD values;
there is no blur, sharpening, filtering or HF downsampling to create LF.
The small reference overview locates the comparison window in the full domain.
[Permanent still](nnlci_comparison_poster.png) · [Editable SVG](nnlci_comparison_poster.svg).

Both LF insets contain exactly nine locations at the same physical offsets,
±0.01 around the point of interest. Colors are the nearest native density
samples. The nearest node can change as the point moves; the CFD solution is
stationary. These patches illustrate local alignment. They are not a literal
rendering of the bootstrap preprocessing, which bilinearly upsamples trimmed
fields and extracts dilation-4 stencils. That implementation is unchanged.

The [method animation](../../../README.md#how-it-works) teaches one step at a time.
The first five chapters hold for 4.2–9 seconds each. Water drops descend onto the tilted LF planes,
lift and move to a second point, then land again. The values fill the input vector
in groups of nine. Current-like pulses cross the neural layers, gather four
variables, and place a single point on the target grid. The final full field holds
for 4.5 seconds.

| Chapter | What to watch | Editable SVG |
| --- | --- | --- |
| 1. Align | Lit 3D water drops select two aligned 3 × 3 neighborhoods | [Step 1](nnlci_method_step_1.svg) |
| 2. Collect | Each of nine locations holds density, x/y velocity and pressure | [Step 2](nnlci_method_step_2.svg) |
| 3. Assemble | 36 values from each LF become 72 inputs | [Step 3](nnlci_method_step_3.svg) |
| 4. Predict | Electric pulses traverse the MLP, then place four variables at one point | [Step 4](nnlci_method_step_4.svg) |
| 5. Train | HF labels and predicted values meet at the loss; the loss updates weights | [Step 5](nnlci_method_step_5.svg) |
| 6. Repeat | Place outputs at their coordinates; illustrate full-field assembly | [Step 6](nnlci_method_step_6.svg) |

[Motion-free storyboard](nnlci_method_storyboard.png) ·
[Final full field and interaction detail](nnlci_method_final.png) ·
[Motion and verification review](SIGNAL_REVIEW.md).

The network uses the installed **Graphviz** tool: its exported node layout drives
the animated connections. Node/layer counts are
reduced for teaching; this is not a plot of learned activations or the literal
10 × 600 hidden-layer architecture in the Config 3 training script.
[Editable graph](../../../scripts/figure_tools/neural_signal.dot) ·
[Graphviz SVG](signal_geometry/neural_signal.svg).

The water drops are closed **OpenSCAD** meshes with real x/y/z displacement and
a changing camera angle. They mark sampling locations, not simulated liquid.
Diffuse, specular and rim shading give the drops volume; soft contact shadows
change with height, and the field slabs cast shadows. Lighting affects display
geometry, while the CFD surface textures retain their original density colors.
[Editable drop](../../../scripts/figure_tools/water_drop.scad) ·
[Exported 3D mesh](signal_geometry/water_drop.off).

The illustrated feature order is LF1 then LF2, with four variable blocks of
nine locations each. **2 × 9 × 4 = 72** appears once, in the assembly chapter.
HF supervision reaches only the training loss; it does not feed the input
vector. The final sweep reveals all **801 × 801 = 641,601** native reference
locations, retaining all previous values between temporally skipped frames.
It is a schematic reveal of reference values, not inference or physical time.
Both LF patches stay blank near boundaries where a complete stencil is
unavailable; displaying boundary reference values makes no neural boundary
prediction claim. The final frame retains the complete HF field and enlarged
interaction detail accepted in the earlier presentation.

The packaged Config 3 checkpoint remains a two-epoch software check whose
evaluator reads training arrays. Its bootstrap data remains separate from the
independent LF solves. No new scientific training ensemble or validated 2D
model output is claimed. For a real train-to-predict path, use the
[new-user Sod walkthrough](../../first_training_run.md).

### Reproduce

With Graphviz, OpenSCAD, NumPy, Matplotlib, Pillow and CairoSVG installed
([setup](../../figure_tools/README.md#setup-on-debian--ubuntu)) and LFS fields retrieved,
run from the repository root:

```bash
python3 scripts/render_promo_followup.py
python3 scripts/check_promo_followup.py

# Permanent PNG alternatives; removes stale GIFs and updates README embeds.
python3 scripts/render_promo_followup.py --static-only
```

The static gate is 700 MiB effective available RAM. GIFs require **2,500 MiB**,
checking host memory and ancestor cgroup limits before encoding each animation.
The two GIFs are encoded sequentially at 960 × 540, retaining at most 190 indexed
frames (<94 MiB) and one transient raster frame. Static SVG/PNG artifacts are
always produced. An isolated `--output-dir .smoke/promo-static` render does not
change the README or published assets.

To independently rerun LF CFD without overwriting the saved sources:

```bash
python3 scripts/regenerate_promo_lf.py --output-dir .smoke/followup-cfd
python3 scripts/check_promo_followup.py --rerun-cfd-dir .smoke/followup-cfd
```

This optional source-verification step needs gfortran and first runs the CPU
solver's existing physical-invariant, bounds and floating-point checks. In the
preceding PR #8 verification, both full solves reproduced
the saved LF fields **byte-for-byte**. The rerun is provenance verification,
not a model benchmark. [Verification record](FOLLOWUP_REVIEW.md).

### Earlier presentation and scientific views

PR #7's [OpenSCAD/Graphviz animation](nnlci_local_to_hf.gif),
[mechanism SVG](nnlci_local_mechanism.svg), and
[provenance](gatech_provenance.json) are retained as historical artifacts.
Their original renderer is `scripts/render_gatech_promo.py`; it does not
produce the current two-animation presentation. The earlier
`check_promo_poi.py` checks PR #7's archived assets;
use `check_promo_followup.py` for the current page.

The [resolution comparison](config3_independent_resolution.png) retains physical
axes and a colorbar ([SVG](config3_independent_resolution.svg),
[PDF](config3_independent_resolution.pdf)). The
[density/pressure view](config3_contour_hero.png) and
[full-domain reference](config3_full_domain.png) retain unchanged source data.
The [earlier review record](VISUAL_REVIEW.md) documents their original rendering.

## Sod · packaged checkpoint evidence

The Sod figures are CPU inference outputs from `best_model_sod_final.pth` and
the packaged Sod fields. They visualize **CFD-informed fidelity lift**. They
contain no measured wall-clock or FLOP comparison.

## Read the comparison

![Checkpoint result · Sod P1.25: LF1, refined prediction (HF-like), HF reference and signed error](sod_P1.25_density_overlay.png)

The amber baseline is **LF1 (50 points)** interpolated onto the 800-point
grid. The teal NNLCI prediction uses **both LF1 and LF2 (100 points)**; the
dark HF curve is the supplied 800-point reference. The lower panel subtracts
HF point by point, exposing local differences that a full-field overlay can
hide. It is a density error profile, not a conservation or stability test.

**Checkpoint result · six packaged Sod cases**

![Checkpoint results: LF1, NNLCI refined predictions (HF-like), and HF references at six Sod pressures](sod_pred_density_sweep.png)

Axes are shared across cases. The CFD generator uses training pressures
1.0–5.0 in increments of 0.5. P1.25, P2.25, P3.75, and P4.25 are distinct
prediction pressures inside that range; P5.50 and P6.00 are outside it.
This describes the supplied case construction, not independent verification
of every checkpoint's training history or generalization to other flows.

[Visual review and schematic regeneration](VISUAL_REVIEW.md)

## Artifacts and provenance

| Artifact | Purpose |
| --- | --- |
| [Triptych](sod_P1.25_fidelity_lift_triptych.png) | LF1, NNLCI, HF on shared position/density axes |
| [Permanent strip](sod_P1.25_fidelity_lift_strip.png) | Static comparison available with or without animation |
| [Density arrays](arrays/) | `x.npy` and P1.25 LF1, NNLCI, HF; each has 800 values |
| [Source hashes](provenance.json) | Checkpoint/raw-data SHA-256 hashes, case list, transforms, library versions |

When memory permits, regeneration also writes
`sod_P1.25_fidelity_lift.gif`: three comparison frames of the same physical
case, **not time evolution or iterative solver steps**. Primary Markdown
images use permanent PNGs so low-memory regeneration never breaks the page.

The renderer mirrors [Sod prediction preprocessing](../../../problems/1d_sod/src/prediction/predict.py):
linear LF alignment, training-LF1 z-score statistics, two three-point
stencils, and LF1 values at the first/last prediction points. HF is loaded
for comparison, not as a network input. Density is shown here; inspect the
[problem result](../../../problems/1d_sod/results/predictions/P1.25.png) for
all three output variables.

## Replot saved P1.25 arrays (preferred)

```bash
MPLBACKEND=Agg python scripts/replot_sod_promo.py
```

This changes only the triptych, permanent strip, and signed-error overlay layout.
It reads the saved arrays without inference or smoothing, preserves all original
source hashes, and records renderer versions and array hashes in `provenance.json`.
It checks host and process-cgroup memory; output is always static PNG.

## Regenerate without training

From the repository root, after installing dependencies and retrieving LFS:

```bash
MPLBACKEND=Agg python scripts/promo_from_sod_results.py
# Optional: explicitly request only static output
MPLBACKEND=Agg python scripts/promo_from_sod_results.py --static-only
```

The script uses Agg, CPU inference, and two PyTorch threads. It checks
`free -m` before rendering and also respects a tighter cgroup v2 memory limit.
If effective available RAM drops below **2,500 MiB**, animation is skipped
and any stale GIF removed; the PNG strip remains. PNG plots are deliberately
small and figures are closed after saving. No CFD solves or training occur.

Regeneration overwrites this gallery's PNGs, arrays, and provenance file.
Source CFD data and model weights are only read. Conceptual images elsewhere
under `docs/assets/` are illustrative and are not outputs of this script.
