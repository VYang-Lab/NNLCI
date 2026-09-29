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

### Presentation and motion

The [4.41-second opening](nnlci_local_to_hf.gif) uses three equal-sized density
tiles with the same x/y = 0.30–0.50 window and one linear color scale spanning
all three complete fields. Each display pixel uses the nearest native node.
LF2 therefore visibly retains its coarse samples and broader features relative
to HF; no blur, artificial sharpening, per-panel rescaling or HF-derived LF data
is used. Counts identify intervals per axis, not node counts.

Only stencil and signal highlights move. Field values remain fixed at the final
physical time. The dashed arrow points to an **HF target**, not an inferred
prediction. Short labels replace the previous animated headlines and paragraphs.
The [poster](nnlci_motion_poster.png) and [permanent strip](nnlci_local_to_hf_concept.png)
show the same workflow without motion. These are workflow schematics using actual
CFD imagery, not a measured NNLCI reconstruction.

The rebuilt [mechanism](nnlci_local_mechanism.png) uses three stages and a separate
training lane. Its 72 colored scalars correspond to two nine-point neighborhoods
with four variables. HF reaches only the loss. [SVG](nnlci_local_mechanism.svg)
is the editable vector master; [PDF](nnlci_local_mechanism.pdf) and PNG are exports.
The resolution comparison also has [SVG](config3_independent_resolution.svg) and
[PDF](config3_independent_resolution.pdf) exports; axes/text are vector, dense fields rasterized.

The [density/pressure hero](config3_contour_hero.png) still shows the packaged
HF/bootstrap source, in x = 0.24–0.54, y = 0.22–0.52. The
[full-domain view](config3_full_domain.png) locates that window. These contour
views use unchanged CFD values and each variable's full-domain min/max; standard
contour interpolation is only a display operation. They are labeled as source
visualizations inside each image.

The packaged Config 3 model remains a two-epoch software check whose evaluator
reads training arrays. Its coordinate-free prediction table is not used for dense
field artwork. No new inference, training, accuracy or speedup claim is made.
The [review record](VISUAL_REVIEW.md) and [render provenance](gatech_provenance.json)
record the delivered assets and their limits.

### Reproduce

After retrieving LFS data and installing NumPy, Matplotlib, Pillow, CairoSVG and
gfortran (see [tool setup](../../figure_tools/README.md)), run from the repo root:

```bash
# Optional: independently regenerate both LF fields (~2 minutes on this host).
# Runs compiler diagnostics + physical-invariant checks before the full solves.
python3 scripts/regenerate_promo_lf.py

# Render from saved LF/HF fields; this does not rerun the solver.
python3 scripts/render_gatech_promo.py
# Or explicitly choose the permanent PNG opening:
python3 scripts/render_gatech_promo.py --static-only
```

The solver and static renderer require at least 700 MiB effective available RAM;
the GIF gate is 1,500 MiB. Host and ancestor-cgroup availability are checked.
Animation is bounded to 40 indexed 1080 × 430 frames (about 18 MiB of index data),
with no high-resolution RGB frame stack. If GIF generation is deferred, the
renderer removes the outdated GIF and points the first README image at the PNG.
No NVIDIA GPU is used. The original solver and bootstrap remain available.

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
