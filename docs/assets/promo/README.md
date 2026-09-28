# Visual gallery · fields, concept, checkpoint evidence

[Project overview](../../../README.md#results-from-this-repository) ·
[Sod workflow](../../../problems/1d_sod/README.md) ·
[Documentation](../../README.md)

## 2D fields and the motion open

![Illustrative field visualization: Config 3 density and pressure interaction detail from the packaged CFD bootstrap source](config3_contour_hero.png)

The contours read the actual `X, Y, D, U, V, P` values in
[`flow_standard_config3.dat`](../../../problems/2d_riemann/data/raw/flow_standard_config3.dat).
They show **the CFD field used to construct the bootstrap**, not NNLCI output.
The 801 × 801 node domain spans 0–1 in both directions. The hero focuses on
x = 0.24–0.54, y = 0.22–0.52; the [full-domain view](config3_full_domain.png)
marks this window. Both exports use identical, linear, full-domain scales for
each variable: inferno for density, viridis for pressure. No smoothing or
alteration of the field values is applied. Contour interpolation is a display operation.

The [7.52-second motion open](nnlci_local_to_hf.gif) leads from the refinement
goal to two local inputs, NNLCI, and field assembly. Only the layout and
highlights animate; there is no physical time axis or solver iteration.
The [permanent poster](nnlci_motion_poster.png) and
[local→HF concept strip](nnlci_local_to_hf_concept.png) remain available without motion.
**Schematic / Conceptual:** the LF tiles are stride-8/stride-4 samples of
the same CFD density field (matching bootstrap subsampling), and the goal
image uses its full-resolution values. This illustrates the objective; it is
not a measured LF→NNLCI→HF comparison. The
[mechanism figure](nnlci_local_mechanism.png) keeps HF supervision at training loss.

**Why not contour `predictions_config3.npy`?** It contains 291,195 sampled
four-variable rows, not a square spatial grid, and the packaged arrays do
not save their spatial index map. Its density channel also includes negative
values. Reshaping the rows or filling unspecified locations would invent a
prediction field. The inspected array's hash, shape, and density range are
recorded in [Gatech-pass provenance](gatech_provenance.json).
The [2D guide](../../../problems/2d_riemann/README.md) documents the two-epoch
bootstrap and default evaluation on training arrays. These visuals establish
neither held-out accuracy nor conservation, stability, generalization, or speedup.

### Reproduce this opening

With NumPy, Matplotlib, and Pillow installed, run from the repository root:

```bash
git lfs pull --include='problems/2d_riemann/data/raw/flow_standard_config3.dat,problems/2d_riemann/results/NNLCI_Output/predictions_config3.npy'
python scripts/render_gatech_promo.py
```

The renderer checks host and ancestor-cgroup memory before drawing and again
before the GIF. It uses Agg/Pillow on CPU, 240-dpi PNGs, and a 960 × 430 indexed
GIF with 21 frames. Below 2,500 MiB available RAM, or with `--static-only`, it
removes its old GIF and changes the first README image to the permanent PNG.
The concept strip always remains in place. There is no training, inference,
CFD regeneration, or GPU dependency.

[gatech_provenance.json](gatech_provenance.json) records the source and output
SHA-256 hashes, coordinate checks, crop, color scales, library versions, and
memory readings. It is separate from the unchanged Sod inference provenance.
[VISUAL_REVIEW.md](VISUAL_REVIEW.md) maps the reviewer requests to the delivered assets.

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
