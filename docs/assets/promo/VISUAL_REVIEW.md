# Promo review · equal patches and POI assembly

2026-09-29 · follow-up to merged [PR #4](https://github.com/LADDER-Net/NNLCI/pull/4)
(`091e20d`, merge `094cf1d`) · branch `promo/readme-redesign`.
This is the implementation author's review, not reviewer acceptance or
independent scientific validation. The earlier PR #4 record is retained below.

## lyao76 follow-up

The former opening used equal outer field tiles but a native 3 × 3 highlight
on each resolution. LF2's highlight was half the physical width and height.
Both highlights now use **one fixed 0.04 × 0.04 physical footprint**, with
**5 × 5 LF1 versus 9 × 9 LF2 native nodes**, including the patch boundary.
The enlarged insets also share display dimensions. The mechanism's aligned
model stencils have matching support and retain the 72-feature input contract.

The new sequence drops a shared POI onto both tilted LF planes, visits a sharp
transition and a smooth region, rotates into a 2D view, then fills a puzzle-like
field with grouped point predictions. The [poster](nnlci_motion_poster.png) and
[three-scene storyboard](nnlci_local_to_hf_concept.png) preserve the explanation
without animation. The [GIF](nnlci_local_to_hf.gif) is 1200 × 750, 15,830 ms,
86 encoded frames (132 rendered frames, with identical holds coalesced), and
2,205,883 bytes.

OpenSCAD 2021.01 exports the actual plane, frame, native-sample marker and POI
meshes used by the CPU compositor. Graphviz 2.42.4 lays out the inference strip
shown in every frame. Editable SCAD/DOT sources and small OFF/SVG exports are
committed. These tools contribute directly to the delivered figures. See the
[tool guide](../../figure_tools/README.md) and [provenance](gatech_provenance.json)
for commands, versions, source/export hashes and memory readings.

## Evidence and resource checks

- Both LF fields are the unchanged, independently solved Config 3 CFD fields
  from PR #4. Native nearest-node colors use one crop and density scale.
- The prediction mosaic is explicitly labeled “Schematic · illustrative colors”
  and “No model output shown.” It uses a documented piecewise analytic teal
  motif; neither packaged HF values nor checkpoint predictions enter it.
  Packaged HF remains in the separate scientific comparison only.
- No new training, inference, CFD solves, accuracy metrics or speedup claims.
  The model, training arrays, supplied Sod results and CFD source data are unchanged.
- Effective available memory: 5,652 MiB before rendering, 5,357 MiB before GIF
  generation; gates remain 700 / 1,500 MiB. External tools run sequentially with
  fresh checks. At most 132 indexed frames occupy about 113 MiB; RGB animation
  frames are transient. CPU only; no NVIDIA GPU or Blender required.

## Validation

- Checked all 37 distinct POIs actually visited in the rendered sequence:
  identical closed .04-square physical bounds, 5 × 5 / 9 × 9 native nodes,
  and a 2:1 spacing ratio. Checked OpenSCAD plane and patch dimensions.
- Verified that the sharp-region density variation exceeds the smooth-region
  variation in each actual LF source; fixed POIs are selected without HF input.
- Checked descending POIs, monotone camera flattening, and completed assembly.
- Executed `--static-only`: stale GIF removed, README switched to the permanent
  poster, storyboard retained. Then regenerated the full motion output.
- Decoded every GIF frame and checked dimensions/duration; viewed landing,
  sharp/smooth patches, intermediate rotation and assembly at full and reduced
  widths. Main shapes remain legible at 390 px; scientific annotations need
  full-size viewing. Inspected the updated mechanism and permanent storyboard.
- Checked source/output hashes, native mesh validity, SVG XML, PNG decoding,
  PDF headers, local Markdown file targets, Python syntax and diff whitespace.

## Reproduce this follow-up

```bash
python3 scripts/render_gatech_promo.py --static-only
python3 scripts/render_gatech_promo.py
```

No CFD regeneration is required. GitHub main already contained `091e20d` at task
start, so this branch needs a new follow-up PR. Repository visibility is retained;
this task sends no email and does not merge the PR.

---

## Historical PR #4 record · Gatech visual and data review

2026-09-28 (America/New_York) · baseline `c699f44` · `promo/readme-redesign`.
This is the implementation author's review, not independent scientific validation
or reviewer acceptance.

### Requests and delivered changes

| Raw email request | Delivered |
| --- | --- |
| ishe3: distinguish LF2 from HF | Equal-sized density tiles with native-node display, one physical crop and shared color scale. Independently solved LF2 has visibly coarser samples and broader features than the packaged HF reference. No artificial blur/sharpening. |
| bjin60: much less text in the first animation | Removed headlines, numbered sentences and explanatory paragraphs. Short LF1/LF2/HF-target/NNLCI labels remain; sampling and signal highlights carry the motion. |
| bjin60: generate LF directly from CFD | Solved standard Config 3 independently at 100 and 200 intervals per axis to t = 0.25, using CPU execution of the existing Fortran numerical kernels. No HF input to either solver process. Fields, namelists, logs and hashes are committed separately from the bootstrap. |
| bjin60: completely redesign the attached mechanism | New three-stage SVG composition: aligned stencils, 72 scalar features, learned map and four-component refined value. A quiet lower training lane routes HF only to loss. Editable SVG, PDF and 2400 × 1260 PNG. |
| lyao76: scriptable open-source tool workflow | Installed and tested OpenSCAD, Graphviz, FreeCAD, draw.io, SVG/Python and Blender. Two examples per desktop tool; 34 validated exports and a preview sheet. Godot deferred as optional. [Tool guide](../../figure_tools/README.md). |

### CFD provenance and limits

The [CPU adapter](../../../scripts/build_cpu_cfd.py) mechanically copies the
repository's CUDA-Fortran kernels, replacing device storage/thread dispatch
with host arrays and serial loops. Ordered boundary sweeps eliminate redundant
GPU writes; unused storage is initialized and invalid grid/time inputs rejected.
Characteristic WENO5-JS reconstruction, numerical flux, RK3 and the inherited
positivity-floor expressions remain intact. A pinned source hash prevents
silently accepting future solver edits. The original GPU source is untouched.

Both runs use standard Config 3 states, γ = 1.4, dt = 0.0001 and transmissive
boundaries. LF1 has 101 × 101 nodes; LF2 has 201 × 201 nodes. Counts in the images
refer to intervals per axis. The full solves completed in approximately 22 and
87 seconds on this host. These are recorded run observations, not speedups.
[CFD provenance](../../../problems/2d_riemann/data/promo_cfd/provenance.json)
includes original/adapter/generated-source hashes, compiler flags, field/config
hashes, ranges and checks. Logs reach step 2500 at t = 0.25.

Checks passed before the full solves: uniform-flow preservation, planar-wave
transverse invariance, x/y transpose symmetry, debug/release agreement for
planar and quadrant cases, array bounds and floating-point traps. Full outputs
are finite, have positive density/pressure, and have the expected coordinates.
A post-solve guard confirms both differ from subsamples of the packaged HF.
This guard is a provenance check, not an NNLCI performance metric.

No NVIDIA GPU or `nvfortran` is available, so CPU/GPU numerical parity remains
untested. The existing 801 × 801 HF field is reused and labeled as a reference;
its nominal case/time follows the repository configuration, but its original
execution log is unavailable. One case was generated for the promo, not a
multi-case training ensemble. No new HF solve, NNLCI training or inference was
performed. Bootstrap arrays, the two-epoch model, supplied Sod evidence and
all original CFD/model inputs remain unchanged.

### Figure interpretation

The opening and resolution comparison use x/y = 0.30–0.50, identical panel size,
and one global density scale across all three complete fields. Nearest-node
rendering exposes the actual sampling resolution; each source gets the same
display rule. LF data is not derived from HF. The dashed output arrow denotes
the HF **target**, not a measured network prediction.

Only sampling/signal highlights animate. The GIF is 1080 × 430, 40 frames,
4,410 ms. There is no physical time axis or solver iteration in the GIF.
The original long-text cover is gone. The PNG poster and permanent strip retain
the same short-label design. The mechanism SVG keeps training supervision
separate from inference; feature tiles actually count 2 × 9 × 4 = 72.

The existing contour hero uses the packaged HF/bootstrap source and a different,
explicitly documented larger crop; it is now behind a README details disclosure.
No coordinate-free model output was reshaped into a fabricated prediction field.

### Memory and tool checks

- CPU solve gate: 2,752 MiB effective available RAM at entry; minimum 700 MiB.
- Final render gate: 2,704 MiB before static work and 2,470 MiB before animation.
- GIF minimum: 1,500 MiB, with about 18 MiB raw indexed-frame storage and no
  high-resolution RGB frame stack. Static work requires 700 MiB.
- Static-only fallback was executed and checked: the stale GIF was removed,
  and the README opening switched to the permanent PNG. Full rendering then
  restored the GIF and refreshed output hashes.
- Figure-tool smoke runs are sequential, recheck memory before each external
  process, and use CPU/software rendering. The [report](../../figure_tools/smoke_report.json)
  records versions, commands, successful exits and export hashes. Desktop
  formats stay editable; no tool binaries are committed.

### Visual and pre-commit review

- Viewed final mechanism, resolution comparison, opening poster, contour views
  and the tool preview sheet. Corrected a clipped title and increased space
  between axes and the comparison colorbar during review.
- Viewed motion frames at 960 and 390 px. LF/HF structure and the main route
  survive at phone width; small scientific annotations require full-size viewing.
- Decoded all 40 GIF frames and confirmed dimensions and total duration.
- Validated source/output hashes, finite positive LF fields and final solver
  times, SVG XML, PDF headers, PNG decoding, all 34 tool exports, local Markdown
  file targets, Python syntax and diff whitespace.
- No invented metrics, visibility changes, email, or PR merge action.

### Repository state

GitHub reported PR #3 **already merged** before this work began, with head
`c699f44a8cc3250d2f707097909cc82af9e7896d`. The requested branch remains the push
destination. Pushing new commits to it cannot update that merged PR's code
review; a follow-up PR is needed for a new review. No replacement PR is created
by this task.

### Reproduce

```bash
python3 scripts/regenerate_promo_lf.py
python3 scripts/render_gatech_promo.py
python3 scripts/figure_tools/smoke.py
```

See the [gallery](README.md#reproduce) for dependencies and data interpretation,
and the [tool guide](../../figure_tools/README.md) for installation/export choices.
