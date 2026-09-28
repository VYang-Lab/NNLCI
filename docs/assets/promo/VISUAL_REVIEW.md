# Gatech feedback pass · visual review

2026-09-28 · baseline `3741986` · branch `promo/readme-redesign`.
Author visual review, not an independent reviewer acceptance or scientific validation.
The session runtime record confirms **gpt-6-astra / xhigh**. No model downgrade.

## Reviewer requests → delivered changes

| Request | Delivered |
| --- | --- |
| lyao76: higher visual quality and vibrant 2D contours | [Density/pressure hero](config3_contour_hero.png): 2400 × 1680, 240 dpi; inferno/viridis, true physical coordinates, readable numerical colorbars. [Full domain](config3_full_domain.png) locates the detail window. |
| ishe3: understand local information → high fidelity at a glance | [Concept strip](nnlci_local_to_hf_concept.png): two visibly coarse local inputs → NNLCI → resolved goal imagery. Three stages, large arrows, little prose. |
| bjin60: open with motion, then show how it works | The first README element is a [7.52-second GIF](nnlci_local_to_hf.gif). It opens on the field and fidelity-lift identity, then highlights inputs, the learned map, and the field goal. [Permanent poster](nnlci_motion_poster.png) and concept PNG remain available. |
| Airphonse: CFD-informed fidelity/resolution lift first | Opening says “Coarse CFD. Finer fields.” Local information supplies the refinement map. No runtime or speedup claim. |

The revised order is motion → core concept → 2D field hero → compact mechanism
→ packaged Sod evidence → runnable entry and details. The project heading
follows the opening figures. The old expanded alignment schematic is folded
into a details block; historical artwork remains in the repository.

The user's feedback overrides the standing spec's earlier GIF/2D demotion,
Sod-first evidence order, and previous opening storyboard. White backgrounds,
role colors, honest labels, inference separation, and source provenance remain.

## Data boundary and source choice

- The numerical contours use **the packaged CFD bootstrap source**,
  `problems/2d_riemann/data/raw/flow_standard_config3.dat`, with its original
  density and pressure values. They are labeled **Illustrative field visualization**
  and **Packaged CFD bootstrap source** inside the image.
- The detail window is x = 0.24–0.54, y = 0.22–0.52. The entire 0–1 domain is
  provided separately with the window marked. Each variable's complete source
  min/max sets the linear color scale in both views. There is no smoothing,
  invented field, hidden rescaling between views, or fabricated metric.
- `predictions_config3.npy` was inspected, not reshaped. It is a 291,195 × 4
  sampled training table without saved spatial coordinates; density includes
  negative values. A dense prediction contour would need a validated index
  map or new inference. Neither is invented here. The array's hash, shape,
  and density range are recorded in [provenance](gatech_provenance.json).
- Concept artwork uses stride-8/stride-4 samples and full-resolution values
  from **one** source field. All such artwork is labeled **Schematic / Conceptual**;
  the output is **HF-like goal**, not measured NNLCI reconstruction. The conceptual
  imagery has no numerical colorbar. The independent contour panels do.
- Config 3 remains a two-epoch bootstrap software check whose default evaluator
  reads training arrays. The top has one short evidence-boundary callout.
  No held-out 2D, exactness, conservation, stability, generalization, or speed claim.
- Sod checkpoint figures and their original arrays/provenance are unchanged
  and appear below the mechanism. No inference, training, or CFD was run.

## Motion and memory

Before final rendering, `free -m` and the process cgroup/ancestors reported
5,718 MiB effective available RAM; the check before animation reported
5,558 MiB. The process cgroup was unlimited, so host availability governed.
No NVIDIA GPU is available or required.

The GIF has 21 indexed frames at 960 × 430, lasts 7,520 ms, and is 500,866 bytes.
It is a conceptual presentation sequence: clean fades, numbered explanations, and stage highlights,
not physics evolving, an iterative corrector, or progressive accuracy gains.
The first frame already contains a field and the fidelity-lift message.

No motion deliverable was deferred. The renderer checks memory again before
animation; below 2,500 MiB (or `--static-only`) it deletes its old GIF and points
the first README image at the permanent PNG. The concept strip always remains.

## Visual and pre-ship checks

- [x] Viewed exported poster, concept, mechanism, contour detail, and full-domain figures.
- [x] Viewed GIF frames for the cover, transition, and all three highlighted stages.
- [x] Reviewed 960 px and 390 px storyboards. At 960 px, stage labels, arrows,
  physical axes, colorbars, and evidence labels are readable. At 390 px, the
  primary local→NNLCI→field direction survives; the small scientific details
  require the linked full-size image. No claim that every footer is phone-readable.
- [x] Enlarged the actual contour panels after the first visual inspection;
  the final plots occupy most of the figure instead of small centered squares.
- [x] Glance test (author): first frame says coarse CFD → finer fields;
  the adjacent still directly shows two local neighborhoods → NNLCI → HF-like goal.
  This does not substitute for a new external reviewer test.
- [x] HF supervision reaches only training loss. LF1/LF2 share physical location
  and time. The schematic uses structured stencils and does not imply arbitrary-mesh support.
- [x] Blue/green LF roles, teal prediction/map, red training supervision, and
  unchanged Sod amber/teal/navy. Labels and arrows carry meaning beyond color.
- [x] Every new result-like visual identifies its source/evidence class inside
  the image; descriptive alt text and linked provenance accompany the README.
- [x] Source/output hashes, PNG dimensions/DPI, complete GIF decoding/duration,
  Markdown local targets, GitHub API Markdown rendering, and diff whitespace checked before commit.
- [x] No fake metrics, training, CFD regeneration, email, visibility mutation,
  or merge action. The repository was verified PRIVATE.

## Repository state discovered

The branch began clean at `3741986` and matched its remote after fetch.
GitHub reported PR #2 **already merged** at `2026-09-28T21:31:46Z`, with head
`3ebb24f5c983cbc23ffb03f5df13ddb245aefce2`, before this work. The requested
branch is still the push destination. A push to that branch cannot reopen or
update the merged PR's code review; no replacement PR or merge is performed.

## Reproduce

From the repository root with NumPy, Matplotlib, Pillow, and retrieved LFS data:

```bash
python scripts/render_gatech_promo.py
```

The renderer reads packaged values, uses Agg/Pillow on CPU, closes Matplotlib
figures, and writes the six new media assets and `gatech_provenance.json`.
See the [gallery guide](README.md#2d-fields-and-the-motion-open) for LFS retrieval,
source interpretation, and static-only operation. Existing Sod rendering
commands and original inference provenance remain in the same guide.
