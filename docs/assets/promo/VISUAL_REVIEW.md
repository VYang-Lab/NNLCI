# Promo visual review · 2026-09-28

Scope: approved visual specification, implemented on `promo/readme-redesign`.
This is an author self-review of presentation, not an independent scientific validation.

- [x] Opening message: coarse LF CFD → learned refinement → HF-like field.
- [x] Conceptual schematics and Sod checkpoint evidence are explicitly classified.
- [x] Refined prediction and HF reference are separate; no exactness guarantee.
- [x] LF1/LF2 share case, location, variables, and time; neither is a timestep.
- [x] HF supervision terminates at loss. New-case HF is excluded from inference features, point selection, and correction.
- [x] Schematics use structured grids; no arbitrary-mesh or viscous capability is implied.
- [x] Blue/green LF inputs, red training supervisor, teal prediction; Sod amber/teal/navy with dashed/solid/dotted curves. Grayscale preview preserves roles and line distinctions.
- [x] PNGs inspected at 900–960 px: no overlaps, clipping, or missing glyphs. Schematic key labels are at least 16 display px at 900 px; secondary text at least 14 px. Plot titles and legends are at least 16 px, secondary labels at least 14 px.
- [x] 390 px previews retain the principal flow and evidence classification; original PNG links provide detail.
- [x] Sod panels share x/density limits. Overlay errors retain both signs, the zero line, and the full spatial/error range.
- [x] No invented metrics, badges, speedup claims, or new science.
- [x] Short captions; local Markdown targets checked, with protocol/provenance linked.
- [x] One opening evidence-boundary callout. 2D two-epoch/training-array limit is explicit; WC supplied-array/missing-generator limit remains in workflows and guide.
- [x] Descriptive presentation paths under `docs/assets/`; source artwork and original problem figures retained.
- [x] Alt text describes LF1/prediction/reference or schematic roles.
- [x] `free -m` and the process cgroup plus ancestor limits checked before result rendering. Agg/CPU static replot; figures closed after saving.
- [x] P1.25 arrays read without writes, inference, or smoothing. Original checkpoint/data hashes preserved; array hashes and renderer versions recorded separately.
- [x] No animation generated in this execution; permanent PNG strip exists. No Markdown image depends on a GIF.
- [x] Ten-second author check: opening identifies the refinement objective, LF inputs, HF supervision, and the first real Sod result. One principle hero leads; mechanism and evidence follow below.

No checklist waivers. Full CFD, model training, and inference were not run.

## Reproduce the schematic assets

From the repository root with Pillow installed:

```bash
python scripts/generate_concept_overview.py
python scripts/generate_training_tokenization_diagram.py
python scripts/annotate_local_stencil_diagram.py
python scripts/add_lf_hf_arrow.py
```

The shared `scripts/promo_drawing.py` uses scalable DejaVu Sans fonts on Linux;
there is no silent bitmap-font fallback. Opaque PNGs export at twice their
960 px design width. The overview and field comparison reuse the unchanged
`lf-field-sample.png` and `hf-field-sample.png` artwork, without a numerical
colorbar. The local diagram redraws the alignment concept from the preserved
`nnlci-local-stencil-base.png`, replacing its legacy HF-selection branch with
LF-only target-grid alignment. The training-only HF branch is shown separately
in the tokenization schematic.

Result regeneration and saved-array replot commands are in the [gallery guide](README.md).
