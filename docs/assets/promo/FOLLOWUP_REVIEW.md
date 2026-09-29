# Follow-up visual and workflow review

This presentation follows merged PR #7 (`555f132`, merged into `f740485`).
It addresses the requests for a simpler opening, a slower explanation of the
method, and a practical introduction to training and inference.

## Delivered presentation

- **Opening:** three columns, LF1/LF2 nine-point patches and an explicitly
  labeled HF output target. No network, equation, counter or explanatory
  paragraph inside the opening. 11.96 seconds, 15 encoded frames, 960 × 540.
- **Method:** six paced chapters showing alignment, four variables, feature
  assembly, an MLP predicting one point, training supervision, and full-field
  assembly. 34.88 seconds, 50 encoded frames, 960 × 540. A 4.5-second final
  hold retains the full HF reference and enlarged interaction detail.
- **Scientific meaning:** both inputs retain nine locations; the MLP's four
  outputs belong to one point. The complete reference reveal includes all
  641,601 locations. Boundary patches stay blank. HF enters only the training
  loss, and reference artwork is labeled separately from predictions.
- **New-user path:** data inspection, one-epoch training to a separate Sod
  checkpoint, prediction using that checkpoint, supplied-model comparison,
  longer training, optional CFD generation and troubleshooting.

Visual inspection covered the comparison, all six chapters, the final frame,
and a phone-width comparison preview. Density views share physical windows
and scales; role colors and typography are consistent. GIFs total about
1.3 MiB. PNG alternatives and editable SVG chapters remain available.

## Executed checks

All commands ran on CPU. Plotting used Agg; training/inference used two CPU
threads. The existing `/workspace/.nnlci-venv` supplied PyTorch; system Python
supplied the figure tools. Reproduction commands use `python`/`python3` after
installing their documented dependencies.

```bash
python3 scripts/regenerate_promo_lf.py --output-dir .smoke/followup-cfd
python3 scripts/render_promo_followup.py
python3 scripts/check_promo_followup.py --rerun-cfd-dir .smoke/followup-cfd
python3 scripts/render_promo_followup.py --static-only --output-dir .smoke/promo-static
python3 scripts/check_promo_followup.py --output-dir .smoke/promo-static
python3 scripts/check_promo_poi.py
```

The independent 100/200 CPU solves reproduced both saved fields byte-for-byte.
The solver's uniform-flow, planar symmetry/invariance, debug/release agreement,
bounds and floating-point checks passed. The new asset checker verified source,
renderer and output hashes; PNG/SVG decoding; complete native reference colors;
unchanged previously revealed values; boundary blanks; GIF dimensions,
duration and final-frame identity; and the static-only outputs. Archived PR #7
assets retain their own checker and provenance.

The walkthrough's data check produced finite `(7182, 18)` inputs and
`(7182, 3)` targets. Its one-epoch training command completed using
`smoke_model_first_sod.pth`. Prediction with that checkpoint produced six
PNGs in `.smoke/first-sod/predictions/`; the supplied checkpoint produced six
PNGs in `.smoke/supplied-sod/predictions/`. Named packaged weights and result
figures were preserved. Training metrics from this smoke run are not promoted
as reconstruction evidence.

Python compilation, changed-document local links and `git diff --check` were
also checked. Scratch CFD runs, checkpoints, predictions and test renders stay
under ignored paths and are not publication artifacts.

## Limits

There is no new validated 2D prediction or training ensemble. The LF fields are
independent CFD, while the output target and assembly colors remain the
packaged HF reference. Its historical execution log is unavailable. GPU/CPU
solver parity was not tested; no NVIDIA GPU was used. The packaged 2D bootstrap
and its two-epoch checkpoint remain unchanged. No timing or accuracy claim
is inferred from animation pacing or image sharpness.

The [machine-readable render record](followup_provenance.json) contains exact
source hashes, software versions, sizes and memory readings.
