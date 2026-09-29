# Airphonse animation follow-up

PR #8 was already merged as `f9afe35` when this work began. This revision keeps
its opening comparison and training walkthrough and updates the separate
[How It Works animation](nnlci_method.gif).

- **神經網路電流：** Graphviz lays out a visible network. Signals cross its
  layers, gather four output variables, and deliver one point to the target
  grid. The forward pass plays twice. Training then routes HF labels and the
  predicted point to the loss, with a separate weight-update pulse.
- **水滴與 3D 位移：** OpenSCAD supplies actual teardrop meshes. They descend,
  land on matching LF sampling locations, lift, move sideways, and land at
  another location. Camera elevation and yaw also change. The two LF planes
  have visible thickness, and both patches retain nine samples.
- **光影：** CPU diffuse/specular/rim shading, soft shadows beneath the planes
  and moving drops, luminous nodes, and bright signal trails. Surface density
  colors continue to come from the verified CFD fields.

The water drops are sampling markers. The neural layers and activations are
schematic, with reduced node counts. The field reveal remains explicitly
**HF reference**, and the blank output grid shows only a conceptual predicted
point. No liquid simulation, measured 2D prediction, or speedup is claimed.

The 960 × 540 method GIF is **42.28 seconds / 178 encoded frames**, about
**2.07 MiB**, with a 4.5-second final full-field hold. Its SVG chapters and PNG
storyboard are regenerated alongside it. The unchanged three-column opening
remains 11.96 seconds. [Exact hashes and timings](followup_provenance.json).

## Validation

```bash
python3 scripts/render_promo_followup.py
python3 scripts/check_promo_followup.py
python3 scripts/check_promo_poi.py
python3 scripts/render_promo_followup.py --static-only --output-dir .smoke/signal-static
python3 scripts/check_promo_followup.py --output-dir .smoke/signal-static
```

The current and archived asset checks passed: source/export/output hashes,
nine-point patches, blank boundary patches, all 641,601 reference locations,
GIF decoding/timing, and final-frame identity. New checks establish a closed,
volumetric drop mesh; continuous vertical movement; forward-only graph edges;
visible changes in decoded motion frames; and a target point that appears
only after the signal passes through the network.

OpenSCAD's 128 zero-area faces on the rotation axis are removed during export;
the 290 vertices and 576 nonzero-area faces are retained. The resulting mesh
has two incident faces per edge and nonzero enclosed volume. The compositor
uses the exported mesh and Graphviz layout directly.

Memory-gate smoke checks inject readings of 2,499 MiB before animation, and
separately after the initial animation gate but before each encoder. Both
paths remove stale GIFs and retain PNG/SVG fallbacks. A 699 MiB reading stops
before rendering or creating the output directory. Isolated renders leave the
published README, GIF and provenance unchanged. The actual production render
started with 5,699 MiB effective available RAM and had 5,607 MiB before animation;
the threshold remains 2,500 MiB, respecting host and cgroup limits.

Visual review covered decoded frames at both landings and between neural layers,
the training branch, all six chapters, the final field, and a 390-pixel-wide
preview. Text and arrows do not overlap. Small explanatory text is best read
at the full image size; the water-drop and signal motion remain visible on a
phone-width view. Python compilation, local documentation links and
`git diff --check` passed.

The opening GIF, opening PNG/SVG, final HF PNG, collection/feature/assembly
chapters, CFD arrays, and new-user Sod walkthrough are byte-identical to the
merged PR #8 files. No CFD solve, training, inference, or NVIDIA tooling was
needed for this presentation change. The preceding
[PR #8 review](FOLLOWUP_REVIEW.md) records the original CFD and training checks.
