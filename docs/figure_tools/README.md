# Scriptable research figures

Use the smallest tool that expresses the figure. The current promo uses
**Graphviz, OpenSCAD and SVG/Python on CPU**: a minimal three-column CFD comparison
and a separate, paced method animation. Graphviz positions neural nodes for
current-like signals; OpenSCAD exports the moving water-drop meshes. The CPU
compositor shades and projects the geometry with soft shadows. Matplotlib renders
native-field contours; CairoSVG and Pillow compose the GIFs. [Renderer](../../scripts/render_promo_followup.py) ·
[Editable chapters, provenance and commands](../assets/promo/README.md#two-animations).

The current [neural graph](../../scripts/figure_tools/neural_signal.dot) and
[water drop](../../scripts/figure_tools/water_drop.scad) are editable sources.
Their exported layouts/geometry are consumed directly by
[the compositor](../../scripts/promo_signal.py). No installation occurs during rendering.

The preceding PR #7 presentation used [OpenSCAD meshes](../../scripts/figure_tools/poi_planes.scad)
and a [Graphviz diagram](../../scripts/figure_tools/poi_pipeline.dot) directly
in its [CPU compositor](../../scripts/promo_poi.py). Those sources and exports
remain available as historical artwork. Neither presentation needs a GPU.

| Figure | First tool | Editable source | Tested exports |
| --- | --- | --- | --- |
| Architecture, pipeline, network | Graphviz or draw.io | `.dot`, `.drawio` | SVG, PDF, PNG |
| Simple 3D concepts | OpenSCAD | `.scad` | STL, PNG; SVG via 2D projection, then PDF via CairoSVG |
| Engineering geometry | FreeCAD Python | Python + parametric `.FCStd` | STEP, projected SVG, then PDF/PNG via CairoSVG |
| Scientific plots and finishing | SVG / Python | SVG, Python + numerical arrays | SVG, PDF, high-resolution PNG |
| Lighting and presentation rendering | Blender, when needed | Python + `.blend` | CPU-rendered PNG |
| Interactive scenes | Godot, when needed | Not installed in this pass | Deferred: no interactive deliverable |

For diagrams, prefer SVG in documents that support it and PDF for typesetting;
use high-resolution PNG for GitHub and slide compatibility. A projected CAD SVG
is a 2D drawing, not an editable 3D model. Preserve SCAD/FCStd/STEP for geometry.
Blender's test renders are 640 × 400 smoke checks, not publication exports.

## Earlier installation smoke checks

The [machine-readable report](smoke_report.json) records exact package versions,
commands, memory readings, successful exit codes and hashes of the generated
files. [Preview sheet](tool_previews.png) shows both examples per desktop tool.
The examples are conceptual geometry/diagrams, not CFD simulation results.

- OpenSCAD: layered blocks with an arrow; a 3 × 3 sampling tile.
- Graphviz: inference pipeline; training-supervision graph.
- draw.io: editable inference and training diagrams; XML plus real desktop exports.
- FreeCAD: a parametric domain with a cylindrical obstacle removed; a sampling rake.
  Both shapes pass FreeCAD validity/positive-volume checks before export.
- SVG/Python: the actual NNLCI mechanism and numerical resolution comparison.
- Blender: layers and sample markers, Cycles on CPU with two threads and eight samples.

The [example sources](../../scripts/figure_tools/) are checked in. Smoke-check build products stay
under `.smoke/figure-tools/` and are intentionally ignored; rerun to obtain the
native CAD/Blender documents, vector exports and PNGs. Installation is separate
from rendering, so normal figure regeneration never requires administrator access.

## Setup on Debian / Ubuntu

Core tools and the CPU CFD compiler:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
  gfortran g++ openscad graphviz freecad blender xvfb xauth \
  python3-numpy python3-matplotlib python3-pil python3-cairosvg
```

The session installed draw.io 31.5.3 from its official Linux amd64 release:

```bash
curl -fL https://github.com/jgraph/drawio-desktop/releases/download/v31.5.3/drawio-amd64-31.5.3.deb \
  -o /tmp/nnlci-drawio.deb
# Verify against the publisher's release digest before installing:
printf '%s  %s\n' b2fd41f02567929c5fae4becbdc16bf118b1ef508a827a29ad9a883924263999 /tmp/nnlci-drawio.deb | sha256sum -c -
sudo apt-get install /tmp/nnlci-drawio.deb
```

This was checked against the [official release](https://github.com/jgraph/drawio-desktop/releases/tag/v31.5.3).
Other platforms should use the publisher's matching build. The headless smoke
runner uses Xvfb, software OpenGL, a temporary profile and disabled update checks.
Its Electron `--no-sandbox` flag is for this managed container's local, generated
examples; ordinary desktop editing can keep the application sandbox enabled.

## Run the examples

```bash
MPLCONFIGDIR=/tmp/nnlci-matplotlib python3 scripts/figure_tools/smoke.py
```

This runs sequentially and rechecks effective available memory before every
external tool (minimum 1,500 MiB). Exports are validated by decoding PNGs,
parsing SVG XML and checking PDF headers. The report also records geometry and
native project files. On Debian, FreeCAD imports from `/usr/lib/freecad/lib`;
use its bundled Python interpreter or adjust that path on other distributions.
Use system Python for the installed FreeCAD bindings. The project's inference
virtual environment need not contain any CAD or Blender dependencies.

For the actual research visuals:

```bash
python3 scripts/render_promo_followup.py
```

See the [gallery](../assets/promo/README.md#reproduce) for CFD inputs and provenance.

Reference documentation: [OpenSCAD](https://openscad.org/documentation.html),
[Graphviz output formats](https://graphviz.org/docs/outputs/),
[FreeCAD source and releases](https://github.com/FreeCAD/FreeCAD),
[draw.io desktop](https://github.com/jgraph/drawio-desktop),
[CairoSVG](https://cairosvg.org/documentation/),
[Blender command line](https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html).
