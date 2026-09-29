# Scriptable research figures

Use the smallest tool that expresses the figure. The promo opening now uses
**OpenSCAD meshes and a Graphviz diagram directly in the delivered animation**;
Python projects the geometry, paints native CFD values, and assembles the GIF.
The mechanism remains an editable SVG generated with Python. No Blender render
is needed for these figures.

The production sources are [poi_planes.scad](../../scripts/figure_tools/poi_planes.scad),
[poi_pipeline.dot](../../scripts/figure_tools/poi_pipeline.dot), and the
[CPU compositor](../../scripts/promo_poi.py). Both LF planes share the same CAD
slab and patch frame, with exactly nine feature markers on each LF input.
The MLP emits one POI and the compositor places one display cell per prediction
until the HF-like puzzle is complete. OpenSCAD OFF meshes and the Graphviz SVG are committed in
the [promo gallery](../assets/promo/); they are inputs to the compositor, not
separate tool demos. The [render provenance](../assets/promo/gatech_provenance.json)
records their commands, versions, hashes and memory gates.

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
python3 scripts/render_gatech_promo.py
```

See the [gallery](../assets/promo/README.md#reproduce) for CFD inputs and provenance.

Reference documentation: [OpenSCAD](https://openscad.org/documentation.html),
[Graphviz output formats](https://graphviz.org/docs/outputs/),
[FreeCAD source and releases](https://github.com/FreeCAD/FreeCAD),
[draw.io desktop](https://github.com/jgraph/drawio-desktop),
[CairoSVG](https://cairosvg.org/documentation/),
[Blender command line](https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html).
