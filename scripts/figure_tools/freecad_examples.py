"""Two headless, editable CAD examples; no CFD meshing or simulation claim."""
import os
from pathlib import Path
import sys

# Debian path; FreeCAD's own Python console already has this module available.
sys.path.append('/usr/lib/freecad/lib')
import FreeCAD as App
import Part
import TechDraw
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

out = Path(os.environ['NNLCI_FIGURE_OUTPUT'])
for name in ('domain', 'sampling'):
    doc = App.newDocument(name)
    if name == 'domain':
        box = doc.addObject('Part::Box', 'Domain')
        box.Length, box.Width, box.Height = 80, 40, 8
        cylinder = doc.addObject('Part::Cylinder', 'Obstacle')
        cylinder.Radius, cylinder.Height = 6, 8
        cylinder.Placement.Base = App.Vector(25, 20, 0)
        result = doc.addObject('Part::Cut', 'FluidDomain')
        result.Base, result.Tool = box, cylinder
    else:
        base = doc.addObject('Part::Box', 'RakeBase')
        base.Length, base.Width, base.Height = 65, 3, 2
        probes = []
        for x in range(5, 65, 10):
            probe = doc.addObject('Part::Cylinder', f'Probe{x}')
            probe.Radius, probe.Height = 1, 18
            probe.Placement.Base = App.Vector(x, 1.5, 1)
            probes.append(probe)
        result = doc.addObject('Part::MultiFuse', 'SamplingRake')
        result.Shapes = [base] + probes
    doc.recompute()
    assert result.Shape.isValid() and result.Shape.Volume > 0
    doc.saveAs(str(out / f'freecad_{name}.FCStd'))
    Part.export([result], str(out / f'freecad_{name}.step'))
    projection = TechDraw.projectToSVG(result.Shape, App.Vector(0, -1, 1))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="-12 -50 105 65">'
           '<rect x="-12" y="-50" width="105" height="65" fill="white"/>'
           '<g transform="rotate(-90)">' + projection + '</g></svg>')
    (out / f'freecad_{name}.svg').write_text(svg)
    App.closeDocument(doc.Name)
print('FreeCAD:', '.'.join(App.Version()[:3]), 'two valid CAD documents, STEP and SVG exported')
