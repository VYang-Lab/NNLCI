"""Graphviz signal paths and lit OpenSCAD water-drop meshes, projected on CPU.

No physical-time simulation or model activations: these are method illustrations.
CFD textures are supplied by the existing verified-source renderer, unchanged.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess

import numpy as np

from cfd_io import sha
from replot_sod_promo import memory_gate

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts/figure_tools"
TEAL = "#087f8c"
DEFS = '''<defs>
  <linearGradient id="signal-panel" x2="1" y2="1">
    <stop stop-color="#172d47"/><stop offset="1" stop-color="#0b192d"/>
  </linearGradient>
  <radialGradient id="signal-glow">
    <stop stop-color="#c4ffef" stop-opacity=".85"/>
    <stop offset=".4" stop-color="#59efdb" stop-opacity=".28"/>
    <stop offset="1" stop-color="#59efdb" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="soft-shadow">
    <stop stop-color="#1b3655" stop-opacity=".28"/>
    <stop offset="1" stop-color="#1b3655" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="signal-node" cx=".28" cy=".24" r=".8">
    <stop stop-color="#f0fff8"/><stop offset=".4" stop-color="#80efdb"/>
    <stop offset="1" stop-color="#16879e"/>
  </radialGradient>
</defs>'''


def prepare_tools(out):
    """Run installed drawing tools; geometry and layouts are used by the movie."""
    folder = out / "signal_geometry"
    folder.mkdir(parents=True, exist_ok=True)
    exports = [folder / "water_drop.off", folder / "neural_signal.plain",
               folder / "neural_signal.svg"]
    commands = [
        ["openscad", "--hardwarnings", "-o", str(exports[0]), str(SRC / "water_drop.scad")],
        ["dot", "-Tplain", str(SRC / "neural_signal.dot"), "-o", str(exports[1])],
        ["dot", "-Tsvg", str(SRC / "neural_signal.dot"), "-o", str(exports[2])],
    ]
    runs = []
    for command in commands:
        available = memory_gate()
        if available < 700:
            raise RuntimeError("Drawing-tool export deferred below 700 MiB available RAM")
        result = subprocess.run(command, capture_output=True, text=True, timeout=90,
                                env=dict(os.environ, QT_QPA_PLATFORM="offscreen",
                                         LIBGL_ALWAYS_SOFTWARE="1"))
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        runs.append({"command": command, "exit_code": 0, "memory_before_mib": available})
    # OpenSCAD emits zero-area triangles along the rotation axis. Remove only
    # those collapsed faces; retain every vertex and every nonzero-area face.
    rows = exports[0].read_text().splitlines()
    header = rows.pop(0).split()
    nv, nf, _ = map(int, header[1:] or rows.pop(0).split())
    vertices = np.array([list(map(float, line.split())) for line in rows[:nv]])
    faces = []
    for row in rows[nv:nv+nf]:
        face = list(map(int, row.split()))[1:]
        p = vertices[face]
        if np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0])) > 1e-9:
            faces.append(row.rstrip())
    exports[0].write_text(f"OFF\n{nv} {len(faces)} 0\n"+
                         "\n".join([line.rstrip() for line in rows[:nv]]+faces)+"\n")
    report = {"execution": "Graphviz layout + OpenSCAD mesh consumed by CPU SVG compositor; no GPU",
              "mesh_cleanup": {"removed_zero_area_axis_faces": nf-len(faces),
                               "retained_faces": len(faces), "vertices_unchanged": nv},
              "commands": runs,
              "versions": {tool: subprocess.run([tool, flag], capture_output=True, text=True,
                           check=True).stderr.strip() for tool, flag in
                           (("dot", "-V"), ("openscad", "--version"))},
              "sources": {str(p.relative_to(ROOT)): sha(p) for p in
                          (SRC / "neural_signal.dot", SRC / "water_drop.scad")},
              "exports": {str(p.relative_to(out)): sha(p) for p in exports}}
    return SignalArtwork(folder), report


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3-2*t)


def drop_state(t):
    """Two landings at different physical points, with actual z displacement."""
    blend = float(smooth((t-.48)/.26))
    point = (.34+.12*blend, .44+.04*blend)
    if t < .40:
        lift = 30*(1-float(smooth(t/.28)))
    elif t < .64:
        lift = 28*float(smooth((t-.40)/.24))
    else:
        lift = 28*(1-float(smooth((t-.64)/.28)))
    return point, lift, 37+10*float(smooth(t)), -18+11*float(smooth(t))


def project(vertices, center, elevation, yaw, scale=2.55):
    x, y, z = np.asarray(vertices).T
    a, e = math.radians(yaw), math.radians(elevation)
    xx, yy = x*np.cos(a)-y*np.sin(a), x*np.sin(a)+y*np.cos(a)
    xy = np.column_stack((center[0]+scale*xx,
                          center[1]-scale*(yy*np.sin(e)+z*np.cos(e))))
    depth = -yy*np.cos(e)+z*np.sin(e)
    return xy, depth


def polygon(svg, points, fill, stroke="none", width=.5):
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    svg.parts.append(f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def pulse(svg, start, end, t, color="#8fffe3"):
    if not 0 <= t <= 1:
        return
    a, b = np.array(start), np.array(end)
    p = a+(b-a)*t
    tail = a+(b-a)*max(0, t-.22)
    svg.parts.append(f'<path d="M {tail[0]:.2f} {tail[1]:.2f} L {p[0]:.2f} {p[1]:.2f}" '
                     f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
    svg.dot(*p, 10, "url(#signal-glow)", "none")
    svg.dot(*p, 2.5, "#effff6", "none")


def pulse_path(svg, points, t, color):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.clip(t, 0., 1.)*lengths.sum()
    for start, end, length in zip(points, points[1:], lengths):
        if distance <= length:
            pulse(svg, start, end, distance/length, color)
            break
        distance -= length


class SignalArtwork:
    def __init__(self, folder):
        lines = (folder / "water_drop.off").read_text().splitlines()
        header = lines.pop(0).split()
        assert header[0] == "OFF"
        nv, nf, _ = map(int, header[1:] or lines.pop(0).split())
        self.vertices = np.array([list(map(float, line.split())) for line in lines[:nv]])
        self.faces = [list(map(int, line.split()))[1:] for line in lines[nv:nv+nf]]
        assert self.vertices.shape == (nv, 3) and nf and np.isfinite(self.vertices).all()
        self.nodes, self.edges = {}, []
        for line in (folder / "neural_signal.plain").read_text().splitlines():
            bits = line.split()
            if bits[0] == "node":
                self.nodes[bits[1]] = np.array([float(bits[2]), float(bits[3])])
            elif bits[0] == "edge":
                self.edges.append((bits[1], bits[2]))
        low, high = np.min(list(self.nodes.values()), axis=0), np.max(list(self.nodes.values()), axis=0)
        for key, point in self.nodes.items():
            x, y = (point-low)/(high-low)
            self.nodes[key] = np.array([374+294*x, 307-124*y])
        self.layers = {name: "ihjo".index(name[0]) for name in self.nodes}

    def mesh(self, svg, offset, center, elevation, yaw, base):
        xyz = self.vertices*1.35+offset
        points, depths = project(xyz, center, elevation, yaw)
        e, a = math.radians(elevation), math.radians(yaw)
        view = np.array([-math.sin(a)*math.cos(e), -math.cos(a)*math.cos(e), math.sin(e)])
        light = np.array([-.55, -.38, .85]); light /= np.linalg.norm(light)
        half = light+view; half /= np.linalg.norm(half)
        for face in sorted(self.faces, key=lambda f: float(depths[f].mean())):
            p = xyz[face]
            normal = np.cross(p[1]-p[0], p[2]-p[0])
            length = np.linalg.norm(normal)
            if length < 1e-9:
                continue
            normal /= length
            if np.dot(normal, view) <= 0:
                continue
            diffuse = max(0., np.dot(normal, light))
            specular = max(0., np.dot(normal, half))**22
            rim = (1-max(0., np.dot(normal, view)))**3
            rgb = np.clip(np.array(base)*(.30+.72*diffuse)+205*specular+30*rim, 0, 255).astype(int)
            color = "#"+"".join(f"{v:02x}" for v in rgb)
            polygon(svg, points[face], color, color, .4)

    def sampling(self, svg, art, t):
        svg.parts.append(DEFS)
        point, lift, elevation, yaw = drop_state(t)
        for level, center in enumerate(((251, 277), (699, 277))):
            role = ("#2563eb", "#159947")[level]
            svg.text(center[0], 115, f"LF{level+1}", 24, role, True, "middle")
            # The upper surface uses unmodified CFD colors. Only geometry, the
            # water-drop marker, and shadows receive presentation lighting.
            corners = np.array([[-48, 48, 0], [48, 48, 0], [48, -48, 0], [-48, -48, 0]])
            top, _ = project(corners, center, elevation, yaw)
            bottom, _ = project(corners+[0, 0, -4], center, elevation, yaw)
            svg.parts.append(f'<ellipse cx="{center[0]+14}" cy="{center[1]+54}" rx="159" ry="61" fill="url(#soft-shadow)"/>')
            for k in (1, 2):
                n = (k+1) % 4
                polygon(svg, [top[k], top[n], bottom[n], bottom[k]], ("#778eaa", "#a5b7cc")[k-1])
            a = (top[1]-top[0])/320; b = (top[3]-top[0])/320
            svg.parts.append(f'<g transform="matrix({a[0]} {a[1]} {b[0]} {b[1]} {top[0,0]} {top[0,1]})">')
            svg.image(art.details[level], 0, 0, 320)
            svg.parts.append('</g>')
            polygon(svg, top, "none", "#ecf5ff", 1.3)
            px, py = [(v-.39)/.26*96 for v in point]
            patch = np.array([[px+dx, py+dy, .4] for dx, dy in
                              ((-4,-4), (4,-4), (4,4), (-4,4))])
            projected, _ = project(patch, center, elevation, yaw)
            polygon(svg, projected, "none", "white", 1.4)
            sample_xyz = np.array([[px+i*.01/.26*96, py+j*.01/.26*96, .6]
                                   for j in (1, 0, -1) for i in (-1, 0, 1)])
            sample_xy, _ = project(sample_xyz, center, elevation, yaw)
            for sx, sy in sample_xy:
                svg.dot(sx, sy, 2.3, "white", role)
            contact = project([[px, py, .6]], center, elevation, yaw)[0][0]
            sx, sy = contact
            # Projected soft shadow grows with height, contracts at contact.
            svg.parts.append(f'<ellipse cx="{sx+lift*.28}" cy="{sy+2}" rx="{12+lift*.18}" ry="{5+lift*.07}" fill="url(#soft-shadow)"/>')
            self.mesh(svg, (px, py, lift+1.2), center, elevation, yaw,
                      (46, 154, 244) if level == 0 else (36, 199, 159))
            if lift < 4:
                svg.parts.append(f'<ellipse cx="{sx}" cy="{sy}" rx="{16+(4-lift)*2}" ry="{6+(4-lift)}" fill="none" stroke="white" stroke-width="1.3"/>')
            x = center[0]+54
            svg.line(f"M {sx} {sy+12} L {x+39} 357", role)
            art.patch(svg, level, point, x, 357, 78)
            svg.text(center[0]-23, 389, "3 × 3", 23, role, True, "end")
            svg.text(center[0]-23, 416, "same offsets", 17, "#596879", anchor="end")
        svg.text(480, 478, "Same physical point, case and time", 23, anchor="middle")
        svg.text(480, 145, "Water drops mark the sampling point", 16, "#596879", anchor="middle")

    def network(self, svg, phase):
        svg.parts.append(DEFS)
        svg.rect(346, 152, 352, 194, "#e4eaf0", radius=16)
        svg.rect(339, 144, 352, 194, "url(#signal-panel)", radius=16)
        front = phase*4.7
        for start, end in self.edges:
            a, b = self.nodes[start], self.nodes[end]
            layer = self.layers[start]
            t = front-layer
            color = "#427e8c" if t >= 1 else "#29435c"
            svg.line(f"M {a[0]} {a[1]} L {b[0]} {b[1]}", color)
            pulse(svg, a, b, t)
        for name, (x, y) in self.nodes.items():
            lit = front >= self.layers[name]
            if lit:
                svg.dot(x, y, 13, "url(#signal-glow)", "none")
            svg.dot(x, y, 6, "url(#signal-node)" if lit else "#253f5a", "#86cabc" if lit else "#50718d")
        # Gather four variables into one location, then place that location.
        hub = (730, 246)
        for name, xy in self.nodes.items():
            if name.startswith("o"):
                svg.line(f"M {xy[0]} {xy[1]} L {hub[0]} {hub[1]}", "#a4c6c9")
                pulse(svg, xy, hub, front-3, TEAL)
        svg.line("M 738 246 H 851", "#a4c6c9", arrow=True)
        # A blank coordinate grid, NOT a field falsely presented as inference.
        for k in range(5):
            svg.line(f"M {799+k*22} 202 V 290", "#dbe7eb")
            svg.line(f"M 799 {202+k*22} H 887", "#dbe7eb")
        svg.parts.append('<g id="predicted-point" data-lit="'+str(front >= 4.7).lower()+'">')
        if front >= 4:
            dest = np.array(hub)+(np.array([843,246])-hub)*min(1, (front-4)/.7)
            svg.dot(*dest, 18, "url(#signal-glow)", "none")
            svg.rect(dest[0]-6, dest[1]-6, 12, 12, TEAL, "#a7f6de", 2)
        svg.parts.append('</g>')
        svg.text(516, 122, "NNLCI · MLP", 23, TEAL, True, "middle")
        svg.text(516, 366, "Schematic layers / nodes", 16, "#596879", anchor="middle")
        svg.text(843, 174, "One point", 22, TEAL, True, "middle")
        svg.text(843, 329, "ρ̂, û, v̂, p̂", 23, TEAL, anchor="middle")

    def prediction(self, svg, phase, training=False):
        svg.text(94, 192, "72 inputs", 21, "#2563eb", True, "middle")
        for row in range(6):
            for col in range(12):
                svg.rect(31+col*11, 214+row*12, 7, 8, "#2563eb" if row < 3 else "#159947", radius=1)
        svg.line("M 171 246 H 203", arrow=True)
        svg.rect(209, 200, 112, 91, "#edf3f7", "#c9dbe2", 10)
        svg.text(265, 225, "Fixed", 18, anchor="middle")
        svg.text(265, 250, "training", 18, anchor="middle")
        svg.text(265, 275, "transforms", 17, bold=True, anchor="middle")
        svg.line("M 324 246 H 356", arrow=True)
        self.network(svg, phase if not training else 1.)
        if not training:
            svg.text(480, 440, "Signals cross the layers. Four values reach one point.", 23, anchor="middle")
            svg.text(480, 477, "Repeat at target coordinates to build the HF-like field", 20, TEAL, anchor="middle")
            return
        svg.rect(31, 392, 891, 95, "#fff5f5", radius=10)
        svg.text(49, 417, "TRAINING ONLY", 15, "#dc2626", True)
        svg.text(51, 461, "HF label at this point", 20, "#dc2626")
        svg.line("M 282 452 H 364", "#dc2626", arrow=True)
        svg.rect(378, 430, 91, 40, "white", "#dc2626", 6)
        svg.text(423, 458, "Loss", 20, "#dc2626", True, "middle")
        svg.line("M 843 347 V 452 H 482", TEAL, arrow=True)
        svg.line("M 423 426 V 383 H 680 V 330", "#dc2626", arrow=True, dash=True)
        svg.text(655, 415, "Update weights", 18, "#dc2626")
        # Supervision converges at loss, then updates weights; never LF inputs.
        t = phase*3
        if t <= 1:
            pulse(svg, (282,452), (364,452), t, "#e37368")
            pulse_path(svg, [(843,347), (843,452), (482,452)], t, TEAL)
        elif t <= 2:
            svg.rect(378, 430, 91, 40, "#ffe2d8", "#dc2626", 6)
            svg.text(423, 458, "Loss", 20, "#dc2626", True, "middle")
        else:
            pulse_path(svg, [(423,426), (423,383), (680,383), (680,330)], t-2, "#e37368")
