"""Validate scientific source identity, field assembly, GIF delivery and fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from cfd_io import sha
from render_promo_followup import (Artwork, COUNTS, GRID, H, HOLDS, OUT, ROOT, TOTAL, W,
                                   complete, load_sources, samples)
from replot_sod_promo import memory_gate
from promo_signal import SignalArtwork, drop_state, project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--rerun-cfd-dir", type=Path)
    args = parser.parse_args()
    if memory_gate() < 700:
        raise SystemExit("Validation deferred below 700 MiB available RAM")
    out = args.output_dir.resolve()
    report = json.loads((out / "followup_provenance.json").read_text())
    for source in report["sources"] + [report["cfd_provenance"]]:
        assert sha(ROOT / source["path"]) == source["sha256"], source["path"]
    for name, digest in report["renderers"].items():
        assert sha(ROOT / "scripts" / name) == digest, name
    drawing = report["drawing_tools"]
    for name, digest in drawing["sources"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in drawing["exports"].items():
        assert sha(out / name) == digest, name
    for name, item in report["outputs"].items():
        path = out / name
        assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], name
        if path.suffix == ".svg":
            ET.parse(path)
        elif path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
    fields, _ = load_sources()
    signal = SignalArtwork(out / "signal_geometry")
    art = Artwork(fields, signal)
    # A closed, volumetric teardrop; each mesh edge has two incident faces.
    from collections import Counter
    edge_counts = Counter()
    volume = 0.
    for face in signal.faces:
        assert min(face) >= 0 and max(face) < len(signal.vertices)
        for a, b in zip(face, face[1:]+face[:1]):
            edge_counts[tuple(sorted((a, b)))] += 1
        vertices = signal.vertices[face]
        for i in range(1, len(vertices)-1):
            volume += np.dot(vertices[0], np.cross(vertices[i], vertices[i+1]))/6
    assert set(edge_counts.values()) == {2} and abs(volume) > 100
    assert np.all(np.ptp(signal.vertices, axis=0) > 5)
    # Movement is continuous through lift-off, and has visible depth, not just
    # an SVG marker translated over a flat image.
    states = [drop_state(float(t)) for t in np.linspace(0, 1, 101)]
    assert max(s[1] for s in states) - min(s[1] for s in states) >= 28
    assert max(abs(a[1]-b[1]) for a, b in zip(states, states[1:])) < 2
    projected, _ = project([[0, 0, 0], [0, 0, 28]], (251, 277), 40, -12)
    assert np.linalg.norm(projected[0]-projected[1]) > 45
    assert len(signal.nodes) == 18 and len(signal.edges) == 65
    assert all(signal.layers[b] == signal.layers[a]+1 for a, b in signal.edges)
    expected = np.asarray(art.reference).reshape(-1, 3)
    assert COUNTS[0] == 0 and COUNTS[-1] == TOTAL == 641601
    assert all(a < b for a, b in zip(COUNTS, COUNTS[1:]))
    for count in COUNTS:
        actual = np.asarray(art.reveal(count)).reshape(-1, 3)
        np.testing.assert_array_equal(actual[:count], expected[:count])
        assert np.all(actual[count:] == [241, 246, 247])
    for point in ((0, 0), (1, 1), (0, .5), (.5, 1), (.005, .5)):
        assert not complete(point)
        row, col = round((1-point[1])*(GRID-1)), round(point[0]*(GRID-1))
        count = row*GRID+col+1
        frame = np.asarray(art.method(5, count).raster())
        for x in (65, 245):
            assert np.all(frame[149:269, x:x+120] == 255), point
    for point in ((.34, .44), (.46, .48), (.40, .36)):
        assert complete(point) and len(np.unique(samples(point), axis=0)) == 9
    final = Image.open(out / "nnlci_method_final.png").convert("RGB")
    np.testing.assert_array_equal(final, art.finished().raster())
    for key, motion in report["animations"].items():
        if motion.get("deferred"):
            assert not (out / f"nnlci_{key}.gif").exists()
            continue
        with Image.open(out / motion["file"]) as gif:
            palette = gif.copy()
            assert gif.size == (W, H) and gif.n_frames == motion["frames"]
            duration = 0
            observed = {}
            # Check actual delivered GIF pixels across the causal signal path.
            predict_start = sum(HOLDS[:3])
            probe_times = (500, 2800, predict_start+100, predict_start+1800,
                           predict_start+3900)
            for i in range(gif.n_frames):
                gif.seek(i)
                gif.load()
                end = duration+gif.info["duration"]
                if key == "method":
                    for instant in probe_times:
                        if duration <= instant < end:
                            observed[instant] = np.array(gif.convert("RGB"))
                duration = end
            assert duration == motion["duration_ms"]
            assert gif.info["duration"] == motion["final_hold_ms"]
            if key == "method":
                quantized = final.quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")
                np.testing.assert_array_equal(gif.convert("RGB"), quantized)
                assert len(observed) == len(probe_times)
                early, middle, late = [observed[t] for t in probe_times[2:]]
                assert np.count_nonzero(early[144:338,339:691] != middle[144:338,339:691]) > 1000
                assert np.count_nonzero(observed[500][150:340,90:380] != observed[2800][150:340,90:380]) > 2000
                np.testing.assert_array_equal(late[246,843], [8,127,140])
                assert not np.array_equal(early[246,843], late[246,843])
                assert not np.array_equal(middle[246,843], late[246,843])
    if not any(m.get("deferred") for m in report["animations"].values()):
        assert report["animations"]["method"]["duration_ms"] > 2*report["animations"]["comparison"]["duration_ms"]
    if out == OUT:
        readme = (ROOT / "README.md").read_text()
        for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', readme):
            if not target.startswith("http"):
                assert (ROOT / target).exists(), target
        for key, fallback in (("comparison", "nnlci_comparison_poster.png"), ("method", "nnlci_method_storyboard.png")):
            filename = fallback if report["animations"][key].get("deferred") else f"nnlci_{key}.gif"
            assert f"](docs/assets/promo/{filename})" in readme
    checks = ["source/renderer/output hashes and SVG/PNG decoding",
              "Graphviz and OpenSCAD source/export hashes; real layered graph and closed volumetric mesh",
              "continuous water-drop lift/landing and visible projected z displacement",
              "independent LF sources differ from HF subsamples",
              "all 641601 reference colors, monotone reveal and no changed prior values",
              "nine-point interior stencils and blank boundary patches",
              "motion-free assets and current README image targets"]
    if not report["animations"]["method"].get("deferred"):
        checks.extend(["decoded water-drop motion and layer signals; target point appears only after propagation",
                       "decoded method GIF dimensions, durations, final hold and full-field frame identity"])
    if not report["animations"]["comparison"].get("deferred"):
        checks.append("decoded opening GIF dimensions, durations and final hold")
    if args.rerun_cfd_dir:
        rerun = json.loads((args.rerun_cfd_dir / "provenance.json").read_text())
        for run, source in zip(rerun["runs"], report["sources"][:2], strict=True):
            assert sha(args.rerun_cfd_dir / run["field"]) == source["sha256"]
        for key, result in rerun["checks"].items():
            if key != "gpu_cpu_parity":
                assert result == "passed", (key, result)
        checks.append("fresh 100/200 CPU CFD runs reproduce saved sources byte-for-byte; solver checks pass")
    print(json.dumps({"checked": checks, "animations": report["animations"]}, indent=2))


if __name__ == "__main__":
    main()
