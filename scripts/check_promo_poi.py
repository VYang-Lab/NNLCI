"""Check dense POI coverage, source identity, boundary panels and delivered frames.

Run after render_gatech_promo.py (also supports its --static-only fallback).
These checks validate the reference illustration, not NNLCI accuracy.
"""
from __future__ import annotations
import json
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nnlci-matplotlib")
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PIL import Image

from cfd_io import read_field, sha
from promo_poi import (
    ASSEMBLY_COUNTS, ASSEMBLY_END, FRAME_COUNT, HEIGHT, INTRO_FRAMES,
    OUT, POINT_COUNT, ROOT, WIDTH, Scene, State, complete_stencil, point_at,
    read_off, state_at, stencil,
)
from replot_sod_promo import memory_gate


def main():
    if memory_gate() < 700:
        raise SystemExit("Validation deferred below 700 MiB effective available RAM")
    report = json.loads((OUT / "gatech_provenance.json").read_text())
    for source in report["sources"]:
        assert sha(ROOT / source["path"]) == source["sha256"], source["path"]
    for name, digest in report["renderers"].items():
        assert sha(ROOT / "scripts" / name) == digest, name
    for name, item in report["outputs"].items():
        path = OUT / name
        assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], name
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
        elif path.suffix == ".svg":
            ET.parse(path)
        elif path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF-")
        elif path.suffix == ".off":
            read_off(path)
    for name, digest in report["production_tools"]["sources"].items():
        assert sha(ROOT / name) == digest, name
    manifest = report["cfd_provenance"]
    assert sha(ROOT / manifest["path"]) == manifest["sha256"]
    assert report["poi_workflow"]["reference_assembly"]["point_count"] == 641601

    fields = [read_field(ROOT / source["path"]) for source in report["sources"]]
    norm = Normalize(*report["opening_and_comparison"]["density_scale"])
    scene = Scene(fields, norm)
    expected = colormaps["inferno"](norm(fields[-1][::-1, :, 2]), bytes=True)[:, :, :3]
    assert expected.shape == (801, 801, 3)
    expected = expected.reshape(-1, 3)
    # Check every displayed native image, including partial rows and boundary
    # nodes, against the source values, not just the final integer counter.
    assert ASSEMBLY_COUNTS[0] == 0 and ASSEMBLY_COUNTS[-1] == 641601
    assert all(b > a for a, b in zip(ASSEMBLY_COUNTS, ASSEMBLY_COUNTS[1:]))
    assert sum(np.diff(ASSEMBLY_COUNTS)) == POINT_COUNT
    for offset, filled in enumerate(ASSEMBLY_COUNTS):
        actual = np.asarray(scene.assembly_image(filled)).reshape(-1, 3)
        np.testing.assert_array_equal(actual[:filled], expected[:filled])
        assert np.all(actual[filled:] == [241, 246, 247]), filled
        state = state_at(INTRO_FRAMES+offset)
        assert state.filled == filled
        if filled:
            # Round-trip coordinates through the actual packaged x/y axes.
            x, y = state.point
            row, col = round((1-y)*800), round(x*800)
            assert row*801+col == filled-1
            np.testing.assert_allclose(fields[-1][800-row, col, :2], [x, y], atol=6e-7)
    assert point_at(0) == (0, 1)
    assert point_at(800) == (1, 1)
    assert point_at(640800) == (0, 0)
    assert point_at(641600) == (1, 0)
    assert state_at(ASSEMBLY_END).filled == POINT_COUNT
    assert state_at(FRAME_COUNT-1).stage == 4

    blank_points = ((0,0),(1,0),(0,1),(1,1),(.5,0),(0,.5),(1,.5),(.5,1),(.005,.5))
    for point in blank_points:
        assert not complete_stencil(point), point
        frame = np.asarray(scene.assembly_frame(State(point, 0, 0, 3, 1)))
        for left in (109, 414):
            assert np.all(frame[503:640, left:left+137] == 255), point
    for point in ((.5,.5),(.34,.44),(.01,.5),(.99,.5)):
        assert complete_stencil(point), point
        assert len(np.unique(stencil(point), axis=0)) == 9
        frame = np.asarray(scene.assembly_frame(State(point, 0, 0, 3, 1)))
        for left in (109, 414):
            assert np.any(frame[503:640, left:left+137] != 255), point
    svg = ET.parse(OUT / "nnlci_local_mechanism.svg")
    ns = {"s": "http://www.w3.org/2000/svg"}
    for name in ("lf1-stencil", "lf2-stencil"):
        group = svg.find(f".//s:g[@id='{name}']", ns)
        assert group is not None and len(group.findall("s:circle", ns)) == 9

    final = Image.open(OUT / "nnlci_hf_final.png").convert("RGB")
    np.testing.assert_array_equal(final, scene.hero())
    np.testing.assert_array_equal(final, Image.open(OUT / "nnlci_motion_poster.png").convert("RGB"))
    motion = report["animation"]
    if motion.get("deferred"):
        assert not (OUT / "nnlci_local_to_hf.gif").exists()
        assert "nnlci_motion_poster.png" in (ROOT / "README.md").read_text().splitlines()[0]
    else:
        with Image.open(OUT / motion["file"]) as gif:
            palette = gif.copy()
            total_ms = 0
            assert gif.size == (WIDTH, HEIGHT) and gif.n_frames == motion["frames"]
            for index in range(gif.n_frames):
                gif.seek(index)
                gif.load()
                total_ms += gif.info["duration"]
            assert total_ms == motion["duration_ms"]
            assert gif.info["duration"] == 4500
            # The last decoded frame must be the actual final scientific image,
            # with only the GIF's expected shared-palette quantization.
            quantized_final = final.quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")
            np.testing.assert_array_equal(gif.convert("RGB"), quantized_final)
        assert "nnlci_local_to_hf.gif" in (ROOT / "README.md").read_text().splitlines()[0]
    print(json.dumps({
        "checked": ["all source, renderer and output hashes", "PNG/SVG/PDF/OFF decoding",
                    "641601 native reference colors and complete coverage",
                    "91 monotone assembly states with fixed previous values",
                    "corner/edge blank panels and nine-point interior stencils",
                    "final lossless image and static poster identity",
                    "decoded GIF final frame, final hold, dimensions and duration" if not motion.get("deferred")
                    else "static-only fallback and README opening"],
        "animation": motion,
    }, indent=2))


if __name__ == "__main__":
    main()
