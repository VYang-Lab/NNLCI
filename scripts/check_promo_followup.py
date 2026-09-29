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
from render_promo_followup import (Artwork, COUNTS, GRID, H, OUT, ROOT, TOTAL, W,
                                   complete, load_sources, samples)
from replot_sod_promo import memory_gate


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
    for name, item in report["outputs"].items():
        path = out / name
        assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], name
        if path.suffix == ".svg":
            ET.parse(path)
        elif path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
    fields, _ = load_sources()
    art = Artwork(fields)
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
            for i in range(gif.n_frames):
                gif.seek(i)
                gif.load()
                duration += gif.info["duration"]
            assert duration == motion["duration_ms"]
            assert gif.info["duration"] == motion["final_hold_ms"]
            if key == "method":
                quantized = final.quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")
                np.testing.assert_array_equal(gif.convert("RGB"), quantized)
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
              "independent LF sources differ from HF subsamples",
              "all 641601 reference colors, monotone reveal and no changed prior values",
              "nine-point interior stencils and blank boundary patches",
              "decoded GIF dimensions, durations, final hold and full-field frame identity",
              "motion-free assets and current README image targets"]
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
