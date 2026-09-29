"""Independently solve Config 3 LF grids using the existing CFD kernels on CPU.

No HF file is an input to either solve. The optional HF comparison is done only
after both subprocesses exit, to catch accidental use of bootstrap subsamples.
"""
from pathlib import Path
import argparse
import datetime
import json
import re
import subprocess
import time
import numpy as np
from build_cpu_cfd import ROOT, SOURCE, build
from cfd_io import sha, read_field
from check_cpu_cfd import check, CONFIG
from replot_sod_promo import memory_gate

OUT = ROOT / "problems/2d_riemann/data/promo_cfd"
STANDARD_CONFIG_SHA256 = "d87695a3d236874d566febbb5094b342cc28113d37260ffc59740576b3c3ac2b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    available = memory_gate()
    if available < 700:
        raise SystemExit("CPU CFD deferred: less than 700 MiB effective available RAM")
    if sha(CONFIG) != STANDARD_CONFIG_SHA256:
        raise SystemExit("Standard Config 3 changed: review case/time provenance before regenerating the promo")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build_dir = ROOT / ".smoke/cpu-cfd"
    command = build(build_dir)
    checks = check(build_dir / "debug", build_dir / "weno_cpu")
    metadata = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "operation": "Independent CPU CFD solves from Config 3 initial conditions; no HF input",
        "solver_source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)},
        "adapter": {"path": "scripts/build_cpu_cfd.py", "sha256": sha(Path(__file__).with_name("build_cpu_cfd.py")),
                    "generated_source_sha256": sha(build_dir / "weno_cpu.f90"),
                    "changes": "CPU loops replace CUDA thread dispatch; host arrays replace device allocation; ordered boundary sweeps; unused storage initialization; input checks. Numerical flux, WENO5-JS, RK3 and positivity floor expressions retained."},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "compile_command": command,
        "config_source": {"path": str(CONFIG.relative_to(ROOT)), "sha256": sha(CONFIG)},
        "physics": {"case": "Riemann Config 3", "gamma": 1.4, "domain": [0, 1, 0, 1],
                    "t_end": .25, "dt": .0001, "boundary": "transmissive", "lower_fidelity_setup": "100 and 200 intervals per axis versus packaged 800; same equations and numerical kernels"},
        "effective_available_memory_mib": available, "checks": checks, "runs": [],
        "limitations": "One standard physical case for figure provenance, not a training ensemble. No new NNLCI training/inference or HF solve; GPU/CPU parity untested. Positivity floor inherited from original solver; no conservation or generalization claim.",
    }
    for n in (100, 200):
        config = out / f"config3_{n}.nml"
        config_text, count = re.subn(r"NX=800, NY=800", f"NX={n}, NY={n}", CONFIG.read_text())
        if count != 1:
            raise ValueError("Expected exactly one resolution setting in the standard configuration")
        config.write_text(config_text)
        target = out / f"flow_{n}x{n}.dat"
        log = out / f"solve_{n}.log"
        print(f"Solving {n} x {n} intervals at t=0.25 from initial conditions", flush=True)
        started = time.monotonic()
        with log.open("w") as stream:
            subprocess.run([str(build_dir / "weno_cpu"), str(config), str(target)], check=True,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
        elapsed = time.monotonic() - started
        field = read_field(target)
        metadata["runs"].append({"resolution_intervals": n, "shape": list(field.shape), "field": target.name,
            "field_sha256": sha(target), "config": config.name, "config_sha256": sha(config),
            "log": log.name, "elapsed_seconds_this_host": round(elapsed, 3),
            "density_range": [float(field[:, :, 2].min()), float(field[:, :, 2].max())],
            "pressure_range": [float(field[:, :, 5].min()), float(field[:, :, 5].max())],
            "finite_positive_and_coordinates": "passed"})
        print(f"Validated {target.name}: {elapsed:.1f}s", flush=True)
    hf_path = ROOT / "problems/2d_riemann/data/raw/flow_standard_config3.dat"
    hf = read_field(hf_path)
    metadata["packaged_hf_reference"] = {"path": str(hf_path.relative_to(ROOT)), "sha256": sha(hf_path),
        "identity": "Existing 801 x 801 CFD field; not regenerated, not a prediction. Nominal case/time matched to repository standard config; original execution log unavailable."}
    for run in metadata["runs"]:
        lf = read_field(out / run["field"])
        sampled = hf[::800 // run["resolution_intervals"], ::800 // run["resolution_intervals"], :]
        difference = np.abs(lf[:, :, 2:] - sampled[:, :, 2:])
        if not np.any(difference > 1e-5):
            raise ValueError("LF unexpectedly equals HF subsamples")
        run["not_hf_subsamples"] = {"passed": True, "max_absolute_primitive_difference": float(difference.max()),
            "meaning": "Provenance guard only; not an NNLCI error metric"}
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
