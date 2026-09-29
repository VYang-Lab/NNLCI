"""Small physical invariants and debug/release checks for CPU CFD dispatch."""
from pathlib import Path
import re
import subprocess
import numpy as np
from build_cpu_cfd import ROOT, build
from cfd_io import read_field

CONFIG = ROOT / "problems/2d_riemann/configs/input_Config3_std.nml"


def run_case(exe, directory, name, states, nx=12, ny=16, t=.001):
    config = directory / f"{name}.nml"
    lines = ["&CASE_PARAMS"]
    for q, values in enumerate(states, 1):
        lines += [", ".join(f"q{q}_{v}={a}" for v, a in zip(("r", "u", "v", "p"), values)) + ","]
    lines += [f"NX={nx}, NY={ny}, DT=0.0001, T_END={t}", "/"]
    config.write_text("\n".join(lines) + "\n")
    out = directory / f"{name}.dat"
    subprocess.run([str(exe), str(config), str(out)], check=True, capture_output=True, timeout=60)
    return read_field(out)


def check(directory, release):
    directory = Path(directory).resolve()
    build(directory, debug=True)
    debug = directory / "weno_cpu"
    uniform = [1., .2, -.1, 1.]
    for exe in (debug, release):
        field = run_case(exe, directory, "uniform", [uniform] * 4)
        np.testing.assert_allclose(field[:, :, 2:], np.broadcast_to(uniform, field[:, :, 2:].shape), atol=1e-7)
    left, right = [1., 0., 0., 1.], [.125, 0., 0., .1]
    x = run_case(debug, directory, "x_planar", [right, left, left, right])
    y = run_case(debug, directory, "y_planar", [right, right, left, left], nx=16, ny=12)
    np.testing.assert_allclose(x[:, :, 2:], y.transpose(1, 0, 2)[:, :, [2, 4, 3, 5]], atol=1e-6)
    np.testing.assert_allclose(x[:, :, 2:], np.broadcast_to(x[0, :, 2:], x[:, :, 2:].shape), atol=1e-7)
    xr = run_case(release, directory, "x_release", [right, left, left, right])
    np.testing.assert_allclose(x, xr, atol=1e-7)
    # A quadrant interaction exercises both fluxes, ghost cells and every RK stage.
    states = [[1.5, 0, 0, 1.5], [.5323, 1.206, 0, .3], [.138, 1.206, 1.206, .029], [.5323, 0, 1.206, .3]]
    d = run_case(debug, directory, "quadrants_debug", states, nx=20, ny=20, t=.01)
    r = run_case(release, directory, "quadrants_release", states, nx=20, ny=20, t=.01)
    np.testing.assert_allclose(d, r, atol=1e-6)
    return {"uniform_flow": "passed", "planar_transpose_and_transverse_invariance": "passed",
            "debug_release_planar_and_quadrants": "passed", "bounds_and_floating_point_traps": "passed",
            "gpu_cpu_parity": "not tested: no NVIDIA GPU/nvfortran"}


if __name__ == "__main__":
    directory = ROOT / ".smoke/cpu-cfd"
    build(directory)
    print(check(directory / "debug", directory / "weno_cpu"))
