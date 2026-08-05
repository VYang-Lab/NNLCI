"""Build a minimal Config 3 case from the standard CFD result.

This is an end-to-end bootstrap. Formal training should instead use
independently solved 100/200/800 grids for many perturbed cases.
"""

from pathlib import Path
import re

import numpy as np

PROBLEM_DIR = Path(__file__).resolve().parents[2]
SOURCE = PROBLEM_DIR / "data" / "raw" / "flow_standard_config3.dat"
CASE_DIR = PROBLEM_DIR / "data" / "bootstrap" / "DATA_TRAIN" / "Config3" / "case_000"


def read_tecplot(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    zone_index = next(i for i, line in enumerate(lines) if "ZONE" in line)
    numeric = re.sub(r"(\d)([+-]\d{3})", r"\1E\2", "\n".join(lines[zone_index + 1 :]))
    data = np.loadtxt(numeric.splitlines())
    side = int(round(np.sqrt(len(data))))
    if side * side != len(data):
        raise ValueError(f"Expected a square Tecplot grid, got {len(data)} rows")
    return data.reshape(side, side, data.shape[1])


def write_tecplot(path: Path, grid: np.ndarray) -> None:
    ny, nx, _ = grid.shape
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(' TITLE="RIEMANN_2D_CONFIG3_BOOTSTRAP"\n')
        stream.write(' VARIABLES= "X" "Y" "D" "U" "V" "P"\n')
        stream.write(f" ZONE I={nx} J={ny} F=POINT\n")
        np.savetxt(stream, grid.reshape(-1, 6), fmt="%.9E")


def main() -> None:
    full = read_tecplot(SOURCE)
    if full.shape[:2] != (801, 801):
        raise ValueError(f"Expected 801x801, got {full.shape[:2]}")
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    for resolution, grid in {100: full[::8, ::8], 200: full[::4, ::4], 800: full}.items():
        destination = CASE_DIR / f"flow_{resolution}x{resolution}.dat"
        write_tecplot(destination, grid)
        print(f"Wrote {destination} with node grid {grid.shape[:2]}")


if __name__ == "__main__":
    main()
