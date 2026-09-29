"""Read the solver's structured Tecplot field, including Fortran exponents."""
from pathlib import Path
import hashlib
import re
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_field(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("version https://git-lfs"):
                raise ValueError(f"Retrieve LFS data first: {path}")
            match = re.search(r"ZONE I=\s*(\d+)\s+J=\s*(\d+)", line)
            if match:
                nx, ny = map(int, match.groups())
                break
        else:
            raise ValueError(f"No structured Tecplot header: {path}")
        rows = np.loadtxt((re.sub(r"(\d)([+-]\d{3})", r"\1E\2", line) for line in stream))
    if rows.shape != (nx * ny, 6) or not np.isfinite(rows).all():
        raise ValueError(f"Wrong shape or nonfinite CFD field: {path}")
    field = rows.reshape(ny, nx, 6)
    np.testing.assert_allclose(field[:, :, 0], np.broadcast_to(np.linspace(0, 1, nx), (ny, nx)), atol=6e-7)
    np.testing.assert_allclose(field[:, :, 1], np.broadcast_to(np.linspace(0, 1, ny)[:, None], (ny, nx)), atol=6e-7)
    if np.any(field[:, :, [2, 5]] <= 0):
        raise ValueError(f"Nonpositive density/pressure: {path}")
    return field
