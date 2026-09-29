"""Build a serial CPU execution adapter for the repository's CUDA CFD source.

Numerical kernels are copied mechanically, not reimplemented. Only dispatch,
storage qualifiers and boundary traversal change. The pinned source hash makes
upstream numerical edits explicit; review the adapter before accepting a new hash.
"""
from pathlib import Path
import argparse
import hashlib
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "problems/2d_riemann/src/cfd/weno_gpu_solver.f90"


def cpu_source(source):
    source = re.sub(r"^\s*USE CUDAFOR\s*$", "", source, flags=re.M)
    source = source.replace(", DEVICE, ALLOCATABLE", ", ALLOCATABLE")
    source = source.replace("ATTRIBUTES(DEVICE) ", "")
    source = re.sub(r"^\s*TYPE\(DIM3\).*\n", "", source, flags=re.M)
    source = re.sub(r"^\s*dim(?:Block|Grid) = DIM3.*\n", "", source, flags=re.M)
    source = source.replace("<<<dimGrid, dimBlock>>>", "")
    source = source.replace("        istat = cudaDeviceSynchronize()", "")

    def kernel(match):
        name, body = match.group(1), match.group(2)
        # Exact numerical body, with each valid GPU thread visited once on CPU.
        body, count = re.subn(r"        [ij] = \(blockIdx%.*\n", "", body)
        if count != 2:
            raise ValueError(f"Unexpected thread mapping in {name}")
        if name == "KR_BOUNDARY":
            # Two ordered sweeps avoid GPU's redundant writes and corner races.
            body = body.replace("IF (j >= 0 .AND. j <= NY) THEN", "DO j = 0, NY")
            body = body.replace("IF (i >= -MT .AND. i <= NX+MT) THEN", "DO i = -MT, NX+MT")
            body = body.replace("        END IF", "        END DO")
        else:
            ilo, jlo = (-1, 0) if name.endswith("_X") else (0, -1) if name.endswith("_Y") else (0, 0)
            pos = body.index("        IF (i >=")
            body = body[:pos] + f"        DO j = {jlo}, NY\n        DO i = {ilo}, NX\n" + body[pos:]
            body += "        END DO\n        END DO\n"
        return f"SUBROUTINE {name}{body}    END SUBROUTINE {name}"

    source, count = re.subn(
        r"ATTRIBUTES\(GLOBAL\) SUBROUTINE (KR_\w+)(.*?)    END SUBROUTINE \1",
        kernel, source, flags=re.S,
    )
    if count != 6 or any(s in source.lower() for s in ("cudafor", "blockidx", "<<<", "attributes(")):
        raise ValueError("CUDA adapter did not match the expected six kernels")
    # Unused RK slots / primitive ghost cells are initialized for diagnostic builds.
    source = source.replace("    ! Upload", "    D_d = 0; U_d = 0; V_d = 0; P_d = 0\n    ! Upload")
    source = source.replace("    DO j = -MT, NY+MT\n        DO i = -MT, NX+MT\n           UC_temp", "    UC_temp = 0\n    DO j = -MT, NY+MT\n        DO i = -MT, NX+MT\n           UC_temp", 1)
    # Reject bad inputs instead of entering a zero/negative-timestep loop.
    source = source.replace('    WRITE(*,*) ">>> Grid Size:"',
                            '    IF (NX < 3 .OR. NY < 3 .OR. DT <= 0 .OR. T_END < 0) ERROR STOP "Invalid grid/time"\n    WRITE(*,*) ">>> Grid Size:"')
    return "! Generated CPU dispatch adapter; numerical expressions from weno_gpu_solver.f90.\n" + source


def build(directory, debug=False):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    raw = SOURCE.read_bytes()
    # Filled from the reviewed source, recorded again in the run manifest.
    expected = "c4d3fa6d278133bddcb2159c904277f3c968df9367c17794ca0b2fa79bf2dfd2"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("CFD source changed: review CPU transformation and update its pinned hash")
    generated = directory / "weno_cpu.f90"
    generated.write_text(cpu_source(raw.decode()))
    flags = ["-O0", "-g", "-fcheck=all", "-finit-real=snan"] if debug else ["-O3"]
    command = ["gfortran", *flags, "-ffree-line-length-none", "-ffp-contract=off",
               "-ffpe-trap=invalid,zero,overflow", str(generated), "-o", str(directory / "weno_cpu")]
    subprocess.run(command, cwd=directory, check=True)
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".smoke/cpu-cfd")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(build(args.build_dir, args.debug))
