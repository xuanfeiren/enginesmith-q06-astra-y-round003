"""LNC=2: latency of a dependent SBUF->SBUF exchange between the two physical cores (nisa.sendrecv).
Each core holds a [128, 8] fp32 tile a; R times: b <- other core's a; a <- a + b. Result (x0 + x1) * 2^(R-1).
mode 0: dma_engine.dma (default), mode 1: dma_engine.gpsimd_dma."""
import numpy as np, sys, os
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, nrun, LAB


def _ex(x, R, mode):
    pid = nl.program_id(0)
    a = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=x[pid])
    for i in range(R):
        b = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
        if mode == 0:
            nisa.sendrecv(src=a, dst=b, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
        else:
            nisa.sendrecv(src=a, dst=b, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                          dma_engine=nisa.dma_engine.gpsimd_dma)
        a2 = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a2, data1=a, data2=b, op=nl.add)
        a = a2
    out = nl.ndarray((2, 128, 8), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[pid], src=a)
    return out

@nki.jit
def ex1_m0(x): return _ex(x, 1, 0)
@nki.jit
def ex64_m0(x): return _ex(x, 64, 0)
@nki.jit
def ex1_m1(x): return _ex(x, 1, 1)
@nki.jit
def ex64_m1(x): return _ex(x, 64, 1)

if __name__ == "__main__":
    x = np.random.default_rng(0).standard_normal((2, 128, 8)).astype(np.float32)
    for nm in sys.argv[1:]:
        R = int(nm[2:].split("_")[0])
        build(globals()[nm], "p8_" + nm, x, lnc=2)
        r, so = nrun(f"{LAB}/neff/p8_{nm}.neff", {"x": x}, {"output_0": ((2, 128, 8), np.float32)}, bench=200)
        ref = (x[0] + x[1]) * 2.0 ** (R - 1)
        err = max(np.abs(r["output_0"][c] - ref).max() for c in range(2)) / np.abs(ref).max()
        us = float([l for l in so.splitlines() if "BENCH queued" in l][0].split()[2])
        print(f"p8_{nm}: rel err {err:.2g}; queued {us:.1f} us/exec", flush=True)
