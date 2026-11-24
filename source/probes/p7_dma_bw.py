"""HBM -> SBUF DMA bandwidth. w is [nprog*128, F] bf16, each partition row contiguous in HBM.
Each program streams its 128 rows in slabs of C elements per partition into a rotating set of NB SBUF
buffers; every slab is touched by one tiny tensor_copy so the DMA is not dead code."""
import numpy as np, sys, os
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun, LAB

F = int(os.environ.get("BW_F", "65536"))
NB = 4

def _bw(w, C, mode):
    nprog = nl.num_programs()
    pid = nl.program_id(0) if nprog > 1 else 0
    R, Ftot = w.shape
    acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    bufs = []
    for b in range(NB):
        bufs.append(nl.ndarray((128, C), dtype=w.dtype, buffer=nl.sbuf))
    ns = Ftot // C
    for s in range(ns):
        t = bufs[s % NB]
        src = w[pid * 128:(pid + 1) * 128, s * C:(s + 1) * C]
        if mode == 0:
            nisa.dma_copy(dst=t, src=src)
        elif mode == 1:
            nisa.dma_copy(dst=t, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync if s % 2 == 0 else nisa.engine.scalar)
        elif mode == 2:
            if s % 3 == 2:
                nisa.dma_copy(dst=t, src=src, dge_mode=nisa.dge_mode.swdge)
            else:
                nisa.dma_copy(dst=t, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync if s % 3 == 0 else nisa.engine.scalar)
        elif mode == 3:
            nisa.dma_copy(dst=t, src=src, dge_mode=nisa.dge_mode.none)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=t[:, 0:1], op=nl.add)
    out = nl.ndarray((nprog * 128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[pid * 128:(pid + 1) * 128, :], src=acc)
    return out

@nki.jit
def bw_c2k_m0(w): return _bw(w, 2048, 0)
@nki.jit
def bw_c8k_m0(w): return _bw(w, 8192, 0)
@nki.jit
def bw_c16k_m0(w): return _bw(w, 16384, 0)
@nki.jit
def bw_c8k_m1(w): return _bw(w, 8192, 1)
@nki.jit
def bw_c8k_m2(w): return _bw(w, 8192, 2)
@nki.jit
def bw_c8k_m3(w): return _bw(w, 8192, 3)
@nki.jit
def bw_c2k_m1(w): return _bw(w, 2048, 1)

if __name__ == "__main__":
    names = sys.argv[1:] or ["bw_c2k_m0", "bw_c8k_m0", "bw_c16k_m0", "bw_c8k_m1", "bw_c8k_m2", "bw_c8k_m3", "bw_c2k_m1"]
    for lnc in (1, 2):
        w = (np.random.default_rng(0).standard_normal((lnc * 128, F)) * 0.1).astype(bf16)
        for nm in names:
            n = f"p7_{nm}_lnc{lnc}"
            try:
                build(globals()[nm], n, w, lnc=lnc)
            except Exception as e:
                msg = str(e); i = msg.find("ERROR"); print(n, "COMPILE FAIL", msg[i:i+200]); continue
            r, so = nrun(f"{LAB}/neff/{n}.neff", {"w": w}, {"output_0": ((lnc * 128, 1), np.float32)}, bench=100)
            us = float([l for l in so.splitlines() if "BENCH queued" in l][0].split()[2])
            print(f"{n}: queued {us:.1f} us for {w.nbytes/1e6:.1f} MB -> {w.nbytes/us/1e3:.0f} GB/s (incl. fixed cost)", flush=True)
