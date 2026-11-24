"""LNC=2 weight-stationary GEMV y = W x, K = 4096, N per core = NT*128, as in the 8B LM head.
layout 'row': w [2, 128, NT*KT*128] (ws_layout per core: one partition's row is NT*KT*256 B long, so the 128 rows of
               one slab DMA are NT*KT*256 B apart in HBM)
layout 'blk': w [2, NT/SL, 128, SL*KT*128] (each slab DMA reads one contiguous block of 128*SL*KT*256 B)"""
import sys, os, numpy as np
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun, LAB

KT = int(os.environ.get("KT", "32"))
CH = int(os.environ.get("CH", "512"))   # output columns per PSUM tile


def _gv(x, w, blk, SL):
    pid = nl.program_id(0)
    if blk:
        NB = w.shape[1]
    else:
        NB = w.shape[2] // (SL * KT * 128)
    NT = NB * SL
    xT = nl.ndarray((128, KT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=x)
    y = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    c0 = 0
    while c0 < NT:
        n = CH
        if NT - c0 < CH:
            n = NT - c0
        ps = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
        for b in range(c0 // SL, (c0 + n) // SL):
            wt = nl.ndarray((128, SL * KT * 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            if blk:
                nisa.dma_copy(dst=wt, src=w[pid, b])
            else:
                nisa.dma_copy(dst=wt, src=w[pid, :, b * SL * KT * 128:(b + 1) * SL * KT * 128])
            for t in range(SL):
                for k in range(KT):
                    nisa.nc_matmul(dst=ps[:, b * SL + t - c0:b * SL + t - c0 + 1],
                                   stationary=wt[:, (t * KT + k) * 128:(t * KT + k + 1) * 128], moving=xT[:, k:k + 1])
        nisa.tensor_copy(dst=y[:, c0:c0 + n], src=ps[:, 0:n])
        c0 += n
    out = nl.ndarray((2, 128, NT), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[pid], src=y)
    return out

@nki.jit
def row_s1(x, w): return _gv(x, w, 0, 1)
@nki.jit
def blk_s1(x, w): return _gv(x, w, 1, 1)
@nki.jit
def blk_s2(x, w): return _gv(x, w, 1, 2)
@nki.jit
def row_s2(x, w): return _gv(x, w, 0, 2)

if __name__ == "__main__":
    NT = int(os.environ.get("NT", "592"))
    rng = np.random.default_rng(0)
    xT = (rng.standard_normal((128, KT)) * 0.5).astype(bf16)
    wrow = (rng.standard_normal((2, 128, NT * KT * 128)) * 0.02).astype(bf16)
    x = xT.T.reshape(-1).astype(np.float32)
    for nm in sys.argv[1:]:
        SL = int(nm[-1])
        if nm.startswith("blk"):
            w = np.ascontiguousarray(wrow.reshape(2, 128, NT // SL, SL * KT * 128).transpose(0, 2, 1, 3))
        else:
            w = wrow
        full = f"p11_{nm}_NT{NT}_KT{KT}_CH{CH}"
        build(globals()[nm], full, xT, w, lnc=2)
        r, so = nrun(f"{LAB}/neff/{full}.neff", {"x": xT, "w": w}, {"output_0": ((2, 128, NT), np.float32)}, bench=20)
        W4 = wrow.astype(np.float32).reshape(2, 128, NT, KT, 128)        # [c, p(k), nt, kt, j(n)]
        ref = np.einsum("cpntj,pt->cjn", W4, xT.astype(np.float32))      # y[c, j, nt]
        err = np.abs(r["output_0"] - ref).max() / np.abs(ref).max()
        us = float([l for l in so.splitlines() if "BENCH queued" in l][0].split()[2])
        print(f"{full}: rel err {err:.2g}; queued {us:.1f} us for {wrow.nbytes/1e6:.0f} MB -> {wrow.nbytes/us/1e3:.0f} GB/s", flush=True)
