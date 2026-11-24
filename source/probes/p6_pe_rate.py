"""TensorE throughput for GEMV-shaped matmuls with the weights already in SBUF (no DMA in the loop).
wstat: weights stationary [128k,128n] tiles, x moving [128,1]  -> out column [128, n-tiles]
xstat: x stationary [128,1], weights moving [128, 512]          -> out row [1, 512]
Each kernel does REPS passes over a resident 2 MB weight block (64 x [128,128] or 16 x [128,512] ... )."""
import numpy as np, sys
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun, LAB

KT, NT = 8, 8          # block: 1024 x 1024 bf16 = 2 MB
REPS = 16              # -> 32 MB of weights "streamed" through TensorE

@nki.jit
def wstat(x, w):
    xT = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=x.ap(pattern=[[1, 128], [128, KT]], offset=0))
    wb = nl.ndarray((128, KT, NT * 128), dtype=w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wb, src=w.reshape((KT, 128, NT * 128)).ap(pattern=[[NT * 128, 128], [128 * NT * 128, KT], [1, NT * 128]], offset=0))
    acc = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for r in range(REPS):
        ps = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.psum)
        for n in range(NT):
            for k in range(KT):
                nisa.nc_matmul(dst=ps[:, n:n + 1], stationary=wb[:, k, n * 128:(n + 1) * 128], moving=xT[:, k:k + 1])
        nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)
    out = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=acc)
    return out

@nki.jit
def xstat(x, w):
    xT = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=x.ap(pattern=[[1, 128], [128, KT]], offset=0))
    wb = nl.ndarray((128, KT, NT * 128), dtype=w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wb, src=w.reshape((KT, 128, NT * 128)).ap(pattern=[[NT * 128, 128], [128 * NT * 128, KT], [1, NT * 128]], offset=0))
    acc = nl.ndarray((1, NT * 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for r in range(REPS):
        ps = nl.ndarray((1, NT * 128), dtype=nl.float32, buffer=nl.psum)
        for j in range(NT * 128 // 512):
            for k in range(KT):
                nisa.nc_matmul(dst=ps[0:1, j * 512:(j + 1) * 512], stationary=xT[:, k:k + 1], moving=wb[:, k, j * 512:(j + 1) * 512])
        nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)
    out = nl.ndarray((1, NT * 128), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=acc)
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, N = KT * 128, NT * 128
    x = rng.standard_normal((1, K)).astype(bf16)
    w = (rng.standard_normal((K, N)) * 0.05).astype(bf16)
    ref = REPS * (x.astype(np.float32) @ w.astype(np.float32))[0]
    for nm, shp in (("wstat", (128, NT)), ("xstat", (1, N))):
        build(globals()[nm], "p6_" + nm, x, w, lnc=1)
        r, so = nrun(f"{LAB}/neff/p6_{nm}.neff", {"x": x, "w": w}, {"output_0": (shp, np.float32)}, bench=200)
        o = r["output_0"]
        o = o.T.reshape(-1) if nm == "wstat" else o.reshape(-1)
        print(nm, "max rel err", np.abs(o - ref).max() / np.abs(ref).max(), [l for l in so.splitlines() if "queued" in l])
