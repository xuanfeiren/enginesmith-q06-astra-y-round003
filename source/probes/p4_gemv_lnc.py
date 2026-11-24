"""Does an LNC=1 NEFF (one physical core) get the same HBM bandwidth as LNC=2? bf16 GEMV 4096x4096,
weight pre-swizzled (example/ gemv_swz4 kernel, copied)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/es-dev/skills/specialized-inference-engine/backends/nki-cpp/example"))
import kernels as EX   # read-only import of the backend example kernels
from nkibuild import build, bf16, nrun, LAB

def swizzle(wT, nprog):
    K, N = wT.shape; KT, NPER = K // 128, N // nprog
    return np.ascontiguousarray(wT.reshape(KT, 128, nprog, NPER).transpose(2, 1, 0, 3)).reshape(K, N)

if __name__ == "__main__":
    K, N = 4096, 2048
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, K)).astype(bf16)
    wT = (rng.standard_normal((K, N)) * 0.02).astype(bf16)
    ref = x.astype(np.float32) @ wT.astype(np.float32)
    for lnc in (1, 2):
        nm = f"gemv_swz4_4096x2048_lnc{lnc}"
        ws = swizzle(wT, lnc)
        build(EX.gemv_swz4, nm, x, ws, lnc=lnc)
        r, so = nrun(f"{LAB}/neff/{nm}.neff", {"x": x, "wS": ws}, {"output_0": ((1, N), bf16)}, bench=300)
        err = np.abs(r["output_0"].astype(np.float32) - ref).max()
        us = float([l for l in so.splitlines() if "BENCH queued" in l][0].split()[2])
        print(f"lnc{lnc}: max err {err:.3g} (|ref| {np.abs(ref).max():.3g}); queued {us:.1f} us/exec -> {K*N*2/us/1e3:.0f} GB/s")
