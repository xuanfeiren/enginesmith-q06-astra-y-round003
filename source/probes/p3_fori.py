import numpy as np
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun

NTMAX = 8

@nki.jit
def fori_scores(kT, q, nt):
    """S[t, j, c] = sum_d kT[d, j*128+t] q[d, c] for j < nt (runtime), 0 elsewhere."""
    D, T = kT.shape
    S = nl.ndarray((128, NTMAX, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=S, value=0.0)
    q_sb = nl.ndarray((128, 2), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q)
    nt_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nt_sb, src=nt)
    ntr = nisa.register_alloc()
    nisa.register_load(ntr, nt_sb)
    kT3 = kT.reshape((D, NTMAX, 128))

    def body(j):
        kt = nl.ndarray((128, 128), dtype=kT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=kt, src=kT3.ap(pattern=[[T, 128], [1, 128]], offset=0, scalar_offset=j, indirect_dim=1))
        ps = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=kt, moving=q_sb)
        nisa.tensor_copy(dst=S.ap(pattern=[[NTMAX * 2, 128], [1, 2]], offset=0, scalar_offset=j, indirect_dim=1), src=ps)

    nl.fori_loop(0, ntr, body)
    out = nl.ndarray((128, NTMAX, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=S)
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = NTMAX * 128
    kT = rng.standard_normal((128, T)).astype(bf16)
    q = rng.standard_normal((128, 2)).astype(bf16)
    nt = np.array([[5]], np.int32)
    build(fori_scores, "p3_fori", kT, q, nt, lnc=1)
    r, so = nrun(f"{LAB}/neff/p3_fori.neff", {"kT": kT, "q": q, "nt": nt}, {"output_0": ((128, NTMAX, 2), np.float32)})
    ref = (kT.astype(np.float32).T @ q.astype(np.float32)).reshape(NTMAX, 128, 2).transpose(1, 0, 2).copy()
    ref[:, 5:, :] = 0
    o = r["output_0"]
    print("max err", np.abs(o - ref).max(), "ref max", np.abs(ref).max(), "zero tail", (o[:, 5:] == 0).all())
