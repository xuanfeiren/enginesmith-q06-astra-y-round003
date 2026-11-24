import numpy as np
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun

@nki.jit
def dyn_step(cache, pos, row):
    """cache[pos, :] = row + pos (in place); pos += 1 (in place). Output: copy of pos before increment."""
    D = row.shape[1]
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    reg = nisa.register_alloc()
    nisa.register_load(reg, p_sb)
    pf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pf, src=p_sb)
    r_sb = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=r_sb, src=row)
    r2 = nl.ndarray((1, D), dtype=row.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r2, data=r_sb, op0=nl.add, operand0=pf)
    nisa.dma_copy(dst=cache.ap(pattern=[[D, 1], [1, D]], offset=0, scalar_offset=reg, indirect_dim=0), src=r2)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
    nisa.dma_copy(dst=pos, src=p1)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p_sb)
    return out

if __name__ == "__main__":
    T, D = 64, 256
    cache = np.full((T, D), -1, dtype=bf16)
    pos = np.array([[37]], dtype=np.int32)
    row = np.arange(D, dtype=np.float32).astype(bf16)[None, :]
    for lnc in (1, 2):
        build(dyn_step, f"p2_lnc{lnc}", cache, pos, row, lnc=lnc)
        for mode in ("sync", "queue"):
            r, so = nrun(ff"{LAB}/neff/p2_lnc{lnc}.neff", {"cache": cache, "pos": pos, "row": row},
                         {"cache": ((T, D), bf16), "pos": ((1, 1), np.int32), "output_0": ((1, 1), np.int32)}, n_exec=5, mode=mode)
            c = r["cache"].astype(np.float32)
            exp = np.full((T, D), -1, np.float32)
            for p in range(37, 42): exp[p] = np.arange(D) + p
            exp = exp.astype(bf16).astype(np.float32)
            print(f"lnc{lnc} {mode}: pos={r['pos'][0,0]} out={r['output_0'][0,0]} cache ok={np.array_equal(c, exp)}")
