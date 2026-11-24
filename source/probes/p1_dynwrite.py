import numpy as np, sys
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16

@nki.jit
def dyn_write(cache, pos, row):
    """cache[pos[0], :] = row  (in place, pos read on device); returns pos+1 (int32)."""
    D = row.shape[1]
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    reg = nisa.register_alloc()
    nisa.register_load(reg, p_sb)
    r_sb = nl.ndarray((1, D), dtype=row.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=r_sb, src=row)
    nisa.dma_copy(dst=cache.ap(pattern=[[D, 1], [1, D]], offset=0, scalar_offset=reg, indirect_dim=0), src=r_sb)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p1)
    return out

if __name__ == "__main__":
    T, D = 64, 256
    cache = np.full((T, D), -1, dtype=bf16)
    pos = np.array([[37]], dtype=np.int32)
    row = np.arange(D, dtype=np.float32).astype(bf16)[None, :]
    out, _ = build(dyn_write, "p1_dynwrite", cache, pos, row, lnc=1)
    print("out", out)
    print("row37 ok", np.array_equal(cache[37], row[0]), "others untouched", (cache[np.arange(T) != 37] == -1).all())
    from nkibuild import nrun
    r, so = nrun(f"{LAB}/neff/p1_dynwrite.neff", {"cache": cache, "pos": pos, "row": row},
                 {"cache": ((T, D), bf16), "output_0": ((1, 1), np.int32)}, quiet=False)
    c = r["cache"]
    print("C++: out", r["output_0"], "row37 ok", np.array_equal(c[37], row[0]), "others untouched", (c[np.arange(T) != 37] == -1).all())
