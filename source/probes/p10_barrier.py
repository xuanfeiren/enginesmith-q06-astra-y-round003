"""LNC=2: core c writes its tile into an internal nl.shared_hbm tensor, core_barrier, reads the other core's tile.
Does the read see the other core's write?  variants:
  v0: internal shared_hbm scratch, core_barrier(data=scratch)
  v1: same, plus a second barrier
  v2: scratch is a kernel INPUT tensor (caller's buffer), core_barrier(data=it)"""
import numpy as np, sys
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, nrun, LAB


def _body(x, scr, nbar):
    pid = nl.program_id(0)
    t = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x[pid])
    t2 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t2, data=t, op0=nl.add, operand0=1.0)
    nisa.dma_copy(dst=scr[pid], src=t2)
    for i in range(nbar):
        nisa.core_barrier(data=scr, cores=(0, 1))
    r = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=r, src=scr[1 - pid])
    out = nl.ndarray((2, 128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[pid], src=r)
    return out

@nki.jit
def v0(x):
    scr = nl.ndarray((2, 128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    return _body(x, scr, 1)
@nki.jit
def v1(x):
    scr = nl.ndarray((2, 128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    return _body(x, scr, 2)
@nki.jit
def v2(x, scr):
    return _body(x, scr, 1)

if __name__ == "__main__" and sys.argv[1].startswith("v"):
    x = np.random.default_rng(0).standard_normal((2, 128, 512)).astype(np.float32)
    for nm in sys.argv[1:]:
        args = [x] if nm != "v2" else [x, np.zeros_like(x)]
        build(globals()[nm], "p10_" + nm, *args, lnc=2)
        ins = {"x": x} if nm != "v2" else {"x": x, "scr": np.zeros_like(x)}
        r, so = nrun(f"{LAB}/neff/p10_{nm}.neff", ins, {"output_0": ((2, 128, 512), np.float32)})
        o = r["output_0"]
        for c in range(2):
            ok = np.allclose(o[c], x[1 - c] + 1)
            print(f"p10_{nm}: core {c} read the other core's tile: {'yes' if ok else 'NO'} (zeros: {np.all(o[c] == 0)})")


def _body2(x, scr, mode):
    """x [128, 8, 256]; core c writes token block c of scr [128, 8, 256], barrier, reads all 256 tokens."""
    pid = nl.program_id(0)
    t = nl.ndarray((128, 8, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x[:, :, nl.ds(pid * 128, 128)])
    if mode == 0:
        nisa.dma_copy(dst=scr[:, :, nl.ds(pid * 128, 128)], src=t)
    elif mode == 2:     # slow producer: 400 matmuls before the write (the value written depends on them)
        w = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=w, value=0.0)
        mv = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=mv, value=1.0)
        ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        for i in range(400):
            nisa.nc_matmul(dst=ps, stationary=w, moving=mv)
        z = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=z, src=ps)
        t3 = nl.ndarray((128, 8, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t3, data1=t, data2=z.ap(pattern=[[512, 128], [0, 8], [1, 128]], offset=0), op=nl.add)
        nisa.dma_copy(dst=scr[:, :, nl.ds(pid * 128, 128)], src=t3)
    else:
        for b in range(2):     # static slices, predicated by arithmetic: write both blocks, other block from x too
            pass
    nisa.core_barrier(data=scr, cores=(0, 1))
    r = nl.ndarray((128, 8, 256), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=r, src=scr[:, :, nl.ds(0, 256)])
    out = nl.ndarray((2, 128, 8, 256), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[pid], src=r)
    return out

@nki.jit
def w0(x):
    scr = nl.ndarray((128, 8, 256), dtype=nl.float32, buffer=nl.shared_hbm)
    return _body2(x, scr, 0)
@nki.jit
def w3(x):
    scr = nl.ndarray((128, 8, 256), dtype=nl.float32, buffer=nl.shared_hbm)
    return _body2(x, scr, 2)
@nki.jit
def w2(x, scr):
    return _body2(x, scr, 0)

if __name__ == "__main__" and sys.argv[1].startswith("w"):
    x = np.random.default_rng(0).standard_normal((128, 8, 256)).astype(np.float32)
    for nm in sys.argv[1:]:
        args = [x] if nm in ("w0", "w3") else [x, np.zeros_like(x)]
        build(globals()[nm], "p10_" + nm, *args, lnc=2)
        ins = {"x": x} if nm in ("w0", "w3") else {"x": x, "scr": np.zeros_like(x)}
        r, so = nrun(f"{LAB}/neff/p10_{nm}.neff", ins, {"output_0": ((2, 128, 8, 256), np.float32)})
        o = r["output_0"]
        for c in range(2):
            print(f"p10_{nm}: core {c}: block0 ok {np.array_equal(o[c][:, :, :128], x[:, :, :128])}, "
                  f"block1 ok {np.array_equal(o[c][:, :, 128:], x[:, :, 128:])}")
