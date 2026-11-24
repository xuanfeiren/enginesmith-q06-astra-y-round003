"""Which dynamic-offset operations compile inside nl.fori_loop?"""
import numpy as np, sys
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun, LAB
NT = 8

def _common(kT, q, nt):
    q_sb = nl.ndarray((128, 2), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q)
    nt_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nt_sb, src=nt)
    r = nisa.register_alloc()
    nisa.register_load(r, nt_sb)
    return q_sb, r

@nki.jit
def vA(kT, q, nt):
    """DMA HBM -> SBUF big buffer at a dynamic offset inside the loop; use after the loop."""
    q_sb, r = _common(kT, q, nt)
    big = nl.ndarray((128, NT, 128), dtype=kT.dtype, buffer=nl.sbuf)
    nisa.memset(dst=big, value=0.0)
    k3 = kT.reshape((128, NT, 128))
    def body(j):
        nisa.dma_copy(dst=big.ap(pattern=[[NT * 128, 128], [1, 128]], offset=0, scalar_offset=j, indirect_dim=1),
                      src=k3.ap(pattern=[[NT * 128, 128], [1, 128]], offset=0, scalar_offset=j, indirect_dim=1))
    nl.fori_loop(0, r, body)
    out = nl.ndarray((128, NT, 128), dtype=kT.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=big)
    return out

@nki.jit
def vB(kT, q, nt):
    """matmul with the stationary operand at a dynamic SBUF offset (buffer filled statically)."""
    q_sb, r = _common(kT, q, nt)
    big = nl.ndarray((128, NT, 128), dtype=kT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=big, src=kT.reshape((128, NT, 128)))
    S = nl.ndarray((128, NT, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=S, value=0.0)
    def body(j):
        ps = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=big.ap(pattern=[[NT * 128, 128], [1, 128]], offset=0, scalar_offset=j, indirect_dim=1), moving=q_sb)
        nisa.tensor_copy(dst=S.ap(pattern=[[NT * 2, 128], [1, 2]], offset=0, scalar_offset=j, indirect_dim=1), src=ps)
    nl.fori_loop(0, r, body)
    out = nl.ndarray((128, NT, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=S)
    return out

@nki.jit
def vC(kT, q, nt):
    """PSUM accumulation across loop iterations (psum declared outside the loop)."""
    q_sb, r = _common(kT, q, nt)
    k3 = kT.reshape((128, NT, 128))
    acc = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=q_sb[:, 0:1].ap(pattern=[[2, 1], [1, 128]], offset=0) if False else q_sb, moving=q_sb, accumulate=False) if False else None
    zero = nl.ndarray((1, 128), dtype=kT.dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    z2 = nl.ndarray((1, 2), dtype=kT.dtype, buffer=nl.sbuf)
    nisa.memset(dst=z2, value=0.0)
    nisa.nc_matmul(dst=acc, stationary=zero, moving=z2, accumulate=False)
    def body(j):
        kt = nl.ndarray((128, 128), dtype=kT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=kt, src=k3.ap(pattern=[[NT * 128, 128], [1, 128]], offset=0, scalar_offset=j, indirect_dim=1))
        nisa.nc_matmul(dst=acc, stationary=kt, moving=q_sb, accumulate=True)
    nl.fori_loop(0, r, body)
    o = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=acc)
    out = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=o)
    return out

@nki.jit
def vD(kT, q, nt):
    """SBUF accumulation across iterations instead of PSUM: acc_sb += matmul result."""
    q_sb, r = _common(kT, q, nt)
    k3 = kT.reshape((128, NT, 128))
    acc = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    def body(j):
        kt = nl.ndarray((128, 128), dtype=kT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=kt, src=k3.ap(pattern=[[NT * 128, 128], [1, 128]], offset=0, scalar_offset=j, indirect_dim=1))
        ps = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=kt, moving=q_sb)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)
    nl.fori_loop(0, r, body)
    out = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=acc)
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    kT = rng.standard_normal((128, NT * 128)).astype(bf16)
    q = rng.standard_normal((128, 2)).astype(bf16)
    nt = np.array([[5]], np.int32)
    kf = kT.astype(np.float32); qf = q.astype(np.float32)
    S = (kf.T @ qf).reshape(NT, 128, 2)
    refs = {"vA": (lambda: np.where(np.arange(NT)[None, :, None] < 5, kf.reshape(128, NT, 128), 0), ((128, NT, 128), bf16)),
            "vB": (lambda: np.where(np.arange(NT)[None, :, None] < 5, S.transpose(1, 0, 2), 0), ((128, NT, 2), np.float32)),
            "vC": (lambda: S[:5].sum(0), ((128, 2), np.float32)),
            "vD": (lambda: S[:5].sum(0), ((128, 2), np.float32))}
    for nm in sys.argv[1:] or ["vA", "vB", "vC", "vD"]:
        try:
            build(globals()[nm], "p5_" + nm, kT, q, nt, lnc=1)
        except Exception as e:
            msg = str(e); i = msg.find("ERROR"); print(nm, "COMPILE FAIL:", msg[i:i + 220] if i >= 0 else msg[-300:]); continue
        rf, (shp, dt) = refs[nm]
        r, _ = nrun(f"{LAB}/neff/p5_{nm}.neff", {"kT": kT, "q": q, "nt": nt}, {"output_0": (shp, dt)})
        ref = rf(); o = r["output_0"].astype(np.float32)
        print(nm, "max err", np.abs(o - ref).max(), "|ref|", np.abs(ref).max())
