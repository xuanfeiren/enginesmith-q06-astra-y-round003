"""LNC=2: which forms of program_id indexing work for DMA writes into an INPUT tensor (in place)?
buf [4, 2, 128, 64] fp32 zeros; core c writes value (c+1)*10+variant into buf[variant, c] using different forms."""
import numpy as np
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, nrun, LAB

@nki.jit
def pw(buf):
    pid = nl.program_id(0)
    for v in range(4):
        t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=t, value=0.0)
        val = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=val, pattern=[[0, 64]], offset=0, channel_multiplier=0)
        nisa.tensor_scalar(dst=val, data=val, op0=nl.add, operand0=(pid + 1) * 10.0 + v)
        if v == 0:
            nisa.dma_copy(dst=buf[0, pid], src=val)                         # direct index
        elif v == 1:
            nisa.dma_copy(dst=buf[1, pid, :, nl.ds(0, 64)], src=val)        # index + ds slice
        elif v == 2:
            nisa.dma_copy(dst=buf[2, pid * 1 + 0, :, :], src=val)           # arithmetic index
        else:
            nisa.dma_copy(dst=buf[3, pid].ap(pattern=[[64, 128], [1, 64]], offset=0), src=val)   # .ap on the view
    o = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    z = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z, value=1.0)
    nisa.dma_copy(dst=o, src=z)
    return o

if __name__ == "__main__":
    buf = np.zeros((4, 2, 128, 64), np.float32)
    build(pw, "p9_pid_write", buf, lnc=2)
    r, so = nrun(f"{LAB}/neff/p9_pid_write.neff", {"buf": buf}, {"buf": (buf.shape, np.float32)})
    b = r["buf"]
    for v in range(4):
        print(f"variant {v}: buf[v,0] values {np.unique(b[v, 0])}, buf[v,1] values {np.unique(b[v, 1])}")
