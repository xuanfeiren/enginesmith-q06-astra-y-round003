import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def rsq(pos, x, o):
    a = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=x)
    g = nl.ndarray((128, 2, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g[:, 0, :], data=a, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    nisa.activation(dst=g[:, 1, :], op=nl.rsqrt, data=a)
    nisa.dma_copy(dst=o, src=g)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out
