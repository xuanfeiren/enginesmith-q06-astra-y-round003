import nki
import nki.language as nl
import nki.isa as nisa


def _b(pos, x, o, N, REP):
    pid = nl.program_id(0)
    a = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=x[:, 0:N])
    for r in range(REP):
        b = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=a, dst=b, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                      dma_engine=nisa.dma_engine.dma)
        a2 = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a2, data1=a, data2=b, op=nl.add)
        a = a2
    nisa.dma_copy(dst=o[pid, :, 0:N], src=a)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out


@nki.jit
def sr2048_1(pos, x, o):
    return _b(pos, x, o, 2048, 1)


@nki.jit
def sr2048_9(pos, x, o):
    return _b(pos, x, o, 2048, 9)


@nki.jit
def sr8192_1(pos, x, o):
    return _b(pos, x, o, 8192, 1)


@nki.jit
def sr8192_5(pos, x, o):
    return _b(pos, x, o, 8192, 5)
