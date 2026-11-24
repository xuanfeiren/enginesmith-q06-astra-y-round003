import nki
import nki.language as nl
import nki.isa as nisa


def _body(pos, x, o, MODE, N, REP):
    a = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=x)
    big = nl.ndarray((128, REP, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    for r in range(REP):
        if MODE == 0:      # 4 transposes into one psum tile, copy out (DVE)
            tp = nl.ndarray((128, 4, 128), dtype=nl.bfloat16, buffer=nl.psum)
            for j in range(4):
                nisa.nc_transpose(dst=tp[:, j, :], data=a[:, j * 128:(j + 1) * 128], engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=big[:, r, :], src=tp.ap(pattern=[[512, 128], [1, 512]], offset=0))
        elif MODE == 1:    # 4 matmuls, different stationaries, moving N cols, accumulate
            ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            for j in range(4):
                nisa.nc_matmul(dst=ps[:, 0:N], stationary=a[:, j * 128:(j + 1) * 128], moving=a[:, 0:N])
            nisa.tensor_copy(dst=big[:, r, 0:N], src=ps[:, 0:N])
        elif MODE == 2:    # same stationary
            ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            for j in range(4):
                nisa.nc_matmul(dst=ps[:, 0:N], stationary=a[:, 0:128], moving=a[:, 0:N])
            nisa.tensor_copy(dst=big[:, r, 0:N], src=ps[:, 0:N])
        elif MODE == 3:    # DVE max reduce from psum, N elems
            ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps[:, 0:N], stationary=a[:, 0:128], moving=a[:, 0:N])
            m = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(4):
                nisa.tensor_reduce(dst=m, op=nl.maximum, data=ps[:, 0:N], axis=1)
            nisa.tensor_copy(dst=big[:, r, 0:1], src=m)
        elif MODE == 4:    # ACT exp from psum with accumulate, N elems
            ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps[:, 0:N], stationary=a[:, 0:128], moving=a[:, 0:N])
            sm = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(4):
                nisa.activation_reduce(dst=big[:, r, 0:N], op=nl.exp, data=ps[:, 0:N], reduce_op=nl.add, reduce_res=sm[:, j:j + 1], scale=0.01)
            nisa.tensor_copy(dst=big[:, r, 1:5], src=sm)
    nisa.dma_copy(dst=o, src=big, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out


@nki.jit
def pt0(pos, x, o):
    return _body(pos, x, o, 0, 128, 1)


@nki.jit
def pt(pos, x, o):
    return _body(pos, x, o, 0, 128, 100)


@nki.jit
def pm128(pos, x, o):
    return _body(pos, x, o, 1, 128, 100)


@nki.jit
def pm512(pos, x, o):
    return _body(pos, x, o, 1, 512, 100)


@nki.jit
def ps128(pos, x, o):
    return _body(pos, x, o, 2, 128, 100)


@nki.jit
def ps512(pos, x, o):
    return _body(pos, x, o, 2, 512, 100)


@nki.jit
def dm512(pos, x, o):
    return _body(pos, x, o, 3, 512, 100)


@nki.jit
def dm128(pos, x, o):
    return _body(pos, x, o, 3, 128, 100)


@nki.jit
def ae512(pos, x, o):
    return _body(pos, x, o, 4, 512, 100)


@nki.jit
def ae128(pos, x, o):
    return _body(pos, x, o, 4, 128, 100)
