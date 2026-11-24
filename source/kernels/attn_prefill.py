"""Causal prefill attention for Qwen3-0.6B (16 q heads, 8 kv heads, head_dim 128), LNC=2, in the prefill's layouts.

Core c (nl.program_id(0)) handles q heads 8c..8c+7 and kv heads 4c..4c+3 (the prefill's QKV split, so it only reads
what it computed itself). Inputs (HBM, bf16): qT [16, 128 d, S], kT [8, 128 d, S], v [8, S, 128 d]. Output:
oT [16, 128 d, S] bf16 (feature-major, what the o_proj phase reads).

Per kv head g and query block i (128 queries, keys 0 .. 128(i+1)):
  scores  s[q, k] = q . k in PSUM, row layout (q on partitions): stationary q^T block [128 d, 128 q], moving k^T
          chunks [128 d, <= 512 k]
  mask    the diagonal 128 x 128 block gets an additive -30000 above the diagonal (VectorE, into SBUF)
  softmax row max over the keys (VectorE, straight from PSUM); e = exp(scale * s - scale * max) rounded to bf16
          with the fp32 row sum accumulated by ScalarE (activation reduce); 1 / sum on VectorE
  PV      e blocks transposed on the PE (exact), O^T [128 d, q] = sum_j V_j^T e_j^T with V_j [128 k, 128 d]
          stationary and the two q heads of the kv head side by side as the moving operand
  norm    O^T * (1 / sum)[q], the per-query factor broadcast over the d partitions with a diag(1/sum) matmul
"""
import nki
import nki.language as nl
import nki.isa as nisa

D = 128
NQL, NKL = 8, 4
SCALE = 0.08838834764831845
NEG = -30000.0
import os as _os
CPE = int(_os.environ.get('AP_CPE', '0'))
NOBC = int(_os.environ.get('AP_NOBC', '0'))
DT = int(_os.environ.get('AP_DT', '2'))   # 2: DMA transposes for S >= 1024 (measured: 172 -> 151 us), PE below
DEPTH0 = int(_os.environ.get('AP_DEPTH', '0'))   # 0: 2 for S < 1024, 3 above (measured 1024: 150 -> 141 us)
DTMIN = int(_os.environ.get('AP_DTMIN', '0'))   # ... for query blocks with more than DTMIN key blocks   # 1: e^T by DMA transposes instead of PE transposes + PSUM copies
OFFE = nisa.engine.gpsimd
if int(_os.environ.get('AP_OFFE', '0')) == 0:
    OFFE = nisa.engine.vector


def _attn_consts():
    """identity (bf16, fp32) and the additive causal mask of a diagonal block (0 where k <= q, NEG above)."""
    idf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=idf, pattern=[[-1, 128]], offset=0, channel_multiplier=1)      # q - k
    msk = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=msk, data=idf, op0=nl.less, operand0=0.0, op1=nl.multiply, operand1=NEG)
    nisa.tensor_scalar(dst=idf, data=idf, op0=nl.equal, operand0=0.0)
    ident = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ident, src=idf)
    ones = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    return idf, ident, msk, ones


def _hd(dst, src):
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def attn_prefill(qT, kT, v, oT, pid, S):
    NB = S // 128
    idf, ident, msk, ones = _attn_consts()
    for g in range(NKL):
        kv = pid * NKL + g
        kt = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        _hd(dst=kt, src=kT[kv])
        vt = nl.ndarray((128, NB, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        _hd(dst=vt, src=v[kv].ap(pattern=[[D, 128], [128 * D, NB], [1, D]], offset=0))
        qts = []
        for hh in range(2):
            qt = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
            _hd(dst=qt, src=qT[pid * NQL + 2 * g + hh])
            qts.append(qt)
        for i in range(NB):
            K = (i + 1) * 128
            eT = nl.ndarray((128, NB, 256), dtype=nl.bfloat16, buffer=nl.sbuf)   # [k, block j, (head, q)]
            rinv = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
            for hh in range(2):
                ps = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
                c0 = 0
                while c0 < K:
                    n = 512
                    if K - c0 < 512:
                        n = K - c0
                    nisa.nc_matmul(dst=ps[:, c0:c0 + n], stationary=qts[hh][:, i * 128:(i + 1) * 128],
                                   moving=kt[:, c0:c0 + n])
                    c0 += n
                dg = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dg, data1=ps[:, i * 128:K], data2=msk, op=nl.add)
                mx = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=mx[:, 0:1], op=nl.maximum, data=dg, axis=1)
                if i > 0:
                    nisa.tensor_reduce(dst=mx[:, 1:2], op=nl.maximum, data=ps[:, 0:i * 128], axis=1)
                    nisa.tensor_tensor(dst=mx[:, 0:1], data1=mx[:, 0:1], data2=mx[:, 1:2], op=nl.maximum)
                nm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=nm, data=mx[:, 0:1], op0=nl.multiply, operand0=-SCALE)
                e = nl.ndarray((128, S), dtype=nl.bfloat16, buffer=nl.sbuf)
                sm = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=e[:, i * 128:K], op=nl.exp, data=dg, reduce_op=nl.add,
                                       reduce_res=sm[:, 0:1], bias=nm, scale=SCALE)
                if i > 0:
                    nisa.activation_reduce(dst=e[:, 0:i * 128], op=nl.exp, data=ps[:, 0:i * 128], reduce_op=nl.add,
                                           reduce_res=sm[:, 1:2], bias=nm, scale=SCALE)
                    nisa.tensor_tensor(dst=sm[:, 0:1], data1=sm[:, 0:1], data2=sm[:, 1:2], op=nl.add)
                nisa.reciprocal(dst=rinv[:, hh:hh + 1], data=sm[:, 0:1])
                # e^T blocks, four per PSUM tile
                j0 = 0
                while j0 <= i:
                    nj = 4
                    if i + 1 - j0 < 4:
                        nj = i + 1 - j0
                    tp = nl.ndarray((128, 4, 128), dtype=nl.bfloat16, buffer=nl.psum)
                    for jj in range(nj):
                        nisa.nc_transpose(dst=tp[:, jj, :], data=e[:, (j0 + jj) * 128:(j0 + jj + 1) * 128],
                                          engine=nisa.engine.tensor)
                    nisa.tensor_copy(dst=eT[:, j0:j0 + nj, hh * 128:(hh + 1) * 128], src=tp[:, 0:nj, :])
                    j0 += nj
            po = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.psum)
            for j in range(i + 1):
                nisa.nc_matmul(dst=po, stationary=vt[:, j, :], moving=eT[:, j, :])
            # broadcast 1 / sum over the d partitions: ones^T @ diag(rinv)
            dgm = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)
            for hh in range(2):
                nisa.tensor_scalar(dst=dgm[:, hh * 128:(hh + 1) * 128], data=idf, op0=nl.multiply,
                                   operand0=rinv[:, hh:hh + 1])
            pb = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pb, stationary=ones, moving=dgm, is_stationary_onezero=True)
            bc = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bc, src=pb)
            ob = nl.ndarray((D, 256), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ob, data1=po, data2=bc, op=nl.multiply)
            for hh in range(2):
                _hd(dst=oT[pid * NQL + 2 * g + hh, :, i * 128:(i + 1) * 128], src=ob[:, hh * 128:(hh + 1) * 128])


def _ap_s1(n, items, qts, kts, ps_l):
    g = items[n][0]
    i = items[n][1]
    hh = items[n][2]
    K = (i + 1) * 128
    ps = nl.ndarray((128, K), dtype=nl.float32, buffer=nl.psum)
    c0 = 0
    while c0 < K:
        w = 512
        if K - c0 < 512:
            w = K - c0
        nisa.nc_matmul(dst=ps[:, c0:c0 + w], stationary=qts[2 * g + hh][:, i * 128:(i + 1) * 128],
                       moving=kts[g][:, c0:c0 + w])
        c0 += w
    ps_l[n] = ps


def _ap_s2(n, items, msk, ps_l, e_l, rinv_l, eT_l, NB):
    if NB <= 2:
        _ap_s2_late(n, items, msk, ps_l, e_l, rinv_l, eT_l, NB)
        return
    # Preserve the original instruction/allocation order for the 1024 bucket.
    g = items[n][0]
    i = items[n][1]
    hh = items[n][2]
    K = (i + 1) * 128
    ps = ps_l[n]
    # Mask the diagonal block, then store z = -SCALE * scores while reducing min(z).
    # exp(-z + min(z)) needs no separate row-maximum scaling instruction.
    nisa.tensor_tensor(dst=ps[:, i * 128:K], data1=ps[:, i * 128:K], data2=msk, op=nl.add)
    nm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_reduce(dst=ps[:, 0:K], data=ps[:, 0:K], op0=nl.multiply, operand0=-SCALE,
                              reduce_op=nl.minimum, reduce_res=nm)
    e = nl.ndarray((128, K), dtype=nl.bfloat16, buffer=nl.sbuf)
    sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=e, op=nl.exp, data=ps[:, 0:K], reduce_op=nl.add, reduce_res=sm, bias=nm, scale=-1.0)
    pi = n // 2
    if hh == 0:
        rinv_l[pi] = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
        eT_l[pi] = nl.ndarray((128, NB, 256), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.reciprocal(dst=rinv_l[pi][:, hh:hh + 1], data=sm)
    e_l[n] = e


def _ap_s2_late(n, items, msk, ps_l, e_l, rinv_l, eT_l, NB):
    g = items[n][0]
    i = items[n][1]
    hh = items[n][2]
    K = (i + 1) * 128
    ps = ps_l[n]
    # Mask the diagonal block, then store z = -SCALE * scores while reducing min(z).
    # exp(-z + min(z)) needs no separate row-maximum scaling instruction.
    nisa.tensor_tensor(dst=ps[:, i * 128:K], data1=ps[:, i * 128:K], data2=msk, op=nl.add)
    nm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_reduce(dst=ps[:, 0:K], data=ps[:, 0:K], op0=nl.multiply, operand0=-SCALE,
                              reduce_op=nl.minimum, reduce_res=nm)
    pi = n // 2
    if hh == 0:
        rinv_l[pi] = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
        eT_l[pi] = nl.ndarray((128, NB, 256), dtype=nl.bfloat16, buffer=nl.sbuf)
    e = nl.ndarray((128, K), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation_reduce(dst=e, op=nl.exp, data=ps[:, 0:K], reduce_op=nl.add,
                           reduce_res=rinv_l[pi][:, hh:hh + 1], bias=nm, scale=-1.0)
    e_l[n] = e


def _ap_transpose(n, items, e_l, eT_l, NB):
    i = items[n][1]
    hh = items[n][2]
    e = e_l[n]
    pi = n // 2
    eT = eT_l[pi]
    j0 = 0
    while j0 <= i and (DT == 1 or (DT == 2 and NB >= 8)) and i + 1 > DTMIN:
        nj = 4
        if i + 1 - j0 < 4:
            nj = i + 1 - j0
        K = (i + 1) * 128
        NBT = eT.shape[1]
        nisa.dma_transpose(dst=eT.ap([[NBT * 256, 128], [1, 1], [256, nj], [1, 128]], offset=j0 * 256 + hh * 128),
                           src=e.ap([[K, 128], [1, 1], [128, nj], [1, 128]], offset=j0 * 128))
        j0 += nj
    while j0 <= i:
        nj = 4
        if i + 1 - j0 < 4:
            nj = i + 1 - j0
        tp = nl.ndarray((128, 4, 128), dtype=nl.bfloat16, buffer=nl.psum)
        for jj in range(nj):
            nisa.nc_transpose(dst=tp[:, jj, :], data=e[:, (j0 + jj) * 128:(j0 + jj + 1) * 128],
                              engine=nisa.engine.tensor)
        if CPE == 1:
            nisa.tensor_copy(dst=eT[:, j0:j0 + nj, hh * 128:(hh + 1) * 128], src=tp[:, 0:nj, :],
                             engine=nisa.engine.gpsimd)
        elif CPE == 2 or (CPE == 3 and (j0 // 4) % 2 == 1):
            nisa.activation(dst=eT[:, j0:j0 + nj, hh * 128:(hh + 1) * 128], op=nl.copy, data=tp[:, 0:nj, :])
        else:
            nisa.tensor_copy(dst=eT[:, j0:j0 + nj, hh * 128:(hh + 1) * 128], src=tp[:, 0:nj, :])
        j0 += nj


def _ap_pv(n, items, rinv_l, eT_l, vts, idf, ones, oT, pid, NB, osb=None, late_norm=False):
    g = items[n][0]
    i = items[n][1]
    hh = items[n][2]
    pi = n // 2
    eT = eT_l[pi]
    if hh == 1:
        if late_norm:
            # Both head sums are ready; elementwise reciprocal has the same
            # arithmetic as the original two separate instructions.
            rinv = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rinv, data=rinv_l[pi])
        else:
            rinv = rinv_l[pi]
        po = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.psum)
        for j in range(i + 1):
            nisa.nc_matmul(dst=po, stationary=vts[g][:, j, :], moving=eT[:, j, :])
        dgm = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)
        if DG1:
            nisa.tensor_tensor(dst=dgm, data1=idf.ap(pattern=[[128, 128], [0, 2], [1, 128]], offset=0),
                               data2=rinv.ap(pattern=[[2, 128], [1, 2], [0, 128]], offset=0), op=nl.multiply)
        else:
            for h2 in range(2):
                nisa.tensor_scalar(dst=dgm[:, h2 * 128:(h2 + 1) * 128], data=idf, op0=nl.multiply,
                                   operand0=rinv[:, h2:h2 + 1], engine=OFFE)
        pb = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=pb, stationary=ones, moving=dgm, is_stationary_onezero=True)
        if osb is not None:
            # SBUF output [128 d, 8 local heads, S]: the pair's two heads are adjacent
            bc = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bc, src=pb)
            S_ = osb.shape[2]
            nisa.tensor_tensor(dst=osb.ap(pattern=[[8 * S_, 128], [S_, 2], [1, 128]], offset=2 * g * S_ + i * 128),
                               data1=po, data2=bc, op=nl.multiply)
            return
        ob = nl.ndarray((D, 256), dtype=nl.bfloat16, buffer=nl.sbuf)
        if NOBC:
            nisa.tensor_tensor(dst=ob, data1=po, data2=pb, op=nl.multiply)
        else:
            bc = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bc, src=pb)
            nisa.tensor_tensor(dst=ob, data1=po, data2=bc, op=nl.multiply)
        for h2 in range(2):
            _hd(dst=oT[pid * NQL + 2 * g + h2, :, i * 128:(i + 1) * 128], src=ob[:, h2 * 128:(h2 + 1) * 128])


def _ap_s3(n, items, e_l, rinv_l, eT_l, vts, idf, ones, oT, pid, NB, osb=None):
    if NB <= 2:
        _ap_transpose(n, items, e_l, eT_l, NB)
        _ap_pv(n, items, rinv_l, eT_l, vts, idf, ones, oT, pid, NB, osb, True)
        return
    # Keep this path intact: sharing its allocation scope with the split stages
    # regressed the 1024 layer by 0.85% despite identical arithmetic and bytes.
    g = items[n][0]
    i = items[n][1]
    hh = items[n][2]
    e = e_l[n]
    pi = n // 2
    eT = eT_l[pi]
    j0 = 0
    while j0 <= i and (DT == 1 or (DT == 2 and NB >= 8)) and i + 1 > DTMIN:
        nj = 4
        if i + 1 - j0 < 4:
            nj = i + 1 - j0
        K = (i + 1) * 128
        NBT = eT.shape[1]
        nisa.dma_transpose(dst=eT.ap([[NBT * 256, 128], [1, 1], [256, nj], [1, 128]], offset=j0 * 256 + hh * 128),
                           src=e.ap([[K, 128], [1, 1], [128, nj], [1, 128]], offset=j0 * 128))
        j0 += nj
    while j0 <= i:
        nj = 4
        if i + 1 - j0 < 4:
            nj = i + 1 - j0
        tp = nl.ndarray((128, 4, 128), dtype=nl.bfloat16, buffer=nl.psum)
        for jj in range(nj):
            nisa.nc_transpose(dst=tp[:, jj, :], data=e[:, (j0 + jj) * 128:(j0 + jj + 1) * 128],
                              engine=nisa.engine.tensor)
        if CPE == 1:
            nisa.tensor_copy(dst=eT[:, j0:j0 + nj, hh * 128:(hh + 1) * 128], src=tp[:, 0:nj, :],
                             engine=nisa.engine.gpsimd)
        elif CPE == 2 or (CPE == 3 and (j0 // 4) % 2 == 1):
            nisa.activation(dst=eT[:, j0:j0 + nj, hh * 128:(hh + 1) * 128], op=nl.copy, data=tp[:, 0:nj, :])
        else:
            nisa.tensor_copy(dst=eT[:, j0:j0 + nj, hh * 128:(hh + 1) * 128], src=tp[:, 0:nj, :])
        j0 += nj
    if hh == 1:
        rinv = rinv_l[pi]
        po = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.psum)
        for j in range(i + 1):
            nisa.nc_matmul(dst=po, stationary=vts[g][:, j, :], moving=eT[:, j, :])
        dgm = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)
        if DG1:
            nisa.tensor_tensor(dst=dgm, data1=idf.ap(pattern=[[128, 128], [0, 2], [1, 128]], offset=0),
                               data2=rinv.ap(pattern=[[2, 128], [1, 2], [0, 128]], offset=0), op=nl.multiply)
        else:
            for h2 in range(2):
                nisa.tensor_scalar(dst=dgm[:, h2 * 128:(h2 + 1) * 128], data=idf, op0=nl.multiply,
                                   operand0=rinv[:, h2:h2 + 1], engine=OFFE)
        pb = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=pb, stationary=ones, moving=dgm, is_stationary_onezero=True)
        if osb is not None:
            # SBUF output [128 d, 8 local heads, S]: the pair's two heads are adjacent
            bc = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bc, src=pb)
            S_ = osb.shape[2]
            nisa.tensor_tensor(dst=osb.ap(pattern=[[8 * S_, 128], [S_, 2], [1, 128]], offset=2 * g * S_ + i * 128),
                               data1=po, data2=bc, op=nl.multiply)
            return
        ob = nl.ndarray((D, 256), dtype=nl.bfloat16, buffer=nl.sbuf)
        if NOBC:
            nisa.tensor_tensor(dst=ob, data1=po, data2=pb, op=nl.multiply)
        else:
            bc = nl.ndarray((D, 256), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bc, src=pb)
            nisa.tensor_tensor(dst=ob, data1=po, data2=bc, op=nl.multiply)
        for h2 in range(2):
            _hd(dst=oT[pid * NQL + 2 * g + h2, :, i * 128:(i + 1) * 128], src=ob[:, h2 * 128:(h2 + 1) * 128])


def attn_prefill_pipe(qT, kT, v, oT, pid, S, DEPTH=0):
    """attn_prefill with the work items (kv head, query block, q head) software-pipelined in program order: the
    QK matmuls of item n + DEPTH, the softmax of item n + 1 and the transposes / PV of item n are interleaved, so
    each engine's in-order stream has independent work while the previous item's dependency chain completes."""
    NB = S // 128
    if DEPTH == 0:
        DEPTH = DEPTH0
        if DEPTH0 == 0:
            DEPTH = 2
            if S >= 1024:
                DEPTH = 3
    idf, ident, msk, ones = _attn_consts()
    kts = []
    vts = []
    qts = []
    for g in range(NKL):
        kv = pid * NKL + g
        kt = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        _hd(dst=kt, src=kT[kv])
        kts.append(kt)
        vt = nl.ndarray((128, NB, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        _hd(dst=vt, src=v[kv].ap(pattern=[[D, 128], [128 * D, NB], [1, D]], offset=0))
        vts.append(vt)
        for hh in range(2):
            qt = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
            _hd(dst=qt, src=qT[pid * NQL + 2 * g + hh])
            qts.append(qt)
    items = []
    for g in range(NKL):
        for i in range(NB):
            for hh in range(2):
                items.append((g, i, hh))
    N = len(items)
    ps_l = []
    e_l = []
    for n in range(N):
        ps_l.append(0)
        e_l.append(0)
    rinv_l = []
    eT_l = []
    for n in range(N // 2):
        rinv_l.append(0)
        eT_l.append(0)
    for n in range(DEPTH):
        _ap_s1(n, items, qts, kts, ps_l)
    _ap_s2(0, items, msk, ps_l, e_l, rinv_l, eT_l, NB)
    for n in range(N):
        if n + DEPTH < N:
            _ap_s1(n + DEPTH, items, qts, kts, ps_l)
        if n + 1 < N:
            _ap_s2(n + 1, items, msk, ps_l, e_l, rinv_l, eT_l, NB)
        _ap_s3(n, items, e_l, rinv_l, eT_l, vts, idf, ones, oT, pid, NB)


def attn_prefill_sb(qsb, ksb, vsb, osb, S, DEPTH=2):
    """attn_prefill_pipe on SBUF-resident tiles of this core's heads: qsb [128 d, 8, S], ksb [128 d, 4, S],
    vsb [128 t, S / 128, 4, 128 d] (bf16); output osb [128 d, 8, S] bf16 (SBUF)."""
    NB = S // 128
    idf, ident, msk, ones = _attn_consts()
    kts = []
    vts = []
    qts = []
    for g in range(NKL):
        kts.append(ksb[:, g, :])
        vts.append(vsb[:, :, g, :])
        for hh in range(2):
            qts.append(qsb[:, 2 * g + hh, :])
    items = []
    for g in range(NKL):
        for i in range(NB):
            for hh in range(2):
                items.append((g, i, hh))
    N = len(items)
    ps_l = []
    e_l = []
    for n in range(N):
        ps_l.append(0)
        e_l.append(0)
    rinv_l = []
    eT_l = []
    for n in range(N // 2):
        rinv_l.append(0)
        eT_l.append(0)
    for n in range(DEPTH):
        _ap_s1(n, items, qts, kts, ps_l)
    _ap_s2(0, items, msk, ps_l, e_l, rinv_l, eT_l, NB)
    for n in range(N):
        if n + DEPTH < N:
            _ap_s1(n + DEPTH, items, qts, kts, ps_l)
        if n + 1 < N:
            _ap_s2(n + 1, items, msk, ps_l, e_l, rinv_l, eT_l, NB)
        _ap_s3(n, items, e_l, rinv_l, eT_l, vts, idf, ones, 0, 0, NB, osb)


BDEPTH = int(_os.environ.get('AP_BDEPTH', '3'))
NMACT = int(_os.environ.get('AP_NMACT', '1'))   # attn_prefill_big: -scale * max on ScalarE (1,104 -> 1,090 us)
DG1 = int(_os.environ.get('AP_DG1', '0'))
ACC = int(_os.environ.get('AP_ACC', '1'))   # attn_prefill_big: row max chained in the VectorE accumulator
SEC = int(_os.environ.get('AP_SEC', '1024'))   # attn_prefill_big: keys per PSUM section (2 banks)


def _ab_s1(n, items, qts, kts, msk, pred, sc_l, mx_l, S):
    """scores of item n: QK in PSUM sections of SEC keys -> fp32 SBUF row (VectorE copy fused with the row max per
    section); the diagonal 128-key block gets the causal mask."""
    g = items[n][0]
    i = items[n][1]
    hh = items[n][2]
    K = (i + 1) * 128
    D0 = i * 128
    NSEC = (S + SEC - 1) // SEC + 1
    sc = nl.ndarray((128, K), dtype=nl.float32, buffer=nl.sbuf)
    mx = nl.ndarray((128, NSEC), dtype=nl.float32, buffer=nl.sbuf)
    k0 = 0
    s = 0
    while k0 < K:
        k1 = k0 + SEC
        if k1 > K:
            k1 = K
        ps = nl.ndarray((128, k1 - k0), dtype=nl.float32, buffer=nl.psum)
        c0 = k0
        while c0 < k1:
            w = 512
            if k1 - c0 < 512:
                w = k1 - c0
            nisa.nc_matmul(dst=ps[:, c0 - k0:c0 - k0 + w], stationary=qts[2 * g + hh][:, i * 128:(i + 1) * 128],
                           moving=kts[g][:, c0:c0 + w])
            c0 += w
        e0 = k1
        if e0 > D0:
            e0 = D0
        if e0 > k0:
            if ACC:
                cmd = nisa.reduce_cmd.reduce
                if k0 == 0:
                    cmd = nisa.reduce_cmd.reset_reduce
                nisa.tensor_scalar_reduce(dst=sc[:, k0:e0], data=ps[:, 0:e0 - k0], op0=nl.add, operand0=0.0,
                                          reduce_op=nl.maximum, reduce_res=None, reduce_cmd=cmd)
            else:
                nisa.tensor_scalar_reduce(dst=sc[:, k0:e0], data=ps[:, 0:e0 - k0], op0=nl.add, operand0=0.0,
                                          reduce_op=nl.maximum, reduce_res=mx[:, s:s + 1])
                s += 1
        if k1 == K:
            if ACC:
                cmd = nisa.reduce_cmd.reduce
                if D0 == 0:
                    cmd = nisa.reduce_cmd.reset_reduce
                nisa.select_reduce(dst=sc[:, D0:K], predicate=pred, on_true=ps[:, D0 - k0:K - k0], on_false=NEG,
                                   reduce_res=mx[:, 0:1], reduce_cmd=cmd, reduce_op=nl.maximum)
                s = 1
            else:
                nisa.tensor_tensor(dst=sc[:, D0:K], data1=ps[:, D0 - k0:K - k0], data2=msk, op=nl.add)
                nisa.tensor_reduce(dst=mx[:, s:s + 1], op=nl.maximum, data=sc[:, D0:K], axis=1)
                s += 1
        k0 = k1
    sc_l[n] = sc
    mx_l[n] = (mx, s)


def _ab_s2(n, items, sc_l, mx_l, e_l, rinv_l, eT_l, NB):
    g = items[n][0]
    i = items[n][1]
    hh = items[n][2]
    K = (i + 1) * 128
    sc = sc_l[n]
    mx = mx_l[n][0]
    ns = mx_l[n][1]
    if ns > 1:
        m = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=mx[:, 0:ns], axis=1)
    else:
        m = mx[:, 0:1]
    nm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    if NMACT:
        nisa.activation(dst=nm, op=nl.copy, data=m, scale=-SCALE)
    else:
        nisa.tensor_scalar(dst=nm, data=m, op0=nl.multiply, operand0=-SCALE, engine=nisa.engine.vector)
    pi = n // 2
    if hh == 0:
        rinv_l[pi] = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
        eT_l[pi] = nl.ndarray((128, NB, 256), dtype=nl.bfloat16, buffer=nl.sbuf)
    e = nl.ndarray((128, K), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation_reduce(dst=e, op=nl.exp, data=sc, reduce_op=nl.add,
                           reduce_res=rinv_l[pi][:, hh:hh + 1], bias=nm, scale=SCALE)
    e_l[n] = e


def attn_prefill_big(qT, kT, v, oT, pid, S, DEPTH=BDEPTH):
    """attn_prefill_pipe for long prompts: scores staged in SBUF (fp32 row per query block) so that PSUM holds only
    SEC keys at a time; one exp per (q head, query block). Transpose e immediately after exp, then issue PV
    farther behind QK so DMA can overlap independent work. The max view and deferred reciprocal keep VectorE
    from waiting on the current ScalarE exponential. Per-head arithmetic and PV summation order are unchanged."""
    NB = S // 128
    idf, ident, msk, ones = _attn_consts()
    pred = nl.ndarray((128, 128), dtype=nl.uint8, buffer=nl.sbuf)       # 1 where k <= q (diagonal block)
    pf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pf, data=msk, op0=nl.equal, operand0=0.0)
    nisa.tensor_copy(dst=pred, src=pf)
    kts = []
    vts = []
    qts = []
    for g in range(NKL):
        kts.append(0)
        vts.append(0)
        qts.append(0)
        qts.append(0)
    items = []
    for g in range(NKL):
        for i in range(NB):
            for hh in range(2):
                items.append((g, i, hh))
    N = len(items)
    sc_l = []
    mx_l = []
    e_l = []
    for n in range(N):
        sc_l.append(0)
        mx_l.append(0)
        e_l.append(0)
    rinv_l = []
    eT_l = []
    for n in range(N // 2):
        rinv_l.append(0)
        eT_l.append(0)

    def_loaded = []
    for g in range(NKL):
        def_loaded.append(0)
    for n in range(N + DEPTH):
        if n < N:
            g = items[n][0]
            if def_loaded[g] == 0:
                kv = pid * NKL + g
                kt = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
                _hd(dst=kt, src=kT[kv])
                kts[g] = kt
                vt = nl.ndarray((128, NB, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                _hd(dst=vt, src=v[kv].ap(pattern=[[D, 128], [128 * D, NB], [1, D]], offset=0))
                vts[g] = vt
                for hh in range(2):
                    qt = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
                    _hd(dst=qt, src=qT[pid * NQL + 2 * g + hh])
                    qts[2 * g + hh] = qt
                def_loaded[g] = 1
            _ab_s1(n, items, qts, kts, msk, pred, sc_l, mx_l, S)
        m2 = n - 1
        if m2 >= 0 and m2 < N:
            _ab_s2(m2, items, sc_l, mx_l, e_l, rinv_l, eT_l, NB)
            _ap_transpose(m2, items, e_l, eT_l, NB)
        m3 = n - DEPTH
        if m3 >= 0:
            _ap_pv(m3, items, rinv_l, eT_l, vts, idf, ones, oT, pid, NB, late_norm=True)


@nki.jit
def attn_test_256(pos, qT, kT, v, oT):
    attn_prefill(qT, kT, v, oT, nl.program_id(0), 256)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out


@nki.jit
def attn_test_1024(pos, qT, kT, v, oT):
    attn_prefill(qT, kT, v, oT, nl.program_id(0), 1024)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out


@nki.jit
def attnp_test_256(pos, qT, kT, v, oT):
    attn_prefill_pipe(qT, kT, v, oT, nl.program_id(0), 256)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out


@nki.jit
def attnp_test_1024(pos, qT, kT, v, oT):
    attn_prefill_pipe(qT, kT, v, oT, nl.program_id(0), 1024)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out


@nki.jit
def attnb_test_1024(pos, qT, kT, v, oT):
    attn_prefill_big(qT, kT, v, oT, nl.program_id(0), 1024)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out


@nki.jit
def attnb_test_4096(pos, qT, kT, v, oT):
    attn_prefill_big(qT, kT, v, oT, nl.program_id(0), 4096)
    p = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p, src=pos)
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=p)
    return out
