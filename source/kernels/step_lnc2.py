"""Whole Qwen3-0.6B decode step in one LNC=2 NEFF: both physical cores of the logical core stream half of every
weight and exchange [128, 8] fp32 partial sums through nisa.sendrecv (SBUF -> SBUF, gpsimd_dma, ~0.85 us).

Split per layer (core c = nl.program_id(0)):
  QKV     N-split by heads: q heads 8c..8c+7, kv heads 4c..4c+3 (16 of the 32 output tiles)
  attn    local: the core's 8 q heads against its 4 kv heads; reads / writes only its half of the cache
  o_proj  K-split (its 8 heads' rows) -> partial h -> exchange + add
  gate/up N-split (intermediate tiles 12c..12c+11 of gate and of up) -> local SiLU*up
  down    K-split (the same 12 intermediate tiles) -> partial -> exchange + add
  LM head N-split (vocab tiles 594c..594c+593, vocab padded to 1188 tiles), local argmax, exchange (max, id).
Everything else (embedding, RMSNorms, residual) is computed redundantly on both cores.

Inputs: like step_decode, but weights pre-split per core (see tests/test_step2.py `split_inputs`):
  wqkv [L, 2, 128, 16*8*128], wo [L, 2, 128, 8*8*128], wgu [L, 2, 128, 24*8*128], wd [L, 2, 128, 8*12*128],
  wlm [2, 128, 594*8*128]; kcache [L, 2, 4, 128, TMAX], vcache [L, 2, TMAX, 512].
Both cores write the same values to toks[pos+1] and to output_0 (= pos+1): identical bytes, no read-modify-write.
"""
import nki
import nki.language as nl
import nki.isa as nisa
from layer_decode import _consts, _gemv, _rms_col, _dma, HT, HID, D, EPS, NEG, SCALE
from step_decode import _token_in, _embed, BIGC

NQL = 8      # q heads per core
NKL = 4      # kv heads per core
ITL = 12     # intermediate tiles per core
VOCAB = 151936
NVL = 594    # vocab tiles per core (2 * 594 * 128 = 152064 >= VOCAB)


def _xchg_add(part, pid):
    """part [128, n] fp32 on each core -> sum over both cores (on both)."""
    n = part.shape[1]
    mine = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mine, src=part)
    other = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=mine, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    s = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s, data1=mine, data2=other, op=nl.add)
    return s


def _layer2(h, l, pid, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, kcache, vcache,
            preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw, kcw=None, vcw=None):
    TMAX = vcache.shape[2]
    if kcw is None:
        kcw = kcache
        vcw = vcache
    hwc = hw
    if hw >= 10:     # DMA modes (layer_decode._dma): weights hw % 10, KV-cache loads hw // 10
        hwc = hw // 10
        hw = hw % 10
        if hwc == 9:
            hwc = 0
    NT = TB // 128
    NS = NT + 1
    qi = 0
    kall = nl.ndarray((D, NKL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
    vall = nl.ndarray((128, NT, NKL * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    for hh in range(NKL):
        _dma(kall[:, hh, :], kcache[l, pid, hh, :, 0:TB], qi, hwc)
        qi += 1
    for j in range(NT):
        _dma(vall[:, j, :], vcache[l, pid, j * 128:(j + 1) * 128, :], qi, hwc)
        qi += 1
    w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1, src=ln1[l])
    w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w2, src=ln2[l])
    qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qnc, src=qn[l])
    knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=knc, src=kn[l])

    x1 = _rms_col(h, w1, HT, HID, ones, nl.bfloat16)
    qkv_ps = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(x1, wqkv[l, pid], HT, 0, 16, qkv_ps, 0, SL, hw, qi)
    qkv = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qkv, src=qkv_ps)
    NQK = NQL + NKL   # 12
    sq = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=qkv[:, 0:NQK], data2=qkv[:, 0:NQK], op=nl.multiply)
    ss = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ss, stationary=ones, moving=sq)
    rstd = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xn, data1=qkv[:, 0:NQK], data2=rstd, op=nl.multiply)
    qkn = nl.ndarray((128, NQK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qkn[:, 0:NQL], data=xn[:, 0:NQL], op0=nl.multiply, operand0=qnc)
    nisa.tensor_scalar(dst=qkn[:, NQL:NQK], data=xn[:, NQL:NQK], op0=nl.multiply, operand0=knc)
    rps = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rps, stationary=rot_sb, moving=qkn)
    xc = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xc, data=qkn, op0=nl.multiply, operand0=cs_col)
    qkT = nl.ndarray((128, NQK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=qkT, data=rps, op0=nl.multiply, operand0=sn_col, op1=nl.add, operand1=xc)
    vT = nl.ndarray((128, NKL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vT, src=qkv[:, NQK:16])
    vr_ps = nl.ndarray((1, NKL * D), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        nisa.nc_matmul(dst=vr_ps[0:1, hh * D:(hh + 1) * D], stationary=vT[:, hh:hh + 1], moving=ident)
    vrow = nl.ndarray((1, NKL * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vrow, src=vr_ps)
    nisa.dma_copy(dst=kcw[l, pid].ap(pattern=[[TMAX, D], [D * TMAX, NKL]], offset=0, scalar_offset=preg, indirect_dim=2),
                  src=qkT[:, NQL:NQK])
    nisa.dma_copy(dst=vcw[l, pid].ap(pattern=[[NKL * D, 1], [1, NKL * D]], offset=0, scalar_offset=preg, indirect_dim=0),
                  src=vrow)

    io = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[128, NT]], offset=0, channel_multiplier=1)
    madd = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=madd, data=io, op0=nl.greater_equal, operand0=pb, op1=nl.multiply, operand1=NEG)
    S = nl.ndarray((128, NS, NQL), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(NT):
        ps = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NKL):
            nisa.nc_matmul(dst=ps[:, 2 * hh:2 * hh + 2], stationary=kall[:, hh, j * 128:(j + 1) * 128],
                           moving=qkT[:, 2 * hh:2 * hh + 2])
        nisa.tensor_scalar(dst=S[:, j, :], data=ps, op0=nl.add, operand0=madd[:, j:j + 1])
    nisa.memset(dst=S[:, NT, :], value=NEG)
    psc = nl.ndarray((1, NQL), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        nisa.nc_matmul(dst=psc[0:1, 2 * hh:2 * hh + 2], stationary=qkT[:, NQL + hh:NQL + hh + 1], moving=qkT[:, 2 * hh:2 * hh + 2])
    nisa.tensor_copy(dst=S[0:1, NT, :], src=psc)
    pm = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pm, op=nl.maximum, data=S.ap(pattern=[[NS * NQL, 128], [1, NQL], [NQL, NS]], offset=0), axis=2)
    pmT = nl.ndarray((NQL, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pmT, data=pm)
    m = nl.ndarray((NQL, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=pmT, axis=1)
    dg = nl.ndarray((NQL, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dg, data=idf[0:NQL, 0:NQL], op0=nl.multiply, operand0=m)
    mb = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=mb, stationary=ones16[0:NQL, :], moving=dg)
    mbs = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mbs, src=mb)
    Sm = nl.ndarray((128, NS, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=Sm, data1=S, data2=mbs.ap(pattern=[[NQL, 128], [0, NS], [1, NQL]], offset=0), op=nl.subtract)
    E = nl.ndarray((128, NS, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=E, op=nl.exp, data=Sm, scale=SCALE)
    es = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=es, op=nl.add, data=E.ap(pattern=[[NS * NQL, 128], [1, NQL], [NQL, NS]], offset=0), axis=2)
    den = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=den, stationary=ones, moving=es)
    rden = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden, data=den)
    ops_ = nl.ndarray((D, NQL), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        for j in range(NT):
            nisa.nc_matmul(dst=ops_[:, 2 * hh:2 * hh + 2], stationary=vall[:, j, hh * D:(hh + 1) * D],
                           moving=E[:, j, 2 * hh:2 * hh + 2])
        nisa.nc_matmul(dst=ops_[:, 2 * hh:2 * hh + 2], stationary=vrow[0:1, hh * D:(hh + 1) * D],
                       moving=E[0:1, NT, 2 * hh:2 * hh + 2])
    oT = nl.ndarray((D, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=oT, data1=ops_, data2=rden, op=nl.multiply)

    y_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(oT, wo[l, pid], NQL, 0, HT, y_ps, 0, SL, hw, qi)
    y = _xchg_add(y_ps, pid)
    h2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h, data2=y, op=nl.add)

    x2 = _rms_col(h2, w2, HT, HID, ones, nl.bfloat16)
    gu_ps = nl.ndarray((128, 2 * ITL), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(x2, wgu[l, pid], HT, 0, 2 * ITL, gu_ps, 0, SL, hw, qi)
    sg = nl.ndarray((128, ITL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=gu_ps[:, 0:ITL])
    act = nl.ndarray((128, ITL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act, data1=sg, data2=gu_ps[:, ITL:2 * ITL], op=nl.multiply)
    d_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(act, wd[l, pid], ITL, 0, HT, d_ps, 0, SL, hw, qi)
    d = _xchg_add(d_ps, pid)
    h3 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h3, data1=h2, data2=d, op=nl.add)
    return h3


def _lm_argmax2(xf, wlm_c, pid, SL, hw):
    lg = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    qi = 0
    c0 = 0
    while c0 < NVL:
        n = 512
        if NVL - c0 < 512:
            n = NVL - c0
        ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        qi = _gemv(xf, wlm_c, HT, c0, c0 + n, ps, 0, SL, hw, qi)
        nisa.tensor_copy(dst=lg[:, c0:c0 + n], src=ps[:, 0:n])
        c0 += n
    # global ids of this core's logits: (pid*NVL + col)*128 + p; padded ids (>= VOCAB) masked out
    gid = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=gid, pattern=[[128, NVL]], offset=0, channel_multiplier=1)
    pf_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pf_i, pattern=[[0, 1]], offset=0, channel_multiplier=0)
    base = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=base, data=pf_i, op0=nl.add, operand0=pid * (NVL * 128))
    nisa.tensor_scalar(dst=gid, data=gid, op0=nl.add, operand0=base)
    msk = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=msk, data=gid, op0=nl.greater_equal, operand0=float(VOCAB), op1=nl.multiply, operand1=-3.0e38)
    nisa.tensor_tensor(dst=lg, data1=lg, data2=msk, op=nl.add)
    pm = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pm[:, 0:1], op=nl.maximum, data=lg, axis=1)
    nisa.tensor_scalar(dst=gid, data=gid, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGC)
    eq = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=pm[:, 0:1])
    nisa.tensor_tensor(dst=eq, data1=eq, data2=gid, op=nl.multiply)
    nisa.tensor_reduce(dst=pm[:, 1:2], op=nl.maximum, data=eq, axis=1)
    rows = nl.ndarray((1, 2, 128), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(2):
        tp = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=pm[:, c:c + 1])
        nisa.tensor_copy(dst=rows[:, c, :], src=tp)
    loc = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)      # (max, BIGC - lowest id) of this core
    nisa.tensor_reduce(dst=loc[:, 0:1], op=nl.maximum, data=rows[:, 0, :], axis=1)
    e2 = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e2, data=rows[:, 0, :], op0=nl.equal, operand0=loc[:, 0:1])
    nisa.tensor_tensor(dst=e2, data1=e2, data2=rows[:, 1, :], op=nl.multiply)
    nisa.tensor_reduce(dst=loc[:, 1:2], op=nl.maximum, data=e2, axis=1)
    oth = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=loc, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    both = nl.ndarray((1, 2, 2), dtype=nl.float32, buffer=nl.sbuf)   # [., core-ish, (max, score)]
    nisa.tensor_copy(dst=both[:, 0, :], src=loc)
    nisa.tensor_copy(dst=both[:, 1, :], src=oth)
    gm = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gm, data1=loc[:, 0:1], data2=oth[:, 0:1], op=nl.maximum)
    e3 = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e3, data=both[:, :, 0], op0=nl.equal, operand0=gm)
    nisa.tensor_tensor(dst=e3, data1=e3, data2=both[:, :, 1], op=nl.multiply)
    bs = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=bs, op=nl.maximum, data=e3, axis=1)
    tokf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tokf, data=bs, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGC)
    return tokf


def _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
           kcache, vcache, TB, SL, hw, kcw=None, vcw=None):
    L = wqkv.shape[0]
    pid = nl.program_id(0)
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb_i, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pb, src=pb_i)
    cs_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    sn_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for half in range(2):
        nisa.dma_copy(dst=cs_col[half * 64:(half + 1) * 64, :],
                      src=cos.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
        nisa.dma_copy(dst=sn_col[half * 64:(half + 1) * 64, :],
                      src=sin.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    ones, ident, idf, ones16 = _consts()
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    t_sb = _token_in(pos, plen, prompt, toks, p_sb)
    h = _embed(embed, t_sb)
    for l in range(L):
        h = _layer2(h, l, pid, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, kcache, vcache,
                    preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw, kcw, vcw)
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(h, wf, HT, HID, ones, nl.bfloat16)
    hwl = hw % 10
    tokf = _lm_argmax2(xf, wlm[pid], pid, SL, hwl)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok_i)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=p1)
    return pos_out


def _tail2(plen, toks, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, kcw, vcw,
           SL, hw):
    """The prefill's last layer for the last prompt token only, then the LM head and the argmax: the decode layer
    (_layer2) at pos = plen - 1, with the residual taken from the prefill's s_h [128, 8, S] (column plen - 1, already
    the decode column layout) and attention over the prompt bucket S (keys [0, plen - 1) from the cache, which the
    prefill's pf_kv wrote, plus the current token). Writes toks[plen] and returns plen."""
    L = wqkv.shape[0]
    S = s_h.shape[2]
    pid = nl.program_id(0)
    pl_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl_sb, src=plen)
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p_sb, data=pl_sb, op0=nl.add, operand0=-1)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb_i, src=plen.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pb, src=pb_i)
    nisa.tensor_scalar(dst=pb, data=pb, op0=nl.add, operand0=-1.0)
    cs_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    sn_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for half in range(2):
        nisa.dma_copy(dst=cs_col[half * 64:(half + 1) * 64, :],
                      src=cos.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
        nisa.dma_copy(dst=sn_col[half * 64:(half + 1) * 64, :],
                      src=sin.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    ones, ident, idf, ones16 = _consts()
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    h = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=h, src=s_h.ap(pattern=[[HT * S, 128], [S, HT]], offset=0, scalar_offset=preg, indirect_dim=2))
    h = _layer2(h, L - 1, pid, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, kcache, vcache,
                preg, pb, cs_col, sn_col, ones, ident, idf, ones16, S, SL, hw, kcw, vcw)
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(h, wf, HT, HID, ones, nl.bfloat16)
    tokf = _lm_argmax2(xf, wlm[pid], pid, SL, hw % 10)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok_i)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=pl_sb)
    return pos_out


@nki.jit
def pf_tail(plen, toks, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, kcw, vcw):
    S = s_h.shape[2]
    if S >= 3072:
        return _tail2(plen, toks, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache,
                      kcw, vcw, 4, 62)
    return _tail2(plen, toks, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache,
                  kcw, vcw, 8, 2)


# DMA modes per bucket (layer_decode._dma): weights and the LM head on the Sync engine's hardware DGE queue (an
# in-order queue that runs nothing else, so weight prefetch never waits behind dependent compute); KV-cache loads on
# the same queue for small buckets, pre-generated descriptors (dge_mode.none) for large ones; weight slabs of 8 output
# tiles (2 MB per DMA) for small buckets, 4 for large ones (session 002, measured).
# kcw / vcw: the same device buffers as kcache / vcache (the host binds both names to one tensor); the cache writes
# go through these names so the compiler sees no dependency between layer l's write and layer l+1's load.
STEP_BUCKETS = [384, 512, 1024, 1152, 1536, 2048, 3072, 4224, 4608]


@nki.jit
def step2_b384(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 384, 8, 2, kcw, vcw)


@nki.jit
def step2_b512(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 512, 8, 2, kcw, vcw)


@nki.jit
def step2_b1024(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 1024, 8, 2, kcw, vcw)


@nki.jit
def step2_b1152(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 1152, 8, 2, kcw, vcw)


@nki.jit
def step2_b1536(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 1536, 8, 2, kcw, vcw)


@nki.jit
def step2_b2048(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 2048, 8, 2, kcw, vcw)


@nki.jit
def step2_b3072(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 3072, 4, 62, kcw, vcw)


@nki.jit
def step2_b4224(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 4224, 4, 62, kcw, vcw)


@nki.jit
def step2_b4608(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
               kcache, vcache, kcw, vcw):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
                  kcache, vcache, 4608, 4, 62, kcw, vcw)
