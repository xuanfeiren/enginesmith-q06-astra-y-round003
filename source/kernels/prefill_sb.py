"""One Qwen3-0.6B prefill layer for short prompt buckets (S <= 256) with every activation resident in SBUF, split
over the two physical cores like the decode step (step_lnc2.py) instead of the phase / barrier structure of
prefill_lnc2.py:

  both cores hold the full residual h^T [128, 8, S] fp32 (identical copies)
  A  norm1 + QKV N-split (q heads 8c.., kv heads 4c.., the decode wqkv layout) + q/k norm + RoPE; q^T, k^T, V stay
     in SBUF (K / V also written to the cache)
  B  causal attention on the core's own heads (attn_prefill_sb, SBUF in / out)
  C  o_proj K-split over the core's 8 heads (decode wo layout) -> partial [128, 8, S] -> exchanged with the other
     core (nisa.sendrecv) -> h2 = h + (mine + other)   (fp32 addition is commutative: both cores get the same bytes)
  D  norm2 + gate/up N-split (decode wgu layout) + SiLU * up -> act [128, 12, S] (the core's intermediate tiles)
  E  down K-split over those tiles (decode wd layout) -> partial -> exchange -> h3 = h2 + (mine + other)
The residual goes from s_h and back to s_h (both cores write identical bytes) between layer NEFFs. Two cross-core
sync points per layer (the exchanges), no HBM round trips of q / k / v, attention output or MLP activations.
"""
import nki
import nki.language as nl
import nki.isa as nisa
from layer_decode import _consts, HT, HID, D, EPS
from attn_prefill import attn_prefill_sb

NQL, NKL, ITL = 8, 4, 12
import os as _os
APD = int(_os.environ.get('SB_APD', '3'))
XCG = int(_os.environ.get('SB_XCG', '0'))
VCA = int(_os.environ.get('SB_VCA', '0'))
SBC_CH = int(_os.environ.get('SB_CH', '512'))   # chunk of pf_sbc (1024 bucket: 528 us at 256, 502 at 512)
WQS = int(_os.environ.get('SB_WQS', '1'))   # weight queues: 0 GpSimd SWDGE, 1 Sync HWDGE, 2 Sync (wq, wo) + Scalar (wgu, wd)   # phase A: QKV tiles issued ahead of the epilogues


def _hd(dst, src):
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def _hs(dst, src):
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)


def _norm_sb(h, w_sb, onesb, S):
    """RMSNorm over the 1024 features of h [128, 8, S] fp32 -> bf16 (squares on ScalarE in bf16, sums on the PE, the
    same arithmetic as prefill_lnc2._norm_chunk with FAST and XG = 3)."""
    sq = nl.ndarray((128, HT, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=h)
    ss = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / HID, bias=EPS)
    xo = nl.ndarray((128, HT, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in range(HT):
        nisa.scalar_tensor_tensor(dst=xo[:, kt, :], data=h[:, kt, :], op0=nl.multiply, operand0=w_sb[:, kt:kt + 1],
                                  op1=nl.multiply, operand1=rstd)
    return xo


def _load_w(w, n, eng):
    t = nl.ndarray((128, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    q = n // 4
    for i in range(4):
        if eng == 0:
            nisa.dma_copy(dst=t[:, i * q:(i + 1) * q], src=w[:, i * q:(i + 1) * q])
        elif eng == 2:
            _hd(dst=t[:, i * q:(i + 1) * q], src=w[:, i * q:(i + 1) * q])
        else:
            _hs(dst=t[:, i * q:(i + 1) * q], src=w[:, i * q:(i + 1) * q])
    return t


def _xchg_add(part, pid, S):
    """part [128, 8, S] fp32 on each core -> mine + other (identical on both cores)."""
    other = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=part, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.dma)
    s = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s, data1=part, data2=other, op=nl.add)
    return s


def _xchg_into(dst, part, pid):
    """Exchange a completed projection tile while later independent tiles compute."""
    other = nl.ndarray(part.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=part, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.dma)
    nisa.tensor_tensor(dst=dst, data1=part, data2=other, op=nl.add)


def _sa_m(nt, x1, wq, S, ps_l):
    ps = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ps, stationary=wq[:, (nt * HT + kt) * 128:(nt * HT + kt + 1) * 128], moving=x1[:, kt, :])
    ps_l[nt] = ps


def _sa_e1(nt, ps_l, yb_l, onesb, ident, qnc, knc, vsb, vcache, pid, S, t0=0):
    ps = ps_l[nt]
    if nt < NQL + NKL:
        sq = nl.ndarray((128, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=ps)
        ss = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq)
        rs = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=EPS)
        yb = nl.ndarray((128, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        if nt < NQL:
            nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=qnc, op1=nl.multiply, operand1=rs)
        else:
            nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=knc, op1=nl.multiply, operand1=rs)
        yb_l[nt] = yb
    else:
        j = nt - NQL - NKL
        NB = S // 128
        vb = nl.ndarray((128, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=vb, op=nl.copy, data=ps)
        for tb in range(NB):
            tp = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=tp, stationary=vb[:, tb * 128:(tb + 1) * 128], moving=ident)
            b = t0 // 128 + tb
            if VCA:
                nisa.activation(dst=vsb[:, b, j, :], op=nl.copy, data=tp)
            else:
                nisa.tensor_copy(dst=vsb[:, b, j, :], src=tp)
            _hd(dst=vcache[0, pid, b * 128:(b + 1) * 128, j * D:(j + 1) * D], src=vsb[:, b, j, :])


def _sa_e2(nt, yb_l, rot_sb, cs, sn, qsb, ksb, kcache, pid, S, t0=0):
    if nt >= NQL + NKL:
        return
    yb = yb_l[nt]
    rp = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=yb)
    xc = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    if XCG:
        nisa.tensor_tensor(dst=xc, data1=yb, data2=cs, op=nl.multiply, engine=nisa.engine.gpsimd)
    else:
        nisa.tensor_tensor(dst=xc, data1=yb, data2=cs, op=nl.multiply)
    xs = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xs, data1=rp, data2=sn, op=nl.multiply)
    if nt < NQL:
        nisa.tensor_tensor(dst=qsb[:, nt, t0:t0 + S], data1=xc, data2=xs, op=nl.add, engine=nisa.engine.gpsimd)
    else:
        j = nt - NQL
        nisa.tensor_tensor(dst=ksb[:, j, t0:t0 + S], data1=xc, data2=xs, op=nl.add, engine=nisa.engine.gpsimd)
        _hd(dst=kcache[0, pid, j, :, t0:t0 + S], src=ksb[:, j, t0:t0 + S])


def _layer_sb(plen, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cosT, sinT, kcache, vcache):
    S = s_h.shape[2]
    NB = S // 128
    l = 0
    pid = nl.program_id(0)
    ones, ident, idf, ones16 = _consts()
    onesb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onesb, value=1.0)
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=w1, src=ln1[l])
    w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=w2, src=ln2[l])
    qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=qnc, src=qn[l])
    knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=knc, src=kn[l])
    h = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=h, src=s_h)
    e1 = 0
    e2 = 0
    if WQS == 1:
        e1 = 2
        e2 = 2
    if WQS == 2:
        e1 = 2
        e2 = 1
    wq = _load_w(wqkv[l, pid], 16 * HT * 128, e1)
    cs = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=cs, src=cosT[:, 0:S])
    sn = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=sn, src=sinT[:, 0:S])
    wo_sb = _load_w(wo[l, pid], HT * NQL * 128, e1)
    wg = _load_w(wgu[l, pid], 2 * ITL * HT * 128, e2)
    wd_sb = _load_w(wd[l, pid], HT * ITL * 128, e2)

    # ================= A
    x1 = _norm_sb(h, w1, onesb, S)
    qsb = nl.ndarray((128, NQL, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    ksb = nl.ndarray((128, NKL, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    vsb = nl.ndarray((128, NB, NKL, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    ps_l = []
    yb_l = []
    for nt in range(16):
        ps_l.append(0)
        yb_l.append(0)
    for nt in range(APD):
        _sa_m(nt, x1, wq, S, ps_l)
    for nt in range(16):
        if nt + APD < 16:
            _sa_m(nt + APD, x1, wq, S, ps_l)
        _sa_e1(nt, ps_l, yb_l, onesb, ident, qnc, knc, vsb, vcache, pid, S)
        if nt >= 1:
            _sa_e2(nt - 1, yb_l, rot_sb, cs, sn, qsb, ksb, kcache, pid, S)
    _sa_e2(15, yb_l, rot_sb, cs, sn, qsb, ksb, kcache, pid, S)

    # ================= B
    osb = nl.ndarray((128, NQL, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    attn_prefill_sb(qsb, ksb, vsb, osb, S)

    # ================= C: o_proj K-split (this core's 8 heads) + exchange + residual
    part = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.sbuf)
    for nt in range(HT):
        ps = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
        for kt in range(NQL):
            nisa.nc_matmul(dst=ps, stationary=wo_sb[:, (nt * NQL + kt) * 128:(nt * NQL + kt + 1) * 128],
                           moving=osb[:, kt, :])
        nisa.tensor_copy(dst=part[:, nt, :], src=ps)
    sc = _xchg_add(part, pid, S)
    h2 = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h, data2=sc, op=nl.add)

    # ================= D: norm2 + gate/up (this core's 12 intermediate tiles) + SiLU * up
    x2 = _norm_sb(h2, w2, onesb, S)
    act = nl.ndarray((128, ITL, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(ITL):
        pg = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=pg, stationary=wg[:, (i * HT + kt) * 128:(i * HT + kt + 1) * 128], moving=x2[:, kt, :])
        pu = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=pu, stationary=wg[:, ((ITL + i) * HT + kt) * 128:((ITL + i) * HT + kt + 1) * 128],
                           moving=x2[:, kt, :])
        sg = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.silu, data=pg)
        nisa.tensor_tensor(dst=act[:, i, :], data1=sg, data2=pu, op=nl.multiply)

    # ================= E: down K-split + exchange + residual
    part2 = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.sbuf)
    for nt in range(HT):
        ps = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.psum)
        for kt in range(ITL):
            nisa.nc_matmul(dst=ps, stationary=wd_sb[:, (nt * ITL + kt) * 128:(nt * ITL + kt + 1) * 128],
                           moving=act[:, kt, :])
        nisa.tensor_copy(dst=part2[:, nt, :], src=ps)
    sd = _xchg_add(part2, pid, S)
    h3 = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h3, data1=h2, data2=sd, op=nl.add)
    _hd(dst=s_h, src=h3)
    pl_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl_sb, src=plen)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=pl_sb)
    return pos_out


def _layer_sbc(plen, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cosT, sinT, kcache, vcache, CH):
    """_layer_sb for longer buckets: the same core split and SBUF residency, every phase except attention walked in
    token chunks of CH (one residual tensor per chunk, so chunks pipeline independently); weights loaded one phase
    ahead in program order (SBUF)."""
    S = s_h.shape[2]
    NB = S // 128
    NC = S // CH
    l = 0
    pid = nl.program_id(0)
    ones, ident, idf, ones16 = _consts()
    onesb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onesb, value=1.0)
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=w1, src=ln1[l])
    w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=w2, src=ln2[l])
    qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=qnc, src=qn[l])
    knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=knc, src=kn[l])
    wq = _load_w(wqkv[l, pid], 16 * HT * 128, 2)
    hs = []
    for c in range(NC):
        hc = nl.ndarray((128, HT, CH), dtype=nl.float32, buffer=nl.sbuf)
        _hd(dst=hc, src=s_h[:, :, c * CH:(c + 1) * CH])
        hs.append(hc)
    cs = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=cs, src=cosT[:, 0:S])
    sn = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    _hd(dst=sn, src=sinT[:, 0:S])

    # ================= A
    qsb = nl.ndarray((128, NQL, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    ksb = nl.ndarray((128, NKL, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    vsb = nl.ndarray((128, NB, NKL, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in range(NC):
        t0 = c * CH
        x1 = _norm_sb(hs[c], w1, onesb, CH)
        ps_l = []
        yb_l = []
        for nt in range(16):
            ps_l.append(0)
            yb_l.append(0)
        for nt in range(APD):
            _sa_m(nt, x1, wq, CH, ps_l)
        for nt in range(16):
            if nt + APD < 16:
                _sa_m(nt + APD, x1, wq, CH, ps_l)
            _sa_e1(nt, ps_l, yb_l, onesb, ident, qnc, knc, vsb, vcache, pid, CH, t0)
            if nt >= 1:
                _sa_e2(nt - 1, yb_l, rot_sb, cs[:, t0:t0 + CH], sn[:, t0:t0 + CH], qsb, ksb, kcache, pid, CH, t0)
        _sa_e2(15, yb_l, rot_sb, cs[:, t0:t0 + CH], sn[:, t0:t0 + CH], qsb, ksb, kcache, pid, CH, t0)
    wo_sb = _load_w(wo[l, pid], HT * NQL * 128, 2)

    # ================= B
    osb = nl.ndarray((128, NQL, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    attn_prefill_sb(qsb, ksb, vsb, osb, S)
    wg = _load_w(wgu[l, pid], 2 * ITL * HT * 128, 2)
    wd_sb = _load_w(wd[l, pid], HT * ITL * 128, 2)

    # ================= C
    h2s = []
    for c in range(NC):
        t0 = c * CH
        part = nl.ndarray((128, HT, CH), dtype=nl.float32, buffer=nl.sbuf)
        sc = nl.ndarray((128, HT, CH), dtype=nl.float32, buffer=nl.sbuf)
        for nt in range(HT):
            ps = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
            for kt in range(NQL):
                nisa.nc_matmul(dst=ps, stationary=wo_sb[:, (nt * NQL + kt) * 128:(nt * NQL + kt + 1) * 128],
                               moving=osb[:, kt, t0:t0 + CH])
            nisa.tensor_copy(dst=part[:, nt, :], src=ps)
            # Issue this tile's exchange before the remaining projection matmuls.
            _xchg_into(sc[:, nt:nt + 1, :], part[:, nt:nt + 1, :], pid)
        h2 = nl.ndarray((128, HT, CH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=h2, data1=hs[c], data2=sc, op=nl.add)
        h2s.append(h2)

    # ================= D + E per chunk
    for c in range(NC):
        t0 = c * CH
        x2 = _norm_sb(h2s[c], w2, onesb, CH)
        act = nl.ndarray((128, ITL, CH), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(ITL):
            pg = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
            for kt in range(HT):
                nisa.nc_matmul(dst=pg, stationary=wg[:, (i * HT + kt) * 128:(i * HT + kt + 1) * 128],
                               moving=x2[:, kt, :])
            pu = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
            for kt in range(HT):
                nisa.nc_matmul(dst=pu, stationary=wg[:, ((ITL + i) * HT + kt) * 128:((ITL + i) * HT + kt + 1) * 128],
                               moving=x2[:, kt, :])
            sg = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sg, op=nl.silu, data=pg)
            nisa.tensor_tensor(dst=act[:, i, :], data1=sg, data2=pu, op=nl.multiply)
        part2 = nl.ndarray((128, HT, CH), dtype=nl.float32, buffer=nl.sbuf)
        sd = nl.ndarray((128, HT, CH), dtype=nl.float32, buffer=nl.sbuf)
        for nt in range(HT):
            ps = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
            for kt in range(ITL):
                nisa.nc_matmul(dst=ps, stationary=wd_sb[:, (nt * ITL + kt) * 128:(nt * ITL + kt + 1) * 128],
                               moving=act[:, kt, :])
            nisa.tensor_copy(dst=part2[:, nt, :], src=ps)
            # Issue this tile's exchange before the remaining projection matmuls.
            _xchg_into(sd[:, nt:nt + 1, :], part2[:, nt:nt + 1, :], pid)
        h3 = nl.ndarray((128, HT, CH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=h3, data1=h2s[c], data2=sd, op=nl.add)
        _hd(dst=s_h[:, :, t0:t0 + CH], src=h3)
    pl_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl_sb, src=plen)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=pl_sb)
    return pos_out


@nki.jit
def pf_sb(plen, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cosT, sinT, kcache, vcache):
    """one layer (inputs are the per-layer slices of the decode-layout weights and caches)"""
    return _layer_sb(plen, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cosT, sinT, kcache, vcache)


@nki.jit
def pf_sbc(plen, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cosT, sinT, kcache, vcache):
    """one layer, chunked SBUF-resident variant (buckets > 256)"""
    return _layer_sbc(plen, s_h, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cosT, sinT, kcache, vcache, SBC_CH)
