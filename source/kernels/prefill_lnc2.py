"""Whole Qwen3-0.6B prefill in one LNC=2 NEFF: S prompt tokens (bucket, padded) -> 28 layers -> KV cache written in
the decode layout of step_lnc2.py -> LM head on the last real token -> argmax = first generated token.

Activations are feature-major ("transposed"): h^T [128, 8, S] fp32 lives in an internal shared-HBM tensor; every
phase walks the tokens in chunks of C = min(512, S) with the layer's weights for this core resident in SBUF.
GEMMs: stationary = weight tile [128 k, 128 n], moving = activation chunk [128 k, C tokens] -> PSUM [128 n, C].
Per layer (core c = program 0/1), with nisa.core_barrier between the phases:
  A  norm1 + QKV (q heads 8c..8c+7, kv heads 4c..4c+3) + q/k RMSNorm + RoPE; q^T -> qT_s, k^T -> kT_s and kcache,
     V transposed on the PE -> v_s and vcache
  B  causal attention over the 16 heads: nkilib _attention_cte (Apache-2.0, neuronx-cc 2.27 wheel), q/k/v from
     HBM in our layouts (tp_q=False, tp_k=False, tp_out=True), its own LNC=2 split of the heads
  C  o_proj, N-split (output tiles 4c..4c+3, all 16 heads as K) + residual, in place into h
  D  norm2 + gate/up (intermediate tiles 12c..12c+11) + SiLU*up -> act_s
  E  down, N-split (output tiles 4c..4c+3, all 24 intermediate tiles as K) + residual, in place into h
Inputs: ids [S, 1] int32, plen [1, 1] int32, toks, embed, ln1, ln2, qn, kn, wqkv (decode split), wo_p [L, 2, 128,
4*16*128], wgu (decode split), wd_p [L, 2, 128, 4*24*128], normf, wlm (decode split), rot, cosT / sinT [128, TMAX]
fp32 (cosT[d, t] = cos(t * f[d % 64])), kcache, vcache (decode layouts, written in place).
Output: output_0 = plen [1, 1] int32 (the next decode position); toks[plen] = first generated token.
"""
import nki
import nki.language as nl
import nki.isa as nisa
from layer_decode import _consts, _rms_col, HT, HID, D, EPS
from step_lnc2 import _lm_argmax2
from nkilib.core.attention.attention_cte import _attention_cte
from attn_prefill import attn_prefill_pipe, attn_prefill_big

SCALE = 0.08838834764831845
NQL, NKL, ITL = 8, 4, 12
import os as _os
ATTN = int(_os.environ.get('PF_ATTN', '1'))
NLAY = int(_os.environ.get('PF_NLAY', '-1'))
DBG = int(_os.environ.get('PF_DBG', '0'))
FAST = int(_os.environ.get('PF_FAST', '1'))
HWDGE = int(_os.environ.get('PF_HWDGE', '1'))
FUSE = int(_os.environ.get('PF_FUSE', '1'))   # RMSNorm: scale / eps fused into the ScalarE rsqrt
XG = int(_os.environ.get('PF_XG', '3'))       # 3: every RMSNorm rsqrt on ScalarE (LUT, max rel err 4.4e-5) instead of GpSimd
PREF = int(_os.environ.get('PF_PREF', '1'))   # 1: each phase's weights are loaded one phase ahead (PWQ queue)
PWQ = int(_os.environ.get('PF_PWQ', '3'))
APIPE = int(_os.environ.get('PF_APIPE', '1024'))   # buckets >= this: phase A tiles software-pipelined
PA_D = int(_os.environ.get('PF_PA_D', '3'))   # tiles of QKV matmuls issued ahead of the epilogues
NOBAR = int(_os.environ.get('PF_NOBAR', '0'))   # 1: no barrier before attn_prefill (measured: no gain)
HOIST = int(_os.environ.get('PF_HOIST', '256'))   # buckets up to this size load all of a layer's weights at its top
LMHW = int(_os.environ.get('PF_LMHW', '2'))     # LM head weight stream: 2 = sync HWDGE queue, 0 = default
MYATTN = int(_os.environ.get('PF_MYATTN', '4096'))  # prompt buckets up to this size use attn_prefill.py


def _norm_chunk(hc, w_sb, ones, C, onesb=None):
    """RMSNorm over the 1024 features of a chunk hc [128, 8, C] fp32 -> bf16 [128, 8, C].
    FAST (onesb given): squares on ScalarE in bf16, sums on the PE (fp32 PSUM), one fused VectorE pass."""
    if onesb is None:
        sq = nl.ndarray((128, HT, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=hc, data2=hc, op=nl.multiply)
        ss = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=ss, stationary=ones, moving=sq[:, kt, :])
    else:
        sq = nl.ndarray((128, HT, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=hc)
        ss = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
    if XG and FUSE:   # rsqrt(ss / n + eps) in one ScalarE instruction (PSUM in)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / HID, bias=EPS)
    elif XG == 3:
        nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / HID, op1=nl.add, operand1=EPS)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=rstd)
    else:
        nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / HID, op1=nl.add, operand1=EPS)
        nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xo = nl.ndarray((128, HT, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    if onesb is None:
        xn = nl.ndarray((128, HT, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xn, data1=hc, data2=rstd.ap(pattern=[[C, 128], [0, HT], [1, C]], offset=0), op=nl.multiply)
        for kt in range(HT):
            nisa.tensor_scalar(dst=xo[:, kt, :], data=xn[:, kt, :], op0=nl.multiply, operand0=w_sb[:, kt:kt + 1])
    else:
        for kt in range(HT):
            nisa.scalar_tensor_tensor(dst=xo[:, kt, :], data=hc[:, kt, :], op0=nl.multiply, operand0=w_sb[:, kt:kt + 1],
                                      op1=nl.multiply, operand1=rstd)
    return xo


def _hd(dst, src):
    """DMA on a hardware DGE queue triggered from the Sync engine (default DMAs are software DGE: ~0.65 us of
    GpSimd time each, which made phase A GpSimd-bound)."""
    if HWDGE:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    else:
        nisa.dma_copy(dst=dst, src=src)


WQ = int(_os.environ.get('PF_WQ', '0'))   # hoisted weight-load queue: 0 default (GpSimd SWDGE), 2 sync HWDGE, 3 scalar HWDGE


def _load_w(w, n, mode=0):
    t = nl.ndarray((128, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    q = n // 4
    for i in range(4):
        if mode == 2:
            nisa.dma_copy(dst=t[:, i * q:(i + 1) * q], src=w[:, i * q:(i + 1) * q], dge_mode=nisa.dge_mode.hwdge,
                          engine=nisa.engine.sync)
        elif mode == 3:
            nisa.dma_copy(dst=t[:, i * q:(i + 1) * q], src=w[:, i * q:(i + 1) * q], dge_mode=nisa.dge_mode.hwdge,
                          engine=nisa.engine.scalar)
        else:
            nisa.dma_copy(dst=t[:, i * q:(i + 1) * q], src=w[:, i * q:(i + 1) * q])
    return t


def _pa_m(nt, x1, wq, C, ps_l):
    ps = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ps, stationary=wq[:, (nt * HT + kt) * 128:(nt * HT + kt + 1) * 128], moving=x1[:, kt, :])
    ps_l[nt] = ps


def _pa_e1(nt, ps_l, yb_l, onesb, ident, qnc, knc, v_s, vcache, l, pid, t0, C):
    ps = ps_l[nt]
    if nt < NQL + NKL:
        sq = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=ps)
        ss = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq)
        rs = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
        if XG and FUSE:   # rsqrt(ss / n + eps) in one ScalarE instruction (PSUM in)
            nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=EPS)
        elif XG:
            nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
            nisa.activation(dst=rs, op=nl.rsqrt, data=rs)
        else:
            nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
            nisa.tensor_scalar(dst=rs, data=rs, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
        yb = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        if nt < NQL:
            nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=qnc, op1=nl.multiply, operand1=rs)
        else:
            nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=knc, op1=nl.multiply, operand1=rs)
        yb_l[nt] = yb
    else:
        j = nt - NQL - NKL
        vb = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=vb, op=nl.copy, data=ps)
        vr = nl.ndarray((128, C // 128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for tb in range(C // 128):
            tp = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=tp, stationary=vb[:, tb * 128:(tb + 1) * 128], moving=ident)
            nisa.tensor_copy(dst=vr[:, tb, :], src=tp)
        for tb in range(C // 128):
            _hd(dst=v_s[pid * NKL + j, nl.ds(t0 + tb * 128, 128), :], src=vr[:, tb, :])
            _hd(dst=vcache[l, pid, nl.ds(t0 + tb * 128, 128), j * D:(j + 1) * D], src=vr[:, tb, :])


def _pa_e2(nt, yb_l, rot_sb, cs, sn, qT_s, kT_s, kcache, l, pid, t0, C):
    if nt >= NQL + NKL:
        return
    yb = yb_l[nt]
    rp = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=yb)
    xc = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xc, data1=yb, data2=cs, op=nl.multiply)
    xs = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xs, data1=rp, data2=sn, op=nl.multiply)
    ro = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ro, data1=xc, data2=xs, op=nl.add, engine=nisa.engine.gpsimd)
    if nt < NQL:
        _hd(dst=qT_s[pid * NQL + nt, :, nl.ds(t0, C)], src=ro)
    else:
        j = nt - NQL
        _hd(dst=kT_s[pid * NKL + j, :, nl.ds(t0, C)], src=ro)
        _hd(dst=kcache[l, pid, j, :, nl.ds(t0, C)], src=ro)


def _phase_a_pipe(x1, wq, onesb, rot_sb, ident, qnc, knc, cs, sn, qT_s, kT_s, v_s, kcache, vcache, l, pid, t0, C,
                  NT0):
    """Phase A's 16 output tiles of one chunk, software-pipelined in program order: the QKV matmuls of tile nt + 2,
    the norm part of tile nt's epilogue and the RoPE part of tile nt - 1 are interleaved, so the PE's in-order
    stream never waits on an epilogue (same arithmetic as the unpipelined loop)."""
    ps_l = []
    yb_l = []
    for nt in range(16):
        ps_l.append(0)
        yb_l.append(0)
    for nt in range(NT0, min(NT0 + PA_D, 16)):
        _pa_m(nt, x1, wq, C, ps_l)
    for nt in range(NT0, 16):
        if nt + PA_D < 16:
            _pa_m(nt + PA_D, x1, wq, C, ps_l)
        _pa_e1(nt, ps_l, yb_l, onesb, ident, qnc, knc, v_s, vcache, l, pid, t0, C)
        if nt - 1 >= NT0:
            _pa_e2(nt - 1, yb_l, rot_sb, cs, sn, qT_s, kT_s, kcache, l, pid, t0, C)
    _pa_e2(15, yb_l, rot_sb, cs, sn, qT_s, kT_s, kcache, l, pid, t0, C)


def _prefill(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT,
             kcache, vcache, sh, sq_, sk, sv, so, sa, SCR, STAGE=0):
    """STAGE 4: phase A of one layer for the k / v tiles only (KV cache writes).
    STAGE 0: everything; 1: embedding only; 2: the layers only (for a 1-layer NEFF run once per layer on
    per-layer tensor slices); 3: final norm + LM head + argmax only. Every stage returns output_0 = plen."""
    S = ids.shape[0]
    L = wqkv.shape[0]
    pid = nl.program_id(0)
    C = 512
    if S < 512:
        C = S
    NC = S // C
    NB = S // 128
    ones, ident, idf, ones16 = _consts()
    onesb = None
    if FAST:
        onesb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=onesb, value=1.0)
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)

    if SCR == 0:
        h = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.shared_hbm)
        qT_s = nl.ndarray((16, D, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        kT_s = nl.ndarray((8, D, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        v_s = nl.ndarray((8, S, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        oT_s = nl.ndarray((16, D, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        act_s = nl.ndarray((2 * ITL, 128, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    else:
        h = sh
        qT_s = sq_
        kT_s = sk
        v_s = sv
        oT_s = so
        act_s = sa

    pl_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl_sb, src=plen)
    NT0 = 0
    if STAGE == 4:
        NT0 = NQL       # k and v output tiles only
    if STAGE == 2 or STAGE == 3 or STAGE == 4:
        NB = 0
    if NLAY >= 0:
        L = NLAY
    if STAGE == 1 or STAGE == 3:
        L = 0
    # ---- embedding: core c gathers token blocks c, c+2, ... (128 rows per indirect DMA), PE-transposes them
    for bi in range(NB // 2):
        b = 2 * bi + pid
        idx = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=ids[nl.ds(b * 128, 128), :])
        er = nl.ndarray((128, HID), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=er, src=embed.ap(pattern=[[HID, 128], [1, HID]], offset=0, vector_offset=idx, indirect_dim=0))
        et = nl.ndarray((128, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
        for kt in range(HT):
            tp = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=tp, stationary=er[:, kt * 128:(kt + 1) * 128], moving=ident)
            nisa.tensor_copy(dst=et[:, kt, :], src=tp)
        nisa.dma_copy(dst=h[:, :, nl.ds(b * 128, 128)], src=et)
    if NB > 0:
        nisa.core_barrier(data=h, cores=(0, 1))

    for l in range(L):
        TMAX = vcache.shape[2]
        # ================= A: norm1 + QKV + q/k norm + RoPE + cache writes
        w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        _hd(dst=w1, src=ln1[l])
        qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        _hd(dst=qnc, src=qn[l])
        knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        _hd(dst=knc, src=kn[l])
        wq = _load_w(wqkv[l, pid], 16 * HT * 128)
        if HOIST >= S and STAGE != 4:
            wo_sb = _load_w(wo_p[l, pid], 4 * 16 * 128, WQ)
            wg = _load_w(wgu[l, pid], 2 * ITL * HT * 128, WQ)
            wd_sb = _load_w(wd_p[l, pid], 4 * 24 * 128, WQ)
        for c in range(NC):
            t0 = c * C
            hc = nl.ndarray((128, HT, C), dtype=nl.float32, buffer=nl.sbuf)
            _hd(dst=hc, src=h[:, :, nl.ds(t0, C)])
            if DBG == 2 and l == 0 and c == 0:
                hseen = nl.ndarray((2, 128, HT, C), dtype=nl.float32, buffer=nl.shared_hbm)
                _hd(dst=hseen[pid], src=hc)
            x1 = _norm_chunk(hc, w1, ones, C, onesb)
            cs = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
            _hd(dst=cs, src=cosT[:, nl.ds(t0, C)])
            sn = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
            _hd(dst=sn, src=sinT[:, nl.ds(t0, C)])
            if APIPE and S >= APIPE:
                _phase_a_pipe(x1, wq, onesb, rot_sb, ident, qnc, knc, cs, sn, qT_s, kT_s, v_s, kcache, vcache,
                              l, pid, t0, C, NT0)
                continue_tiles = 0
            else:
                continue_tiles = 1
            for nt in range(NT0, 16 * continue_tiles):
                ps = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                for kt in range(HT):
                    nisa.nc_matmul(dst=ps, stationary=wq[:, (nt * HT + kt) * 128:(nt * HT + kt + 1) * 128],
                                   moving=x1[:, kt, :])
                if nt < NQL + NKL:
                    if FAST:
                        sq = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.activation(dst=sq, op=nl.square, data=ps)
                        ss = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq)
                        rs = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        if XG and FUSE:   # rsqrt(ss / n + eps) in one ScalarE instruction (PSUM in)
                            nisa.activation(dst=rs, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=EPS)
                        elif XG:
                            nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
                            nisa.activation(dst=rs, op=nl.rsqrt, data=rs)
                        else:
                            nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
                            nisa.tensor_scalar(dst=rs, data=rs, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
                        yb = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                        if nt < NQL:
                            nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=qnc, op1=nl.multiply, operand1=rs)
                        else:
                            nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=knc, op1=nl.multiply, operand1=rs)
                        rp = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=yb)
                        xc = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=xc, data1=yb, data2=cs, op=nl.multiply)
                        xs = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=xs, data1=rp, data2=sn, op=nl.multiply)
                        ro = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                        if XG == 1:
                            nisa.tensor_tensor(dst=ro, data1=xc, data2=xs, op=nl.add, engine=nisa.engine.vector)
                        else:
                            nisa.tensor_tensor(dst=ro, data1=xc, data2=xs, op=nl.add, engine=nisa.engine.gpsimd)
                    else:
                        y = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=y, src=ps)
                        sq = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=sq, data1=y, data2=y, op=nl.multiply)
                        ss = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=ss, stationary=ones, moving=sq)
                        rs = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
                        nisa.tensor_scalar(dst=rs, data=rs, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
                        yn = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=yn, data1=y, data2=rs, op=nl.multiply)
                        yb = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                        if nt < NQL:
                            nisa.tensor_scalar(dst=yb, data=yn, op0=nl.multiply, operand0=qnc)
                        else:
                            nisa.tensor_scalar(dst=yb, data=yn, op0=nl.multiply, operand0=knc)
                        rp = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=yb)
                        xc = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=xc, data1=yb, data2=cs, op=nl.multiply)
                        xs = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=xs, data1=rp, data2=sn, op=nl.multiply)
                        ro = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=ro, data1=xc, data2=xs, op=nl.add)
                    if nt < NQL:
                        _hd(dst=qT_s[pid * NQL + nt, :, nl.ds(t0, C)], src=ro)
                    else:
                        j = nt - NQL
                        _hd(dst=kT_s[pid * NKL + j, :, nl.ds(t0, C)], src=ro)
                        _hd(dst=kcache[l, pid, j, :, nl.ds(t0, C)], src=ro)
                else:
                    j = nt - NQL - NKL
                    vb = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if FAST:
                        nisa.activation(dst=vb, op=nl.copy, data=ps)
                    else:
                        nisa.tensor_copy(dst=vb, src=ps)
                    vr = nl.ndarray((128, C // 128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for tb in range(C // 128):
                        tp = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=tp, stationary=vb[:, tb * 128:(tb + 1) * 128], moving=ident)
                        nisa.tensor_copy(dst=vr[:, tb, :], src=tp)
                    for tb in range(C // 128):
                        _hd(dst=v_s[pid * NKL + j, nl.ds(t0 + tb * 128, 128), :], src=vr[:, tb, :])
                        _hd(dst=vcache[l, pid, nl.ds(t0 + tb * 128, 128), j * D:(j + 1) * D], src=vr[:, tb, :])
        if STAGE != 4:   # STAGE 4: the cache-writing part of phase A only (the last layer, see pf_tail)
            if not (NOBAR and ATTN == 1 and S <= MYATTN):
                # (attn_prefill reads only the heads this core wrote in phase A: no barrier needed)
                nisa.core_barrier(data=qT_s, cores=(0, 1))
                nisa.core_barrier(data=kT_s, cores=(0, 1))
                nisa.core_barrier(data=v_s, cores=(0, 1))
            # ================= B: attention
            if HOIST < S and PREF:
                wo_sb = _load_w(wo_p[l, pid], 4 * 16 * 128, PWQ)
            if ATTN == 1 and S <= MYATTN and S <= 1024:
                attn_prefill_pipe(qT_s, kT_s, v_s, oT_s, pid, S)
            elif ATTN == 1 and S <= MYATTN:
                attn_prefill_big(qT_s, kT_s, v_s, oT_s, pid, S)
            elif ATTN == 1:
                _attention_cte(q=qT_s, k=kT_s, v=v_s, scale=SCALE, causal_mask=True, tp_q=False, tp_k=False, tp_out=True,
                               out_o_hbm=oT_s)
            else:
                for hh in range(NQL):
                    tq = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
                    _hd(dst=tq, src=qT_s[pid * NQL + hh])
                    _hd(dst=oT_s[pid * NQL + hh], src=tq)
            nisa.core_barrier(data=oT_s, cores=(0, 1))
            # ================= C: o_proj (N-split) + residual
            if HOIST < S and not PREF:
                wo_sb = _load_w(wo_p[l, pid], 4 * 16 * 128)
            if HOIST < S and PREF:
                wg = _load_w(wgu[l, pid], 2 * ITL * HT * 128, PWQ)
            for c in range(NC):
                t0 = c * C
                oc = nl.ndarray((128, 16, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                for hh in range(16):
                    _hd(dst=oc[:, hh, :], src=oT_s[hh, :, nl.ds(t0, C)])
                hc = nl.ndarray((128, 4, C), dtype=nl.float32, buffer=nl.sbuf)
                _hd(dst=hc, src=h[:, nl.ds(pid * 4, 4), nl.ds(t0, C)])
                ho = nl.ndarray((128, 4, C), dtype=nl.float32, buffer=nl.sbuf)
                for i in range(4):
                    ps = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(16):
                        nisa.nc_matmul(dst=ps, stationary=wo_sb[:, (i * 16 + kt) * 128:(i * 16 + kt + 1) * 128],
                                       moving=oc[:, kt, :])
                    nisa.tensor_tensor(dst=ho[:, i, :], data1=hc[:, i, :], data2=ps, op=nl.add)
                _hd(dst=h[:, nl.ds(pid * 4, 4), nl.ds(t0, C)], src=ho)
            nisa.core_barrier(data=h, cores=(0, 1))
            # ================= D: norm2 + gate/up + SiLU*up
            w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
            _hd(dst=w2, src=ln2[l])
            if HOIST < S and not PREF:
                wg = _load_w(wgu[l, pid], 2 * ITL * HT * 128)
            if HOIST < S and PREF:
                wd_sb = _load_w(wd_p[l, pid], 4 * 24 * 128, PWQ)
            for c in range(NC):
                t0 = c * C
                hc = nl.ndarray((128, HT, C), dtype=nl.float32, buffer=nl.sbuf)
                _hd(dst=hc, src=h[:, :, nl.ds(t0, C)])
                x2 = _norm_chunk(hc, w2, ones, C, onesb)
                for i in range(ITL):
                    pg = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(HT):
                        nisa.nc_matmul(dst=pg, stationary=wg[:, (i * HT + kt) * 128:(i * HT + kt + 1) * 128], moving=x2[:, kt, :])
                    pu = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(HT):
                        nisa.nc_matmul(dst=pu, stationary=wg[:, ((ITL + i) * HT + kt) * 128:((ITL + i) * HT + kt + 1) * 128],
                                       moving=x2[:, kt, :])
                    sg = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=sg, op=nl.silu, data=pg)
                    ab = nl.ndarray((128, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=ab, data1=sg, data2=pu, op=nl.multiply)
                    _hd(dst=act_s[pid * ITL + i, :, nl.ds(t0, C)], src=ab)
            nisa.core_barrier(data=act_s, cores=(0, 1))
            # ================= E: down (N-split) + residual
            if HOIST < S and not PREF:
                wd_sb = _load_w(wd_p[l, pid], 4 * 24 * 128)
            for c in range(NC):
                t0 = c * C
                ac = nl.ndarray((128, 24, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i in range(24):
                    _hd(dst=ac[:, i, :], src=act_s[i, :, nl.ds(t0, C)])
                hc = nl.ndarray((128, 4, C), dtype=nl.float32, buffer=nl.sbuf)
                _hd(dst=hc, src=h[:, nl.ds(pid * 4, 4), nl.ds(t0, C)])
                ho = nl.ndarray((128, 4, C), dtype=nl.float32, buffer=nl.sbuf)
                for i in range(4):
                    ps = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(24):
                        nisa.nc_matmul(dst=ps, stationary=wd_sb[:, (i * 24 + kt) * 128:(i * 24 + kt + 1) * 128],
                                       moving=ac[:, kt, :])
                    nisa.tensor_tensor(dst=ho[:, i, :], data1=hc[:, i, :], data2=ps, op=nl.add)
                _hd(dst=h[:, nl.ds(pid * 4, 4), nl.ds(t0, C)], src=ho)
            nisa.core_barrier(data=h, cores=(0, 1))

    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    if STAGE == 1 or STAGE == 2 or STAGE == 4:
        nisa.dma_copy(dst=pos_out, src=pl_sb)
        return pos_out
    # ---- LM head on the last real token (position plen - 1), first generated token
    lm1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lm1, data=pl_sb, op0=nl.add, operand0=-1)
    lreg = nisa.register_alloc()
    nisa.register_load(lreg, lm1)
    hl = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hl, src=h.ap(pattern=[[HT * S, 128], [S, HT]], offset=0, scalar_offset=lreg, indirect_dim=2))
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(hl, wf, HT, HID, ones, nl.bfloat16)
    tokf = _lm_argmax2(xf, wlm[pid], pid, 4, LMHW)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    preg = nisa.register_alloc()
    nisa.register_load(preg, pl_sb)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0), src=tok_i)
    nisa.dma_copy(dst=pos_out, src=pl_sb)
    if DBG == 2:
        return pos_out, hseen
    if DBG:
        h_out = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.shared_hbm)
        for c in range(NC):
            tt = nl.ndarray((128, HT, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tt, src=h[:, :, nl.ds(c * C, C)])
            nisa.dma_copy(dst=h_out[:, :, nl.ds(c * C, C)], src=tt)
        return pos_out, h_out
    return pos_out


@nki.jit
def prefill_int(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache):
    """variant with internal nl.shared_hbm scratch tensors: WRONG results (each core sees only its own writes)"""
    return _prefill(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT,
                    kcache, vcache, None, None, None, None, None, None, 0)


@nki.jit
def prefill(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache,
            s_h, s_q, s_k, s_v, s_o, s_a):
    """scratch tensors passed in by the caller (device buffers, contents irrelevant):
    s_h [128, 8, S] fp32, s_q [16, 128, S] bf16, s_k [8, 128, S] bf16, s_v [8, S, 128] bf16, s_o [16, 128, S] bf16,
    s_a [24, 128, S] bf16"""
    return _prefill(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT,
                    kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a, 1)


@nki.jit
def pf_emb(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a):
    return _prefill(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a, 1, 1)


@nki.jit
def pf_layer(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a):
    return _prefill(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a, 1, 2)


@nki.jit
def pf_head(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a):
    return _prefill(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a, 1, 3)



@nki.jit
def pf_kv(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a):
    """the last layer's KV cache writes (phase A, k / v tiles); the rest of that layer runs in pf_tail for the last token"""
    return _prefill(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo_p, wgu, wd_p, normf, wlm, rot, cosT, sinT, kcache, vcache, s_h, s_q, s_k, s_v, s_o, s_a, 1, 4)
