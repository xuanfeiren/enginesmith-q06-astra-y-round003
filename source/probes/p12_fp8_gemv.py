"""Weight-only fp8 GEMV (LNC=2): W stored as float8_e4m3 (legacy Trn2 format, max 240) with one fp32 scale per output
channel; stationary = fp8 weight tile [128 k, 128 n], moving = bf16 x column, fp32 PSUM, y[n] = s[n] * (W8 x)[n].
Same loop as p11 (row layout, PSUM tile width CH). Compares (a) kernel vs numpy with the dequantised weights
(kernel arithmetic), (b) vs the bf16 weights (quantisation error), and the queued time vs the bf16 kernel."""
import sys, os, numpy as np, ml_dtypes
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun, LAB

KT = int(os.environ.get("KT", "32"))
CH = int(os.environ.get("CH", "256"))
F8 = ml_dtypes.float8_e4m3


def _gv8(x, w, sc, SL):
    pid = nl.program_id(0)
    NB = w.shape[2] // (SL * KT * 128)
    NT = NB * SL
    xT = nl.ndarray((128, KT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=x)
    s_sb = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_sb, src=sc[pid])
    y = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    c0 = 0
    while c0 < NT:
        n = CH
        if NT - c0 < CH:
            n = NT - c0
        ps = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
        for b in range(c0 // SL, (c0 + n) // SL):
            wt = nl.ndarray((128, SL * KT * 128), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt, src=w[pid, :, b * SL * KT * 128:(b + 1) * SL * KT * 128])
            for t in range(SL):
                for k in range(KT):
                    nisa.nc_matmul(dst=ps[:, b * SL + t - c0:b * SL + t - c0 + 1],
                                   stationary=wt[:, (t * KT + k) * 128:(t * KT + k + 1) * 128], moving=xT[:, k:k + 1])
        nisa.tensor_tensor(dst=y[:, c0:c0 + n], data1=ps[:, 0:n], data2=s_sb[:, c0:c0 + n], op=nl.multiply)
        c0 += n
    out = nl.ndarray((2, 128, NT), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[pid], src=y)
    return out

@nki.jit
def f8_s1(x, w, sc): return _gv8(x, w, sc, 1)
@nki.jit
def f8_s2(x, w, sc): return _gv8(x, w, sc, 2)
@nki.jit
def f8_s4(x, w, sc): return _gv8(x, w, sc, 4)

if __name__ == "__main__":
    NT = int(os.environ.get("NT", "592"))
    rng = np.random.default_rng(0)
    xT = (rng.standard_normal((128, KT)) * 0.5).astype(bf16)
    W = (rng.standard_normal((2, NT * 128, KT * 128)) * 0.02 * rng.uniform(0.5, 2, (2, NT * 128, 1))).astype(bf16).astype(np.float32)
    s = np.abs(W).max(2, keepdims=True) / 240.0                      # per output channel
    W8 = (W / s).astype(F8)
    Wd = W8.astype(np.float32) * s
    # ws layout per core: w[c, p, ((nt*KT)+kt)*128 + j] = W[c, nt*128 + j, kt*128 + p]
    lay = lambda A: np.ascontiguousarray(A.reshape(2, NT, 128, KT, 128).transpose(0, 4, 1, 3, 2)).reshape(2, 128, -1)
    w8 = lay(W8)
    sc = np.ascontiguousarray(s[:, :, 0].reshape(2, NT, 128).transpose(0, 2, 1)).astype(np.float32)   # [2, 128 j, NT]
    x = xT.T.reshape(-1).astype(np.float32)
    for nm in sys.argv[1:]:
        full = f"p12_{nm}_NT{NT}_KT{KT}_CH{CH}"
        build(globals()[nm], full, xT, w8, sc, lnc=2)
        r, so = nrun(f"{LAB}/neff/{full}.neff", {"x": xT, "w": w8, "sc": sc}, {"output_0": ((2, 128, NT), np.float32)}, bench=20)
        o = r["output_0"].transpose(0, 2, 1).reshape(2, -1)            # y[c, nt*128 + j]
        ref_d = np.einsum("cnk,k->cn", Wd, x); ref_b = np.einsum("cnk,k->cn", W, x)
        e_d = np.abs(o - ref_d).max() / np.abs(ref_d).max(); e_b = np.abs(o - ref_b).max() / np.abs(ref_b).max()
        rel_rms = np.sqrt(((o - ref_b) ** 2).mean() / (ref_b ** 2).mean())
        us = float([l for l in so.splitlines() if "BENCH queued" in l][0].split()[2])
        print(f"{full}: vs dequantised ref {e_d:.2g}, vs bf16 weights max {e_b:.3g} rms {rel_rms:.3g}; queued {us:.1f} us "
              f"for {w8.nbytes/1e6:.0f} MB fp8 -> {w8.nbytes/us/1e3:.0f} GB/s ({2*w8.nbytes/us/1e3:.0f} GB/s bf16-equivalent)", flush=True)
