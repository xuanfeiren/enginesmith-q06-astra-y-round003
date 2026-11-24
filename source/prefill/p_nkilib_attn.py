"""Baseline: nkilib core/attention/attention_cte.py (Apache-2.0, shipped in the neuronx-cc wheel, public NKI API)
at Qwen3-0.6B prefill shapes: 16 q heads, 8 kv heads (GQA), d 128, causal. Correctness vs numpy, queued time."""
import sys, os, numpy as np
import nki, nki.language as nl
from nkibuild import build, bf16, nrun, LAB
from nkilib.core.attention.attention_cte import attention_cte

SC = 1.0 / np.sqrt(128)

def ref(q, k, v):
    q, k, v = (np.asarray(a, np.float32) for a in (q, k, v))
    H, S, D = q.shape
    out = np.zeros_like(q)
    mask = np.triu(np.full((S, S), -np.inf, np.float32), 1)
    for h in range(H):
        s = q[h] @ k[h // (H // k.shape[0])] * SC + mask
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        out[h] = p @ v[h // (H // k.shape[0])]
    return out


if __name__ == "__main__":
    lnc = int(os.environ.get("LNC", "2"))
    for S in [int(a) for a in sys.argv[1:]]:
        rng = np.random.default_rng(0)
        HQ = int(os.environ.get("HQ", "16"))
        q = rng.standard_normal((HQ, S, 128)).astype(bf16)
        k = rng.standard_normal((8, 128, S)).astype(bf16)
        v = rng.standard_normal((8, S, 128)).astype(bf16)
        nm = f"nkilib_attn_H{HQ}_S{S}_lnc{lnc}"
        _, ct = build(attention_cte, nm, q, k, v, lnc=lnc, kw=dict(scale=SC, causal_mask=True, tp_q=True, tp_k=False, tp_out=False))
        r, so = nrun(f"{LAB}/neff/{nm}.neff", {"q": q, "k": k, "v": v}, {"output_0": ((HQ, S, 128), bf16)}, bench=20)
        o = r["output_0"].astype(np.float32); rf = ref(q, k, v)
        err = np.abs(o - rf).max()
        us = float([l for l in so.splitlines() if "BENCH queued" in l][0].split()[2])
        fl = 4.0 * 128 * HQ * S * S / 2
        print(f"{nm}: compile {ct:.1f}s max abs err {err:.3g} (|ref| {np.abs(rf).max():.3g}); queued {us:.1f} us -> "
              f"{fl / us / 1e6:.1f} TFLOP/s (causal flops 4*d*H*S^2/2)", flush=True)
