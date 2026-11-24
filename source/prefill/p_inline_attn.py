"""Can nkilib's _attention_cte (undecorated) be inlined in our own kernel, after our own DMAs into shared HBM?
q^T [16, 128, S] (tp_q=False), k^T [8, 128, S], v [8, S, 128] -> out o^T [16, 128, S] (tp_out=True)."""
import sys, os, numpy as np
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun, LAB
from nkilib.core.attention.attention_cte import _attention_cte
from p_nkilib_attn import ref, SC


@nki.jit
def inl(qT, kT, v):
    H, D, S = qT.shape
    # copy the inputs through SBUF into internal HBM tensors (stand-in for a producer in the same NEFF)
    q2 = nl.ndarray((H, D, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    k2 = nl.ndarray((8, D, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    for i in range(H // 2):
        t = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=qT[pid * (H // 2) + i])
        nisa.dma_copy(dst=q2[pid * (H // 2) + i], src=t)
    for i in range(4):
        t = nl.ndarray((D, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=kT[pid * 4 + i])
        nisa.dma_copy(dst=k2[pid * 4 + i], src=t)
    nisa.core_barrier(data=q2, cores=(0, 1))
    nisa.core_barrier(data=k2, cores=(0, 1))
    o = nl.ndarray((H, D, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    _attention_cte(q=q2, k=k2, v=v, scale=SC, causal_mask=True, tp_q=False, tp_k=False, tp_out=True, out_o_hbm=o)
    return o


if __name__ == "__main__":
    S = int(sys.argv[1])
    rng = np.random.default_rng(0)
    q = rng.standard_normal((16, S, 128)).astype(bf16)
    k = rng.standard_normal((8, 128, S)).astype(bf16)
    v = rng.standard_normal((8, S, 128)).astype(bf16)
    qT = np.ascontiguousarray(q.transpose(0, 2, 1))
    nm = f"inline_attn_S{S}"
    build(inl, nm, qT, k, v, lnc=2)
    r, so = nrun(f"{LAB}/neff/{nm}.neff", {"qT": qT, "kT": k, "v": v}, {"output_0": ((16, 128, S), bf16)}, bench=20)
    o = r["output_0"].astype(np.float32).transpose(0, 2, 1)
    rf = ref(q, k, v)
    print(f"{nm}: max abs err {np.abs(o - rf).max():.3g} (|ref| {np.abs(rf).max():.3g}); per head max err",
          np.abs(o - rf).max((1, 2)).round(3).tolist())
    print([l for l in so.splitlines() if "BENCH queued" in l][0])
