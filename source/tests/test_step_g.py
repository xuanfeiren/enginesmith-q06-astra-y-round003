"""Self-check of the generic LNC=2 decode step (kernels/step_g.py) for any Qwen3 size (default Qwen3-8B): real weights,
prompt fed token by token then greedy generation, one chain of queued executions (host/chain), compared with HF
transformers on the CPU (fp32). usage: python tests/test_step_g.py <kernel e.g. g8b_b512_s1> [G=16] [L=all]
env: MODEL (default ~/models/Qwen3-8B), NOBUILD, REPS, PROMPT"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
from nkibuild import build, bf16, chain, LAB
import step_g as SG
from ref_qwen3 import rope_tables
from qwen3_weights import SafeTensors, ws_layout, col_layout, rope_rot_matrix

TMAX = 4608
ORDER = ["pos", "plen", "prompt", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "normf",
         "wlm", "rot", "cos", "sin", "kcache", "vcache"]


def inputs_g(st, cfg, L, ids):
    H = cfg["hidden_size"]; HT = H // 128; NQ = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
    IT = cfg["intermediate_size"] // 128
    g = lambda l, n: st.get(f"model.layers.{l}.{n}")
    lay = lambda f: np.stack([f(l) for l in range(L)])
    per_core = lambda f: lay(lambda l: np.stack([f(l, c) for c in range(2)]))
    def qkv(l, c):
        w = ws_layout(np.concatenate([g(l, "self_attn.q_proj.weight"), g(l, "self_attn.k_proj.weight"),
                                      g(l, "self_attn.v_proj.weight")])).reshape(128, NQ + 2 * NKV, HT * 128)
        q0, k0 = c * NQ // 2, c * NKV // 2
        return np.concatenate([w[:, q0:q0 + NQ // 2], w[:, NQ + k0:NQ + k0 + NKV // 2],
                               w[:, NQ + NKV + k0:NQ + NKV + k0 + NKV // 2]], 1).reshape(128, -1)
    def wo(l, c):
        w = ws_layout(g(l, "self_attn.o_proj.weight")).reshape(128, HT, NQ, 128)
        return np.ascontiguousarray(w[:, :, c * NQ // 2:(c + 1) * NQ // 2]).reshape(128, -1)
    def gu(l, c):
        w = ws_layout(np.concatenate([g(l, "mlp.gate_proj.weight"), g(l, "mlp.up_proj.weight")])).reshape(128, 2 * IT, HT * 128)
        i0 = c * IT // 2
        return np.concatenate([w[:, i0:i0 + IT // 2], w[:, IT + i0:IT + i0 + IT // 2]], 1).reshape(128, -1)
    def wd(l, c):
        w = ws_layout(g(l, "mlp.down_proj.weight")).reshape(128, HT, IT, 128)
        return np.ascontiguousarray(w[:, :, c * IT // 2:(c + 1) * IT // 2]).reshape(128, -1)
    lmn = "lm_head.weight" if "lm_head.weight" in st.index else "model.embed_tokens.weight"
    lm = ws_layout(st.get(lmn)).reshape(128, 1187, HT, 128)
    lm = np.concatenate([lm, np.zeros((128, 1, HT, 128), bf16)], 1)
    cos, sin = rope_tables(TMAX)
    pr = np.zeros((TMAX, 1), np.int32); pr[:len(ids), 0] = ids
    return dict(
        pos=np.zeros((1, 1), np.int32), plen=np.array([[len(ids)]], np.int32), prompt=pr,
        toks=np.zeros((TMAX + 1, 1), np.int32), embed=np.ascontiguousarray(st.get("model.embed_tokens.weight")),
        ln1=lay(lambda l: col_layout(g(l, "input_layernorm.weight")).astype(np.float32)),
        ln2=lay(lambda l: col_layout(g(l, "post_attention_layernorm.weight")).astype(np.float32)),
        qn=lay(lambda l: g(l, "self_attn.q_norm.weight").reshape(128, 1).astype(np.float32)),
        kn=lay(lambda l: g(l, "self_attn.k_norm.weight").reshape(128, 1).astype(np.float32)),
        wqkv=per_core(qkv), wo=per_core(wo), wgu=per_core(gu), wd=per_core(wd),
        normf=col_layout(st.get("model.norm.weight")).astype(np.float32),
        wlm=np.ascontiguousarray(np.stack([lm[:, 594 * c:594 * (c + 1)] for c in range(2)])).reshape(2, 128, -1),
        rot=rope_rot_matrix().astype(bf16), cos=cos, sin=sin,
        kcache=np.zeros((L, 2, NKV // 2, 128, TMAX), bf16), vcache=np.zeros((L, 2, TMAX, NKV // 2 * 128), bf16))


F8 = None


def quantize_fp8(inp, cfg):
    """per core, per output channel absmax scaling to float8_e4m3 (max 240); returns inputs + scale tensors."""
    import ml_dtypes
    H = cfg["hidden_size"]; HT = H // 128; NQ = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
    IT = cfg["intermediate_size"] // 128
    out = dict(inp)
    def q(w, NT, KT):
        lead = w.shape[:-2]
        a = w.astype(np.float32).reshape(*lead, 128, NT, KT, 128)          # [..., p(k), nt, kt, j(n)]
        amax = np.abs(a).max(axis=(-4, -2), keepdims=True)                  # per (nt, j)
        sc = np.maximum(amax, 1e-30) / 240.0
        w8 = (a / sc).astype(ml_dtypes.float8_e4m3).reshape(w.shape)
        scol = np.ascontiguousarray(np.moveaxis(sc[..., 0, :, 0, :], -1, -2)).reshape(*lead, 128, NT)   # [..., j, nt]
        return w8, scol.astype(np.float32)
    nqkv = NQ // 2 + NKV
    out["wqkv"], out["sqkv"] = q(inp["wqkv"], nqkv, HT)
    out["wo"], out["so"] = q(inp["wo"], HT, NQ // 2)
    out["wgu"], out["sgu"] = q(inp["wgu"], IT, HT)
    out["wd"], out["sd"] = q(inp["wd"], HT, IT // 2)
    out["wlm"], out["slm"] = q(inp["wlm"], 594, HT)
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3-8B"))
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    L = int(sys.argv[3]) if len(sys.argv) > 3 else cfg["num_hidden_layers"]
    import transformers, torch
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    text = open(os.environ["PROMPT_FILE"]).read() if os.environ.get("PROMPT_FILE") else \
        os.environ.get("PROMPT", "The capital of France is Paris. The capital of Germany is")
    ids = tok(text)["input_ids"][:int(os.environ.get("PROMPT_TOKENS", "100000"))]
    P = len(ids)
    inp = inputs_g(SafeTensors(MODEL), cfg, L, ids)
    order = ORDER
    if os.environ.get("Q8"):
        inp = quantize_fp8(inp, cfg)
        order = ORDER + ["sqkv", "so", "sgu", "sd", "slm"]
    print("inputs ready:", sum(v.nbytes for v in inp.values()) / 1e9, "GB", flush=True)
    nm = f"{name}_L{L}"
    if not os.environ.get("NOBUILD"):
        build(getattr(SG, name), nm, *[inp[k] for k in order], lnc=2)
    r, so = chain(f"{LAB}/neff/{nm}.neff", inp, {"toks": ((TMAX + 1,), np.int32)}, P + G - 1, "pos:output_0",
                  reps=int(os.environ.get("REPS", "1")))
    for l in so.splitlines():
        if "CHAIN" in l or "nrt_load" in l or "upload" in l: print(l)
    got = r["toks"][1:P + G].tolist()
    del inp
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if L < cfg["num_hidden_layers"]:
        m.model.layers = m.model.layers[:L]; m.config.num_hidden_layers = L
    with torch.no_grad():
        full = m.generate(torch.tensor([ids]), max_new_tokens=G, do_sample=False, num_beams=1)[0].tolist()
        lg = m(torch.tensor([full])).logits[0].float().numpy()
    ref_pred = lg.argmax(-1)[:P + G - 1].tolist()
    tf = sum(int(a == b) for a, b in zip(got[:P - 1], ref_pred[:P - 1]))
    gen_k, gen_r = got[P - 1:], full[P:P + G]
    same = next((i for i in range(G) if gen_k[i] != gen_r[i]), G)
    print(f"[fp32] prompt teacher-forced argmax agree {tf}/{P - 1}; generated: first {same}/{G} tokens identical")
    mis = [i for i in range(P - 1) if got[i] != ref_pred[i]]
    if mis:
        gaps = [float(lg[i][ref_pred[i]] - lg[i][got[i]]) for i in mis]
        print(f"   teacher-forced mismatches: fp32 logit gap median {float(np.median(gaps)):.3f} max {max(gaps):.3f}")
    ok = tf >= min(P - 2, int(0.98 * (P - 1))) and same >= min(G, 8)
    if same < G:
        i = P - 1 + same
        gap = float(lg[i][gen_r[same]] - lg[i][gen_k[same]])
        print(f"   divergence at gen {same}: kernel {gen_k[same]} ref {gen_r[same]}, fp32 logit gap {gap:.4f}")
        ok = ok or gap < 0.1
    print("kernel text:", repr(tok.decode(gen_k)))
    print("PASS" if ok else "FAIL")
