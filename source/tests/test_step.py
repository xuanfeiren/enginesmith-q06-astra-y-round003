"""Self-check of the whole-step NEFF (kernels/step_decode.py): a real prompt is fed token by token and G tokens
are generated greedily, all as one chain of queued executions (host/chain); the token history is compared with
HF transformers on the CPU (fp32 and bf16), teacher-forced over the prompt and free-running afterwards.
usage: python tests/test_step.py <kernel> [G=32] [L=28]      env: LNC, NOBUILD, REPS, PROMPT"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nkibuild import build, bf16, chain, LAB
import step_decode as SD
from ref_qwen3 import rope_tables
from qwen3_weights import SafeTensors, ws_layout, col_layout, rope_rot_matrix
from test_layer import layer_weights, MODEL, TMAX

ORDER = ["pos", "plen", "prompt", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "normf",
         "wlm", "rot", "cos", "sin", "kcache", "vcache"]


def step_inputs(st, L, prompt_ids, tmax=TMAX):
    Ws = [layer_weights(st, l) for l in range(L)]
    stk = lambda f: np.stack([f(W) for W in Ws])
    cos, sin = rope_tables(tmax)
    pr = np.zeros((tmax, 1), np.int32); pr[:len(prompt_ids), 0] = prompt_ids
    return dict(
        pos=np.zeros((1, 1), np.int32), plen=np.array([[len(prompt_ids)]], np.int32), prompt=pr,
        toks=np.zeros((tmax + 1, 1), np.int32), embed=np.ascontiguousarray(st.get("model.embed_tokens.weight")),
        ln1=stk(lambda W: col_layout(W["ln1"]).astype(np.float32)), ln2=stk(lambda W: col_layout(W["ln2"]).astype(np.float32)),
        qn=stk(lambda W: W["qn"].reshape(128, 1).astype(np.float32)), kn=stk(lambda W: W["kn"].reshape(128, 1).astype(np.float32)),
        wqkv=stk(lambda W: ws_layout(np.concatenate([W["q"], W["k"], W["v"]]))),
        wo=stk(lambda W: ws_layout(W["o"])), wgu=stk(lambda W: ws_layout(np.concatenate([W["gate"], W["up"]]))),
        wd=stk(lambda W: ws_layout(W["down"])),
        normf=col_layout(st.get("model.norm.weight")).astype(np.float32),
        wlm=ws_layout(st.get("lm_head.weight")), rot=rope_rot_matrix().astype(bf16), cos=cos, sin=sin,
        kcache=np.zeros((L, 8, 128, tmax), bf16), vcache=np.zeros((L, tmax, 1024), bf16))


def split_inputs(inp):
    """Per-core weight / cache layouts of kernels/step_lnc2.py (core c = program 0/1 of an LNC=2 NEFF)."""
    L = inp["wqkv"].shape[0]
    t = lambda w, nt, kt: w.reshape(w.shape[0], 128, nt, kt, 128)
    q = t(inp["wqkv"], 32, 8); o = t(inp["wo"], 8, 16); gu = t(inp["wgu"], 48, 8); dn = t(inp["wd"], 8, 24)
    cat = lambda xs: np.ascontiguousarray(np.stack(xs, 1)).reshape(L, 2, 128, -1)
    out = dict(inp)
    out["wqkv"] = cat([np.concatenate([q[:, :, 8 * c:8 * c + 8], q[:, :, 16 + 4 * c:20 + 4 * c], q[:, :, 24 + 4 * c:28 + 4 * c]], 2) for c in range(2)])
    out["wo"] = cat([o[:, :, :, 8 * c:8 * c + 8] for c in range(2)])
    out["wgu"] = cat([np.concatenate([gu[:, :, 12 * c:12 * c + 12], gu[:, :, 24 + 12 * c:36 + 12 * c]], 2) for c in range(2)])
    out["wd"] = cat([dn[:, :, :, 12 * c:12 * c + 12] for c in range(2)])
    lm = inp["wlm"].reshape(128, 1187, 8, 128)
    lm = np.concatenate([lm, np.zeros((128, 1, 8, 128), lm.dtype)], 1)
    out["wlm"] = np.ascontiguousarray(np.stack([lm[:, 594 * c:594 * (c + 1)] for c in range(2)])).reshape(2, 128, -1)
    T = inp["vcache"].shape[1]
    out["kcache"] = np.zeros((L, 2, 4, 128, T), bf16)
    out["vcache"] = np.zeros((L, 2, T, 512), bf16)
    return out


def hf_reference(ids, G, dtype, L):
    import torch, transformers
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    if L < m.config.num_hidden_layers:
        m.model.layers = m.model.layers[:L]
        m.config.num_hidden_layers = L
    m.eval()
    with torch.no_grad():
        out = m.generate(torch.tensor([ids]), max_new_tokens=G, do_sample=False, num_beams=1)
        full = out[0].tolist()
        lg = m(torch.tensor([full])).logits[0].float().numpy()
    return full, lg


if __name__ == "__main__":
    name = sys.argv[1]
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 28
    lnc = int(os.environ.get("LNC", "1"))
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    text = os.environ.get("PROMPT", "The capital of France is Paris. The capital of Germany is")
    if os.environ.get("PROMPT_FILE"):
        text = open(os.environ["PROMPT_FILE"]).read()
    ids = tok(text)["input_ids"]
    if os.environ.get("PROMPT_TOKENS"):
        ids = ids[:int(os.environ["PROMPT_TOKENS"])]
    P = len(ids)
    st = SafeTensors(MODEL)
    inp = step_inputs(st, L, ids)
    mod = SD
    if lnc == 2:
        import step_lnc2 as mod
        if name.startswith(("g06b", "p06b", "q06b", "c06b")):
            import step_g as mod
        inp = split_inputs(inp)
    nm = f"{name}_L{L}_lnc{lnc}"
    if not os.environ.get("NOBUILD"):
        _, ct = build(getattr(mod, name), nm, *[inp[k] for k in ORDER], lnc=lnc)
    nsteps = P + G - 1
    r, so = chain(f"{LAB}/neff/{nm}.neff", inp, {"toks": ((TMAX + 1,), np.int32)}, nsteps, "pos:output_0",
                  reps=int(os.environ.get("REPS", "1")), vncs=1)
    for l in so.splitlines():
        if "CHAIN" in l or "nrt_load" in l or "upload" in l: print(l)
    got = r["toks"][1:P + G].tolist()          # prediction after each position 0..P+G-2
    import torch
    ok = True
    for dt in (torch.float32, torch.bfloat16):
        full, lg = hf_reference(ids, G, dt, L)
        ref_pred = lg.argmax(-1)[:P + G - 1].tolist()   # teacher-forced on the ref's own sequence
        gen_k, gen_r = got[P - 1:], full[P:P + G]
        n_same = next((i for i in range(G) if gen_k[i] != gen_r[i]), G)
        tf = sum(int(a == b) for a, b in zip(got[:P - 1], ref_pred[:P - 1]))
        mis = [i for i in range(P - 1) if got[i] != ref_pred[i]]
        if mis:
            gaps = [float(lg[i][ref_pred[i]] - lg[i][got[i]]) for i in mis]
            print(f"   [{dt}] teacher-forced mismatches: ref logit(top1) - logit(kernel's pick): max {max(gaps):.3f}, "
                  f"median {float(np.median(gaps)):.3f} over {len(mis)}")
        print(f"[{dt}] prompt teacher-forced argmax agree {tf}/{P - 1}; generated: first {n_same}/{G} tokens identical")
        if n_same < G:
            i = P - 1 + n_same
            top = np.argsort(-lg[i])[:3]
            print(f"   diverges at gen {n_same}: kernel {gen_k[n_same]} ref {gen_r[n_same]}; ref top3 {top.tolist()} "
                  f"logits {lg[i][top].round(3).tolist()}")
        if dt == torch.float32:
            ok &= tf >= min(P - 2, int(0.99 * (P - 1))) and n_same >= min(G, 16)
    print("kernel text:", repr(tok.decode(got[P - 1:])))
    print("PASS" if ok else "FAIL")
