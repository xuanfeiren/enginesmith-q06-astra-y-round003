"""Self-check of kernels/prefill_lnc2.py: real prompt (padded to the bucket S), compares the first generated token
with HF transformers (fp32, CPU) and the K/V cache written for every layer with HF's past_key_values.
usage: python tests/test_prefill.py <S> [plen]     env: NOBUILD, BENCH (queued timing iterations), PROMPT_FILE"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nkibuild import build, bf16, nrun, LAB
import prefill_lnc2 as PF
from qwen3_weights import SafeTensors
from test_layer import MODEL, TMAX
from test_step import step_inputs, split_inputs

ORDER = ["ids", "plen", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo_p", "wgu", "wd_p", "normf", "wlm",
         "rot", "cosT", "sinT", "kcache", "vcache", "s_h", "s_q", "s_k", "s_v", "s_o", "s_a"]


def prefill_inputs(st, L, ids, S):
    inp = step_inputs(st, L, ids)
    full_wo, full_wd = inp["wo"], inp["wd"]
    inp = split_inputs(inp)
    o = full_wo.reshape(L, 128, 8, 16, 128); d = full_wd.reshape(L, 128, 8, 24, 128)
    cat = lambda xs: np.ascontiguousarray(np.stack(xs, 1)).reshape(L, 2, 128, -1)
    inp["wo_p"] = cat([o[:, :, 4 * c:4 * c + 4] for c in range(2)])
    inp["wd_p"] = cat([d[:, :, 4 * c:4 * c + 4] for c in range(2)])
    cos, sin = inp["cos"], inp["sin"]                      # [TMAX, 64]
    inp["cosT"] = np.ascontiguousarray(np.concatenate([cos, cos], 1).T).astype(np.float32)
    inp["sinT"] = np.ascontiguousarray(np.concatenate([sin, sin], 1).T).astype(np.float32)
    idp = np.zeros((S, 1), np.int32); idp[:len(ids), 0] = ids
    inp["ids"] = idp
    inp.update(s_h=np.zeros((128, 8, S), np.float32), s_q=np.zeros((16, 128, S), bf16), s_k=np.zeros((8, 128, S), bf16),
               s_v=np.zeros((8, S, 128), bf16), s_o=np.zeros((16, 128, S), bf16), s_a=np.zeros((24, 128, S), bf16))
    return inp


if __name__ == "__main__":
    S = int(sys.argv[1])
    L = int(os.environ.get("L", "28"))
    import transformers, torch
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    text = open(os.environ["PROMPT_FILE"]).read() if os.environ.get("PROMPT_FILE") else \
        "The capital of France is Paris. The capital of Germany is"
    ids = tok(text)["input_ids"]
    P = int(sys.argv[2]) if len(sys.argv) > 2 else min(len(ids), S)
    ids = (ids * (P // len(ids) + 1))[:P]
    st = SafeTensors(MODEL)
    inp = prefill_inputs(st, L, ids, S)
    nm = f"prefill_S{S}_L{L}"
    if not os.environ.get("NOBUILD"):
        build(PF.prefill, nm, *[inp[k] for k in ORDER], lnc=2)
    r, so = nrun(f"{LAB}/neff/{nm}.neff", {k: inp[k] for k in ORDER},
                 {"toks": ((TMAX + 1,), np.int32), "output_0": ((1, 1), np.int32),
                  "kcache": (inp["kcache"].shape, bf16), "vcache": (inp["vcache"].shape, bf16)},
                 bench=int(os.environ.get("BENCH", "0")))
    for l in so.splitlines():
        if "BENCH" in l or "nrt_load" in l: print(l)
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if L < 28:
        m.model.layers = m.model.layers[:L]; m.config.num_hidden_layers = L
    with torch.no_grad():
        out = m(torch.tensor([ids]), use_cache=True)
    lg = out.logits[0, -1].numpy()
    first = int(r["toks"][P]); ref_first = int(lg.argmax())
    print(f"first token: kernel {first} ref {ref_first} (logit gap ref_top1 - kernel pick {lg[ref_first] - lg[first]:.4f}); "
          f"output_0 = {int(r['output_0'][0, 0])} (plen {P})")
    pkv = out.past_key_values
    mb = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    if L < 28:
        mb.model.layers = mb.model.layers[:L]; mb.config.num_hidden_layers = L
    with torch.no_grad():
        pkb = mb(torch.tensor([ids]), use_cache=True).past_key_values
    worst = 0.0; worst_b = 0.0
    for l in range(L):
        Kr = pkv.layers[l].keys[0].numpy(); Vr = pkv.layers[l].values[0].numpy()        # [8, P, 128]
        kc = r["kcache"][l].astype(np.float32)          # [2, 4, 128, T]
        vc = r["vcache"][l].astype(np.float32)          # [2, T, 512]
        Kk = kc.reshape(8, 128, -1)[:, :, :P].transpose(0, 2, 1)
        Vk = vc[:, :P].reshape(2, P, 4, 128).transpose(0, 2, 1, 3).reshape(8, P, 128)
        ek = np.abs(Kk - Kr).max() / np.abs(Kr).max(); ev = np.abs(Vk - Vr).max() / np.abs(Vr).max()
        worst = max(worst, ek, ev)
        Kb = pkb.layers[l].keys[0].float().numpy(); Vb = pkb.layers[l].values[0].float().numpy()
        eb = max(np.abs(Kb - Kr).max() / np.abs(Kr).max(), np.abs(Vb - Vr).max() / np.abs(Vr).max())
        worst_b = max(worst_b, eb)
        if l in (0, 1, L // 2, L - 1) or max(ek, ev) > 0.03:
            print(f"layer {l}: K max rel err {ek:.4f}, V max rel err {ev:.4f}   (HF bf16 vs fp32: {eb:.4f})")
    ok = first == ref_first or lg[ref_first] - lg[first] < 0.05
    ok &= worst < max(0.05, 1.5 * worst_b) and int(r["output_0"][0, 0]) == P
    print(f"worst K/V rel err over layers {worst:.4f}; HF bf16 vs HF fp32 on the same metric {worst_b:.4f}")
    print("PASS" if ok else "FAIL")
