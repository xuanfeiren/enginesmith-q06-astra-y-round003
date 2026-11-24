"""End-to-end: prefill NEFF + decode NEFF from C++ (host/generate), greedy tokens vs HF transformers (fp32 CPU).
usage: python tests/test_generate.py <S> <decode bucket> <plen> <n_new>   (NEFFs must exist: prefill_S<S>_L28.neff,
step2_b<bucket>_L28_lnc2.neff)  env: PROMPT_FILE, REPS"""
import sys, os, subprocess, tempfile, shutil, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nkibuild import LAB
from qwen3_weights import SafeTensors
from test_layer import MODEL, TMAX
from test_prefill import prefill_inputs

S, B, P, G = (int(a) for a in sys.argv[1:5])
import transformers, torch
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
text = open(os.environ["PROMPT_FILE"]).read() if os.environ.get("PROMPT_FILE") else \
    "The capital of France is Paris. The capital of Germany is"
ids = tok(text)["input_ids"]
ids = (ids * (P // len(ids) + 1))[:P]
inp = prefill_inputs(SafeTensors(MODEL), 28, ids, S)
inp["plen"] = np.array([[P]], np.int32)
pf_arg = f"{LAB}/neff/prefill_S{S}_L28.neff"
if os.environ.get("LAYERED"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
    import prefill_lnc2 as PF
    from nkibuild import build
    from test_prefill import ORDER
    names = []
    per_layer = ("ln1", "ln2", "qn", "kn", "wqkv", "wo_p", "wgu", "wd_p", "kcache", "vcache")
    for st in ("emb", "layer", "head"):
        nm = f"pf_{st}_S{S}"
        names.append(f"{LAB}/neff/{nm}.neff")
        if not os.path.exists(names[-1]) or os.environ.get("REBUILD"):
            args = [inp[k][0:1] if (st == "layer" and k in per_layer) else inp[k] for k in ORDER]
            build(getattr(PF, f"pf_{st}"), nm, *args, lnc=2)
    pf_arg = ",".join(names)
d = tempfile.mkdtemp(prefix="gen_", dir=os.path.join(LAB, "build"))
for k, v in inp.items():
    np.ascontiguousarray(v).tofile(os.path.join(d, k + ".bin"))
p = subprocess.run([f"{LAB}/host/generate", pf_arg, f"{LAB}/neff/step2_b{B}_L28_lnc2.neff",
                    d, str(G - 1), os.environ.get("REPS", "2")], capture_output=True, text=True)
print("GENOUT", p.returncode, p.stdout[-2000:], p.stderr[-3000:])
toks = np.fromfile(os.path.join(d, "out_toks.bin"), np.int32)
shutil.rmtree(d, ignore_errors=True)
got = toks[P:P + G].tolist()
m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
with torch.no_grad():
    ref = m.generate(torch.tensor([ids]), max_new_tokens=G, do_sample=False, num_beams=1)[0, P:].tolist()
same = next((i for i in range(G) if got[i] != ref[i]), G)
print(f"generated {G} tokens: first {same} identical to HF fp32 greedy")
ok = same >= min(G, 16)
if same < G:
    # at the divergence point both sequences share the prefix: is the kernel's pick a near-tie in fp32?
    with torch.no_grad():
        lg = m(torch.tensor([ids + ref[:same]])).logits[0, -1].numpy()
    gap = float(lg[ref[same]] - lg[got[same]])
    srt = np.sort(lg)[::-1]
    print(f"   divergence at {same}: ref {ref[same]} kernel {got[same]}; fp32 logit gap {gap:.4f} (top1-top2 {srt[0]-srt[1]:.4f})")
    mb = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    with torch.no_grad():
        refb = mb.generate(torch.tensor([ids]), max_new_tokens=G, do_sample=False, num_beams=1)[0, P:].tolist()
    sb = next((i for i in range(G) if refb[i] != ref[i]), G)
    print(f"   HF bf16 vs HF fp32: first {sb} identical; kernel vs HF bf16: first "
          f"{next((i for i in range(G) if got[i] != refb[i]), G)} identical")
    ok = ok or gap < 0.1
print("kernel:", repr(tok.decode(got)))
print("PASS" if ok else "FAIL")
