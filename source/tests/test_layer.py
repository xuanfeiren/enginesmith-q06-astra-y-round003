"""Self-check of kernels/layer_decode.py with real Qwen3-0.6B weights against ref_qwen3.layer_ref.
usage: python test_layer.py <kernel name> <L> <pos> [n_exec]"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
from nkibuild import build, bf16, nrun, LAB
import layer_decode as LD
from ref_qwen3 import rope_tables, layer_ref
from qwen3_weights import SafeTensors, ws_layout, col_layout, rope_rot_matrix

TMAX = 4608
MODEL = os.path.expanduser("~/models/Qwen3-0.6B")


def layer_weights(st, l):
    g = lambda n: st.get(f"model.layers.{l}.{n}")
    return dict(ln1=g("input_layernorm.weight"), ln2=g("post_attention_layernorm.weight"),
                qn=g("self_attn.q_norm.weight"), kn=g("self_attn.k_norm.weight"),
                q=g("self_attn.q_proj.weight"), k=g("self_attn.k_proj.weight"), v=g("self_attn.v_proj.weight"),
                o=g("self_attn.o_proj.weight"), gate=g("mlp.gate_proj.weight"), up=g("mlp.up_proj.weight"),
                down=g("mlp.down_proj.weight"))


def kernel_inputs(Ws, rng, pos):
    L = len(Ws)
    st = lambda f: np.stack([f(W) for W in Ws])
    cos, sin = rope_tables(TMAX)
    kc = (rng.standard_normal((L, 8, 128, TMAX)) * 1.5).astype(bf16)
    vc = (rng.standard_normal((L, TMAX, 1024)) * 0.5).astype(bf16)
    kc[:, :, :, pos:] = 0; vc[:, pos:] = 0
    return dict(
        h_in=(rng.standard_normal((128, 8)) * 0.5).astype(np.float32),
        ln1=st(lambda W: col_layout(W["ln1"])), ln2=st(lambda W: col_layout(W["ln2"])),
        qn=st(lambda W: W["qn"].reshape(128, 1)), kn=st(lambda W: W["kn"].reshape(128, 1)),
        wqkv=st(lambda W: ws_layout(np.concatenate([W["q"], W["k"], W["v"]]))),
        wo=st(lambda W: ws_layout(W["o"])), wgu=st(lambda W: ws_layout(np.concatenate([W["gate"], W["up"]]))),
        wd=st(lambda W: ws_layout(W["down"])), rot=rope_rot_matrix().astype(bf16), cos=cos, sin=sin,
        kcache=kc, vcache=vc, pos=np.array([[pos]], np.int32))


ORDER = ["h_in", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "rot", "cos", "sin", "kcache", "vcache", "pos"]

if __name__ == "__main__":
    name = sys.argv[1]; L = int(sys.argv[2]); pos = int(sys.argv[3])
    st = SafeTensors(MODEL)
    L0 = int(os.environ.get("L0", "0"))
    Ws = [layer_weights(st, l) for l in range(L0, L0 + L)]
    rng = np.random.default_rng(0)
    inp = kernel_inputs(Ws, rng, pos)
    if os.environ.get("TOKEN"):      # real token embedding as h_in (column layout)
        e = st.get("model.embed_tokens.weight")[int(os.environ["TOKEN"])].astype(np.float32)
        inp["h_in"] = np.ascontiguousarray(e.reshape(8, 128).T)
    lnc = int(os.environ.get("LNC", "1"))
    nm = f"{name}_L{L}_lnc{lnc}"
    if not os.environ.get("NOBUILD"):
        _, ct = build(getattr(LD, name), nm, *[inp[k] for k in ORDER], lnc=lnc)
    r, so = nrun(f"{LAB}/neff/{nm}.neff", inp, {"output_0": ((128, 8), np.float32), "kcache": (inp["kcache"].shape, bf16),
                                               "vcache": (inp["vcache"].shape, bf16)}, bench=int(os.environ.get("BENCH", "0")))
    # reference
    f32 = lambda a: np.asarray(a, np.float32)
    h = inp["h_in"].T.reshape(-1).copy()
    cos, sin = inp["cos"], inp["sin"]
    ok = True
    for l in range(L):
        W = {k: f32(v) for k, v in Ws[l].items()}
        h, kn_, vn_ = layer_ref(h, W, f32(inp["kcache"][l]), f32(inp["vcache"][l]), pos, cos, sin)
        kerr = np.abs(f32(r["kcache"][l][:, :, pos]) - kn_).max(); verr = np.abs(f32(r["vcache"][l][pos]) - vn_).max()
        print(f"layer {l}: k write max err {kerr:.3g} (|k| {np.abs(kn_).max():.3g}), v write max err {verr:.3g} (|v| {np.abs(vn_).max():.3g})")
        ok &= kerr <= 0.02 * np.abs(kn_).max() + 1e-6 and verr <= 0.02 * np.abs(vn_).max() + 1e-6
    o = r["output_0"].T.reshape(-1)
    err = np.abs(o - h).max(); rel = err / np.abs(h).max()
    print(f"h_out max abs err {err:.4g}, |ref| max {np.abs(h).max():.4g}, rel {rel:.3g}; cos sim {np.dot(o, h) / np.linalg.norm(o) / np.linalg.norm(h):.7f}")
    ok &= rel < 0.01
    for l in so.splitlines():
        if "BENCH" in l or "nrt_load" in l: print(l)
    print("PASS" if ok else "FAIL")
