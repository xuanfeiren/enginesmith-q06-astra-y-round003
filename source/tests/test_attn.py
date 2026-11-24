"""Self-check of kernels/attn_decode.py against ref_qwen3.attn_decode_ref, via host/nrun (libnrt, C++).
usage: python test_attn.py [bucket=512] [positions...]"""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "kernels"))
from nkibuild import build, bf16, nrun, LAB
import attn_decode as A
from ref_qwen3 import rope_tables, attn_decode_ref

TMAX = 4608


def make_inputs(rng, pos):
    qkv = (rng.standard_normal((1, 4096)) * 2).astype(bf16)
    qn_w = (1 + 0.3 * rng.standard_normal((1, 128))).astype(bf16)
    kn_w = (1 + 0.3 * rng.standard_normal((1, 128))).astype(bf16)
    cos, sin = rope_tables(TMAX)
    kcache = (rng.standard_normal((8, 128, TMAX))).astype(bf16)
    vcache = (rng.standard_normal((TMAX, 1024))).astype(bf16)
    kcache[:, :, pos:] = 50.0   # stale garbage beyond pos must be masked out
    vcache[pos:] = 1000.0
    return dict(qkv=qkv, qn_w=qn_w, kn_w=kn_w, cos=cos, sin=sin, kcache=kcache, vcache=vcache,
                pos=np.array([[pos]], np.int32))


def check(name, neff, inp, pos):
    r, so = nrun(neff, inp, {"output_0": ((1, 2048), bf16), "kcache": ((8, 128, TMAX), bf16), "vcache": ((TMAX, 1024), bf16)})
    ref, kref, vref = attn_decode_ref(inp["qkv"], inp["qn_w"], inp["kn_w"], inp["cos"], inp["sin"], inp["kcache"], inp["vcache"], pos)
    o = r["output_0"].astype(np.float32).reshape(-1)
    err = np.abs(o - ref).max()
    tol = np.abs(ref).max() * 2 ** -7 + 1e-3     # a few bf16 ulps
    kw = r["kcache"][:, :, pos].astype(np.float32); vw = r["vcache"][pos].astype(np.float32)
    kerr = np.abs(kw - kref).max(); verr = np.abs(vw - vref.astype(bf16).astype(np.float32)).max()
    kc = r["kcache"].copy(); kc[:, :, pos] = inp["kcache"][:, :, pos]
    vc = r["vcache"].copy(); vc[pos] = inp["vcache"][pos]
    untouched = np.array_equal(kc.view(np.uint16), inp["kcache"].view(np.uint16)) and np.array_equal(vc.view(np.uint16), inp["vcache"].view(np.uint16))
    ok = err <= tol and kerr <= np.abs(kref).max() * 2 ** -7 and verr == 0 and untouched
    print(f"{name} pos={pos}: out max err {err:.4g} (tol {tol:.3g}, |ref| {np.abs(ref).max():.3g}); "
          f"k write err {kerr:.3g}; v write err {verr}; rest of cache untouched {untouched} => {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    a1 = sys.argv[1] if len(sys.argv) > 1 else "512"
    name = f"attn_b{a1}" if a1.isdigit() else a1          # bucket number or kernel name (e.g. attn_dyn4)
    b = int(a1) if a1.isdigit() else TMAX
    poss = [int(a) for a in sys.argv[2:]] or [0, 1, 127, 128, 300, b - 1]
    rng = np.random.default_rng(1)
    suf = __import__("os").environ.get("SUF", "")
    name += suf
    kern = getattr(A, name)
    inp = make_inputs(rng, 5)
    build(kern, f"{name}_lnc1", *[inp[k] for k in ["qkv", "qn_w", "kn_w", "cos", "sin", "kcache", "vcache", "pos"]], lnc=1)
    allok = True
    for p in poss:
        inp = make_inputs(rng, p)
        allok &= check(name, f"{LAB}/neff/{name}_lnc1.neff", inp, p)
    print("ALL PASS" if allok else "SOME FAILED")
