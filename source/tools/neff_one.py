"""Build time only: compile one engine kernel to a NEFF through NKI's documented route.

NKI_ARTIFACTS_DIR is set (by build_neffs.py) before nki is imported; calling the @nki.jit kernel with numpy arrays then
compiles it, runs it once on a NeuronCore (NEURON_RT_VISIBLE_CORES names a free one) and leaves kernel.neff there.
The inputs are zeros of the shapes the engine binds at run time (only the shapes matter for the NEFF).
usage: python neff_one.py <kind> <arg>      kind = step (arg = attention bucket) | pf_emb | pf_layer | pf_kv | pf_tail | pf_sb (arg = S)
"""
import os
import sys

assert os.environ.get("NKI_ARTIFACTS_DIR"), "run through build_neffs.py"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kernels"))
sys.path.insert(0, os.path.join(HERE, "..", "third_party"))   # the vendored nkilib copy (THIRD_PARTY.md)
import numpy as np  # noqa: E402
import ml_dtypes  # noqa: E402

bf16 = ml_dtypes.bfloat16
L, TMAX, V, H = 28, 4608, 151936, 1024
i32, f32 = np.int32, np.float32


def common():
    return dict(
        toks=np.zeros((TMAX + 1, 1), i32), embed=np.zeros((V, H), bf16),
        ln1=np.zeros((L, 128, 8), f32), ln2=np.zeros((L, 128, 8), f32),
        qn=np.zeros((L, 128, 1), f32), kn=np.zeros((L, 128, 1), f32),
        wqkv=np.zeros((L, 2, 128, 16 * 8 * 128), bf16), wgu=np.zeros((L, 2, 128, 24 * 8 * 128), bf16),
        normf=np.zeros((128, 8), f32), wlm=np.zeros((2, 128, 594 * 8 * 128), bf16),
        rot=np.zeros((128, 128), bf16),
        kcache=np.zeros((L, 2, 4, 128, TMAX), bf16), vcache=np.zeros((L, 2, TMAX, 512), bf16))


kind, arg = sys.argv[1], int(sys.argv[2])
if kind == "step":
    import step_lnc2 as K
    c = common()
    args = [np.zeros((1, 1), i32), np.zeros((1, 1), i32), np.zeros((TMAX, 1), i32), c["toks"], c["embed"],
            c["ln1"], c["ln2"], c["qn"], c["kn"], c["wqkv"], np.zeros((L, 2, 128, 8 * 8 * 128), bf16), c["wgu"],
            np.zeros((L, 2, 128, 8 * 12 * 128), bf16), c["normf"], c["wlm"], c["rot"],
            np.zeros((TMAX, 64), f32), np.zeros((TMAX, 64), f32), c["kcache"], c["vcache"],
            c["kcache"].copy(), c["vcache"].copy()]   # kcw / vcw (aliases of the caches at run time)
    getattr(K, f"step2_b{arg}")[2](*args)
elif kind == "pf_sb" or kind == "pf_sbc":
    import prefill_sb as K
    c = common()
    S = arg
    one = lambda a: np.ascontiguousarray(a[0:1])
    args = [np.ones((1, 1), i32), np.zeros((128, 8, S), f32), one(c["ln1"]), one(c["ln2"]), one(c["qn"]), one(c["kn"]),
            one(c["wqkv"]), np.zeros((1, 2, 128, 8 * 8 * 128), bf16), one(c["wgu"]),
            np.zeros((1, 2, 128, 8 * 12 * 128), bf16), c["rot"], np.zeros((128, TMAX), f32),
            np.zeros((128, TMAX), f32), one(c["kcache"]), one(c["vcache"])]
    getattr(K, kind)[2](*args)
elif kind == "pf_tail":
    import step_lnc2 as K
    c = common()
    S = arg
    args = [np.ones((1, 1), i32), c["toks"], np.zeros((128, 8, S), f32), c["ln1"], c["ln2"], c["qn"], c["kn"],
            c["wqkv"], np.zeros((L, 2, 128, 8 * 8 * 128), bf16), c["wgu"], np.zeros((L, 2, 128, 8 * 12 * 128), bf16),
            c["normf"], c["wlm"], c["rot"], np.zeros((TMAX, 64), f32), np.zeros((TMAX, 64), f32), c["kcache"],
            c["vcache"], c["kcache"].copy(), c["vcache"].copy()]
    K.pf_tail[2](*args)
else:
    import prefill_lnc2 as K
    import nkilib
    assert os.path.realpath(nkilib.__file__).startswith(os.path.realpath(os.path.join(HERE, "..", "third_party"))), \
        f"nkilib must be the vendored copy, not {nkilib.__file__}"
    S = arg
    c = common()
    d = dict(ids=np.zeros((S, 1), i32), plen=np.ones((1, 1), i32), **c,
             wo_p=np.zeros((L, 2, 128, 4 * 16 * 128), bf16), wd_p=np.zeros((L, 2, 128, 4 * 24 * 128), bf16),
             cosT=np.zeros((128, TMAX), f32), sinT=np.zeros((128, TMAX), f32),
             s_h=np.zeros((128, 8, S), f32), s_q=np.zeros((16, 128, S), bf16), s_k=np.zeros((8, 128, S), bf16),
             s_v=np.zeros((8, S, 128), bf16), s_o=np.zeros((16, 128, S), bf16), s_a=np.zeros((24, 128, S), bf16))
    if kind == "pf_layer" or kind == "pf_kv":
        for k in ("ln1", "ln2", "qn", "kn", "wqkv", "wo_p", "wgu", "wd_p", "kcache", "vcache"):
            d[k] = np.ascontiguousarray(d[k][0:1])
    order = ["ids", "plen", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo_p", "wgu", "wd_p", "normf", "wlm",
             "rot", "cosT", "sinT", "kcache", "vcache", "s_h", "s_q", "s_k", "s_v", "s_o", "s_a"]
    getattr(K, kind)[2](*[d[k] for k in order])
print("built", kind, arg, flush=True)
