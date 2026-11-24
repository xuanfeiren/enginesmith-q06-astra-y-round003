import numpy as np, os, subprocess, tempfile, sys, ml_dtypes
bf = ml_dtypes.bfloat16
S = int(sys.argv[1]); neff_a = sys.argv[2]; neff_b = sys.argv[3]
TMAX = 4608
rng = np.random.default_rng(0)
def ws(W, nts, kts):
    N, K = W.shape
    out = np.zeros((128, len(nts) * len(kts) * 128), W.dtype)
    for i, nt in enumerate(nts):
        for k, kt in enumerate(kts):
            blk = W[nt*128:(nt+1)*128, kt*128:(kt+1)*128]  # [j, p]
            out[:, (i*len(kts)+k)*128:(i*len(kts)+k+1)*128] = blk.T
    return out
def col(v): return np.ascontiguousarray(v.reshape(-1, 128).T)
H = 1024
Wq = (rng.standard_normal((4096, H)) * 0.03).astype(bf)
Wo = (rng.standard_normal((H, 2048)) * 0.03).astype(bf)
GU = (rng.standard_normal((6144, H)) * 0.03).astype(bf)
Wd = (rng.standard_normal((H, 3072)) * 0.03).astype(bf)
r = lambda a, b: list(range(a, b))
d = {}
d['wqkv'] = np.stack([ws(Wq, r(8*c, 8*c+8) + r(16+4*c, 20+4*c) + r(24+4*c, 28+4*c), r(0, 8)) for c in range(2)])[None]
d['wo'] = np.stack([ws(Wo, r(0, 8), r(8*c, 8*c+8)) for c in range(2)])[None]
d['wgu'] = np.stack([ws(GU, r(12*c, 12*c+12) + r(24+12*c, 36+12*c), r(0, 8)) for c in range(2)])[None]
d['wd'] = np.stack([ws(Wd, r(0, 8), r(12*c, 12*c+12)) for c in range(2)])[None]
d['wo_p'] = np.stack([ws(Wo, r(4*c, 4*c+4), r(0, 16)) for c in range(2)])[None]
d['wd_p'] = np.stack([ws(Wd, r(4*c, 4*c+4), r(0, 24)) for c in range(2)])[None]
d['ln1'] = col((1 + 0.2 * rng.standard_normal(H)).astype(np.float32))[None]
d['ln2'] = col((1 + 0.2 * rng.standard_normal(H)).astype(np.float32))[None]
d['qn'] = (1 + 0.2 * rng.standard_normal((1, 128, 1))).astype(np.float32)
d['kn'] = (1 + 0.2 * rng.standard_normal((1, 128, 1))).astype(np.float32)
rot = np.zeros((128, 128), np.float32)
for m in range(64): rot[(m+64), m] = -1; rot[m, m+64] = 1
d['rot'] = rot.astype(bf)
inv = 1.0 / (1e6 ** (np.arange(64) * 2 / 128.0)); t = np.arange(TMAX)[None, :] * inv[:, None]
d['cosT'] = np.concatenate([np.cos(t), np.cos(t)]).astype(np.float32); d['sinT'] = np.concatenate([np.sin(t), np.sin(t)]).astype(np.float32)
h = rng.standard_normal((S, H)).astype(np.float32)
d['s_h'] = np.ascontiguousarray(h.T.reshape(8, 128, S).transpose(1, 0, 2))
d['plen'] = np.array([[S]], np.int32)
def run(neff):
    dd = tempfile.mkdtemp(dir='.')
    for k, v in d.items(): np.ascontiguousarray(v).tofile(f"{dd}/{k}.bin")
    rr = subprocess.run(["./nrun", neff, dd, "1", "sync", "0"], env={**os.environ, "NEURON_RT_VISIBLE_CORES": "42"}, capture_output=True, text=True)
    if rr.returncode: print(rr.stdout[-1500:], rr.stderr[-1500:]); sys.exit(1)
    o = {k: np.fromfile(f"{dd}/out_{k}.bin", dt) for k, dt in (("s_h", np.float32), ("kcache", bf), ("vcache", bf))}
    import shutil; shutil.rmtree(dd); return o
a = run(neff_a); b = run(neff_b)
ha, hb = a['s_h'], b['s_h']
print("s_h max abs diff", np.abs(ha - hb).max(), "max |h|", np.abs(ha).max(), "rel", np.abs(ha - hb).max() / np.abs(ha).max(), "changed from input", np.abs(ha - d['s_h'].ravel()).max())
for k in ("kcache", "vcache"):
    x, y = a[k].astype(np.float32), b[k].astype(np.float32)
    print(k, "max abs diff", np.abs(x - y).max(), "max", np.abs(x).max(), "nonzero", (x != 0).sum(), (y != 0).sum())
