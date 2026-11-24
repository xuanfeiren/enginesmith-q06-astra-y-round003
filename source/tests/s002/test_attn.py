import sys, os, subprocess, tempfile, numpy as np, ml_dtypes
S = int(sys.argv[1]); neff = sys.argv[2]; core = sys.argv[3] if len(sys.argv) > 3 else "42"
bf = ml_dtypes.bfloat16
rng = np.random.default_rng(0)
q = (rng.standard_normal((16, S, 128)) * 1.5).astype(bf)
k = (rng.standard_normal((8, S, 128)) * 1.5).astype(bf)
v = rng.standard_normal((8, S, 128)).astype(bf)
d = tempfile.mkdtemp(dir=".")
np.zeros((1, 1), np.int32).tofile(f"{d}/pos.bin")
np.ascontiguousarray(q.transpose(0, 2, 1)).tofile(f"{d}/qT.bin")
np.ascontiguousarray(k.transpose(0, 2, 1)).tofile(f"{d}/kT.bin")
v.tofile(f"{d}/v.bin")
(np.ones((16, 128, S)) * 7).astype(bf).tofile(f"{d}/oT.bin")
r = subprocess.run(["./nrun", neff, d, "1", "sync", "0"], env={**os.environ, "NEURON_RT_VISIBLE_CORES": core}, capture_output=True, text=True)
if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit(1)
o = np.fromfile(f"{d}/out_oT.bin", dtype=bf).reshape(16, 128, S).astype(np.float32)
qf, kf, vf = q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)
err = 0
for h in range(16):
    s = qf[h] @ kf[h // 2].T * 0.08838834764831845
    s = s + np.triu(np.full((S, S), -1e30), 1)
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    ref = (p @ vf[h // 2]).T
    e = np.abs(o[h] - ref).max(); err = max(err, e)
    if h in (0, 5, 15): print(h, "maxerr", e, "refmax", np.abs(ref).max())
print("overall max abs err", err)
import shutil; shutil.rmtree(d)
