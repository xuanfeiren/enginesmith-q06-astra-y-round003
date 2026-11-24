"""Qwen3-8B decode step NEFF (kernels/step_g.py, LNC=2) with L layers: compile time, NEFF size, queued ms/step from
host/chain with zero weights (DMA/compute time does not depend on the values). usage: bench_8b.py <kernel> <L>..."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
from nkibuild import build, bf16, chain, LAB
import step_g as SG

TMAX = 4608
HT, NQL, NKL, ITL = 32, 16, 4, 48


def shapes(L):
    z = lambda *s, dt=bf16: np.zeros(s, dt)
    return dict(pos=z(1, 1, dt=np.int32), plen=np.array([[1]], np.int32), prompt=z(TMAX, 1, dt=np.int32),
                toks=z(TMAX + 1, 1, dt=np.int32), embed=z(151936, HT * 128),
                ln1=z(L, 128, HT, dt=np.float32), ln2=z(L, 128, HT, dt=np.float32),
                qn=z(L, 128, 1, dt=np.float32), kn=z(L, 128, 1, dt=np.float32),
                wqkv=z(L, 2, 128, (NQL + 2 * NKL) * HT * 128), wo=z(L, 2, 128, HT * NQL * 128),
                wgu=z(L, 2, 128, 2 * ITL * HT * 128), wd=z(L, 2, 128, HT * ITL * 128),
                normf=z(128, HT, dt=np.float32), wlm=z(2, 128, 594 * HT * 128), rot=z(128, 128),
                cos=z(TMAX, 64, dt=np.float32), sin=z(TMAX, 64, dt=np.float32),
                kcache=z(L, 2, NKL, 128, TMAX), vcache=z(L, 2, TMAX, NKL * 128))


ORDER = ["pos", "plen", "prompt", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "normf",
         "wlm", "rot", "cos", "sin", "kcache", "vcache"]

if __name__ == "__main__":
    name = sys.argv[1]
    for L in [int(a) for a in sys.argv[2:]]:
        inp = shapes(L)
        nm = f"{name}_L{L}"
        if not os.environ.get("NOBUILD"):
            _, ct = build(getattr(SG, name), nm, *[inp[k] for k in ORDER], lnc=2)
        else:
            ct = float("nan")
        small = {k: v for k, v in inp.items() if v.nbytes < (1 << 20)}
        r, so = chain(f"{LAB}/neff/{nm}.neff", small, {}, 20, "pos:output_0", reps=3)
        us = [float(l.split("=")[1].split()[0]) for l in so.splitlines() if l.startswith("CHAIN")]
        wbytes = sum(inp[k].nbytes for k in ("wqkv", "wo", "wgu", "wd", "wlm")) + 2 * inp["kcache"][:, :, :, :, :512].nbytes
        print(f"{nm}: compile {ct:.1f}s, NEFF {os.path.getsize(f'{LAB}/neff/{nm}.neff')/1e6:.2f} MB, "
              f"queued {min(us)/1e3:.3f} ms/step; weights+cache(b512) streamed {wbytes/1e9:.2f} GB -> "
              f"{wbytes / min(us) / 1e3:.0f} GB/s", flush=True)
        for l in so.splitlines():
            if "nrt_load" in l or "upload" in l: print("   ", l)
