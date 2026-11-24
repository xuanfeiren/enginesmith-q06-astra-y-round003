"""Build time only: compile every NEFF the engine embeds, through NKI's documented route (tools/neff_one.py, one
process and one artifacts directory per kernel), into build/neff/<name>.neff.

NEFFs are cached in target/neff-cache/ keyed by the kernel sources and the job, so a rebuild of an unchanged kernel is
a copy. Kernels run once on a NeuronCore while they are built: one worker per core in NEURON_BUILD_CORES (default:
ENGINESMITH_NEURON_CORE + 2 and + 3).
"""
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STEP_BUCKETS = [384, 512, 1024, 1152, 1536, 2048, 3072, 4224, 4608]
PREFILL_BUCKETS = [256, 1024, 4096]
SB_MAX = 256   # prompt buckets up to this size run their layers as pf_sb (kernels/prefill_sb.py)
SBC_MAX = 1024  # ... up to this size as pf_sbc (the chunked variant); larger ones as pf_layer (prefill_lnc2.py)
JOBS = [(f"step_b{b}", "step", b) for b in STEP_BUCKETS] + \
       [(f"{k}_S{s}", k, s) for s in reversed(PREFILL_BUCKETS)
        for k in (("pf_sb" if s <= SB_MAX else "pf_sbc" if s <= SBC_MAX else "pf_layer"), "pf_kv", "pf_tail",
                  "pf_emb")]


def source_hash():
    h = hashlib.sha256()
    for d in ("kernels", "tools", "third_party"):
        for root, _, files in sorted(os.walk(os.path.join(ROOT, d))):
            for fn in sorted(files):
                if fn.endswith(".py"):
                    h.update(fn.encode())
                    h.update(open(os.path.join(root, fn), "rb").read())
    return h.hexdigest()[:16]


def main():
    base = int(os.environ.get("ENGINESMITH_NEURON_CORE", "0"))
    cores = os.environ.get("NEURON_BUILD_CORES", f"{base + 2},{base + 3}").split(",")
    out = os.path.join(ROOT, "build", "neff")
    cache = os.path.join(ROOT, "target", "neff-cache")
    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    key = source_hash()
    todo = []
    for name, kind, arg in JOBS:
        c = os.path.join(cache, f"{name}-{key}.neff")
        if os.path.exists(c):
            shutil.copy(c, os.path.join(out, f"{name}.neff"))
        else:
            todo.append((name, kind, arg, c))
    print(f"neffs: {len(JOBS) - len(todo)} cached, {len(todo)} to build on cores {cores}", flush=True)
    lock = threading.Lock()
    errors = []

    def worker(core):
        while True:
            with lock:
                if not todo or errors:
                    return
                name, kind, arg, c = todo.pop(0)
            art = os.path.join(ROOT, "build", "art", name)
            shutil.rmtree(art, ignore_errors=True)
            os.makedirs(art)
            env = {**os.environ, "NKI_ARTIFACTS_DIR": art, "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
                   "NEURON_RT_VISIBLE_CORES": core}
            t0 = time.time()
            with open(os.path.join(art, "build.log"), "w") as log:
                r = subprocess.run([sys.executable, os.path.join(HERE, "neff_one.py"), kind, str(arg)], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
            neff = os.path.join(art, "kernel.neff")
            if r.returncode != 0 or not os.path.exists(neff):
                with lock:
                    errors.append(name)
                print(f"FAILED {name} (exit {r.returncode}); log tail:", flush=True)
                print(open(os.path.join(art, "build.log")).read()[-4000:], flush=True)
                return
            shutil.copy(neff, c + ".tmp")
            os.replace(c + ".tmp", c)
            shutil.copy(neff, os.path.join(out, f"{name}.neff"))
            print(f"built {name} on core {core} in {time.time() - t0:.0f} s ({os.path.getsize(neff) / 1e6:.2f} MB)",
                  flush=True)
            shutil.rmtree(art, ignore_errors=True)

    threads = [threading.Thread(target=worker, args=(c,)) for c in cores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        sys.exit(f"NEFF build failed: {errors}")
    # remove stale cache entries
    for fn in os.listdir(cache):
        if fn.endswith(".neff") and not fn.endswith(f"-{key}.neff"):
            os.remove(os.path.join(cache, fn))
    with open(os.path.join(ROOT, "build", "neffs.list"), "w") as f:
        for name, _, _ in JOBS:
            f.write(name + "\n")


if __name__ == "__main__":
    main()
