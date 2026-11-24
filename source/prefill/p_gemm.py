"""bf16 GEMM on TensorE at prefill shapes, feature-major ("transposed") activations:
  Y^T [N, S] = W [N, K] @ X^T [K, S]; stationary = W^T tile [128 k, 128 n], moving = X^T [128 k, 512 tokens],
  PSUM [128 n, 512 tokens] fp32, accumulated over K tiles, written back bf16. Inputs: xT [128, KT, S] (x^T in
  column layout: xT[p, kt, s] = x[s, kt*128 + p]), w = ws_layout(W) [128, NT*KT*128]. LNC=2 splits N.
Loop orders: o0 = for nt: for s-chunk: for kt (stationary changes every matmul)
             o1 = for nt: for kt: for s-chunk  (one stationary reused for S/512 matmuls; S/512 PSUM banks live)"""
import sys, os, numpy as np
import nki, nki.language as nl, nki.isa as nisa
from nkibuild import build, bf16, nrun, LAB
from qwen3_weights import ws_layout


def _gemm(xT, w, N, order):
    KT, S = xT.shape[1], xT.shape[2]
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    NT = N // 128
    NTL = NT // nprog
    SC = S // 512
    if S < 512:
        SC = 1
    CW = S // SC
    x = nl.ndarray((128, KT, S), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kt in range(KT):
        nisa.dma_copy(dst=x[:, kt, :], src=xT[:, kt, :])
    yT = nl.ndarray((N, S), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i in range(NTL):
        nt = pid * NTL + i
        wt = nl.ndarray((128, KT * 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w[:, nl.ds(nt * KT * 128, KT * 128)])
        yo = nl.ndarray((128, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        if order == 0:
            for c in range(SC):
                ps = nl.ndarray((128, CW), dtype=nl.float32, buffer=nl.psum)
                for kt in range(KT):
                    nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=x[:, kt, c * CW:(c + 1) * CW])
                nisa.tensor_copy(dst=yo[:, c * CW:(c + 1) * CW], src=ps)
        else:
            pss = []
            for c in range(SC):
                pss.append(nl.ndarray((128, CW), dtype=nl.float32, buffer=nl.psum))
            for kt in range(KT):
                for c in range(SC):
                    nisa.nc_matmul(dst=pss[c], stationary=wt[:, kt * 128:(kt + 1) * 128], moving=x[:, kt, c * CW:(c + 1) * CW])
            for c in range(SC):
                nisa.tensor_copy(dst=yo[:, c * CW:(c + 1) * CW], src=pss[c])
        nisa.dma_copy(dst=yT[nl.ds(nt * 128, 128), :], src=yo)
    return yT


@nki.jit
def gemm_o0(xT, w, N): return _gemm(xT, w, N, 0)
@nki.jit
def gemm_o1(xT, w, N): return _gemm(xT, w, N, 1)


if __name__ == "__main__":
    lnc = int(os.environ.get("LNC", "2"))
    order = int(os.environ.get("ORDER", "0"))
    shapes = [(1024, 4096), (1024, 6144), (3072, 1024), (2048, 1024)]
    if os.environ.get("MODEL8B"):    # Qwen3-8B: QKV, gate/up (per half), o_proj, down
        shapes = [(4096, 6144), (4096, 12288), (4096, 4096), (12288, 4096)]
    for S in [int(a) for a in sys.argv[1:]]:
        for K, N in shapes:
            rng = np.random.default_rng(0)
            X = rng.standard_normal((S, K)).astype(bf16)
            W = (rng.standard_normal((N, K)) * 0.03).astype(bf16)
            xT = np.ascontiguousarray(X.T.reshape(K // 128, 128, S).transpose(1, 0, 2))
            w = ws_layout(W)
            nm = f"gemm_o{order}_S{S}_K{K}_N{N}_lnc{lnc}"
            _, ct = build(gemm_o0 if order == 0 else gemm_o1, nm, xT, w, lnc=lnc, kw=dict(N=N))
            r, so = nrun(f"{LAB}/neff/{nm}.neff", {"xT": xT, "w": w}, {"output_0": ((N, S), bf16)}, bench=20)
            ref = W.astype(np.float32) @ X.astype(np.float32).T
            err = np.abs(r["output_0"].astype(np.float32) - ref).max() / np.abs(ref).max()
            us = float([l for l in so.splitlines() if "BENCH queued" in l][0].split()[2])
            print(f"{nm}: rel err {err:.2g}; queued {us:.1f} us -> {2.0 * S * K * N / us / 1e6:.1f} TFLOP/s", flush=True)
