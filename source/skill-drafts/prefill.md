# Prefill on Trainium2

Purpose: how to run the prompt through the model on the device as one (or a few) NEFFs that also write the KV cache
in the decode layout and produce the first token, and what limits its speed. Measured on the trn2.48xlarge of
`README.md`, Qwen3-0.6B, LNC=2, queued from C++; code in the NKI lab (`kernels/prefill_lnc2.py`,
`host/generate.cpp`, `prefill/`, FINDINGS.md section 3).

| prompt bucket | ours (whole model, incl. LM head + argmax) | reference | TFLOP/s (GEMM + causal attention) |
|---|---|---|---|
| 256 | 7.5 ms | 11 ms | 31 |
| 1,024 | 19.8 ms | 36 ms | 52 |
| 4,096 | 76.3 ms | 154 ms | 73 |

First token = HF fp32 argmax; prefill + 31 decode steps reproduce HF fp32 greedy (32/32 at a 4,093-token prompt).

## What to do

1. **Keep activations feature-major** (`h^T [128, 8, S]` fp32, the decode column layout with a token axis). Every
   GEMM: weight tile `[128 k, 128 n]` stationary, activation chunk `[128 k, 512 tokens]` moving, PSUM `[128 n, 512]`
   fp32. Outputs are again feature-major, so no activation transposes between GEMMs; q^T / k^T come out in the
   layouts nkilib's attention takes (`tp_q=False`, `tp_k=False`) and k^T *is* the decode K^T cache layout. V
   needs a PE transpose per 128x128 block (matmul with the identity, exact) for the `[T, 512]` V cache.
   GEMM rate at S = 4,096 (LNC=2, queued incl. ~20 us fixed): 96-130 TFLOP/s (`prefill/p_gemm.py`).
2. **Per layer, phases over 512-token chunks with that layer's weights resident in SBUF** (each core's half: QKV
   32 KB, o_proj 16 KB, gate/up 48 KB, down 24 KB per partition), `nisa.core_barrier` between phases:
   A norm + QKV + q/k norm + RoPE + cache writes; B attention (nkilib `_attention_cte` inlined, `attention.md`);
   C o_proj + residual; D norm + gate/up + SiLU*up; E down + residual. o_proj and down are **N-split** in prefill
   (each core writes its own 4 of the 8 hidden tiles of h, in place) — keep N-split copies of those two weights
   (+224 MB for 0.6B) and share everything else with the decode NEFF.
3. **LM head only on the last real token**: read column `plen - 1` of h with a register offset, reuse the decode
   LM head + argmax, write `toks[plen]` and `output_0 = plen`: the decode chain starts right after it with no host
   work.
4. **Long prompts: one 1-layer NEFF executed per layer.** Unrolling 28 layers x 8 chunks x the attention kernel in
   Python made a 190 MB MLIR at S = 1,024 (162 s compile, 8.2 MB NEFF) and did not finish tracing in 10 minutes at
   S = 4,096. Build three NEFFs (embedding, one layer, LM head); bind the layer NEFF's per-layer inputs to slices
   of the stacked tensors with `nrt_tensor_allocate_slice(src, l * size, size, name, &slice)` and queue all 30
   executions: S = 4,096 builds in 37 s (1.8 MB), and the per-execution fixed cost is noise (7.51 vs 7.30 ms
   at S = 256, 22.3 vs 21.8 ms at S = 1,024 before the phase-A changes).

## What limited it (S = 4,096 layer NEFF, neuron-explorer, 100 us buckets of engine busy time)

- Before: 3.11 ms per layer; phase A 0.95 ms with VectorE ~83% busy (8 fp32 elementwise passes per head per
  chunk for the q/k norm and RoPE), attention ~1.2 ms, the other GEMM phases ~0.8 ms.
- Removing VectorE passes (squares on ScalarE straight from PSUM, `activation(op=nl.square)`; sums of squares as
  bf16 x ones matmuls into fp32 PSUM; norm scale x weight in one `scalar_tensor_tensor`): **no gain** — GpSimd
  became the bottleneck (~95% busy).
- GpSimd was issuing the ~48 small DMAs per chunk (default `dma_copy` is software DGE run by GpSimd). Moving the
  activation DMAs to hardware DGE (`dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync`): **88.8 -> 76.3 ms**
  (S = 4,096), 22.3 -> 19.8 ms (S = 1,024). Lesson: look at all four engines in the profile; DMA issue cost lands
  on GpSimd.
- Remaining at S = 4,096: attention ~45% of the time, GEMMs at ~100 TFLOP/s; at S = 256 the time is dominated by
  per-phase latency (barriers, small chunks), not FLOPs.

## Pitfalls

- Cross-core scratch in **internal** `nl.shared_hbm` tensors was private per core in this kernel (each core saw
  only its own writes, deterministic, with `core_barrier`); scratch passed in as kernel inputs works. The KV cache
  (an input) is shared correctly.
- Check K/V against the reference model's `past_key_values` per layer and compare with the reference's own bf16
  error on the same metric: single positions of middle layers reach 15% relative error (HF bf16: 40%) while
  tokens still match.
- `scale=` of nkilib's attention must be passed as a kernel kwarg (`build(..., kw=...)`), not via a wrapper jit.
