# GEMV bandwidth and weight quantization

Purpose: how to stream weights at the HBM rate in a batch-1 decode GEMV, which knobs matter and which do not, and
what fp8 weights buy. Measured on the trn2.48xlarge of `README.md`, queued executions from C++ (`host/nrun`,
`host/chain`), logical cores 56-59; code in the NKI lab (`probes/p6_pe_rate.py`, `p7_dma_bw.py`,
`p11_gemv_layout.py`, `p12_fp8_gemv.py`, `kernels/step_g.py`, FINDINGS.md sections 2 and 4).

## Ceilings

| what | LNC=1 (1 physical core) | LNC=2 (logical core) |
|---|---|---|
| pure HBM -> SBUF stream, 16.8 / 33.6 MB | 258-275 GB/s | 446-466 GB/s |
| same, 67 / 134 MB (fixed cost amortised) | 374 GB/s | 535 GB/s |
| weight-stationary GEMV, 1.24 GB (8B LM head shape) | - | **661 GB/s** |
| GEMV K = 1,024, 1.24 GB | - | 674 GB/s |
| whole 0.6B decode step (1.25 GB) | 360 GB/s | 561 GB/s |
| whole 8B decode step (15.2 GB) | - | 637 GB/s |

Use LNC=2 (one program per physical core, split every weight in two contiguous halves): it is the only way to
reach ~650 GB/s. Floors: 0.6B bf16 ~1.9 ms/step, 8B bf16 ~22.7 ms/step.

## What matters

1. **Weight-stationary with a one-column moving x** (`kernels-fusion.md`): TensorE pipelines one 128x128 tile per
   ~30 ns (LDWEIGHTS 105 ns + MATMUL 165-187 ns each, overlapped), enough for ~1 TB/s per logical core. In a
   profile, a start-to-start gap of 70 ns with matmul pacing at 120-200 ns per tile per core means TensorE is
   starved or stalled, not slow.
2. **PSUM accumulation depth <= ~8k matmuls per PSUM tile.** Output columns x K tiles accumulated into one PSUM
   tile before it is read: 512 x 32 = 16k -> 176 GB/s; 592 tiles with 512-wide PSUM tiles -> 292 GB/s; the same
   with 256- or 128-wide tiles -> 661 GB/s; 512 x 8 = 4k -> 674 GB/s. This was the whole difference between a
   4.89 and a 2.52 ms 8B (1 layer + LM head) step.
3. **1 MB weight DMAs** (`[128 partitions, 8 KB]`): [128 x 2048] slabs instead of [128 x 512] were the big win in
   the backend example (123.6 -> 81.9 us).

## What does not matter (measured negatives)

- HBM layout of the weight: rows with a 4.8 MB partition stride vs fully contiguous 1 MB blocks: 292 vs 292 GB/s
  (same kernel, both limited by the PSUM cliff above); 1 vs 2 n-tiles per DMA: 623 vs 608 GB/s (fp8).
- Hardware DGE for the weight stream: slower (whole 0.6B step 2.378 vs 2.235 ms) than the default software DGE.
- Pure DMA chunk size 2-16 KB per partition: 258-275 GB/s at LNC=1, 446-466 at LNC=2.

## fp8 weights (weight-only, per-output-channel scale)

`nc_matmul` accepts a `float8_e4m3` stationary with a bf16 moving operand (fp32 accumulation). Store
`W8 = W / s` with `s[n] = max_k |W[n, k]| / 240` (legacy Trn2 e4m3, max 240; `ml_dtypes.float8_e4m3`), multiply
the PSUM column by `s` once per GEMV (a `[128, NT]` scale tile in column layout; for K-split GEMVs before the
cross-core exchange).

| | bf16 | fp8 |
|---|---|---|
| 8B LM-head GEMV (1.24 GB bf16), LNC=2 | 1.88 ms (661 GB/s) | **1.00 ms** (623 GB/s of fp8 = 1.25 TB/s bf16-equivalent) |
| kernel arithmetic vs dequantised weights | - | 6e-7 relative |
| GEMV output vs bf16 weights (random weights) | - | 2.65% rms |
| whole 0.6B decode step (b512) | 2.228 ms | **1.353 ms** |
| 0.6B teacher-forced argmax vs HF fp32, 479 positions | 478/479 | **447/479** (mismatch logit gap median 0.12, max 0.66) |
| whole 8B decode step (b512, 36 layers) | 23.86 ms | **13.84 ms** |
| 8B teacher-forced argmax vs HF fp32, 199 positions | 197/199 | **184/199** (gap median 0.197) |

fp8 weight-only is 1.65x (0.6B) / 1.72x (8B) faster but costs ~6.5 points of top-1 agreement with the fp32 model: it changes
the model, not just the rounding. Use it only where the task's correctness gates allow it, measure agreement on
long teacher-forced text (not on a few greedy tokens: the first 24 generated tokens were still identical), and
keep bf16 as the default. `double_row` fp8 performance mode needs both operands in fp8 (activation quantisation
as well) and was not measured.
