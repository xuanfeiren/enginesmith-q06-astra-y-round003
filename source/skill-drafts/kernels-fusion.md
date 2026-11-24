# Kernel fusion: the whole decode step in one NEFF

Purpose: how to put an entire decode step (embedding, every layer, LM head, argmax, KV write, position update) into
one NEFF so the host only queues executions, and how to use both physical cores of a logical core (LNC=2).
Measured on the trn2.48xlarge of `README.md`; code in the NKI lab (`kernels/step_lnc2.py`, `kernels/step_g.py`,
`host/chain.cpp`, `host/generate.cpp`, FINDINGS.md sections 2 and 4).

## Why

A queued execution costs at least ~20 us of fixed time (`README.md`); ten kernels per layer x 28 layers would
spend more than the reference's whole step on fixed costs. Inside one NEFF the compiler overlaps the weight
stream of layer l+1 with the compute of layer l; the whole step becomes one DMA-bound stream.

| Qwen3-0.6B decode, b512 attention bucket | ms/step (queued from C++) | device | MBU |
|---|---|---|---|
| reference (NxD Inference) | 6.26 | 6.42 ms | 38% |
| one NEFF, LNC=1 (`step_b512`) | 3.46 | 3.47 ms | 50% |
| **one NEFF, LNC=2 (`step2_b512`)** | **2.235** | 2.42 ms (profiling on) | 72% |
| Qwen3-8B, one NEFF, 36 layers, LNC=2 (`g8b_b512_s1`) | 23.86 | - | 637 GB/s |

Compile: 26-33 s and 1.5 MB for the 0.6B step, 325 s and 13.8 MB for the 8B step (36 layers, ~8 s per layer);
`nrt_load` 0.2 s / 2.4 s. Tokens identical to HF transformers fp32 greedy (40/40 for 0.6B, 24/24 for 8B).

## What to do

1. **One layout for every vector: "column layout"** `[128, K/128]` with `v[kt*128 + p]` at `[p, kt]`. Every GEMV
   is weight-stationary: tile (nt, kt) of W^T, `[128 k, 128 n]`, stationary, `x[:, kt]` (one column) moving, the
   result `y[:, nt]` lands in PSUM already in column layout. No transposes between ops anywhere in the layer.
   Host layout `ws_layout(W)[p, (nt*KT + kt)*128 + j] = W[nt*128 + j, kt*128 + p]`, streamed in slabs of
   1 MB (e.g. 4 n-tiles x 8 k-tiles x 128 x 2 B = 8 KB per partition). TensorE pipelines these at ~30 ns per
   tile; the x-stationary form (x `[128, 1]` stationary, 512-wide weight rows moving) took twice the TensorE
   time for the same bytes (probe p6: 30.4 vs 60.2 us for 2 MB).
2. **Norms with the PE**: sum of squares over the partition dim = `ones[128, 128]^T @ sq` (every partition gets the
   total); rsqrt on GpSimd (`tensor_scalar(op0=nl.rsqrt, engine=nisa.engine.gpsimd)`, more precise than the
   ScalarE LUT). RoPE rotate-half = a fixed permutation matrix as one matmul.
3. **Residual stream in fp32 SBUF** across all layers; activations rounded to bf16 only as matmul inputs. This
   kept a 4,000-token teacher-forced run closer to HF fp32 (3,972/3,999 argmax agreement) than HF's own bf16
   run (3,904/3,999).
4. **Argmax with ties to the lowest id, exact**: logits `lg[p, c]` (id = c*128 + p) fp32; per partition `m_p =
   max`, score `(lg == m_p) * (2^20 - id)`, `max` -> lowest id; PE-transpose both columns to one partition and
   repeat. Ids < 2^20 are exact in fp32; equality is only taken between values that went the same path.
5. **Device-side request loop**: input token = `prompt[pos]` if `pos < plen` else `toks[pos]`; output
   `toks[pos+1] = argmax` (dynamic DMA writes) and `output_0 = pos + 1`. Two tensor sets swapping `pos`/
   `output_0`; queue all steps with `nrta_execute_schedule`; one read of `toks` per request. Queued time = device
   time (3.46 vs 3.466 ms at LNC=1).
6. **Prefill + decode as one queue**: the prefill NEFF (`prefill.md`) shares every tensor of the same name with
   the decode NEFF (weights, caches, `toks`); its `output_0` (= plen) is the decode's `pos`. `host/generate.cpp`
   queues prefill + all decode steps at once: 7.4 ms prefill + 2.22 ms/step, 64/64 tokens identical to HF.

## LNC=2: two physical cores, one program

Both cores run the same program; `nl.program_id(0)` (0/1) works as an index (`w[l, pid]`), in arithmetic
(`1 - pid`, `pid * 4`, `nl.ds(pid * NT + i, ...)`) and as a `tensor_scalar` operand. Split each layer so that
each core streams one contiguous half of every weight (re-lay the weights per core on the host):

| op | split | exchange |
|---|---|---|
| QKV | by heads (q 8c..8c+7, kv 4c..4c+3) | none: attention is local |
| attention | local heads, own half of the cache | none |
| o_proj | K (the core's 8 heads) | partial `[128, 8]` fp32, sendrecv + add |
| gate/up | N (intermediate tiles 12c..12c+11) | none: SiLU*up local |
| down | K (the same 12 tiles) | partial, sendrecv + add |
| LM head | N (vocab tiles 594c.., padded to 1,188 tiles, padding masked) | (max, lowest-id score) pair |

Exchange: `nisa.sendrecv(src=mine, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
dma_engine=nisa.dma_engine.gpsimd_dma)` SBUF to SBUF: 0.85 us per dependent exchange + add (2.0 us with the
default `dma_engine.dma`; probe p8). 57 exchanges per step ~ 50 us, against ~1.2 ms saved.

## Pitfalls (measured)

- **PSUM accumulation depth**: more than ~8k matmuls accumulating into one PSUM tile before it is read made the
  GEMV 3.7x slower (512 columns x 32 k-tiles: 176 GB/s; 256 x 32: 661 GB/s). The 8B LM head went 4.25 -> 1.9 ms
  by using 256-column PSUM tiles. Chunk the output columns so `columns x k_tiles <= 8192`.
- **Software vs hardware DGE**: default `dma_copy` is software DGE driven by GpSimd (~0.65 us of GpSimd time per
  DMA). For the big decode weight slabs keep the default (hardware DGE on two queues was slower: 2.378 vs 2.235
  ms/step); for many small DMAs next to GpSimd compute switch to `dge_mode=nisa.dge_mode.hwdge,
  engine=nisa.engine.sync` (prefill S = 4,096: 88.8 -> 76.3 ms).
- **Internal `nl.shared_hbm` scratch** used for cross-core exchange through HBM was private per core in a large
  kernel (each core read back only its own writes), even with `core_barrier`. Pass such scratch buffers as kernel
  inputs allocated by the host.
- Test whole-model kernels on **real token sequences** against a real reference. With random KV caches the
  28-layer stack and even a bf16-rounding numpy reference disagreed with the fp32 reference (cos 0.85): attention
  over random caches is chaotic.
- Python loops are unrolled at trace time (`nl.affine_range` / `sequential_range` are aliases of `range` in nki
  0.6); a 28-layer step is ~30k instructions per core and traces + compiles in ~30 s; 36 8B layers ~212k
  matmuls per core, 325 s.
