# Attention on Trainium2 (decode and prefill)

Purpose: how to write decode attention against a KV cache that stays in HBM, with the new K/V written at a position
the device reads itself, and how to get causal prefill attention. Everything below was measured on the
trn2.48xlarge of `README.md` (neuronx-cc 2.27, nki 0.6, logical core config 2) at Qwen3-0.6B shapes (16 q heads,
8 kv heads, head dim 128, up to 4,608 positions) unless it says otherwise; the code is in the NKI lab
(`kernels/attn_decode.py`, `kernels/step_lnc2.py`, `kernels/prefill_lnc2.py`, FINDINGS.md sections 1 and 3).

## Decode: the position lives on the device

Keep `pos` as a `[1, 1]` int32 device tensor. In the kernel, load it into a register and use it as a dynamic
offset on the HBM side of a DMA:

```python
p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
nisa.dma_copy(dst=p_sb, src=pos)
preg = nisa.register_alloc(); nisa.register_load(preg, p_sb)
# V cache [T, kv_heads*d]: row pos
nisa.dma_copy(dst=vcache.ap(pattern=[[1024, 1], [1, 1024]], offset=0, scalar_offset=preg, indirect_dim=0), src=vrow)
# K cache stored transposed [kv_head, d, T]: column pos of every (head, d) row, from an SBUF tile [128 d, 8 heads]
nisa.dma_copy(dst=kcache.ap(pattern=[[T, 128], [128 * T, 8]], offset=0, scalar_offset=preg, indirect_dim=2), src=kT_new)
# RoPE row pos of a [T, 64] table, broadcast to 24 partitions (partition step 0)
nisa.dma_copy(dst=cs, src=cos.ap(pattern=[[0, 24], [1, 64]], offset=0, scalar_offset=preg, indirect_dim=0))
```

- `scalar_offset` counts in units of the stride of `indirect_dim` of the tensor `.ap()` is called on. The writes
  change the caller's buffers in place; every other element stays untouched (checked bitwise).
- Increment on the device (`tensor_scalar(p1, p_sb, add, 1)`) and write `pos + 1` to a **second** tensor; the
  host builds two tensor sets that swap the two buffers (A reads X writes Y, B the reverse) and queues every
  step. Writing `pos` in place works at LNC=1 but is a read-modify-write race between the two cores at LNC=2.
- The same register trick selects the input token (`prompt[pos]` or `toks[pos]`), gathers its embedding row
  straight into column layout (`embed.ap(pattern=[[1, 128], [128, 8]], scalar_offset=treg, indirect_dim=0)`),
  and stores the argmax at `toks[pos + 1]`: a whole request is a chain of identical executions.

## Decode: layouts that need no transposes

- **K transposed `[kv_head, d, T]`**: a 128-position tile is `[128 d, 128 t]`, the stationary operand of
  `scores[t, h] = sum_d K[d, t] q[d, h]` with q^T `[128 d, 2 heads]` moving (GQA: the q heads of one kv head
  together). **V natural `[T, kv_heads*d]`**: tile `[128 t, 128 d]` is the stationary of `O^T[d, h] = sum_t V[t, d]
  P[t, h]` with P `[128 t, 2 heads]` moving. P never needs a transpose and O^T `[128 d, 16 heads]` is already the
  stationary layout the o_proj GEMV wants. Price: the new K is a scatter of 1,024 2-byte elements per layer.
- Scores live as `[128 t, tiles, heads]`; the per-head max goes through a PE transpose + reduce + a
  `ones[16, 128]^T @ diag(m)` broadcast, the denominators through `ones[128, 128]^T @ rowsums`.
- Attend over `[0, pos)` from HBM plus the current token from SBUF: the kernel never reads what it writes in the
  same execution, so no HBM read-after-write ordering is needed. Zero-initialise the cache once (masked garbage
  must be finite); mask with an additive -30000 before the max (not -inf).
- **Issue the cache loads first in program order**: each engine's instruction stream is in order, so a DMA
  placed after the norm/RoPE chain starts late. b512 52.6 -> 34.1 us, b4608 87.8 -> 73.9 us (queued, LNC=1).

## Decode: static buckets, not device loops

| variant (LNC=1, one physical core) | 512 | 1,024 | 2,048 | 4,608 positions |
|---|---|---|---|---|
| static bucket, device exec (us) | 21.3 | 34.3 | 42.6 | 64.2 (18.9 MB, ~300 GB/s) |
| `nl.fori_loop` over 512-position chunks, queued (us) | 69.9 (pos 256) | 93.1 | - | 226 (pos 4,095) |

The dynamic loop works (a register bound loaded from a device tensor, the loop register as a DMA offset on the HBM
side) but each iteration drains the pipeline (~10 us per chunk per loop). Inside a `fori_loop` body: DMA into an
SBUF tensor at a dynamic offset compiles but `nrt_load` fails (`TDRV:calculate_max_reg_offset OOB access`);
matmul operands at a dynamic SBUF offset fail to compile (`NCC_IBIR040`); a PSUM tile accumulated across
iterations fails (`NCC_IBIR092`). The host always knows the step count, so pick one NEFF per bucket from the
host; in the whole-step NEFF (LNC=2) the buckets cost 2.235 / 2.371 / 2.623 / 3.148 ms per step at 512 / 1,024 /
2,048 / 4,608 positions (each 512 positions of bucket = 59 MB of cache, 0.13 ms).

At LNC=2 split the heads: core c owns q heads 8c..8c+7 and kv heads 4c..4c+3, with the caches laid out
`kcache [L, 2, 4, 128, T]`, `vcache [L, 2, T, 512]` so each core streams one contiguous half; attention needs no
exchange at all (the o_proj after it is K-split, see `kernels-fusion.md`).

## Prefill: use nkilib's `attention_cte`

`nkilib/core/attention/attention_cte.py` (AWS NKI Library, Apache-2.0, a copy ships in the neuronx-cc wheel,
public NKI API only) is a tuned flash-attention kernel with native GQA, causal masking and LNC=2 support.

- Standalone (`build(attention_cte, ..., kw=dict(scale=1/sqrt(128), causal_mask=True, tp_q=True))`, LNC=2,
  queued): S = 256: 73 us (3.7 TFLOP/s), 1,024: 250 us (17.2), 4,096: 1,384 us (49.6), max error 0.008 vs fp32.
- **Inline it in your own kernel**: call the undecorated `_attention_cte(q=qT, k=kT, v=v, scale=..., causal_mask=True,
  tp_q=False, tp_k=False, tp_out=True, out_o_hbm=oT)` with `qT [16, 128, S]`, `kT [8, 128, S]`, `v [8, S, 128]`
  in HBM. `tp_q=False` / `tp_k=False` / `tp_out=True` are exactly the feature-major layouts a prefill that keeps
  activations transposed produces and consumes, and kT is the decode K-cache layout. It splits the 16 heads over
  the two cores itself (8 each). Inlined it costs the same as standalone (257 vs 250 us at S = 1,024).
- Do not wrap the jit kernel in another jit through a Python conditional: it traced to an empty kernel
  (`NCC_INKI003 Insufficient number of outputs param`).
- In a 28-layer prefill at S = 4,096 attention is ~1.2 ms of the ~2.7 ms per layer (profile), i.e. ~34 ms of
  the 76 ms whole-model prefill.

Record a copied nkilib kernel in `THIRD_PARTY.md` (path, neuronx-cc 2.27.5334, Apache-2.0).
