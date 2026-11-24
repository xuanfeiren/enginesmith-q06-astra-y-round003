# Qwen3-0.6B engine on Trainium2: notes

Machine: trn2.48xlarge, logical NeuronCore config 2, Neuron runtime 2.34.10, neuronx-cc 2.27.5334, nki 0.6.0.
Every number is measured unless marked as an estimate.

## Session 001 (2026-11-03): first engine, passes the gates

What the engine is:
- `target/engine --model DIR` (`host/engine.cpp`, C++17, links libnrt only) speaks protocol 1. At `hello` it loads
  `model.safetensors` (mmap), re-lays out the weights on the host (values unchanged: the weight-stationary tile layout
  of `kernels/qwen3_weights.py ws_layout`, split per physical core like `tests/test_step.py split_inputs` /
  `tests/test_prefill.py prefill_inputs`; norm weights bf16 -> fp32 exactly; RoPE tables in fp64 -> fp32), uploads
  them, zero-initialises the KV cache, loads the 15 embedded NEFFs, binds tensor sets once and runs warm-up requests
  that touch every NEFF. It uses only the first visible logical core (rewrites `NEURON_RT_VISIBLE_CORES` to its first
  entry before `nrt_init`).
- Kernels (all from the seed's lab, FINDINGS.md): decode = `kernels/step_lnc2.py` (one NEFF per step: 28 layers,
  LM head, argmax with ties to the lowest id, K/V written at a device-held position), one NEFF per static attention
  bucket 512 / 1,024 / 1,536 / 2,048 / 3,072 / 4,608 (1,536 and 3,072 added this session); the host picks the
  smallest bucket > pos for every step. Prefill = layered `kernels/prefill_lnc2.py` (`pf_emb`, `pf_layer` x 28 on
  per-layer slices via `nrt_tensor_allocate_slice`, `pf_head`) for buckets 256 / 1,024 / 4,096 (prompt padded to the
  bucket; causal attention, so padding never reaches real tokens).
- Request flow: write prompt ids, `plen` (decode) and `plen_pf` (prefill: min(P, 4096)); queue prefill, then one
  decode step per remaining prompt token (only for P > 4,096) and per generated token, all with
  `nrta_execute_schedule`, the position ping-ponging between `posX` / `posY` on the device; wait once; read the reply's
  tokens from `toks` once. `reset: false` continues from the device state (the last emitted token is `toks[pos]`).
  Requests beyond 4,608 positions get an error reply.
- Build (`build.sh`): `tools/build_neffs.py` compiles the 15 NEFFs through NKI's documented route
  (`NKI_ARTIFACTS_DIR` + calling the `@nki.jit` kernel with zero numpy inputs, `tools/neff_one.py`), one process per
  kernel, two workers on cores ENGINESMITH_NEURON_CORE + 2 / + 3 (override: `NEURON_BUILD_CORES`); cached in
  `target/neff-cache/` keyed by a hash of `kernels/`, `tools/`, `third_party/`. `build/neffs.S` embeds them with
  `.incbin`; `nrt_load` from memory. Clean build from a fresh clone: 4 min 05 s (measured); cached rebuild 2 s.
  NEFF bytes are not reproducible across builds of the same source (all 15 differed byte-wise), so compare engines by
  tokens, not NEFF bytes.
- nkilib's `attention_cte` (Apache-2.0) is vendored in `third_party/` (THIRD_PARTY.md). The lab's
  `tests/nkibuild.py` uses private nki entry points; it is a lab-only test helper, never used by `build.sh`.

Verified:
- Smoke test (core 42): "The capital of France is Paris. The capital of Germany is" -> " Berlin. The capital of Italy is
  Rome. The capital of Spain is Madrid. The"; generate(P, 1) + generate((), 16) identical to generate(P, 17).
- `enginesmith dev-eval --lengths all` (1 round), dev-eval 001: PASSED; all 17 control prompts pass (1 to 4,097
  tokens; 4 not identical, worst 1 mismatching step at reference rank 3, 13 allowed); timed cases 1 / 0 / 0
  mismatching steps.

| case | prefill ms ref / ours | prefill speedup | decode (64 tokens) ms ref / ours | decode speedup |
|---|---|---|---|---|
| p256 | 11.7 / 8.0 | 1.456 | 275.9 / 143.4 (2.24 ms/token) | 1.924 |
| p1024 | 37.9 / 20.4 | 1.859 | 438.5 / 159.1 (2.49 ms/token, b1536) | 2.756 |
| p4096 | 97.3 / 78.0 | 1.247 | 347.9 / 201.3 (3.15 ms/token, b4608) | 1.729 |

Score 1.772 (prefill 1.500, decode 2.093) on dev-eval 001, one round. dev-eval 002 on the final tree (vendored
nkilib, NEFFs rebuilt): PASSED, score 1.747 (prefill 1.469, decode 2.077); our times within 0.1 ms of 001, the
difference is the reference's own run-to-run noise (p256 prefill 11.7 vs 11.4 ms). Load (hello) about
10-15 s, mostly the host re-layout and upload.

## Where to go next after session 001 (estimates, not measured; superseded, see the end of session 002)

- p4096 decode reads the whole 4,608 bucket each step: a 4,352 or 4,224 bucket (decode positions 4,096..4,160) saves
  roughly 0.03-0.06 ms/step (59 MB per 512 positions at ~520 GB/s marginal, FINDINGS.md).
- p1024 decode: a 1,152 or 1,280 bucket instead of 1,536 (positions 1,024..1,088).
- p256 prefill: the one-NEFF prefill measured 7.30 vs 7.51 ms layered in the lab; the reference here took 11.7.
- p4096 prefill (78 ms vs the reference's 97 ms in this evaluation; the model card says 155 ms for the reference,
  so the reference was faster here than when the card was written): the prefill layer NEFF's phase profile is in
  FINDINGS.md section 3 (attention 1.2 ms of 2.7 ms per layer).
- Decode is latency-bound per layer (FINDINGS.md section 4, "Also measured"): 2.23 ms vs ~1.9 ms floor at b512.

## Session 002 (2026-11-04): DMA queues, own prefill attention, SBUF-resident short-prompt layers, host spin

Summary. dev-eval 008 (3 rounds, all lengths, final tree, built by dev-eval): PASSED, score 2.2075 (prefill 2.0305,
decode 2.4000); dev-eval 007 (before item 15, which changes no timing): 2.2095 (prefill 2.0375, decode 2.3960);
round 001 was 1.7486. Per case (ref / ours, ms, median of 3): p256 prefill 11.2 / 5.2, decode (64) 271 / 125.8;
p1024 prefill 36.5 / 14.5, decode 435 / 137.1; p4096 prefill 97.2 / 63.8, decode 347 / 172.1. Controls: 6 of 17 not
identical, at most 2 mismatching steps per stream (worst rank 2-3, 13 allowed); timed cases 0 mismatches.
What the engine is now (differences from session 001):
- decode step (step_lnc2.py): weight / LM-head / cache DMAs on the Sync engine's HWDGE queue, cache writes through
  aliased inputs kcw / vcw, cache loads with `dge_mode.none` for buckets >= 3,072, 2 MB weight slabs for buckets
  < 3,072; buckets 384 / 512 / 1,024 / 1,152 / 1,536 / 2,048 / 3,072 / 4,224 / 4,608. ~1.96 ms/step at b384 (1.28 GB,
  655 GB/s: at the measured streaming ceiling), 2.13 at b1152, 2.67 at b4224.
- prefill per bucket: 256 -> pf_emb, 27 x pf_sb (prefill_sb.py), pf_kv, pf_tail; 1,024 -> 27 x pf_sbc (chunked
  pf_sb); 4,096 -> 27 x pf_layer (prefill_lnc2.py with attn_prefill_big, software-pipelined phase A, weights one
  phase ahead). pf_kv = the last layer's KV-cache writes (phase A k / v tiles), pf_tail = the last layer for the last
  prompt token with the decode layer code + LM head + argmax. Own causal attention (attn_prefill.py) in all buckets.
- host: busy-polling stdin between requests.
Verified: same.py (40 prompts x 3 request shapes) after every bitwise-neutral change (identical), dev-eval after
every numerics change (001-007 all PASSED), a fresh clone of the final commit builds with `bash build.sh` (5 min 38 s,
no NEFF cache) and gives tokens identical to the development build on all 40 same.py cases.


Starting point: round 001, score 1.7486 (prefill 1.4705, decode 2.0794). All timings below are measured on logical
cores 42 / 43 (`ENGINESMITH_NEURON_CORE` + 2 / + 3); "chain" = `host/chain` queued executions of one NEFF with zero
inputs (us per execution, median of 3 x 51); "A/B" = `brief/skill/scripts/ab.py`, 6 interleaved rounds (engines
started through wrappers that set `NEURON_RT_VISIBLE_CORES`, because the scripts' clean environment would otherwise
let the runtime claim every core). Profiles: `neuron-explorer capture` with zero-filled input files + `view
--output-format json`, per-engine busy time in 5-10 us buckets.

### Decode (kernels/step_lnc2.py, kernels/layer_decode.py `_dma` modes)
Profile of the round-001 step (b384): DMA active 97% but only 520 GB/s; every weight DMA was software DGE issued
from GpSimd's in-order instruction stream, which also runs the q/k-norm rsqrt, iota, casts and the sendrecv
exchanges, so the weight prefetch stalled behind every dependent compute op (gaps of 10-25 us per layer).
1. All weight / LM-head / KV-cache DMAs on the Sync engine's hardware DGE queue (`dge_mode.hwdge,
   engine=sync`; Sync runs nothing else): chain b384 2205 -> 2078 us, b4224 3052 -> 2831 us. (Round 001's
   FINDINGS tried sync/scalar alternating: slower, because the Scalar engine's stream blocks on compute too.)
2. The cache writes go through separate kernel inputs `kcw` / `vcw` that the host binds to the same device tensors
   as `kcache` / `vcache`: the compiler had serialised layer l+1's cache load behind layer l's (dynamic-offset) cache
   write on the whole tensor. Safe: a step never reads what it writes (current token from SBUF, the tile containing
   pos is masked), and NEFF executions are ordered. b384 2078 -> 1987 us, b4224 2831 -> 2819 us.
3. KV-cache loads with `dge_mode.none` (descriptors pre-generated by the runtime) for buckets >= 3072: b4224
   2819 -> 2679 us, b3072 2483 -> 2455 us; worse for small buckets (b384 1987 -> 2002, b1152 2162 -> 2202, b2048
   2294 -> 2333), so small buckets keep the cache loads on the Sync queue. Rejected: caches on the GpSimd queue
   (b4224 2894), on the Scalar HWDGE queue (b384 2289, b4224 3052), `dge_mode.none` for the weights (b4224 3947),
   explicit swdge (2918).
4. Weight slabs of 8 output tiles (2 MB per DMA) instead of 4 for buckets < 3072: b384 1987 -> 1958 us, b1152
   2164 -> 2134 us; b4224 2674 -> 2690 (kept at 4). SL 12 / 16: same as 8.
5. Buckets 384 / 1152 / 4224 added (timed decode positions are P .. P + 63): A/B decode +0.8% / +2.9% / +2.6%.
A/B (round 001 -> 1+2+3+5, tokens identical on all 40 same.py cases): decode +12.0% / +14.9% / +16.7% at
256 / 1024 / 4096.
Remaining (b384 profile after 1-2): the Sync queue now waits only on buffer reuse (WAR on PE), i.e. the prefetch
depth is bounded by SBUF, and the step moves 1.24 GB in 1.96 ms (633 GB/s, floor at the best GEMV rate ~1.85 ms).

### Prefill (kernels/prefill_lnc2.py, kernels/attn_prefill.py)
Per-layer NEFF profile at S = 256 (247 us): 15 us preamble, phase A 50 us, 20 us gap, nkilib attention 60 us,
o_proj weights loaded only after the attention barrier (on GpSimd's queue behind nkilib's ~150 small software-DGE
DMAs), C / D / E ~85 us.
1. `HOIST` (S <= 256): all four weight matrices of the layer are loaded at the top of the layer, on the Scalar
   engine's HWDGE queue (`_load_w(..., WQ=3)`): 247 -> 231 us per layer. Hoisting alone (still on GpSimd): no change;
   on the Sync queue: 233 us. At S = 1,024: no gain (684 -> 682); at S = 4,096 SBUF allocation fails.
2. nkilib attention_cte DMAs moved to HWDGE (vendored copy patched, per call site): at most 2% (227-233 us); the
   output store does not compile with HWDGE (`NCC_IBIR098`), `dge_mode.none` is 2.5x slower. Reverted (vendored file
   unchanged).
3. Own causal attention `attn_prefill.py` for buckets <= 1,024 (core c: q heads 8c.., kv heads 4c.., the QKV split):
   scores in row layout in PSUM (q on partitions, moving = k^T chunks of 512), causal mask added in place on the
   diagonal block, row max on VectorE straight from PSUM, exp(scale*s - scale*max) to bf16 with the fp32 row sum
   from the ScalarE accumulator, e blocks transposed on the PE (4 per PSUM tile), O^T = sum_j V_j^T e_j^T with both
   q heads of a kv head as one 256-wide moving operand, 1/sum broadcast over d with a ones^T @ diag(1/sum) matmul.
   Work items (kv head, q block, q head) software-pipelined in program order (QK of item n+2, softmax of n+1,
   transposes / PV of n), since every engine executes its stream in order. Max abs error vs numpy fp32 0.0089
   (S = 256) / 0.0082 (1,024), the same as nkilib's 0.008. Standalone (chain, incl. ~20 us NEFF overhead):
   S = 256 50 us (nkilib 73), S = 1,024 172 us (nkilib 250); unpipelined first version 72 / 239. In the layer:
   S = 256 231 -> 214 us, S = 1,024 684 -> 590 us. Not used at 4,096: its cost is per element (VectorE max 1.1 ns
   and PSUM copy 0.8 ns per element, ScalarE exp 0.83 ns + ~0.3 us per instruction), extrapolated ~2 ms vs nkilib's
   1.38 ms. Measured instruction costs (pe_probe-style kernels): ScalarE activation ~310 ns fixed, VectorE reduce from
   PSUM ~60 ns + 1.1 ns/element, PE transpose of 128 x 128 ~90 ns, 512-column bf16 matmul ~250-630 ns (the profiler
   shows power throttling to 50% for ~35% of a 4,096-token layer).
4. Last layer: `pf_kv` (phase A for the k / v tiles only: the cache writes for every prompt token) + `pf_tail` (the
   decode layer `_layer2` at pos = plen - 1 with the residual column from `s_h`, attention over the prompt bucket,
   then final norm + LM head + argmax): replaces the last pf_layer + pf_head. chain S = 256: 214 + 520 -> 65 + 581
   us; S = 4,096: 2,700 + 520 -> 406 + 611 us. A/B prefill +0.0% / +1.7% / +2.9%; 39/40 same.py cases identical to
   the previous build (one differs: the last token's K/V now come from the decode path).
5. LM head of the prefill on the Sync HWDGE queue: 524 -> 520 us (kept, noise level).
Rejected: one NEFF for the whole 256-token prefill (6,655 us vs 6,562 us for the 30 layered executions, chain).

dev-eval 001 (after decode 1-3,5 and prefill 1,3,5; 1 round, all lengths): PASSED, score 1.9846 (prefill 1.6301,
decode 2.4161); controls: 6 of 17 not identical, 3 single mismatching steps (worst rank 3, 13 allowed).

Host: the engine's request handling is device-bound (submission waits on the queue); ~0.4 ms per request is spent
outside generate() (pipe / wake-up). A busy-polling stdin reader cut a no-op request after GPU work from 0.3 to
0.06 ms but did not change generate timings (7.0 ms at P = 256 either way): rejected.

### Session 002, continued (after dev-eval 001)
6. RMSNorm rsqrt of the prefill (norm1 / norm2 / q,k norms) on ScalarE (`activation(op=rsqrt)`, LUT; max relative
   error 4.4e-5 vs GpSimd's 7e-8, measured on 65k values, applied before the bf16 rounding of q / k / x) instead
   of GpSimd, which made phase A GpSimd-bound (each [128, 512] rsqrt 2.3 us): layer S = 256 214 -> 205 us,
   1,024 591 -> 557 us, 4,096 2,702 -> 2,550 us. (VectorE has no rsqrt / sqrt / pow: `tensor_scalar_valid_ops`.)
7. Hoisted weights (S = 256) on GpSimd's queue now that GpSimd is idle: 204 -> 197 us. Weights one phase ahead
   (`PREF`, o_proj loaded before attention, gate/up before o_proj, down before gate/up) on the Scalar HWDGE queue for
   1,024 / 4,096: 557 -> 550 us, 2,560 -> 2,543 us.
8. attn_prefill at 1,024: e^T by `nisa.dma_transpose` (the DMA queues are idle; what nkilib does) instead of PE
   transposes + PSUM copies: 172 -> 151 us standalone; pipeline depth 3: 141 us. At 256 the DMA transposes are slower
   (50 -> 75 us), so the PE path stays there.
9. attn_prefill_big for the 4,096 bucket: scores staged in an fp32 SBUF row per (q head, query block) through PSUM
   sections of 1,024 keys, VectorE copy fused with the running row max (`tensor_scalar_reduce` with the max chained
   in the VectorE accumulator across sections, `select_reduce` with a 0/1 predicate for the diagonal block), one exp
   per row, DMA transposes, per-kv-head q / k / v tiles loaded when first needed (SBUF). Standalone 1,140 -> 1,101 us
   with the chained max, vs nkilib attention_cte 1,311 us (same max abs error 0.0084 vs numpy fp32). In the layer:
   2,546 -> 2,428 us. Rejected: sections of 2,048 or 512 (1,202 us), depth 3 / 4 (1,146 / 1,141), the small ops on
   GpSimd (1,131).
   nkilib's own exp per 512 keys widened to 1,024 / 2,048 (vendored copy patched, `_exp_wide_large_tile`): 1,367 /
   1,569 us vs 1,311: rejected, file restored.
10. Phase A software-pipelined (`_phase_a_pipe`: QKV matmuls of tile nt + 3 issued before the norm epilogue of tile
   nt and the RoPE epilogue of tile nt - 1; the PE stream used to stall on every epilogue): 4,096 2,422 -> 2,339 us,
   1,024 545 -> 537 us; at 256 slightly slower (197 -> 200), so only for buckets >= 1,024.
Rejected: two layers per NEFF at 256 (195 vs 197 us per layer); no core barrier before attn_prefill (no change).
A/B after 6-8 (6 rounds): prefill +3.6% / +2.2% / +4.9%; after 10: +0.0% / +1.7% / +3.3%; decode unchanged.
dev-eval 003 (after 9): PASSED, score 2.0939 (prefill 1.7946, decode 2.4431).
11. `prefill_sb.py` `pf_sb` for the 256 bucket: the whole layer SBUF-resident and split over the cores like the decode
   step (QKV / gate-up N-split, o_proj / down K-split over the core's own heads / intermediate tiles with the decode
   weight layouts, partial sums exchanged with `nisa.sendrecv` (dma engine; 8 KB per partition ~2.4 us incl. the add;
   `gpsimd_dma` is limited to 1 KB per partition) and added as mine + other, which is commutative, so both cores hold
   identical residuals), attention on SBUF q / k / v (`attn_prefill_sb`). No barriers, no HBM round trips except the
   residual between layer NEFFs. vs pf_layer on random weights (`exp/test_psb.py`): K / V caches bitwise equal,
   residual within 3.5e-4 relative (summation order of o_proj / down). chain 197 -> 158 us per layer; phase A
   software-pipelined 152.5 us. A/B p256 prefill +21.7%, 38/40 same.py cases identical.
12. `pf_sbc` for the 1,024 bucket: the same with every phase except attention in 512-token chunks (one residual
   tensor per chunk), weights loaded one phase ahead in program order: 537 -> 502 us per layer (256-token chunks:
   528). A/B p1024 prefill +6.2%. Not possible at 4,096 (q / k / v alone would be 128 KB per partition).
   Rejected for pf_sb: xc on GpSimd (163 vs 153 us), V copies on ScalarE (154), weights split Sync / Scalar (156).
dev-eval 005: PASSED, score 2.1799 (prefill 1.9838, decode 2.3953).
Rejected (4,096 attention): PV for two query blocks per V-tile matmul (512-wide moving, `_ab_s3q`): standalone
1,213 us at depth 2, 1,079 at depth 6 (vs 1,101), but 2,392 vs 2,339 us in the layer; depth 6 without it: 2,400.
Rejected (decode): weight slabs alternating over queues (b384): Sync + `dge_mode.none` 2,214 us, Sync + none + GpSimd
2,165, Sync + GpSimd 2,478, vs 1,957 on the Sync queue alone. (A pure-DMA probe, `exp/bw_probe.py`, streamed more
with 3 queues than with 1, 334 vs 234 GB/s, but the probe is far below what the step itself reaches.)
13. RMSNorm: `activation(op=rsqrt, data=ss (PSUM), scale=1/n, bias=eps)` instead of a VectorE tensor_scalar + the
   rsqrt (one instruction less per norm and per q / k head; output bitwise equal in the pf_sb test): pf_sbc 504 ->
   497 us, others unchanged. attn_prefill_big: -scale * max on ScalarE instead of VectorE (1,104 -> 1,090 us); the
   1 / sum broadcast tile built with one VectorE instruction instead of two: no gain (rejected).
   A/B (6 rounds): prefill +0.6% / +3.3% / +1.3%.
Rejected: 3 layers per pf_sb NEFF with the residual kept in SBUF and the next layer's weights requested during the
MLP: 153.2 us per layer, same as one layer per NEFF. The pf_sb layer is PE-bound under power throttling (profile:
PE active 104 of 154 us, `throttle_avg_util_limit` 0.61-0.64, 256-column matmuls at ~305 ns instead of ~110).
14. Host: stdin is read non-blocking and the engine spins for up to 10 s after the last input before it blocks in
   poll(). A process that blocked in read() between requests was measurably slower on the next request, beyond the
   pipe latency (interleaved, both core assignments): generate(P, 1) 5.55 -> 5.28 / 15.07 -> 14.58 / 65.77 ->
   64.16 ms at P = 256 / 1,024 / 4,096; A/A control (both blocking): 5.36 vs 5.47 / 15.08 vs 15.07 / 65.69 vs 65.70.
   A/B (ab.py, 8 rounds): prefill +4.7% / +3.5% / +2.2%, decode within noise. (The earlier rejection in this session
   compared single back-to-back runs on one core and missed it.)

## Where to go next (after session 002)
- p4096 prefill is the weakest case (1.52): per layer ~2.3 ms = attention ~1.0 (attn_prefill_big; VectorE, PE and
  ScalarE each ~85% busy: the copy + max pass at ~1 ns per score element is the floor of this design), phase A ~0.45
  (PE-bound with ~37% overhead matmuls: q/k-norm sums, RoPE rotation, V transposes), C / D / E near the PE rate. An
  SBUF-resident 4,096 layer is not possible (q / k / v alone 128 KB per partition); a K-split + exchange structure
  with q / k / v / o through HBM is untried (estimate: a few %).
- Power throttling: profiles of compute-heavy NEFFs show `throttle_avg_util_limit` 0.6-0.8 (PE limited to 50% for a
  third of the time). Not understood; it caps every PE-bound phase.
- Decode is at the streaming ceiling; only fewer bytes would help, and the weights must stay bf16.
- p256 prefill: pf_sb is PE-bound under throttling (153 us per layer vs ~65 us of weight streaming); pf_tail is
  0.58 ms of the 5.2 (the LM head alone streams 311 MB).
15. Robustness: libnrt prints its log lines to stdout (seen with a configuration warning: the WARN line arrived
   before the hello reply). The engine now writes replies to a private dup of the original stdout and points fd 1 at
   stderr before nrt_init. (`NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS` is ignored by this runtime;
   `NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1`: no measurable change, not used.)

## Session round 001 (2026-11-04): optimization work

Starting point: SESSION.md score 2.2094, prefill 5.2 / 14.4 / 63.8 ms and decode 125.7 / 137.2 / 172.1 ms
at prompt lengths 256 / 1024 / 4096. Decode is already near the measured streaming ceiling.

Initial hypotheses (estimates, before measurement):
- Phase A of the 4096 prefill spends matrix-engine work on RoPE's signed permutation and value transposes.
  Test a two-half stream shuffle for RoPE, and DMA transposes for V, independently. Expected upside is a few
  percent of phase A (~0.45 ms/layer); falsifier: no reduction in full-layer queued execution time.
  The expressions and reduction orders stay unchanged; compare residual and KV bytes before timing.
- The phased long-prompt MLP writes and reloads its activations through HBM and synchronizes at phase boundaries.
  A chunk-local C/D/E schedule with exchanges could remove those trips while preserving each projection's original
  N split and accumulation order. Not implemented or measured yet.
- Wider prefill chunks can reuse each Tensor Engine stationary weight across two adjacent 512-token matmuls.
  Test 1024-token chunks with each matmul explicitly split into legal 512-column instructions; per-token K
  accumulation order stays unchanged. Estimate: remove up to half the stationary-load instructions in phases
  A/C/D/E, at the cost of more SBUF/PSUM occupancy. Falsifier: no improvement in full-layer queued time.
- Gate/up matmuls currently precede each tile's SiLU/multiply/store epilogue. Explicitly issue the next tile's
  matmuls before the current epilogue (as phase A already does). This may avoid PSUM reuse dependencies and
  overlap the tensor/vector/scalar streams. Estimate: a few percent of the long layer; verify the same residual
  and cache bytes and measure queued layer time. Pipeline depths 1 and 2 are the initial tests.

Initial measured results (lab cores 26/27, one accelerator user under the shared GPU lock):
- Fresh baseline build: all 21 kernels compiled, native executable produced. A/A `ab.py`, 8 rounds, two warmups:
  all tokens identical. B/A median times correspond to A/B ratios (256 / 1024 / 4096) prefill
  0.9890 / 0.9982 / 0.9984, decode 0.9979 / 0.9974 / 1.0046. Prefill ratio SD
  2.74% / 4.02% / 0.14%; decode SD 0.05% / 0.08% / 0.23%. Short-prefill outliers and core bias require care.
- Full-layer random-weight checks at S=4096 (seed 0): V DMA transpose, chunk-local N-split C/D/E with exchanges,
  and 1024-token chunks all produced byte-identical residuals, K caches and V caches.
- Queued layer timing, 4 alternating-order rounds, each sample median of 3 x 51 executions:
  baseline 2307.18 us; V DMA transpose 2353.70 us (+2.02%, rejected); chunk-local N-split C/D/E
  2399.46 us (+4.00%, rejected); 1024-token chunks with phase-A depth 1 2344.31 us (+1.61%, rejected).
  These are diagnostic layer times, not full-model speedup claims. Raw timings remain in this session's lab/logs.
- RoPE shuffle first failed NKI tracing because list(range(...)) is unsupported inside a traced function;
  an explicit constant mask compiles and gives byte-identical layer residuals and caches.
- Further hypothesis: normalization currently repeats the identical reduction and rsqrt across all 128 output
  partitions. Computing one output partition with the same PE accumulation and broadcasting its fp32 result may
  reduce power/compute costs. Investigate exact PE, stream-shuffle and DMA broadcasts in a small public-NKI probe
  before modifying the layer. Not measured yet.
- Additional long-layer candidates, two alternating-order rounds (3 x 51 executions each): baseline
  2311.12 us, RoPE shuffle 2450.60 us, 1024-token chunks / phase-A depth 2 2357.19 us, MLP pipeline depth 1
  2354.33 us, depth 2 2341.59 us. All residual and KV bytes match, all candidates rejected for regressions.
  Some individual repetitions had host outliers; medians and the larger regressions are clear.
- Rebuilt controls: untouched source copied into a lab module 2307.28 us, experiment scaffolding with all flags
  disabled 2306.82 us, original baseline 2305.49 us (2 alternating rounds). Scaffolding is neutral.
- Short MLP pipeline depth 1: 256 baseline/candidate 154.27 / 154.17 us; 1024 497.79 / 498.46 us (2 alternating
  rounds, 3 x 51 executions each). Residual and caches byte-identical at both shapes; no useful gain, rejected.
- Chunk-local K-split C/D/E (decode's wo/wd layouts, same approach as the short engine) at 4096:
  random-layer KV bytes identical, residual max abs difference 0.00325155 (0.000428757 relative to max residual).
  This is a correctness change. Full `dev-eval 001 --no-build --lengths all` PASSED (206 s), all timed cases
  0 mismatches. Candidate prefill 5.2 / 14.6 / 69.4 ms, decode 125.8 / 137.3 / 173.2 ms, score 2.1936
  against that run's reference. Long prefill regresses from ~64 ms, so rejected and original source/binary restored.
- Structural phase-A hypothesis: make each 128-query activation tile the stationary GEMM operand, and four
  heads of weights the moving operand. Projection outputs then have queries on partitions and head dimensions
  on free axes. This permits VectorE head-norm reductions and free-axis RoPE swaps, avoiding the old replicated
  reduction/rotation matmuls; V is already in cache layout. Q/K need a grouped DMA transpose afterward.
  Preserve projection accumulation order, norm square rounding and q * weight * rsqrt order. Vector reduction
  order may differ, so compare bytes and use reference gates if necessary. Estimate: recover part of phase A's
  ~37% non-projection matmul overhead; whole-layer timing is the falsifier.
- Normalization broadcast probe (32 blocks of 128x512 nonnegative bf16 inputs, LNC=2): PE broadcast and four
  stream-shuffle broadcasts both give exactly the baseline's fp32 output bytes. Queued medians (3 x 51 calls,
  one diagnostic round): original repeated reduction/rsqrt 41.41 us; one partition + PE broadcast 62.17 us;
  one partition + stream shuffles 70.55 us. Rejected. DMA broadcast cannot compile: SBUF partition stride 0
  is invalid. This probe uses only public NKI APIs.
- Query-stationary phase A prototype at 4096: random-layer residual max abs diff 0.00916189 (0.00120811 relative
  to max), K max diff 0.0078125, V bytes identical. Vector head-norm reduction changes summation order; this is
  a correctness change and is being evaluated against the reference before adoption.
- Layout follow-ups to query-stationary QKV (hypotheses): RoPE tables can be stored query-major at load time,
  and QKV weights can be reordered from [partition, output tile, input tile, output lane] to
  [partition, input tile, output tile, output lane]. This leaves all values and arithmetic unchanged while making
  the row kernel's table reads and 512-column moving weight operands contiguous. Test each against its row
  parent by bytes/tokens before timing; a faster layer is required before considering this route.
- Public NKI nc_matmul documentation identifies an exact optimization for fp32 operands known to contain only
  ones/zeros: is_stationary_onezero=True can avoid unnecessary work. Prefill attention's final normalization
  broadcast uses a known all-ones stationary matrix but currently omits this hint. Test the hint in that
  broadcast only, across all prompt buckets. Arithmetic should be unchanged; require byte/token identity.
- Query-stationary first prototype: `dev-eval 002 --no-build --lengths all` PASSED in 189 s, all timed cases
  0 mismatches; long prefill 89.1 ms versus baseline ~64 ms (score 2.1101 in that reference pairing).
  Original source/binary restored while layout follow-ups are measured.
- Query-major RoPE tables give byte-identical layer residual/K/V to that prototype; `same.py` 40 prompts x
  3 request shapes: 40/40 identical, 74.6 s. Reordered QKV weights likewise give byte-identical layer outputs
  and 40/40 identical full-engine cases against the table-layout parent (75.6 s). Timings pending.
- Short-prompt launch grouping hypothesis: three original SBUF-layer calls in one NEFF, with their original HBM
  residual hand-off and an explicit core barrier between layers. This preserves the per-layer boundary and does
  not prefetch the next layer's weights during the preceding MLP, unlike the previous session's SBUF-resident
  three-layer fusion. Aim to remove 18 NEFF launches at S=256; estimated upper bound ~0.36 ms/request from the
  measured ~20 us launch cost. Test full token/state flow before timing. Includes the separately checked ones hint.
- Attention epilogue hypothesis: the long-attention score copy/max pass already visits every score. Scale scores
  by -SCALE during that pass and reduce their minimum, obtaining the negative scaled row maximum directly.
  Then exp(-scaled_score + negative_scaled_max) needs no separate max copy/scale instructions. The diagonal mask
  needs one additional short VectorE instruction. Estimated gain ~40-50 us/layer, pending measurement.
  Moving scaling from ScalarE to VectorE may change rounding; treat any byte difference as a correctness change.
- The same attention scaling mechanism can combine the small-bucket PSUM row reduction with scaling in place,
  removing the separate maximum-scaling instruction there as well. No extra diagonal-mask pass is needed for
  small buckets, since their original code already masks PSUM before reduction. Test separately at 256/1024.
- First scaled-score candidate failed `dev-eval 003` during hello warmup (NRT_EXEC_OOB in the next decode's
  embedding gather after the 1200-token prompt). Candidate rejected; original source/binary restored. The first
  long-attention implementation carried a minimum in VectorE's shared accumulator across a select_reduce
  mask operation. Hypothesis: that operation clobbers the minimum, allowing exp overflow on real-model scores.
  The revised version resets and stores one minimum per PSUM section after masking, then reduces those explicit
  minima. This avoids shared accumulator state across the mask. A hello/warmup smoke check precedes re-evaluation.
- Grouped short-prefill follow-up: construct the layer-independent identity/ones/causal-mask tiles and RoPE
  rotation tile once per group, pass them read-only to all three layers, and retain the explicit layer barriers.
  Values and arithmetic remain unchanged. Estimate: a few microseconds per layer from repeated constant setup;
  require full token identity and interleaved request timing before adoption.
- Ones-hint full check: `same.py` 40/40 prompts, all three request shapes identical to baseline (74.9 s).
  Two 8-round A/B runs with core assignments reversed, all tokens identical: prefill ratios
  1.0069 / 1.0079 / 1.0039 and 1.0290 / 0.9993 / 1.0002 (256 / 1024 / 4096).
  Core-balanced geometric ratios approximately 1.0179 / 1.0036 / 1.0020; too small alone for acceptance.
- Three-layer grouping with original per-layer HBM hand-off/barriers: full `same.py` vs the hinted parent,
  40/40 identical cases (72.4 s). Two 8-round 256-only A/B runs, cores swapped: prefill ratios
  1.0057 and 1.0047; about 0.52% over the hinted parent. This saves 18 launches but much of the estimated
  launch saving does not appear end to end; keep as a candidate only if useful with another improvement.
- Row-layout diagnostic, two alternating-order rounds (3 x 51): baseline 2307.22 us; first row QKV 3240.54 us;
  query-major RoPE tables 2620.58 us; plus contiguous moving QKV weights 2620.83 us. Tables recover 620 us/layer,
  weight layout is neutral, entire route remains ~14% slower and is rejected.
- Revised scaled attention: random 4096-layer residual max diff 0.00750828 (0.000990058 relative to max);
  KV bytes identical. The engine now passes hello and every built-in warmup, including the former failing case.
  The following full reference evaluation completed this check.

- `dev-eval 004` of the corrected scaled-score paths PASSED (204 s, all timed cases 0 mismatches):
  prefill 5.0 / 14.0 / 66.0 ms, decode 125.7 / 137.1 / 172.2 ms; score 2.2074.
  Keep the small-bucket scaling/reduction, restore original long-attention scores/max, retain the ones hint.
  Long scaling is slower, so it is excluded from the final candidate.
- Shared constants within three-layer groups: `same.py` 40/40 identical (77.8 s). Two 8-round 256-only A/B
  runs vs ordinary groups, cores swapped: prefill ratios 0.9628 and 1.0094, balanced ~0.9858. Rejected.
  The final candidate uses the original one-layer execution schedule and host code.

Final candidate source: `kernels/attn_prefill.py` only (plus notes/experiment evidence). For small-bucket attention,
mask scores, then combine multiplication by -SCALE with their minimum reduction in PSUM. The following exp uses
scale -1 and that minimum as bias, removing the separate row-maximum scaling instruction. Both attention paths
mark the known all-ones fp32 stationary matrix in the normalization broadcast with the public NKI one/zero hint.
No weights, model shapes, decode arithmetic, host scheduling, or protocol behavior changed. The short-path change
moves an fp32 rounding point, so final reference gates, rather than parent byte identity, validate its numerics.
Fresh-checkout build and final evaluation follow.

- Fresh checkout of the original commit plus this candidate's source changes built all 21 NEFFs without a cache
  using `bash build.sh`, producing the 19 MB native executable. `ldd` shows libnrt and standard system libraries.
  `same.py` against the development candidate: 40/40 prompts identical across all three request shapes (72.7 s).
  This fresh executable is now `engine/target/engine`; the final 3-round all-length evaluation uses it.


### Final result: clean executable, dev-eval 005

`enginesmith dev-eval --no-build --rounds 3 --lengths all` PASSED in 404 s. All control gates passed;
all nine timed streams had 0 mismatching steps. The native executable was built from a fresh checkout and
matched the development candidate on all 40 `same.py` cases and all three request shapes.

| case | prefill ms, candidate rounds | decode ms (64 tokens), candidate rounds | prefill speedup | decode speedup |
|---|---|---|---|---|
| p256 | 5.1 / 5.1 / 5.1 | 125.7 / 125.7 / 125.8 | 2.318 | 2.142 |
| p1024 | 14.1 / 14.0 / 14.0 | 137.3 / 137.1 / 137.2 | 2.620 | 3.253 |
| p4096 | 63.9 / 63.8 / 63.7 | 172.1 / 172.0 / 173.3 | 1.528 | 2.029 |

Score **2.2539**, prefill mean **2.1011**, decode mean **2.4179**. Against SESSION.md's starting
scores (2.2094 / 2.0417 / 2.3908), these are +2.01% / +2.91% /
+1.13%, respectively. This dev evaluation clears the 1.5% score-improvement criterion.
These are reference-relative scores; the reference's timings vary between evaluations. Native decode code is
unchanged. The candidate's median prefills are 5.1 / 14.0 / 63.8 ms versus the starting report's approximately
5.2 / 14.4 / 63.8 ms.

Final production changes are confined to `kernels/attn_prefill.py`; the host, weight layouts, decode kernels,
launch schedule, build interface and runtime dependencies retain their original implementation. Rejected
experiment patches and measurement samples are preserved in `experiments/round001/`, outside the build path.

## Round 002: attention pipeline exploration

Starting from round 001 (score 2.2322). Production kernels are initially unchanged.
Hypothesis: long attention performs the probability DMA transpose in the same stage as PV and normalization.
Separating that transpose from PV can issue it earlier, overlapping it with independent QK/softmax work.
The existing pipeline-depth sweeps kept those operations together; this tests a different dependency schedule.
No arithmetic, rounding points, query ownership or summation order should change.
Estimate: recover 5-10% of the ~1.1 ms attention stage, or ~2-5% of the long prefill layer.
Falsifier: byte-identical standalone/layer tests followed by alternating queued timings show no gain.
Only a full-engine improvement with correctness and fresh-build verification will be retained.

Second independent hypothesis: long-prefill C/D/E currently split output channels and intermediate channels
between cores, exchanging residual/MLP activations through HBM and three barriers. Give each core alternate
256-token chunks and the complete o_proj/MLP for those chunks. The existing N-split weight tensors can be loaded
from both halves without changing values. This removes C-to-D residual and D-to-E activation HBM round trips,
and duplicate RMSNorm work; every GEMM still sums K tiles in the original order.
Unlike the rejected K-split/exchange experiment, this requires no partial-sum exchanges.
Estimated saving: ~50 MB/layer of intermediate traffic (~75 us at 650 GB/s), less extra weight reads, and two
barriers; an upper bound of roughly 5% of long-layer time. Smaller chunks may lose matrix efficiency.
Require identical residual/KV bytes, then queued-layer and full-request comparisons.

Attention dependency follow-up: `_ab_s2` issues a VectorE reciprocal immediately after ScalarE exp/sum.
That forces the in-order VectorE queue to wait for exp before it can copy/reduce the next QK scores.
Write sums directly from ScalarE into the existing pair buffer and delay reciprocal until PV consumes the pair.
One reciprocal instruction handles both heads with unchanged elementwise arithmetic. This is independent of
the earlier-transpose experiment. Estimate: 5-10% of attention by reducing VectorE/ScalarE serialization;
a profile and unchanged output bytes distinguish this from a numerical change.

A narrower token-split variant keeps the original o_proj phase and changes only norm2/gate/up/down, using
512-token chunks. It avoids the whole C/D/E variant's smaller GEMM tiles and extra o_proj weight reads while
still eliminating the 3072-wide activation HBM round trip (~50 MB/layer) and duplicate norm2.
Extra full-MLP weights cost ~19 MB/layer; estimated net traffic saving ~31 MB (~48 us/layer).
Compare both variants because the smaller all-phase variant may sacrifice matrix throughput.

The same dependency review found a redundant VectorE copy of the already-reduced row maximum in `_ab_s2`.
Using its original SBUF view directly lets ScalarE start the previous item's exp without waiting for the
current item's VectorE score-copy/max pass. Test this on top of delayed reciprocal, with unchanged values.

Initial attention checks (random Q/K/V, seed 472, all 8,388,608 output bf16 values): early transposes and delayed
reciprocal each match the parent byte for byte. Four alternating-order diagnostic rounds, each 3 x 51 queued
executions, on the same logical core: baseline 1085.72 us; early transposes / PV depth 3 1000.08 us (8.56% faster);
delayed reciprocal 1044.845 us (3.91% faster). These are standalone attention measurements, not engine scores.
Baseline full-engine A/A (8 rounds): prefill ratios 0.9859 / 0.9999 / 0.9984, decode 0.9989 / 0.9974 / 0.9932
at 256 / 1024 / 4096, all tokens identical. Swap core assignments for final A/B to cancel that bias.

The short-bucket helper has the same immediate reciprocal dependency. Test writing its ScalarE sums directly
into the pair buffer and taking the reciprocal only at PV consumption, independently for 256 and 1024 buckets.
The original scaled-score formula and reduction order remain unchanged.

Both token-split feed-forward variants pass byte comparisons for all layer residual/K/V outputs (random weights,
seed 0). Four alternating rounds, 3 x 51 queued calls, with all candidates loaded once: parent 2304.848 us,
independent parent copy 2300.399 us (0.19% A/A difference), C/D/E token split at 256 2444.341 us (6.05% slower),
MLP-only token split at 512 2317.526 us (0.55% slower). Both rejected; original projection/MLP code retained.

Full-layer attention comparisons (all residual and K/V bytes identical; 4 alternating rounds, 3 x 51 calls):
parent 2304.407 us; earlier transposes 2226.841 us (+3.48%); delayed reciprocal + direct max view 2300.590 us
(+0.17%, neutral by itself); all three together 2208.770 us (+4.33%). Keep the combined candidate for full-engine
verification. The alone-versus-combined difference demonstrates that the schedule interactions matter.

Further hypothesis: the DMA transpose path inherited its four-key-block batch limit from the PE transpose path.
The public DMA API allows batching the entire probability row (up to 32 key blocks / 1 MB), with unchanged
source/destination addresses and values. Test one DMA call per work item in place of up to eight, leaving the
PE fallback untouched. Estimated upside: some of the DMA descriptor/issue overhead, bounded by ~80 us/layer.

Short-bucket delayed reciprocal: residual and K/V bytes identical at both shapes. Four alternating rounds,
3 x 51 calls: 256 parent/candidate 150.805 / 147.894 us (+1.97%); 1024 481.909 / 489.507 us (1.58% slower).
Retain it only for the 256 bucket; the 1024 path keeps its original scheduling.

Wider DMA batches: residual/K/V bytes match the combined parent. Four alternating rounds, 3 x 51 calls:
original baseline 2303.944 us, combined schedule 2202.556 us, 32-block DMA batches 2248.787 us. Rejected
(2.10% slower than the combined schedule); retain four-block transfers so PV can consume smaller ready regions.

Selected production change is confined to `kernels/attn_prefill.py`: split probability transpose from PV for the
4096 path (PV depth 3), remove its redundant max copy, and defer reciprocal to PV for 256 and 4096 only.
The 1024 path keeps its original immediate reciprocal. Refactor shared transpose/PV helpers without changing
expressions, reduction order, rounding, weights, host schedule or protocol. Verify the refactored build against
the measured lab NEFFs before full-engine token/state checks and paired timings.

Production helper cleanup passes all residual/KV byte comparisons. Layer timings: baseline 2305.235 us,
measured lab schedule 2203.105 us, cleaned long helper 2207.527 us (+4.43% vs baseline, 0.20% behind lab).
The 1024 helper cleanup regressed 481.801 -> 485.915 us (0.85%) despite identical bytes. Preserve the original
1024 helper bodies and allocation scope; dispatch 256 to its own delayed-reciprocal helper. Recheck the result.
Lab validation interruptions were tooling-only (compiler PATH, a stdlib profile-module name collision, and the
profiler requiring all 25 NEFF inputs including unused ones); no compiler/runtime/library patches were made.

The preserved 1024 helper resolves the regression: 481.280 us baseline vs 481.164 us release (4 alternating
rounds, 3 x 51 calls), within noise. Final release 256/1024/4096 residual and K/V outputs match their verified
parents byte for byte. A native development executable embeds those three NEFFs plus the unchanged baseline
NEFFs; full-engine validation is next.

Full-engine `same.py` release vs round-001 parent: all 40 prompts, all three request shapes, identical tokens
and self-consistent continuation state (73.5 s; both engine processes exited 0).
Device profile on the same finite random layer inputs, third execution: parent/release total time
2.313398 / 2.224340 ms. HBM reads 258,033,672 bytes and writes 125,829,128 bytes in both; 15,616 matrix
instructions in both; no spills. Vector instructions 5,986 -> 5,748. The byte/FLOP work is unchanged;
the schedule and fewer vector dependencies account for the improvement. Profile overhead is separate from
request timing; the next measurements use the complete native engine and real model weights.

Full-engine interleaved A/B, two 8-round runs with development-core assignments swapped, every token identical:
prefill ratios 1.023161 / 1.003476 / 1.044076, then 1.017781 / 0.998648 / 1.042864 (256 / 1024 / 4096).
Core-balanced ratios 1.020468 / 1.001059 / 1.043470.
Balanced decode ratios 0.999445 / 0.999338 / 0.998058; decode code is unchanged.
Prefill geometric improvement 2.15%, decode -0.11%, combined 1.02%.
Long prefill wins all 16 pairs; all raw request samples and round spreads are in experiments/round002/measurements.json.
These are candidate/parent comparisons; the authoritative reference evaluation follows the fresh-checkout build.

Fresh checkout: all 21 NEFFs rebuilt without a cache using `bash build.sh`, 341.86 s measured from the build log's
creation/final-write timestamps. Produced a 19 MB executable; `ldd` lists libnrt and standard system libraries.
Fresh binary vs development release: `same.py` 40/40 prompts and all three request shapes identical, 72.7 s;
both processes exited 0. The fresh binary is now `engine/target/engine`.
Final `enginesmith dev-eval --no-build --rounds 3 --lengths all` completed on it; results below.

### Round 002 final result: fresh executable, dev-eval 001

PASSED in 411.93 s, three rounds at all task lengths. All 17 control gates pass: six mismatching steps in total,
at most two in one control stream, worst reference rank 2 (13 steps / rank 100 allowed). All nine timed streams
have zero mismatching steps. Overall six mismatches in 1,690 checked steps, below the allowed 169.

| case | candidate prefill ms, rounds | candidate decode ms (64 tokens), rounds | prefill speedup | decode speedup |
|---|---|---|---|---|
| p256 | 5.0 / 5.2 / 5.1 | 125.8 / 125.8 / 125.6 | 2.381 | 2.148 |
| p1024 | 14.0 / 14.1 / 14.0 | 137.3 / 137.4 / 137.1 | 2.630 | 3.186 |
| p4096 | 61.1 / 61.3 / 61.1 | 173.4 / 172.3 / 172.1 | 1.593 | 2.053 |

Score **2.27948**, prefill mean **2.15311**, decode mean **2.41326**. SESSION.md's starting score is 2.2322:
this development result is +2.12%, above the 1.5% score-improvement threshold. Reference timings vary across
runs; the controlled parent/release comparison attributes +2.15% prefill and +1.02% combined speed to this change,
with decode unchanged within noise. Long prefill improves 4.35% in that comparison and wins all 16 paired samples.

Delivery: only production `kernels/attn_prefill.py` changes, plus notes and experiment evidence. Arithmetic,
rounding, reduction order, model weights, decode code, host scheduling and the protocol remain intact.
`target/engine` is the fresh-checkout executable that passed the final evaluation; target/ remains ignored.
All comparison/profiling/build processes finished, the evaluation exited successfully, and the engine process
registry is empty. No harness, reference, SDK, model or installed library was modified.

## Round 003: medium-prefill dependency experiments

Starting point: round 002, score 2.2275; prefill 5.0 / 14.0 / 61.1 ms, decode 125.7 / 137.1 / 172 ms
at 256 / 1024 / 4096 (SESSION.md, five evaluation rounds). Decode is near the measured bandwidth ceiling.
This session first tests the 1024-token SBUF attention schedule: move probability DMA transposes to just after
softmax, independently of the later PV stage, with PV distances 2/3/4 and with/without deferred reciprocal.
This mechanism was helpful for the long path in round 002; deferred reciprocal alone was previously slower
at 1024. Per-head expressions and PV accumulation order stay fixed.
Estimate: a 5-10% gain in attention would save roughly 1-3% of this prefill; layer and full-engine measurements
will decide. A separate experiment stages scaled scores in SBUF instead of overwriting PSUM, with the same
scaling/minimum instruction, to release PSUM earlier. No gain is claimed before byte checks and timing.

Second mechanism: the 1024 layer currently exchanges each entire 1024-feature projection result after computing
it. Test 2- or 4-feature-tile groups exchanged as soon as they are ready, for both o_proj and down. The data,
addition and ownership do not change; only the point at which the two cores communicate changes. Hypothesis:
transfer latency can overlap the remaining independent projection tiles, at the cost of more sendrecv calls.
The upper bound is the existing exchange stalls (not yet measured in this session); reject if the extra calls
outweigh the overlap. These variants are separate from the attention experiments.

First batch: all eight variants produce identical residual and K/V bytes to the baseline on finite random
1024-token layer inputs (seed 0; every byte compared). Four alternating-order rounds, three batches of 51 queued
executions, same logical core: median microseconds per layer baseline 482.479; split-transpose/deferred-reciprocal
PV distances 2/3/4: 480.961 / 476.550 / 463.724; distance 3 with immediate reciprocal 476.047; scaled scores in
SBUF 487.346; distance 3 plus SBUF scores 475.349. Exchange groups of 2/4 tiles: 456.014 / 468.256.
The last round had large host-time outliers in three arms (baseline up to 1236.865 us); preserve the samples and
confirm promising candidates separately and end to end. These are diagnostics, not full-engine speedup claims.
Next candidate combines distance-4 attention and two-tile exchanges; also check single-tile exchange granularity.

Baseline profile (finite random inputs, third execution): 478.091 us device time, PE active 78.97%, VectorE 66.06%,
ScalarE 45.30%, no spills. Baseline full-engine A/A (8 interleaved rounds, both development cores): all tokens
identical, prefill ratios 0.9722 / 1.0028 / 0.9994, decode 0.9978 / 0.9971 / 0.9987. Final A/B swaps cores to
cancel their bias. Earlier prefetch experiment: load gate/up or both MLP matrices before attention instead of
after it; same values and operations, potentially overlapping weight reads with attention if SBUF capacity allows.

Second batch: combined schedule, one-tile exchanges, and both prefetch variants all pass full residual/K/V byte
comparisons. Clean four-round diagnostic (3 x 51): baseline 482.718 us, combined 454.523 (+6.20%), one-tile
exchange alone 454.151 (+6.29%), early gate/up prefetch 483.971 (-0.26%), early entire MLP 489.086 (-1.30%).
Prefetching rejected. Attention and exchange gains do not add: the combined schedule is only ~0.3% better than
two-tile exchange alone. Compare the simpler one-tile exchange engine before selecting production source.

Full-engine combined candidate vs parent: same.py, 40/40 prompts, all three request shapes identical, 71.8 s,
both processes exited 0. Profiles on the same finite random input sequence, third execution: baseline 478.091 us,
combined 453.594 us, one-tile exchange alone 448.311 us. Each reads 58,804,232 HBM bytes and writes 12,582,920;
no spills. Exchange-only is the selected simpler mechanism, pending release cleanup checks and request timing.
The production cleanup removes the always-true experiment branch and simplifies equivalent tile slice indices;
this is checked again because allocation/scheduling changes can affect NKI code generation.

Release cleanup verified: all residual and cache bytes identical. Four alternating rounds, 3 x 51 calls:
baseline 482.540 us, one-tile experimental source 455.131 us, cleaned release 453.865 us (+6.32%).
Selected production change is only `kernels/prefill_sb.py`: exchange each completed 128-feature o_proj/down tile
in the 1024-token chunked layer, so communication overlaps subsequent independent projection work. No changes
to expressions, reduction order, rounding, model weights, attention, host schedule, decode or other prompt buckets.

Full-engine release vs parent: same.py 40/40 prompts and all three request shapes identical (72.4 s), both
processes exited 0. This includes resets after longer histories, split continuations, and lengths around tiled
boundaries through 4097. Paired request timing runs next, with development-core assignments swapped.

Full-engine interleaved A/B, two 8-round runs with core assignments swapped, all tokens identical:
prefill ratios 1.035122 / 1.068247 / 1.002249, then 1.017660 / 1.060980 / 1.000997 at 256 / 1024 / 4096.
Balanced ratios: 1.026354 / 1.064607 / 1.001623. Balanced decode: 1.000672 / 1.000664 / 1.000429.
The changed 1024 bucket wins all 16 pairs; ratio range 1.02279..1.14299, standard deviation 0.02518.
Observed geometric prefill improvement 3.05%, decode +0.059%, combined +1.545%. The unchanged 256 bucket's
apparent 2.64% gain is within its A/A variation and is not attributed to this optimization. The clear targeted
result is +6.46% at 1024; other kernels are unchanged. Raw per-round samples are in experiments/round003/measurements.json.

A fresh source export (tracked files plus this session's additions, no build artifacts/cache) is compiling all
21 NEFFs with `bash build.sh`. Its native executable will be compared to the verified release and evaluated
against the reference before delivery.

Clean build succeeded: all 21 NEFFs compiled without cached artifacts, producing a 19 MB native executable.
`ldd` lists libnrt and standard system libraries only. Production kernel source exactly matches the fresh
export. The fresh executable has been copied to `engine/target/engine`; fresh/development token comparison
and the final reference evaluation follow.

Fresh vs development release: same.py 40/40 prompts, all three request shapes identical, 76.7 s, both processes
exited 0. The fresh executable in `engine/target/engine` has SHA-256
`e032be88a4551afc763c7d3a94ed608786fe0a26a7cdbb99651a689a4663f253`.
Final `enginesmith dev-eval --no-build --rounds 3 --lengths all` now runs on that executable.

### Round 003 final result: fresh executable, dev-eval 001

`enginesmith dev-eval --no-build --rounds 3 --lengths all` PASSED in 412.84 s. All 17 control gates pass:
six mismatching steps total, at most two in a control stream, worst reference rank 2. All nine timed streams
have zero mismatching steps. Overall 6/1690 mismatches, below the allowance of 169.

| case | candidate prefill ms, rounds | candidate decode ms (64 tokens), rounds | prefill speedup | decode speedup |
|---|---|---|---|---|
| p256 | 6.71 / 5.15 / 4.91 | 125.72 / 126.62 / 125.77 | 2.216 | 2.062 |
| p1024 | 13.26 / 13.44 / 13.14 | 137.14 / 137.27 / 137.02 | 2.779 | 3.200 |
| p4096 | 61.19 / 61.25 / 61.12 | 171.89 / 172.96 / 171.92 | 1.593 | 2.003 |

Score **2.249728**, prefill mean **2.140480**, decode mean **2.364552**. SESSION.md starts at 2.2275;
this development score is +1.00%. The p1024 prefill speedup improves from 2.607 to 2.779 (+6.59%), above the
4% case-improvement threshold, with the overall score improving. Reference timing varies and p256 has a
6.71 ms first-round outlier; the controlled parent comparison is the stronger attribution evidence:
**6.46% faster p1024 prefill, all 16 pairs faster, all tokens identical**. Decode is unchanged within noise.

Delivery: production changes only in `kernels/prefill_sb.py`, plus these notes and `experiments/round003/`.
The final native `target/engine` is the clean-build executable tested above; target/ remains ignored.
All build, profiling, comparison and evaluation processes have exited, and the session engine registry is empty.
No harness, reference, SDK, model or installed library was changed.
