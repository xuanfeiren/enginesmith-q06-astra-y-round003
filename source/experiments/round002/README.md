# Round 002 experiment evidence

Only `kernels/attn_prefill.py` is changed in production. These patches preserve the measured alternatives;
`build.sh` does not use this directory. See `../../NOTES.md` for mechanisms, checks and results.

Patches use the round-001 parent (`18fe812`) unless their prerequisite is listed:

- `early_transpose.patch`: split the probability DMA transpose from PV; PV depth 3.
- `delayed_reciprocal.patch`: store ScalarE sums directly, take reciprocal when PV consumes the pair.
- `direct_max.patch`: apply after delayed reciprocal; use the already-reduced maximum directly.
- `combined_schedule.patch`: the preceding mechanisms together, before production helper cleanup.
- `wide_transpose.patch`: apply after combined schedule; rejected 32-block DMA batches.
- `sequence_cde.patch`: rejected token partition for o_proj and the MLP, 256-token chunks.
- `sequence_mlp.patch`: rejected token partition for only norm2/gate/up/down, 512-token chunks.
- `short_reciprocal.patch`: apply after delayed reciprocal and direct max; short-path counterpart,
  helpful at 256 and slower at 1024. Production restricts it to 256.

`measurements.json` preserves the diagnostic samples (microseconds per queued execution) and later request
comparisons. Profile total/active times are seconds; byte and instruction counts have explicit field names.
All timings are measurements; hypotheses and estimates are explicitly labeled in NOTES.

`bench_multi.cpp` loads all named NEFFs once on the visible logical NeuronCore, warms each with 10 executions,
and runs three batches of 51 queued executions per arm in each alternating-order round. Usage:
`bench_multi ROUNDS COUNT NAME NEFF [NAME NEFF ...]`. Compile like `host/chain.cpp` (libnrt and its public headers).
Use the shared GPU lock and the session's assigned development core. Only use this zero-input diagnostic with
stable, fixed-shape prefill/attention kernels; it deliberately does not implement decode position ping-pong.
Random-layer residual/KV byte checks precede timing. Full-engine checks and timings use the brief's protocol tools.
