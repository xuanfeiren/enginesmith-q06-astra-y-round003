# Round 001 experiments

These patches and probes preserve rejected or exploratory work; they are not used by `build.sh`.
See `../../NOTES.md` for hypotheses, verification, measured outcomes, and the final implementation.

- `scheduling.patch`: independent environment-selected RoPE shuffle, V DMA transpose, wider chunks,
  chunk-local N-split C/D/E, and MLP scheduling trials.
- `mlp_pipeline.patch`: the short-prompt MLP schedule trial.
- `k_split.patch`: long-prompt chunk-local K-split C/D/E (passes reference gates, slower).
- `query_rows.patch`, `query_tables.patch`, `moving_weights.patch`: successive QKV layout trials.
- `rejected_accumulator_carry.patch`: the rejected first scaled-score reduction. It fails real-model
  warm-up; do not use its carried minimum across the masking instruction.
- `group.py`, `group_constants.patch`: grouped short-layer calls, host slicing, and the rejected shared-constant follow-up.
- `norm_probe.py`: public-NKI normalization reduction/broadcast probe. `NORM_MODE=0/1/2` selects
  original/PE/shuffle broadcast. Mode 3 records a rejected, unsupported SBUF DMA broadcast.
- `measurements.json`: A/A and interleaved A/B samples and full-engine comparison summaries.

Patch imports with laboratory module names describe the experiment composition; they are not production imports.
All numeric timing entries are measurements. No model weights, NEFFs, private paths, or evaluation prompts are included.
