# Round 003 experiment evidence

The production change is in `kernels/prefill_sb.py`: the chunked 1024-token layer exchanges each completed
128-feature projection tile immediately, overlapping communication with subsequent independent matmuls.
Both o_proj and down use this schedule. Per-element values, additions and reduction order are unchanged.

All patches are against the round-002 parent. `build.sh` does not read this directory.

- `exchange1.patch`: one feature tile per exchange; selected mechanism before source cleanup.
- `exchange2.patch`, `exchange4.patch`: coarser groups; faster than baseline, less effective than single tiles.
- `split2_late.patch`, `split3_late.patch`, `split4_late.patch`: earlier probability DMA transposes, with PV
  delayed by 2/3/4 work items and reciprocal delayed until PV. The distance-4 variant helped alone.
- `split3_immediate.patch`: split transpose/PV with the original immediate reciprocal.
- `scores_sbuf.patch`, `split3_sbuf.patch`: scaled scores stored in SBUF, alone or with distance-3 scheduling.
- `combined.patch`: distance-4 attention plus two-tile exchanges. No material gain over exchanges alone;
  the simpler exchange-only implementation was selected.
- `prefetch_gate.patch`, `prefetch_mlp.patch`: load gate/up or all MLP weights before attention; rejected.

`measurements.json` contains raw diagnostic samples and full-request checks/timings. Kernel diagnostics compare
all residual and K/V output bytes on finite random inputs (seed 0) before timing. Timings use the existing
`../round002/bench_multi.cpp`: four alternating-order rounds, three batches of 51 queued executions per arm,
all NEFFs loaded once on the same development core, under the shared accelerator lock. All numbers are measured.
The first batch has several large host-time outliers; its complete samples are retained. Subsequent clean
comparisons and full protocol timings determine the selected change. The device profiles use the third
execution of the same finite random input sequence; profile totals are separate from uninstrumented timing.

Full engine checks use the brief's `same.py` and interleaved `ab.py`; core assignments are swapped for the
paired comparison. The authoritative reference check is `enginesmith dev-eval`. See `../../NOTES.md` for results.

The cleaned production source passed full parent token checks, both core assignments of request timing, a fresh
21-NEFF build and fresh/development token identity, and three-round all-length dev-eval. `final-dev-eval.json`
records its aggregate correctness and timing data (judged token IDs omitted). Final controlled p1024 prefill
speedup over the parent is 1.064607; final reference-relative score is 2.249728. See NOTES for variability limits.
