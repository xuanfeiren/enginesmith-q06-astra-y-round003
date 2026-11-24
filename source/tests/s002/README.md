Session 002 lab tools (not used by build.sh). Paths inside are those of the session's scratch directory; adapt them.
- build_any.py / mkany.sh: compile any `@nki.jit` kernel with zero inputs of given shapes to a NEFF
  (`mkany.sh MODULE FUNC 'SPEC-JSON' OUTNAME [core]`, SPEC = [[shape, "bf16"|"f32"|"i32"], ...]).
- test_attn.py S NEFF: attention kernels (pos, qT, kT, v, oT) vs numpy fp32 through host/nrun.
- test_psb.py S NEFF_A NEFF_B: two prefill-layer NEFFs (pf_layer / pf_sb / pf_sbc) on the same random weights in
  both layouts; compares the residual and the K / V caches.
- bench.sh CORE NEFF...: queued us per execution through host/chain (zero inputs, pos:output_0 swap).
- tl.py ntff.json BUCKET_US T0 T1 [sg01]: per-bucket DMA GB/s and engine busy % from `neuron-explorer view
  --output-format json`; gaps.py: idle gaps of one engine's instruction stream and what it waited on.
- pe_probe.py, sr_probe.py, rsq_probe.py: instruction-cost probes (PE transposes / matmuls, sendrecv sizes, rsqrt
  precision per engine).
