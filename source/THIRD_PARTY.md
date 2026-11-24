# Third-party code

| project | files in this engine | license | notes |
|---|---|---|---|
| AWS NKI Library (nkilib), github.com/aws-neuron/nki-library, the copy shipped in the neuronx-cc 2.27.5334 wheel (`nkilib/version`: 0.0.0.0dev0+3b542be2) | `third_party/nkilib/` (unchanged copies of `core/attention/attention_cte.py`, the `core/utils/` modules it imports: `allocator`, `common_types`, `interleave_copy`, `kernel_assert`, `kernel_helpers`, `logging`, `modular_allocator`, `stream_shuffle_broadcast`, `tree_logger`, and the package `__init__` files) | Apache-2.0 (license header kept in every file) | used at build time only: `kernels/prefill_lnc2.py` inlines `_attention_cte` (causal prefill flash attention) into the prefill-layer NEFF. Public NKI API only. `tools/neff_one.py` asserts that the build imports this copy and not the installed wheel. |

The decode kernels, the rest of the prefill kernels and the host code are this engine's own (originally written in the
NKI kernel lab session, see FINDINGS.md).
