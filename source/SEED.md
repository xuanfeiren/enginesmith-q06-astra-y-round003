# Seed: the NKI kernel lab

This directory is the work of a kernel lab session, not yet an engine. It holds measured, tested NKI kernels for
Qwen3-0.6B on this machine (see FINDINGS.md): a whole decode step in one NEFF (kernels/step_lnc2.py, 2.23 ms/step at
a 512-position bucket), whole-model prefill (kernels/prefill_lnc2.py, 7.5 / 19.8 / 76.3 ms at 256 / 1,024 / 4,096
tokens), C++ drivers on libnrt (host/), and tests against HF transformers.

To turn it into the engine: a native executable that speaks the EngineSmith protocol (build.sh -> target/engine),
with every NEFF embedded, built through NKI\x27s documented route only (NKI_ARTIFACTS_DIR + @nki.jit, see the skill\x27s
backends/nki-cpp/runtime.md; the lab\x27s kernels/nkibuild.py uses private nki entry points and must be replaced).
