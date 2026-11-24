#!/bin/bash
# Build the engine: target/engine (one native executable, every NEFF embedded, links libnrt only).
# Build time uses Python + NKI + neuronx-cc to compile the kernels (tools/build_neffs.py, NKI's documented route,
# which runs each kernel once on a free NeuronCore: ENGINESMITH_NEURON_CORE + 2 and + 3, or NEURON_BUILD_CORES).
# NEFFs are cached in target/neff-cache/ keyed by the kernel sources.
set -euo pipefail
cd "$(dirname "$0")"
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
NRT=/opt/aws/neuron
export PATH=$V/bin:$PATH
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export PYTHONPATH=$PWD/third_party:$PWD/kernels
mkdir -p build target
python tools/build_neffs.py
names=$(cat build/neffs.list)
{
  echo '.section .rodata'
  for n in $names; do
    echo '.balign 64'
    echo "neff_${n}_start: .incbin \"build/neff/${n}.neff\""
    echo "neff_${n}_end:"
    echo "neff_${n}_name: .asciz \"${n}\""
  done
  echo '.section .data.rel.ro,"aw"'
  echo '.balign 8'
  echo '.globl neff_table'
  echo 'neff_table:'
  for n in $names; do echo "  .quad neff_${n}_name, neff_${n}_start, neff_${n}_end"; done
  echo '.globl neff_count'
  echo '.balign 4'
  echo "neff_count: .long $(echo $names | wc -w)"
  echo '.section .note.GNU-stack,"",@progbits'
} > build/neffs.S
g++ -O2 -std=c++17 -I$NRT/include host/engine.cpp build/neffs.S -o target/engine \
  -L$NRT/lib -lnrt -Wl,-rpath,$NRT/lib
echo "built target/engine ($(du -h target/engine | cut -f1))"
