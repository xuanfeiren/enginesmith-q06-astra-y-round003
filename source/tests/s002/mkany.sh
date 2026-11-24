#!/bin/bash
# mkany.sh module func 'spec-json' outname [core]
set -e
E=/home/ubuntu/es-dev/tasks/q06-lab/runs/q06-lab-20261103-231731/work/002
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
art=$E/exp/art/$4; rm -rf $art; mkdir -p $art $E/exp/neff
NKI_ARTIFACTS_DIR=$art NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_VISIBLE_CORES=${5:-43} \
 PYTHONPATH=$E/exp:$E/engine/third_party:$E/engine/kernels $V/bin/python $E/exp/build_any.py "$1" "$2" "$3" > $art/build.log 2>&1 || { grep -v "^ " $art/build.log | tail -8 | cut -c1-400; exit 1; }
cp $art/kernel.neff $E/exp/neff/$4.neff; echo built $4
