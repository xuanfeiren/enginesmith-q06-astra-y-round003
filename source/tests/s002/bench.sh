#!/bin/bash
# bench.sh core neff... : chain 51 steps x 3 reps each, prints median us/step
core=$1; shift
for k in "$@"; do
  r=$(NEURON_RT_VISIBLE_CORES=$core ./chain neff/$k.neff io 51 pos:output_0 3 2>&1 | grep CHAIN | awk '{print $(NF-1)}' | sort -n | sed -n 2p)
  echo "$k $r"
done
