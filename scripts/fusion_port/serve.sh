#!/bin/bash
# Serve a local Qwen3.5-35B-A3B checkpoint dir on the Coder-Next fused bundle's code (a patched copy mounted over the autoport dir;
# the published image and the coder's container are untouched). Usage: serve.sh NAME MODELDIR PORT [extra docker args...]
set -euo pipefail
S=${ORCHARD_SCRATCH:?set the scratch directory}
NAME=$1; MODEL=$2; PORT=$3; shift 3
IMG=tt-model/qwen3-coder-next:0c170421ca1b
exec docker run --rm --name "$NAME" --device /dev/tenstorrent/2 --device /dev/tenstorrent/3 --ipc host --security-opt label=disable \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G --user 1000:1000 -p 127.0.0.1:$PORT:$PORT \
  -v $MODEL:/m:ro -v $S/run/cache:/cache -v $S/run/tensors:/tensor-cache -v $S/run/weights:/weight-cache \
  -v $S/code/qwen_qwen3_coder_next:/opt/tt-metal/models/autoports/qwen_qwen3_coder_next:ro \
  -e HF_MODEL=/m -e HF_HUB_OFFLINE=1 -e ARCH_NAME=blackhole -e MESH_DEVICE=P300 -e TT_VISIBLE_DEVICES=0,1 \
  -e TT_CACHE_PATH=/tensor-cache -e TT_DIT_CACHE_DIR=/weight-cache \
  -e QWEN35_ATTN_DECODE_OPS=1 -e QWEN35_CHUNKED_PREFILL=1 -e QWEN35_DECODE_REPLICATED=1 -e QWEN35_GDN_DECODE_OPS=1 \
  -e QWEN35_MOE_FUSED_DECODE=1 -e QWEN35_PROGRAM_GUARD=1 -e QWEN35_ROWS_TOPK=1 -e QWEN35_TRACE_REGION_MIB=768 "$@" \
  $IMG vllm serve /m --served-model-name m --max-model-len 4096 --max-num-seqs 1 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 805306368, "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}' \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048 --port $PORT
