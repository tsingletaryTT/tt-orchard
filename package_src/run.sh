#!/bin/bash
# Serve Qwen3.5-35B-A3B on one Blackhole p300 board (two dies) with the fused decode bundle of raahemnabeel/qwen3-coder-next and a patched
# weight loader mounted over the image's loader. Nothing is changed inside the image. The weights are not in this package: make the model
# directory first with fetch_weights.py (see docs/running.md).
#
#   MODEL_DIR=/path/to/model ./run.sh
#
# Environment: MODEL_DIR (required), IMAGE, WORK (caches and the patched loader; default ./work), DEVICES (device numbers of the board's two
# dies, default "0 1"), PORT (default 8000), NAME (container name), MAX_MODEL_LEN (default 4096).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-tt-model/qwen3-coder-next:0c170421ca1b}
IMAGE_ID=sha256:0c170421ca1b5b95f17ebf28d0c8256ccb8c5d320957bc8c4e72e7a07c43acfb      # the image this package was measured on
LOADER=/opt/tt-metal/models/autoports/qwen_qwen3_coder_next/tt/weights.py
ORIG_SHA=a8be9a77d9190b6d13fa64a57bde649b6b7885b0711da6b2825cf65e924c8245            # the image's loader before the patch
PATCHED_SHA=1b94978020b09501cdc404d55041f143b67258dbb609c485e1ac68bae54e8a7b          # and after
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the directory fetch_weights.py wrote}
WORK=${WORK:-$here/work}; DEVICES=${DEVICES:-"0 1"}; PORT=${PORT:-8000}; NAME=${NAME:-qwen35-35b-a3b-p300}; MAXLEN=${MAX_MODEL_LEN:-4096}

[ -f "$MODEL_DIR/model.safetensors" ] || { echo "run.sh: $MODEL_DIR has no model.safetensors; run fetch_weights.py first" >&2; exit 2; }
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "run.sh: image $IMAGE is not on this machine. Pull the package that ships it: tt-model pull raahemnabeel/qwen3-coder-next" >&2; exit 2
fi
have=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[ "$have" = "$IMAGE_ID" ] || echo "run.sh: warning: image id $have differs from the measured $IMAGE_ID; the patch may not apply" >&2

mkdir -p "$WORK/cache" "$WORK/tensors" "$WORK/weights"
if [ ! -f "$WORK/weights.py" ]; then
  docker run --rm --entrypoint cat "$IMAGE" "$LOADER" > "$WORK/weights.orig.py"
  [ "$(sha256sum "$WORK/weights.orig.py" | cut -d' ' -f1)" = "$ORIG_SHA" ] || { echo "run.sh: the image's loader is not the file the patch was made for" >&2; exit 2; }
  patch -s -o "$WORK/weights.py" "$WORK/weights.orig.py" < "$here/weights_qwen35_moe.patch"
  [ "$(sha256sum "$WORK/weights.py" | cut -d' ' -f1)" = "$PATCHED_SHA" ] || { echo "run.sh: the patched loader does not match the measured one" >&2; exit 2; }
fi

devs=(); vis=(); i=0
for d in $DEVICES; do devs+=(--device "/dev/tenstorrent/$d"); vis+=("$i"); i=$((i+1)); done
[ "$i" = 2 ] || { echo "run.sh: this package serves exactly two dies (one p300 board); DEVICES has $i" >&2; exit 2; }

exec docker run --rm --name "$NAME" "${devs[@]}" --ipc host --security-opt label=disable \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G --user "$(id -u):$(id -g)" -p "127.0.0.1:$PORT:$PORT" \
  -v "$MODEL_DIR:/m:ro" -v "$WORK/cache:/cache" -v "$WORK/tensors:/tensor-cache" -v "$WORK/weights:/weight-cache" \
  -v "$WORK/weights.py:$LOADER:ro" \
  -e HF_MODEL=/m -e HF_HUB_OFFLINE=1 -e ARCH_NAME=blackhole -e MESH_DEVICE=P300 -e "TT_VISIBLE_DEVICES=$(IFS=,; echo "${vis[*]}")" \
  -e TT_CACHE_PATH=/tensor-cache -e TT_DIT_CACHE_DIR=/weight-cache \
  -e QWEN35_ATTN_DECODE_OPS=1 -e QWEN35_CHUNKED_PREFILL=1 -e QWEN35_DECODE_REPLICATED=1 -e QWEN35_GDN_DECODE_OPS=1 \
  -e QWEN35_MOE_FUSED_DECODE=1 -e QWEN35_PROGRAM_GUARD=1 -e QWEN35_ROWS_TOPK=1 -e QWEN35_TRACE_REGION_MIB=768 \
  "$IMAGE" vllm serve /m --served-model-name m --max-model-len "$MAXLEN" --max-num-seqs 1 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 805306368, "l1_small_size": 24576, "fabric_config": "FABRIC_1D"}}' \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048 --port "$PORT"
