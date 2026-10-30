#!/bin/bash
# Serve Qwen3.5-35B-A3B on one Blackhole p300 board (two dies) with the fused decode bundle of raahemnabeel/qwen3-coder-next and a patched
# weight loader mounted over the image's loader. Nothing is changed inside the image. The weights are not in this package: make the model
# directory first with fetch_weights.py (see docs/running.md).
#
#   MODEL_DIR=/path/to/model ./run.sh
#
# Environment: MODEL_DIR (required); TT_VISIBLE_DEVICES (set by a gozer lease: the PCI addresses or device numbers you may use) or DEVICES (two device
# numbers, used only when there is no TT_VISIBLE_DEVICES, and checked against it when there is); WORK (caches and the patched loader, default ./work);
# IMAGE; PORT (default 8000); NAME (container name); MAX_MODEL_LEN (default 4096). The measured image, loader and patch hashes are in pins.env.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pins.env
. "${PINS:-$here/pins.env}"
IMAGE=${IMAGE:-$IMAGE_TAG}      # another tag is fine when it resolves to the recorded image id (checked below)
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the directory fetch_weights.py wrote}
WORK=${WORK:-$here/work}; PORT=${PORT:-8000}; NAME=${NAME:-qwen35-35b-a3b-p300}; MAXLEN=${MAX_MODEL_LEN:-4096}
PCI=${SYSFS_PCI:-/sys/bus/pci/devices}

die() { echo "run.sh: $*" >&2; exit 2; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

[ -f "$MODEL_DIR/model.safetensors" ] || die "$MODEL_DIR has no model.safetensors; run fetch_weights.py first"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE is not on this machine. Pull the package that ships it: tt-model pull raahemnabeel/qwen3-coder-next"
have=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[ "$have" = "$IMAGE_ID" ] || die "image $IMAGE has id $have, which differs from the recorded $IMAGE_ID; the patch was made for the recorded image"

# Device nodes. The chips this process may use are the ones named in TT_VISIBLE_DEVICES (a gozer lease exports PCI addresses there).
# Each address is mapped to its /dev/tenstorrent/N node through sysfs. DEVICES, when given, must name exactly the leased nodes.
# With no TT_VISIBLE_DEVICES there is no lease to check against, so DEVICES must be given.
nodes=()
if [ -n "${TT_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra ids <<< "$TT_VISIBLE_DEVICES"
  for id in "${ids[@]}"; do
    id=${id// /}
    if [[ "$id" =~ ^[0-9]+$ ]]; then nodes+=("$id")
    else
      n=$(ls "$PCI/$id/tenstorrent" 2>/dev/null | sed -n 's/^tenstorrent!\([0-9][0-9]*\)$/\1/p' | head -1)
      [ -n "$n" ] || die "TT_VISIBLE_DEVICES names $id and no /dev/tenstorrent node is behind it"
      nodes+=("$n")
    fi
  done
  if [ -n "${DEVICES:-}" ]; then
    want=$(printf '%s\n' $DEVICES | sort -n | tr '\n' ' '); got=$(printf '%s\n' "${nodes[@]}" | sort -n | tr '\n' ' ')
    [ "$want" = "$got" ] || die "DEVICES ($want) is not the leased set ($got); drop DEVICES or take a lease for those chips"
  fi
elif [ -n "${DEVICES:-}" ]; then
  read -ra nodes <<< "$DEVICES"
else
  die "set TT_VISIBLE_DEVICES (a gozer lease does this) or DEVICES to the two device numbers of one board"
fi
[ "${#nodes[@]}" = 2 ] || die "this package serves exactly two dies (one p300 board); got ${#nodes[@]}"
devs=(); vis=(); i=0
for n in "${nodes[@]}"; do [ -e "${DEV_ROOT:-/dev/tenstorrent}/$n" ] || die "/dev/tenstorrent/$n is not a device"; devs+=(--device "/dev/tenstorrent/$n"); vis+=("$i"); i=$((i+1)); done

mkdir -p "$WORK/cache" "$WORK/tensors" "$WORK/weights"
if [ ! -f "$WORK/weights.py" ]; then
  docker run --rm --entrypoint cat "$IMAGE" "$LOADER" > "$WORK/weights.orig.py"
  [ "$(sha "$WORK/weights.orig.py")" = "$ORIG_SHA" ] || die "the image's loader is not the file the patch was made for"
  patch -s -o "$WORK/weights.py" "$WORK/weights.orig.py" < "$here/weights_qwen35_moe.patch"
fi
# checked on every launch: a patched loader left in WORK from an earlier run could have been edited since
[ "$(sha "$WORK/weights.py")" = "$PATCHED_SHA" ] || die "$WORK/weights.py does not match the measured patched loader; delete it and start again"

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
