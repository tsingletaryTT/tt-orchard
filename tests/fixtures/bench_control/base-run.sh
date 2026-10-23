#!/usr/bin/env bash
# Serve this model on TT hardware. Assumes ./install.sh has been run.
set -euo pipefail
HERE="/home/ttuser/.cache/tt-model/models/episod/qwen3.8-27b-dflash2-p300"
VENV="${VENV:-$HERE/venv}"
PYBIN="$VENV/bin/python"

# Locate ttnn WITHOUT importing it — importing loads _ttnn.so, which is exactly what needs the
# LD_PRELOAD below (chicken-and-egg). find_spec resolves the path without executing the module.
TTNN_DIR="$("$PYBIN" -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("ttnn").origin))')"
# _ttnncpp.so lives in ttnn.libs/ for an auditwheel-repaired (portable) wheel, or build/lib/ for a
# raw one; preload it to avoid the glibc "static TLS block" error on late dlopen.
# Prefer the auditwheel-vendored copy in *.libs/ (that's the one _ttnn.so actually loads via
# RPATH); fall back to build/lib for a raw (unrepaired) wheel — e.g. the plain `ttnn` PyPI wheel
# a v6 thin bundle installs, which isn't auditwheel-repaired at all.
# `|| true` on each probe: under `set -e -o pipefail`, a `ls <no-match> | head -1` pipe fails
# (pipefail surfaces ls's nonzero exit) and set -e would kill the script on THIS line, before the
# fallback below — or the deliberate `:?` error a few lines down — ever runs.
LD_PRELOAD="$(ls "$TTNN_DIR"/../*.libs/_ttnncpp*.so 2>/dev/null | head -1)" || true
[ -n "$LD_PRELOAD" ] || LD_PRELOAD="$(ls "$TTNN_DIR"/build/lib/_ttnncpp*.so 2>/dev/null | head -1)" || true
export LD_PRELOAD="${LD_PRELOAD:?could not locate _ttnncpp.so in the ttnn install}"
export TT_METAL_HOME="$TTNN_DIR"
# EXTRA_MODELS_DIR is a PARENT of per-model bundle folders; the plugin scans its children for
# each vllm_metadata.json (so the metadata lives in vllm_models/<model>/, not the bundle root).
export EXTRA_MODELS_DIR="$HERE/vllm_models"
export TT_VLLM_BUILTIN_MODELS=0
# Do NOT set VLLM_PLUGINS: it is an ALLOW-LIST — setting it suppresses the vllm.general_plugins
# group, so the model's tool/reasoning-parser overrides would silently not load. The TT platform
# + model registry load via entry points without it.
export PYTHONPATH="$HERE:${PYTHONPATH:-}"   # resolves the adapter/model imports
export MESH_DEVICE="${MESH_DEVICE:-P150x2}"
NCHIPS=2
if [ -z "${TT_METAL_VISIBLE_DEVICES:-}" ]; then
  if [ -n "${TT_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -ra _GRANT <<< "$TT_VISIBLE_DEVICES"
    if [ "${#_GRANT[@]}" -lt "$NCHIPS" ]; then
      echo "run.sh: this model needs $NCHIPS chip(s) but TT_VISIBLE_DEVICES grants ${#_GRANT[@]} ($TT_VISIBLE_DEVICES)" >&2
      exit 1
    fi
    TT_VISIBLE_DEVICES="$(IFS=,; echo "${_GRANT[*]:0:$NCHIPS}")"
    export TT_VISIBLE_DEVICES
    TT_METAL_VISIBLE_DEVICES="0,1"
  else
    TT_METAL_VISIBLE_DEVICES="0,1"
  fi
fi
export TT_METAL_VISIBLE_DEVICES
# HERMETIC RUNTIME: keep every cache/home INSIDE the folder wall, so serving writes and reads
# nothing outside it (the ttnn tensor cache even DEFAULTS to a hard-coded /mnt/... path upstream —
# a classic other-machine leak we must override). Each is overridable if the operator sets it.
export HF_HOME="${HF_HOME:-$HERE/.hf}"                  # HF weights + hub cache
export TT_CACHE_PATH="${TT_CACHE_PATH:-$HERE/.tt_cache}"    # ttnn weight/tensor cache
export TT_CACHE_HOME="${TT_CACHE_HOME:-$HERE/.tt_cache}"    # override upstream's /mnt/... default
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HERE/.cache}"     # generic catch-all (triton, etc.)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HERE/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HERE/.cache/inductor}"
export HF_MODEL="/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
export TT_MODEL_WEIGHTS_REVISION="${TT_MODEL_WEIGHTS_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
export ARCH_NAME="blackhole"
export QWEN36_SKIP_VISION="1"
export QWEN_SDPA_BF8="1"
export TORCHDYNAMO_DISABLE="1"
export TT_QWEN35_TEXT_VER="qwen36_blackhole"
export VLLM_RPC_TIMEOUT="900000"
export VLLM_CONFIGURE_LOGGING="1"
export QWEN36_DRAFTER=""
export DFLASH_WEIGHTS="incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4"
export QWEN36_DFLASH_TP="1"
export QWEN36_DFLASH_BLOCK="8"
export QWEN36_DFLASH_FOLD_SEED="1"
export QWEN36_DFLASH_SERVE_BLOCK="32"
export QWEN36_PREFILL_BUCKET_TRACE="0"
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="0"
export QWEN36_GDN_SPEC_FUSED="1"
export QWEN36_MAX_TOKENS_ALL_USERS="262144"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model /home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --max_num_seqs 4 --block_size 64 --max_model_len 262144 --additional-config '{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}' --max-num-batched-tokens 65536 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning_parser qwen3 --no-async-scheduling "$@")
# TT_MODEL_PRINT=1 (set by `tt-model serve --print`) echoes the fully-resolved command+env
if [ "${TT_MODEL_PRINT:-0}" = "1" ]; then
  printf 'LD_PRELOAD=%s TT_METAL_HOME=%s EXTRA_MODELS_DIR=%s MESH_DEVICE=%s HF_MODEL=%s
  %s
' \
    "$LD_PRELOAD" "$TT_METAL_HOME" "$EXTRA_MODELS_DIR" "$MESH_DEVICE" "${HF_MODEL:-}" "${CMD[*]}"
  exit 0
fi
exec "${CMD[@]}"
