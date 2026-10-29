#!/bin/bash
# Boot check of a staged (or pulled) package under a lease: start `bash run.sh` from the package directory with the package's own venv,
# wait for the ready line, run tools/smoke.py against it, then stop the server by the pid this script launched.
# usage: GRANITE_VENV=<venv> TT_METAL_HOME=<tt-metal> bash boot.sh PKG_DIR OUT_DIR [kernel]
#   READY_TRIES / READY_SLEEP (default 120 / 2 s) bound the wait for the ready line.
#   The optional third argument adds the kernel-mode speed check; it needs PAIR_DIR (the tt-lang pair venv directory) and SAFETENSORS_DIR.
PKG=$1; OUT=$2; mkdir -p "$OUT"
VENV=${GRANITE_VENV:?set GRANITE_VENV to the venv of the package}
export TT_METAL_HOME=${TT_METAL_HOME:?set TT_METAL_HOME}
export PORT=${PORT:-20110}
cd "$PKG" || exit 2
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a "$OUT/conditions.txt"
( source "$VENV/bin/activate"; exec bash run.sh > "$OUT/server.log" 2>&1 ) &
SPID=$!
for i in $(seq 1 "${READY_TRIES:-120}"); do
  grep -q '"event": "ready"' "$OUT/server.log" 2>/dev/null && break
  kill -0 "$SPID" 2>/dev/null || break
  sleep "${READY_SLEEP:-2}"
done
grep '"event": "ready"' "$OUT/server.log" 2>/dev/null | tee -a "$OUT/conditions.txt"
if grep -q '"event": "ready"' "$OUT/server.log" 2>/dev/null; then
  "$VENV/bin/python" tools/smoke.py --port "$PORT" --out "$OUT/smoke.json" 2>&1 | tee "$OUT/smoke.txt"
  echo "smoke rc=${PIPESTATUS[0]} $(date -Is)" | tee -a "$OUT/conditions.txt"
else
  echo "server did not become ready"; tail -20 "$OUT/server.log" 2>/dev/null | cut -c1-300
fi
# Stop the server by the pid this script launched ($SPID: the subshell execs run.sh, which execs python, so it is the server).
# The pid in the ready line is not used: it does not exist when the server never became ready.
kill "$SPID" 2>/dev/null
for i in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
kill -0 "$SPID" 2>/dev/null && { kill -9 "$SPID"; echo "server needed SIGKILL" | tee -a "$OUT/conditions.txt"; }
wait "$SPID" 2>/dev/null
if [ "$3" = "kernel" ]; then
  # optional kernel modes need ttl, so use the pair venv; the package's own runtime and kernels, bf16 state, decode speed only
  source "${PAIR_DIR:?set PAIR_DIR}/venv-light/bin/activate"
  eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir "$TT_METAL_HOME")"
  export PYTHONPATH=$PYTHONPATH:${SAFETENSORS_DIR:-}
  MD=$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/*)
  cd /tmp
  for m in composed state_full_ro; do
    timeout 400 python "$PKG/tools/evaluate.py" bench --model-dir "$MD" --max-len 2048 --prefill-sizes "" --ssd $m --ssd-precision bf16 --out "$OUT/kernel-bench-$m.json" > "$OUT/kernel-bench-$m.log" 2>&1
    echo "kernel bench $m rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a "$OUT/conditions.txt"
  done
fi
echo "end $(date -Is) load $(cat /proc/loadavg)" | tee -a "$OUT/conditions.txt"
