#!/bin/bash
# Boot check of a staged (or pulled) package under a lease: start ./run.sh from the package directory with the package's own venv,
# wait for the ready line, run tools/smoke.py against it, then stop the server by its exact pid.
# usage: bash boot.sh PKG_DIR OUT_DIR [kernel]     (the third argument adds the optional kernel-mode speed check)
PKG=$1; OUT=$2; mkdir -p $OUT
VENV=<pkg-run-dir>/venv
export TT_METAL_HOME=$HOME/tt-metal
export PORT=20110
cd $PKG
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
( source $VENV/bin/activate; exec bash run.sh > $OUT/server.log 2>&1 ) &
SPID=$!
for i in $(seq 1 120); do grep -q '"event": "ready"' $OUT/server.log && break; kill -0 $SPID 2>/dev/null || break; sleep 2; done
grep '"event": "ready"' $OUT/server.log | tee -a $OUT/conditions.txt
if grep -q '"event": "ready"' $OUT/server.log; then
  $VENV/bin/python tools/smoke.py --port $PORT --out $OUT/smoke.json 2>&1 | tee $OUT/smoke.txt
  echo "smoke rc=${PIPESTATUS[0]} $(date -Is)" | tee -a $OUT/conditions.txt
else
  echo "server did not become ready"; tail -20 $OUT/server.log | cut -c1-300
fi
# stop the server: the pid recorded in the ready line is the python process
RPID=$(grep '"event": "ready"' $OUT/server.log | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['pid'])" 2>/dev/null)
[ -n "$RPID" ] && kill "$RPID" 2>/dev/null
for i in $(seq 1 30); do kill -0 ${RPID:-0} 2>/dev/null || break; sleep 1; done
[ -n "$RPID" ] && kill -0 "$RPID" 2>/dev/null && { kill -9 "$RPID"; echo "server needed SIGKILL" | tee -a $OUT/conditions.txt; }
wait $SPID 2>/dev/null
if [ "$3" = "kernel" ]; then
  # optional kernel modes: needs ttl, so use the pair venv; the package's own runtime and kernels, bf16 state, decode speed only
  source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
  eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
  export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>
  MD=$(ls -d $HOME/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/*)
  cd /tmp
  for m in composed state_full_ro; do
    timeout 400 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len 2048 --prefill-sizes "" --ssd $m --ssd-precision bf16 --out $OUT/kernel-bench-$m.json > $OUT/kernel-bench-$m.log 2>&1
    echo "kernel bench $m rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
    python3 -c "import json;d=json.load(open('$OUT/kernel-bench-$m.json'));print('$m',round(d['decode']['mean'],2),[round(x,1) for x in d['decode']['tok_s_runs']])" 2>&1 | tail -1
  done
fi
echo "end $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
