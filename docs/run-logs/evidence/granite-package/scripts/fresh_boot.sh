#!/bin/bash
# Fresh-machine check of a staged package, then a calm third measurement session.
# 1. a venv built from the package's requirements.txt (made earlier, no cache), 2. an EMPTY Hugging Face cache (HF_HOME in a scratch directory),
# 3. `bash run.sh --self-test`, `bash run.sh` (first start downloads the pinned revision), the smoke test, the self-test again,
# 4. decode, TTFT, prefill (bench) from the same cache with the pair venv. One process at a time. Under a lease.
OUT=$1; PKGSRC=$2; mkdir -p $OUT
[ -n "$GOZER_LEASE" ] && gozer renew --expect 40m $GOZER_LEASE | tail -1
R=<pkg-run-dir>
export HF_HOME=$R/fresh/hf-home; rm -rf $HF_HOME; mkdir -p $HF_HOME
export TT_METAL_HOME=$HOME/tt-metal PORT=20110
VENV=$R/fresh/venv
PKG=$R/fresh/pkg; rm -rf $PKG; cp -r $PKGSRC $PKG
cd $PKG
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
( source $VENV/bin/activate; bash run.sh --self-test ) > $OUT/selftest-before.txt 2>&1; echo "selftest-before rc=$?" | tee -a $OUT/conditions.txt; cat $OUT/selftest-before.txt
T0=$(date +%s); echo "first start $(date -Is)" | tee -a $OUT/conditions.txt
( source $VENV/bin/activate; exec bash run.sh > $OUT/server.log 2>&1 ) &
SPID=$!
for i in $(seq 1 600); do grep -q '"event": "ready"' $OUT/server.log && break; kill -0 $SPID 2>/dev/null || break; sleep 2; done
T1=$(date +%s)
if grep -q '"event": "ready"' $OUT/server.log; then
  echo "ready after $((T1 - T0)) s (download of the weights and load)" | tee -a $OUT/conditions.txt; grep '"event": "ready"' $OUT/server.log | tee -a $OUT/conditions.txt
  $VENV/bin/python tools/smoke.py --port $PORT --out $OUT/smoke.json 2>&1 | tee $OUT/smoke.txt
  echo "smoke rc=${PIPESTATUS[0]} $(date -Is)" | tee -a $OUT/conditions.txt
else
  echo "server did not become ready after $((T1 - T0)) s" | tee -a $OUT/conditions.txt; grep -v "^$" $OUT/server.log | tail -25 | cut -c1-300
fi
kill $SPID 2>/dev/null; for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -0 $SPID 2>/dev/null && { kill -9 $SPID; echo "needed SIGKILL" | tee -a $OUT/conditions.txt; }; wait $SPID 2>/dev/null
( source $VENV/bin/activate; bash run.sh --self-test ) > $OUT/selftest-after.txt 2>&1; echo "selftest-after rc=$?" | tee -a $OUT/conditions.txt; cat $OUT/selftest-after.txt
du -sh $HF_HOME | tee -a $OUT/conditions.txt
# third session: bench from the same cache, pair venv
MD=$(ls -d $HF_HOME/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/*)
source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $TT_METAL_HOME)"
export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>
cd /tmp
unset HF_HOME
echo "bench start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
timeout 600 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len 2048 --prefill-sizes 128,512,1024 --out $OUT/bench-fast.json > $OUT/bench-fast.log 2>&1; echo "bench-fast rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
timeout 600 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len 2048 --prefill-sizes "" --no-trace --out $OUT/bench-untraced.json > $OUT/bench-untraced.log 2>&1; echo "bench-untraced rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
echo "end $(date -Is)" | tee -a $OUT/conditions.txt
