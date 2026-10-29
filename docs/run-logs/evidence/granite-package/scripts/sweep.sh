#!/bin/bash
# Session 1: decode speed against the context limit, and agreement (4 prompts and a 1024-token sequence) at the default limit.
# Needs TT_VISIBLE_DEVICES from a gozer lease. Python is the pair venv with the host tt-metal exports (no ttl import in the default path).
OUT=$1; mkdir -p $OUT
PKG=$HOME/code/tt-orchard/.claude/worktrees/granite-package/pkgsrc
source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>
MD=$HOME/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/d18cca4c121edb87d022116d281ce212c9136f57
REF=$HOME/orchard-runs/ibm-granite--granite-4.0-h-1b/reference
cd /tmp
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
for L in 2048 128 512 1024 4096; do
  timeout 600 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len $L --prefill-sizes 128 --out $OUT/bench-L$L.json > $OUT/bench-L$L.log 2>&1
  echo "bench L=$L rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
  python3 -c "import json;d=json.load(open('$OUT/bench-L$L.json'));print(d['max_len'],round(d['decode']['mean'],2),round(d['decode']['min'],2),round(d['decode']['max'],2),d['prefill'])" 2>&1 | cut -c1-300
done
timeout 900 python $PKG/tools/evaluate.py agree --model-dir $MD --max-len 2048 --ref $REF --long <pkg-run-dir>/ref/long1024.npz --out $OUT/agree-L2048.json > $OUT/agree-L2048.log 2>&1
echo "agree L=2048 rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
grep -E '^\{|^Traceback|Error' $OUT/agree-L2048.log | cut -c1-400 | tail -8
echo "end $(date -Is)" | tee -a $OUT/conditions.txt
