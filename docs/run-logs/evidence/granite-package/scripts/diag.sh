#!/bin/bash
OUT=$1; mkdir -p $OUT
PKG=$HOME/code/tt-orchard/.claude/worktrees/granite-package/pkgsrc
source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>
MD=$HOME/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/d18cca4c121edb87d022116d281ce212c9136f57
REF=$HOME/orchard-runs/ibm-granite--granite-4.0-h-1b/reference
LONG=<pkg-run-dir>/ref/long1024.npz
cd /tmp
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
for P in fp32 bf16; do
  timeout 600 python $PKG/tools/evaluate.py agree --model-dir $MD --max-len 2048 --ref $REF --long $LONG --ssd-precision $P --out $OUT/agree-$P.json > $OUT/agree-$P.log 2>&1
  echo "agree $P rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
  grep -E '^Traceback|Error' $OUT/agree-$P.log | tail -3 | cut -c1-300
  python3 - <<PY
import json
try:
    d=json.load(open("$OUT/agree-$P.json"))
    print("$P", [round(p["rel_l2_mean"],4) for p in d["prompts"]], [round(p["top1"],3) for p in d["prompts"]], [p["free_run_prefix"] for p in d["prompts"]])
    l=d["long"]; print(" long chip", {k:round(v,4) for k,v in l["chip_vs_fp32"].items()}); print(" bins", [round(x,3) for x in l["rel_l2_by_128_positions"]["chip"]])
except Exception as e: print("no result", e)
PY
done
timeout 600 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len 2048 --prefill-sizes 128 --ssd-precision fp32 --out $OUT/bench-fp32.json > $OUT/bench-fp32.log 2>&1
echo "bench fp32 rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
python3 -c "import json;d=json.load(open('$OUT/bench-fp32.json'));print('bench fp32',[round(x,1) for x in d['decode']['tok_s_runs']])" 2>&1 | tail -2
echo "end $(date -Is)" | tee -a $OUT/conditions.txt
