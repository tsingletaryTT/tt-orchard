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
timeout 600 python $PKG/tools/evaluate.py agree --model-dir $MD --max-len 2048 --ref $REF --long $LONG --ssd-precision fp32r --out $OUT/agree-fp32r.json > $OUT/agree-fp32r.log 2>&1
echo "agree fp32r rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
grep -E '^Traceback|Error' $OUT/agree-fp32r.log | tail -3 | cut -c1-300
python3 - <<PY
import json
d=json.load(open("$OUT/agree-fp32r.json"))
print("fp32r", [round(p["rel_l2_mean"],4) for p in d["prompts"]], [round(p["top1"],3) for p in d["prompts"]], [p["free_run_prefix"] for p in d["prompts"]])
l=d["long"]; print(" long chip", {k:round(v,4) for k,v in l["chip_vs_fp32"].items()})
PY
for r in 1 2; do for P in bf16 fp32 fp32r; do
  timeout 600 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len 2048 --prefill-sizes 128 --ssd-precision $P --out $OUT/bench-$P-$r.json > $OUT/bench-$P-$r.log 2>&1
  echo "bench $P r$r rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
  python3 -c "import json;d=json.load(open('$OUT/bench-$P-$r.json'));print('bench $P',round(d['decode']['mean'],2),[round(x,1) for x in d['decode']['tok_s_runs']])" 2>&1 | tail -1
done; done
echo "end $(date -Is)" | tee -a $OUT/conditions.txt
