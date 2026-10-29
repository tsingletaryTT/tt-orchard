#!/bin/bash
# Session F1: agreement (4 prompts and a 2048-token sequence), decode and prefill speed at the default context limit (2048, fp32 state),
# and the optional kernel mode against the composed bf16 path at the same limit.
OUT=$1; mkdir -p $OUT
PKG=$HOME/code/tt-orchard/.claude/worktrees/granite-package/pkgsrc
source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>
MD=$HOME/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/d18cca4c121edb87d022116d281ce212c9136f57
REF=$HOME/orchard-runs/ibm-granite--granite-4.0-h-1b/reference
LONG=<pkg-run-dir>/ref/long2048.npz
cd /tmp
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
timeout 900 python $PKG/tools/evaluate.py agree --model-dir $MD --max-len 2048 --ref $REF --long $LONG --out $OUT/agree-fp32.json > $OUT/agree-fp32.log 2>&1
echo "agree rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
python3 - <<PY
import json
d=json.load(open("$OUT/agree-fp32.json"))
print("fp32", [round(p["rel_l2_mean"],4) for p in d["prompts"]], [round(p["top1"],3) for p in d["prompts"]], [p["free_run_prefix"] for p in d["prompts"]])
l=d["long"]; print(" long", {k:round(v,4) for k,v in l["chip_vs_fp32"].items()}, "bf16cpu", {k:round(v,4) for k,v in l["bf16_cpu_vs_fp32"].items()})
PY
timeout 700 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len 2048 --prefill-sizes 128,512,1024 --out $OUT/bench-fp32.json > $OUT/bench-fp32.log 2>&1
echo "bench fp32 rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
python3 - <<PY
import json
d=json.load(open("$OUT/bench-fp32.json"))
print("decode", round(d["decode"]["mean"],2), [round(x,1) for x in d["decode"]["tok_s_runs"]], "first token", round(d["first_token_s_12_token_prompt"]["median"],3))
for p in d["prefill"]: print(" prefill", p["n"], round(p["loop_read_last"]["median_tok_s"],1), round(p["loop_read_every_step"]["median_tok_s"],1))
PY
for spec in "composed:bf16" "state_full_ro:bf16"; do
  m=${spec%%:*}
  timeout 400 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len 2048 --prefill-sizes "" --ssd $m --ssd-precision bf16 --out $OUT/bench-$m-bf16.json > $OUT/bench-$m-bf16.log 2>&1
  echo "bench $m bf16 rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
  python3 -c "import json;d=json.load(open('$OUT/bench-$m-bf16.json'));print('$m',round(d['decode']['mean'],2),[round(x,1) for x in d['decode']['tok_s_runs']])" 2>&1 | tail -1
done
echo "end $(date -Is)" | tee -a $OUT/conditions.txt
