#!/bin/bash
# Session F2: decode speed against the context limit (traced), and untraced at the default limit. fp32 state (the default).
OUT=$1; mkdir -p $OUT
PKG=$HOME/code/tt-orchard/.claude/worktrees/granite-package/pkgsrc
source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>
MD=$HOME/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/d18cca4c121edb87d022116d281ce212c9136f57
cd /tmp
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
for spec in "512:" "2048:" "1024:" "4096:" "2048:--no-trace"; do
  L=${spec%%:*}; X=${spec#*:}; tag=L$L${X:+-untraced}
  timeout 400 python $PKG/tools/evaluate.py bench --model-dir $MD --max-len $L --prefill-sizes "" $X --out $OUT/bench-$tag.json > $OUT/bench-$tag.log 2>&1
  echo "bench $tag rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
  python3 -c "import json;d=json.load(open('$OUT/bench-$tag.json'));print('$tag',round(d['decode']['mean'],2),[round(x,1) for x in d['decode']['tok_s_runs']])" 2>&1 | tail -1
done
echo "end $(date -Is)" | tee -a $OUT/conditions.txt
