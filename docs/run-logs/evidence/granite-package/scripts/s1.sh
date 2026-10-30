#!/bin/bash
# Session S1: decode, TTFT and prefill (with and without the no-head trace), untraced decode, agreement on three text types.
OUT=$1; mkdir -p $OUT
PKG=$HOME/code/tt-orchard/.claude/worktrees/granite-package/pkgsrc
source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>
MD=$HOME/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/d18cca4c121edb87d022116d281ce212c9136f57
REF=$HOME/orchard-runs/ibm-granite--granite-4.0-h-1b/reference
R=<pkg-run-dir>/ref
cd /tmp
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
run() { name=$1; shift; timeout 600 python $PKG/tools/evaluate.py "$@" --out $OUT/$name.json > $OUT/$name.log 2>&1; echo "$name rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt; }
run bench-fast bench --model-dir $MD --max-len 2048 --prefill-sizes 128,512,1024
run bench-nofast bench --model-dir $MD --max-len 2048 --prefill-sizes 512 --no-fast-prefill
run bench-untraced bench --model-dir $MD --max-len 2048 --prefill-sizes "" --no-trace
run agree-prose agree --model-dir $MD --max-len 2048 --ref $REF --long $R/long2048.npz
run agree-code agree --model-dir $MD --max-len 2048 --ref $REF --long $R/code2048.npz --skip-prompts
run agree-multi agree --model-dir $MD --max-len 2048 --ref $REF --long $R/multi.npz --skip-prompts
for f in bench-fast bench-nofast bench-untraced; do python3 -c "
import json;d=json.load(open('$OUT/$f.json'));print('$f',round(d['decode']['mean'],2),round(d['first_token_s_12_token_prompt']['median'],3),[(p['n'],round(p['loop_read_last']['median_tok_s'],1),p.get('prefill_vs_stepwise_last_logits')) for p in d.get('prefill',[])])" 2>&1 | tail -1; done
for f in agree-prose agree-code agree-multi; do python3 -c "
import json;d=json.load(open('$OUT/$f.json'));l=d['long'];print('$f',l['tokens'],{k:round(v,4) for k,v in l['chip_vs_fp32'].items()},'floor',{k:round(v,4) for k,v in l['bf16_cpu_vs_fp32'].items()},l.get('first_1024_positions'))" 2>&1 | tail -1; done
echo "end $(date -Is)" | tee -a $OUT/conditions.txt
