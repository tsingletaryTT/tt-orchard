#!/bin/bash
# Final lease: boot the final staged package from its own directory in the fresh venv (weights already in the scratch HF cache), then
# an untraced decode run and the kernel-mode and bf16-state agreement runs (for the release notes). Keep under 14 minutes.
OUT=$1; PKGSRC=$2; mkdir -p $OUT
R=<pkg-run-dir>
export HF_HOME=$R/fresh/hf-home TT_METAL_HOME=$HOME/tt-metal PORT=20110
VENV=$R/fresh/venv
PKG=$R/fresh/pkgfinal; rm -rf $PKG; cp -r $PKGSRC $PKG
echo "start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
export GRANITE_VENV=$VENV READY_TRIES=120 READY_SLEEP=2
( source $VENV/bin/activate; cd $PKG; bash run.sh --self-test ) > $OUT/selftest-before.txt 2>&1; echo "selftest-before rc=$?" | tee -a $OUT/conditions.txt; cat $OUT/selftest-before.txt
bash $R/../granite-package/boot.sh $PKG $OUT/boot 2>&1 | tail -16
( source $VENV/bin/activate; cd $PKG; bash run.sh --self-test ) > $OUT/selftest-after.txt 2>&1; echo "selftest-after rc=$?" | tee -a $OUT/conditions.txt; cat $OUT/selftest-after.txt
MD=$(ls -d $HF_HOME/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/*)
unset HF_HOME
source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $TT_METAL_HOME)"
export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>
REF=$HOME/orchard-runs/ibm-granite--granite-4.0-h-1b/reference; LONG=$R/ref/long2048.npz
cd /tmp
run() { name=$1; shift; timeout 500 python $PKG/tools/evaluate.py "$@" --out $OUT/$name.json > $OUT/$name.log 2>&1; echo "$name rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt; }
run bench-untraced bench --model-dir $MD --max-len 2048 --prefill-sizes "" --no-trace
run agree-kernel agree --model-dir $MD --max-len 2048 --ref $REF --long $LONG --ssd state_full_ro --ssd-precision bf16
run agree-composed-bf16 agree --model-dir $MD --max-len 2048 --ref $REF --long $LONG --ssd-precision bf16 --skip-prompts
for f in agree-kernel agree-composed-bf16; do python3 -c "
import json;d=json.load(open('$OUT/$f.json'));l=d['long'];print('$f',d['ssd'],d['ssd_precision'],{k:round(v,4) for k,v in l['chip_vs_fp32'].items()},l.get('first_1024_positions',{}).get('chip_vs_fp32'),[round(p['rel_l2_mean'],4) for p in d.get('prompts',[])],[round(p['top1'],3) for p in d.get('prompts',[])])" 2>&1 | tail -1; done
python3 -c "import json;d=json.load(open('$OUT/bench-untraced.json'));print('untraced',round(d['decode']['mean'],2))" 2>&1 | tail -1
echo "end $(date -Is)" | tee -a $OUT/conditions.txt
