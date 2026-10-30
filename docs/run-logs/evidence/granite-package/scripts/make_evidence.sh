#!/bin/bash
# Collect the evidence files of the package into $1 (names as the card cites them), replacing home paths with placeholders.
E=$1; rm -rf $E; mkdir -p $E; R=$HOME/orchard-runs/granite-package
for L in 512 1024 2048 4096; do cp $R/f2/bench-L$L.json $E/bench-L$L.json; done
cp $R/f2/bench-L2048-untraced.json $E/; cp $R/f1/bench-fp32.json $E/bench-fp32-L2048-prefill.json; cp $R/bootA2/smoke.json $E/server-smoke.json
cp $R/s1/bench-fast.json $E/s2-bench-fast.json; cp $R/s1/bench-nofast.json $E/s2-bench-nofast.json; cp $R/s1/bench-untraced.json $E/s2-bench-untraced.json
cp $R/s1/agree-prose.json $E/agree-prose-2048.json; cp $R/s1/agree-code.json $E/agree-code-2048.json; cp $R/s1/agree-multi.json $E/agree-nonenglish-653.json
cp $R/extra/long1024-agree-fp32.json $R/extra/long1024-agree-bf16.json $E/
cp $HOME/orchard-runs/ibm-granite--granite-4.0-h-1b/reference/bf16_cpu_control.json $E/
cp $R/kb/kernel-bench-composed.json $E/kernel-bench-composed-bf16-L2048.json; cp $R/kb/kernel-bench-state_full_ro.json $E/kernel-bench-state_full_ro-bf16-L2048.json
cp $R/f1/conditions.txt $E/conditions-f1.txt; cp $R/f2/conditions.txt $E/conditions-f2.txt; cp $R/s1/conditions.txt $E/conditions-s1.txt
cp $R/final/agree-kernel.json $E/agree-kernel-state_full_ro-bf16-2048.json; cp $R/final/agree-composed-bf16.json $E/agree-composed-bf16-2048.json
cp $R/final/bench-untraced.json $E/final-bench-untraced-loaded-host.json; cp $R/final/boot/smoke.json $E/final-server-smoke.json; cp $R/final/conditions.txt $E/conditions-final.txt
if [ -d "$2" ]; then
  cp $2/bench-fast.json $E/s3-bench-fast.json; cp $2/selftest-before.txt $E/s3-selftest-before.txt; cp $2/selftest-after.txt $E/s3-selftest-after.txt; cp $2/conditions.txt $E/conditions-s3.txt
  [ -f $2/smoke.json ] && cp $2/smoke.json $E/s3-server-smoke.json
fi
if [ -d "$3" ]; then       # session 4 (0.3.0, chunked prefill): the chunked bench, the step-loop control in the same session, the conditions
  cp $3/bench-fast.json $E/s4-bench-fast.json; cp $3/bench-step.json $E/s4-bench-step.json; cp $3/conditions.txt $E/conditions-s4.txt
  [ -f $3/smoke.json ] && cp $3/smoke.json $E/s4-server-smoke.json
fi
sed -i "s#$HOME/orchard-runs/ibm-granite--granite-4.0-h-1b#<run-dir>#g; s#$HOME/orchard-runs/granite-package#<pkg-run-dir>#g; s#$HOME/.cache/huggingface#<hf-cache>#g; s#$HOME#<home>#g" $E/*
grep -l "/home/\|ttuser" $E/* && echo "HOME PATH LEFT" || echo "evidence clean"
