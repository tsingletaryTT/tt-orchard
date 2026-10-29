#!/bin/bash
# The full matrix, one session, one lease (run it as: gozer run --exact 2 --who ... -- bash matrix.sh OUTDIR). One process at a time.
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1
bash $HERE/session2.sh $OUT \
 "composed-un-1|--ssd composed" "composed-tr-1|--ssd composed --trace" \
 "sf-repeat-un|--ssd state_full --expand repeat" "sf-repeat-tr|--ssd state_full --expand repeat --trace" \
 "sf-mul-un|--ssd state_full --expand mul" "sf-mul-tr|--ssd state_full --expand mul --trace" \
 "sfro-repeat-un|--ssd state_full_ro --expand repeat" "sfro-repeat-tr|--ssd state_full_ro --expand repeat --trace" \
 "sfro-mul-un|--ssd state_full_ro --expand mul" "sfro-mul-tr|--ssd state_full_ro --expand mul --trace" \
 "bcast-un|--ssd bcast" "bcast-tr|--ssd bcast --trace" \
 "bcastro-un|--ssd bcast_ro" "bcastro-tr|--ssd bcast_ro --trace" \
 "composed-un-2|--ssd composed" "composed-tr-2|--ssd composed --trace"
# the earlier dynamic port (granite_tt.py, composed), same venv and session, for continuity with the 2026-10-28 and 10-29 logs
source $HOME/orchard-runs/ttlang-pair/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
export PYTHONPATH=$PYTHONPATH:${ST_DIR:-<safetensors-only-dir>}
cd $HERE/../port
timeout 300 python bench_decode.py composed --out $OUT/dynamic-composed.json > $OUT/dynamic-composed.log 2>&1
echo "dynamic-composed rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
