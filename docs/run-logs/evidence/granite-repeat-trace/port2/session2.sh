#!/bin/bash
# usage (under a gozer lease, TT_VISIBLE_DEVICES set): bash session2.sh OUTDIR "NAME|args" ...   one process at a time, each under timeout
HERE=$(cd "$(dirname "$0")" && pwd)
PAIR=$HOME/orchard-runs/ttlang-pair
OUT=$1; shift; mkdir -p $OUT
source $PAIR/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
export PYTHONPATH=$PYTHONPATH:${ST_DIR:-/tmp/claude-1000/st}     # safetensors only
cd $HERE
echo "session start $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
for spec in "$@"; do
  name=${spec%%|*}; args=${spec#*|}
  timeout 900 python run2.py $name $args --out $OUT > $OUT/$name.log 2>&1
  echo "$name rc=$? $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
  grep -E '^\{|^Traceback|^[A-Za-z_.]*(Error|Exception)' $OUT/$name.log | tail -8 | cut -c1-300
done
echo "session end $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
