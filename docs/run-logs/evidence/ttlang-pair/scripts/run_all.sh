#!/bin/bash
# usage: run_all.sh <mode: sim|chip> <outdir>; runs each kernel in its own process under a wall limit
MODE=$1; OUT=$2; mkdir -p $OUT
cd ~/orchard-runs/ttlang-pair
for n in $(python3 -c "import json;print(' '.join(e['name'] for e in json.load(open('k/manifest.json'))))"); do
  if [ $MODE = sim ]; then
    timeout 300 ~/orchard-venvs/ttlang/bin/tt-lang-sim scripts/run_one.py $n > $OUT/$n.log 2>&1
  elif [ $MODE = simlight ]; then
    timeout 300 venv-light/bin/tt-lang-sim scripts/run_one.py $n > $OUT/$n.log 2>&1
  else
    source venv-light/bin/activate
    eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir ~/tt-metal)"
    timeout 300 python scripts/run_one.py $n > $OUT/$n.log 2>&1
  fi
  echo "$n rc=$? $(grep -c RESULT $OUT/$n.log)"
done
