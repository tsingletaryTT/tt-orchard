#!/bin/bash
# One session on the pair (venv-light + host tt-metal), one process at a time, each under timeout.
# Usage: TT_VISIBLE_DEVICES=<ids> bash session.sh
# ORCH_RUN = run directory of the Granite port; PAIR = the venv-light directory (see the run log).
ORCH_RUN=$HOME/orchard-runs/ibm-granite--granite-4.0-h-1b
PAIR=$HOME/orchard-runs/ttlang-pair
OUT=$ORCH_RUN/port/pair-session
mkdir -p $OUT
source $PAIR/venv-light/bin/activate
eval "$(tt-lang-setup-external-tt-metal --tt-metal-dir $HOME/tt-metal)"
export PYTHONPATH=$PYTHONPATH:<safetensors-only-dir>      # safetensors only (not in venv-light); appended so it cannot shadow torch
cd $ORCH_RUN/port
echo "session start $(date -Is) load: $(cat /proc/loadavg)" | tee $OUT/conditions.txt
python - <<'PY' 2>&1 | tee -a $OUT/conditions.txt
import ttl; print("build_info", ttl.build_info())
PY
# order: composed, state-only, state+readout kernel, composed again (brackets drift)
run_bench() { # name mode [module:build]
  local k=(); [ -n "$3" ] && k=(--kernels "$3")
  timeout 900 python bench_decode.py $2 "${k[@]}" --out $OUT/bench-$1.json 2>&1 | grep -E '^\{|^Traceback|^[A-Za-z_.]*Error' | tail -15
  echo "bench $1 rc=${PIPESTATUS[0]} $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
}
run_agree() { # name mode [module:build]
  local k=(); [ -n "$3" ] && k=(--kernels "$3")
  timeout 1500 python run_port.py $2 "${k[@]}" --prompts 4 --out $OUT/agree-$1 2>&1 | grep -E '^\{|^Traceback|^[A-Za-z_.]*Error' | tail -30 | tee $OUT/agree-$1.txt
  echo "agree $1 rc=${PIPESTATUS[0]} $(date -Is) load $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
}
case "$1" in
 pilot) run_agree full kernel kernel_mod:build ;;
 *)
  run_bench composed-a composed
  run_bench state-composed kernel kernel_mod_state_only:build
  run_bench state-readout kernel kernel_mod:build
  run_bench composed-b composed
  run_agree composed composed
  run_agree state-composed kernel kernel_mod_state_only:build
  run_agree state-readout kernel kernel_mod:build ;;
esac
echo "session end $(date -Is) load: $(cat /proc/loadavg)" | tee -a $OUT/conditions.txt
