#!/bin/bash
# One boot per model: session.sh LABEL(trim|base) OUTNAME SCALE(pilot|full) EXACT PORT [extra session_all.py args...]
# Code is frozen in this worktree for the whole job.
W=<CHECKOUT>
cd $W || exit 9
R=<RUN_DIR>
B=$R/benchmark
label=$1; outname=$2; scale=$3; exact=$4; port=$5; shift 5
PY=<HOME>/.tenstorrent-venv/bin/python3
TRIM_ID="ProprietaryLegal/Qwen3.8-27B-CJK-Trim@e5476cfa7df694f10fa9d00494f08e51010327b5"
BASE_ID="Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
if [ "$label" = trim ]; then id=$TRIM_ID; run=$R/stages/2/run.sh; dir=$R/stages/2/model-dir; ctl=()
else id=$BASE_ID; run=$B/base-run.sh; dir=<HOME>/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; ctl=(--control-record $B/base-run.sh.control.json); fi
cache=<HOME>/orchard-runs/cache/proprietarylegal--qwen3.8-27b-cjk-trim/bench-$label/tt_cache
mkdir -p "$cache"
export ORCHARD_CHECKOUT=$W
echo "SESSION start $label $outname $scale $(date -u +%H:%M:%SZ)"
gozer run --chips 2 --exact $exact --who claude:bench-cjk-trim --reason "benchmark $label ($outname, $scale), CJK-Trim card" -- \
  timeout 18000 $PY $W/docs/run-logs/evidence/2026-10-27-cjk-trim-benchmark/session_all.py --label $label --model-id "$id" --run-sh $run --model-dir $dir "${ctl[@]}" \
    --tt-cache $cache --hf-home <HOME>/.cache/huggingface --port $port --out-dir $B/$outname --bfcl-dir $B/bfcl-data --scale $scale "$@" > $B/$outname.log 2>&1
rc=$?
echo "SESSION end $label $outname exit=$rc $(date -u +%H:%M:%SZ)"
exit $rc
