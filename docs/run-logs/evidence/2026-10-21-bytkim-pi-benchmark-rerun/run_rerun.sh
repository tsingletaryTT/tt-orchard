#!/bin/bash
# Re-run of the items whose reply was cut off at 8192 tokens, now with a 32768 token limit. Same bundle and thinking setting as the full run.
# Frozen code: this worktree is not edited while the run is in progress.
cd /home/ttuser/code/tt-orchard/.claude/worktrees/bench-rerun || exit 9
B=/home/ttuser/orchard-runs/bytkim--qwen3.8-27b-pi/benchmark-rerun
PI_ID="bytkim/Qwen3.8-27B-pi@34d3f21045f7b41453e66c39e137cfc818a603ce"
BASE_ID="Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PI_DIR=/home/ttuser/orchard-runs/bytkim--qwen3.8-27b-pi/stages/4/configs/2/model-dir
BASE_DIR=/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
session() {  # label
  local label=$1 out=$B/$1-32k id run dir cache
  if [ "$label" = pi ]; then id=$PI_ID; run=/home/ttuser/orchard-runs/bytkim--qwen3.8-27b-pi/stages/4/configs/2/run.sh; dir=$PI_DIR
    cache=/home/ttuser/orchard-runs/cache/bytkim--qwen3.8-27b-pi/2chip-episod--qwen3.8-27b-dflash2-p300/tt_cache
  else id=$BASE_ID; run=/home/ttuser/orchard-runs/bytkim--qwen3.8-27b-pi/benchmark-pilot/base-run.sh; dir=$BASE_DIR; cache=/mnt/bonus/models/orchard-runs/bytkim-package/base-tt-cache; fi
  echo "SESSION start $label $(date -u +%H:%M:%SZ)"
  gozer run --chips 2 --exact 2 --who claude:bench-rerun --reason "re-run of cut-off items at 32768 tokens, $label" -- \
    timeout 36000 /home/ttuser/.tenstorrent-venv/bin/python3 -m orchard.bench_session --label $label --model-id "$id" --run-sh $run --model-dir $dir --tt-cache $cache \
      --hf-home /home/ttuser/.cache/huggingface --port 8100 --out-dir $out --phases humaneval,mbpp,ifeval --items $B/items-$label.json \
      --concurrency 4 --thinking default --max-gen-toks 32768 --quiet-s 2400 --phase-backstop-s 21600 > $B/$label-32k.log 2>&1
  local rc=$?
  echo "SESSION end $label exit=$rc $(date -u +%H:%M:%SZ)"
  /home/ttuser/.tenstorrent-venv/bin/python3 - "$out/summary.json" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception as e:
    print("  no summary:", e); sys.exit(0)
print(f"  served={s.get('served_name')} ready={s.get('server_ready_s')}s wall={s.get('wall_s')}s")
for p in s["phases"]:
    pr = p.get("proxy") or {}
    print(f"  {p['name']}: ok={p['ok']} problem={p['problem']} wall={p.get('wall_s')}s requests={pr.get('requests')} errors={pr.get('errors')} cut_off={pr.get('cut_off')}")
PYEOF
  return $rc
}
for label in pi base; do
  session $label; rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 1 ]; then echo "STOP: session $label exited $rc (not a phase failure)"; exit $rc; fi
done
echo "RERUN done $(date -u +%H:%M:%SZ)"
