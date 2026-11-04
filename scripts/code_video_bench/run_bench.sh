#!/usr/bin/env bash
# Run one code-video benchmark: Qwen Code, in a locked-down container, drives the public `code-video` skill against a model endpoint.
#
# usage: run_bench.sh RUN_NAME BRIEF.md [WALL_MINUTES=120] [MAX_TURNS=1500]
# env (required): WORK_ROOT   directory that holds one folder per run (must not contain a folder named RUN_NAME)
#                 SKILL_SRC   skills/code-video of a checkout of the public SenseNova-6.8-Code2Video repository, unchanged
# env (optional): ENDPOINT    OpenAI-compatible server (default http://127.0.0.1:8012)
#                 IMAGE       docker image built from the Dockerfile next to this script (default cvbench:0.1)
#                 PROMPT      first message to the agent (default: the upstream usage text with the Qwen Code skill path)
#
# What the container gets: the task directory only (read-write, same path inside), a HOME inside it, no credentials, all capabilities
# dropped, no new privileges, a read-only root file system, memory and process limits. It shares the host network namespace only because
# the endpoint listens on loopback; the host's loopback services are therefore reachable from inside.
set -euo pipefail
NAME=${1:?run name}; BRIEF=${2:?brief file}; WALL_MIN=${3:-120}; MAX_TURNS=${4:-1500}
: "${WORK_ROOT:?}"; : "${SKILL_SRC:?}"
ENDPOINT=${ENDPOINT:-http://127.0.0.1:8012}; IMAGE=${IMAGE:-cvbench:0.1}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$WORK_ROOT/$NAME; TASK=$RUN/task; LOGS=$RUN/logs
[ ! -e "$RUN" ] || { echo "refusing to reuse $RUN"; exit 1; }
MODEL=$(curl -sf -m 20 "$ENDPOINT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
PROMPT=${PROMPT:-"Use the project code-video skill. Read .qwen/skills/code-video/SKILL.md and dense_prompt.md, then the relevant craft guides. This directory is the project root. Produce the complete video, run the rendering tools, and deliver out/final.mp4, SPEC.md and VERIFY.md."}

mkdir -p "$TASK/.qwen/skills" "$TASK/home/.qwen" "$TASK/out" "$TASK/assets" "$LOGS"
cp -r "$SKILL_SRC" "$TASK/.qwen/skills/code-video"
ln -s /opt/pw/node_modules "$TASK/node_modules"          # resolves inside the image: Playwright and @napi-rs/canvas
cp "$BRIEF" "$TASK/dense_prompt.md"
ENDPOINT="$ENDPOINT" MODEL="$MODEL" python3 - "$TASK/home/.qwen/settings.json" <<'PY'
import json, os, sys
m, base = os.environ["MODEL"], os.environ["ENDPOINT"].rstrip("/") + "/v1"
json.dump({
  "general": {"enableAutoUpdate": False},
  "privacy": {"usageStatisticsEnabled": False},
  "memory": {"enableManagedAutoMemory": False, "enableManagedAutoDream": False},   # background model calls that would compete for a slow endpoint
  "telemetry": {"enabled": False},
  "tools": {"approvalMode": "yolo"},
  "security": {"auth": {"selectedType": "openai"}},
  "model": {"name": m},
  "modelProviders": {"openai": [{
      "id": m, "name": "model under test", "baseUrl": base, "envKey": "OPENAI_API_KEY",
      "generationConfig": {"contextWindowSize": 262144, "timeout": 1800000, "maxRetries": 2,
                           "samplingParams": {"max_tokens": 32768}}}]},
}, open(sys.argv[1], "w"), indent=1)
PY
cp "$TASK/home/.qwen/settings.json" "$LOGS/qwen-settings.json"

python3 "$HERE/poll_metrics.py" "$ENDPOINT" "$LOGS/metrics.jsonl" 15 & POLL=$!
cleanup() { kill "$POLL" 2>/dev/null || true; docker rm -f "cvbench-$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
python3 "$HERE/poll_metrics.py" "$ENDPOINT" "$LOGS/metrics-start.json" once
date -u +%FT%TZ > "$LOGS/started_at"

set +e
systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 -- \
  timeout "$((WALL_MIN + 5))m" docker run --rm --init --name "cvbench-$NAME" \
    --user "$(id -u):$(id -g)" --network host --cap-drop ALL --security-opt no-new-privileges \
    --read-only --tmpfs /tmp:rw,exec,size=2g --shm-size 2g \
    --memory 16g --memory-swap 16g --cpus 8 --pids-limit 2048 \
    -v "$TASK:$TASK" -w "$TASK" -e "HOME=$TASK/home" -e OPENAI_API_KEY=local-endpoint-has-no-auth -e NODE_PATH=/opt/pw/node_modules \
    "$IMAGE" qwen --approval-mode yolo --output-format stream-json --max-session-turns "$MAX_TURNS" --max-wall-time "${WALL_MIN}m" \
         --openai-logging --openai-logging-dir "$TASK/home/openai-logs" -p "$PROMPT" \
    > "$LOGS/stream.jsonl" 2> "$LOGS/stderr.log"
echo $? > "$LOGS/exit_code"
set -e
date -u +%FT%TZ > "$LOGS/ended_at"
python3 "$HERE/poll_metrics.py" "$ENDPOINT" "$LOGS/metrics-end.json" once
echo "run finished, exit $(cat "$LOGS/exit_code"); task in $TASK, logs in $LOGS"
