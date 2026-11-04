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
# The run directory is reserved with one atomic `mkdir` before anything slow starts, so two invocations with the same WORK_ROOT and run
# name cannot both proceed: the second is refused and touches nothing of the first. The container is named cvbench-RUN_NAME-HASH, where HASH
# comes from the reserved directory's path, so two work roots that use the same run name do not collide on the container name either.
# Every python call here uses `python3 -I -S` (no current directory, PYTHON* variable, user site or site-packages on the import path), and the
# script first unsets every PYTHON* variable, so nothing a model could write onto a path reaches a host interpreter. A container removal in
# the exit trap is bounded by CVBENCH_CLEANUP_TIMEOUT_S (default 60); one that fails or times out is written to logs/cleanup.log with the id.
#
# The run name may use letters, digits, dots, underscores and hyphens. WORK_ROOT may be relative (it is made absolute); neither it nor
# the resolved path may contain a colon or a line break, because they become a docker mount source and destination.
#
# What the container gets: the task directory only (read-write, same path inside), a HOME inside it, no credentials, all capabilities
# dropped, no new privileges, a read-only root file system, memory and process limits. It shares the host network namespace only because
# the endpoint listens on loopback; the host's loopback services are therefore reachable from inside.
set -euo pipefail
for v in $(env | sed -n 's/^\(PYTHON[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
NAME=${1:?run name}; BRIEF=${2:?brief file}; WALL_MIN=${3:-120}; MAX_TURNS=${4:-1500}
: "${WORK_ROOT:?}"; : "${SKILL_SRC:?}"
case $NAME in *[!A-Za-z0-9._-]*|"") echo "run name '$NAME' may use letters, digits, '.', '_' and '-' only"; exit 1;; esac
nl=$'\n'
bad_path() { case $1 in *:*|*"$nl"*) return 0;; esac; return 1; }
bad_path "$WORK_ROOT" && { echo "WORK_ROOT may not contain a colon or a line break (it becomes a docker mount): $WORK_ROOT"; exit 1; }
mkdir -p "$WORK_ROOT"
WORK_ROOT=$(cd "$WORK_ROOT" && pwd -P)          # docker needs absolute paths for the mount destination and the working directory
SKILL_SRC=$(cd "$SKILL_SRC" && pwd -P)
bad_path "$WORK_ROOT" && { echo "WORK_ROOT resolves to a path with a colon or a line break: $WORK_ROOT"; exit 1; }
ENDPOINT=${ENDPOINT:-http://127.0.0.1:8012}; IMAGE=${IMAGE:-cvbench:0.1}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$WORK_ROOT/$NAME; TASK=$RUN/task; LOGS=$RUN/logs; CIDFILE=$LOGS/container.id
# Reserve the run directory. `mkdir` without -p fails if it exists, so exactly one of two racing invocations gets it. Until the run has
# written anything, a failed start gives the name back (rmdir removes only an empty directory, which is only ever this invocation's own).
mkdir "$RUN" 2>/dev/null || { echo "refusing to reuse $RUN: that run name is already taken under $WORK_ROOT"; exit 1; }
trap 'rmdir "$RUN" 2>/dev/null || true' EXIT
RUN_ID=$(printf '%s' "$RUN" | sha256sum | cut -c1-8); CNAME="cvbench-$NAME-$RUN_ID"
MODEL=$(curl -sf -m 20 "$ENDPOINT/v1/models" | python3 -I -S -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
PROMPT=${PROMPT:-"Use the project code-video skill. Read .qwen/skills/code-video/SKILL.md and dense_prompt.md, then the relevant craft guides. This directory is the project root. Produce the complete video, run the rendering tools, and deliver out/final.mp4, SPEC.md and VERIFY.md."}

mkdir -p "$TASK/.qwen/skills" "$TASK/home/.qwen" "$TASK/out" "$TASK/assets" "$LOGS"
cp -r "$SKILL_SRC" "$TASK/.qwen/skills/code-video"
# The trusted copy: collect.py checks the film with the checker from this directory, mounted read-only, and not with the task's copy, which
# the model can edit. The hashes let the summary list any skill file the run changed.
printf '%s\n' "$SKILL_SRC" > "$LOGS/skill_src"
( cd "$SKILL_SRC" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) > "$LOGS/skill.sha256"
ln -s /opt/pw/node_modules "$TASK/node_modules"          # resolves inside the image: Playwright and @napi-rs/canvas
cp "$BRIEF" "$TASK/dense_prompt.md"
ENDPOINT="$ENDPOINT" MODEL="$MODEL" python3 -I -S - "$TASK/home/.qwen/settings.json" <<'PY'
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

python3 -I -S "$HERE/poll_metrics.py" "$ENDPOINT" "$LOGS/metrics.jsonl" 15 & POLL=$!
# Remove only the container this invocation created: docker writes its id to $CIDFILE after the daemon made it. A start that failed (a
# name already used by another run, say) leaves the file empty, and then there is nothing of ours to remove.
cleanup() {
  kill "$POLL" 2>/dev/null || true
  cid=""; if [ -s "$CIDFILE" ]; then cid=$(cat "$CIDFILE"); fi
  if [ -n "$cid" ]; then
    limit=${CVBENCH_CLEANUP_TIMEOUT_S:-60}
    # Bounded: a daemon that never answers must not hang the exit trap. A removal that fails is written down with the id, so it can be done by hand.
    if timeout -k 5 "$limit" docker rm -f "$cid" >/dev/null 2>&1; then :; else
      rc=$?
      case $rc in 124|137) why="timed out after $limit s";; *) why="docker rm exited $rc";; esac
      printf 'cleanup of container %s failed: %s\n' "$cid" "$why" >> "$LOGS/cleanup.log" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT
python3 -I -S "$HERE/poll_metrics.py" "$ENDPOINT" "$LOGS/metrics-start.json" once
date -u +%FT%TZ > "$LOGS/started_at"

set +e
systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 -- \
  timeout -k 60 "$((WALL_MIN + 5))m" docker run --rm --init --name "$CNAME" --cidfile "$CIDFILE" \
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
python3 -I -S "$HERE/poll_metrics.py" "$ENDPOINT" "$LOGS/metrics-end.json" once
echo "run finished, exit $(cat "$LOGS/exit_code"); task in $TASK, logs in $LOGS"
echo "summarize it with: $HERE/collect.sh $RUN WIDTH HEIGHT FPS MIN_DUR MAX_DUR"
