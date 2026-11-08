#!/bin/bash
# Boot check of the staged package through tt-model's own serve path, under the lease the caller holds (lease_run.sh), then the benchmarks.
# Exit 0 only when the server came up healthy and both benchmarks succeeded; logs/boot-status.txt says OK or FAILED: <why>. Result files from an
# earlier run are removed first, so a failed run never leaves stale numbers. The container this invocation started is always stopped and removed.
# env: CAN_CAD_WORK (work directory with staged/, logs/ and the scripts), HF_HOME (holds the weights), TT_MODEL_PYTHON (the tt-model interpreter),
#      BENCH_PYTHON (default python3), BOOT_TIMEOUT_S (default 2400), BOOT_POLL_S (default 10). The package cache directory under
#      ~/.cache/tt-model is expected to be empty for a cold boot and may be a link to a bigger disk: it is not removed here.
set -u
cd "${CAN_CAD_WORK:?set CAN_CAD_WORK}" || exit 2
: "${HF_HOME:?set HF_HOME}"
export HF_HOME
NAME=tt-model-berkelium-qwen3-32b-cadquery-p300-p300
PORT=20161
mkdir -p logs
rm -f logs/boot-ready.txt logs/boot-start.txt logs/boot-serve.log logs/boot-bench.log logs/boot-container.log logs/boot-status.txt \
      bench-package-p300.json prefill-package-p300.json
started=0
cleanup() {
  if [ "$started" = 1 ]; then
    docker logs "$NAME" > logs/boot-container.log 2>&1
    docker stop "$NAME" > /dev/null 2>&1
    docker rm -f "$NAME" > /dev/null 2>&1
  fi
}
trap cleanup EXIT
fail() { echo "FAILED: $1" | tee logs/boot-status.txt >&2; exit "${2:-1}"; }
date -u +%FT%TZ > logs/boot-start.txt
started=1
"${TT_MODEL_PYTHON:-python3}" serve_staged.py --profile p300 --port "$PORT" --detach > logs/boot-serve.log 2>&1 || fail "serve_staged.py exited non-zero (see logs/boot-serve.log)"
t0=$(date +%s)
until curl --fail --silent --show-error -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
  sleep "${BOOT_POLL_S:-10}"
  docker ps --format '{{.Names}}' | grep -q "$NAME" || fail "container gone before the server was healthy"
  [ $(( $(date +%s) - t0 )) -gt "${BOOT_TIMEOUT_S:-2400}" ] && fail "health check timed out"
done
echo "ready_after_s=$(( $(date +%s) - t0 ))" > logs/boot-ready.txt
"${BENCH_PYTHON:-python3}" bench.py --port "$PORT" --out bench-package-p300.json --repeats 3 --max-tokens 400 --prefill 2048,7000 \
  --cache-state "tensor cache and kernel cache warm (built by an earlier boot of this package through tt-model serve); first boot with empty caches took 532 s" \
  > logs/boot-bench.log 2>&1 || fail "bench.py failed (see logs/boot-bench.log)"
"${BENCH_PYTHON:-python3}" bench_prefill.py "$PORT" prefill-package-p300.json >> logs/boot-bench.log 2>&1 || fail "bench_prefill.py failed (see logs/boot-bench.log)"
echo OK > logs/boot-status.txt
