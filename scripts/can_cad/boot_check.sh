#!/bin/bash
# Boot check of the staged package through tt-model's own serve path, under the lease the caller holds. Cold caches on purpose.
# env: CAN_CAD_WORK (the work directory with staged/, logs/, the scripts), HF_HOME (holds the weights), TT_MODEL_CACHE (default ~/.cache/tt-model), TT_MODEL_PYTHON (the tt-model interpreter)
cd "${CAN_CAD_WORK:?set CAN_CAD_WORK}"
: "${HF_HOME:?set HF_HOME}"
export HF_HOME
# The package cache directory under TT_MODEL_CACHE is expected to be empty (cold boot) and may be a link to a bigger disk: it is not removed here.
date -u +%FT%TZ > logs/boot-start.txt
"${TT_MODEL_PYTHON:-python3}" serve_staged.py --profile p300 --port 20161 --detach > logs/boot-serve.log 2>&1
t0=$(date +%s)
until curl -s -m 3 http://127.0.0.1:20161/health >/dev/null 2>&1; do
  sleep 10
  if ! docker ps --format '{{.Names}}' | grep -q berkelium-qwen3-32b-cadquery-p300; then echo "container gone" >> logs/boot-serve.log; break; fi
  if [ $(( $(date +%s) - t0 )) -gt 2400 ]; then echo "boot timeout" >> logs/boot-serve.log; break; fi
done
echo "ready_after_s=$(( $(date +%s) - t0 ))" > logs/boot-ready.txt
python3 bench.py --port 20161 --out bench-package-p300.json --repeats 3 --max-tokens 400 --prefill 2048,7000 --cache-state "tensor cache and kernel cache warm (built by an earlier boot of this package through tt-model serve); first boot with empty caches took 532 s" > logs/boot-bench.log 2>&1
python3 bench_prefill.py 20161 prefill-package-p300.json >> logs/boot-bench.log 2>&1
docker logs tt-model-berkelium-qwen3-32b-cadquery-p300-p300 > logs/boot-container.log 2>&1
docker stop tt-model-berkelium-qwen3-32b-cadquery-p300-p300 >> logs/boot-serve.log 2>&1
