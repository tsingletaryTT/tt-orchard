"""Sample a vLLM-style /metrics endpoint. usage: poll_metrics.py ENDPOINT OUT_FILE (INTERVAL_S | once)
With an interval, append one JSON line per sample to OUT_FILE until killed; with `once`, write one JSON object to OUT_FILE.
Keeps the counters and gauges that show how busy the server was and who else used it."""
import json, re, sys, time, urllib.request

KEEP = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
        "vllm:request_success_total", "vllm:num_preemptions_total")


def sample(endpoint):
    out = {"t": time.time()}
    try:
        text = urllib.request.urlopen(endpoint.rstrip("/") + "/metrics", timeout=10).read().decode()
    except Exception as exc:                      # a failed poll is recorded and polling goes on
        return {**out, "error": str(exc)[:120]}
    for line in text.splitlines():
        m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m and m.group(1) in KEEP:
            key = m.group(1) + (re.sub(r'(engine|model_name)="[^"]*",?', "", m.group(2) or "")).replace("{}", "")
            out[key] = float(m.group(3))
    return out


endpoint, path, mode = sys.argv[1:4]
if mode == "once":
    json.dump(sample(endpoint), open(path, "w"))
else:
    with open(path, "a") as f:
        while True:
            f.write(json.dumps(sample(endpoint)) + "\n"); f.flush()
            time.sleep(float(mode))
