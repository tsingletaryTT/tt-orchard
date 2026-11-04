"""Sample a vLLM-style /metrics endpoint. usage: poll_metrics.py ENDPOINT OUT_FILE (INTERVAL_S | once)
With an interval, append one JSON line per sample to OUT_FILE until killed; with `once`, write one JSON object to OUT_FILE.
Keeps the counters and gauges that show how busy the server was and who else used it. The engine and model_name labels are dropped, so
series that differ only in those are added together (two engines with one running request each are two running requests); every other
label stays in the key, in sorted order."""
import json, re, sys, time, urllib.request

DROP = ("engine", "model_name")
KEEP = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
        "vllm:request_success_total", "vllm:num_preemptions_total")


def sample(endpoint):
    out = {"t": time.time()}
    try:
        text = urllib.request.urlopen(endpoint.rstrip("/") + "/metrics", timeout=10).read(8 << 20).decode(errors="replace")   # a body over 8 MiB is cut
    except Exception as exc:                      # a failed poll is recorded and polling goes on
        return {**out, "error": str(exc)[:120]}
    for line in text.splitlines():
        m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m and m.group(1) in KEEP:
            try:
                value = float(m.group(3))
            except ValueError:
                continue
            labels = sorted((k, v) for k, v in re.findall(r'(\w+)="([^"]*)"', m.group(2) or "") if k not in DROP)
            key = m.group(1) + ("{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else "")
            out[key] = out.get(key, 0.0) + value
    return out


endpoint, path, mode = sys.argv[1:4]
if mode == "once":
    json.dump(sample(endpoint), open(path, "w"))
else:
    with open(path, "a") as f:
        while True:
            f.write(json.dumps(sample(endpoint)) + "\n"); f.flush()
            time.sleep(float(mode))
