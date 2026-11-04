"""Summarize one finished run of run_bench.sh.

usage: collect.py RUN_DIR WIDTH HEIGHT FPS MIN_DUR MAX_DUR [IMAGE]

Reads the run's logs (Qwen Code's stream-json output, the server metric samples) and checks out/final.mp4 with ffprobe, ffmpeg and the
skill's own check_video.sh, both inside a read-only container that has no network. Writes RUN_DIR/logs/summary.json and prints it.
Every figure comes from a file in RUN_DIR/logs or RUN_DIR/task; the summary names which."""
import json, os, re, subprocess, sys, datetime

run, W, H, FPS, DMIN, DMAX = sys.argv[1:7]
image = sys.argv[7] if len(sys.argv) > 7 else "cvbench:0.1"
task, logs = os.path.join(run, "task"), os.path.join(run, "logs")
out = {"run": os.path.basename(run)}


def iso(path):
    return datetime.datetime.strptime(open(path).read().strip(), "%Y-%m-%dT%H:%M:%SZ")


def rd(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


# ---- wall time and exit ------------------------------------------------------------------------------------------------------
if rd(f"{logs}/started_at") and rd(f"{logs}/ended_at"):
    out["wall_s"] = int((iso(f"{logs}/ended_at") - iso(f"{logs}/started_at")).total_seconds())
out["exit_code"] = rd(f"{logs}/exit_code")

# ---- the agent's stream ------------------------------------------------------------------------------------------------------
st = {"assistant_messages": 0, "tool_calls": {}, "tool_errors": 0, "tool_results": 0, "compaction_events": 0, "thinking_chars": 0,
      "text_chars": 0, "usage_sum": {"input_tokens": 0, "output_tokens": 0}, "result": None, "web_fetch_calls": 0, "image_tool_calls": {}}
errors_sample = []
last_in = 0
try:
    lines = open(f"{logs}/stream.jsonl", errors="replace").read().splitlines()
except OSError:
    lines = []
for line in lines:
    try:
        d = json.loads(line)
    except ValueError:
        continue
    t = d.get("type")
    kinds = " ".join(str(x).lower() for x in (t, d.get("subtype"), (d.get("event") or {}).get("type")))
    if t != "system" and re.search(r"compact|compress", kinds):       # an event of that kind; the init event only lists slash commands
        st["compaction_events"] += 1
    if t == "assistant":
        m = d.get("message", {})
        st["assistant_messages"] += 1
        u = m.get("usage") or {}
        n_in = u.get("input_tokens", 0) or 0
        if n_in and last_in and n_in < 0.7 * last_in:                 # the prompt shrank by 30% or more between two calls
            st["context_drops"] = st.get("context_drops", 0) + 1
        if n_in:
            last_in = n_in
            st["max_prompt_tokens"] = max(st.get("max_prompt_tokens", 0), n_in)
        st["usage_sum"]["input_tokens"] += n_in
        st["usage_sum"]["output_tokens"] += u.get("output_tokens", 0) or 0
        for b in m.get("content", []):
            if b.get("type") == "tool_use":
                n = b.get("name"); st["tool_calls"][n] = st["tool_calls"].get(n, 0) + 1
                if n == "web_fetch":
                    st["web_fetch_calls"] += 1
                if n in ("zoom_image", "read_file"):
                    inp = json.dumps(b.get("input"))
                    if re.search(r"\.(png|jpg|jpeg|webp)", inp):
                        st["image_tool_calls"][n] = st["image_tool_calls"].get(n, 0) + 1
            elif b.get("type") == "thinking":
                st["thinking_chars"] += len(b.get("thinking", ""))
            elif b.get("type") == "text":
                st["text_chars"] += len(b.get("text", ""))
    elif t == "user":
        for b in (d.get("message", {}).get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                st["tool_results"] += 1
                if b.get("is_error"):
                    st["tool_errors"] += 1
                    if len(errors_sample) < 8:
                        errors_sample.append(str(b.get("content"))[:200])
    elif t == "result":
        st["result"] = {k: d.get(k) for k in ("subtype", "is_error", "duration_ms", "num_turns", "usage", "result")}
        if isinstance(st["result"].get("result"), str):
            st["result"]["result"] = st["result"]["result"][:400]
st["tool_error_samples"] = errors_sample
out["stream"] = st

# ---- server load (metric samples every 15 s) ------------------------------------------------------------------------------------
def js(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None

s0, s1 = js(f"{logs}/metrics-start.json"), js(f"{logs}/metrics-end.json")
samples = []
try:
    samples = [json.loads(x) for x in open(f"{logs}/metrics.jsonl") if x.strip()]
except OSError:
    pass
run_vals = [s.get("vllm:num_requests_running", 0) for s in samples if "error" not in s]
srv = {"samples": len(samples), "max_requests_running": max(run_vals) if run_vals else None,
       "samples_with_more_than_one_running": sum(1 for v in run_vals if v > 1),
       "max_requests_waiting": max([s.get("vllm:num_requests_waiting", 0) for s in samples if "error" not in s] or [0])}
if s0 and s1:
    for k in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total"):
        if k in s0 and k in s1:
            srv[k.split(":")[1] + "_delta"] = int(s1[k] - s0[k])
out["server"] = srv

# ---- the deliverables ----------------------------------------------------------------------------------------------------------
def in_container(cmd):
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                        "--user", f"{os.getuid()}:{os.getgid()}", "--memory", "4g", "-v", f"{task}:{task}:ro", "-w", task, image, "sh", "-c", cmd],
                       capture_output=True, text=True, timeout=900)
    return (r.stdout + r.stderr).strip()

final = os.path.join(task, "out", "final.mp4")
delivered = os.path.exists(final)
if not delivered:                                    # the agent never delivered out/final.mp4: measure the newest mp4 in out/ and say so
    cand = sorted((os.path.join(task, "out", f) for f in os.listdir(os.path.join(task, "out")) if f.endswith(".mp4")),
                  key=os.path.getmtime) if os.path.isdir(os.path.join(task, "out")) else []
    if cand:
        final = cand[-1]
out["final_mp4_delivered"] = delivered
files = {}
for name in ("SPEC.md", "VERIFY.md", "STRUCTURE.md", "out/final.mp4"):
    p = os.path.join(task, name)
    files[name] = os.path.getsize(p) if os.path.exists(p) else None
out["files"] = files
rel = os.path.relpath(final, task)
out["measured_file"] = rel if os.path.exists(final) else None
if os.path.exists(final):
    probe = in_container(f"ffprobe -v error -count_frames -show_streams -show_format -of json {rel}")
    try:
        pj = json.loads(probe)
        v = [s for s in pj["streams"] if s["codec_type"] == "video"][0]
        out["video"] = {"codec": v["codec_name"], "size": f"{v['width']}x{v['height']}", "pix_fmt": v.get("pix_fmt"), "avg_frame_rate": v["avg_frame_rate"],
                        "frames": int(v.get("nb_read_frames") or 0), "duration_s": float(pj["format"]["duration"]),
                        "audio_streams": sum(1 for s in pj["streams"] if s["codec_type"] == "audio"), "bytes": int(pj["format"]["size"])}
    except Exception as exc:
        out["video"] = {"probe_error": str(exc)[:160], "raw": probe[:300]}
    out["check_video"] = in_container(f"sh .qwen/skills/code-video/scripts/check_video.sh {rel} {W} {H} {FPS} {DMIN} {DMAX} 0")
    md5 = in_container(f"ffmpeg -v error -i {rel} -f framemd5 -an - 2>/dev/null | grep -v '^#'")
    hashes = [ln.split(",")[-1].strip() for ln in md5.splitlines() if ln.strip()]
    longest = cur = 0
    for i in range(1, len(hashes)):
        cur = cur + 1 if hashes[i] == hashes[i - 1] else 0
        longest = max(longest, cur)
    out["frame_hashes"] = {"frames": len(hashes), "distinct": len(set(hashes)),
                           "identical_to_previous": sum(1 for i in range(1, len(hashes)) if hashes[i] == hashes[i - 1]),
                           "longest_identical_run_frames": longest + 1 if hashes else 0}
json.dump(out, open(f"{logs}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
