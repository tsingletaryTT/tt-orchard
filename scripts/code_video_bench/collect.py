"""Summarize one finished run of run_bench.sh.

usage: collect.py RUN_DIR WIDTH HEIGHT FPS MIN_DUR MAX_DUR [IMAGE]

Reads the run's logs (Qwen Code's stream-json output, the server metric samples) and checks out/final.mp4 with ffprobe, ffmpeg and the
skill's own check_video.sh, both inside a read-only container that has no network. Writes RUN_DIR/logs/summary.json and prints it.
Every figure comes from a file in RUN_DIR/logs or RUN_DIR/task; the summary names which.

The checker is run from a trusted copy of the skill, mounted read-only: the directory in the SKILL_SRC environment variable, or else the one
run_bench.sh recorded in RUN_DIR/logs/skill_src (the logs directory is never mounted into the model's container). The task's own copy of
the skill is not used for the check, because the model can edit it; the summary lists the skill files the run changed (hashes recorded by
run_bench.sh in RUN_DIR/logs/skill.sha256). A figure that is not measured is null, never zero.
Isolation: the collector re-runs itself with `python3 -I` when it was not started that way, so nothing in the current directory or in a
PYTHON* variable can be imported in its place. Inside the containers the checker runs with its working directory set to the trusted copy of
the skill (the unchanged files, mounted read-only), the video named by its absolute path, and PYTHONSAFEPATH, PYTHONNOUSERSITE and an empty
PYTHONPATH, so a file the model wrote into the task directory (a json.py, say) cannot be imported by the checker. Every other container call
runs from an empty read-only scratch directory.
Failures: a measurement that could not be taken (a container that timed out or that docker refused, ffmpeg that failed or wrote anything but
framemd5 records, a hash count that disagrees with ffprobe's frame count) is reported as null with the reason in an `error` field, and
`measurement_failures` lists and counts them. A failed measurement is never a figure.
Environment: CVBENCH_DOCKER_TIMEOUT_S (default 900) bounds each container; a container that overruns is removed by its id."""
import os, sys
if not sys.flags.isolated:
    os.execv(sys.executable, [sys.executable, "-I", *sys.argv])
import hashlib, json, re, shlex, stat, subprocess, tempfile, datetime

run, W, H, FPS, DMIN, DMAX = sys.argv[1:7]
run = os.path.abspath(run)                           # docker needs absolute mount destinations and working directories
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
st = {"assistant_messages": 0, "assistant_messages_with_usage": 0, "tool_calls": {}, "tool_errors": 0, "tool_results": 0, "compaction_events": 0, "thinking_chars": 0,
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
        if u:
            st["assistant_messages_with_usage"] = st.get("assistant_messages_with_usage", 0) + 1
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
def gauge(name):                                      # only samples that carry the gauge count; a sample without it is no measurement
    return [s[name] for s in samples if "error" not in s and isinstance(s.get(name), (int, float)) and not isinstance(s.get(name), bool)]

run_vals, wait_vals = gauge("vllm:num_requests_running"), gauge("vllm:num_requests_waiting")
srv = {"samples": len(samples), "samples_with_error": sum(1 for s in samples if "error" in s),
       "samples_with_running_gauge": len(run_vals), "samples_with_waiting_gauge": len(wait_vals),
       "max_requests_running": max(run_vals) if run_vals else None,
       "samples_with_more_than_one_running": sum(1 for v in run_vals if v > 1),
       "max_requests_waiting": max(wait_vals) if wait_vals else None}
if s0 and s1:
    for k in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total"):
        if k in s0 and k in s1:
            srv[k.split(":")[1] + "_delta"] = int(s1[k] - s0[k])
out["server"] = srv

# ---- the trusted copy of the skill and what the run changed in its own copy ------------------------------------------------------
def trusted_skill():
    """The directory of the unchanged skill: $SKILL_SRC, else the path run_bench.sh recorded in logs/skill_src. None when neither names a
    directory that holds scripts/check_video.sh."""
    for cand in (os.environ.get("SKILL_SRC"), rd(f"{logs}/skill_src")):
        if cand and os.path.isfile(os.path.join(cand, "scripts", "check_video.sh")):
            return os.path.realpath(cand)
    return None


def digest(path, cap=16 << 20):
    """sha256 of a regular file of at most `cap` bytes, opened without following a link and without blocking (the model could leave a
    FIFO or a link at the path); None for anything else."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        st_ = os.fstat(fd)
        if not stat.S_ISREG(st_.st_mode) or st_.st_size > cap:
            return None
        h = hashlib.sha256()
        with os.fdopen(os.dup(fd), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    finally:
        os.close(fd)


recorded = rd(f"{logs}/skill.sha256")
if recorded is None:
    out["skill_files_changed_by_run"] = None            # no record, so it is unknown and not "none"
else:
    copy_root = os.path.join(task, ".qwen", "skills", "code-video")
    known, changed = set(), []
    for ln in recorded.splitlines():
        h, _, name = ln.partition("  ")
        name = name[2:] if name.startswith("./") else name
        if not h or not name:
            continue
        known.add(name)
        if digest(os.path.join(copy_root, name)) != h:
            changed.append(name)
    out["skill_files_changed_by_run"] = sorted(changed)
    added = []
    for root_, _dirs, files_ in os.walk(copy_root):
        for f_ in files_:
            rel_ = os.path.relpath(os.path.join(root_, f_), copy_root)
            if rel_ not in known:
                added.append(rel_)
    out["skill_files_added_by_run"] = sorted(added)

# ---- the deliverables ----------------------------------------------------------------------------------------------------------
DOCKER_TIMEOUT_S = float(os.environ.get("CVBENCH_DOCKER_TIMEOUT_S", "900"))


SCRATCH = "/scratch"                                   # an empty read-only tmpfs: the working directory of every call that needs none
PY_SCRUB = ("PYTHONSAFEPATH=1", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=")
DOCKER_ERROR_CODES = (125, 126, 127)                    # docker's own failures: the daemon refused, the command could not be run or was not found
OUTPUT_CAP = int(os.environ.get("CVBENCH_OUTPUT_CAP_BYTES", 64 << 20))      # the most frame-record text one ffmpeg call may write


def in_container(cmd, extra_mounts=(), workdir=SCRATCH):
    """Run `sh -c cmd` in a read-only container with no network, the task directory mounted read-only, an empty scratch directory and a Python
    environment that imports nothing from the working directory. The container id is written to a --cidfile; when the call overruns, the docker
    client is killed by the timeout and the container is removed by that id, so the timeout also bounds the container's own life.
    Returns {"ok", "rc", "stdout", "stderr", "error", "text"}: `ok` is false when no command result exists (a timeout, a docker error, or docker
    missing) and `error` says why; `rc` is the command's exit status when it ran, and a nonzero one is the caller's to judge."""
    res = {"ok": False, "rc": None, "stdout": "", "stderr": "", "error": None, "text": ""}
    with tempfile.TemporaryDirectory(prefix="cvbench-collect-") as tmp:
        cidfile = os.path.join(tmp, "cid")
        argv = ["docker", "run", "--rm", "--cidfile", cidfile, "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}", "--memory", "4g",
                "--tmpfs", f"{SCRATCH}:ro,size=1m", *[a for kv in PY_SCRUB for a in ("-e", kv)],
                "-v", f"{task}:{task}:ro", *[a for m in extra_mounts for a in ("-v", m)], "-w", workdir, image, "sh", "-c", cmd]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=DOCKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            cid = rd(cidfile)
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, text=True, timeout=60)
            res["error"] = f"timed out after {DOCKER_TIMEOUT_S:g} s" + (f"; container {cid[:12]} removed" if cid else "; no container id was recorded")
            res["text"] = res["error"]
            return res
        except OSError as exc:
            res["error"] = f"could not run docker: {type(exc).__name__}"
            res["text"] = res["error"]
            return res
    res.update(rc=r.returncode, stdout=r.stdout, stderr=r.stderr, text=(r.stdout + r.stderr).strip())
    if r.returncode in DOCKER_ERROR_CODES and r.stderr.lstrip().startswith("docker"):
        res["error"] = f"docker error (exit {r.returncode}): " + " ".join(r.stderr.split())[:200]
    else:
        res["ok"] = True
    return res


failures = []                                           # the measurements that could not be taken, in the order they were tried

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
FRAMEMD5 = re.compile(r"^\d+,\s*-?\d+,\s*-?\d+,\s*\d+,\s*\d+,\s*([0-9a-fA-F]{32})$")      # stream, dts, pts, duration, size, md5


def frame_hash_figures(r, probe_frames):
    """The frame-hash measurement from one container result: the figures, or every figure null with the reason in `error`. Only framemd5
    records count as frames; any other line, any ffmpeg error output, a nonzero exit, a container that did not run, or a record count that
    disagrees with the frame count ffprobe reported makes the whole measurement a failure."""
    fh = {"frames": None, "distinct": None, "identical_to_previous": None, "longest_identical_run_frames": None, "invalid_lines": 0, "error": None}
    if not r["ok"]:
        fh["error"] = r["error"]
        return fh
    if r["rc"] != 0:
        fh["error"] = f"ffmpeg exited {r['rc']}: " + (" ".join(r["stderr"].split())[:200] or "no message")
        return fh
    if r["stderr"].strip():
        fh["error"] = "ffmpeg reported errors: " + " ".join(r["stderr"].split())[:200]
        return fh
    if len(r["stdout"]) > OUTPUT_CAP:
        fh["error"] = f"ffmpeg wrote more than {OUTPUT_CAP} bytes of frame records"
        return fh
    hashes, bad = [], []
    for ln in r["stdout"].splitlines():
        s_ = ln.strip()
        if not s_ or s_.startswith("#"):
            continue
        m = FRAMEMD5.match(s_)
        if m:
            hashes.append(m.group(1).lower())
        else:
            bad.append(s_)
    fh["invalid_lines"] = len(bad)
    if bad:
        fh["error"] = f"{len(bad)} invalid line(s) in the ffmpeg output, which are not framemd5 records (first: {bad[0][:80]!r})"
    elif not hashes:
        fh["error"] = "ffmpeg wrote no frame records"
    elif probe_frames is not None and probe_frames != len(hashes):
        fh["error"] = f"{len(hashes)} frame hashes against {probe_frames} frames counted by ffprobe"
    else:
        longest = cur = 0
        for i in range(1, len(hashes)):
            cur = cur + 1 if hashes[i] == hashes[i - 1] else 0
            longest = max(longest, cur)
        fh.update(frames=len(hashes), distinct=len(set(hashes)), longest_identical_run_frames=longest + 1,
                  identical_to_previous=sum(1 for i in range(1, len(hashes)) if hashes[i] == hashes[i - 1]))
    return fh


if os.path.exists(final):
    qa = shlex.quote(final)                           # absolute, and quoted once: a file name may hold a space or a shell metacharacter
    probe = in_container(f"ffprobe -v error -count_frames -show_streams -show_format -of json {qa}")
    if not probe["ok"]:
        out["video"] = {"probe_error": probe["error"], "raw": ""}
        failures.append("video")
    else:
        try:
            pj = json.loads(probe["stdout"])
            v = [s for s in pj["streams"] if s["codec_type"] == "video"][0]
            out["video"] = {"codec": v["codec_name"], "size": f"{v['width']}x{v['height']}", "pix_fmt": v.get("pix_fmt"), "avg_frame_rate": v["avg_frame_rate"],
                            "frames": int(v["nb_read_frames"]) if str(v.get("nb_read_frames", "")).isdigit() else None, "duration_s": float(pj["format"]["duration"]),
                            "audio_streams": sum(1 for s in pj["streams"] if s["codec_type"] == "audio"), "bytes": int(pj["format"]["size"])}
        except Exception as exc:
            out["video"] = {"probe_error": str(exc)[:160], "raw": probe["text"][:300]}
            failures.append("video")
    trusted = trusted_skill()
    if trusted:
        out["check_video_source"] = f"trusted copy of the skill, read-only mount ({os.path.basename(trusted)})"
        chk = in_container("sh /trusted-skill/scripts/check_video.sh " + " ".join(shlex.quote(x) for x in (final, W, H, FPS, DMIN, DMAX, "0")),
                           extra_mounts=(f"{trusted}:/trusted-skill:ro",), workdir="/trusted-skill")
        if chk["ok"]:
            out["check_video"] = chk["text"]
        else:
            out["check_video"] = "not measured: " + chk["error"]
            failures.append("check_video")
    else:
        out["check_video_source"] = None
        out["check_video"] = ("not run: no trusted copy of the skill (set SKILL_SRC, or use a run made by run_bench.sh, which records "
                              "logs/skill_src); the task's own copy can be edited by the model")
    md5 = in_container(f"ffmpeg -v error -i {qa} -f framemd5 -an -")
    out["frame_hashes"] = frame_hash_figures(md5, out["video"].get("frames") if isinstance(out.get("video"), dict) else None)
    if out["frame_hashes"]["error"]:
        failures.append("frame_hashes")
out["measurement_failures"] = {"count": len(failures), "names": failures}
json.dump(out, open(f"{logs}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
