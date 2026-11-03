#!/usr/bin/env python3
"""Serve one published can from the Hub the way a stranger would, send it one real request, and record what happened.

Run this INSIDE a gozer lease (the lease sets TT_VISIBLE_DEVICES), for example:

    gozer run --chips 1 --who "claude:hw1-cans" --reason "hub can serve check" -- \
        python3 scripts/hub_can_check.py granite-4.0-h-1b-blackhole --root <scratch dir>

What it does, in order:
  1. starts `tt-model serve episod/<can> --port <card port>` in its own process group, with
     XDG_CACHE_HOME, TT_MODEL_MODELS_DIR and HF_HOME all inside --root, so the can's venv, caches and
     downloaded weights land in one directory and nothing is written to the operator's own caches;
  2. polls the health route until it answers 200 (or the stated timeout passes) and records the start time;
  3. sends the can's kind of request (chat for text cans, a music job for audio cans), several times for the
     speed numbers, and checks the output (the text is printed for a person to read; audio is decoded with
     ffmpeg and its duration, level and silence are measured);
  4. stops the server by the exact pid it started (SIGTERM to the process group, then SIGKILL after a wait).

It writes <root>/<can>/result.json and the server log <root>/<can>/serve.log. It never prints HF_TOKEN.
Use `tt-model pull episod/<can>` first (outside the lease) with the same environment, which `--print-env` shows.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Per can: port (from the card), health route, kind, and the request settings the card's numbers were taken with.
CANS = {
    "granite-4.0-h-1b-blackhole": {"port": 20110, "health": "/health", "kind": "chat", "model": None,
                                   "card_decode_tps": 34.0, "ready_timeout": 900},
    "minicpm3-4b-blackhole": {"port": 20130, "health": "/health", "kind": "chat", "model": None,
                              "card_decode_tps": 32.7, "ready_timeout": 1800},
    "ace-step-1.5-turbo-blackhole": {"port": 20140, "health": "/health", "kind": "ace", "ready_timeout": 1800},
    "stable-audio-3-small-music-blackhole": {"port": 20170, "health": "/health", "kind": "stable",
                                             "ready_timeout": 1800},
    "minimax-music3-fiona-crapple-blackhole": {"port": 20150, "health": "/v1/health", "kind": "fiona",
                                               "ready_timeout": 2400},
    "qwen3.5-35b-a3b-blackhole-p300": {"port": 20160, "health": "/health", "kind": "chat",
                                       "model": "qwen3.5-35b-a3b", "card_decode_tps": 125.75,
                                       "chat_extra": {"chat_template_kwargs": {"enable_thinking": False}},
                                       "ready_timeout": 4800},
}
OWNER = "episod"
PROMPT = "Explain in about 150 words how a mountain range forms."
FIRST_PROMPT = "Name the three primary colors."


def env_for(root: Path) -> dict:
    """The environment that keeps every write inside `root` (HF_TOKEN is inherited, never printed)."""
    root = Path(os.path.abspath(root))              # the child runs in another working directory
    e = dict(os.environ)
    e["XDG_CACHE_HOME"] = str(root / "xdg")
    e["TT_MODEL_MODELS_DIR"] = str(root / "models")
    e["HF_HOME"] = str(root / "hf")
    e["TT_METAL_CACHE"] = str(root / "kernel-cache")
    # variables that would send writes elsewhere when the caller's shell sets them
    e["HF_HUB_CACHE"] = str(root / "hf" / "hub")
    e["TT_DIT_CACHE_DIR"] = str(root / "tt-dit-cache")
    e["TT_CACHE_PATH"] = str(root / "tt-cache")
    e["TT_CACHE_HOME"] = str(root / "tt-cache")
    for name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        e.pop(name, None)
    return e


def port_busy(port: int) -> bool:
    """True when something already listens on 127.0.0.1:port (its health route could answer for a server this script did not start)."""
    import socket
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def poll_job(get, terminal, deadline_s: float, interval: float, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Call `get(timeout)` until its status is in `terminal`; TimeoutError at the deadline. Each request is capped by the time left."""
    end = clock() + deadline_s
    while True:
        left = end - clock()
        if left <= 0:
            raise TimeoutError(f"job still not finished after {deadline_s} s")
        st = get(left)
        if st.get("status") in terminal:
            return st
        sleep(min(interval, max(end - clock(), 0)))


JOB_TIMEOUT_S = 1800
OK_STATES = ("done", "completed")


def run_ok(result: dict) -> bool:
    """The exit condition: ready, no error, and every job that reports a status ended in done or completed."""
    if not result.get("ready") or "error" in result:
        return False
    return all(r.get("status") in OK_STATES for r in result.get("requests", []) if "status" in r)


def http(method: str, url: str, body: dict | None = None, timeout: float = 600, raw: bool = False):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read()
        return r.status, {k.lower(): v for k, v in r.headers.items()}, (payload if raw else json.loads(payload or b"{}"))


def chat_stream(base: str, model, extra: dict, max_tokens: int) -> dict:
    """One streamed chat request. Decode speed = (tokens - 1) / (last chunk time - first chunk time)."""
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, **extra}
    if model:
        body["model"] = model
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    chunks, text, usage = 0, [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            obj = json.loads(line[5:])
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    now = time.monotonic()
                    first = first or now
                    last = now
                    chunks += 1
                    text.append(piece)
    total = time.monotonic() - t0
    n = (usage or {}).get("completion_tokens") or chunks
    tps = (n - 1) / (last - first) if first and last and last > first and n > 1 else None
    return {"latency_s": round(total, 3), "ttft_s": round(first - t0, 3) if first else None, "tokens": n,
            "chunks": chunks, "decode_tps": round(tps, 2) if tps else None, "text": "".join(text)}


def audio_facts(path: Path) -> dict:
    """Decode with ffmpeg: duration, rate, channels, peak, and the share of 100 ms windows above -40 dBFS."""
    import numpy as np
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, check=True, timeout=120).stdout)
    st = probe["streams"][0]
    rate, ch = int(st["sample_rate"]), int(st["channels"])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-"],
                         capture_output=True, check=True, timeout=300).stdout
    x = np.frombuffer(raw, dtype="<f4").reshape(-1, ch)
    win = rate // 10
    n = len(x) // win
    rms = np.sqrt((x[: n * win].reshape(n, win, ch) ** 2).mean(axis=(1, 2))) if n else np.array([0.0])
    return {"bytes": path.stat().st_size, "codec": st.get("codec_name"), "sample_rate": rate, "channels": ch,
            "duration_s": round(len(x) / rate, 3), "peak": round(float(np.abs(x).max()), 4),
            "windows_above_-40dBFS": round(float((20 * np.log10(np.maximum(rms, 1e-9)) > -40).mean()), 3),
            "finite": bool(np.isfinite(x).all())}


def music_requests(kind: str, base: str, out: Path) -> list:
    """The can's own request, as its card shows it. Each entry: label, asked duration, latency, file facts."""
    runs = []
    if kind == "ace":
        plan = [("first 10 s", 10), ("30 s run 1", 30), ("30 s run 2", 30)]
        for label, dur in plan:
            f = out / f"ace-{label.replace(' ', '-')}.wav"
            t0 = time.monotonic()
            status, hdr, data = http("POST", base + "/v1/music", {"prompt": "warm lo-fi hip hop, mellow rhodes piano",
                                                                   "duration": dur, "seed": 42}, raw=True)
            lat = time.monotonic() - t0
            f.write_bytes(data)
            runs.append({"label": label, "asked_s": dur, "latency_s": round(lat, 3), "http": status,
                         "x_timings": hdr.get("x-timings"), **audio_facts(f)})
    elif kind == "stable":
        plan = [("first 10 s", 10), ("10 s run 2", 10), ("30 s", 30)]
        for label, dur in plan:
            t0 = time.monotonic()
            _, _, job = http("POST", base + "/v1/audio/jobs", {
                "prompt": "A warm acoustic guitar melody with soft percussion, 100 BPM", "duration": dur,
                "steps": 8, "cfg_scale": 1.0, "seed": 42})
            st = poll_job(lambda left: http("GET", base + f"/v1/audio/jobs/{job['id']}", timeout=min(60, left))[2],
                          ("done", "error", "cancelled"), JOB_TIMEOUT_S, 1)
            lat = time.monotonic() - t0
            rec = {"label": label, "asked_s": dur, "latency_s": round(lat, 3), "status": st["status"],
                   "server_stats": st.get("stats"), "error": st.get("error")}
            if st["status"] == "done":
                f = out / f"stable-{label.replace(' ', '-')}.flac"
                _, _, data = http("GET", base + f"/v1/audio/jobs/{job['id']}/audio", raw=True)
                f.write_bytes(data)
                rec.update(audio_facts(f))
            runs.append(rec)
    elif kind == "fiona":
        plan = [("first 5 s", 5, 11), ("10 s run 1", 10, 11)]
        for label, dur, seed in plan:
            t0 = time.monotonic()
            _, _, job = http("POST", base + "/v1/music/jobs", {
                "input": "[Verse 1]\nThe kettle sings at half past three\n[Chorus]\nStay a little longer",
                "instructions": "fiona crapple, jazzy piano, raw female vocal, upright bass, brushed drums, intimate",
                "audio_duration": dur, "seed": seed, "response_format": "wav"})
            jid = job.get("id") or job.get("job_id")
            st = poll_job(lambda left: http("GET", base + f"/v1/music/jobs/{jid}", timeout=min(60, left))[2],
                          ("done", "completed", "error", "failed", "cancelled"), JOB_TIMEOUT_S, 2)
            lat = time.monotonic() - t0
            rec = {"label": label, "asked_s": dur, "latency_s": round(lat, 3), "status": st.get("status"),
                   "job": {k: v for k, v in st.items() if k != "audio"}}
            if st.get("status") in ("done", "completed"):
                f = out / f"fiona-{label.replace(' ', '-')}.wav"
                _, _, data = http("GET", base + f"/v1/music/jobs/{jid}/audio", raw=True)
                f.write_bytes(data)
                rec.update(audio_facts(f))
            runs.append(rec)
    for r in runs:
        if r.get("duration_s") and r.get("latency_s"):
            r["rtf_latency_per_audio_s"] = round(r["latency_s"] / r["duration_s"], 3)
    return runs


def group_alive(pgid: int) -> bool:
    """True while any process of the group exists (a zombie counts until its parent reaps it)."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_group(proc: subprocess.Popen, wait_s: int = 60) -> str:
    """SIGTERM the process group this script started (its pid is the group id), then SIGKILL it if members remain after wait_s.
    The launcher exiting first does not end the check: a server child can outlive it and hold the board."""
    pgid = proc.pid
    launcher_gone = proc.poll() is not None
    if not group_alive(pgid):
        return f"already exited {proc.returncode}"
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        if proc.poll() is None:
            try:
                proc.wait(0.2)
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(0.2)
        if not group_alive(pgid):
            return "stopped on SIGTERM"
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    end = time.monotonic() + 30
    while time.monotonic() < end and group_alive(pgid):
        if proc.poll() is None:
            try:
                proc.wait(0.2)
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(0.2)
    return "needed SIGKILL" + (" (launcher had exited)" if launcher_gone else "") + ("" if not group_alive(pgid) else "; group still alive")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("can")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--print-env", action="store_true")
    ap.add_argument("--pull-only", action="store_true", help="run `tt-model pull` in the isolated environment and exit")
    ap.add_argument("--ready-timeout", type=int)
    ap.add_argument("--hf-token-path", help="set HF_TOKEN_PATH for the server (a token FILE the Hub client reads; "
                    "the stable-audio can needs one, see the run log). The file is read by the client, never copied or printed.")
    a = ap.parse_args()
    if a.can not in CANS:
        ap.error(f"unknown can {a.can!r}; known: {', '.join(CANS)}")
    spec = CANS[a.can]
    a.root = Path(os.path.abspath(a.root))
    work = a.root / a.can
    work.mkdir(parents=True, exist_ok=True)
    env = env_for(work)
    if a.hf_token_path:
        env["HF_TOKEN_PATH"] = os.path.abspath(a.hf_token_path)
    if a.print_env:
        for k in ("XDG_CACHE_HOME", "TT_MODEL_MODELS_DIR", "HF_HOME", "TT_METAL_CACHE"):
            print(f"export {k}={env[k]}")
        return 0
    if a.pull_only:
        t = time.monotonic()
        cp = subprocess.run(["tt-model", "pull", f"{OWNER}/{a.can}"], env=env, cwd=str(work),
                            capture_output=True, text=True)
        (work / "pull.log").write_text(cp.stdout + cp.stderr)
        print(json.dumps({"can": a.can, "pull_exit": cp.returncode, "pull_s": round(time.monotonic() - t, 1)}))
        print("\n".join((cp.stdout + cp.stderr).splitlines()[-25:]))
        return cp.returncode
    base = f"http://127.0.0.1:{spec['port']}"
    timeout = a.ready_timeout or spec["ready_timeout"]
    result = {"can": a.can, "repo": f"{OWNER}/{a.can}", "port": spec["port"], "ready_timeout_s": timeout,
              "tt_visible_devices": env.get("TT_VISIBLE_DEVICES")}
    if port_busy(spec["port"]):
        result["error"] = f"port {spec['port']} is already in use; refusing to start (its health route could answer for another server)"
        result["ready"] = False
        (work / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1
    log = open(work / "serve.log", "wb")
    t0 = time.monotonic()
    proc = subprocess.Popen(["tt-model", "serve", f"{OWNER}/{a.can}", "--port", str(spec["port"]),
                             "--no-update-check"], env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=str(work))
    result["serve_pid"] = proc.pid
    try:
        ready = False
        while time.monotonic() - t0 < timeout:
            if proc.poll() is not None:
                result["serve_exit"] = proc.returncode
                break
            try:
                if http("GET", base + spec["health"], timeout=5)[0] == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(3)
        result["ready"] = ready
        result["start_to_ready_s"] = round(time.monotonic() - t0, 1)
        result["loadavg_at_ready"] = os.getloadavg()
        if ready:
            kind = spec["kind"]
            if kind == "chat":
                runs = []
                extra = spec.get("chat_extra", {})
                body = {"messages": [{"role": "user", "content": FIRST_PROMPT}], "max_tokens": 96, **extra}
                if spec.get("model"):
                    body["model"] = spec["model"]
                t1 = time.monotonic()
                st, _, obj = http("POST", base + "/v1/chat/completions", body)
                result["first_request"] = {"latency_s": round(time.monotonic() - t1, 3), "http": st,
                                           "reply": obj["choices"][0]["message"]["content"],
                                           "finish": obj["choices"][0].get("finish_reason"),
                                           "usage": obj.get("usage"), "timings": obj.get("timings")}
                for _ in range(3):
                    runs.append(chat_stream(base, spec.get("model"), extra, 64))
                result["stream_runs"] = runs
                tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
                result["decode_tps_median"] = sorted(tps)[len(tps) // 2] if tps else None
                result["card_decode_tps"] = spec["card_decode_tps"]
            else:
                outdir = work / "clips"
                outdir.mkdir(exist_ok=True)
                result["requests"] = music_requests(kind, base, outdir)
    except Exception as exc:  # the record must survive any failure; the traceback goes to the log
        import traceback
        result["error"] = f"{type(exc).__name__}: {exc}"
        log.write(traceback.format_exc().encode())
    finally:
        result["loadavg_at_end"] = os.getloadavg()
        result["stop"] = stop_group(proc)
        log.close()
        result["log_tail"] = (work / "serve.log").read_text(errors="replace").splitlines()[-60:]
        (work / "result.json").write_text(json.dumps(result, indent=2))
    printable = {k: v for k, v in result.items() if k != "log_tail"}
    print(json.dumps(printable, indent=2)[:6000])
    return 0 if run_ok(result) else 1


if __name__ == "__main__":
    sys.exit(main())
