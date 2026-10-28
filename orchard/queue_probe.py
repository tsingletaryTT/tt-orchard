# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A queue and concurrency probe for any local server that speaks the audio job API (issue 174). Standard library only.

The API it needs (the job server of the Stable Audio 3 demo is one implementation; this module has no code from it):

    GET    /v1/health                    {"status": "ok" | "loading" | "error", "queue_depth": n, "queue_max": n, "busy": bool, ...}
    POST   /v1/audio/jobs                {prompt, duration, steps, cfg_scale, seed} -> 202 {"id"} | 429 queue full | 503 not ready
    GET    /v1/audio/jobs/<id>           {"status": queued | running | done | error | cancelled, "queue_position": n, "stats": {"total_s": s} | null}
    DELETE /v1/audio/jobs/<id>           drops a queued job

For each concurrency level N (default 1, 2 and 4) the probe sends N job requests at the same instant from N threads, then polls every accepted job until
all have ended, and records for each request the HTTP status and when the reply came; for each job the queue positions seen, the time it waited, the
server's own service time and its final status. The level summary says how many were accepted and how many refused (by status code), the largest queue
position seen, whether the jobs ran one at a time (their service intervals, from the server's own `total_s`, do not overlap by more than two polls),
whether the queue was first in, first out, and how long the whole level took. A job that is still queued at the timeout is cancelled so the server is left
clean. Between levels the probe waits until the server reports itself idle.

Limits, written into every result: times are seen by polling, so they have a resolution of about the poll interval; the service time is the server's own
number; one clip length and one seed are used; the host is shared and its one-minute load average is recorded at the start and end of each level.

Safety: only a server on this machine's loopback address is probed, never on a port that other work uses (8000, 8100 to 8103, 8780); no proxy named in the
environment is used; every reply is capped at 1 MB and parsed with `safeio.parse_json_bytes` (no NaN, no repeated keys, no deep nesting), and a reply that
fails any of that is recorded as a problem and does not stop the probe.

    python -m orchard.queue_probe --url http://127.0.0.1:20100 --out probe.json --levels 1,2,4 --prompt "..." --duration 10 --steps 8 --cfg-scale 1.0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from orchard import resident, safeio

DEFAULT_LEVELS = (1, 2, 4)
MAX_LEVEL = 16
FORBIDDEN_PORTS = frozenset({8000, 8100, 8101, 8102, 8103, 8780})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_REPLY_BYTES = 1 << 20
TERMINAL = frozenset({"done", "error", "cancelled"})
ID_PATTERN = re.compile(r"[0-9A-Za-z_-]{1,64}")
UNREADABLE_LIMIT = 5                      # consecutive unreadable polls of one job before it is called unknown
SETTING_KEYS = ("prompt", "duration", "steps", "cfg_scale", "seed")


class ProbeError(Exception):
    """The probe cannot run as asked."""


def check_url(url) -> str:
    """The base URL of a local server on an allowed port, without a trailing slash; anything else is refused."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except (ValueError, AttributeError, TypeError):
        raise ProbeError("the URL cannot be read") from None
    if parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS:
        raise ProbeError("the probe talks to http servers on this machine's loopback address only")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ProbeError("the URL is a base address: no user, path or query")
    if port is None:
        raise ProbeError("the URL needs a port")
    if port in FORBIDDEN_PORTS:
        raise ProbeError(f"port {port} belongs to other work and is never probed")
    return f"http://{parts.netloc}"


def check_levels(levels) -> tuple:
    """The concurrency levels as a tuple of whole numbers from 1 to MAX_LEVEL."""
    out = []
    for n in levels:
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_LEVEL:
            raise ProbeError(f"a concurrency level is a whole number from 1 to {MAX_LEVEL}, got {n!r}")
        out.append(n)
    if not out:
        raise ProbeError("at least one concurrency level is needed")
    return tuple(out)


def check_settings(s) -> dict:
    """The job request: the documented keys only, `prompt` a non-empty string, the numbers finite and the whole numbers whole."""
    if not isinstance(s, dict) or not isinstance(s.get("prompt"), str) or not s["prompt"].strip():
        raise ProbeError("the settings need a prompt")
    if set(s) - set(SETTING_KEYS):
        raise ProbeError(f"the job request has these keys only: {', '.join(SETTING_KEYS)}")
    for key in ("duration", "cfg_scale"):
        if key in s and (not isinstance(s[key], (int, float)) or isinstance(s[key], bool) or not math.isfinite(s[key])):
            raise ProbeError(f"{key} must be a finite number")
    for key in ("steps", "seed"):
        if key in s and (not isinstance(s[key], int) or isinstance(s[key], bool)):
            raise ProbeError(f"{key} must be a whole number")
    return dict(s)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _opener():
    """An opener that uses no proxy from the environment and follows no redirect (the server is local and answers directly)."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def request(method: str, url: str, body: dict | None = None, timeout: float = 30.0):
    """(http status or None, JSON object or None, problem or None). A reply over MAX_REPLY_BYTES, a reply that is invalid or non-strict JSON, and a reply that is no object give a
    problem and no object; a connection that fails gives no status."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        resp = _opener().open(req, timeout=timeout)
        code = resp.status
    except urllib.error.HTTPError as exc:
        resp, code = exc, exc.code
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, None, f"{method} {url.split('://', 1)[-1].split('/', 1)[-1] or '/'}: no reply ({type(exc).__name__})"
    try:
        raw = resp.read(MAX_REPLY_BYTES + 1)
    except OSError as exc:
        return code, None, f"{method}: the reply could not be read ({type(exc).__name__})"
    finally:
        resp.close()
    if len(raw) > MAX_REPLY_BYTES:
        return code, None, f"{method}: the reply is larger than {MAX_REPLY_BYTES} bytes"
    try:
        obj = safeio.parse_json_bytes(raw, max_bytes=MAX_REPLY_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        return code, None, f"{method}: the reply is not a usable JSON object ({type(exc).__name__})"
    return code, obj, None


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def probe_level(base: str, n: int, settings: dict, *, poll_s: float = 0.25, timeout_s: float = 600.0, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Send `n` job requests at once and follow the accepted jobs to the end (or to `timeout_s`). Returns the record described in the module text."""
    base = check_url(base)
    t0 = clock()
    load_start = resident.host_loadavg_1m()
    barrier = threading.Barrier(n)
    requests: list = [None] * n
    problems: list[str] = []
    lock = threading.Lock()

    def post(i: int) -> None:
        barrier.wait()
        sent = clock() - t0
        code, obj, problem = request("POST", base + "/v1/audio/jobs", settings)
        job_id = obj.get("id") if code == 202 and obj else None
        if code == 202 and not (isinstance(job_id, str) and ID_PATTERN.fullmatch(job_id)):
            problem = problem or "POST: the 202 reply has no usable id"
            job_id = None
        requests[i] = {"index": i, "sent_at_s": round(sent, 4), "replied_at_s": round(clock() - t0, 4), "http_status": code, "job_id": job_id}
        if problem:
            with lock:
                problems.append(problem)
    threads = [threading.Thread(target=post, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rejected: dict[str, int] = {}
    for r in requests:
        if r["job_id"] is None:
            key = "no reply" if r["http_status"] is None else (str(r["http_status"]) if r["http_status"] != 202 else "202 unusable")
            rejected[key] = rejected.get(key, 0) + 1
    jobs = {r["job_id"]: {"id": r["job_id"], "accepted_at_s": r["replied_at_s"], "positions": [], "first_position": None, "first_running_s": None,
                          "done_s": None, "final_status": None, "server_total_s": None, "audio_s": None, "error": "", "unreadable": 0}
            for r in requests if r["job_id"] is not None}
    timed_out = False
    pending = set(jobs)
    while pending:
        if clock() - t0 > timeout_s:
            timed_out = True
            break
        for jid in sorted(pending):
            job = jobs[jid]
            code, obj, problem = request("GET", f"{base}/v1/audio/jobs/{jid}")
            now = round(clock() - t0, 4)
            status = obj.get("status") if obj else None
            if problem or code != 200 or not isinstance(status, str):
                job["unreadable"] += 1
                if problem and problem not in problems:
                    problems.append(problem)
                if job["unreadable"] >= UNREADABLE_LIMIT:
                    job["final_status"] = "unknown"
                    pending.discard(jid)
                continue
            job["unreadable"] = 0
            pos = obj.get("queue_position")
            if isinstance(pos, int) and not isinstance(pos, bool):
                if job["first_position"] is None:
                    job["first_position"] = pos
                if not job["positions"] or job["positions"][-1] != pos:
                    job["positions"].append(pos)
            if status == "running" and job["first_running_s"] is None:
                job["first_running_s"] = now
            job["final_status"] = status
            if status in TERMINAL:
                job["done_s"] = now
                stats = obj.get("stats")
                if isinstance(stats, dict):
                    job["server_total_s"], job["audio_s"] = _num(stats.get("total_s")), _num(stats.get("audio_s"))
                err = obj.get("error")
                job["error"] = err[:200] if isinstance(err, str) else ""
                pending.discard(jid)
        if pending:
            sleep(poll_s)
    for jid in sorted(pending):                       # at the timeout: drop what is still queued, so the server is left clean
        job = jobs[jid]
        if job["final_status"] == "queued":
            code, obj, _problem = request("DELETE", f"{base}/v1/audio/jobs/{jid}")
            if code == 200 and obj and isinstance(obj.get("status"), str):
                job["final_status"] = obj["status"]
        elif job["final_status"] is None:
            job["final_status"] = "timeout"
    rows = list(jobs.values())
    for job in rows:
        start = (job["done_s"] - job["server_total_s"]) if job["done_s"] is not None and job["server_total_s"] is not None else job["first_running_s"]
        job["start_s"] = None if start is None else round(start, 4)
        job["wait_s"] = None if start is None else round(max(0.0, start - job["accepted_at_s"]), 4)
        job["end_to_end_s"] = None if job["done_s"] is None else round(job["done_s"] - job["accepted_at_s"], 4)
        del job["unreadable"]
    ended = [j for j in rows if j["start_s"] is not None and j["done_s"] is not None]
    ended.sort(key=lambda j: j["start_s"])
    one_at_a_time = None
    if len(ended) >= 2:
        one_at_a_time = all(b["start_s"] >= a["done_s"] - 2 * poll_s for a, b in zip(ended, ended[1:]))
    fifo = None
    if len(ended) >= 2 and all(j["first_position"] is not None for j in ended):
        firsts = [j["first_position"] for j in ended]
        fifo = all(x <= y for x, y in zip(firsts, firsts[1:]))
    positions = [p for j in rows for p in j["positions"]]
    done_times = [j["done_s"] for j in rows if j["done_s"] is not None]
    return {"concurrency": n, "requests": requests, "accepted": len(jobs), "rejected": rejected, "jobs": rows,
            "max_queue_position": max(positions) if positions else 0, "ran_one_at_a_time": one_at_a_time, "first_in_first_out": fifo,
            "makespan_s": round(max(done_times) - min(r["sent_at_s"] for r in requests), 4) if done_times else None,
            "timed_out": timed_out, "problems": problems, "host_loadavg_1m_at_start": load_start, "host_loadavg_1m_at_end": resident.host_loadavg_1m()}


def wait_idle(base: str, timeout_s: float, poll_s: float, clock=time.monotonic, sleep=time.sleep) -> dict:
    """The server's health once it says it is idle (not busy, nothing queued); a ProbeError when it does not within `timeout_s`."""
    t0 = clock()
    while True:
        code, obj, _problem = request("GET", base + "/v1/health")
        if code == 200 and obj and obj.get("busy") is False and obj.get("queue_depth") == 0:
            return obj
        if clock() - t0 > timeout_s:
            raise ProbeError("the server did not become idle in time, so the next level would start on a busy server")
        sleep(poll_s)


def run_probe(url: str, settings: dict, *, levels=DEFAULT_LEVELS, poll_s: float = 0.25, timeout_s: float = 600.0, out=None) -> dict:
    """Probe each level in turn and return (and write, when `out` is given) the record."""
    base = check_url(url)
    levels = check_levels(levels)
    settings = check_settings(settings)
    if not isinstance(poll_s, (int, float)) or isinstance(poll_s, bool) or not 0 < poll_s <= 10 or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ProbeError("poll_s must be above 0 and at most 10, and timeout_s above 0")
    if out is not None and Path(out).exists():
        raise ProbeError(f"{out} exists; a result is never overwritten")
    code, health, problem = request("GET", base + "/v1/health")
    if code != 200 or not health:
        raise ProbeError(f"the health check did not answer ({problem or code})")
    if health.get("status") != "ok":
        raise ProbeError(f"the health check says the server is {health.get('status')!r}, which is not ready")
    wait_idle(base, timeout_s, poll_s)
    rec_levels = []
    for n in levels:
        rec_levels.append(probe_level(base, n, settings, poll_s=poll_s, timeout_s=timeout_s))
        wait_idle(base, timeout_s, poll_s)
    _code, after, _problem = request("GET", base + "/v1/health")
    record = {"settings": settings, "poll_s": poll_s, "timeout_s": timeout_s, "health_before": health, "health_after": after, "levels": rec_levels,
              "limits": [f"Times are seen by polling every {poll_s:g} s, so they have a resolution of about that.",
                         "The service time of a job is the server's own total_s. Whether jobs ran one at a time is judged from those service intervals.",
                         "One clip length, one prompt and one seed are used at every level.",
                         "The host is shared; its one-minute load average is recorded at the start and end of each level."]}
    if out is not None:
        try:
            safeio.atomic_write(out, json.dumps(record, indent=1) + "\n", replace=False)
        except FileExistsError:
            raise ProbeError(f"{out} exists; a result is never overwritten") from None
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True, help="the local server, for example http://127.0.0.1:20100")
    p.add_argument("--out", required=True)
    p.add_argument("--levels", default=",".join(str(n) for n in DEFAULT_LEVELS))
    p.add_argument("--poll-s", type=float, default=0.25)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--cfg-scale", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args(argv)
    try:
        levels = [int(x) for x in args.levels.split(",")]
        record = run_probe(args.url, {"prompt": args.prompt, "duration": args.duration, "steps": args.steps, "cfg_scale": args.cfg_scale, "seed": args.seed},
                           levels=levels, poll_s=args.poll_s, timeout_s=args.timeout_s, out=args.out)
    except (ProbeError, ValueError) as exc:
        print(f"queue probe: {exc}", file=sys.stderr)
        return 2
    for lv in record["levels"]:
        waits = [j["wait_s"] for j in lv["jobs"] if j["wait_s"] is not None]
        print(f"concurrency {lv['concurrency']}: accepted {lv['accepted']}, refused {lv['rejected'] or 'none'}, max queue position {lv['max_queue_position']}, "
              f"one at a time {lv['ran_one_at_a_time']}, first in first out {lv['first_in_first_out']}, longest wait {max(waits) if waits else 'n/a'} s, "
              f"whole level {lv['makespan_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
