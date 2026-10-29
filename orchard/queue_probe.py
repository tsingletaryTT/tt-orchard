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
position seen, whether the jobs ran one at a time (their service intervals, from the server's own `total_s`, do not overlap by more than two polls), and how long
the whole level took. A job that is still queued at the timeout is cancelled so the server is left clean. Between levels the probe waits until the server reports
itself idle.

Order. Requests sent at the same instant have no known order of arrival, and the queue positions seen after a job has started cannot give one. So the order test
is separate: it sends three jobs one after another (each reply comes before the next request, so the order of arrival is known) and says whether they started in that
order. It says "unknown" when fewer than two of them ran to the end.

Deadlines. Every request is bounded by the time that is left for its level (and by 30 s), the polling loop checks the deadline between jobs, and the clean-up
of queued jobs has its own small budget, so a stalled server cannot hold the probe much past `--timeout-s`.

Limits, written into every result: times are seen by polling, so they have a resolution of about the poll interval; the service time is the server's own
number; one clip length and one seed are used; the host is shared and its one-minute load average is recorded at the start and end of each level.

Safety: only a server on this machine's loopback address is probed, never on a port that other work uses (8000, 8100 to 8103, 8780); no proxy named in the
environment is used; every reply is capped at 1 MB and parsed with `safeio.parse_json_bytes` (no NaN, no repeated keys, no deep nesting), and a reply that
fails any of that is recorded as a problem and does not stop the probe.

    python -m orchard.queue_probe --url http://127.0.0.1:20100 --out probe.json --levels 1,2,4 --prompt "..." --duration 10 --steps 8 --cfg-scale 1.0 --seed 42
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import re
import socket
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from orchard import resident, safeio

DEFAULT_LEVELS = (1, 2, 4)
MAX_LEVEL = 16
FORBIDDEN_PORTS = frozenset({8000, 8100, 8101, 8102, 8103, 8780})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_REPLY_BYTES = 1 << 20
TERMINAL = frozenset({"done", "error", "cancelled"})
REQUEST_TIMEOUT_S = 30.0                  # no single request waits longer than this
CLEANUP_TIMEOUT_S = 5.0                   # one DELETE of a queued job at the end of a level
CLEANUP_BUDGET_S = 20.0                   # all the DELETEs of a level together
ORDER_JOBS = 3                            # jobs in the order test
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


def request(method: str, url: str, body: dict | None = None, timeout: float = 30.0):
    """(http status or None, JSON object or None, problem or None). The whole request, from connecting to the last byte of the reply, has one absolute
    limit of `timeout` seconds: a timer shuts the socket at the limit, and the body is read in chunks that check the clock, so a server that sends one byte
    at a time cannot hold the probe past it. No proxy and no redirect is used (http.client has neither). A reply over MAX_REPLY_BYTES, a reply that is
    invalid or non-strict JSON, and a reply that is no object give a problem and no object; a connection that fails gives no status."""
    where = f"{method} {urllib.parse.urlsplit(url).path or '/'}"
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        return None, None, f"{where}: the time limit is not a positive number"
    parts = urllib.parse.urlsplit(url)
    data = None if body is None else json.dumps(body).encode()
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    t_end = time.monotonic() + timeout

    held: dict = {"sock": None}                    # the socket is kept here: a reply that closes the connection makes http.client forget it

    def stop() -> None:
        sock = held["sock"]
        try:
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    timer = threading.Timer(timeout, stop)
    timer.daemon = True
    timer.start()
    code = None
    try:
        conn.connect()
        held["sock"] = conn.sock
        if time.monotonic() >= t_end:
            return None, None, f"{where}: no complete reply within {timeout:g} s"
        conn.request(method, parts.path or "/", body=data, headers={"Content-Type": "application/json"} if data is not None else {})
        resp = conn.getresponse()
        code = resp.status
        chunks, size = [], 0
        while size <= MAX_REPLY_BYTES:
            if time.monotonic() >= t_end:
                return code, None, f"{where}: the reply did not finish within {timeout:g} s"
            chunk = resp.read(min(65536, MAX_REPLY_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        raw = b"".join(chunks)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        if time.monotonic() >= t_end:
            return code, None, f"{where}: no complete reply within {timeout:g} s"
        return code, None, f"{where}: no reply ({type(exc).__name__})"
    finally:
        timer.cancel()
        conn.close()
    if len(raw) > MAX_REPLY_BYTES:
        return code, None, f"{where}: the reply is larger than {MAX_REPLY_BYTES} bytes"
    try:
        obj = safeio.parse_json_bytes(raw, max_bytes=MAX_REPLY_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        return code, None, f"{where}: the reply is not a usable JSON object ({type(exc).__name__})"
    return code, obj, None


def check_times(poll_s, timeout_s) -> None:
    """poll_s above 0 and at most 10, timeout_s above 0, both finite numbers (a NaN would make every deadline comparison false)."""
    for name, v in (("poll_s", poll_s), ("timeout_s", timeout_s)):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ProbeError(f"{name} must be a finite number")
    if not 0 < poll_s <= 10 or timeout_s <= 0:
        raise ProbeError("poll_s must be above 0 and at most 10, and timeout_s above 0")


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _bound(deadline: float, clock, cap: float = REQUEST_TIMEOUT_S) -> float:
    """The timeout of one request: at most `cap`, and at most the time left until `deadline` (never below 50 ms)."""
    return max(0.05, min(cap, deadline - clock()))


def _new_job(job_id: str, accepted_at: float) -> dict:
    return {"id": job_id, "accepted_at_s": accepted_at, "positions": [], "first_position": None, "first_running_s": None, "done_s": None,
            "final_status": None, "server_total_s": None, "audio_s": None, "error": "", "unreadable": 0}


def _follow(base: str, jobs: dict, problems: list, *, t0: float, deadline: float, poll_s: float, clock, sleep) -> bool:
    """Poll the jobs until each has ended, was called unknown, or `deadline` passes; then cancel what is still queued. Returns True when the deadline
    passed first. The deadline is checked between jobs, and every request is bounded by it."""
    pending = set(jobs)
    timed_out = False
    while pending and not timed_out:
        for jid in [j for j in jobs if j in pending]:          # in the order the jobs were accepted (an id sort would be arbitrary)
            if clock() >= deadline:
                timed_out = True
                break
            job = jobs[jid]
            code, obj, problem = request("GET", f"{base}/v1/audio/jobs/{jid}", timeout=_bound(deadline, clock))
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
        if pending and not timed_out:
            sleep(poll_s)
    cleanup_end = clock() + CLEANUP_BUDGET_S
    for jid in [j for j in jobs if j in pending]:     # at the deadline: drop what is still queued, so the server is left clean
        job = jobs[jid]
        if job["final_status"] in ("queued", None):          # a job never observed may still be queued, so it is cancelled too
            code, obj, _problem = request("DELETE", f"{base}/v1/audio/jobs/{jid}", timeout=_bound(cleanup_end, clock, CLEANUP_TIMEOUT_S))
            if code == 200 and obj and isinstance(obj.get("status"), str):
                job["final_status"] = obj["status"]
            elif job["final_status"] is None:
                job["final_status"] = "timeout"
        elif job["final_status"] is None:
            job["final_status"] = "timeout"
    return timed_out


def _derive(rows: list, poll_s: float) -> None:
    """Add each job's start, wait and end-to-end time. The start is the end of the job minus the server's own service time when both are known, and the
    first time the job was seen running otherwise."""
    for job in rows:
        start = (job["done_s"] - job["server_total_s"]) if job["done_s"] is not None and job["server_total_s"] is not None else job["first_running_s"]
        job["start_s"] = None if start is None else round(start, 4)
        job["wait_s"] = None if start is None else round(max(0.0, start - job["accepted_at_s"]), 4)
        job["end_to_end_s"] = None if job["done_s"] is None else round(job["done_s"] - job["accepted_at_s"], 4)
        del job["unreadable"]


def probe_level(base: str, n: int, settings: dict, *, poll_s: float = 0.25, timeout_s: float = 600.0, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Send `n` job requests at once and follow the accepted jobs to the end (or to `timeout_s`). Returns the record described in the module text."""
    base = check_url(base)
    check_times(poll_s, timeout_s)
    t0 = clock()
    deadline = t0 + timeout_s
    load_start = resident.host_loadavg_1m()
    barrier = threading.Barrier(n)
    requests: list = [None] * n
    problems: list[str] = []
    lock = threading.Lock()

    def post(i: int) -> None:
        barrier.wait()
        sent = clock() - t0
        code, obj, problem = request("POST", base + "/v1/audio/jobs", settings, timeout=_bound(deadline, clock))
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
    jobs = {r["job_id"]: _new_job(r["job_id"], r["replied_at_s"]) for r in requests if r["job_id"] is not None}
    timed_out = _follow(base, jobs, problems, t0=t0, deadline=deadline, poll_s=poll_s, clock=clock, sleep=sleep)
    rows = list(jobs.values())
    _derive(rows, poll_s)
    ended = [j for j in rows if j["start_s"] is not None and j["done_s"] is not None]
    ended.sort(key=lambda j: j["start_s"])
    one_at_a_time = None
    if len(ended) >= 2:
        one_at_a_time = all(b["start_s"] >= a["done_s"] - 2 * poll_s for a, b in zip(ended, ended[1:]))
    positions = [p for j in rows for p in j["positions"]]
    done_times = [j["done_s"] for j in rows if j["done_s"] is not None]
    return {"concurrency": n, "requests": requests, "accepted": len(jobs), "rejected": rejected, "jobs": rows,
            "max_queue_position": max(positions) if positions else 0, "ran_one_at_a_time": one_at_a_time,
            "makespan_s": round(max(done_times) - min(r["sent_at_s"] for r in requests), 4) if done_times else None,
            "timed_out": timed_out, "problems": problems, "host_loadavg_1m_at_start": load_start, "host_loadavg_1m_at_end": resident.host_loadavg_1m()}


def probe_order(base: str, settings: dict, *, n: int = ORDER_JOBS, poll_s: float = 0.25, timeout_s: float = 600.0, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Send `n` jobs one after another (each reply before the next request, so the order of arrival is known) and say whether they started in that order.
    `first_in_first_out` is None when fewer than two of the accepted jobs ran to the end."""
    base = check_url(base)
    check_times(poll_s, timeout_s)
    t0 = clock()
    deadline = t0 + timeout_s
    problems: list[str] = []
    jobs: dict = {}
    submit_order: list = []
    refused: dict[str, int] = {}
    for _ in range(n):
        code, obj, problem = request("POST", base + "/v1/audio/jobs", settings, timeout=_bound(deadline, clock))
        job_id = obj.get("id") if code == 202 and obj else None
        if code == 202 and not (isinstance(job_id, str) and ID_PATTERN.fullmatch(job_id)):
            problem, job_id = problem or "POST: the 202 reply has no usable id", None
        if problem:
            problems.append(problem)
        if job_id is None:
            key = "no reply" if code is None else (str(code) if code != 202 else "202 unusable")
            refused[key] = refused.get(key, 0) + 1
            continue
        jobs[job_id] = _new_job(job_id, round(clock() - t0, 4))
        submit_order.append(job_id)
    timed_out = _follow(base, jobs, problems, t0=t0, deadline=deadline, poll_s=poll_s, clock=clock, sleep=sleep)
    rows = list(jobs.values())
    _derive(rows, poll_s)
    ended = [jobs[j] for j in submit_order if jobs[j]["start_s"] is not None and jobs[j]["done_s"] is not None]
    fifo = None
    if len(ended) >= 2 and len(ended) == len(submit_order):
        starts = [j["start_s"] for j in ended]
        # Jobs that end within two polls of each other get start estimates that follow the order they were polled in; no order can be read from those.
        if all(abs(b - a) >= 2 * poll_s for a, b in zip(starts, starts[1:])):
            fifo = all(a <= b for a, b in zip(starts, starts[1:]))
    return {"jobs_sent": n, "accepted": len(jobs), "refused": refused, "submit_order": submit_order, "first_in_first_out": fifo,
            "start_s_in_submit_order": [jobs[j]["start_s"] for j in submit_order], "timed_out": timed_out, "problems": problems}


def wait_idle(base: str, timeout_s: float, poll_s: float, clock=time.monotonic, sleep=time.sleep):
    """(the server's last health answer or None, idle). `idle` is True once the server says it is not busy and has nothing queued, and False when it
    does not within `timeout_s` (a job that hangs while running keeps it busy). It never raises for a busy server; the caller reports it."""
    check_times(poll_s, timeout_s)
    t0 = clock()
    deadline = t0 + timeout_s
    while True:
        code, obj, _problem = request("GET", base + "/v1/health", timeout=_bound(deadline, clock))
        if code == 200 and obj and obj.get("busy") is False and obj.get("queue_depth") == 0:
            return obj, True
        if clock() - t0 > timeout_s:
            return (obj if code == 200 else None), False
        sleep(poll_s)


def run_probe(url: str, settings: dict, *, levels=DEFAULT_LEVELS, poll_s: float = 0.25, timeout_s: float = 600.0, out=None) -> dict:
    """Probe each level in turn and return (and write, when `out` is given) the record."""
    base = check_url(url)
    levels = check_levels(levels)
    settings = check_settings(settings)
    check_times(poll_s, timeout_s)
    if out is not None and Path(out).exists():
        raise ProbeError(f"{out} exists; a result is never overwritten")
    code, health, problem = request("GET", base + "/v1/health", timeout=min(REQUEST_TIMEOUT_S, timeout_s))
    if code != 200 or not health:
        raise ProbeError(f"the health check did not answer ({problem or code})")
    if health.get("status") != "ok":
        raise ProbeError(f"the health check says the server is {health.get('status')!r}, which is not ready")
    _h, idle = wait_idle(base, timeout_s, poll_s)
    if not idle:
        raise ProbeError("the server did not become idle in time, so a level would start on a busy server")
    rec_levels, notes = [], []
    order = None
    for n in levels:
        rec_levels.append(probe_level(base, n, settings, poll_s=poll_s, timeout_s=timeout_s))
        _h, idle = wait_idle(base, timeout_s, poll_s)
        if not idle:                                   # a hung running job: say so, and do not start the next level on a busy server
            notes.append(f"The server was still busy after the level of {n}, so the later levels and the order test were not run.")
            break
    else:
        order = probe_order(base, settings, poll_s=poll_s, timeout_s=timeout_s)
        _h, idle = wait_idle(base, timeout_s, poll_s)
        if not idle:
            notes.append("The server was still busy after the order test.")
    _code, after, _problem = request("GET", base + "/v1/health", timeout=min(REQUEST_TIMEOUT_S, timeout_s))
    record = {"settings": settings, "poll_s": poll_s, "timeout_s": timeout_s, "health_before": health, "health_after": after, "levels": rec_levels, "order": order, "notes": notes,
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
              f"one at a time {lv['ran_one_at_a_time']}, longest wait {max(waits) if waits else 'n/a'} s, "
              f"whole level {lv['makespan_s']} s")
    order = record["order"]
    if order is not None:
        print(f"order test: {order['accepted']} of {order['jobs_sent']} jobs accepted one after another, first in first out {order['first_in_first_out']}")
    for note in record["notes"]:
        print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
