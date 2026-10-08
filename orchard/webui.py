# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard ui`: watch and control the runs on this machine from a browser.

    tt-orchard ui [--host 127.0.0.1] [--port 8780]

It listens on this machine's loopback address (the same rule the tier config keeps, `tiers.LOCAL_HOSTS`);
from another computer, forward the port: `ssh -L 8780:localhost:8780 <box>`. `--lan` listens on the local
network instead, with no login: anyone who can reach the port can pause, abort and start runs. Use it only
on a network you trust, and only when the operator asked for it.

The hardware view is tt-toplike itself (github.com/tenstorrent/tt-toplike): its own terminal UI, run in a
pseudo-terminal of the page's size and streamed to the page, where xterm.js draws it. It is view-only. The
page sends no keystrokes; it picks a mode from TOPLIKE_MODES and tt-toplike is started with that `--mode`
(or `--rotate`), so nothing typed in a browser reaches a program on this machine. Each viewer gets its own
tt-toplike (its rendering sends only what changed, so one stream cannot be shared), at most TOPLIKE_MAX at
once, stopped when the page goes away.

What it reads, and how. Everything is read the way `status` and `watch` read it: `status.collect` for a
run's facts, `ledger.read_entries` for the ledger, and a `narrate.Narrator` for the live feed. None of
these takes the ledger lock, so the UI cannot get in a run's way, and a UI failure cannot stop one.
`gozer status` is a subprocess, so its answer is shared for GOZER_TTL_S seconds.

What it can change, and nothing else:
- a control word for a run (`supervisor.Control`): pause and abort while a supervisor is alive, resume
  while it is paused. A word the run cannot act on is refused, so no stale word waits for a later run;
- a detached `tt-orchard bringup` for a new model, only when the preflight has no block, or for a run
  that ended blocked or stopped (a retry, which the supervisor records as `resume by=retry`).
There is no publish, upload, push, reset, lease or delete action, and there will not be one here.

Requests from other web pages are refused: on loopback every request must name a loopback host (a page that
reaches 127.0.0.1 under its own name by DNS rebinding is turned away), and every POST needs the token this
process made at start, which only this page can read, plus an Origin that matches the host the request was
sent to (the page's own origin).

Files are served from an allow-list of run-relative paths (the gate files, evidence, the bundle's text,
BLOCKED.md and the coder log), never the ledger, the agent transcripts, the control file or anything a
link points outside the run. What is served has tokens, the home path and the host name taken out,
with the scrub's own patterns (orchard/scrub.py).
"""
from __future__ import annotations

import base64
import fcntl
import hmac
import io
import json
import os
import re
import secrets
import select
import signal
import socket
import struct
import subprocess
import termios
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from orchard import __version__, bringup_config, lexicon, narrate, status, ui
from orchard.ledger import LedgerCorrupt, read_entries
from orchard.scrub import HOME_PATH, TOKEN_PATTERNS
from orchard.stages import STAGES
from orchard.supervisor import Control
from orchard.tiers import LOCAL_HOSTS

DEFAULT_PORT = 8780
GOZER_TTL_S = 15.0              # one `gozer status` serves every request in this window
FILE_TAIL_BYTES = 256 * 1024    # a longer file is served as its last 256 KB
BODY_LIMIT = 64 * 1024
SSE_PING_S = 15.0
CPU_TIER_PORT = 11434           # ollama's default, which the setup script configures
CHECKOUT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).with_name("web")
JS, CSS, WOFF2 = "text/javascript; charset=utf-8", "text/css; charset=utf-8", "font/woff2"
# Every file the page loads, by URL. Nothing else under orchard/web is served (not the vendored licences or maps).
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/static/app.js": ("app.js", JS),
          "/static/scene.js": ("scene.js", JS),
          "/static/app.css": ("app.css", CSS),
          "/static/vendor/xterm/xterm.js": ("vendor/xterm/xterm.js", JS),
          "/static/vendor/xterm/xterm.css": ("vendor/xterm/xterm.css", CSS),
          "/static/vendor/xterm/addon-fit.js": ("vendor/xterm/addon-fit.js", JS),
          "/static/vendor/pixelify-sans/pixelify-sans-latin-400-normal.woff2":
              ("vendor/pixelify-sans/pixelify-sans-latin-400-normal.woff2", WOFF2),
          "/static/vendor/pixelify-sans/pixelify-sans-latin-700-normal.woff2":
              ("vendor/pixelify-sans/pixelify-sans-latin-700-normal.woff2", WOFF2)}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
ALLOWED = re.compile(
    r"^(?:BLOCKED\.md|blocked\.json|coder\.log"
    r"|stages/\d+(?:\.partial-\d+)?/(?:delta|reference|result|hw_test|hw_tests|handoff|test-result)\.json"
    r"|stages/\d+(?:\.partial-\d+)?/evidence/[A-Za-z0-9._-]+\.(?:json|txt|md)"
    r"|stages/4/tests/\d+/test-result\.json"
    r"|stages/8/bundle/[A-Za-z0-9._-]+\.(?:md|txt))$")
LISTED = ("BLOCKED.md", "blocked.json", "coder.log", "stages/*/*.json", "stages/*/evidence/*",
          "stages/4/tests/*/test-result.json", "stages/8/bundle/*")
# Runs that need a person first, then the ones still going, then the finished ones.
ATTENTION = {"blocked": 0, "stopped-or-crashed": 0, "unreadable": 0, "paused": 1, "running": 2,
             "not-started": 3, "ready-for-operator-review": 4, "aborted": 5}
RETRYABLE = ("blocked", "stopped-or-crashed")
# tt-toplike's own views (`--mode`), plus its kiosk rotation (`--rotate`). `hivemind`, its opt-in sniffer of other
# processes' activity, is left out: an open page should not start it.
TOPLIKE_MODES = ("normal", "starfield", "castle", "flow", "arcade", "training", "rotate")
TOPLIKE_MAX = 4
CHIP_LINE = re.compile(r"^\s*chip\s+(\d+)\s+(\S+)\s+(\S+)\s*(.*)$")
BOARD_LINE = re.compile(r"^\s*board\s+(\S+)(?:\s+\(([^)]*)\))?")


class WebError(Exception):
    code = HTTPStatus.INTERNAL_SERVER_ERROR


class BadRequest(WebError):
    code = HTTPStatus.BAD_REQUEST


class Forbidden(WebError):
    code = HTTPStatus.FORBIDDEN


class NotFound(WebError):
    code = HTTPStatus.NOT_FOUND


class Conflict(WebError):
    code = HTTPStatus.CONFLICT


class Unavailable(WebError):
    code = HTTPStatus.SERVICE_UNAVAILABLE


def spawn_detached(argv, *, log_path, cwd) -> int:
    """Start `argv` in its own session, output to `log_path`, and return its pid. The supervisor installs
    its signal handlers only on a main thread, so a run is always its own process, never a UI thread."""
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(cwd), start_new_session=True)
    return proc.pid


def pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        os.waitpid(pid, os.WNOHANG)          # reap our own exited child, which would linger as a zombie
    except ChildProcessError:
        pass
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][:1] != "Z"
    except (OSError, IndexError):
        return True


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def parse_gozer(text: str) -> list[dict]:
    """Boards and chips from `gozer status`. Lines that look like neither are skipped."""
    boards: list[dict] = []
    for line in text.splitlines():
        if m := BOARD_LINE.match(line):
            boards.append({"id": m.group(1), "kind": m.group(2) or "", "chips": []})
        elif (m := CHIP_LINE.match(line)) and boards:
            owner, _, note = m.group(4).partition(" — ")
            boards[-1]["chips"].append({"index": int(m.group(1)), "bdf": m.group(2), "state": m.group(3),
                                        "owner": owner.strip(), "note": note.strip()})
    return boards


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Feed(narrate.Narrator):
    """The narrator, with each line kept as a dict instead of printed. `poll()` returns the new lines."""

    def __init__(self, run_dir, *, replay=30, clock=time.time, redact=lambda text: text):
        super().__init__(run_dir, ui.Style("none", False), io.StringIO(), replay=replay, clock=clock,
                         tz=timezone.utc)
        self._events: list[dict] = []
        self._redact = redact

    def _emit(self, line, ts=None):
        self._events.append({"ts": ts or _iso(self.clock()), "actor": line.actor,
                             "icon": narrate.ACTOR_ICON.get(line.actor, ""),
                             "role": "bad" if line.bad else narrate.ACTOR_ROLE.get(line.actor, "dim"),
                             "text": self._redact(line.text), "bad": bool(line.bad)})
        self._last_print = self.clock()

    def poll(self) -> list[dict]:
        super().poll()
        out, self._events = self._events, []
        return out


class WebApp:
    """What the pages ask for. Every outside signal is a parameter, so a test supplies its own."""

    def __init__(self, *, runs_root, config_path, cfg, token=None, gozer_status=None, clock=time.time,
                 spawn=spawn_detached, preflight=None, collect_kwargs=None, port_open=port_open,
                 pid_running=pid_running, hostname=None, home=None, sse_poll_s=1.0, toplike=None, lan=False):
        self.runs_root = Path(runs_root)
        self.config_path = Path(config_path)
        self.cfg = cfg
        self.token = token or secrets.token_urlsafe(24)
        self._gozer_fn = gozer_status or status._gozer_status
        self.clock = clock
        self.spawn = spawn
        self._preflight = preflight
        self.collect_kwargs = dict(collect_kwargs or {})
        self.port_open = port_open
        self.pid_running = pid_running
        self.hostname = socket.gethostname() if hostname is None else hostname
        self.home = os.path.expanduser("~") if home is None else home
        self.sse_poll_s = sse_poll_s
        self.toplike_cmd = list(toplike) if toplike else None     # argv of tt-toplike, or None when absent
        self._toplike: dict[str, Toplike] = {}
        self.lan = lan                  # True: listen on the network, no login (the operator's choice)
        self._gozer_cache: tuple[float, str] | None = None
        self._lock = threading.Lock()
        self._launches: dict[str, dict] = {}

    # ---- shared signals ----
    def _gozer(self) -> str:
        with self._lock:
            now = self.clock()
            if self._gozer_cache is None or now - self._gozer_cache[0] > GOZER_TTL_S:
                self._gozer_cache = (now, self._gozer_fn())
            return self._gozer_cache[1]

    def _run(self, name: str) -> Path:
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise NotFound(f"no run named {name!r}")
        run = self.runs_root / name
        if not (run / "ledger.jsonl").is_file():
            raise NotFound(f"no run named {name!r} in {self.runs_root}")
        return run

    def _collect(self, run: Path) -> dict:
        return status.collect(run, gozer_status=self._gozer, **self.collect_kwargs)

    def _launch_alive(self, name: str) -> dict | None:
        info = self._launches.get(name)
        if info and self.pid_running(info["pid"]):
            return info
        return None

    # ---- reading ----
    def meta(self) -> dict:
        coder = getattr(self.cfg, "coder", None)
        return {
            "version": __version__, "token": self.token, "hostname": self.hostname, "lan": self.lan,
            "palette": dict(ui.ROLES),
            "states": {k: {"emoji": v.emoji, "role": v.role, "word": v.phrase} for k, v in lexicon.STATES.items()},
            "stages": [{"number": s.number, "real": s.name, "emoji": lexicon.STAGES[s.number].emoji,
                        "name": lexicon.STAGES[s.number].name} for s in STAGES],
            "actors": {k: {"icon": narrate.ACTOR_ICON.get(k, ""), "role": narrate.ACTOR_ROLE.get(k, "dim"),
                           "about": v} for k, v in lexicon.ACTORS.items()},
            "marks": dict(lexicon.STATUS_MARKS),
            "toplike": {"available": bool(self.toplike_cmd), "modes": list(TOPLIKE_MODES),
                        "command": " ".join(self.toplike_cmd[-1:]) if self.toplike_cmd else None},
            "config": {"path": str(self.config_path), "runs_root": str(self.runs_root),
                       "coder": None if coder is None else {"target": getattr(coder, "target", None),
                                                            "port": getattr(coder, "port", None),
                                                            "chips": getattr(coder, "chips", None)}},
        }

    def machine(self) -> dict:
        text = self._gozer()
        coder_port = getattr(getattr(self.cfg, "coder", None), "port", None)
        disk = self.collect_kwargs.get("disk_free_gb", status._disk_free_gb)
        boards = parse_gozer(text)
        return {
            "gozer_ok": bool(text.strip()),
            "lan": self.lan,
            "boards": boards,
            "coder": {"port": coder_port, "up": bool(coder_port and self.port_open(coder_port))},
            "cpu_tier": {"port": CPU_TIER_PORT, "up": self.port_open(CPU_TIER_PORT)},
            "disk_free_gb": disk(self.runs_root),
            "now": self.clock(),
        }

    def list_runs(self) -> list[dict]:
        rows = []
        if self.runs_root.is_dir():
            for run in sorted(self.runs_root.iterdir()):
                if not run.is_dir() or not NAME_RE.match(run.name) or not (run / "ledger.jsonl").is_file():
                    continue
                try:
                    f = self._collect(run)
                except (LedgerCorrupt, OSError, ValueError, KeyError) as exc:
                    rows.append({"name": run.name, "state": "unreadable", "error": f"{type(exc).__name__}: {exc}",
                                 "model": None, "stage": None, "hint": "the ledger cannot be read; see "
                                 "`tt-orchard status --run-dir` for the error", "last_ts": None,
                                 "supervisor_alive": False, "blocked": None})
                    continue
                rows.append({"name": run.name, "model": f["model"], "state": f["state"], "stage": f["stage"],
                             "hint": f["hint"], "supervisor_alive": f["supervisor"]["alive"],
                             "blocked": f["blocked"], "pause": f["pause"],
                             "last_ts": f["last_events"][-1]["ts"] if f["last_events"] else None,
                             "launching": self._launch_alive(run.name) is not None, "error": None})
        rows.sort(key=lambda r: r["last_ts"] or "", reverse=True)
        rows.sort(key=lambda r: ATTENTION.get(r["state"], 9))
        return rows

    def run_detail(self, name: str) -> dict:
        run = self._run(name)
        try:
            f = self._collect(run)
        except LedgerCorrupt as exc:
            raise Conflict(f"the ledger of {name} fails its hash check: {exc}") from exc
        f["name"] = name
        f["files"] = self.list_files(run)
        f["launch"] = self._launch_alive(name)
        f["block"] = None
        if f["state"] == "blocked" and f["blocked"]:
            stage = None
            try:
                stage = json.loads((run / "blocked.json").read_text(encoding="utf-8")).get("stage")
            except (OSError, ValueError):
                stage = f["stage"]["current"]
            code = f["blocked"].get("code") or "unclassified"
            f["block"] = {"code": code, "reason": f["blocked"].get("reason"), "stage": stage,
                          "tried": [self.redact(t) for t in narrate.what_was_tried(run, stage=stage)]
                          if stage is not None else [],
                          "unblock": narrate.how_to_unblock(code, f["model"])}
        return f

    def ledger(self, name: str, after: int = 0, limit: int = 500) -> dict:
        run = self._run(name)
        try:
            entries = read_entries(run / "ledger.jsonl")
        except LedgerCorrupt as exc:
            raise Conflict(str(exc)) from exc
        rows = [{"seq": e["seq"], "ts": e["ts"], "event": e["event"], "stage": e["stage"],
                 "summary": self.redact(status.summarize(e)),
                 "kind": e["data"].get("decision") or e["data"].get("what") or e["data"].get("result")}
                for e in entries if e["seq"] > after]
        return {"entries": rows[:limit], "last": entries[-1]["seq"] if entries else 0, "more": len(rows) > limit}

    # ---- files ----
    def list_files(self, run: Path) -> list[str]:
        found = set()
        for pattern in LISTED:
            for p in run.glob(pattern):
                rel = p.relative_to(run).as_posix()
                if p.is_file() and not p.is_symlink() and ALLOWED.match(rel):
                    found.add(rel)
        return sorted(found)[:400]

    def redact(self, text: str) -> str:
        for _, pattern in TOKEN_PATTERNS:
            text = pattern.sub("<token>", text)
        if self.home and self.home != "/":
            text = text.replace(self.home, "~")
        text = HOME_PATH.sub("<home>", text)
        if self.hostname and self.hostname != "localhost":
            text = re.sub(rf"\b{re.escape(self.hostname)}\b", "<host>", text)
        return text

    def read_file(self, name: str, rel: str) -> str:
        run = self._run(name)
        if not isinstance(rel, str) or "\\" in rel or rel.startswith("/") or ".." in rel.split("/") \
                or not ALLOWED.match(rel):
            raise Forbidden(f"{rel!r} is not a file the UI shows")
        path = run / rel
        try:
            real = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise NotFound(f"{rel} does not exist in {name}") from exc
        if not real.is_relative_to(run.resolve()) or path.is_symlink():
            raise Forbidden(f"{rel} points outside the run")
        size = real.stat().st_size
        with open(real, "rb") as fh:
            if size > FILE_TAIL_BYTES:
                fh.seek(size - FILE_TAIL_BYTES)
                data = fh.read()
                data = data[data.find(b"\n") + 1:]
                head = f"… (the last {FILE_TAIL_BYTES // 1024} KB of {size / 1e6:.1f} MB)\n"
            else:
                data, head = fh.read(), ""
        return head + self.redact(data.decode("utf-8", errors="replace"))

    # ---- control ----
    def control(self, name: str, word: str) -> dict:
        run = self._run(name)
        f = self._collect(run)
        allowed = {"pause": ("running",), "abort": ("running", "paused"), "resume": ("paused",)}
        if word not in Control.COMMANDS:
            raise Conflict(f"{word!r} is not a control word; the words are {', '.join(Control.COMMANDS)}")
        if f["state"] not in allowed[word]:
            raise Conflict(f"{name} is {f['state']}, so {word} has nothing to act on")
        Control(run).write(word)
        return self.run_detail(name)

    def _launch(self, name: str, argv: list[str]) -> dict:
        if self._launch_alive(name):
            raise Conflict(f"a bring-up for {name} was started from this page and is still starting")
        log_path = self.runs_root / f"{name}.ui-launch.log"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        pid = self.spawn(argv, log_path=log_path, cwd=CHECKOUT)
        info = {"pid": pid, "argv": argv, "log": str(log_path), "started": self.clock(), "model": argv[3]}
        self._launches[name] = info
        return {"run": name, "pid": pid, "log": str(log_path)}

    def launches(self) -> list[dict]:
        """Bring-ups started from this page, newest first, with the end of their output: what a run says
        before its ledger exists (the preflight, the download) and why it stopped if it never started."""
        rows = []
        for name, info in sorted(self._launches.items(), key=lambda kv: -kv[1]["started"]):
            try:
                with open(info["log"], "rb") as fh:
                    fh.seek(max(0, os.path.getsize(info["log"]) - 8192))
                    tail = fh.read().decode("utf-8", errors="replace")
            except OSError:
                tail = ""
            rows.append({"run": name, "model": info["model"], "pid": info["pid"], "started": info["started"],
                         "alive": self.pid_running(info["pid"]),
                         "has_ledger": (self.runs_root / name / "ledger.jsonl").is_file(),
                         "log_tail": self.redact("\n".join(tail.splitlines()[-40:]))})
        return rows

    def _bringup_argv(self, model: str, extra: list[str]) -> list[str]:
        return [sys.executable, str(CHECKOUT / "bin" / "tt-orchard"), "bringup", model, "--quiet",
                *extra, "--config", str(self.config_path)]

    def retry(self, name: str) -> dict:
        run = self._run(name)
        f = self._collect(run)
        if f["state"] not in RETRYABLE:
            raise Conflict(f"{name} is {f['state']}; only a blocked or stopped run is retried")
        if not f["model"] or not MODEL_RE.match(f["model"]):
            raise Conflict(f"the ledger of {name} names no model to retry")
        return self._launch(name, self._bringup_argv(f["model"], ["--run-dir", str(run)]))

    @staticmethod
    def _check_ids(model, base):
        if not isinstance(model, str) or not MODEL_RE.match(model):
            raise BadRequest("give a Hugging Face model id as org/name")
        if base not in (None, "") and (not isinstance(base, str) or not MODEL_RE.match(base)):
            raise BadRequest("the base must be a bundle or model id as org/name")
        return model, base or None

    def preflight(self, model, base=None) -> dict:
        model, base = self._check_ids(model, base)
        checks = self._preflight(model, base)
        name = bringup_config.slug(model)
        rows = []
        for c in checks:
            data = c.data if isinstance(c.data, dict) else {}
            rows.append({"name": c.name, "status": c.status, "detail": self.redact(c.detail), "reason": c.reason,
                         "candidates": data.get("candidates"), "base": data.get("base")})
        return {"model": model, "base": base, "run": name, "run_dir": str(self.runs_root / name),
                "resuming": (self.runs_root / name / "ledger.jsonl").is_file(),
                "ok": not any(c.status == "block" for c in checks), "checks": rows}

    def bringup(self, model, base=None) -> dict:
        model, base = self._check_ids(model, base)
        name = bringup_config.slug(model)
        if (self.runs_root / name / "ledger.jsonl").is_file():
            state = self._collect(self.runs_root / name)["state"]
            if state in ("running", "paused"):
                raise Conflict(f"{name} is already {state}")
        checks = self._preflight(model, base)
        stops = [c for c in checks if c.status == "block"]
        if stops:
            raise Conflict("the preflight blocks: " + ", ".join(f"{c.name} ({c.reason})" for c in stops))
        return self._launch(name, self._bringup_argv(model, ["--base", base] if base else []))


    # ---- tt-toplike ----
    def toplike_open(self, mode: str, cols: int, rows: int) -> tuple[str, "Toplike"]:
        if not self.toplike_cmd:
            raise Unavailable("tt-toplike is not installed here; start the UI with --toplike PATH")
        if mode not in TOPLIKE_MODES and mode != "":
            raise BadRequest(f"unknown view {mode!r}; the views are {', '.join(TOPLIKE_MODES)}")
        cols, rows = _term_size(cols, rows)
        argv = [*self.toplike_cmd, "--quiet"] + (["--rotate"] if mode == "rotate" else
                                                  ["--mode", mode] if mode else [])
        with self._lock:
            self._toplike = {k: t for k, t in self._toplike.items() if t.alive()}
            if len(self._toplike) >= TOPLIKE_MAX:
                raise Conflict(f"{TOPLIKE_MAX} hardware views are already open; close one first")
            sid = secrets.token_urlsafe(12)
            t = Toplike(argv, cols, rows)
            self._toplike[sid] = t
        return sid, t

    def toplike_close(self, sid: str) -> None:
        with self._lock:
            t = self._toplike.pop(sid, None)
        if t:
            t.close()

    def _session(self, sid: str) -> "Toplike":
        t = self._toplike.get(sid)
        if t is None:
            raise NotFound("no such hardware view")
        return t

    def toplike_resize(self, sid: str, cols, rows) -> None:
        self._session(sid).resize(*_term_size(cols, rows))

    def toplike_size(self, sid: str) -> tuple[int, int]:
        return self._session(sid).size()

    def toplike_pid(self, sid: str) -> int | None:
        t = self._toplike.get(sid)
        return t.proc.pid if t else None

    def toplike_sessions(self) -> int:
        return len(self._toplike)


def _term_size(cols, rows) -> tuple[int, int]:
    try:
        cols, rows = int(cols), int(rows)
    except (TypeError, ValueError) as exc:
        raise BadRequest("cols and rows must be numbers") from exc
    if not (20 <= cols <= 400 and 8 <= rows <= 200):
        raise BadRequest("the terminal must be 20-400 columns by 8-200 rows")
    return cols, rows


class Toplike:
    """tt-toplike in a pseudo-terminal of a fixed size. Output is read from `fd`; nothing is ever written to it,
    so the program gets no input from the page."""

    def __init__(self, argv, cols, rows):
        master, slave = os.openpty()
        self.fd = master
        self._winsize(slave, cols, rows)
        # The environment tt-toplike's own terminal app gives the TUI (src/bin/app.rs). TMUX is dropped: the TUI
        # runs in this pseudo-terminal, not in tmux, and with TMUX set it gives up true colour (src/ui/colors.rs).
        env = {k: v for k, v in os.environ.items() if k not in ("TMUX", "TMUX_PANE", "STY")}
        env.update({"COLORTERM": "truecolor", "TERM": "xterm-256color", "LANG": "en_US.UTF-8"})
        try:
            self.proc = subprocess.Popen(list(argv), stdin=slave, stdout=slave, stderr=slave, env=env,
                                         start_new_session=True, close_fds=True)
        finally:
            os.close(slave)
        self._size = (cols, rows)

    @staticmethod
    def _winsize(fd, cols, rows):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def resize(self, cols, rows):
        self._winsize(self.fd, cols, rows)
        self._size = (cols, rows)
        try:
            os.killpg(self.proc.pid, signal.SIGWINCH)
        except ProcessLookupError:
            pass

    def size(self) -> tuple[int, int]:
        return self._size

    def alive(self) -> bool:
        return self.proc.poll() is None

    def read(self, timeout: float) -> bytes | None:
        """Bytes the program wrote, b"" when there are none yet, None when it has exited."""
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return b"" if self.alive() else None
        try:
            data = os.read(self.fd, 65536)
        except OSError:                      # EIO: the program closed the terminal
            return None
        return data if data else None

    def close(self):
        for sig, wait in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            if self.proc.poll() is not None:
                break
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.close(self.fd)
        except OSError:
            pass


# ---- HTTP -------------------------------------------------------------------------------------------

def _host_ok(header: str | None) -> bool:
    if not header:
        return False
    host = header[1:].split("]", 1)[0] if header.startswith("[") else header.rsplit(":", 1)[0] \
        if header.count(":") == 1 else header
    return host in LOCAL_HOSTS


class Handler(BaseHTTPRequestHandler):
    server_version = "tt-orchard-ui"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> WebApp:
        return self.server.app

    def log_message(self, fmt, *args):     # quiet: the terminal shows the URL and errors only
        pass

    def _headers(self, code, ctype, length=None, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy",
                         # xterm.js styles its terminal cells inline (tt-toplike's colours), so inline styles are
                         # allowed. Scripts stay 'self' only: no inline or remote script can run.
                         "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
                         "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self._headers(code, "application/json", len(body))
        self.wfile.write(body)

    def _error(self, code, message):
        self._json(code, {"error": message})

    def _guard_host(self) -> bool:
        if not self.app.lan and not _host_ok(self.headers.get("Host")):
            self._error(HTTPStatus.FORBIDDEN, "this UI answers only on a loopback address")
            return False
        return True

    def _route(self, method):
        parts = urlsplit(self.path)
        q = {k: v[-1] for k, v in parse_qs(parts.query).items()}
        segs = [s for s in parts.path.split("/") if s]
        try:
            if method == "GET":
                return self._get(parts.path, segs, q)
            return self._post(segs)
        except WebError as exc:
            self._error(exc.code, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                 # a UI bug answers 500 and the server goes on
            try:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")
            except OSError:
                pass

    def do_GET(self):
        if self._guard_host():
            self._route("GET")

    def do_POST(self):
        if not self._guard_host():
            return
        token = self.headers.get("X-Orchard-Token") or ""
        origin = self.headers.get("Origin")
        if not hmac.compare_digest(token.encode(), self.app.token.encode()):
            return self._error(HTTPStatus.FORBIDDEN, "missing or wrong token; reload the page")
        if origin is not None and urlsplit(origin).netloc.lower() != (self.headers.get("Host") or "").lower():
            return self._error(HTTPStatus.FORBIDDEN, "a request from another site is refused")
        self._route("POST")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > BODY_LIMIT:
            raise BadRequest("request too large")
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError as exc:
            raise BadRequest("the body is not JSON") from exc
        if not isinstance(data, dict):
            raise BadRequest("the body must be a JSON object")
        return data

    def _get(self, path, segs, q):
        if path in ASSETS:
            name, ctype = ASSETS[path]
            body = (WEB / name).read_bytes()
            self._headers(HTTPStatus.OK, ctype, len(body))
            return self.wfile.write(body)
        if segs[:1] != ["api"]:
            raise NotFound("not found")
        app = self.app
        if segs == ["api", "meta"]:
            return self._json(200, app.meta())
        if segs == ["api", "machine"]:
            return self._json(200, app.machine())
        if segs == ["api", "toplike"]:
            return self._toplike_stream(q)
        if segs == ["api", "launches"]:
            return self._json(200, {"launches": app.launches()})
        if segs == ["api", "runs"]:
            return self._json(200, {"runs": app.list_runs(), "runs_root": str(app.runs_root)})
        if len(segs) == 3 and segs[1] == "runs":
            return self._json(200, app.run_detail(segs[2]))
        if len(segs) == 4 and segs[1] == "runs":
            name, what = segs[2], segs[3]
            if what == "ledger":
                try:
                    after = int(q.get("after", "0"))
                except ValueError as exc:
                    raise BadRequest("after must be a number") from exc
                return self._json(200, app.ledger(name, after))
            if what == "file":
                text = app.read_file(name, q.get("path", "")).encode()
                self._headers(200, "text/plain; charset=utf-8", len(text))
                return self.wfile.write(text)
            if what == "events":
                return self._events(name, q.get("replay", "30"))
        raise NotFound("not found")

    def _post(self, segs):
        app = self.app
        body = self._body()
        if segs == ["api", "preflight"]:
            return self._json(200, app.preflight(body.get("model"), body.get("base")))
        if segs == ["api", "bringup"]:
            return self._json(200, app.bringup(body.get("model"), body.get("base")))
        if len(segs) == 4 and segs[:2] == ["api", "toplike"] and segs[3] == "resize":
            app.toplike_resize(segs[2], body.get("cols"), body.get("rows"))
            return self._json(200, {"ok": True})
        if len(segs) == 4 and segs[:2] == ["api", "runs"]:
            if segs[3] == "control":
                return self._json(200, app.control(segs[2], body.get("word")))
            if segs[3] == "retry":
                return self._json(200, app.retry(segs[2]))
        raise NotFound("not found")

    def _toplike_stream(self, q):
        return _toplike_stream_impl(self, q)

    def _events(self, name, replay_arg):
        run = self.app._run(name)
        replay = True if replay_arg == "all" else (int(replay_arg) if replay_arg.isdigit() else 30) or False
        feed = Feed(run, replay=replay, redact=self.app.redact)
        first = feed.poll()
        self.close_connection = True
        self._headers(HTTPStatus.OK, "text/event-stream; charset=utf-8", extra={"Connection": "close"})

        def send(chunk: str):
            self.wfile.write(chunk.encode())
            self.wfile.flush()

        send(": connected\n\n")
        for e in first:
            send("data: " + json.dumps(e) + "\n\n")
        last = time.monotonic()
        while not self.server.stopping.is_set():
            time.sleep(self.app.sse_poll_s)
            try:
                events = feed.poll()
            except Exception as exc:             # the feed is best effort; the page reconnects
                send("event: feed-error\ndata: " + json.dumps({"error": str(exc)}) + "\n\n")
                return
            for e in events:
                send("data: " + json.dumps(e) + "\n\n")
            if events:
                last = time.monotonic()
            elif time.monotonic() - last > SSE_PING_S:
                send(": ping\n\n")
                last = time.monotonic()


def _peer_gone(sock) -> bool:
    """True once the browser has closed the connection. A quiet tt-toplike screen sends nothing, so a failed
    write alone would leave its process running after the page went away."""
    try:
        ready, _, _ = select.select([sock], [], [], 0)
        return bool(ready) and sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


def _toplike_stream_impl(handler, q):
    app = handler.app
    sid, term = app.toplike_open(q.get("mode", ""), q.get("cols", "120"), q.get("rows", "36"))
    try:
        handler.close_connection = True
        handler._headers(HTTPStatus.OK, "text/event-stream; charset=utf-8", extra={"Connection": "close"})

        def send(chunk: str):
            handler.wfile.write(chunk.encode())
            handler.wfile.flush()

        send("event: session\ndata: " + json.dumps({"id": sid, "cols": term.size()[0], "rows": term.size()[1]}) + "\n\n")
        last = time.monotonic()
        while not handler.server.stopping.is_set():
            if _peer_gone(handler.connection):
                return
            data = term.read(0.25)
            if data is None:
                send("event: exit\ndata: " + json.dumps({"code": term.proc.poll()}) + "\n\n")
                return
            if data:
                send("data: " + base64.b64encode(data).decode() + "\n\n")
                last = time.monotonic()
            elif time.monotonic() - last > SSE_PING_S:
                send(": ping\n\n")
                last = time.monotonic()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        app.toplike_close(sid)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, app, family):
        self.address_family = family
        self.app = app
        self.stopping = threading.Event()
        super().__init__(addr, handler)

    def shutdown(self):
        self.stopping.set()
        super().shutdown()


def make_server(app: WebApp, host: str, port: int) -> Server:
    if host not in LOCAL_HOSTS and not app.lan:
        raise ValueError(f"{host} is not a loopback address; the UI listens only on {', '.join(sorted(LOCAL_HOSTS))} "
                         "unless it is started with --lan")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    return Server((host, port), Handler, app, family)
