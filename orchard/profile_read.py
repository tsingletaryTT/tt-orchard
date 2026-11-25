# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard profile-read`: a bounded summary of a TT-NN report folder, read through the report viewer's MCP server (issues 551 and 113).

    tt-orchard profile-read REPORT_DIR [--baseline DIR] [--venv DIR] [--out FILE] [--wheel FILE] [--accept-untested-version]
                                       [--call-timeout S] [--total-timeout S]

What it does, in order:

  1. Checks the report folder (a directory that is not a link, holding `db.sqlite` and/or an `ops_perf_results*.csv`) and the baseline folder when one is given.
  2. Checks the server: `VENV/bin/ttnn-visualizer-mcp` is a regular executable file, the installed package is a version this module has been tested with
     (`TESTED`) and the files of its `agent` package hash to the tested digest. A version it has not been tested with, or files that differ, are refused
     unless `--accept-untested-version` is given, and the summary says the run was untested. With `--wheel FILE` the wheel's sha256 is computed and
     compared with the tested wheel digest; without it the summary says the wheel digest was not measured.
  3. Starts the server (stdio, newline-delimited JSON-RPC) with `USAGE_RECORDING_DISABLED=true`, a private empty home directory, a minimal environment
     and no secrets, and asks `load_report` for the folder. The server's own `answerable` and `unanswerable` lists decide which questions can be asked.
  4. Asks a fixed list of questions (`QUESTIONS`): the slowest ops with their kernel sources and advice, time per op and per op category, the memory
     profile, the errors, the distinct argument combinations of each op, and, with `--baseline`, the change between the two reports.
  5. Writes `profile-read.json` (or `--out`): every answer cut to fixed sizes, which questions were answered and which could not be and why, what could not
     be measured, and what was cut. The whole file stays under `SUMMARY_MAX_BYTES`.

It needs no board and no lease. It never imports ttnn and never opens a network connection. The server process is a child that leads its own process group;
it is stopped through its stdin and, if it does not leave within two seconds, killed through its exact process group (the child's own pid). Replies are read
through a bounded line reader, so a server that never ends a line, never answers or floods its error output cannot hold this command past its time limits
or its memory caps. Links the server puts in its replies (`app_link`) are dropped: they point at a web page this command never opens.

Install the tested server once, then point the command at it (the default is `~/.cache/tt-orchard/ttnn-visualizer/venv`):

    python3 -m venv ~/.cache/tt-orchard/ttnn-visualizer/venv
    ~/.cache/tt-orchard/ttnn-visualizer/venv/bin/pip install ttnn-visualizer==0.112.0

Exit codes: 0 every question answered; 1 some question could not be answered (the summary says which and why); 2 refused (usage, a bad folder, a server that is
missing, a version or files that were not tested); 3 the server could not be run or stopped answering (the summary says where). A summary file is written for
0, 1 and 3.

For a coding agent (not wired into any prompt yet): `CODER_TOOLS` is the short list of server tools worth handing to a model, because the full list of 21
tool descriptions costs about 7,500 prompt tokens on every request; `FIRST_MESSAGE_NOTE` says how the server's `answerable` list belongs in the first message.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from orchard import safeio

EXIT_OK, EXIT_PARTIAL, EXIT_REFUSED, EXIT_SERVER = 0, 1, 2, 3

SUMMARY_FILE = "profile-read.json"
SUMMARY_MAX_BYTES = 96 * 1024        # the whole output file
QUESTION_MAX_BYTES = 16 * 1024       # one question's answer, after cutting
REPLY_MAX_BYTES = 4 * 1024 * 1024    # one line the server writes; a longer line ends the session
STDERR_MAX_BYTES = 16 * 1024         # the tail of the server's error output kept in the summary
CALL_TIMEOUT_S = 60.0
TOTAL_TIMEOUT_S = 300.0
STR_MAX, ITEMS_MAX, KEYS_MAX, DEPTH_MAX = 400, 25, 40, 8     # the first cut of every answer
REASON_MAX = 600
KILL_GRACE_S = 2.0
MAX_UNRELATED_MESSAGES = 50          # notifications skipped while waiting for one reply
MAX_REPORT_ENTRIES = 5000
MAX_AGENT_FILES = 400
MAX_AGENT_FILE_BYTES = 8 * 1024 * 1024
MAX_WHEEL_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
PROTOCOL_VERSION = "2025-06-18"
DEFAULT_VENV = "~/.cache/tt-orchard/ttnn-visualizer/venv"
SERVER_NAME = "ttnn-visualizer-mcp"
DROPPED_KEYS = frozenset({"app_link", "app_links"})

# Versions this module has been tested with (an audition on 2026-11-25), each with the sha256 of the wheel and of the `agent` package's files. The tree
# digest is the sha256 of one line per file, "relative/path<NUL>sha256<LF>", in path order, for every regular file under `ttnn_visualizer/agent` except
# `__pycache__`.
TESTED = {
    "0.112.0": {
        "wheel_sha256": "c47e67e64478a769b6556b77be0881706a05b0f4a0bdbb407178a48ad93a398e",
        "agent_tree_sha256": "139573be6df8aa85cacf724c9344f3b506d9e666464b9e5a520111158015a7bf",
    },
}

# The short list of tools worth giving a model: seven of the 21, so a request carries a third of the tool descriptions.
CODER_TOOLS = ("top_ops", "op_breakdown", "diff_reports", "find_operations", "operation_detail", "memory_profile", "errors")
FIRST_MESSAGE_NOTE = (
    "Call load_report first. Its reply holds two lists: answerable (the tools that can answer for this report) and unanswerable (the tools that cannot). "
    "Copy the answerable list into the first message to the coding agent, and give the agent only the tools in coder_tools that are on it. "
    "Every other tool stays with a person or a later stage."
)


class ProfileReadError(Exception):
    """A refusal or a failure with a sentence for the person who ran the command."""


class ServerFailed(ProfileReadError):
    """The server broke the protocol, stopped answering or died; the session is over."""


@dataclass(frozen=True)
class Question:
    key: str
    tool: str
    args: dict = field(default_factory=dict)
    needs_baseline: bool = False


QUESTIONS = (
    Question("top_ops", "top_ops", {"by": "device_time", "limit": 10, "include": ["config", "kernels", "advice", "flags"]}),
    Question("op_breakdown", "op_breakdown", {"group_by": "op", "by": "device_time", "limit": 15}),
    Question("op_category_breakdown", "op_breakdown", {"group_by": "op_category", "by": "device_time", "limit": 10}),
    Question("memory_profile", "memory_profile", {"limit": 15}),
    Question("errors", "errors", {"limit": 10}),
    Question("op_configurations", "op_configurations", {"limit": 10}),
    Question("baseline_comparison", "diff_reports", {"by": "device_time", "limit": 15}, needs_baseline=True),
)

_UNANSWERABLE_WHY = {
    "zone_timings": "per-zone timings need profile_log_device.csv beside the performance CSV",
    "zone_intervals": "zone intervals need profile_log_device.csv beside the performance CSV",
    "core_spread": "per-core spread needs profile_log_device.csv beside the performance CSV",
    "noc_congestion": "NoC congestion needs a tt-npe trace, which this command does not pass",
}


# ------------------------------------------------------------------------------------------------------------ bounding

@dataclass
class Cuts:
    """What bounding removed, counted so the summary can say so."""
    strings: int = 0
    list_items: int = 0
    keys: int = 0
    depth: int = 0
    app_links: int = 0

    def total(self) -> int:
        return self.strings + self.list_items + self.keys + self.depth

    def as_dict(self) -> dict:
        return {"strings_cut": self.strings, "list_items_cut": self.list_items, "keys_cut": self.keys, "depth_cut": self.depth,
                "app_links_dropped": self.app_links, "cut": self.total() > 0}


def _cut_str(text: str, max_str: int, cuts: Cuts) -> str:
    if len(text) <= max_str:
        return text
    cuts.strings += 1
    return f"{text[:max_str]}...[{len(text) - max_str} more characters cut]"


def bound_value(value, *, max_str: int, max_items: int, max_keys: int, max_depth: int, cuts: Cuts, _depth: int = 0):
    """`value` (parsed JSON) with every string, list, object and nesting level cut to the given sizes. Numbers, booleans and null pass through
    unchanged. Keys named `app_link` and `app_links` are dropped at every depth. What is cut is counted in `cuts` and marked in place."""
    if isinstance(value, str):
        return _cut_str(value, max_str, cuts)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if _depth >= max_depth:
        cuts.depth += 1
        return f"<cut: nested deeper than {max_depth} levels>"
    kw = dict(max_str=max_str, max_items=max_items, max_keys=max_keys, max_depth=max_depth, cuts=cuts, _depth=_depth + 1)
    if isinstance(value, list):
        out = [bound_value(v, **kw) for v in value[:max_items]]
        if len(value) > max_items:
            cuts.list_items += len(value) - max_items
            out.append(f"<cut: {len(value) - max_items} more items>")
        return out
    if isinstance(value, dict):
        out = {}
        kept = 0
        for k, v in value.items():
            if k in DROPPED_KEYS:
                cuts.app_links += 1
                continue
            if kept >= max_keys:
                continue
            out[_cut_str(str(k), 100, cuts)] = bound_value(v, **kw)
            kept += 1
        extra = len([k for k in value if k not in DROPPED_KEYS]) - kept
        if extra > 0:
            cuts.keys += extra
            out["<cut>"] = f"{extra} more keys"
        return out
    return _cut_str(repr(value), max_str, cuts)


def _bounded_json(value, factor: int, cuts: Cuts):
    return bound_value(value, max_str=max(40, STR_MAX // factor), max_items=max(3, ITEMS_MAX // factor), max_keys=max(6, KEYS_MAX // factor),
                       max_depth=DEPTH_MAX, cuts=cuts)


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1, sort_keys=False, allow_nan=False)


# ------------------------------------------------------------------------------------------------------------ the folders

def inspect_report_dir(path) -> dict:
    """The report files in the folder `path`, read through a descriptor so a link in the last component is refused: the file names, the folder's
    identity and which of the two inputs the server can use. Raises ProfileReadError when it is not a report folder."""
    try:
        with safeio.pinned_root(path) as pin:
            st = os.fstat(pin.fd)
            names: list[str] = []
            has_db = has_csv = False
            seen = 0
            with safeio.list_beneath(pin.fd) as listing:
                for entry in listing:
                    seen += 1
                    if seen > MAX_REPORT_ENTRIES:
                        raise ProfileReadError(f"{path} holds more than {MAX_REPORT_ENTRIES} entries")
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    name = entry.name
                    if name == "db.sqlite":
                        has_db = True
                    elif name.startswith("ops_perf_results") and name.endswith(".csv"):
                        has_csv = True
                    elif name not in ("config.json", "profile_log_device.csv", "cluster_descriptor.yaml"):
                        continue
                    names.append(name)
    except safeio.SafeReadError as exc:
        raise ProfileReadError(f"{path} cannot be used as a report folder: {exc}") from None
    if not (has_db or has_csv):
        raise ProfileReadError(f"{path} holds no db.sqlite and no ops_perf_results*.csv as regular files")
    return {"files": sorted(names)[:50], "has_db": has_db, "has_csv": has_csv, "identity": {"device": st.st_dev, "inode": st.st_ino},
            "resolved": os.path.realpath(path)}


def _site_packages(venv: Path) -> Path:
    lib = venv / "lib"
    try:
        names = sorted(n for n in os.listdir(lib) if re.fullmatch(r"python3(\.[0-9]{1,3})?", n))
    except OSError:
        raise ProfileReadError(f"{venv} has no lib/python3* directory: it is not a virtual environment") from None
    for n in names:
        sp = lib / n / "site-packages"
        if sp.is_dir():
            try:
                if any(x.startswith("ttnn_visualizer-") and x.endswith(".dist-info") for x in os.listdir(sp)):
                    return sp
            except OSError:
                continue
    raise ProfileReadError(f"{venv} has no installed ttnn-visualizer package")


def installed_version(site_packages: Path) -> str:
    """The version in the METADATA of the single `ttnn_visualizer-*.dist-info` directory in `site_packages`."""
    dists = sorted(n for n in os.listdir(site_packages) if re.fullmatch(r"ttnn_visualizer-[^/]{1,40}\.dist-info", n))
    if len(dists) != 1:
        raise ProfileReadError(f"expected one ttnn_visualizer dist-info directory in {site_packages}, found {len(dists)}")
    try:
        with safeio.pinned_root(site_packages) as pin:
            raw = safeio.read_beneath(pin.fd, (dists[0], "METADATA"), max_bytes=MAX_METADATA_BYTES)
    except safeio.SafeReadError as exc:
        raise ProfileReadError(f"cannot read the package metadata: {exc}") from None
    name = version = None
    for line in raw.decode("utf-8", "replace").splitlines()[:200]:
        if line.startswith("Name: ") and name is None:
            name = line[6:].strip()
        elif line.startswith("Version: ") and version is None:
            version = line[9:].strip()
        elif line == "":
            break
    if name is None or name.lower().replace("_", "-") != "ttnn-visualizer" or not version or not re.fullmatch(r"[0-9A-Za-z.+!_-]{1,40}", version):
        raise ProfileReadError("the package metadata does not name ttnn-visualizer with a usable version")
    return version


def agent_tree_sha256(site_packages) -> str:
    """The digest of the `ttnn_visualizer/agent` package's files (see TESTED). A link, a device or a pipe inside it is refused, as is a tree of more than
    MAX_AGENT_FILES files, so the digest cannot be made to read something else or to run long."""
    rows: list[tuple[str, str]] = []
    try:
        with safeio.pinned_root(site_packages) as pin:
            def walk(parts: tuple[str, ...], depth: int) -> None:
                if depth > 6:
                    raise ProfileReadError("the agent package is nested too deeply")
                with safeio.list_beneath(pin.fd, parts) as listing:
                    entries = sorted(((e.name, e.is_symlink(), e.is_dir(follow_symlinks=False), e.is_file(follow_symlinks=False)) for e in listing))
                for name, is_link, is_dir, is_file in entries:
                    if name == "__pycache__":
                        continue
                    if is_link or not (is_dir or is_file):
                        raise ProfileReadError(f"{'/'.join(parts + (name,))} is a link or not a regular file")
                    if is_dir:
                        walk(parts + (name,), depth + 1)
                        continue
                    if len(rows) >= MAX_AGENT_FILES:
                        raise ProfileReadError(f"the agent package holds more than {MAX_AGENT_FILES} files")
                    data = safeio.read_beneath(pin.fd, parts + (name,), max_bytes=MAX_AGENT_FILE_BYTES)
                    rows.append(("/".join((parts + (name,))[2:]), hashlib.sha256(data).hexdigest()))
            walk(("ttnn_visualizer", "agent"), 0)
    except safeio.SafeReadError as exc:
        raise ProfileReadError(f"cannot read the agent package: {exc}") from None
    if not rows:
        raise ProfileReadError("the agent package holds no files")
    h = hashlib.sha256()
    for rel, digest in sorted(rows):
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def check_server(venv: Path, *, wheel: str | None, accept_untested: bool) -> dict:
    """Everything known about the server before it starts. Raises ProfileReadError for a refusal."""
    exe = venv / "bin" / SERVER_NAME
    try:
        st = os.lstat(exe)
    except OSError:
        raise ProfileReadError(f"{exe} does not exist: install ttnn-visualizer into {venv} (see `tt-orchard profile-read --help`)") from None
    if not stat.S_ISREG(st.st_mode) or not st.st_mode & 0o111:
        raise ProfileReadError(f"{exe} is not a regular executable file")
    sp = _site_packages(venv)
    version = installed_version(sp)
    tree = agent_tree_sha256(sp)
    tested = TESTED.get(version)
    wheel_digest = None
    wheel_ok = None
    problems = []
    if wheel is not None:
        try:
            wheel_digest = safeio.sha256_file(wheel, max_bytes=MAX_WHEEL_BYTES)
        except safeio.SafeReadError as exc:
            raise ProfileReadError(f"cannot hash the wheel: {exc}") from None
        wheel_ok = bool(tested) and wheel_digest == tested["wheel_sha256"]
        if tested and not wheel_ok:
            problems.append(f"the wheel's sha256 ({wheel_digest[:12]}...) is not the tested wheel's ({tested['wheel_sha256'][:12]}...)")
    if tested is None:
        problems.append(f"ttnn-visualizer {version} has not been tested with this command (tested: {', '.join(sorted(TESTED))})")
    elif tree != tested["agent_tree_sha256"]:
        problems.append(f"the installed agent files differ from the tested {version} files")
    if problems and not accept_untested:
        raise ProfileReadError("; ".join(problems) + ". Pass --accept-untested-version to run it anyway; the summary will say the run was untested.")
    return {
        "package": "ttnn-visualizer", "version": version, "tested": not problems, "tested_versions": sorted(TESTED),
        "agent_tree_sha256": tree, "agent_tree_matches_tested": bool(tested) and tree == tested["agent_tree_sha256"],
        "wheel_sha256": wheel_digest, "wheel_matches_tested": wheel_ok,
        "tested_wheel_sha256": tested["wheel_sha256"] if tested else None, "executable": str(exe),
        "_identity": (st.st_dev, st.st_ino, st.st_mtime_ns), "_exe": str(exe),
    }


# ------------------------------------------------------------------------------------------------------------ the session

class _Session:
    """One server process and its JSON-RPC conversation. Every read has a deadline and a size cap; the child is stopped on every exit path."""

    def __init__(self, exe: str, venv: Path, *, call_timeout: float, total_deadline: float, now=time.monotonic):
        self._now = now
        self._call_timeout = call_timeout
        self._total_deadline = total_deadline
        self._home = tempfile.mkdtemp(prefix="profile-read-")
        self._buf = bytearray()
        self._n = 0
        self._err = bytearray()
        self.proc = None
        env = {"PATH": f"{venv / 'bin'}:/usr/bin:/bin", "HOME": self._home, "TMPDIR": self._home, "LANG": "C.UTF-8",
               "USAGE_RECORDING_DISABLED": "true", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self._home, env=env,
                                         start_new_session=True, close_fds=True)
        except OSError as exc:
            self._cleanup_home()
            raise ServerFailed(f"the server could not be started ({type(exc).__name__})") from None
        self._out_fd = self.proc.stdout.fileno()
        self._thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._thread.start()

    def _drain_stderr(self) -> None:
        fd = self.proc.stderr.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            self._err += chunk
            if len(self._err) > STDERR_MAX_BYTES:
                del self._err[:len(self._err) - STDERR_MAX_BYTES]

    def stderr_tail(self) -> str:
        return bytes(self._err).decode("utf-8", "replace")[-STDERR_MAX_BYTES:]

    def _send(self, message: dict) -> None:
        data = (json.dumps(message, ensure_ascii=True) + "\n").encode("ascii")
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (OSError, ValueError):
            raise ServerFailed("the server closed its input") from None

    def _line(self, deadline: float, what: str) -> bytes:
        while True:
            i = self._buf.find(b"\n")
            if i >= 0:
                if i > REPLY_MAX_BYTES:
                    raise ServerFailed(f"a reply line is longer than {REPLY_MAX_BYTES} bytes")
                line = bytes(self._buf[:i])
                del self._buf[:i + 1]
                return line
            if len(self._buf) > REPLY_MAX_BYTES:
                raise ServerFailed(f"a reply line is longer than {REPLY_MAX_BYTES} bytes")
            now = self._now()
            if now >= self._total_deadline:
                raise ServerFailed(f"the total time limit ended while waiting for {what}")
            if now >= deadline:
                raise ServerFailed(f"no reply within {self._call_timeout:g} s to {what}")
            ready, _, _ = select.select([self._out_fd], [], [], min(0.5, deadline - now, self._total_deadline - now))
            if not ready:
                continue
            chunk = os.read(self._out_fd, 65536)
            if not chunk:
                raise ServerFailed(f"the server ended its output while {what} was waiting")
            self._buf += chunk

    def rpc(self, method: str, params: dict | None, what: str):
        self._n += 1
        message = {"jsonrpc": "2.0", "id": self._n, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        deadline = self._now() + self._call_timeout
        skipped = 0
        while True:
            line = self._line(deadline, what)
            try:
                reply = safeio.parse_json_bytes(line, max_bytes=REPLY_MAX_BYTES, expect=dict)
            except safeio.SafeReadError as exc:
                raise ServerFailed(f"a reply to {what} is not valid JSON ({exc})") from None
            if reply.get("id") == self._n and not isinstance(reply.get("id"), bool):
                break
            skipped += 1
            if skipped > MAX_UNRELATED_MESSAGES:
                raise ServerFailed(f"{MAX_UNRELATED_MESSAGES} messages arrived and none answered {what}")
        if "error" in reply:
            raise ServerFailed("the server answered " + what + " with a protocol error: " + _cut_reason(str(reply.get("error"))))
        result = reply.get("result")
        if not isinstance(result, dict):
            raise ServerFailed(f"the reply to {what} has no result object")
        return result

    def notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def call_tool(self, name: str, args: dict):
        """(value, text, is_error): the tool's answer parsed as JSON when it is JSON, else None with the text."""
        result = self.rpc("tools/call", {"name": name, "arguments": args}, f"the {name} call")
        content = result.get("content")
        if not isinstance(content, list):
            raise ServerFailed(f"the {name} result has no content list")
        text = "".join(c.get("text", "") for c in content[:50] if isinstance(c, dict) and isinstance(c.get("text", ""), str))
        is_error = result.get("isError") is True
        value = None
        if not is_error:
            try:
                value = safeio.parse_json_bytes(text, max_bytes=REPLY_MAX_BYTES)
            except safeio.SafeReadError:
                value = None
        return value, text, is_error

    def _cleanup_home(self) -> None:
        shutil.rmtree(self._home, ignore_errors=True)

    def close(self) -> None:
        p = self.proc
        if p is not None:
            try:
                p.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                p.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)       # the child leads its own process group: the group id is its pid, and it is not reaped yet
                except ProcessLookupError:
                    pass
                p.wait()
            self._thread.join(timeout=2)
            for f in (p.stdout, p.stderr):
                try:
                    f.close()
                except OSError:
                    pass
        self._cleanup_home()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _cut_reason(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= REASON_MAX else text[:REASON_MAX] + "..."


# ------------------------------------------------------------------------------------------------------------ the questions

_HANDLE = re.compile(r"report-[0-9]{1,6}")


def _load(session: _Session, info: dict, label: str) -> dict:
    args = {}
    if info["has_db"]:
        args["memory_report_path"] = info["resolved"]
    if info["has_csv"]:
        args["performance_path"] = info["resolved"]
    value, text, is_error = session.call_tool("load_report", args)
    if is_error or not isinstance(value, dict):
        raise ProfileReadError(f"load_report failed for the {label}: " + _cut_reason(text))
    handle = value.get("handle")
    if not isinstance(handle, str) or not _HANDLE.fullmatch(handle):
        raise ProfileReadError(f"load_report gave no usable handle for the {label}")
    answerable = [t for t in value.get("answerable", []) if isinstance(t, str)][:60] if isinstance(value.get("answerable"), list) else []
    unanswerable = [t for t in value.get("unanswerable", []) if isinstance(t, str)][:60] if isinstance(value.get("unanswerable"), list) else []
    return {"handle": handle, "answerable": answerable, "unanswerable": unanswerable, "raw": value}


def _ask(session: _Session, q: Question, handle_args: dict, tools: set, load: dict) -> dict:
    """One question's entry (status, reason, value). Raises ServerFailed when the session is over."""
    if q.tool not in tools:
        return {"status": "could_not_answer", "tool": q.tool, "reason": f"the server has no tool {q.tool}"}
    if q.tool in load["unanswerable"]:
        return {"status": "could_not_answer", "tool": q.tool,
                "reason": f"the server's load_report lists {q.tool} as unanswerable for this report"}
    if not q.needs_baseline and q.tool not in load["answerable"]:
        return {"status": "could_not_answer", "tool": q.tool, "reason": f"{q.tool} is not in the server's answerable list for this report"}
    value, text, is_error = session.call_tool(q.tool, {**q.args, **handle_args})
    if is_error:
        return {"status": "could_not_answer", "tool": q.tool, "reason": "the server reported an error: " + _cut_reason(text)}
    if value is None:
        return {"status": "could_not_answer", "tool": q.tool, "reason": "the server's answer was not JSON: " + _cut_reason(text)}
    return {"status": "answered", "tool": q.tool, "reply_bytes": len(text), "_value": value}


def _could_not_measure(server: dict, load: dict | None) -> list[str]:
    out = []
    if server.get("wheel_sha256") is None:
        out.append("the wheel's sha256: not measured (pass --wheel FILE to hash it)")
    for tool in (load or {}).get("unanswerable", []):
        if tool in _UNANSWERABLE_WHY:
            out.append(_UNANSWERABLE_WHY[tool])
    return out


def _finish_questions(entries: dict) -> tuple[dict, Cuts]:
    """The `questions` block with every answer cut so each stays under QUESTION_MAX_BYTES and the whole under half of SUMMARY_MAX_BYTES: the cuts
    start at the first sizes and get harder until it fits. At the hardest sizes an answer that is still too large is replaced by a note that lists its
    top-level keys; at the end nothing is left but the statuses."""
    factors = (1, 2, 4, 8, 16)
    for factor in factors:
        cuts = Cuts()
        out = {}
        too_big = False
        for key, e in entries.items():
            e = dict(e)
            value = e.pop("_value", None)
            if value is not None:
                e["reply"] = _bounded_json(value, factor, cuts)
                if len(_dumps(e["reply"])) > QUESTION_MAX_BYTES:
                    if factor != factors[-1]:
                        too_big = True
                    else:
                        e["reply"] = {"cut": f"the answer is larger than {QUESTION_MAX_BYTES} bytes after cutting",
                                      "top_level_keys": sorted(map(str, value))[:30] if isinstance(value, dict) else None}
                        cuts.depth += 1
            out[key] = e
        if not too_big and len(_dumps(out)) <= SUMMARY_MAX_BYTES // 2:
            return out, cuts
    for e in out.values():
        e.pop("reply", None)
        e["cut"] = "the answer was dropped to keep the summary under its size cap"
    return out, cuts


def run(report: dict, baseline: dict | None, server: dict, venv: Path, *, call_timeout: float, total_timeout: float, now=time.monotonic) -> tuple[dict, int]:
    """Start the server, ask the questions, return (summary body, exit code)."""
    started = now()
    deadline = started + total_timeout
    entries: dict[str, dict] = {q.key: {"status": "not_asked", "tool": q.tool, "reason": "the session ended before this question"}
                                for q in QUESTIONS if baseline is not None or not q.needs_baseline}
    body: dict = {"load": None, "failure": None, "stderr_tail": ""}
    load = None
    code = EXIT_OK
    exe = server["_exe"]
    try:
        st = os.lstat(exe)
        if (st.st_dev, st.st_ino, st.st_mtime_ns) != server["_identity"]:
            raise ProfileReadError("the server file changed after it was checked")
        with _Session(exe, venv, call_timeout=call_timeout, total_deadline=deadline, now=now) as s:
            try:
                step = "initialize"
                init = s.rpc("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "tt-orchard-profile-read", "version": "1"}},
                             "the handshake")
                info = init.get("serverInfo") if isinstance(init.get("serverInfo"), dict) else {}
                body["server_info"] = {"name": _cut_reason(str(info.get("name", "")))[:80], "version": _cut_reason(str(info.get("version", "")))[:40]}
                s.notify("notifications/initialized")
                step = "tools/list"
                listed = s.rpc("tools/list", {}, "the tool list")
                tools = {t.get("name") for t in listed.get("tools", [])[:200] if isinstance(t, dict) and isinstance(t.get("name"), str)} \
                    if isinstance(listed.get("tools"), list) else set()
                body["tool_count"] = len(tools)
                step = "load_report"
                try:
                    load = _load(s, report, "report")
                except ProfileReadError as exc:
                    reason = str(exc)
                    for k, e in entries.items():
                        entries[k] = {"status": "could_not_answer", "tool": e["tool"], "reason": reason}
                    code = EXIT_PARTIAL
                    raise _SessionDone() from None
                body["load"] = {"handle": load["handle"], "answerable": load["answerable"], "unanswerable": load["unanswerable"],
                                "raw": load["raw"]}
                base_load = None
                if baseline is not None:
                    step = "load_report (baseline)"
                    try:
                        base_load = _load(s, baseline, "baseline")
                    except ProfileReadError as exc:
                        base_reason = str(exc)
                if baseline is not None and base_load is None:
                    entries["baseline_comparison"] = {"status": "could_not_answer", "tool": "diff_reports", "reason": base_reason}
                for q in QUESTIONS:
                    if q.key not in entries or (q.needs_baseline and base_load is None):
                        continue
                    step = q.key
                    handle_args = {"handle_a": base_load["handle"], "handle_b": load["handle"]} if q.needs_baseline else {"handle": load["handle"]}
                    entries[q.key] = _ask(s, q, handle_args, tools, load)
            except _SessionDone:
                pass
            except ServerFailed as exc:
                body["failure"] = {"what": _cut_reason(str(exc)), "during": step}
                if step in entries and entries[step]["status"] == "not_asked":
                    entries[step] = {"status": "failed", "tool": entries[step]["tool"], "reason": _cut_reason(str(exc))}
                code = EXIT_SERVER
            body["stderr_tail"] = s.stderr_tail()
    except ServerFailed as exc:
        body["failure"] = {"what": _cut_reason(str(exc)), "during": "start"}
        code = EXIT_SERVER
    body["questions_raw"] = entries
    body["elapsed_s"] = round(now() - started, 2)
    answered_all = all(e["status"] == "answered" for e in entries.values())
    if code == EXIT_OK and not answered_all:
        code = EXIT_PARTIAL
    return body, code


class _SessionDone(Exception):
    """Ends the question loop early without a server failure."""


# ------------------------------------------------------------------------------------------------------------ the command

def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=add_help, description="summarise a TT-NN report folder through the report viewer's MCP server")
    p.add_argument("report_dir", help="the report folder (db.sqlite and/or ops_perf_results*.csv; profile_log_device.csv adds per-core and per-zone answers)")
    p.add_argument("--baseline", metavar="DIR", help="a second report folder: adds the change between the two (this one is the baseline)")
    p.add_argument("--venv", metavar="DIR", help=f"the virtual environment with ttnn-visualizer installed (default {DEFAULT_VENV})")
    p.add_argument("--out", default=SUMMARY_FILE, metavar="FILE", help=f"where to write the summary (default ./{SUMMARY_FILE})")
    p.add_argument("--wheel", metavar="FILE", help="the wheel the package was installed from: its sha256 is checked against the tested wheel's")
    p.add_argument("--accept-untested-version", action="store_true",
                   help="run a version, or files, this command has not been tested with; the summary says the run was untested")
    p.add_argument("--call-timeout", type=float, default=CALL_TIMEOUT_S, metavar="S", help=f"seconds one server call may take (default {CALL_TIMEOUT_S:g})")
    p.add_argument("--total-timeout", type=float, default=TOTAL_TIMEOUT_S, metavar="S", help=f"seconds the whole session may take (default {TOTAL_TIMEOUT_S:g})")
    return p


def _refuse(out, message: str) -> int:
    print(f"tt-orchard profile-read: {message}", file=out)
    return EXIT_REFUSED


def main(args, *, out, now=time.monotonic) -> int:
    for flag, value in (("--call-timeout", args.call_timeout), ("--total-timeout", args.total_timeout)):
        if not safeio.fits_float(value) or value <= 0:
            return _refuse(out, f"{flag} must be a finite number of seconds above 0")
    out_path = Path(args.out)
    if out_path.is_dir() or (out_path.parent.exists() and not out_path.parent.is_dir()):
        return _refuse(out, "--out must name a file, and it names a directory")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _refuse(out, f"cannot create the directory for --out ({type(exc).__name__})")
    venv = Path(os.path.expanduser(args.venv or DEFAULT_VENV))
    try:
        report = inspect_report_dir(args.report_dir)
        baseline = inspect_report_dir(args.baseline) if args.baseline else None
        server = check_server(venv, wheel=args.wheel, accept_untested=args.accept_untested_version)
    except ProfileReadError as exc:
        return _refuse(out, str(exc))
    try:
        body, code = run(report, baseline, server, venv, call_timeout=args.call_timeout, total_timeout=args.total_timeout, now=now)
    except ProfileReadError as exc:
        return _refuse(out, str(exc))
    questions, cuts = _finish_questions(body.pop("questions_raw"))
    load = body.get("load")
    load_out = None
    if load is not None:
        load_cuts = Cuts()
        load_out = bound_value({k: v for k, v in load["raw"].items() if k not in ("answerable", "unanswerable")}, max_str=STR_MAX, max_items=ITEMS_MAX,
                               max_keys=KEYS_MAX, max_depth=DEPTH_MAX, cuts=load_cuts)
        load_out["handle"] = load["handle"]
        load_out["answerable"], load_out["unanswerable"] = load["answerable"], load["unanswerable"]
        cuts.app_links += load_cuts.app_links
        cuts.strings += load_cuts.strings
    server_out = {k: v for k, v in server.items() if not k.startswith("_")}
    server_out["stderr_tail"] = body["stderr_tail"][-STDERR_MAX_BYTES:]
    summary = {
        "schema": 1, "command": "tt-orchard profile-read", "report_dir": report["resolved"],
        "baseline_dir": baseline["resolved"] if baseline else None,
        "report": {"files": report["files"], "identity": {"device": report["identity"]["device"], "inode": report["identity"]["inode"]}},
        "server": server_out, "server_info": body.get("server_info"), "tool_count": body.get("tool_count"), "load": load_out,
        "questions": questions,
        "could_not_answer": [k for k, v in questions.items() if v["status"] != "answered"],
        "could_not_measure": _could_not_measure(server, load),
        "coder_tools": list(CODER_TOOLS), "first_message_note": FIRST_MESSAGE_NOTE,
        "limits": {**cuts.as_dict(), "reply_max_bytes": REPLY_MAX_BYTES, "question_max_bytes": QUESTION_MAX_BYTES, "summary_max_bytes": SUMMARY_MAX_BYTES,
                   "call_timeout_s": args.call_timeout, "total_timeout_s": args.total_timeout},
        "elapsed_s": body["elapsed_s"],
    }
    if body.get("failure"):
        summary["failure"] = body["failure"]
    text = _dumps(summary)
    if len(text.encode("utf-8")) > SUMMARY_MAX_BYTES:
        for q in summary["questions"].values():
            q.pop("reply", None)
            q["cut"] = "the answer was dropped to keep the summary under its size cap"
        summary["limits"]["cut"] = True
        text = _dumps(summary)
    try:
        safeio.atomic_write(out_path, text, mode=0o600)
    except (OSError, safeio.SafeReadError) as exc:
        return _refuse(out, f"cannot write {out_path} ({type(exc).__name__})")
    answered = sum(1 for v in questions.values() if v["status"] == "answered")
    print(f"tt-orchard profile-read: answered {answered} of {len(questions)} questions; wrote {out_path}", file=out)
    if summary["could_not_answer"]:
        print("not answered: " + ", ".join(summary["could_not_answer"]), file=out)
    return code
