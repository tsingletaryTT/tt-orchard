"""Agent steps: one fresh model context per stage phase, and the tool-call loop that serves it.

This module owns three things.

1. The agent's environment (`agent_env`). It starts from an allow-list, so GitHub and Hugging
   Face tokens, SSH agent sockets and every other variable the supervisor happens to have are
   absent. HOME points to an empty directory inside the run, so `gh`, git's HTTPS helpers and
   `huggingface_hub` do not look in the operator's home. OpenSSH ignores HOME and reads ~/.ssh
   from the passwd entry, so GIT_SSH_COMMAND gives git's ssh no config and no key; plain ssh is
   refused by the runner. Any process can still read a credential file by absolute path. The
   supervisor's preflight refuses to start while known ones exist, unless the operator accepts
   that. Extra variables the operator passes are refused when the name looks like a credential.
2. The tools (`Tools`): `shell`, which runs every command through orchard/runner.py's checks with
   the run directory as its working directory, and `write_file`, which writes only inside the
   stage directory.
3. The loop (`AgentStep`): it sends the conversation to a tier's OpenAI-compatible endpoint
   (stdlib urllib, non-streaming), runs each tool call, records each new file under the stage's
   `evidence/` directory in the ledger with its sha256, and feeds the watchdog an Event for every
   response, tool call and tool result. A nudge the watchdog's ladder sends is added to the next
   request as a user message. Transport failures are retried once, through the watchdog's
   RetryGuard (spec section 3).

   The environment also sets TT_VISIBLE_DEVICES and TT_METAL_VISIBLE_DEVICES to NO_CHIP, a
   device mask that matches no chip. This is a request to the runtime. Nothing in orchard stops a
   process from opening a device: the same user can open /dev/tenstorrent/* directly, and code
   can clear the variables. Checked on hardware 2026-10-02: a device open with this mask fails (UMD raises RuntimeError: BDF pattern 0000:ff:00.0 did not match any devices).

What it does not do: streaming; a proxy in front of agents the supervisor did not launch (the
watchdog only reads their transcripts); read-only mounts. The same user account runs the agent,
so a file outside the run directory that holds a token can still be read by absolute path, and a
shell redirect can still write outside the run directory. The real limits there are a separate
user account or a mount namespace, which plan 4 does not set up.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from orchard.canary import CanaryError, post_json
from orchard.defaults import (AGENT_MAX_TOKENS, AGENT_MAX_TURNS, AGENT_REQUEST_TIMEOUT_S,
                              TOOL_OUTPUT_CHARS, TOOL_TIMEOUT_S)
from orchard.runner import Denied, check_string
from orchard.stages import evidence_record
from orchard.watchdog import Event, RetryGuard

# ---- environment --------------------------------------------------------------------------------

ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER", "LOGNAME", "TERM")
# The device mask for agent shells: a PCI address that matches no chip on this box, set in both
# TT_VISIBLE_DEVICES and TT_METAL_VISIBLE_DEVICES. Checked on hardware 2026-10-02: a device open
# with this mask fails with a RuntimeError (the BDF matched no device). Code that clears the
# variables still gets every chip. The supervisor's hardware test replaces it with the leased chips.
NO_CHIP = "0000:ff:00.0"
GIT_SSH_COMMAND = "ssh -F /dev/null -o IdentitiesOnly=yes -o IdentityFile=/dev/null -o BatchMode=yes"
DEVICE_VARS = ("TT_VISIBLE_DEVICES", "TT_METAL_VISIBLE_DEVICES")
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|_KEY$|^KEY$|COOKIE", re.IGNORECASE)


def agent_env(run_dir, *, extra: dict | None = None, source=None) -> dict:
    """The whole environment of an agent's shell. Nothing is inherited outside ENV_ALLOW."""
    source = os.environ if source is None else source
    env = {k: source[k] for k in ENV_ALLOW if k in source}
    home = Path(run_dir).resolve() / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "GH_CONFIG_DIR": str(home / ".config" / "gh"),
        "HF_TOKEN_PATH": str(home / "no-hf-token"),      # huggingface_hub reads its token here
        "GIT_CONFIG_GLOBAL": str(home / ".gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        # OpenSSH reads ~/.ssh from the passwd entry and ignores HOME, so git over ssh would find
        # the operator's keys. This gives git's ssh no config file and no key.
        "GIT_SSH_COMMAND": GIT_SSH_COMMAND,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "ORCHARD_RUN_DIR": str(Path(run_dir).resolve()),
    })
    env.update({k: NO_CHIP for k in DEVICE_VARS})
    for k, v in (extra or {}).items():
        if k in DEVICE_VARS:
            raise ValueError(f"{k} is the agent shells' device mask; the operator cannot set it")
        if SECRET_NAME.search(k):
            raise ValueError(f"{k} looks like a credential; agent shells never get one")
        env[k] = str(v)
    return env


# ---- running a checked command ------------------------------------------------------------------

def spawn_checked(command: str, run_dir, env: dict, timeout: float, stdout) -> tuple[int | None, bool]:
    """Run one shell string after the runner's checks. Returns (exit code or None, timed out).

    Denied propagates: a refused string never starts. The command runs in its own session, so a
    timeout kills everything it started. So does anything that interrupts the wait (the
    supervisor's SIGINT or SIGTERM handler, or an error): a terminal's Ctrl-C does not reach a
    process in another session, so without this the command would keep running, possibly with a
    device open, after the supervisor had released its lease.
    """
    check_string(command, run_dir)
    proc = subprocess.Popen(["bash", "-c", command], cwd=run_dir, env=env, stdin=subprocess.DEVNULL,
                            stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        _kill_session(proc)
        return None, True
    except BaseException:
        _kill_session(proc)
        raise


def _kill_session(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters cut ...]\n{text[-half:]}"


# ---- tools --------------------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "shell",
        "description": ("Run one bash command. It starts in the run directory. Heredocs, command "
                        "substitution, subshells, eval and publishing commands are refused, with a "
                        "message that says how to rewrite the command."),
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a text file. The path is relative to your stage directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]


class Tools:
    def __init__(self, run_dir, stage_dir, env: dict, *, timeout: float = TOOL_TIMEOUT_S,
                 limit: int = TOOL_OUTPUT_CHARS):
        self.run_dir, self.stage_dir = Path(run_dir), Path(stage_dir)
        self.env, self.timeout, self.limit = env, timeout, limit

    def call(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments or "{}")
        except ValueError:
            return "error: the arguments are not JSON"
        if not isinstance(args, dict):
            return "error: the arguments must be a JSON object"
        if name == "shell":
            return self.shell(args.get("command"))
        if name == "write_file":
            return self.write_file(args.get("path"), args.get("content"))
        return f"error: there is no tool named {name!r}; the tools are shell and write_file"

    def shell(self, command) -> str:
        if not isinstance(command, str) or not command.strip():
            return "error: command must be a non-empty string"
        out_path = self.stage_dir / "log" / "last-shell-output.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, "wb") as out:
                code, timed_out = spawn_checked(command, self.run_dir, self.env, self.timeout, out)
        except Denied as exc:
            return f"refused: {exc}"
        text = out_path.read_text(encoding="utf-8", errors="replace")
        head = f"killed after {self.timeout} s" if timed_out else f"exit {code}"
        return f"{head}\n{clip(text, self.limit)}"

    def target(self, path) -> Path | None:
        """The real path `path` names inside the stage directory, or None (links resolved)."""
        if not isinstance(path, str) or not path.strip():
            return None
        root = os.path.realpath(self.stage_dir)
        p = os.path.realpath(os.path.join(root, path))
        if p == root or os.path.commonpath([root, p]) != root:
            return None
        return Path(p)

    def write_file(self, path, content) -> str:
        if not isinstance(content, str):
            return "error: content must be a string"
        p = self.target(path)
        if p is None:
            return f"refused: {path!r} is outside your stage directory {self.stage_dir}"
        if p.name.startswith("ledger"):
            return "refused: the ledger is written only by the supervisor"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {os.path.relpath(p, os.path.realpath(self.run_dir))}"


# ---- the model endpoint -------------------------------------------------------------------------

class AgentError(Exception):
    """The model endpoint did not give a usable answer."""


def probe_model(endpoint: str, model: str, timeout: float = 5.0) -> bool:
    """Does the OpenAI-compatible server at `endpoint` (ending in /v1) list `model`?"""
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and any(isinstance(m, dict) and m.get("id") == model
                                          for m in data.get("data") or [])


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Outcome:
    status: str          # done, escalate, pause, operator-pause, abort, turns, error
    turns: int
    final_text: str = ""
    detail: str = ""


class AgentStep:
    """One agent step: a fresh conversation for one stage phase, run until the model stops calling
    tools, the turn limit is reached, or the supervisor says stop.

    `control` is the supervisor's actuator: `take_nudges(agent) -> list[str]` and
    `stop_reason() -> str | None`. `feed` is the watchdog's `feed(Event)`.
    """

    def __init__(self, *, agent: str, endpoint: str, model: str, tools: Tools, ledger, stage: int,
                 phase: str, feed, control, run_dir, evidence_dir, log_path, http=post_json,
                 clock=time.time, guard: RetryGuard | None = None, max_turns: int = AGENT_MAX_TURNS,
                 max_tokens: int = AGENT_MAX_TOKENS, timeout: float = AGENT_REQUEST_TIMEOUT_S):
        self.agent, self.endpoint, self.model, self.tools = agent, endpoint, model, tools
        self.ledger, self.stage, self.phase = ledger, stage, phase
        self.feed, self.control, self.http, self.clock = feed, control, http, clock
        self.run_dir, self.evidence_dir = Path(run_dir), Path(evidence_dir)
        self.log_path = Path(log_path)
        self.guard = guard if guard is not None else RetryGuard()
        self.max_turns, self.max_tokens, self.timeout = max_turns, max_tokens, timeout
        self._seen: dict[str, tuple[int, int]] = {}

    # ---- helpers ----------------------------------------------------------------------------

    def _event(self, kind: str, **fields) -> None:
        self.feed(Event(ts=self.clock(), agent=self.agent, kind=kind, stage=self.stage, **fields))

    def _log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _snapshot(self) -> dict[str, tuple[int, int]]:
        out = {}
        if self.evidence_dir.is_dir():
            for f in sorted(self.evidence_dir.rglob("*")):
                if f.is_file():
                    st = f.stat()
                    out[str(f)] = (st.st_size, st.st_mtime_ns)
        return out

    def _record_new_evidence(self) -> None:
        now = self._snapshot()
        for path, stamp in now.items():
            if self._seen.get(path) != stamp:
                rec = evidence_record(self.run_dir, path)
                self.ledger.append("evidence", self.stage, what="evidence file", phase=self.phase, **rec)
                self._event("evidence", name=rec["path"])
        self._seen = now

    def _send(self, request: dict) -> dict:
        url = self.endpoint.rstrip("/") + "/chat/completions"
        last = None
        # The guard allows the call and one retry of the identical request (spec section 3). The
        # range is only a backstop; the guard is what stops a third send.
        for attempt in range(1, 4):
            if not self.guard.allow(self.agent, request):
                break
            try:
                return self.http(url, request, self.timeout)
            except (OSError, ValueError, CanaryError) as exc:
                last = exc
                self.ledger.append("retry", self.stage, what="model request", phase=self.phase,
                                   attempt=attempt, error=str(exc)[:300])
        raise AgentError(f"the model request to {url} failed and was retried once: {last}")

    def _end(self, status: str, turns: int, final_text: str = "", detail: str = "") -> Outcome:
        if self.log_path.exists():
            self.ledger.append("evidence", self.stage, what="transcript", phase=self.phase,
                               status=status, turns=turns, **evidence_record(self.run_dir, self.log_path))
        return Outcome(status, turns, final_text, detail)

    # ---- the loop ---------------------------------------------------------------------------

    def run(self, system: str, user: str) -> Outcome:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logged = 0
        self._seen = self._snapshot()       # files already here (a resumed stage) are not new
        for turn in range(1, self.max_turns + 1):
            for text in self.control.take_nudges(self.agent):
                messages.append({"role": "user", "content": text})
            reason = self.control.stop_reason()
            if reason:
                return self._end(reason, turn - 1)
            request = {"model": self.model, "messages": messages, "tools": TOOL_SCHEMAS,
                       "temperature": 0, "max_tokens": self.max_tokens, "stream": False}
            try:
                data = self._send(request)
                msg = data["choices"][0]["message"]
                if not isinstance(msg, dict):
                    raise TypeError("message is not an object")
            except AgentError as exc:
                return self._end("error", turn - 1, detail=str(exc))
            except (KeyError, IndexError, TypeError) as exc:
                return self._end("error", turn - 1, detail=f"no choices[0].message: {exc}")
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            details = usage.get("completion_tokens_details") or {}
            assistant = {"role": "assistant", "content": content}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            self._log({"turn": turn, "sent": messages[logged:-1], "received": assistant, "usage": usage})
            logged = len(messages)
            self._event("response", input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        thinking_tokens=details.get("reasoning_tokens") if isinstance(details, dict) else None,
                        text_hash=sha(content) if content else None, had_tool_call=bool(calls))
            if not calls:
                return self._end(self.control.stop_reason() or "done", turn, final_text=content)
            for call in calls:
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name, args = str(fn.get("name", "")), fn.get("arguments") or "{}"
                args = args if isinstance(args, str) else json.dumps(args)
                self._event("tool_call", tool=name, args_hash=sha(args))
                result = self.tools.call(name, args)
                messages.append({"role": "tool", "tool_call_id": str(call.get("id", "")), "content": result})
                self._event("tool_result", tool=name, output_hash=sha(result))
                self._record_new_evidence()
        return self._end("turns", self.max_turns, detail=f"no final answer after {self.max_turns} turns")
