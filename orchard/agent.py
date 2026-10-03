"""Agent steps: one fresh model context per stage phase, and the tool-call loop that serves it.

This module owns three things.

1. The agent's environment (`agent_env`). It starts from an allow-list, so GitHub and Hugging
   Face tokens, SSH agent sockets and every other variable the supervisor happens to have are
   absent. HOME points to an empty directory inside the run, so `gh`, `git` and `huggingface_hub`
   find no stored credentials under the usual paths. Extra variables the operator passes are
   refused when the name looks like a credential.
2. The tools (`Tools`): `shell`, which runs every command through orchard/runner.py's checks with
   the run directory as its working directory, and `write_file`, which writes only inside the
   stage directory.
3. The loop (`AgentStep`): it sends the conversation to a tier's OpenAI-compatible endpoint
   (stdlib urllib, non-streaming), runs each tool call, records each new file under the stage's
   `evidence/` directory in the ledger with its sha256, and feeds the watchdog an Event for every
   response, tool call and tool result. A nudge the watchdog's ladder sends is added to the next
   request as a user message. Transport failures are retried once, through the watchdog's
   RetryGuard (spec section 3).

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
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "ORCHARD_RUN_DIR": str(Path(run_dir).resolve()),
    })
    for k, v in (extra or {}).items():
        if SECRET_NAME.search(k):
            raise ValueError(f"{k} looks like a credential; agent shells never get one")
        env[k] = str(v)
    return env


# ---- running a checked command ------------------------------------------------------------------

def spawn_checked(command: str, run_dir, env: dict, timeout: float, stdout) -> tuple[int | None, bool]:
    """Run one shell string after the runner's checks. Returns (exit code or None, timed out).

    Denied propagates: a refused string never starts. The command runs in its own session, so a
    timeout kills everything it started.
    """
    check_string(command, run_dir)
    proc = subprocess.Popen(["bash", "-c", command], cwd=run_dir, env=env, stdin=subprocess.DEVNULL,
                            stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return None, True


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
