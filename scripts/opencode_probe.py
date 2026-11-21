#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run orchard's kernel probe with OpenCode v2 as the agent loop (an evaluation tool, not part of the harness).

The probe's own plumbing stays in charge: preflights, `prepare_attempt` (the attempt directory), the task text, the unedited
`authoritative_check`, `attempt.json` and the report are the ones `tt-orchard kernel-probe` uses. Only the agent loop changes:
`kernelprobe.AgentStep` is replaced by `OpenCodeStep`, which runs `opencode run` under a Landlock sandbox and gives back an
`orchard.agent.Outcome`.

The sandbox is Landlock (scripts/landlock_exec.py), not bubblewrap: AppArmor refuses unprivileged user namespaces on this host, so
bwrap cannot start. Landlock needs no namespace.

    python3 scripts/opencode_probe.py kernel-probe --op add --attempts 1 ...   (the arguments of `tt-orchard kernel-probe`)
    python3 scripts/opencode_probe.py control kernel-probe ...                  (the orchard agent loop, same checkout, for the control)
    python3 scripts/opencode_probe.py smoke "a prompt"                          (one sandboxed opencode run, events printed)

What the sandbox lets the agent use: the attempt directory (read and write), the system directories and a few /etc files (read
only), the opencode install, the tt-lang simulator environment (read only) and a private HOME and TMPDIR. Opening or listing anything
else, including ~/.ssh, ~/.config/gh, ~/.docker, ~/.cache/huggingface, ~/.config/opencode and /dev/tenstorrent, gives EACCES (Landlock
does not hide names). TCP connects are allowed to the model server's port only, and no TCP port can be bound. The environment holds
only what `kernelprobe.child_env` gives the orchard agent, plus the XDG paths opencode needs.

Differences from the orchard agent that the settings cannot remove are listed in the run log. The ones this file decides:
OpenCode brings its own system prompt, tools (read, write, edit, bash, grep, glob, ...) and context compaction; the probe's skill
text reaches it as an ambient instruction file with a short tool-name mapping in front; OpenCode does not run the check after
each write of kernel.py, so the agent has to run it.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import secrets
import socket
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from orchard import kernelprobe  # noqa: E402
from orchard.agent import Outcome  # noqa: E402

LAUNCHER = Path(__file__).resolve().parent / "landlock_exec.py"
OPENCODE = Path.home() / ".opencode" / "bin" / "opencode"
PROVIDER = "tenstorrent"
# The mapping text put in front of the probe's skill. OpenCode's tools have other names than the orchard agent's, and the skill says
# that each write of kernel.py shows the check's output, which OpenCode does not do.
MAPPING = """\
Tool names in this environment differ from the names in the instructions below:
- `read_file` is the `read` tool (offset and limit count lines); `grep` and `glob` are available too.
- `write_file` is the `write` tool; `edit` changes part of a file.
- `shell` is the `shell` tool.
- Writing kernel.py does not run the check for you here. After each write, run `python3 kernel_check.py` yourself and read its output.
- There is no `hardware_check` tool in this environment.
The instructions below are otherwise the same.

"""
SANDBOX_PATH_TAIL = "/usr/local/bin:/usr/bin:/bin"
ETC_FILES = ("resolv.conf", "ssl", "ca-certificates", "passwd", "group", "hosts", "ld.so.cache", "ld.so.conf", "ld.so.conf.d",
             "localtime", "alternatives", "nsswitch.conf", "ca-certificates.conf")
# Tools for the measured attempts. The pre-registration says the agent has file and shell tools only. Found with a logging proxy on 2026-11-21
# (the request's tool list; see the run log): `tools.question`, `.subagent`, `.execute`, `.webfetch` and `.websearch` set to false remove those tools;
# `tools.skill` does nothing; with `execute` off, eight `opencode_*` tools (sessions, worktrees, MCP resources) appear and a `permission` deny of
# `opencode_*` removes seven of them. `skill` and `opencode_worktree_remove` stay in the list whatever is set, so they are denied by permission:
# the model is offered them and a call is refused. No skills exist in the attempt's project, so `skill` has nothing to load.
TOOLS_OFF = ("webfetch", "websearch", "question", "subagent", "execute")
PERMISSION = {"opencode_*": "deny", "skill": "deny", "opencode_worktree_remove": "deny", "opencode_worktree_create": "deny"}
FORBIDDEN_NAMES = ("tidepool",)          # a transcript that names a model outside the Tenstorrent provider is not a valid attempt


def _resolve(path) -> Path:
    return Path(os.path.realpath(path))


def _ro_binds(paths) -> list:
    out = []
    seen = set()
    for p in paths:
        p = Path(p)
        if str(p) in seen or not p.exists():
            continue
        seen.add(str(p))
        out += ["--ro-bind", str(p), str(p)]
    return out


def sandbox_spec(adir: Path, meta: Path, env: dict, *, sim: str | None, connect: list, argv: list, bind: list = ()) -> dict:
    """The spec file for scripts/landlock_exec.py. `meta` is the host directory that holds opencode's state, as `meta/home`."""
    adir, meta = Path(adir).resolve(), Path(meta).resolve()
    ro = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/proc/{pid}", "/proc/meminfo", "/proc/cpuinfo", "/proc/version",
          "/sys/devices/system/cpu", "/dev/urandom", "/dev/random", "/dev/zero"]
    ro += [str(Path("/etc") / n) for n in ETC_FILES]
    ro.append(str(OPENCODE.parent.parent))
    if sim:
        venv = Path(sim).resolve().parent.parent          # <venv>/bin/tt-lang-sim -> <venv>
        ro += [str(venv), str(_resolve(venv / "bin" / "python3").parent.parent), str(Path.home() / "orchard-src" / "tt-lang")]
    ro.append(str(meta / "skill.md"))
    tmp = meta / "tmp"
    tmp.mkdir(exist_ok=True)
    env = {**env, "TMPDIR": str(tmp)}
    return {"ro": ro, "rw": [str(adir), str(meta / "home"), str(tmp)], "rw_files": ["/dev/null"], "connect_ports": list(connect), "bind_ports": list(bind),
            "env": env, "chdir": str(adir), "argv": argv}


def build_env(adir: Path, meta: Path, sim: str | None) -> dict:
    """What the orchard agent's shell gets (child_env), with OpenCode's own XDG paths inside the private HOME."""
    env = dict(kernelprobe.child_env(meta, adir, hardware=False))
    home = Path(env["HOME"])
    path = [str(OPENCODE.parent)]
    if sim:
        path.append(str(Path(sim).resolve().parent))
    env["PATH"] = ":".join(path + [SANDBOX_PATH_TAIL])
    env.update({"XDG_CONFIG_HOME": str(home / ".config"), "XDG_DATA_HOME": str(home / ".local" / "share"),
                "XDG_STATE_HOME": str(home / ".local" / "state"), "XDG_CACHE_HOME": str(home / ".cache"),
                "OPENCODE_DISABLE_AUTOUPDATE": "1"})
    return env


def write_config(meta: Path, *, endpoint: str, model: str, temperature, top_p, steps: int, max_tokens: int, instructions: str,
                 seed=None, skill_tool: bool = False, skill_paths=None, tools_off=None, permission=None, agent_build_extra=None) -> Path:
    """opencode.json for one attempt: only the Tenstorrent provider is enabled, so there is no fallback to a hosted model."""
    ref = f"{PROVIDER}/{model}"
    # `temperature: true` is a capability flag: without it opencode sends no temperature to a model it does not know, whatever
    # agent.build.temperature says (checked with a logging proxy on 2026-11-21).
    model_entry = {"name": model, "temperature": True, "tool_call": True, "limit": {"context": 262144, "output": max_tokens}}
    # Model `options` go into the request body as they are (the seed arrived that way). agent.build.temperature and a model's
    # temperature flag did not make opencode send a temperature to this provider, so the sampling is set here.
    options = {}
    if seed is not None:
        options["seed"] = seed
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if options:
        model_entry["options"] = options
    build = {"model": ref, "steps": steps}
    if temperature is not None:
        build["temperature"] = temperature
    if top_p is not None:
        build["top_p"] = top_p
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "model": ref, "small_model": ref,
        "enabled_providers": [PROVIDER],
        "provider": {PROVIDER: {"npm": "@ai-sdk/openai-compatible", "name": "Tenstorrent",
                                "options": {"baseURL": endpoint, "apiKey": "tt-local"}, "models": {model: model_entry}}},
        "agent": {"build": build},
        "instructions": [instructions],
        "share": "disabled", "update": "disable",
        # The orchard agent has shell, read_file and write_file. OpenCode adds grep, glob and edit (kept: they are its own toolset) and also
        # `question`, `subagent`, `execute` and `skill`, which are switched off so an attempt is one agent with file and shell tools.
        "tools": {**{name: False for name in (tools_off if tools_off is not None else TOOLS_OFF)}, **({"skill": True} if skill_tool else {})},
    }
    perm = PERMISSION if permission is None else permission
    if perm and not skill_tool:
        cfg["permission"] = perm
    if agent_build_extra:
        cfg["agent"]["build"].update(agent_build_extra)
    if skill_paths:
        cfg["skills"] = list(skill_paths)
    path = meta / "home" / ".config" / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def read_events(path: Path) -> list:
    out = []
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
    except OSError:
        pass
    return out


def summarize(events: list) -> dict:
    """Counts from the event stream: event types, tool calls by tool, steps, tokens when reported, models named."""
    types, tools, models = {}, {}, set()
    steps, tokens = 0, {}
    for e in events:
        t = str(e.get("type"))
        types[t] = types.get(t, 0) + 1
        part = e.get("part") if isinstance(e.get("part"), dict) else {}
        if t == "tool_use" or part.get("type") == "tool":
            name = str(part.get("tool") or e.get("tool") or "?")
            tools[name] = tools.get(name, 0) + 1
        if t == "step_start":
            steps += 1
        if t == "step_finish":
            for k, v in (part.get("tokens") or {}).items():
                if isinstance(v, (int, float)):
                    tokens[k] = tokens.get(k, 0) + v
        blob = json.dumps(e)
        for name in FORBIDDEN_NAMES:
            if name in blob:
                models.add(name)
    return {"event_types": types, "tool_calls": tools, "steps": steps, "tokens": tokens, "foreign_models": sorted(models)}


def run_opencode(adir: Path, meta: Path, *, system: str, user: str, endpoint: str, model: str, temperature, top_p, seed,
                 steps: int, max_tokens: int, wall_deadline_s: float, control=None, title: str = "attempt",
                 claude_config: bool = False, mapping: bool = True, skill_tool: bool = False, skill_paths=None, tools_off=None,
                 permission=None, agent_build_extra=None) -> dict:
    """One sandboxed `opencode run`. Returns a dict: exit code, wall seconds, why it stopped, paths of the raw output."""
    adir, meta = Path(adir).resolve(), Path(meta).resolve()
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "home").mkdir(exist_ok=True)
    skill = meta / "skill.md"
    skill.write_text((MAPPING if mapping else "") + system, encoding="utf-8")
    check = adir / "check_config.json"
    sim = None
    if check.is_file():
        try:
            sim = json.loads(check.read_text()).get("sim")
        except ValueError:
            sim = None
    env = build_env(adir, meta, sim)
    # The server needs a password; the client reads the same variable. OPENCODE_DISABLE_CLAUDE_CODE keeps opencode from loading Claude Code
    # skills and CLAUDE.md files, so only the probe's instruction file counts (the skill-discovery test turns that on). Model-list
    # fetching and the file watcher are switched off: the sandbox has no network beyond the model server.
    env.update({"OPENCODE_PASSWORD": secrets.token_urlsafe(24), "OPENCODE_DISABLE_MODELS_FETCH": "1",
                "OPENCODE_DISABLE_FILEWATCHER": "1"})
    if not claude_config:
        env["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
    write_config(meta, endpoint=endpoint, model=model, temperature=temperature, top_p=top_p, steps=steps, max_tokens=max_tokens,
                 instructions=str(skill), seed=seed, skill_tool=skill_tool, skill_paths=skill_paths, tools_off=tools_off, permission=permission, agent_build_extra=agent_build_extra)
    model_port = int(urlsplit(endpoint).port or 80)
    # The private server `run --standalone` starts needs a TCP bind, which the sandbox denies, so the server is started here as the
    # first sandboxed process (it may bind one port) and the client is a second one (it may connect to that port). The commands the
    # model runs are children of the server, so they inherit its rules: connect to the model port only, bind nothing else.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        server_port = probe.getsockname()[1]
    server_spec = sandbox_spec(adir, meta, env, sim=sim, connect=[model_port], bind=[server_port],
                               argv=[str(OPENCODE), "serve", "--hostname", "127.0.0.1", "--port", str(server_port)])
    client_spec = sandbox_spec(adir, meta, env, sim=sim, connect=[server_port], bind=[],
                               argv=[str(OPENCODE), "run", "--server", f"http://127.0.0.1:{server_port}", "--format", "json", "--auto",
                                     "-m", f"{PROVIDER}/{model}", "--title", title, user])
    paths = {}
    for name, spec in (("server", server_spec), ("client", client_spec)):
        paths[name] = meta / f"sandbox-{name}.json"
        paths[name].write_text(json.dumps(spec, indent=1), encoding="utf-8")
    events, errors = meta / "events.jsonl", meta / "stderr.txt"
    t0 = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    why = "finished"
    host_env = {"PATH": "/usr/bin:/bin"}
    with open(meta / "server.log", "wb") as slog, open(events, "wb") as out, open(errors, "wb") as err:
        server = subprocess.Popen([sys.executable, str(LAUNCHER), str(paths["server"])], stdout=slog, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True, env=host_env)
        (meta / "server.pid").write_text(str(server.pid))
        proc = None
        try:
            up = False
            for _ in range(120):                       # up to 60 s for the server to listen
                if server.poll() is not None:
                    break
                try:
                    socket.create_connection(("127.0.0.1", server_port), timeout=0.5).close()
                    up = True
                    break
                except OSError:
                    time.sleep(0.5)
            if not up:
                why = "server-failed"
                return {"returncode": -1, "wall_s": round(time.monotonic() - t0, 2), "why": why, "events": str(events),
                        "stderr": str(errors), "started_utc": started_utc, "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            proc = subprocess.Popen([sys.executable, str(LAUNCHER), str(paths["client"])], stdout=out, stderr=err,
                                    stdin=subprocess.DEVNULL, start_new_session=True, env=host_env)
            (meta / "pid").write_text(str(proc.pid))
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    pass
                stop = control.stop_reason() if control is not None else None
                if stop or time.monotonic() - t0 > wall_deadline_s:
                    why = stop or "wall-clock"
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    break
        finally:
            for victim in (proc, server):             # each one started its own session, so its pid is its group id
                if victim is not None and victim.poll() is None:
                    try:
                        os.killpg(victim.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    victim.wait()
    return {"returncode": proc.returncode, "wall_s": round(time.monotonic() - t0, 2), "why": why,
            "events": str(events), "stderr": str(errors), "started_utc": started_utc, "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


class OpenCodeStep:
    """The stand-in for orchard.agent.AgentStep. Takes the same keyword arguments and has the same `run(system, user)`."""

    def __init__(self, *, endpoint, model, run_dir, max_turns, temperature=0.0, top_p=None, seed=None, max_tokens=4096, control=None,
                 **ignored):
        self.endpoint, self.model, self.adir = endpoint, model, Path(run_dir)
        self.max_turns, self.temperature, self.top_p, self.seed, self.max_tokens = max_turns, temperature, top_p, seed, max_tokens
        self.control = control

    def run(self, system: str, user: str) -> Outcome:
        probe_dir = self.adir.parent.parent
        meta = probe_dir / "oc-meta" / f"{self.adir.parent.name}-{self.adir.name}"
        shutil.rmtree(meta, ignore_errors=True)
        wall = max(1.0, (self.control.deadline - self.control.clock()) if self.control is not None else 1800.0)
        r = run_opencode(self.adir, meta, system=system, user=user, endpoint=self.endpoint, model=self.model,
                         temperature=self.temperature if self.temperature else None, top_p=self.top_p, seed=self.seed,
                         steps=self.max_turns, max_tokens=self.max_tokens, wall_deadline_s=wall, control=self.control,
                         title=f"{self.adir.parent.name}-{self.adir.name}")
        s = summarize(read_events(r["events"]))
        (meta / "summary.json").write_text(json.dumps({**r, **s}, indent=2), encoding="utf-8")
        detail = (f"opencode exit {r['returncode']}, {s['steps']} steps, tools {s['tool_calls']}, tokens {s['tokens']}"
                  + (f", FOREIGN MODEL NAMED {s['foreign_models']}" if s["foreign_models"] else ""))
        if r["why"] != "finished":
            self.control.detail = self.control.detail or "the attempt's wall-clock budget is spent"
            return Outcome(status=r["why"], turns=s["steps"], detail=detail)
        if s["foreign_models"]:
            return Outcome(status="error", turns=s["steps"], detail=detail)
        if r["returncode"] != 0:
            return Outcome(status="error", turns=s["steps"], detail=detail)
        # A run that used up the step limit ends with a final text like any other, so compare the count.
        status = "turns" if s["steps"] >= self.max_turns else "done"
        return Outcome(status=status, turns=s["steps"], detail=detail)


def smoke(prompt: str) -> int:
    """One sandboxed opencode run in a scratch attempt-like directory, with the events printed."""
    base = Path.home() / "orchard-runs" / "ocprobe" / "smoke"
    shutil.rmtree(base, ignore_errors=True)
    adir = base / "adir"
    adir.mkdir(parents=True)
    r = run_opencode(adir, base / "meta", system="You are a careful assistant.", user=prompt, endpoint="http://127.0.0.1:8011/v1",
                     model="Qwen/Qwen3-Coder-Next", temperature=0.7, top_p=None, seed=12345, steps=6, max_tokens=4096, wall_deadline_s=300)
    print(json.dumps(r, indent=1))
    ev = read_events(r["events"])
    print(json.dumps(summarize(ev), indent=1))
    return 0


def main(argv: list) -> int:
    if argv and argv[0] == "smoke":
        return smoke(" ".join(argv[1:]) or "Reply with exactly: ready")
    if argv and argv[0] == "control":
        # The unpatched orchard agent loop from this same checkout, so both harnesses run the same probe code.
        from orchard import cli
        return cli.main(argv[1:])
    kernelprobe.AgentStep = OpenCodeStep
    from orchard import cli
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
