# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The release watcher: start the bring-up of a new Qwen release when it appears, with nobody at the keyboard.

    tt-orchard drop-watch init --watch-dir DIR --target Qwen/Qwen4-27B --run-dir RUN [--capabilities FILE] ...
    tt-orchard drop-watch tick --watch-dir DIR [--json] [--hub-dir RELEASE]
    tt-orchard drop-watch show --watch-dir DIR

A scheduler (a scheduled Claude task that also sends the notifications, or a timer) runs `tick` every 30 minutes. One tick
does at most one step of this sequence and records it in DIR/state.json:

  waiting            the target repository does not exist (or is private or gated)
  seen               it exists, but the upload is not finished: config.json, the weight index or a shard it lists is missing,
                     or the last commit is younger than quiet_minutes
  waiting-for-chips  the release-day check (orchard/drop_check.py) ran; its route is fixed; gozer shows a chip held
  running            every chip was free, so `tt-orchard bringup <target> --run-dir RUN ...` was started once
  ready / blocked / failed   the run ended; nothing more happens
  needs-operator     the check needs a reference interpreter, or the state cannot be trusted (a corrupt state file, or a tick
                     that died after "starting" and before it recorded the process): nothing is started until the operator
                     acts (fix the cause, then edit or remove state.json)

The route follows the check: covered starts `bringup --base` the Qwen4 package (or the Qwen3.8 bundle for a release with
Qwen3.8's architecture); not covered starts the full-port fallback. Each step appends events to DIR/events.jsonl; `tick --json`
prints the events of this tick for the notifier.

Never: publishes or pushes anything, sends a Hugging Face token (the check and bringup both strip it), stops a process that holds
a chip, starts a second run, or watches an organisation outside drop_check.ALLOWED_ORGS. A lock file keeps two ticks apart.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from orchard import drop_check as dc
from orchard import safeio

STATES = ("waiting", "seen", "check", "needs-operator", "waiting-for-chips", "starting", "running", "ready", "blocked",
          "failed")
TERMINAL = ("ready", "blocked", "failed")
CONFIG_NAME, STATE_NAME, EVENTS_NAME, LOCK_NAME = "watch.toml", "state.json", "events.jsonl", ".lock"
MAX_STATE_BYTES = 4 << 20
MAX_LISTING_BYTES = 4 << 20       # the Hub's model search
MAX_GOZER_BYTES = 4 << 20         # `gozer status --json`
CONFIG_KEYS = ("target", "run_dir", "capabilities", "bundle", "qwen38_bundle", "quiet_minutes", "bringup_args", "command",
               "reference_python", "bundle_dir", "hf_home")


@dataclasses.dataclass
class Config:
    target: str
    run_dir: str
    capabilities: str | None = None
    bundle: str = dc.QWEN4_BUNDLE
    qwen38_bundle: str = dc.QWEN38_BUNDLE
    quiet_minutes: int = 30
    bringup_args: list = dataclasses.field(default_factory=list)
    command: list = dataclasses.field(default_factory=lambda: ["tt-orchard"])
    reference_python: str | None = None
    bundle_dir: str | None = None       # the installed Qwen4 package: its capabilities file and its own weights config
    hf_home: str | None = None          # the Hugging Face cache that holds those weights (default $HF_HOME or ~/.cache/huggingface)


def _validate(c: Config) -> Config:
    p = dc.repo_problem(c.target)
    if p:
        raise ValueError(f"target: {p}")
    for name in ("run_dir",):
        v = getattr(c, name)
        if not isinstance(v, str) or not v or "\0" in v:
            raise ValueError(f"{name} must be a non-empty path")
    for name in ("capabilities", "reference_python", "bundle_dir", "hf_home"):
        v = getattr(c, name)
        if v is not None and (not isinstance(v, str) or not v or "\0" in v):
            raise ValueError(f"{name} must be a path")
    for name in ("bundle", "qwen38_bundle"):
        v = getattr(c, name)
        if not isinstance(v, str) or not dc.REPO_RE.fullmatch(v):
            raise ValueError(f"{name} must be org/name")
    if isinstance(c.quiet_minutes, bool) or not isinstance(c.quiet_minutes, int) or not 0 <= c.quiet_minutes <= 10080:
        raise ValueError("quiet_minutes must be a whole number from 0 to 10080")
    for name in ("bringup_args", "command"):
        v = getattr(c, name)
        if not isinstance(v, list) or not all(isinstance(x, str) and "\0" not in x for x in v):
            raise ValueError(f"{name} must be a list of strings")
    if not c.command:
        raise ValueError("command must name the tt-orchard command")
    return c


def _toml_value(v) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, int):
        return str(v)
    return json.dumps(v)          # a JSON string is a valid TOML basic string for the characters allowed here


def write_config(watch_dir, c: Config) -> None:
    from orchard import safeio
    _validate(c)
    d = Path(watch_dir)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    lines = [f"{k} = {_toml_value(v)}" for k, v in dataclasses.asdict(c).items() if v is not None]
    safeio.atomic_write(d / CONFIG_NAME, "\n".join(lines) + "\n")


def load_config(watch_dir) -> Config:
    import tomllib

    from orchard import safeio
    try:
        raw = safeio.read_bytes(Path(watch_dir) / CONFIG_NAME, max_bytes=1 << 16)
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, safeio.SafeReadError) as exc:
        raise ValueError(f"{CONFIG_NAME} cannot be read: {type(exc).__name__}") from exc
    unknown = sorted(set(data) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"{CONFIG_NAME} has unknown keys: {', '.join(unknown)}")
    if "target" not in data or "run_dir" not in data:
        raise ValueError(f"{CONFIG_NAME} needs target and run_dir")
    try:
        return _validate(Config(**data))
    except TypeError as exc:
        raise ValueError(f"{CONFIG_NAME}: {exc}") from exc


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_iso(s) -> float | None:
    if not isinstance(s, str) or len(s) > 40:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.timestamp() if dt.tzinfo else None


@contextlib.contextmanager
def locked(watch_dir):
    """Yield True while this process holds DIR/.lock, False when another tick holds it."""
    d = Path(watch_dir)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(d / LOCK_NAME, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class StateCorrupt(RuntimeError):
    pass


def load_state(watch_dir) -> dict:
    from orchard import safeio
    path = Path(watch_dir) / STATE_NAME
    if not os.path.lexists(path):
        return {"state": "waiting"}
    try:
        st = safeio.read_json(path, max_bytes=MAX_STATE_BYTES)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        raise StateCorrupt(f"{STATE_NAME} cannot be read ({type(exc).__name__})") from exc
    if not isinstance(st, dict) or st.get("state") not in STATES:
        raise StateCorrupt(f"{STATE_NAME} has no known state")
    problem = _state_problem(st)
    if problem:
        raise StateCorrupt(f"{STATE_NAME} is in state {st['state']} but {problem}")
    return st


def _text_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _state_problem(st: dict) -> str | None:
    """What a state's tick needs that the file lacks (the tick would end in a KeyError or launch a malformed command), or None."""
    state = st["state"]
    if (state == "check" or (state == "needs-operator" and st.get("needs") == "check")) and dc.revision_problem(st.get("revision")):
        return "it has no 40-character revision"
    if state == "waiting-for-chips":
        cmd = st.get("command")
        if not (_text_list(cmd) and cmd[:2] == ["tt-orchard", "bringup"]):
            return "its command is not a tt-orchard bringup command"
    if state in ("starting", "running"):
        run = st.get("run")
        if not (isinstance(run, dict) and isinstance(run.get("run_dir"), str)
                and (state == "starting" or isinstance(run.get("log"), str))):
            return "its run record has no run directory" + ("" if state == "starting" else " and log")
    for key in ("seen_other", "stages_reported"):
        if key in st and not _text_list(st[key]):
            return f"{key} is not a list of names"
    return None


def save_state(watch_dir, st: dict) -> None:
    from orchard import safeio
    safeio.atomic_write(Path(watch_dir) / STATE_NAME, json.dumps(st, indent=1, sort_keys=True) + "\n")


def _append_events(watch_dir, events: list[dict]) -> None:
    if not events:
        return
    fd = os.open(Path(watch_dir) / EVENTS_NAME, os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, "".join(json.dumps(e, sort_keys=True) + "\n" for e in events).encode())
    finally:
        os.close(fd)


def upload_complete(info: dict, hub, repo: str, sha: str, now: float, quiet_s: float) -> tuple[bool, str]:
    """(True, "") when config.json, the weight index and every shard it lists are in the repository with a size, and the last
    commit is at least `quiet_s` old; else (False, why)."""
    sib = info.get("siblings")
    sizes = {s.get("rfilename"): s.get("size") for s in sib if isinstance(s, dict)} if isinstance(sib, list) else {}
    if "config.json" not in sizes:
        return False, "config.json is not uploaded yet"
    if "model.safetensors.index.json" in sizes:
        try:
            raw = hub.get(repo, sha, "model.safetensors.index.json", max_bytes=dc.METADATA_FILES["model.safetensors.index.json"])
        except dc.Unreadable as exc:
            return False, str(exc)
        shards = dc.index_shards(raw)
        if not shards:
            return False, "the weight index cannot be read"
        missing = sorted(s for s in shards if not (isinstance(sizes.get(s), int) and sizes[s] > 0))
        if missing:
            return False, f"{len(missing)} of {len(shards)} shards are not uploaded yet"
    elif not (isinstance(sizes.get("model.safetensors"), int) and sizes["model.safetensors"] > 0):
        return False, "no weight index and no model.safetensors yet"
    modified = parse_iso(info.get("lastModified"))
    if modified is None:
        return False, "the last commit time is not readable"
    if now - modified < quiet_s:
        return False, f"the last commit is {int(now - modified)} s old; waiting for {int(quiet_s)} s without a commit"
    return True, ""


def _summary(res: dict) -> str:
    parts = [f"{res.get('verdict')}, route {res.get('route')}"]
    parts += [f"reason: {r}" for r in res.get("reasons", [])[:6]]
    parts += [f"note: {n}" for n in res.get("notes", [])[:4]]
    return "; ".join(parts)


def tick(watch_dir, *, hub, chips, launch, run_status, alive, now=time.time, class_probe=None) -> dict:
    """One step of the watcher. Returns {"state", "events"} (or {"skipped": why} when another tick holds the lock)."""
    watch_dir = Path(watch_dir)
    with locked(watch_dir) as got:
        if not got:
            return {"state": None, "events": [], "skipped": "another tick holds the lock"}
        cfg = load_config(watch_dir)
        t = now()
        events: list[dict] = []

        def ev(kind, text):
            events.append({"kind": kind, "text": text, "ts": iso(t), "target": cfg.target})

        def hub_down(exc):
            """An outage is reported once, when it starts; the state does not change and the next tick tries again."""
            if not st.get("hub_unreachable"):
                ev("hub-unreachable", f"{exc}; the watcher tries again next tick")
                st["hub_unreachable"] = iso(t)
        try:
            st = load_state(watch_dir)
        except StateCorrupt as exc:
            path = watch_dir / STATE_NAME
            os.replace(path, watch_dir / f"{STATE_NAME}.corrupt-{int(t)}")
            st = {"state": "needs-operator", "needs": "state"}
            ev("needs-operator", f"{exc}; it was moved aside and nothing is started until you check the watcher")
        state = st["state"]
        org, name = cfg.target.split("/", 1)

        if state in ("waiting", "seen"):
            try:
                seen_other = st.setdefault("seen_other", [])
                for other in hub.list_models(org, name):
                    if other != cfg.target and other not in seen_other and not dc.repo_problem(other):
                        seen_other.append(other)
                        ev("other-release", other)
                try:
                    info, unreadable = hub.info(cfg.target), None
                except dc.Unreadable as exc:
                    info, unreadable = None, str(exc)
                if unreadable is not None:
                    ev("needs-operator", f"the Hub record cannot be read: {unreadable}")
                    state, st["needs"] = "needs-operator", "hub"
                elif info is None:
                    if state == "seen":
                        ev("release-gone", f"{cfg.target} is no longer visible")
                    state = "waiting"
                else:
                    sha = info.get("sha")
                    if dc.revision_problem(sha):
                        ev("needs-operator", "the Hub record has no valid commit hash")
                        state, st["needs"] = "needs-operator", "hub"
                    else:
                        st["revision"] = sha
                        if state == "waiting":
                            ev("release-seen", f"{cfg.target} at {sha[:12]}")
                            state = "seen"
                        ok, why = upload_complete(info, hub, cfg.target, sha, t, cfg.quiet_minutes * 60.0)
                        st["upload"] = "complete" if ok else why
                        if ok:
                            state = "check"
            except dc.HubUnreachable as exc:
                hub_down(exc)
            else:
                st.pop("hub_unreachable", None)

        if state == "check" or (state == "needs-operator" and st.get("needs") == "check"):
            rev = st["revision"]
            caps, dev, extra_notes = None, None, []
            if cfg.bundle_dir:
                caps, dev, extra_notes = dc.bundle_inputs(cfg.bundle_dir, cfg.hf_home or dc.default_hf_home())
            if cfg.capabilities:
                from orchard import safeio
                try:
                    caps = safeio.read_json(cfg.capabilities, max_bytes=1 << 20)
                except (OSError, ValueError, safeio.SafeReadError) as exc:
                    caps = f"unreadable ({type(exc).__name__})"
            try:
                files, problems = dc.fetch_metadata(hub, cfg.target, rev)
            except dc.HubUnreachable as exc:
                hub_down(exc)                       # nothing is decided from part of the files: the check runs again next tick
                files = None
            if files is not None:
                st.pop("hub_unreachable", None)
                probe = class_probe if class_probe is not None else dc.class_probe_for(cfg.reference_python)
                res = dc.classify(cfg.target, rev, files, capabilities=caps, dev_config=dev, class_probe=probe,
                                  qwen4_bundle=cfg.bundle, qwen38_bundle=cfg.qwen38_bundle)
                res["notes"] = extra_notes + res["notes"]
                if problems:
                    res["reasons"] = problems + res["reasons"]
                    if res["verdict"] in ("covered", "covered-with-notes"):
                        res["verdict"], res["route"] = "not-covered", "full-port"
                        res["command"] = dc._command("full-port", cfg.target, cfg.bundle, cfg.qwen38_bundle)
                previous = st.get("check")
                st["check"] = res
                if res["verdict"] in ("covered", "covered-with-notes", "not-covered"):
                    ev("check", _summary(res))
                    st["route"], st["command"] = res["route"], res["command"]
                    st.pop("needs", None)
                    state = "waiting-for-chips"
                else:
                    if not previous or previous.get("reasons") != res.get("reasons") or state != "needs-operator":
                        ev("needs-operator", _summary(res))
                    state, st["needs"] = "needs-operator", "check"

        if state == "waiting-for-chips":
            c = chips()
            if c.get("free") is True:
                argv = list(cfg.command) + list(st["command"][1:]) + ["--run-dir", cfg.run_dir] + list(cfg.bringup_args)
                log = str(watch_dir / f"bringup-{int(t)}.log")
                st["state"], st["run"] = "starting", {"argv": argv, "run_dir": cfg.run_dir, "log": log}
                save_state(watch_dir, st)
                started = launch(argv, log)                 # if this raises, the next tick finds "starting" and stops
                st["run"].update(pid=started.get("pid"), start_time=started.get("start_time"), started=iso(t))
                state = "running"
                ev("run-started", " ".join(argv))
            elif st.get("waiting_holders") != c.get("holders"):
                st["waiting_holders"] = c.get("holders")
                ev("waiting-for-chips", "chips held by " + (", ".join(map(str, c.get("holders") or [])) or "unknown")
                   + "; the run starts when every chip is free")
        elif state == "starting":
            st["needs"] = "start-unknown"
            state = "needs-operator"
            ev("needs-operator", f"a run may have started without its process being recorded; check {st['run']['run_dir']}"
               " and the process list, then set state.json")

        if state == "running":
            s = run_status(st["run"]["run_dir"]) or {}
            reported = st.setdefault("stages_reported", [])
            for row in s.get("stages") or []:
                if not isinstance(row, dict) or row.get("status") in (None, "running"):
                    continue
                key = f"{row.get('stage')}:{row.get('status')}:{row.get('attempts')}"
                if key not in reported:
                    reported.append(key)
                    label = f"stage {row.get('stage')}" + (f" ({row['name']})" if row.get("name") else "")
                    ev("stage", f"{label}: {row.get('status')}")
            ss = s.get("state")
            if ss == "ready-for-operator-review":
                state = "ready"
                ev("run-ready", f"{cfg.target}: ready for operator review in {st['run']['run_dir']}")
            elif ss == "blocked":
                b = s.get("blocked") or {}
                state = "blocked"
                ev("run-blocked", f"{b.get('code', 'blocked')}: {b.get('reason', '')}".strip())
            elif ss in ("aborted", "stopped-or-crashed"):
                state = "failed"
                ev("run-failed", f"the run is {ss}")
            elif ss == "paused" and not st.get("paused_reported"):
                st["paused_reported"] = True
                ev("run-paused", "the run is paused")
            elif ss not in ("running", "paused") and not alive(st["run"].get("pid"), st["run"].get("start_time")):
                state = "failed"
                ev("run-failed", f"bringup exited before the run finished (status {ss}); see {st['run']['log']}")

        st["state"], st["updated"] = state, iso(t)
        save_state(watch_dir, st)
        _append_events(watch_dir, events)
        return {"state": state, "events": events}


# ---- the real world ---------------------------------------------------------------------------------------------------

class Hub(dc.HubClient):
    """The Hub client with a model search, for other releases of the same name."""

    def list_models(self, org: str, search: str) -> list[str]:
        if dc.repo_problem(f"{org}/{search}"):
            return []
        try:
            raw = self._open(f"{self.base}/api/models?author={org}&search={search}&limit=50", MAX_LISTING_BYTES)
            data = safeio.parse_json_bytes(raw, max_bytes=MAX_LISTING_BYTES, expect=list) if raw else []
        except (OSError, ValueError, dc.Unreadable, dc.HubUnreachable, safeio.SafeReadError):
            return []
        return [m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]


class SimHub(dc.DirClient):
    """A release laid out in a directory (info.json plus files), for rehearsals."""

    def list_models(self, org: str, search: str) -> list[str]:
        return []


def gozer_chips(run=subprocess.run) -> dict:
    """{"free": every chip FREE, "holders": [...]} from `gozer status --json`. A failure counts as not free."""
    try:
        done = run(["gozer", "status", "--json"], capture_output=True, text=True, timeout=120)
        chips = safeio.parse_json_bytes(done.stdout, max_bytes=MAX_GOZER_BYTES, expect=dict).get("chips")
    except (OSError, ValueError, subprocess.SubprocessError, safeio.SafeReadError) as exc:
        return {"free": False, "holders": [f"gozer status failed ({type(exc).__name__})"]}
    if not isinstance(chips, list) or not chips:
        return {"free": False, "holders": ["gozer status listed no chips"]}
    held = sorted({str(c.get("who") or c.get("state")) for c in chips if isinstance(c, dict) and c.get("state") != "FREE"})
    return {"free": not held, "holders": held}


def _start_time(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()   # hardening: ok a kernel record of an integer pid; /proc files report size 0, so safeio's size checks do not apply
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def launch_process(argv: list[str], log_path: str) -> dict:
    fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fd, stderr=subprocess.STDOUT, start_new_session=True,
                                close_fds=True)
    finally:
        os.close(fd)
    return {"pid": proc.pid, "start_time": _start_time(proc.pid)}


def process_alive(pid, start_time) -> bool:
    return isinstance(pid, int) and pid > 0 and start_time is not None and _start_time(pid) == start_time


def run_status_of(run_dir) -> dict:
    from orchard import status
    d = Path(run_dir)
    if not d.is_dir():
        return {"state": "not-started"}
    try:
        return status.collect(d)
    except Exception as exc:  # noqa: BLE001  (a ledger mid-write or corrupt: report and try again next tick)
        return {"state": "unknown", "error": type(exc).__name__}


# ---- command line ---------------------------------------------------------------------------------------------------

def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard drop-watch", add_help=add_help,
                                description="watch for a release and start its bring-up when the chips are free")
    sub = p.add_subparsers(dest="action", required=True)
    i = sub.add_parser("init", help="write DIR/watch.toml")
    i.add_argument("--watch-dir", required=True)
    i.add_argument("--target", required=True, help="the release to watch, org/name")
    i.add_argument("--run-dir", required=True, help="the run directory bringup uses")
    i.add_argument("--capabilities", help="the Qwen4 package's orchard-capabilities.json (none: every Qwen4 release takes "
                                          "the full-port fallback)")
    i.add_argument("--bringup-arg", action="append", default=[], help="an extra bringup argument (repeatable)")
    i.add_argument("--quiet-minutes", type=int, default=30, help="how long the last commit must be quiet (default 30)")
    i.add_argument("--reference-python", help="the reference interpreter drop-check asks about a new architecture class")
    i.add_argument("--bundle-dir", help="the installed Qwen4 package's directory (its capabilities file and its weights config)")
    i.add_argument("--hf-home", help="the Hugging Face cache that holds the package's weights")
    t = sub.add_parser("tick", help="do one step and record it")
    t.add_argument("--watch-dir", required=True)
    t.add_argument("--json", action="store_true", help="print this tick's state and events as JSON")
    t.add_argument("--hub-dir", help="read the release from a directory (info.json plus files), for tests and rehearsals")
    s = sub.add_parser("show", help="print the recorded state")
    s.add_argument("--watch-dir", required=True)
    return p


def run_args(args, out=None) -> int:
    out = out or sys.stdout
    if args.action == "init":
        write_config(args.watch_dir, Config(target=args.target, run_dir=args.run_dir, capabilities=args.capabilities,
                                            bringup_args=list(args.bringup_arg), quiet_minutes=args.quiet_minutes,
                                            reference_python=args.reference_python, bundle_dir=args.bundle_dir,
                                            hf_home=args.hf_home))
        print(f"wrote {Path(args.watch_dir) / CONFIG_NAME}", file=out)
        return 0
    if args.action == "show":
        try:
            st = load_state(args.watch_dir)
        except StateCorrupt as exc:
            print(f"state: unreadable ({exc})", file=out)
            return 2
        print(json.dumps(st, indent=1, sort_keys=True), file=out)
        return 0
    hub = SimHub(args.hub_dir) if args.hub_dir else Hub()
    res = tick(args.watch_dir, hub=hub, chips=gozer_chips, launch=launch_process, run_status=run_status_of,
               alive=process_alive)
    if args.json:
        print(json.dumps(res, indent=1), file=out)
    else:
        print(f"state: {res['state']}" + (f" (skipped: {res['skipped']})" if res.get("skipped") else ""), file=out)
        for e in res["events"]:
            print(f"  {e['kind']}: {e['text']}", file=out)
    return 0


def main(argv=None) -> int:
    return run_args(parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
