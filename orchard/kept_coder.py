# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A coder kept up between lab runs: the record one run leaves and the next one adopts.

Booting the coder took about 6.5 minutes of a 31-minute lab run. With `[coder] keep_up = true` (lab
mode only), a run that ends ready or blocked with a healthy coder hands the coder's chips to gozer
as an adopted lease (judged by the container's own process, so it outlives the supervisor) and
writes `<cache_root>/coder/kept.json`. The next run adopts that coder only when every check holds:
the same coder (target, kind, port, profile, image, model and chips), a server that lists the model
on the port, gozer showing the chips HELD by the kept coder's processes, and a canary answer equal
to the one the coder gave when it was first started. Otherwise the run records why and boots a coder
as before; it never stops a server it did not record.

`tt-orchard coder status` shows the kept coder and `tt-orchard coder stop` stops it and frees its
chips.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

KEPT_WHO = "orchard:coder-kept"          # the adopted lease's owner name in `gozer status`
IDENTITY_KEYS = ("target", "kind", "port", "profile", "image_id", "model", "chips")


def path(cache_root) -> Path:
    return Path(cache_root) / "coder" / "kept.json"


def read(cache_root) -> dict | None:
    """The kept coder's record, or None when there is none (or it cannot be read)."""
    try:
        rec = json.loads(path(cache_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def write(cache_root, rec: dict) -> None:
    p = path(cache_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def forget(cache_root, lease_id: str | None = None) -> None:
    """Remove the record once its coder is stopped (only the record for `lease_id`, when given)."""
    rec = read(cache_root)
    if rec is None:
        return
    if lease_id is not None and (rec.get("lease") or {}).get("lease_id") != lease_id:
        return
    try:
        path(cache_root).unlink()
    except FileNotFoundError:
        pass


def mismatch(kept: dict, identity: dict | None) -> str | None:
    """Why the kept coder is not this run's coder, or None when it is."""
    if not identity:
        return "this run does not know its coder's identity"
    have = kept.get("identity") if isinstance(kept.get("identity"), dict) else {}
    diff = [k for k in IDENTITY_KEYS if have.get(k) != identity.get(k)]
    if diff:
        return "the kept coder differs in " + ", ".join(f"{k} ({have.get(k)!r}, this run {identity.get(k)!r})"
                                                         for k in diff)
    return None


# ---- tt-orchard coder status | stop ------------------------------------------------------------------

STOP_WAIT_S = 180.0


def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard coder", add_help=add_help,
                                description="show or stop the coder a lab run kept up for the next run")
    p.add_argument("action", choices=["status", "stop"], nargs="?", default="status")
    p.add_argument("--yes", action="store_true", help="stop without asking")
    return p


def main(*, cfg, args, say=print, ask=input, probe=None, server=None, adapter=None,
         clock=time.monotonic, sleep=time.sleep) -> int:
    from orchard.caches import cache_root_of
    root = cache_root_of(cfg)
    rec = read(root)
    if rec is None:
        say(f"no coder is kept up ({path(root)} does not exist)")
        return 0
    ident = rec.get("identity") or {}
    lease = rec.get("lease") or {}
    if probe is None:
        from orchard.agent import probe_model as probe
    endpoint = f"http://127.0.0.1:{ident.get('port')}/v1"
    serving = bool(probe(endpoint, ident.get("model")))
    left = rec.get("left_at")
    say(f"kept coder: {ident.get('target')} on port {ident.get('port')} ({ident.get('chips')} chips)")
    say(f"  left up by {rec.get('run_dir')}"
        + (f" at {time.strftime('%Y-%m-%d %H:%M', time.localtime(left))}" if isinstance(left, (int, float)) else ""))
    say(f"  gozer lease {lease.get('lease_id')} on {', '.join(lease.get('chips') or [])} "
        f"(held by pids {', '.join(str(x) for x in rec.get('holder_pids') or []) or '?'})")
    say(f"  serving {ident.get('model')}: {'yes' if serving else 'no'}")
    if args.action == "status":
        return 0
    if not args.yes:
        answer = ask("stop the kept coder and free its chips? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            say("left running")
            return 1
    from orchard.adapters import AdapterError, Lease
    from orchard.server import ServerControl, ServerError, ServerSpec
    if server is None:
        server = ServerControl(ServerSpec(target=ident["target"], kind=ident["kind"], port=ident["port"],
                                          model=ident["model"], profile=ident.get("profile") or "default",
                                          image_id=ident.get("image_id")), log_path=os.devnull)
    try:
        server.adopt(rec.get("server") or {})
        if not server.confirm_stopped().stopped:
            server.stop()
        deadline = clock() + STOP_WAIT_S
        while not server.confirm_stopped().stopped:
            if clock() > deadline:
                say(f"refused: the coder is not confirmed stopped after {STOP_WAIT_S:.0f} s; its lease was kept")
                return 1
            sleep(2.0)
    except (ServerError, OSError) as exc:
        say(f"refused: stopping the coder failed: {exc}")
        return 1
    if adapter is None:
        from orchard.adapters.gozer import GozerAdapter
        adapter = GozerAdapter(gozer=cfg.gozer, owner_pid=os.getpid())
    try:
        adapter.release(Lease.from_record(lease))
    except AdapterError as exc:
        say(f"the coder stopped, but releasing lease {lease.get('lease_id')} failed: {exc}. "
            f"Release it with `gozer release {lease.get('lease_id')}`")
        return 1
    forget(root)
    say(f"stopped the kept coder and released lease {lease.get('lease_id')}")
    return 0
