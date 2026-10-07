# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Post-run checks for the operator: `python3 -m orchard.operator_checks --run-dir DIR`.

Run it on a run that is "ready-for-operator-review". It is read-only and needs no hardware. It
prints one line per check and exits 0 when nothing is wrong, 1 when it found a problem, and 2 for
a bad run directory. It checks:

- no tool call that an agent made looks like a publish, push or upload (`hf upload`, `git push`,
  `--public`, `--publish` and similar). Only the model's own tool calls in `stages/*/log/*.jsonl`
  are read (the "received" side of each turn). The "sent" side holds the skill text, which names
  these commands as forbidden, so it is not searched;
- none of the repos named in `stages/8/bundle/PUBLISH_COMMANDS.txt` exists on the hub. The lookup
  is anonymous: 404 or 401 means "not found for us", 200 means the repo is public. It cannot tell
  a private repo from a missing one, so "not found" is weaker than "nothing was uploaded", and
  the tool-call check above is the stronger evidence. With `--no-network`, or when the hub cannot
  be reached, this check is skipped with a note and does not fail the run;
- the bundle holds no hostname, token or home path (orchard/scrub.py, the supervisor's own check,
  run again).

It does not verify the ledger chain. `python3 -m orchard.supervisor status` does that.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from orchard.scrub import scrub_bundle

# What a publish-like shell command or tool argument looks like.
PUBLISH_LIKE = re.compile(r"hf upload|huggingface-cli upload|git push|--public\b|--publish\b|"
                          r"docker push|twine upload|upload_folder|upload_file|create_repo")
UPLOAD_LINE = re.compile(r"^\s*hf upload\b.*?\s([\w.-]+/[\w.-]+)\s")


def repos_in_publish_file(run_dir) -> list[str]:
    path = Path(run_dir) / "stages" / "8" / "bundle" / "PUBLISH_COMMANDS.txt"
    if not path.is_file():
        return []
    found = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = UPLOAD_LINE.match(line)          # a commented line starts with # and does not match
        if m:
            found.append(m.group(1))
    return found


def hub_status(repo: str) -> int | None:
    """The anonymous HTTP status of the repo's API page, or None when the hub cannot be reached."""
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=15) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (OSError, urllib.error.URLError):
        return None


def publish_like_calls(run_dir) -> list[str]:
    hits = []
    for log in sorted(Path(run_dir).glob("stages/*/log/*.jsonl")):
        for n, line in enumerate(log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            try:
                calls = json.loads(line).get("received", {}).get("tool_calls") or []
            except (json.JSONDecodeError, AttributeError):
                continue
            for call in calls:
                fn = call.get("function", {})
                if fn.get("name") != "shell":
                    continue        # write_file text (such as PUBLISH_COMMANDS.txt) is not run
                args = str(fn.get("arguments", ""))
                if PUBLISH_LIKE.search(args):
                    hits.append(f"{log.relative_to(run_dir)} line {n}: {' '.join(args.split())[:120]}")
    return hits


def check(run_dir, *, fetch=hub_status, hostname=None, home=None) -> dict:
    """{"ok", "problems", "notes"}. `fetch` returns an HTTP status or None; None skips a repo."""
    problems, notes = [], []
    problems += [f"publish-like tool call: {h}" for h in publish_like_calls(run_dir)]
    repos = repos_in_publish_file(run_dir)
    if not repos:
        notes.append("no repo ids found in PUBLISH_COMMANDS.txt")
    for repo in repos:
        code = fetch(repo)
        if code is None:
            notes.append(f"hub lookup skipped for {repo} (offline, unreachable or --no-network)")
        elif code == 200:
            problems.append(f"the repo {repo} exists and is readable without a login")
    bundle = Path(run_dir) / "stages" / "8" / "bundle"
    if bundle.is_dir():
        problems += [f"bundle scrub: {h}" for h in scrub_bundle(bundle, hostname=hostname, home=home)]
    else:
        notes.append("no stages/8/bundle directory to scrub")
    return {"ok": not problems, "problems": problems, "notes": notes}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m orchard.operator_checks",
                                description="read-only post-run checks for a finished run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--no-network", action="store_true", help="skip the hub lookup")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"refused: {run_dir} is not a directory", file=sys.stderr)
        return 2
    result = check(run_dir, fetch=(lambda repo: None) if args.no_network else hub_status)
    for note in result["notes"]:
        print(f"note: {note}")
    for problem in result["problems"]:
        print(f"PROBLEM: {problem}")
    print("checks: ok" if result["ok"] else "checks: PROBLEMS FOUND. Stop and ask a human.")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
