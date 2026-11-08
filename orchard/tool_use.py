# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Count the tool calls the coder made, per stage, from the agent step logs in a run directory (issue 581).

Each step log (`stages/<n>/log/<phase>-<seq>.jsonl`) has one line per model turn, `{"turn", "sent", "received"}`, and the calls are in
`received.tool_calls`. A stage directory that was moved aside on an escalation (`stages/2.partial-1`) keeps its logs, so its calls count toward
stage 2. Only calls the model made are counted: the text of a reply or of the task is not read, so a tool name written in prose is not a call.
A shell call that runs one of the stage tools is named by its subcommand (`stage_tools last-check`), any other shell call is `shell`.

The logs are a coder's output, so each file is read as untrusted: only a regular file that is not a link is opened (a link or a pipe is counted
as skipped and never followed), the read is bounded, and a line that is not a JSON object is counted as skipped.

    python3 -m orchard.tool_use RUN_DIR
"""
from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path

from orchard import safeio

LOG_MAX_BYTES = 64 * 1024 * 1024
STAGE_DIR = re.compile(r"^(\d+)(?:\.partial-\d+)?$")
STAGE_TOOL = re.compile(r"stage_tools(?:\.py)?\s+([a-z][a-z-]*)")


def _name(call) -> str | None:
    fn = call.get("function") if isinstance(call, dict) else None
    name = fn.get("name") if isinstance(fn, dict) else None
    if not isinstance(name, str) or not name:
        return None
    if name == "shell":
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = {}
        command = args.get("command") if isinstance(args, dict) else None
        m = STAGE_TOOL.search(command) if isinstance(command, str) else None
        if m:
            return f"stage_tools {m.group(1)}"
    return name[:60]


def count(run_dir) -> dict:
    """{stage number as text: {tool: calls}} and `skipped`, the number of files and lines that were not read."""
    out: dict = {"skipped": 0}
    stages = Path(run_dir) / "stages"
    try:
        dirs = sorted(os.scandir(stages), key=lambda e: e.name)
    except OSError:
        return out
    for d in dirs:
        m = STAGE_DIR.match(d.name)
        if not m or d.is_symlink() or not d.is_dir(follow_symlinks=False):
            continue
        tools = out.setdefault(m.group(1), {})
        try:
            files = sorted(os.scandir(Path(d.path) / "log"), key=lambda e: e.name)
        except OSError:
            continue
        for f in files:
            if not f.name.endswith(".jsonl"):
                continue
            try:
                if not stat.S_ISREG(f.stat(follow_symlinks=False).st_mode):
                    out["skipped"] += 1               # a link, a pipe or anything else: never opened
                    continue
                data = safeio.read_bytes(f.path, max_bytes=LOG_MAX_BYTES)
            except (safeio.SafeReadError, OSError):
                out["skipped"] += 1
                continue
            for line in data.decode("utf-8", "replace").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    out["skipped"] += 1
                    continue
                received = rec.get("received") if isinstance(rec, dict) else None
                calls = received.get("tool_calls") if isinstance(received, dict) else None
                if not isinstance(received, dict):
                    out["skipped"] += 1
                    continue
                for call in calls if isinstance(calls, list) else []:
                    name = _name(call)
                    if name:
                        tools[name] = tools.get(name, 0) + 1
    return out


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python3 -m orchard.tool_use RUN_DIR", file=sys.stderr)
        return 2
    got = count(args[0])
    for stage in sorted((k for k in got if k != "skipped"), key=int):
        tools = got[stage]
        total = sum(tools.values())
        print(f"stage {stage}: {total} tool calls")
        for name, n in sorted(tools.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:5d}  {name}")
    if got["skipped"]:
        print(f"{got['skipped']} files or lines were not read (links, pipes, oversize files, lines that are not JSON)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
