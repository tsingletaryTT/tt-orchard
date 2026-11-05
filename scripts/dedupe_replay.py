#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Replay the tool calls recorded in agent step logs through the tool-result dedupe rules, read only (docs/tool-result-dedupe.md).

    python3 scripts/dedupe_replay.py RUN_DIR [RUN_DIR ...]

For each run directory it reads `stages/*/log/*.jsonl` (each file is one agent step: a record per model turn, the assistant message in `received`
and the tool results in the next record's `sent`) and applies the same rules `orchard.agent.Tools` applies live:

- a read_file of a path and offset read earlier in the step, with the same text, would have been a note;
- a read-only shell command (`orchard.agent.read_only_command`) that ran earlier in the step with the same output would have been a note, unless a
  write_file or a command off the list came between;
- a write_file ends the notes for its file and for every shell command.

The recorded logs hold no file identity, so equal text stands for "unchanged". A path is matched without its leading `stages/<n>/`. The count is
therefore an estimate of what the live rule would have done. Nothing is written and no model is called. It prints one line per run and a total:
the calls, how many would have been notes, and the characters of tool output that would not have gone into the context again."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchard import safeio                      # noqa: E402
from orchard.agent import read_only_command    # noqa: E402

LOG_MAX_BYTES = 64 * 1024 * 1024
NOTE_CHARS = 190                                # about the length of a note: the path, the turn numbers and one sentence


def norm(path: str) -> str:
    path = str(path).strip().replace("\\", "/")
    path = re.sub(r"^\./", "", path)
    return re.sub(r"^stages/\d+/", "", path)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def replay_step(records: list, run_dir: Path) -> dict:
    reads: dict[tuple[str, int], str] = {}
    shell: dict[str, str] = {}
    out = {"read_calls": 0, "read_notes": 0, "shell_calls": 0, "shell_notes": 0, "chars": 0, "saved": 0}
    pending: list[tuple[str, dict]] = []
    for rec in records:
        # The tool results answer the calls of the record before: they are the `tool` messages of this record's `sent`.
        results = [m.get("content", "") for m in rec.get("sent", []) if isinstance(m, dict) and m.get("role") == "tool"]
        for (name, args), text in zip(pending, results):
            text = text if isinstance(text, str) else json.dumps(text)
            out["chars"] += len(text)
            if name == "read_file":
                out["read_calls"] += 1
                key = (norm(args.get("path", "")), args.get("offset", 0) if isinstance(args.get("offset", 0), int) else 0)
                if reads.get(key) == digest(text) and not text.startswith(("error:", "refused:")):
                    out["read_notes"] += 1
                    out["saved"] += max(len(text) - NOTE_CHARS, 0)
                elif not text.startswith(("error:", "refused:")):
                    reads[key] = digest(text)
            elif name == "shell":
                out["shell_calls"] += 1
                command = args.get("command")
                if not read_only_command(command, run_dir):
                    shell.clear()
                elif text.startswith("exit 0\n"):
                    if shell.get(command) == digest(text):
                        out["shell_notes"] += 1
                        out["saved"] += max(len(text) - NOTE_CHARS, 0)
                    else:
                        shell[command] = digest(text)
            elif name == "write_file":
                shell.clear()
                target = norm(args.get("path", ""))
                for key in [k for k in reads if k[0] == target]:
                    del reads[key]
        pending = []
        msg = rec.get("received") or {}
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            pending.append((str(fn.get("name", "")), args if isinstance(args, dict) else {}))
    return out


def replay_run(run_dir: Path) -> dict:
    total = {"steps": 0, "read_calls": 0, "read_notes": 0, "shell_calls": 0, "shell_notes": 0, "chars": 0, "saved": 0, "worst_step_notes": 0}
    for log in sorted(run_dir.glob("stages/*/log/*.jsonl")):
        try:
            records = safeio.read_jsonl(log, max_bytes=LOG_MAX_BYTES, expect=dict)
        except safeio.SafeReadError:
            continue
        step = replay_step(records, run_dir)
        total["steps"] += 1
        for k in ("read_calls", "read_notes", "shell_calls", "shell_notes", "chars", "saved"):
            total[k] += step[k]
        total["worst_step_notes"] = max(total["worst_step_notes"], step["read_notes"] + step["shell_notes"])
    return total


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows, grand = [], {}
    for arg in argv:
        run = Path(arg)
        t = replay_run(run)
        rows.append((run.name, t))
        for k, v in t.items():
            grand[k] = max(grand.get(k, 0), v) if k == "worst_step_notes" else grand.get(k, 0) + v
    head = f"{'run':<28}{'steps':>6}{'reads':>7}{'notes':>7}{'shell':>7}{'notes':>7}{'tool chars':>12}{'saved':>10}{'saved %':>9}{'worst step':>11}"
    print(head)
    for name, t in rows + [("TOTAL", grand)]:
        pct = 100.0 * t["saved"] / t["chars"] if t["chars"] else 0.0
        print(f"{name[:27]:<28}{t['steps']:>6}{t['read_calls']:>7}{t['read_notes']:>7}{t['shell_calls']:>7}{t['shell_notes']:>7}"
              f"{t['chars']:>12}{t['saved']:>10}{pct:>8.1f}%{t['worst_step_notes']:>11}")
    reads = grand.get("read_calls", 0)
    print(f"\nread_file calls {reads}, would have been notes {grand.get('read_notes', 0)} "
          f"({100.0 * grand.get('read_notes', 0) / reads:.0f}% of reads)" if reads else "\nno read_file calls found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
