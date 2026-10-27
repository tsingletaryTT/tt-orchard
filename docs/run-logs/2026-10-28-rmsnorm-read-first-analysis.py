#!/usr/bin/env python3
"""Per-attempt measurements for the rmsnorm read-first experiment.

Usage: analyze_read_first.py OUT_DIR [OUT_DIR ...]
Reads OUT_DIR/rmsnorm/attempt-N/{attempt.json,log/probe-*.jsonl,reference/TTLangSpecification.md}
and prints one JSON line per attempt, then a table. Standard library only.

Definitions (fixed before the run):
  passed_sim      attempt.json passed_sim
  failure_class   deadlock | pop_invalid_step | shape_mismatch | numeric | wall_clock | other | pass
                  (from the authoritative category and first line in attempt.json; the stop reason
                  for a wall-clock end)
  first_check_turn   the turn whose tool call produced the first kernel_check result (the check that
                  runs after a write of kernel.py, or a shell call of kernel_check.py)
  block_states_read  count of read_file calls on the specification whose returned lines cover the
                  whole "Block states" section, plus shell calls whose output prints the section's
                  body (a numbered grep line inside it, or its table text)
  read_file_calls count of read_file calls on any file
"""
import json
import re
import sys
from pathlib import Path

SPEC = "TTLangSpecification.md"
SECTIONS = ("Dataflow buffer", "Block", "Block states")


def headings(spec_text):
    """{title: (first_line, last_line)} for every '## ' section, lines counted from 1."""
    lines = spec_text.splitlines()
    starts = [(i + 1, l[3:].strip()) for i, l in enumerate(lines) if l.startswith("## ")]
    out = {}
    for k, (n, title) in enumerate(starts):
        end = starts[k + 1][0] - 1 if k + 1 < len(starts) else len(lines)
        out[title] = (n, end)
    return out


def calls_and_results(rows):
    calls, results = [], {}
    for r in rows:
        for m in r["sent"]:
            if m.get("role") == "tool":
                results[m["tool_call_id"]] = m["content"]
        for c in (r.get("received") or {}).get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:
                args = {}
            calls.append({"turn": r["turn"], "id": c["id"], "name": c["function"]["name"], "args": args})
    return calls, results


def returned_range(result):
    """The line range a read_file result returned: from its footer, else the whole file (no footer)."""
    m = re.search(r"\[lines (\d+) to (\d+) of (\d+)", result)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1, len(result.splitlines())


def classify(a):
    if a.get("passed_sim"):
        return "pass"
    first = (a.get("first_line") or "")
    detail = (a.get("status_detail") or "")
    if "Deadlock" in first:
        return "deadlock"
    if "not a valid next dataflow step" in first:
        return "pop_invalid_step"
    if "Shape mismatch" in first:
        return "shape_mismatch"
    if a.get("category") == "numeric_mismatch":
        return "numeric"
    if a.get("status") in ("wall", "wall_clock") or "wall" in detail.lower():
        return "wall_clock"
    return "other"


def analyse(attempt_dir):
    a = json.loads((attempt_dir / "attempt.json").read_text())
    spec = headings((attempt_dir / "reference" / SPEC).read_text())
    bs_first, bs_last = spec["Block states"]
    rows = []
    for log in sorted((attempt_dir / "log").glob("probe-*.jsonl")):
        rows += [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    calls, results = calls_and_results(rows)
    reads = [c for c in calls if c["name"] == "read_file"]
    covered = {s: set() for s in SECTIONS}
    bs_read_file = 0
    for c in reads:
        if not str(c["args"].get("path", "")).endswith(SPEC):
            continue
        lo, hi = returned_range(results.get(c["id"], ""))
        if results.get(c["id"], "").startswith("error:"):
            continue
        for s in SECTIONS:
            f, l = spec[s]
            covered[s] |= set(range(max(lo, f), min(hi, l) + 1))
        if lo <= bs_first and hi >= bs_last:
            bs_read_file += 1
    bs_shell = 0
    for c in calls:
        if c["name"] != "shell":
            continue
        out = results.get(c["id"], "")
        numbered = {int(n) for n in re.findall(r"(?m)^(?:[\w./-]*TTLangSpecification\.md[:-])?(\d+)[:-]", out)}
        if any(bs_first < n <= bs_last for n in numbered) or "Must be Written" in out or "Read Only while Reading" in out:
            bs_shell += 1
    first_check = None
    for c in calls:
        out = results.get(c["id"], "")
        if (c["name"] == "write_file" and "kernel_check.py ran after this write" in out) or \
                (c["name"] == "shell" and "kernel_check.py" in str(c["args"].get("command", ""))
                 and re.search(r"\b(PASS|FAIL:)", out)):
            first_check = c["turn"]
            break
    shell_greps_ref = sum(1 for c in calls if c["name"] == "shell" and "reference" in str(c["args"].get("command", ""))
                          and "grep" in str(c["args"].get("command", "")))
    return {
        "attempt": a["attempt"], "passed_sim": bool(a.get("passed_sim")), "failure_class": classify(a),
        "first_line": a.get("first_line"), "turns": a.get("turns"), "status": a.get("status"),
        "first_check_turn": first_check, "block_states_read": bs_read_file + bs_shell,
        "block_states_read_file": bs_read_file, "block_states_shell": bs_shell,
        "read_file_calls": len(reads), "grep_calls_on_reference": shell_greps_ref,
        "spec_lines_read_file": {s: f"{len(covered[s])}/{spec[s][1] - spec[s][0] + 1}" for s in SECTIONS},
        "decode_tok_s": (a.get("coder_speed") or {}).get("tok_s"), "agent_wall_s": a.get("agent_wall_s"),
    }


def main(argv):
    for out in argv:
        print(f"## {out}")
        rs = [analyse(d) for d in sorted((Path(out) / "rmsnorm").glob("attempt-*"),
                                         key=lambda p: int(p.name.split("-")[1]))
              if (d / "attempt.json").is_file()]
        for r in rs:
            print(json.dumps(r))
        print("attempt | passed | class | first_check_turn | block_states_read | read_file_calls | tok/s")
        for r in rs:
            print(f"{r['attempt']} | {r['passed_sim']} | {r['failure_class']} | {r['first_check_turn']} | "
                  f"{r['block_states_read']} | {r['read_file_calls']} | {r['decode_tok_s']}")


if __name__ == "__main__":
    main(sys.argv[1:])
