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
                  body (at least 5 of its numbered lines, or at least 3 of its six state names). A grep that prints 1 to 4
                  lines of the section is counted apart, as block_states_grep_hits.
  read_file_calls count of read_file calls on any file
  measurements_complete  False when a call has no result in the transcript (it stores a result on the next model turn, so a stop on
                  the wall budget or the turn limit leaves the last calls without one). Reads, section readings and the first check
                  are then lower bounds; the table prints ">=" and "none seen".
Limits: a shell command's unprefixed numbered output is credited only when the specification is its single file operand (see
single_source); a command that reads two files or runs two commands gets credit only for lines prefixed with the specification's
name. Tool arguments may be a JSON string or an already decoded object; anything that is not an object counts as no arguments.
"""
import json
import re
import shlex
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
            raw = c["function"]["arguments"]
            try:
                # the arguments arrive as a JSON string or as an already decoded object; anything that is
                # not an object (a list, null, a number) is a malformed call and counts as no arguments
                args = json.loads(raw or "{}") if isinstance(raw, (str, bytes)) else raw
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append({"turn": r["turn"], "id": c["id"], "name": c["function"]["name"], "args": args})
    return calls, results


def single_source(command):
    """True when a shell command's output can only come from the specification: one command (no `;`, `&&`, `||`, `&`), with the
    specification as its only file operand before any pipe. Output of a command with a second file or a second command could hold
    numbered lines or state names from elsewhere, so unprefixed numbers in it are not credited as a reading of the specification.
    A file operand is a word with a `.` or `/` that is not an option; a grep pattern with a dot is counted as one (a limit: it can
    only make this check refuse, never credit)."""
    try:
        lexer = shlex.shlex(str(command), posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return False
    if any(w in (";", "&&", "||", "&", "(", ")") for w in words):
        return False
    first_stage = words[:words.index("|")] if "|" in words else words
    operands = [w for w in first_stage[1:] if not w.startswith("-") and ("." in w or "/" in w) and not re.fullmatch(r"[\d,$p]+", w)]
    return len(operands) == 1 and operands[0].endswith(SPEC)


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
    if "shape mismatch" in first.lower() or a.get("category") == "shape_mismatch":
        return "shape_mismatch"
    if a.get("category") == "numeric_mismatch":
        return "numeric"
    # wall-clock means the attempt ended on the wall budget with no simulator result. An attempt that ended on the wall
    # budget after its last check answered with some other error is "other" (its `status` column says how it ended).
    # `kernel.py does not exist` (category no_kernel) is a check that never ran the simulator, so it has no result either.
    if (not first or a.get("category") == "no_kernel") and (a.get("status") in ("wall", "wall_clock", "wall-clock") or "wall" in detail.lower()):
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
    bs_grep_touch = 0
    shell_detail = []
    for c in calls:
        if c["name"] != "shell":
            continue
        out = results.get(c["id"], "")
        # credit output only when the command names the specification file; five `grep -n` hits numbered 744 to 748 in
        # another file are not a reading of "Block states"
        if SPEC not in str(c["args"].get("command", "")):
            continue
        # A line prefixed with the specification's name is from it whatever else the command did. A bare number counts only when
        # the command had no other source (single_source); `cat a.md TTLangSpecification.md` or `grep ... ; grep ... other.py`
        # print numbers and state names that may be another file's.
        prefixed = {int(n) for n in re.findall(r"(?m)^[\w./-]*TTLangSpecification\.md[:-](\d+)[:-]", out)}
        alone = single_source(c["args"].get("command", ""))
        bare = {int(n) for n in re.findall(r"(?m)^(\d+)[:-]", out)} if alone else set()
        numbered = prefixed | bare
        inside = sorted(n for n in numbered if bs_first < n <= bs_last)
        names = ("Must be Written", "Must be Read", "Read-Write", "Read Only while Reading", "No Access while Writing",
                 "Out of Scope")
        shown = [w for w in names if w in out] if alone else []
        if inside or shown:
            bs_grep_touch += 1
            # a grep that prints one or two lines of the section is a hit, not a reading of it; a shell call prints the
            # section when it shows at least 5 of its numbered lines or at least 3 of its six state names (sed -n, cat)
            if len(inside) >= 5 or len(shown) >= 3:
                bs_shell += 1
            shell_detail.append({"turn": c["turn"], "command": str(c["args"].get("command", ""))[:140],
                                 "numbered_lines_inside": inside, "state_names_shown": len(shown)})
    first_check = None
    for c in calls:
        out = results.get(c["id"], "")
        # an automatic check counts only when it produced a result; a check that was killed or crashed did not
        if (c["name"] == "write_file" and "kernel_check.py ran after this write" in out
                and re.search(r"\b(PASS|FAIL:)", out)) or \
                (c["name"] == "shell" and "kernel_check.py" in str(c["args"].get("command", ""))
                 and re.search(r"\b(PASS|FAIL:)", out)):
            first_check = c["turn"]
            break
    # the transcript records a tool result on the next model turn, so a stop on the wall budget or the turn limit can
    # leave the last calls without a result. Their reads and checks are then not counted; the count is reported.
    calls_without_result = sum(1 for c in calls if c["id"] not in results)
    # With any call left without a result, every measurement taken from results is a lower bound: a read, a check or a section
    # reading in the last calls may have happened and cannot be seen. `measurements_complete` says so, and the table marks it.
    complete = calls_without_result == 0
    shell_greps_ref = sum(1 for c in calls if c["name"] == "shell" and "reference" in str(c["args"].get("command", ""))
                          and "grep" in str(c["args"].get("command", "")))
    return {
        "attempt": a["attempt"], "passed_sim": bool(a.get("passed_sim")), "failure_class": classify(a),
        "first_line": a.get("first_line"), "turns": a.get("turns"), "status": a.get("status"),
        "first_check_turn": first_check, "block_states_read": bs_read_file + bs_shell,
        "block_states_read_file": bs_read_file, "block_states_shell": bs_shell, "block_states_grep_hits": bs_grep_touch, "block_states_shell_detail": shell_detail,
        "read_file_calls": len(reads), "calls_without_result": calls_without_result,
        "measurements_complete": complete, "grep_calls_on_reference": shell_greps_ref,
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
            # ">=" marks a count taken with calls that have no result in the transcript: the true count may be higher
            lb = "" if r["measurements_complete"] else ">="
            first = r["first_check_turn"] if r["first_check_turn"] is not None or r["measurements_complete"] else "none seen"
            print(f"{r['attempt']} | {r['passed_sim']} | {r['failure_class']} | {first} | "
                  f"{lb}{r['block_states_read']} | {r['read_file_calls']} | {r['decode_tok_s']}")


if __name__ == "__main__":
    main(sys.argv[1:])
