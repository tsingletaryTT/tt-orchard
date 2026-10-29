#!/usr/bin/env python3
"""Tables for the staged-construction experiment (docs/run-logs/2026-10-28-staged-construction.md).

Usage: 2026-10-28-staged-construction-analysis.py EXP_DIR

EXP_DIR holds one probe output directory per attempt, named <arm>-<op>-<k> (arm W or S, k from 1), each run with
--attempts 1. The record is EXP_DIR/<arm>-<op>-<k>/<op>/attempt-1/attempt.json. Standard library only; it prints Markdown.

Definitions (fixed in the pre-registration, before the first attempt):
  passed        attempt.json passed_sim (the unedited check of the whole op)
  failure class the check's category, with tt_lang_error split into deadlock and "dataflow step" (first line text)
  reached       arm S: staged.reached, the last stage that was started (5 for rmsnorm, 4 for center is the final stage);
                arm W has no stages
  criteria      per op: arm S passes the op if S passes >= 1 attempt AND (S passes - W passes) >= 3, OR S reached the final
                stage in >= 5 of 10 attempts while W passed 0 of 10 (W "never got past the first failing shape error"
                is read as W having no pass; W's shape_mismatch count is printed beside it)
"""
import json
import sys
from pathlib import Path

OPS = ("center", "rmsnorm")
STAGE_COUNT = {"center": 4, "rmsnorm": 5}


def failure_class(category, first_line):
    if category is None:
        return "passed"
    text = first_line or ""
    if category == "tt_lang_error":
        if "Deadlock" in text:
            return "deadlock"
        if "not a valid next dataflow step" in text:
            return "dataflow step"
    return category


def load(exp: Path):
    rows = {}
    for d in sorted(exp.iterdir()):
        parts = d.name.split("-")
        if len(parts) != 3 or parts[0] not in ("W", "S") or parts[1] not in OPS or not parts[2].isdigit():
            continue
        path = d / parts[1] / "attempt-1" / "attempt.json"
        if not path.is_file():
            rows[(parts[0], parts[1], int(parts[2]))] = None
            continue
        rows[(parts[0], parts[1], int(parts[2]))] = json.loads(path.read_text(encoding="utf-8"))
    return rows


def counts(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def fmt(c):
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "none"


def main(exp_dir):
    rows = load(Path(exp_dir))
    print("## Results by arm\n")
    print("| op | arm | attempts recorded | passed | failure classes (failed attempts) | status | median turns | median wall s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    summary = {}
    for op in OPS:
        for arm in ("W", "S"):
            recs = [r for (a, o, k), r in sorted(rows.items(), key=lambda kv: kv[0][2]) if a == arm and o == op and r]
            passed = sum(1 for r in recs if r["passed_sim"])
            fails = [failure_class(r.get("category"), r.get("first_line")) for r in recs if not r["passed_sim"]]
            turns = sorted(r["turns"] or 0 for r in recs)
            walls = sorted(r["agent_wall_s"] or 0 for r in recs)
            med = lambda v: v[len(v) // 2] if v else "-"
            print(f"| {op} | {arm} | {len(recs)} | {passed} | {fmt(counts(fails))} | {fmt(counts(r['status'] for r in recs))} "
                  f"| {med(turns)} | {med(walls)} |")
            summary[(op, arm)] = {"recs": recs, "passed": passed, "fails": fails}
    print("\n## Arm S: the stage each attempt reached\n")
    for op in OPS:
        recs = summary[(op, "S")]["recs"]
        names = recs[0]["staged"]["stage_names"] if recs and recs[0].get("staged") else []
        print(f"### {op}\n")
        print("| stage | attempts that ended here | failed here | failure classes at this stage |")
        print("| --- | --- | --- | --- |")
        for i, name in enumerate(names, start=1):
            here = [r for r in recs if r["staged"]["reached"] == i]
            failed = [r for r in recs if r["staged"]["failed_stage"] == name]
            cls = counts(failure_class(r.get("category"), r.get("first_line")) for r in failed)
            print(f"| {i} {name} | {len(here)} | {len(failed)} | {fmt(cls)} |")
        print()
    print("## Criteria\n")
    print("| op | W passes | S passes | S minus W | S reached final stage | W shape_mismatch | criterion 1 | criterion 2 | S passes the op |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for op in OPS:
        w, s = summary[(op, "W")], summary[(op, "S")]
        reached_final = sum(1 for r in s["recs"] if r["staged"]["reached"] == STAGE_COUNT[op])
        c1 = s["passed"] >= 1 and s["passed"] - w["passed"] >= 3
        c2 = reached_final >= 5 and w["passed"] == 0
        print(f"| {op} | {w['passed']} of {len(w['recs'])} | {s['passed']} of {len(s['recs'])} | {s['passed'] - w['passed']} "
              f"| {reached_final} of {len(s['recs'])} | {counts(w['fails']).get('shape_mismatch', 0)} | {c1} | {c2} | {c1 or c2} |")
    print("\n## Per attempt\n")
    print("| op | arm | k | seed | result | stage reached | failed at | turns | wall s | decode tok/s before | status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (arm, op, k), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        if r is None:
            print(f"| {op} | {arm} | {k} | - | no record | - | - | - | - | - | - |")
            continue
        st = r.get("staged") or {}
        speed = (r.get("coder_speed") or {}).get("tok_s")
        res = "passed" if r["passed_sim"] else failure_class(r.get("category"), r.get("first_line"))
        print(f"| {op} | {arm} | {k} | {(r.get('sampling') or {}).get('seed')} | {res} | {st.get('reached', '-')} "
              f"| {st.get('failed_stage') or '-'} | {r.get('turns')} | {r.get('agent_wall_s')} "
              f"| {round(speed, 1) if isinstance(speed, (int, float)) else '-'} | {r.get('status')} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
