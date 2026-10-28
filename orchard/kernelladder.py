# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The capability ladder of the kernel probe (`tt-orchard kernel-probe --ladder`, issue 175).

The probe measured hard ops only (rmsnorm passed 0 of 10 attempts), so nobody knew where the coder's ability stops.
The ladder asks for a series of ops from easy to hard, each with the same sampling rules, and reports per rung the
pass count and the failure classes. The first rung where passes fall to zero is the target of later kernel
experiments. The rungs are `defaults.KERNEL_PROBE_LADDER`; every rung is an ordinary probe run (kernelprobe.main) in its
own directory, `<out>/<rung>/`, so a rung's attempts, report and records are the same as for `--op <rung>`.

This module holds what the ladder adds:

* `failure_class`: the class of one attempt's end, the classes of the rmsnorm read-first experiment plus a few that
  experiment lumped under "other";
* `size_wall_budget`: the stated way to size one rung's wall budget, so an easy rung does not burn 1800 s per attempt;
* `projected_s`: the projection the run uses to stop early when it would take longer than `--max-hours`;
* `write_ladder_report`: the table (`ladder-report.md` and `.json`) read from the rung directories;
* `run`: the loop over the rungs.

It imports nothing from kernelprobe at load time (kernelprobe imports it), so it can be tested alone.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from orchard import defaults

LADDER = defaults.KERNEL_PROBE_LADDER

# The failure classes, in the order the report lists them. The first five are the classes of the 2026-10-28 rmsnorm
# read-first run log (its "pop invalid step" is `dataflow_step` here). That run's "other" is split into `tt_lang_error`
# (any other error the simulator or tt-lang reports, a type error on a host tensor for example), `import_error`,
# `forbidden_code` and `no_kernel`; "other" keeps what none of these names.
FAILURE_CLASSES = ("deadlock", "dataflow_step", "shape_mismatch", "numeric", "tt_lang_error", "import_error",
                   "forbidden_code", "no_kernel", "wall_clock", "other")

REPORT_NAME = "ladder-report"

# Sizing a rung's wall budget. See size_wall_budget.
WALL_FLOOR_S = 300.0
WALL_CAP_S = 1800.0
WALL_FACTOR = 4.0
# Seconds the probe spends on one attempt outside the agent's wall time: the speed check before it, the sweep and the
# authoritative check after it. Measured as the gap between agent_wall_s and the attempt's elapsed time in the
# 2026-10-28 run (about a minute); a choice, not a constant of the tool.
ATTEMPT_OVERHEAD_S = 60.0


class LadderRefused(Exception):
    """A bad --rungs, --rung-wall, --rung-expect-s or --max-hours; the message is for the operator."""


# ---- one attempt -----------------------------------------------------------------------------------------------

def failure_class(a: dict) -> str | None:
    """The class of an attempt's end, from its record (attempt.json): None for a pass, else one of FAILURE_CLASSES.

    The class comes from the first line and the category of the unedited check, in this order: a first line with
    `Deadlock` is `deadlock`; one with `not a valid next dataflow step` (any operation: pop, store or copy) is
    `dataflow_step`; the category `shape_mismatch` or a first line with `shape mismatch` is `shape_mismatch`;
    `numeric_mismatch` is `numeric`. An attempt that ended on the wall budget with no simulator result (no first line,
    or the category `no_kernel`) is `wall_clock`. Otherwise `tt_lang_error`, `import_error`, `forbidden_code` and
    `no_kernel` by category, and `other` for anything else."""
    if a.get("passed_sim") is True:
        return None
    first = str(a.get("first_line") or "")
    category = a.get("category")
    detail = str(a.get("status_detail") or "")
    if "Deadlock" in first:
        return "deadlock"
    if "not a valid next dataflow step" in first:
        return "dataflow_step"
    if category == "shape_mismatch" or "shape mismatch" in first.lower():
        return "shape_mismatch"
    if category == "numeric_mismatch":
        return "numeric"
    on_wall = a.get("status") in ("wall", "wall_clock", "wall-clock") or "wall" in detail.lower()
    if (not first or category == "no_kernel") and on_wall:
        return "wall_clock"
    if category in ("tt_lang_error", "import_error", "forbidden_code", "no_kernel"):
        return category
    return "other"


# ---- the wall budget -------------------------------------------------------------------------------------------

def size_wall_budget(pass_walls_s: list[float], *, factor: float = WALL_FACTOR, floor_s: float = WALL_FLOOR_S,
                     cap_s: float = WALL_CAP_S) -> float:
    """The wall budget for a rung, in seconds, from the agent wall times of the pilot attempts that PASSED on it.

    The rule: `factor` (4) times the slowest passing pilot, at least `floor_s` (300) and at most `cap_s` (1800, the
    probe's default). A rung whose pilot did not pass has no time to scale from, so it keeps the cap: the question there
    is whether more time lets the coder pass, and a shorter budget would answer it before it is asked.

    A pass is judged on the kernel, not on the clock, so a budget that is long enough for the passes cannot lower the
    pass count; it can only cut the time failing attempts spend. The budget is checked between turns, so an attempt can
    overrun it by up to one tool timeout (420 s at most). A rung's failures that end on the budget are counted as
    `wall_clock` in the table, so the reader can see how many a longer budget could have changed."""
    walls = [float(w) for w in pass_walls_s if isinstance(w, (int, float)) and w > 0]
    if not walls:
        return float(cap_s)
    return float(min(cap_s, max(floor_s, factor * max(walls))))


def projected_s(plan: list[dict]) -> float:
    """Seconds the listed rungs still need: for each, attempts x (expected seconds per attempt + ATTEMPT_OVERHEAD_S).
    `plan` is [{"attempts": int, "expected_s": float}, ...]."""
    return float(sum(int(p["attempts"]) * (float(p["expected_s"]) + ATTEMPT_OVERHEAD_S) for p in plan))


# ---- the options -----------------------------------------------------------------------------------------------

def parse_rungs(text: str | None) -> list[str]:
    """The rungs to run, from `--rungs a,b,c` (any order the operator gives, each a rung of the ladder, none twice).
    No text means the whole ladder."""
    if text is None or not str(text).strip():
        return list(LADDER)
    names = [t.strip() for t in str(text).split(",")]
    for n in names:
        if n not in LADDER:
            raise LadderRefused(f"--rungs: {n!r} is not a rung; the rungs are {', '.join(LADDER)}")
    if len(set(names)) != len(names):
        raise LadderRefused("--rungs: a rung is listed twice")
    return names


def parse_seconds_map(text: str | None, flag: str) -> dict[str, float]:
    """`op=seconds,op=seconds` for --rung-wall and --rung-expect-s. Each op must be a rung, each value above 0."""
    out: dict[str, float] = {}
    if text is None or not str(text).strip():
        return out
    for part in str(text).split(","):
        name, sep, value = part.partition("=")
        name = name.strip()
        if not sep or name not in LADDER:
            raise LadderRefused(f"{flag}: {part.strip()!r} is not rung=seconds with a rung of the ladder "
                                f"({', '.join(LADDER)})")
        try:
            seconds = float(value)
        except ValueError:
            raise LadderRefused(f"{flag}: {value.strip()!r} is not a number") from None
        if not (seconds > 0) or seconds > 7200:                       # NaN fails this too
            raise LadderRefused(f"{flag}: {name}={value.strip()} is not above 0 and at most 7200 seconds")
        if name in out:
            raise LadderRefused(f"{flag}: {name} is listed twice")
        out[name] = seconds
    return out


# ---- the table -------------------------------------------------------------------------------------------------

def rung_row(op: str, attempts: list[dict], header: dict | None = None) -> dict:
    """One rung's numbers from its attempt records: counts only. `walls_s` are the agent wall times of the passes."""
    header = header or {}
    classes = {c: 0 for c in FAILURE_CLASSES}
    for a in attempts:
        c = failure_class(a)
        if c is not None:
            classes[c] += 1
    passes = sum(1 for a in attempts if a.get("passed_sim") is True)
    pass_walls = [a["agent_wall_s"] for a in attempts if a.get("passed_sim") is True
                  and isinstance(a.get("agent_wall_s"), (int, float))]
    all_walls = [a["agent_wall_s"] for a in attempts if isinstance(a.get("agent_wall_s"), (int, float))]
    seeds = [((a.get("sampling") or {}).get("seed")) for a in sorted(attempts, key=lambda a: a.get("attempt", 0))]
    budget = header.get("budget") if isinstance(header.get("budget"), dict) else {}
    return {"op": op, "attempts": len(attempts), "passed": passes,
            "classes": {c: n for c, n in classes.items() if n},
            "pass_walls_s": pass_walls, "median_pass_wall_s": statistics.median(pass_walls) if pass_walls else None,
            "total_wall_s": round(sum(all_walls), 1), "seeds": seeds,
            "temperatures": sorted({float((a.get("sampling") or {}).get("temperature") or 0.0) for a in attempts}),
            "wall_budget_s": budget.get("wall_s"), "max_turns": budget.get("max_turns"),
            "library": (header.get("library") or {}).get("mode"), "probe_id": header.get("probe_id"),
            "stopped": header.get("stopped")}


def first_zero_rung(rows: list[dict]) -> str | None:
    """The first rung, in ladder order, that ran at least one attempt and passed none; None when every rung that ran
    has a pass (or none ran). Rungs not run are skipped, and the report says which they were."""
    for r in sorted(rows, key=lambda r: LADDER.index(r["op"]) if r["op"] in LADDER else len(LADDER)):
        if r["attempts"] and r["passed"] == 0:
            return r["op"]
    return None


def _classes_cell(classes: dict) -> str:
    return ", ".join(f"{c} {n}" for c, n in classes.items()) or "none"


def _seed_text(seeds: list) -> str:
    vals = [s for s in seeds if isinstance(s, int)]
    if not vals:
        return "none (temperature 0)"
    return f"{min(vals)} to {max(vals)}" if len(vals) > 1 else str(vals[0])


def ladder_lines(rows: list[dict], *, skipped: list[str], stopped: str | None, settings: dict) -> list[str]:
    """The markdown of the ladder report."""
    zero = first_zero_rung(rows)
    lines = ["# Kernel probe ladder report", "",
             "Each rung is one op, asked for with the same sampling rules. A pass is the unedited simulator check "
             "passing on every input set. The report gives counts, never rates: 10 attempts cannot tell a pass rate of "
             "0% from one of 20% (0.8 to the tenth power is 0.11).", ""]
    if stopped:
        lines += [f"The ladder stopped: {stopped}", ""]
    lines += ["## Settings", ""]
    for key in ("coder_model", "temperatures", "seed_base", "attempts_per_rung", "library", "phase", "max_turns",
                "sim_version", "tt_lang_source_sha"):
        if settings.get(key) is not None:
            lines.append(f"- {key.replace('_', ' ')}: {settings[key]}")
    lines += ["", "## Passes and failure classes per rung", "",
              "| Rung | Op | Attempts | Passed | Failure classes | Median wall of a pass (s) | Wall budget (s) | Seeds |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rung = LADDER.index(r["op"]) + 1 if r["op"] in LADDER else "-"
        med = f"{r['median_pass_wall_s']:.0f}" if r["median_pass_wall_s"] is not None else "-"
        budget = f"{r['wall_budget_s']:.0f}" if isinstance(r["wall_budget_s"], (int, float)) else "-"
        lines.append(f"| {rung} | {r['op']} | {r['attempts']} | {r['passed']} | {_classes_cell(r['classes'])} | "
                     f"{med} | {budget} | {_seed_text(r['seeds'])} |")
    lines += [""]
    if skipped:
        lines += [f"Rungs not run: {', '.join(skipped)}.", ""]
    if zero:
        lines += [f"The first rung where passes fall to zero: `{zero}` "
                  f"(rung {LADDER.index(zero) + 1} of {len(LADDER)}).", ""]
    elif rows:
        lines += ["No rung that ran had zero passes.", ""]
    walled = [r for r in rows if r["classes"].get("wall_clock")]
    if walled:
        lines += ["Attempts that ended on the wall budget with no simulator result are counted as `wall_clock`: "
                  + ", ".join(f"{r['op']} {r['classes']['wall_clock']}" for r in walled)
                  + ". A longer budget could have changed those.", ""]
    lines += ["Failure classes (from the first line of the unedited check): `deadlock`, `dataflow_step` (any \"not a "
              "valid next dataflow step\"), `shape_mismatch`, `numeric`, `tt_lang_error` (any other error tt-lang or "
              "the simulator reports), `import_error`, `forbidden_code`, `no_kernel`, `wall_clock`, `other`.", ""]
    return lines


def load_rung(out_dir: Path, op: str) -> tuple[dict | None, list[dict]]:
    """The header and attempt records a rung's probe run wrote (`<out>/<op>/kernel-probe-report.json`), or (None, [])."""
    path = Path(out_dir) / op / "kernel-probe-report.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    attempts = [a for a in data.get("attempts", []) if isinstance(a, dict) and a.get("op") == op]
    return (data.get("header") if isinstance(data.get("header"), dict) else {}), attempts


def write_ladder_report(out_dir, rungs: list[str], *, stopped: str | None = None) -> dict:
    """Read the rung directories under `out_dir` and write `ladder-report.md` and `ladder-report.json` there.
    Returns the data written. A rung with no report is listed as not run."""
    out_dir = Path(out_dir)
    rows, skipped, headers = [], [], {}
    for op in rungs:
        header, attempts = load_rung(out_dir, op)
        if header is None or not attempts:
            skipped.append(op)
            continue
        headers[op] = header
        rows.append(rung_row(op, attempts, header))
    first = next(iter(headers.values()), {})
    sampling = first.get("sampling") if isinstance(first.get("sampling"), dict) else {}
    settings = {"coder_model": first.get("coder_model"), "temperatures": sampling.get("temperatures"),
                "seed_base": sampling.get("seed_base"), "attempts_per_rung": sorted({r["attempts"] for r in rows}),
                "library": sorted({r["library"] for r in rows if r["library"]}),
                "phase": "sim", "max_turns": sorted({r["max_turns"] for r in rows if r["max_turns"]}),
                "sim_version": (first.get("tt_lang") or {}).get("sim_version"),
                "tt_lang_source_sha": (first.get("tt_lang") or {}).get("source_sha")}
    data = {"rungs": rows, "not_run": skipped, "first_zero_rung": first_zero_rung(rows), "stopped": stopped,
            "settings": settings, "ladder": list(LADDER)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{REPORT_NAME}.json").write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / f"{REPORT_NAME}.md").write_text(
        "\n".join(ladder_lines(rows, skipped=skipped, stopped=stopped, settings=settings)) + "\n", encoding="utf-8")
    return data


# ---- the run ---------------------------------------------------------------------------------------------------

def run(args, main, *, default_wall_s: float, say=print, clock=time.monotonic) -> int:
    """Run the rungs one after another: `main` is kernelprobe.main with its keyword arguments bound, called once per
    rung with a copy of `args` whose `op`, `out` and `wall_s` are the rung's and whose `ladder` is off. A rung that
    returns a code other than 0 ends the ladder with that code. After each rung the projection (elapsed seconds plus
    projected_s of the rungs left, each at its `--rung-expect-s` or else its wall budget) is compared with
    `--max-hours`; a ladder that would run longer stops, and the report says so. The report is written from whatever
    finished. Returns the exit code (2 for a refused option)."""
    import copy
    out_dir = Path(getattr(args, "out", None) or "kernel-probe")
    try:
        rungs = parse_rungs(getattr(args, "rungs", None))
        walls = parse_seconds_map(getattr(args, "rung_wall", None), "--rung-wall")
        expects = parse_seconds_map(getattr(args, "rung_expect_s", None), "--rung-expect-s")
        max_h = getattr(args, "max_hours", None)
        if max_h is not None:
            try:
                max_h = float(max_h)
            except (TypeError, ValueError):
                raise LadderRefused(f"--max-hours: {max_h!r} is not a number") from None
            if not (max_h > 0):
                raise LadderRefused("--max-hours must be above 0")
        for flag in ("rung_wall", "rung_expect_s"):
            extra = [k for k in parse_seconds_map(getattr(args, flag, None), "--" + flag.replace("_", "-"))
                     if k not in rungs]
            if extra:
                raise LadderRefused(f"--{flag.replace('_', '-')} names {', '.join(extra)}, which --rungs does not run")
        if getattr(args, "op", None):
            raise LadderRefused("--ladder runs rungs; name them with --rungs and do not give --op")
        default_wall = getattr(args, "wall_s", None)
        default_wall = float(default_wall) if default_wall is not None else float(default_wall_s)
    except LadderRefused as exc:
        import sys
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    attempts = int(getattr(args, "attempts", 5) or 0)
    n_temps = max(1, len(str(getattr(args, "temperatures", None) or "x").split(",")))
    per_rung = attempts * n_temps

    def wall_of(op: str) -> float:
        return walls.get(op, default_wall)

    started = clock()
    stopped, code = None, 0
    if getattr(args, "report_only", False):
        data = write_ladder_report(out_dir, rungs)
        say(f"ladder report: {out_dir / (REPORT_NAME + '.md')}")
        return 0
    for i, op in enumerate(rungs):
        sub = copy.copy(args)
        sub.op, sub.out, sub.wall_s, sub.ladder = op, str(out_dir / op), wall_of(op), False
        say(f"ladder rung {i + 1} of {len(rungs)}: {op} (wall budget {wall_of(op):.0f} s)")
        code = main(sub)
        if code != 0:
            stopped = f"rung {op} ended with exit code {code}"
            break
        left = [{"attempts": per_rung, "expected_s": expects.get(r, wall_of(r))} for r in rungs[i + 1:]]
        if max_h is not None and left:
            total_h = (clock() - started + projected_s(left)) / 3600.0
            if total_h > max_h:
                stopped = (f"after rung {op} the projection is {total_h:.1f} hours, above --max-hours {max_h:g}; "
                           f"not run: {', '.join(rungs[i + 1:])}")
                break
    data = write_ladder_report(out_dir, rungs, stopped=stopped)
    say(f"ladder report: {out_dir / (REPORT_NAME + '.md')}")
    if stopped:
        say(f"ladder stopped: {stopped}")
    return code
