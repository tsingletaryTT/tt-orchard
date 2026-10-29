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
import os
import signal
import statistics
import sys
import tempfile
import time
from pathlib import Path

from orchard import defaults

LADDER = defaults.KERNEL_PROBE_LADDER

# The failure classes, in the order the report lists them. The first five are the classes of the 2026-10-28 rmsnorm
# read-first run log (its "pop invalid step" is `dataflow_step` here). That run's "other" is split into `tt_lang_error`
# (any other error the simulator or tt-lang reports, a type error on a host tensor for example), `import_error`,
# `forbidden_code` and `no_kernel`; "other" keeps what none of these names.
FAILURE_CLASSES = ("deadlock", "dataflow_step", "shape_mismatch", "numeric", "tt_lang_error", "import_error",
                   "forbidden_code", "no_kernel", "sim_gap", "wall_clock", "other")

REPORT_NAME = "ladder-report"
READ_LIMIT_BYTES = 64 * 1024 * 1024      # the most a rung report may be: the probe's own limit for a record

# Sizing a rung's wall budget. See size_wall_budget.
WALL_FLOOR_S = 300.0
WALL_CAP_S = 1800.0
WALL_FACTOR = 4.0
# Seconds the probe spends on one attempt outside the agent's wall time: the speed check before it, the sweep and the
# authoritative check after it. A choice that was not measured; the run log compares it with the attempts' timings.
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
    if category in ("tt_lang_error", "import_error", "forbidden_code", "no_kernel", "sim_gap"):
        return category
    return "other"


# ---- the wall budget -------------------------------------------------------------------------------------------

def size_wall_budget(pass_walls_s: list[float], *, factor: float = WALL_FACTOR, floor_s: float = WALL_FLOOR_S,
                     cap_s: float = WALL_CAP_S) -> float:
    """The wall budget for a rung, in seconds, from the agent wall times of the pilot attempts that PASSED on it.

    The rule: `factor` (4) times the slowest passing pilot, at least `floor_s` (300) and at most `cap_s` (1800, the
    probe's default). A rung whose pilot did not pass has no time to scale from, so it keeps the cap: the question there
    is whether more time lets the coder pass, and a shorter budget would answer it before it is asked.

    The unedited check judges the kernel alone, so a budget that is long enough for the passes cannot lower the pass
    count; it can only cut the time failing attempts spend. The budget is checked between turns, so an attempt can
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
    No option (None) means the whole ladder; an option given empty is refused."""
    if text is None:
        return list(LADDER)
    if not str(text).strip():
        raise LadderRefused("--rungs is empty; name rungs or leave the option out")
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
    if text is None:
        return out
    if not str(text).strip():
        raise LadderRefused(f"{flag} is empty; give rung=seconds or leave the option out")
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

def _num(value) -> float:
    """`value` as a float, or 0.0 when it is not a number."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _d(value) -> dict:
    """`value` when it is a dict, else an empty one: a report written by something else may hold any type."""
    return value if isinstance(value, dict) else {}


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
    seeds = [(_d(a.get("sampling")).get("seed")) for a in sorted(attempts, key=lambda a: _num(a.get("attempt")))]
    budget = _d(header.get("budget"))
    return {"op": op, "attempts": len(attempts), "passed": passes,
            "classes": {c: n for c, n in classes.items() if n},
            "pass_walls_s": pass_walls, "median_pass_wall_s": statistics.median(pass_walls) if pass_walls else None,
            "total_wall_s": round(sum(all_walls), 1), "seeds": seeds,
            "temperatures": sorted({float(_d(a.get("sampling")).get("temperature") or 0.0) for a in attempts}),
            "wall_budget_s": budget.get("wall_s"), "max_turns": budget.get("max_turns"),
            "library": _d(header.get("library")).get("mode"), "probe_id": header.get("probe_id"),
            "coder_model": header.get("coder_model"),
            "sim_version": _d(header.get("tt_lang")).get("sim_version"),
            "source_sha": _d(header.get("tt_lang")).get("source_sha"),
            "seed_base": _d(header.get("sampling")).get("seed_base"),
            "top_p": (lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else None)(
                _d(header.get("sampling")).get("top_p")),
            "stopped": header.get("stopped")}


def first_zero_rung(rows: list[dict]) -> str | None:
    """The first rung, in ladder order, that ran at least one attempt, passed none and was not stopped early; None when
    there is none. A rung whose run was stopped has fewer attempts than planned, so its zero is not evidence. Rungs not
    run are skipped, and the report says which they were."""
    for r in sorted(rows, key=lambda r: LADDER.index(r["op"]) if r["op"] in LADDER else len(LADDER)):
        if r["attempts"] and r["passed"] == 0 and not r.get("stopped"):
            return r["op"]
    return None


def _agreed(values: list):
    """The one value the rungs share, or the sorted list of the distinct values when they differ (None when none has one)."""
    by_text = {json.dumps(v, sort_keys=True, default=str): v for v in values if v is not None}     # any type, even a list
    distinct = [by_text[k] for k in sorted(by_text)]
    return distinct[0] if len(distinct) == 1 else (distinct or None)


MIXED_KEYS = (("coder_model", "coder model"), ("sim_version", "simulator version"), ("source_sha", "tt-lang source sha"),
              ("seed_base", "seed base"), ("library", "library mode"), ("sampling", "sampling"),
              ("max_turns", "turn limit"))


def sampling_text(row: dict) -> str:
    """One rung's sampling in words, for comparing rungs: its temperatures and its top-p."""
    temps = ", ".join(f"{t:g}" for t in row.get("temperatures") or [])
    top_p = row.get("top_p")
    return f"temperatures {temps or 'none'}, top-p {'none' if top_p is None else f'{top_p:g}'}"


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
    for key in ("coder_model", "temperatures", "sampling", "seed_base", "seed_base_from", "attempts_per_rung", "library",
                "phase", "max_turns", "sim_version", "source_sha"):
        if settings.get(key) is not None:
            lines.append(f"- {key.replace('_', ' ')}: {settings[key]}")
    if settings.get("mixed"):
        names = dict(MIXED_KEYS)
        lines += ["", "The rungs differ in " + ", ".join(names[k] for k in settings["mixed"]) + ", so their counts are not "
                  "directly comparable. Each rung's own values:", ""]
        lines += [f"- {r['op']}: model {r['coder_model']}, simulator {r['sim_version']}, seed base {r['seed_base']}, "
                  f"library {r['library']}, {sampling_text(r)}, turn limit {r['max_turns']}" for r in rows]
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
        lines += ["No rung that ran to its end had zero passes." if any(r.get("stopped") for r in rows) else "No rung that ran had zero passes.", ""]
    stopped_rungs = [r for r in rows if r.get("stopped")]
    if stopped_rungs:
        lines += ["Rungs stopped early, whose counts are partial and are not used for the first zero rung: "
                  + ", ".join(f"{r['op']} ({r['stopped']})" for r in stopped_rungs) + ".", ""]
    walled = [r for r in rows if r["classes"].get("wall_clock")]
    if walled:
        lines += ["Attempts that ended on the wall budget with no simulator result are counted as `wall_clock`: "
                  + ", ".join(f"{r['op']} {r['classes']['wall_clock']}" for r in walled)
                  + ". A longer budget could have changed those.", ""]
    lines += ["Failure classes (from the first line of the unedited check): `deadlock`, `dataflow_step` (any \"not a "
              "valid next dataflow step\"), `shape_mismatch`, `numeric`, `tt_lang_error` (any other error tt-lang or "
              "the simulator reports), `import_error`, `forbidden_code`, `no_kernel`, `sim_gap` (the pinned simulator lacks a function the specification documents, so the "
              "result does not show what the coder cannot write), `wall_clock`, `other`.", ""]
    return lines


def load_rung(out_dir: Path, op: str) -> tuple[dict | None, list[dict]]:
    """The header and attempt records a rung's probe run wrote (`<out>/<op>/kernel-probe-report.json`), or (None, [])."""
    path = Path(out_dir) / op / "kernel-probe-report.json"
    from orchard import kernelprobe as kp
    raw = kp._read_regular(path, READ_LIMIT_BYTES)       # a regular file only: no link, no FIFO wait, a size limit
    if raw is None or len(raw) > READ_LIMIT_BYTES:
        return None, []
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None, []
    if not isinstance(data, dict):
        return None, []
    if not isinstance(data.get("attempts"), list):
        return None, []
    attempts = [a for a in data["attempts"] if isinstance(a, dict) and a.get("op") == op]
    return (data.get("header") if isinstance(data.get("header"), dict) else {}), attempts


def _write_replacing(path: Path, text: str) -> None:
    """Write `text` to `path` by writing a new file beside it and renaming it over `path`. A link at `path` is replaced,
    and the file it points to is not touched."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")      # O_EXCL, a random name
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(name, 0o644)
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def write_ladder_report(out_dir, rungs: list[str], *, stopped: str | None = None, ran=None) -> dict:
    """Read the rung directories under `out_dir` and write `ladder-report.md` and `ladder-report.json` there.
    Returns the data written. A rung with no report is listed as not run.

    `ran` is the set of rungs the current invocation ran and that wrote a report. A rung outside it is listed as not
    run even when an earlier invocation left a report in its directory, so a stopped ladder does not mix in old
    results. None (`--report-only`) reads every rung's report."""
    out_dir = Path(out_dir)
    rows, skipped, headers = [], [], {}
    for op in rungs:
        if ran is not None and op not in ran:
            skipped.append(op)
            continue
        header, attempts = load_rung(out_dir, op)
        if header is None or not attempts:
            skipped.append(op)
            continue
        headers[op] = header
        rows.append(rung_row(op, attempts, header))
    first = next(iter(headers.values()), {})
    sampling = first.get("sampling") if isinstance(first.get("sampling"), dict) else {}
    # Each rung keeps its own model, simulator and seed base in its row. The settings show the value the rungs share,
    # or every distinct value when they differ (a `--report-only` over directories from different runs), and say so.
    for r in rows:
        r["sampling"] = sampling_text(r)
    settings = {key: _agreed([r[key] for r in rows]) for key, _ in MIXED_KEYS}
    settings["mixed"] = [key for key, _ in MIXED_KEYS if isinstance(settings[key], list)]
    bases = settings["seed_base"]
    settings.update({"temperatures": rows[0]["temperatures"] if rows and "sampling" not in settings["mixed"] else None,
                     "seed_base_from": sampling.get("seed_base_from") if isinstance(bases, int) else None,
                     "attempts_per_rung": sorted({r["attempts"] for r in rows}),
                     "phase": "sim"})
    data = {"rungs": rows, "not_run": skipped, "first_zero_rung": first_zero_rung(rows), "stopped": stopped,
            "settings": settings, "ladder": list(LADDER)}
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_replacing(out_dir / f"{REPORT_NAME}.json", json.dumps(data, indent=2, default=str) + "\n")
    _write_replacing(out_dir / f"{REPORT_NAME}.md",
                     "\n".join(ladder_lines(rows, skipped=skipped, stopped=stopped, settings=settings)) + "\n")
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
        if getattr(args, "phase", "sim") != "sim":
            raise LadderRefused("--ladder measures the simulator only; --phase hw is not supported with it")
        default_wall = getattr(args, "wall_s", None)
        if default_wall is not None:
            try:
                default_wall = float(default_wall)
            except (TypeError, ValueError):
                raise LadderRefused(f"--wall-s: {default_wall!r} is not a number") from None
            if not (0 < default_wall <= 7200):                                  # NaN fails this too
                raise LadderRefused(f"--wall-s {default_wall:g} is not above 0 and at most 7200 seconds")
        else:
            default_wall = float(default_wall_s)
    except LadderRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    attempts = int(getattr(args, "attempts", 5) or 0)
    n_temps = max(1, len(str(getattr(args, "temperatures", None) or "x").split(",")))
    per_rung = attempts * n_temps

    def wall_of(op: str) -> float:
        return walls.get(op, default_wall)

    from orchard import kernelprobe as kp
    try:
        lock = kp.acquire_lock(out_dir)          # held from here to the report: a second ladder cannot mix its rungs in
    except kp._Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    try:
        started = clock()
        stopped, code, held = None, 0, []
        if getattr(args, "report_only", False):
            # Lock every selected rung's directory while its report is read, so a probe that is writing one is not read
            # half done (and is refused if it starts meanwhile).
            for op in rungs:
                try:
                    held.append(kp.acquire_lock(out_dir / op))
                except kp._Refused as exc:
                    print(f"refused: rung {op}: {exc}", file=sys.stderr)
                    return 2
            write_ladder_report(out_dir, rungs)
            say(f"ladder report: {out_dir / (REPORT_NAME + '.md')}")
            return 0
        # One seed base for every rung, so attempt k sends the same seed in each. Without --seed the base comes from
        # one id for the whole ladder, and the rungs' headers say so.
        seed, seed_from = getattr(args, "seed", None), None
        if seed is None:
            seed, seed_from = str(kp.seed_base_from_probe_id(kp.new_probe_id())), "the ladder's id"
        ran = set()
        try:
            previous = signal.signal(signal.SIGTERM, kp._raise_interrupt)      # SIGTERM stops the ladder like Ctrl-C
        except ValueError:                                                     # not the main thread
            previous = None
        try:
            for i, op in enumerate(rungs):
                sub = copy.copy(args)
                sub.op, sub.out, sub.wall_s, sub.ladder = op, str(out_dir / op), wall_of(op), False
                sub.seed, sub.seed_from = seed, seed_from
                sub.rungs = sub.rung_wall = sub.rung_expect_s = sub.max_hours = None      # the ladder's own options
                sub.report_only = False
                say(f"ladder rung {i + 1} of {len(rungs)}: {op} (wall budget {wall_of(op):.0f} s)")
                # The rung's directory is locked from before its run to the end of the ladder, so no other probe can replace
                # its report in between. The run is told, and neither takes the lock again nor releases it.
                try:
                    held.append(kp.acquire_lock(out_dir / op))
                except kp._Refused as exc:
                    stopped, code = f"rung {op}: {exc}", 2
                    break
                sub.dir_locked = True
                before = (load_rung(out_dir, op)[0] or {}).get("probe_id")
                code = main(sub)
                # The rung is this run's only when its report carries a new probe id. A refused run (2) wrote none, and a
                # run that failed to write its report (3) leaves an older one in place.
                after = (load_rung(out_dir, op)[0] or {}).get("probe_id")
                if code != 2 and after is not None and after != before:
                    ran.add(op)
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
        except KeyboardInterrupt:                # Ctrl-C or SIGTERM between two rungs: the report still gets written
            stopped, code = "interrupted", 4
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
        write_ladder_report(out_dir, rungs, stopped=stopped, ran=ran)
        say(f"ladder report: {out_dir / (REPORT_NAME + '.md')}")
        if stopped:
            say(f"ladder stopped: {stopped}")
        return code
    finally:
        for h in held:
            kp.release_lock(h)
        kp.release_lock(lock)
