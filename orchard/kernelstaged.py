# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Staged construction for the kernel probe (issue 176): the coder builds a tt-lang kernel in steps, and each step ends at a
simulator check against a reference for that step.

The whole-kernel probe (kernelprobe.run_attempt) gives the coder one prompt and one check for the finished op. In the
ladder runs the coder passed elementwise ops and `row_sum` and failed `center` and `rmsnorm`, which add a reduction and a
broadcast back. Staged construction builds the same kernel in the order a person would: copy the input through the
dataflow buffers, add one arithmetic step, add the reduction, add the broadcast back, and end at the real op.

How it works. `STAGES[op]` is a table (data, so another op is one more entry). Each `Stage` has a name, a short
instruction that names only that step (no code, no worked solution), the reference function the stage's output is compared
with (a key in staged_references.STAGE_REFERENCES; None on the last stage, which uses the op's own reference), the inputs
and output shape of the stage, and its own turn and wall budgets. `run_attempt` runs one agent conversation per stage, in
order:

  * Stage k's check is the shipped check with a spec made by `stage_spec` and references that know the stage's reference
    (`write_stage_templates`). The simulator bars are the op's own. The LAST stage is judged by the unmodified op spec and
    references, so passing it is passing the whole-kernel probe's check.
  * The verified prefix is kept. When stage k passes, the exact bytes the authoritative check read become stage k+1's
    starting `kernel.py`. The coder edits that file; it does not start again.
  * A stage that fails its check, or spends its turns or its wall clock, ends the attempt. The record names the stage.
  * Every attempt records the stage it reached (`staged.reached`, counting from 1: the last stage that was started), the
    stages it passed, the failed stage's name, and each stage's category, turns and time. The attempt's seed is sent with
    every stage's requests.

The coder gets no hardware tool in a staged attempt (the stage checks run in the simulator only).
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from orchard import kernelprobe as kp
from orchard.agent import AgentStep
from orchard.canary import post_json
from orchard.ledger import Ledger
from orchard.watchdog import RetryGuard

STAGED_NOTE = Path(__file__).resolve().parent / "skills" / "kernel-probe-staged.md"
STAGED_DIR = ".staged"              # DIR/.staged/<op>-<n>-s<k>/: the files stage k's check runs against (outside the agent's reach)
WHOLE_KERNEL_TURNS = kp.BUDGET["max_turns"]      # the whole-kernel probe's budget for one attempt: 40 turns
WHOLE_KERNEL_WALL_S = kp.BUDGET["wall_s"]        # and 1800 s. The stages of an op share these, so both arms get the same totals.


@dataclass(frozen=True)
class Stage:
    name: str                        # short, used in the record and the report
    instruction: str                 # what kernel.py must do at the end of this stage. No code and no solution.
    reference: str | None            # key in staged_references.STAGE_REFERENCES; None: the op's own reference (last stage)
    inputs: tuple | None = None      # names of the op's inputs this stage takes, in order; None: all of them
    output: tuple | None = None      # the output shape of this stage; None: the op's
    max_turns: int = 8               # set by _share_budget: the attempt's turns divided over the op's stages
    wall_s: float = 360.0            # likewise for the wall clock


_COPY = Stage(
    "copy", "Make kernel.py copy x to out unchanged: read x through a dataflow buffer and write it to out through another. "
            "This stage only moves data; the check compares out with x.", "stage_copy", inputs=("x",))

def _share_budget(stages: tuple) -> tuple:
    """The stages with the whole-kernel attempt's turns and wall clock divided evenly over them (turns rounded down), so a
    staged attempt cannot use more turns or time than a whole-kernel one. A stage that finishes early does not pass its
    unused budget on."""
    n = len(stages)
    return tuple(Stage(**{**s.__dict__, "max_turns": WHOLE_KERNEL_TURNS // n, "wall_s": WHOLE_KERNEL_WALL_S / n})
                 for s in stages)


_TABLE: dict[str, tuple[Stage, ...]] = {
    "center": (
        _COPY,
        Stage("scale", "Multiply every element by 1/256 on the way through, so out = x / 256. Same output shape as before.",
              "stage_scale_by_inverse_width", inputs=("x",)),
        Stage("row_mean", "Now reduce. out has a new, smaller shape (see kernel_spec.json). out[i, 0] is the mean of row i "
                          "of x over its 256 columns, and the other 31 columns of out are 0.",
              "stage_row_mean", inputs=("x",), output=(64, 32)),
        Stage("center", "The whole op: out = x - mean(x over the last axis), with the output the shape of x again. "
                        "The mean of each row has to be subtracted from every element of that same row of x.", None),
    ),
    "rmsnorm": (
        _COPY,
        Stage("square", "Square every element on the way through, so out = x * x. Same output shape as before.",
              "stage_square", inputs=("x",)),
        Stage("row_sum_of_squares", "Now reduce. out has a new, smaller shape (see kernel_spec.json). out[i, 0] is the sum "
                                    "over the 256 columns of x[i, :] squared, and the other 31 columns of out are 0.",
              "stage_row_sum_of_squares", inputs=("x",), output=(64, 32)),
        Stage("scale_rows", "Bring the row value back to x: out = x * rsqrt(mean(x^2 over the last axis) + 1e-6), so each "
                            "row of x is scaled by its own value. out has the shape of x again.",
              "stage_scale_rows_by_inverse_rms", inputs=("x",)),
        Stage("rmsnorm", "The whole op: out = x * rsqrt(mean(x^2 over the last axis) + 1e-6) * w. w is a second input with "
                         "the shape of x (see kernel_spec.json), so the operation now takes x, w and out.", None),
    ),
}
STAGES: dict[str, tuple[Stage, ...]] = {op: _share_budget(stages) for op, stages in _TABLE.items()}


# ---- the files a stage runs against ----------------------------------------------------------------------------------

def stage_spec(op: str, stage: Stage) -> dict:
    """The op's spec with the stage's reference, inputs and output. The bars, the seeds and the dtype are the op's. The last
    stage (reference None) is the op's spec exactly. The `op` field names the stage ("rmsnorm.square") so a check result
    says which spec it judged."""
    spec = kp.load_spec(op)
    if stage.reference is None and stage.inputs is None and stage.output is None:
        return spec
    spec = dict(spec)
    spec["op"] = f"{op}.{stage.name}"
    spec["reference"] = stage.reference or spec["reference"]
    if stage.inputs is not None:
        spec["inputs"] = {name: list(spec["inputs"][name]) for name in stage.inputs}
    if stage.output is not None:
        spec["output"] = list(stage.output)
    spec["description"] = f"stage {stage.name} of {op}: {stage.instruction}"
    return spec


def write_stage_templates(tdir, op: str, stage: Stage) -> Path:
    """Make `tdir` hold what the shipped check needs for this stage: kernel_check.py and kernel_harness.py (byte for byte
    the shipped files), references.py (the shipped one plus the stage references), staged_references.py and
    ops/<op>.json (the stage spec). The last stage gets the shipped files only. Returns `tdir`."""
    tdir = Path(tdir)
    kp._real_dir(tdir.parent)
    kp._fresh_dir(tdir)
    (tdir / "ops").mkdir()
    for name in ("kernel_check.py", "kernel_harness.py"):
        shutil.copy2(kp.TEMPLATES / name, tdir / name)
    refs = (kp.TEMPLATES / "references.py").read_text(encoding="utf-8")
    if stage.reference is not None:
        shutil.copy2(kp.TEMPLATES / "staged_references.py", tdir / "staged_references.py")
        refs += ("\n\n# Added by the staged probe (orchard/kernelstaged.py): the references of the intermediate stages.\n"
                 "from staged_references import STAGE_REFERENCES\nREFERENCES.update(STAGE_REFERENCES)\n")
    (tdir / "references.py").write_text(refs, encoding="utf-8")
    (tdir / "ops" / f"{op}.json").write_text(json.dumps(stage_spec(op, stage), indent=2) + "\n", encoding="utf-8")
    return tdir


def _install_in_attempt(adir: Path, tdir: Path, op: str) -> None:
    """Put the stage's spec, check and references where the coder's own `python3 kernel_check.py` reads them. A check.json
    from the previous stage is removed, so nothing of it is read for this stage."""
    for name in ("kernel_check.py", "kernel_harness.py", "references.py", "staged_references.py"):
        target = adir / name
        if (tdir / name).is_file():
            shutil.copy2(tdir / name, target)
        elif target.exists() and name == "staged_references.py":
            target.unlink()
    shutil.copy2(tdir / "ops" / f"{op}.json", adir / "kernel_spec.json")
    check = adir / "check.json"
    if check.exists() or check.is_symlink():
        check.unlink()


# ---- the attempt -----------------------------------------------------------------------------------------------------

def _user_message(stage: Stage, index: int, total: int, op: str) -> str:
    start = ("Write kernel.py" if index == 1 else "Change kernel.py, which passed the previous stage,")
    return (f"Stage {index} of {total} of building the op `{op}`: {stage.name}. {stage.instruction}\n"
            f"{start} for this stage only. kernel_spec.json and kernel_check.py describe this stage. "
            "Start by reading kernel_spec.json" + (" and reference/eltwise_add.py." if index == 1 else " and kernel.py."))


def _stage_record(index: int, stage: Stage, outcome, wall_s: float, check: dict) -> dict:
    return {"index": index, "name": stage.name, "passed": check.get("passed") is True,
            "category": None if check.get("passed") is True else check.get("category"),
            "first_line": "" if check.get("passed") is True else (check.get("first_line") or ""),
            "status": outcome.status, "turns": outcome.turns, "agent_wall_s": wall_s,
            "kernel_sha256": check.get("kernel_sha256")}


def run_attempt(*, op, n, probe_dir, env, endpoint, model, coder_model, probe_id, http=post_json, clock=time.time,
                watchdog=None, run_watched=kp.run_watched, tt_lang: dict | None = None, temperature: float = 0.0,
                top_p: float | None = None, seed: int | None = None, references: list[dict] | None = None,
                skills: dict | None = None, coder_speed: dict | None = None, stage_turns: int | None = None,
                stage_wall_s: float | None = None, **ignored) -> dict:
    """One staged attempt (see the module docstring). Returns the attempt record, saved as attempt.json, with the fields
    kernelprobe.run_attempt writes and one more, `staged`.

    The record's pass or fail, category and first line are those of the LAST check that ran: the last stage's check, which is
    the whole op's, when every stage passed, else the failing stage's. `turns` and `agent_wall_s` are sums over the stages.
    `stage_turns` and `stage_wall_s` replace every stage's budget from the table when given. The arguments `max_turns`,
    `wall_s`, `hw_policy`, `shared_python` and `hw_status` of the whole-kernel attempt are accepted and not used: a staged
    attempt has per-stage budgets and no hardware tool."""
    if op not in STAGES:
        raise ValueError(f"no stage table for op {op!r}; the staged ops are {', '.join(STAGES)}")
    stages = STAGES[op]
    sampling = {"temperature": temperature, "top_p": top_p, "seed": seed}
    adir = Path(probe_dir) / op / f"attempt-{n}"
    facts = tt_lang if tt_lang is not None else kp.tt_lang_facts(env)
    copied = kp.prepare_attempt(adir, op=op, env=env, tt_lang=facts, references=references)
    ledger = Ledger(adir / "ledger.jsonl")
    system = kp.SKILL.read_text(encoding="utf-8") + "\n\n" + STAGED_NOTE.read_text(encoding="utf-8")
    records: list[dict] = []
    last_outcome, last_check, last_detail = None, {}, ""
    verified: bytes | None = None
    started = time.monotonic()
    swept_all: list = []
    try:
        for index, stage in enumerate(stages, start=1):
            final = index == len(stages)
            tdir = write_stage_templates(Path(probe_dir) / STAGED_DIR / f"{op}-{n}-s{index}", op, stage)
            _install_in_attempt(adir, tdir, op)
            if verified is not None:
                # The verified prefix: the bytes stage index-1's check passed are the starting file of this stage.
                kernel = adir / "kernel.py"
                if kernel.is_symlink() or kernel.exists():
                    kernel.unlink()
                kp._write_new(kernel, verified)
            turns = stage_turns if stage_turns is not None else stage.max_turns
            wall = stage_wall_s if stage_wall_s is not None else stage.wall_s
            control = kp.ProbeControl(wall_s=wall)
            dog = watchdog if watchdog is not None else kp.build_watchdog(control, ledger)
            step = AgentStep(agent=kp.AGENT, endpoint=endpoint, model=model, tools=kp.attempt_tools(adir), ledger=ledger,
                             stage=index, phase="probe", feed=dog.feed, control=control, run_dir=adir,
                             evidence_dir=adir / "evidence", log_path=adir / "log" / f"probe-{index:05d}.jsonl",
                             http=http, clock=clock, guard=RetryGuard(), max_turns=turns, temperature=temperature,
                             top_p=top_p, seed=seed, max_tokens=kp.BUDGET["max_tokens"])
            stage_started = time.monotonic()
            try:
                outcome = step.run(system, _user_message(stage, index, len(stages), op))
            finally:
                swept_all += kp.sweep_strays(adir)
            stage_wall = round(time.monotonic() - stage_started, 2)
            vdir = None if final else kp.verdict_dir(probe_dir, op, f"{n}-s{index}")
            try:
                check = kp.authoritative_check(adir, op=op, env=env, run_watched=run_watched, tt_lang=facts, vdir=vdir,
                                               templates=None if final else tdir)
            finally:
                swept_all += kp.sweep_strays(adir)
            records.append(_stage_record(index, stage, outcome, stage_wall, check))
            last_outcome, last_check = outcome, check
            last_detail = outcome.detail or control.detail or ""
            if check.get("passed") is not True:
                break
            clean = (vdir if vdir is not None else kp._verdict_dir_of(adir)) / kp.CLEAN_DIR / "kernel.py"
            verified = kp._read_regular(clean, kp.MAX_KERNEL_BYTES)
    finally:
        ledger.close()
    agent_wall_s = round(time.monotonic() - started, 2)
    passed_n = sum(1 for r in records if r["passed"])
    failed = next((r for r in records if not r["passed"]), None)
    check = last_check
    sets = check.get("sets") if isinstance(check.get("sets"), list) else []
    measures = kp.set_measures(sets)
    passed = failed is None
    staged = {"stage_names": [s.name for s in stages], "stages": records, "stages_passed": passed_n,
              "reached": len(records), "failed_stage": failed["name"] if failed else None,
              "budgets": [{"name": s.name, "max_turns": stage_turns if stage_turns is not None else s.max_turns,
                           "wall_s": stage_wall_s if stage_wall_s is not None else s.wall_s} for s in stages]}
    detail = last_detail
    if failed is not None:
        detail = f"failed at stage {failed['index']} of {len(stages)} ({failed['name']})" + (f": {detail}" if detail else "")
    attempt = {
        "op": op, "attempt": n, "status": last_outcome.status if last_outcome else "error", "status_detail": detail,
        "turns": sum(r["turns"] or 0 for r in records), "agent_wall_s": agent_wall_s, "passed_sim": passed,
        "category": None if passed else check.get("category"), "first_line": check.get("first_line") or "",
        **measures, "violations": check.get("violations") or [], "sets": sets, "hw": None,
        # The agent's own check.json is judged against the shipped hashes of the whole op's files, so it can be compared
        # only on the last stage. An earlier stage's file never counts as untrusted here.
        "agent_check_untrusted": (kp.agent_check_untrusted(kp._read_agent_check(adir / "check.json"), check, op)
                                  if passed else False),
        "swept": swept_all, "sampling": dict(sampling), "coder_speed": coder_speed,
        "agent_hw": {"enabled": False}, "staged": staged,
        "provenance": kp.provenance(kernel_sha256=check.get("kernel_sha256"), tt_lang=facts, coder_model=coder_model,
                                    probe_id=probe_id, simulator_pcc_min=measures["pcc_min"], sampling=dict(sampling),
                                    references=copied or None, skills=skills),
    }
    kp._save_attempt(adir, attempt)
    return attempt


# ---- the report ------------------------------------------------------------------------------------------------------

def failure_class(category, first_line: str) -> str:
    """The failure class of a check result: the check's category, with `tt_lang_error` split into `deadlock` and
    `dataflow step` (the simulator's "not a valid next dataflow step") when the first line says so. "passed" for no category."""
    if category is None:
        return "passed"
    text = first_line or ""
    if category == "tt_lang_error":
        if "Deadlock" in text:
            return "deadlock"
        if "not a valid next dataflow step" in text:
            return "dataflow step"
    return category


def _staged_rows(rows: list[dict], op: str) -> list[dict]:
    return [r for r in rows if r.get("op") == op and isinstance(r.get("staged"), dict)]


def reached_histogram(rows: list[dict], op: str) -> dict:
    """{stage name: attempts whose last started stage was that stage}, in stage order. An attempt that passed everything
    counts under the last stage."""
    names = [s.name for s in STAGES[op]]
    hist = {name: 0 for name in names}
    for r in _staged_rows(rows, op):
        k = r["staged"].get("reached")
        if isinstance(k, int) and 1 <= k <= len(names):
            hist[names[k - 1]] += 1
    return hist


def failure_classes_by_stage(rows: list[dict], op: str) -> dict:
    """{stage name: {failure class: count}} over the attempts that failed at that stage."""
    out = {s.name: {} for s in STAGES[op]}
    for r in _staged_rows(rows, op):
        name = r["staged"].get("failed_stage")
        if name in out:
            cls = failure_class(r.get("category"), r.get("first_line") or "")
            out[name][cls] = out[name].get(cls, 0) + 1
    return out


def report_lines(rows: list[dict], op: str) -> list[str]:
    """The staged section of the report: pass count, then one row per stage with how many attempts ended there (reached),
    how many failed there and the failure classes of those. Counts only."""
    mine = _staged_rows(rows, op)
    passed = sum(1 for r in mine if r.get("passed_sim"))
    hist, classes = reached_histogram(rows, op), failure_classes_by_stage(rows, op)
    lines = [f"## Staged construction: {op}", "",
             f"Attempts: {len(mine)}. Passed the whole op in the simulator: passed {passed} of {len(mine)}.", "",
             "| stage | attempts that ended here | failed here | failure classes |", "| --- | --- | --- | --- |"]
    for name in hist:
        fails = sum(classes[name].values())
        text = ", ".join(f"{c} {k}" for c, k in sorted(classes[name].items())) or "none"
        lines.append(f"| {name} | {hist[name]} | {fails} | {text} |")
    return lines


def _attempt_table(rows: list[dict], op: str) -> list[str]:
    lines = ["", "| attempt | seed | stages passed | reached | failed at | result | turns | wall s | decode tok/s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(_staged_rows(rows, op), key=lambda r: r["attempt"]):
        st, sp = r["staged"], r.get("sampling") or {}
        speed = (r.get("coder_speed") or {}).get("tok_s")
        result = "passed" if r.get("passed_sim") else failure_class(r.get("category"), r.get("first_line") or "")
        lines.append(f"| {r['attempt']} | {sp.get('seed')} | {st.get('stages_passed')} of {len(st.get('stage_names') or [])} "
                     f"| {st.get('reached')} | {st.get('failed_stage') or '-'} | {result} | {r.get('turns')} "
                     f"| {r.get('agent_wall_s')} | {round(speed, 1) if isinstance(speed, (int, float)) else '-'} |")
    return lines


def write_staged_report(probe_dir, rows: list[dict], op: str) -> Path:
    """Write DIR/kernel-probe-staged-report.md and return its path."""
    path = Path(probe_dir) / "kernel-probe-staged-report.md"
    path.write_text("\n".join(report_lines(rows, op) + _attempt_table(rows, op)) + "\n", encoding="utf-8")
    return path
