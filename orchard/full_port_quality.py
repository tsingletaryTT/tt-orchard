# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 6 of a full port (issue 329, stage 6 half): the free-run quality check, the teacher-forced perplexity against the reference, and the
benchmark with the share of the decode floor. This module holds what the gate checks and what the supervisor writes into `result.json`.
It starts nothing, keeps no state, and imports nothing from orchard/stages.py (which imports it).

The files, all under `stages/6/`:

    port/                          the port that passed the earlier stage (stage 3 when it ran, else stage 2), copied by the supervisor.
                                   The ledger holds the sha256 of each copied file.
    port_quality.py                the shipped template (orchard/skills/full-port-templates/port_quality.py), copied by the supervisor.
    port_quality_config.json       the model snapshot, written by the supervisor.
    evidence/port-quality.json     the record the template writes (RECORD_SCHEMA).
    evidence/hw-test-output.txt    the command's output, written by the supervisor. The template prints `PORT_QUALITY_SHA256 <hex>` of the
                                   record as its last act.
    test-result.json               written by the supervisor after the command ends (exit code, timeout, chips, the output's sha256).
    result.json                    written by the supervisor from the record (`derive`). No agent writes it: every field is a copy of, or
                                   a number computed from, the record, and `check` recomputes all of them.

What each number is.

  * Free run. The port's own greedy ids from the stage 1 prompt, 64 tokens. `leading_match` is how many of them equal the reference's ids
    before the first difference. `coherent` is the supervisor's floor on the text (`full_port_result.readable`: printable, mostly words, no
    loops). The rehearsal option (`--full-port-rehearsal`) waives `coherent` and no other field of this stage.
  * Perplexity against the reference. The port is fed the reference's ids one by one (teacher forcing). The record holds the natural log of
    the probability the port gives the reference's next id at each position. `perplexity` is exp of the mean negative log probability, on
    the reference's own greedy continuation, so it is the port's perplexity of the reference text and not of held-out text. Stage 1 recorded
    the reference's top five logits at each position, so the comparison with the reference is made on those: the record holds the port's
    logits at the same five ids, and `top5_gap_diff` is the mean absolute difference between the two models' logit gaps
    (logit of the k-th id minus logit of the reference's first id, k = 2 to 5). A gap is unchanged by a shift of all logits, which a bf16 chip
    and a float32 CPU model do not share. The stage passes when the mean is at most GAP_MAX. The bar is a choice, not a measured noise floor.
  * Benchmark. WARMUP_RUNS discarded runs and at least REPEATS measured runs in one process; each is one prefill and BENCH_STEPS decode steps.
    Spread is min, median and max (`orchard/resident.py summarise_runs`), and the label is the one `resident.label_for` gives. The floor is
    `orchard/decode_floor.py` on the model's config.json, the port's declared dtype policy (or bf16 when it declares none, and the record
    says which) and the dies the test ran on, at the p300 datasheet bandwidth. The share is the median speed over the floor's speed. The
    floor is an estimate; the speed is measured.

Provenance, the same checks as stages 2 and 3: the stage's copy of the template is the shipped file and the record carries its sha256; the
reference ids, the prompt and the top-five file are the stage 1 files on disk (path and sha256); the supervisor's test exited 0 on the
chips the record names; the output carries the record's sha256 once; the port sources are the ones the ledger says were copied in, which are
the ones the earlier stage's record names. Like stage 2's, these tie the record to a test the supervisor ran and do not prove the chip
computed the ids.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

from orchard import decode_floor as df
from orchard import full_port, full_port_result, model_config as mc, resident, safeio

RECORD_SCHEMA = "orchard-port-quality/1"
RESULT_SCHEMA = "orchard-full-port-quality-result/1"
STAGE = 6
SOURCE_STAGES = (2, 3)                # the stage whose port stage 6 measures: 3 when it ran (the configured chips), else 2
RECORD_FILE = "evidence/port-quality.json"
OUTPUT_FILE = "evidence/hw-test-output.txt"
OUTPUT_MARK = "PORT_QUALITY_SHA256"
TEMPLATE = Path(__file__).resolve().parent / "skills" / "full-port-templates" / "port_quality.py"
TEMPLATE_NAME = "port_quality.py"
CONFIG_NAME = "port_quality_config.json"
COMMAND = "python3 -P -s stages/6/port_quality.py"
DEADLINE_S = 3600
PORT_DIR = "stages/6/port/"
BOARD = "p300"                         # the floor's board type (orchard/decode_floor.py BOARDS), as the text-runtime stage 6 uses
TOP1_MIN = full_port.TOP1_MIN          # choice: stage 2's bar, applied to the teacher-forced ids of this stage's own run
MIN_TOKENS = full_port.MIN_TOKENS
GAP_MAX = 0.5                          # choice: mean absolute difference of top-five logit gaps, in logits; a bf16 chip against a float32 CPU model
SHARE_MAX = 1.5                        # a share above this means an input is wrong (the text-runtime stage 6 uses the same bound)
MIN_BENCH_STEPS = 16                   # a benchmark run of fewer decode steps than this is not a decode speed
FREE_RUN_MIN = 32                      # the free run is at least as long as the reference
PROBES = 5
JSON_MAX = full_port.JSON_MAX
LIMITS = ("Quality and speed are measured on the one prompt stage 1 recorded; nothing here covers other prompts or long contexts.",
          "Perplexity is the port's perplexity of the reference's own greedy continuation (32 to 64 tokens), which is an easy path; it says how "
          "close the port's distribution is on that path and not how it does on held-out text.",
          "The decode speed is batch size one with one prompt. Each timed request includes a pipe round trip and an argmax.",
          "The floor is an estimate from the model's config at the datasheet DRAM bandwidth; it ignores the KV cache and activations.",
          "A port that returns the reference's ids without computing them cannot be told from one that computed them.")
TODO_NUMBERS = ("quality on prompts other than the stage 1 prompt", "decode speed at a batch size above one", "quality at a long context")


def command() -> str:
    """The only command the supervisor runs for this stage."""
    return COMMAND


def template_sha256() -> str | None:
    """The sha256 of the shipped template, or None when it cannot be read."""
    return _sha_file(TEMPLATE)


def _sha_file(path) -> str | None:
    try:
        return full_port._sha(path)
    except OSError:
        return None


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _numlist(x, n: int | None = None) -> bool:
    return isinstance(x, list) and (n is None or len(x) == n) and all(_finite(v) for v in x)


# ---- the record's arrays ------------------------------------------------------------------------------------------------------

def record_problems(record: dict) -> list[str]:
    """Why the record's arrays and benchmark runs cannot be summarised. No reasons means `summarise` can run on it."""
    out: list[str] = []
    ref, chip = record.get("reference_ids"), record.get("chip_ids")
    if not (full_port._ids(ref) and full_port._ids(chip)) or not chip or len(ref) != len(chip):
        return ["reference_ids and chip_ids must be non-empty lists of whole numbers of the same length"]
    n = len(chip)
    if n < MIN_TOKENS:
        out.append(f"the record compares {n} positions; at least {MIN_TOKENS} are needed")
    lps = record.get("logprobs")
    if not _numlist(lps, n) or any(v > 1e-9 for v in lps):
        out.append(f"logprobs must be {n} finite numbers at or below zero (the log probability of each reference id)")
    top5, probes = record.get("reference_top5"), record.get("probe_logits")
    if not (isinstance(top5, list) and len(top5) == n and all(
            isinstance(t, dict) and full_port._ids(t.get("ids")) and len(t["ids"]) == PROBES and _numlist(t.get("logits"), PROBES) for t in top5)):
        out.append(f"reference_top5 must hold {n} entries with {PROBES} ids and {PROBES} logits each")
    if not (isinstance(probes, list) and len(probes) == n and all(_numlist(p, PROBES) for p in probes)):
        out.append(f"probe_logits must hold {n} lists of {PROBES} finite numbers")
    free = record.get("free_run_ids")
    if not full_port._ids(free) or len(free) < FREE_RUN_MIN:
        out.append(f"free_run_ids must be a list of at least {FREE_RUN_MIN} whole numbers")
    elif free[0] != chip[0]:
        out.append(f"the port's second prefill of the same prompt gave id {free[0]} where the first gave {chip[0]}: prefill must reset the "
                   "model's state (the plugin reuses a slot for a new request and expects prefill to start it afresh)")
    bench = record.get("bench")
    runs = bench.get("runs") if isinstance(bench, dict) else None
    if not isinstance(runs, list) or not runs:
        out.append("bench.runs must be a non-empty list")
        return out
    if not any(isinstance(r, dict) and r.get("warmup") is False for r in runs):
        # `summarise` takes its steps from the first measured run, so a record of warm-up runs only has to be refused here (review of PR 491).
        out.append("bench.runs holds no measured run: every run is a warm-up")
        return out
    for i, r in enumerate(runs):
        if not (isinstance(r, dict) and type(r.get("warmup")) is bool and _finite(r.get("prefill_s")) and r["prefill_s"] > 0
                and _finite(r.get("decode_s")) and r["decode_s"] > 0 and type(r.get("steps")) is int and r["steps"] >= MIN_BENCH_STEPS
                and _finite(r.get("decode_tok_s"))):
            out.append(f"bench run {i} needs a warmup flag, a positive prefill and decode time, at least {MIN_BENCH_STEPS} steps and a decode speed")
        elif not math.isclose(r["steps"] / r["decode_s"], r["decode_tok_s"], rel_tol=1e-6):
            out.append(f"bench run {i}: its decode speed is not its steps over its decode time")
    return out


def _bench_runs(record: dict) -> tuple[list, list]:
    runs = record["bench"]["runs"]
    return [r for r in runs if r["warmup"]], [r for r in runs if not r["warmup"]]


def leading_match(reference: list, free: list) -> int:
    """How many of the free run's ids equal the reference's, counted from the start until the first difference."""
    k = 0
    for a, b in zip(reference, free):
        if a != b:
            break
        k += 1
    return k


def top5_gap_diff(top5: list, probes: list) -> float:
    """Mean absolute difference between the reference's top-five logit gaps and the port's, over every position and k = 2..5. The gap of the
    k-th id is its logit minus the logit of the reference's first id, in each model."""
    diffs = []
    for t, p in zip(top5, probes):
        r = t["logits"]
        for k in range(1, PROBES):
            diffs.append(abs((p[k] - p[0]) - (r[k] - r[0])))
    return sum(diffs) / len(diffs)


def summarise(record: dict, model_cfg: dict, source_stage: int) -> dict:
    """The numbers of a record that passed `record_problems`: a dict of the result's computed fields. Raises df.FloorInputError for a policy or
    config the floor cannot count."""
    ref, chip, lps = record["reference_ids"], record["chip_ids"], record["logprobs"]
    n = len(chip)
    nll = -sum(lps) / n
    ppl = math.exp(min(nll, 700.0))
    gap = top5_gap_diff(record["reference_top5"], record["probe_logits"])
    free = record["free_run_ids"]
    lead = leading_match(ref, free)
    warm, measured = _bench_runs(record)
    timings = [{k: r[k] for k in ("prefill_s", "decode_s", "decode_tok_s", "host_loadavg_1m") if k in r} for r in measured]   # the load average is not a part: resident lists it per run
    summary = resident.summarise_runs(timings, warmup=len(warm))
    chips = len(record["chips"])
    floor = df.floor_for_board(model_cfg, record["dtype_policy"], BOARD, dies=chips)
    parts = summary["parts"]
    median = parts["decode_tok_s"]["median"] if "decode_tok_s" in parts else None
    share = df.share_of_floor(median, floor) if median is not None else None
    agree = full_port.agreement(ref, chip)
    ok_text, basis = full_port_result.readable(record["free_run_text"])
    free_run = {"n_tokens": len(free), "leading_match": lead, "compared": min(len(ref), len(free)), "label": "measured"}
    perplexity = {"perplexity": ppl, "mean_nll": nll, "n_tokens": n, "top1_agreement": agree, "top5_gap_diff": gap, "label": "measured",
                  "basis": "the port's perplexity of the reference's own greedy continuation, teacher forced"}
    benchmark = {"label": summary["label"], "warmup_runs": len(warm), "repeats": summary["repeats"], "steps": measured[0]["steps"],
                 "decode_tok_s": parts.get("decode_tok_s"), "prefill_s": parts.get("prefill_s"), "load_s": record["load_s"],
                 "chips": chips, "dtype_policy": record["dtype_policy"], "dtype_policy_source": record["dtype_policy_source"],
                 "floor": floor.as_dict(), "share_of_floor_median": share,
                 "share_text": df.share_text(median, floor) if median is not None else None,
                 "host_loadavg_1m": summary["host_loadavg_1m"]}
    return {"source_stage": source_stage, "coherent": ok_text, "coherent_basis": f"supervisor floor (form only): {basis}",
            "free_run": free_run, "perplexity": perplexity, "benchmark": benchmark, "n_tokens": n, "chips": chips,
            "numbers": numbers(free_run, perplexity, benchmark)}


def numbers(free_run: dict, ppl: dict, bench: dict) -> list[dict]:
    """The `numbers` list the operator bundle prints: name, value, unit and label (measured or TODO) for each figure, plus a TODO line for
    each thing this stage does not measure. A figure that has no value is a TODO."""
    def row(name, value, unit):
        return {"name": name, "value": value, "unit": unit, "label": "measured" if _finite(value) else "TODO"}
    d, p = bench.get("decode_tok_s") or {}, bench.get("prefill_s") or {}
    rows = [row("free run: leading ids equal to the reference", free_run["leading_match"], "tokens"),
            row("teacher-forced perplexity of the reference text", ppl["perplexity"], ""),
            row("teacher-forced mean negative log probability", ppl["mean_nll"], "nats"),
            row("teacher-forced top-1 agreement with the reference", ppl["top1_agreement"], "fraction"),
            row("top-5 logit gap difference from the reference", ppl["top5_gap_diff"], "logits"),
            row("decode speed, median", d.get("median"), "tok/s"), row("decode speed, minimum", d.get("min"), "tok/s"),
            row("decode speed, maximum", d.get("max"), "tok/s"), row("prefill time, median", p.get("median"), "s"),
            row("share of the estimated decode floor, median speed", bench.get("share_of_floor_median"), "fraction of the floor"),
            row("model load time", bench.get("load_s"), "s")]
    rows += [{"name": t, "value": None, "unit": "", "label": "TODO"} for t in TODO_NUMBERS]
    return rows


def build_result(record: dict, model_cfg: dict, source_stage: int, evidence: list[str]) -> dict:
    """The whole result.json for a record (without `derived_by`)."""
    s = summarise(record, model_cfg, source_stage)
    return {"schema": RESULT_SCHEMA, "runs": True, "source_stage": s["source_stage"], "chips": s["chips"], "n_tokens": s["n_tokens"],
            "coherent": s["coherent"], "coherent_basis": s["coherent_basis"], "free_run_text": record["free_run_text"],
            "free_run": s["free_run"], "perplexity": s["perplexity"], "benchmark": s["benchmark"], "numbers": s["numbers"],
            "limits": list(LIMITS), "evidence": evidence}


def _model_config(stage_dir: Path, run_dir: Path, record: dict, reasons: list) -> dict | None:
    """config.json of the snapshot the stage's config names, when it is the file the record hashed."""
    cfg = full_port._json(Path(stage_dir) / CONFIG_NAME, CONFIG_NAME, reasons)
    snap = cfg.get("model_snapshot") if isinstance(cfg, dict) else None
    if not isinstance(snap, str) or not snap:
        reasons.append(f"{CONFIG_NAME} names no model snapshot")
        return None
    path = safeio.resolved(Path(snap) / "config.json")
    want = (record.get("model_config") or {}).get("sha256") if isinstance(record.get("model_config"), dict) else None
    if _sha_file(path) != want:
        reasons.append("the model's config.json is not the file the record hashed (missing, or changed after the test), so the floor is not of the "
                       "model that ran")
        return None
    try:
        return mc.read_model_config(path)
    except mc.ModelConfigError as exc:
        reasons.append(str(exc))
        return None


def evidence_paths(stage_dir, run_dir) -> list[str]:
    return [os.path.relpath(Path(stage_dir) / RECORD_FILE, run_dir), os.path.relpath(Path(stage_dir) / OUTPUT_FILE, run_dir)]


def _same(a, b) -> bool:
    import json
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def derive(stage_dir, run_dir, *, source_stage: int, rehearsal: bool = False) -> tuple[dict | None, str]:
    """(result, why): the content of `result.json` for a stage 6 directory whose hardware test passed and whose record is complete, or
    (None, why not). The supervisor runs `check` on what it wrote and deletes a result that does not pass. `rehearsal` only lets a text that
    fails the readable floor through with `coherent` false; `check` waives that field and nothing else."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    problems: list[str] = []
    marker = full_port._json(stage_dir / "test-result.json", "test-result.json", problems)
    if not isinstance(marker, dict):
        return None, "test-result.json is missing or unreadable"
    if not (type(marker.get("returncode")) is int and marker["returncode"] == 0 and marker.get("timed_out") is False):
        return None, "the hardware test did not pass"
    record = full_port._json(stage_dir / RECORD_FILE, RECORD_FILE, problems)
    if not isinstance(record, dict):
        return None, f"{RECORD_FILE} is missing or unreadable"
    if bad := record_problems(record):
        return None, bad[0]
    if not isinstance(record.get("free_run_text"), str) or not record["free_run_text"].strip():
        return None, "the record has no free-run text"
    cfg = _model_config(stage_dir, run_dir, record, problems)
    if cfg is None:
        return None, problems[-1] if problems else "the model config cannot be read"
    try:
        result = build_result(record, cfg, source_stage, evidence_paths(stage_dir, run_dir))
    except (df.FloorInputError, KeyError, TypeError, ValueError) as exc:
        return None, f"the record cannot be summarised: {exc}"
    if not result["coherent"] and not rehearsal:
        return None, f"the free-run text did not pass the readable floor ({result['coherent_basis']})"
    result["derived_by"] = "supervisor"
    return result, "derived"


# ---- the gate -----------------------------------------------------------------------------------------------------------------

def check(stage_dir, run_dir, *, rehearsal: bool = False, waived: list | None = None, expected_chips: int | None = None,
          seeded: list | None = None, source_stage: int = 2, accepted_record_sha: str | None = None) -> tuple[list[str], list[str]]:
    """(reasons, evidence paths) for a stage 6 directory of a full port. No reasons means the result passes.

    `expected_chips` is the chip count the test must have run on (the supervisor's, from the run's `--required-chips` when stage 3 ran, else
    one board); `seeded` is the list of {path, sha256} the ledger holds for the files the supervisor copied into `stages/6/port/`; `source_stage`
    is the stage the port came from; `accepted_record_sha` is the sha256 the ledger holds for that stage's port-check record among the evidence its
    gate accepted (a missing value is a refusal). All four are keywords the caller sets; nothing in the stage directory sets them. `rehearsal` waives the
    `coherent` field and nothing else; `waived` receives its name when it did."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    reasons: list[str] = []
    seen: list[str] = []
    if source_stage not in SOURCE_STAGES:
        return [f"source_stage must be one of {list(SOURCE_STAGES)}, got {source_stage!r}"], []
    result = full_port._json(stage_dir / "result.json", "result.json", reasons)
    if result is None:
        return reasons, seen
    marker = full_port._json(stage_dir / "test-result.json", "test-result.json", reasons)
    record = full_port._json(stage_dir / RECORD_FILE, RECORD_FILE, reasons)

    if isinstance(marker, dict):
        code, timed_out = marker.get("returncode"), marker.get("timed_out")
        if not (type(code) is int and code == 0 and timed_out is False):
            why = ("the hardware test ran past its deadline and was stopped" if timed_out else
                   "the hardware test has no exit code (it was refused or did not start)" if code is None else
                   f"the hardware test exited with code {code!r}")
            reasons.append(f"test-result.json: {why}")
        if marker.get("command") != command():
            reasons.append(f"test-result.json: the hardware test ran {marker.get('command')!r}; the supervisor's command for this stage is {command()!r}")
    if record is None:
        return reasons, seen
    if record.get("schema") != RECORD_SCHEMA:
        reasons.append(f"{RECORD_FILE} schema must be {RECORD_SCHEMA!r}, got {record.get('schema')!r}")
    listed = result.get("evidence")
    if not isinstance(listed, list) or not listed:
        reasons.append("result.json lists no evidence")
    else:
        for rel in listed:
            if full_port._inside(run_dir, rel) is None:
                reasons.append(f"result.json: evidence {rel!r} is not a file inside the run directory")
            else:
                seen.append(rel)
        for name in ("port-quality.json", "hw-test-output.txt"):
            if not any(isinstance(e, str) and Path(e).name == name for e in listed):
                reasons.append(f"result.json evidence must list {name}")

    # ---- the record against the stage 1 files on disk ----
    ref_ids = full_port._reference_ids(record, run_dir, reasons)
    _top5_reasons(record, run_dir, reasons)
    arrays = record_problems(record)
    reasons += arrays
    if not arrays and ref_ids is not None:
        if record["reference_ids"] != ref_ids[:len(record["chip_ids"])]:
            reasons.append("reference_ids are not the first ids of the stage 1 reference file: the comparison was made against ids the reference did not produce")
        agree = full_port.agreement(record["reference_ids"], record["chip_ids"])
        if agree < TOP1_MIN:
            reasons.append(f"top-1 agreement recomputed from the record is {agree:.4f}, below the minimum of {TOP1_MIN}")
        if not isinstance(record.get("free_run_text"), str):
            reasons.append("free_run_text must be text")

    # ---- the record against the supervisor's test ----
    if isinstance(marker, dict):
        rc, mcs = record.get("chips"), marker.get("chips")
        chip_lists = all(isinstance(c, list) and c and all(isinstance(x, str) for x in c) for c in (rc, mcs))
        if not chip_lists or sorted(rc) != sorted(mcs):
            reasons.append(f"{RECORD_FILE} chips {rc!r} are not the chips the supervisor leased for the test {mcs!r} "
                           "(both must be non-empty lists of chip names)")
        elif expected_chips is not None and (len(rc) != expected_chips or len(mcs) != expected_chips):
            reasons.append(f"the hardware test ran on {len(mcs)} chips and the record lists {len(rc)}; stage 6 measures the port on {expected_chips} "
                           f"(the chip count of stage {source_stage}, which the supervisor decides)")
        _output_reasons(stage_dir, run_dir, marker, reasons)
    _template_reasons(record, stage_dir, reasons)
    full_port._source_reasons(record, stage_dir, run_dir, reasons, STAGE)
    _seed_reasons(record, run_dir, seeded, source_stage, accepted_record_sha, reasons)

    # ---- result.json against the numbers the record gives ----
    if arrays or not isinstance(record.get("free_run_text"), str):
        return reasons, seen
    cfg = _model_config(stage_dir, run_dir, record, reasons)
    if cfg is None:
        return reasons, seen
    try:
        expected = build_result(record, cfg, source_stage, listed if isinstance(listed, list) else [])
    except (df.FloorInputError, KeyError, TypeError, ValueError) as exc:
        reasons.append(f"the record cannot be summarised: {exc}")
        return reasons, seen
    for key, want in expected.items():
        if key == "evidence":
            continue
        if key == "coherent":
            if result.get("coherent") is not want:
                reasons.append(f"result.json coherent is {result.get('coherent')!r}; the supervisor's floor on the record's free-run text gives {want!r}")
            elif want is not True and rehearsal:
                if waived is not None:
                    waived.append("coherent")
            elif want is not True:
                reasons.append(f"coherent must be true (the free-run text is readable): {expected['coherent_basis']}")
        elif not _same(result.get(key), want):
            reasons.append(f"result.json {key} is not what the record gives (the gate recomputed it)")
    bench, ppl = expected["benchmark"], expected["perplexity"]
    if bench["warmup_runs"] < resident.WARMUP_RUNS:
        reasons.append(f"{bench['warmup_runs']} warm-up run(s); the rule is {resident.WARMUP_RUNS}")
    if bench["repeats"] < resident.MIN_REPEATS:
        reasons.append(f"{bench['repeats']} measured run(s); the rule is {resident.MIN_REPEATS}")
    if bench["decode_tok_s"] is None:
        reasons.append("the runs report no decode speed")
    share = bench["share_of_floor_median"]
    if not _finite(share) or not 0 < share <= SHARE_MAX:
        reasons.append(f"the share of the decode floor ({share}) is not a share between 0 and {SHARE_MAX}")
    if ppl["top5_gap_diff"] > GAP_MAX:
        reasons.append(f"the port's top-5 logit gaps differ from the reference's by {ppl['top5_gap_diff']:.4f} on average; the bar is {GAP_MAX}")
    text = record["free_run_text"]
    if not full_port._text(text) or len("".join(text.split())) < full_port.MIN_FREE_RUN_CHARS or len(set(text.split())) < full_port.MIN_FREE_RUN_WORDS:
        reasons.append(f"free_run_text must be at least {full_port.MIN_FREE_RUN_CHARS} non-space characters and {full_port.MIN_FREE_RUN_WORDS} different words")
    return reasons, seen


def _top5_reasons(record: dict, run_dir: Path, reasons: list) -> None:
    """The record's reference top-five entries are the ones in the stage 1 file on disk, and the record names that file and its sha256."""
    entry = record.get("top5")
    want = "stages/1/evidence/reference/top5-logits.json"
    if not isinstance(entry, dict) or entry.get("path") != want:
        reasons.append(f"top5.path must be {want!r}, the stage 1 file, got {entry.get('path') if isinstance(entry, dict) else entry!r}")
        return
    f = full_port._inside(run_dir, want)
    if f is None:
        reasons.append(f"top5: {want} is not a file in the run directory (stage 1 wrote no top-five logits, so the comparison with the reference has nothing to compare with)")
        return
    if entry.get("sha256") != full_port._sha(f):
        reasons.append(f"top5.sha256 does not match {want} as it is on disk")
    try:
        data = safeio.read_json(f, max_bytes=JSON_MAX, expect=dict)
        steps = data.get("steps")
        got = record.get("reference_top5")
        if isinstance(got, list):
            for k, g in enumerate(got):
                top = steps[k]["top5"][:PROBES]
                if not isinstance(g, dict) or g.get("ids") != [int(t["id"]) for t in top] or g.get("logits") != [float(t["logit"]) for t in top]:
                    reasons.append(f"reference_top5[{k}] is not the top five in {want}")
                    break
    except (safeio.SafeReadError, KeyError, IndexError, TypeError, ValueError, AttributeError):
        reasons.append(f"{want} is not a readable list of steps with top-five logits for every compared position")


def _template_reasons(record: dict, stage_dir: Path, reasons: list) -> None:
    shipped = _sha_file(TEMPLATE)
    if shipped is None:
        reasons.append(f"the shipped template {TEMPLATE.name} cannot be read, so the record cannot be tied to it")
        return
    copy = stage_dir / TEMPLATE_NAME
    mine = _sha_file(copy) if copy.is_file() and not copy.is_symlink() else None
    if mine is None:
        reasons.append(f"{TEMPLATE_NAME} is not in this stage directory as a regular file: the supervisor copies it there and the gate compares it with the shipped one")
    elif mine != shipped:
        reasons.append(f"{TEMPLATE_NAME} in this stage differs from the shipped template (sha256 {mine[:12]} against {shipped[:12]}), so the record is not the template's")
    if record.get("template_sha256") != shipped:
        reasons.append(f"{RECORD_FILE} template_sha256 {str(record.get('template_sha256'))[:12]} is not the shipped template's ({shipped[:12]}): the record was not written by it")
    cfg = record.get("config")
    cfg_path = stage_dir / CONFIG_NAME
    if not isinstance(cfg, dict) or cfg.get("sha256") != (_sha_file(cfg_path) if cfg_path.is_file() else None):
        reasons.append(f"{CONFIG_NAME} is not the file the record names (missing, or changed after the test)")


def _output_reasons(stage_dir: Path, run_dir: Path, marker: dict, reasons: list) -> None:
    """The supervisor's output file is the file it recorded, and the command printed the record's sha256 into it once."""
    out = stage_dir / OUTPUT_FILE
    rec = marker.get("output")
    if not isinstance(rec, dict) or rec.get("path") != os.path.relpath(out, run_dir) or rec.get("sha256") != full_port._sha(out):
        reasons.append(f"{OUTPUT_FILE} is not the output file the supervisor recorded in test-result.json")
        return
    try:
        text = safeio.read_bytes(out, max_bytes=JSON_MAX * 4).decode("utf-8", "replace")
    except safeio.SafeReadError as exc:
        reasons.append(f"{OUTPUT_FILE} is not readable: {exc.reason}")
        return
    marks = re.findall(rf"^{OUTPUT_MARK} ([0-9a-f]{{64}})\s*$", text, re.M)
    starts = [line for line in text.splitlines() if line.startswith(OUTPUT_MARK)]
    if len(starts) != 1:
        reasons.append(f"{OUTPUT_FILE} has {len(starts)} lines that begin with {OUTPUT_MARK}; the template prints exactly one "
                       "(and prefixes every line the port prints), so something else wrote a line of its own")
    if not marks or marks[-1] != full_port._sha(stage_dir / RECORD_FILE):
        reasons.append(f"{OUTPUT_FILE} does not carry a {OUTPUT_MARK} line that matches {RECORD_FILE}, so the record was not written by the test the supervisor ran")


def _seed_reasons(record: dict, run_dir: Path, seeded: list | None, source_stage: int, accepted_record_sha: str | None, reasons: list) -> None:
    """The port that ran is the port that passed the earlier stage. The files the record names must be exactly the files the supervisor copied
    in (the ledger's hashes), and those must be the files the source stage's own record named (its hashes, read from its evidence)."""
    sources = record.get("port_sources")
    mine = {s["path"][len(PORT_DIR):]: s.get("sha256") for s in sources if isinstance(s, dict) and isinstance(s.get("path"), str)
            and s["path"].startswith(PORT_DIR)} if isinstance(sources, list) else {}
    if not isinstance(seeded, list) or not seeded:
        reasons.append("the ledger holds no record of the files copied into stages/6/port/, so the port cannot be tied to the stage that passed it")
        return
    copied = {s["path"][len(PORT_DIR):]: s.get("sha256") for s in seeded if isinstance(s, dict) and isinstance(s.get("path"), str) and s["path"].startswith(PORT_DIR)}
    if mine != {k: v for k, v in copied.items() if k.endswith(".py")}:
        reasons.append(f"the port files the record names are not the .py files the supervisor copied from stage {source_stage} (the ledger's hashes): "
                       "the port that was measured is not the port that passed")
    # The source record is trusted only when it is the file the source stage's gate accepted: its hash is in the ledger. Without this, a
    # record and a port replaced together after the source stage passed would be accepted as the port that passed (review of PR 491).
    src_path = run_dir / "stages" / str(source_stage) / full_port.RECORD_FILE
    if not isinstance(accepted_record_sha, str) or not accepted_record_sha:
        reasons.append(f"the ledger holds no sha256 for stage {source_stage}'s {full_port.RECORD_FILE} among the evidence its gate accepted, "
                       "so that record cannot be trusted to name the port that passed")
        return
    if full_port._sha(src_path) != accepted_record_sha:
        reasons.append(f"stage {source_stage}'s {full_port.RECORD_FILE} is not the file its gate accepted (the sha256 differs from the ledger's): "
                       "the record that names the port that passed has been replaced")
        return
    src_record = full_port._json(src_path, f"stage {source_stage} {full_port.RECORD_FILE}", [])
    src_prefix = full_port.port_dir(source_stage)
    theirs = {s["path"][len(src_prefix):]: s.get("sha256") for s in (src_record or {}).get("port_sources", [])
              if isinstance(s, dict) and isinstance(s.get("path"), str) and s["path"].startswith(src_prefix)}
    if not theirs or theirs != mine:
        reasons.append(f"the port files the record names are not the ones stage {source_stage}'s own record names: the port that was measured is not "
                       "the port that passed that stage")
