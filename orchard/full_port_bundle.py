# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The operator bundle's part for stage 6 of a full port (issue 329): a section of RESULTS.md and the lines of RISKS.md. The bundle builder
(orchard/skills/operator-bundle-templates/build_bundle.py) calls `section` and `risks` with the parsed `stages/6/result.json`.

Stages 5 and 6 of a full port already appear in the builder's per-stage tables (`field_table`): every scalar field with its label, and the `numbers` list
(name, value, unit, label, where a label is `measured` or `TODO`). This module adds what a table of scalars cannot show for stage 6: the benchmark
as a spread with its label and warm-up count, the floor with its inputs, how each quality figure is defined, and the limits. Every number printed
here is read from the result file; a field that is missing or has the wrong type prints TODO, so a malformed file never prints a number that
nothing recorded. The result file was written by the supervisor and checked by the stage gate (orchard/full_port_quality.py); the builder reads it
again as untrusted text and values.
"""
from __future__ import annotations

import math

from orchard import full_port_quality as fq

HEADING = "Full port: quality against the reference and benchmark"


def _num(x) -> bool:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:         # a JSON integer too large for a float is not a number the bundle can print (review of PR 491, round 6)
        return False


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _fmt(x, digits: int = 4) -> str:
    """A number with its label, or TODO when the value is not a finite number."""
    return f"{x:.{digits}g} (measured)" if _num(x) else "TODO"


def _datasheet(x, digits: int = 4) -> str:
    """A fixed input taken from the board's datasheet. No bandwidth is measured in this stage, so it is never labelled measured."""
    return f"{x:.{digits}g} (datasheet value)" if _num(x) else "TODO"


TEXT_READ_MAX = 20_000        # a record is believed once its hash matches the ledger, so how much of one text is read is bounded here too


def _text(redact, x, limit: int = 400) -> str:
    """`x` as one line: the text is redacted whole first, then its whitespace is collapsed, then it is cut to `limit`. Cutting first can leave a
    credential prefix that no pattern matches (review of PR 491; build_bundle.py's `_record_text` has the same order)."""
    if not (isinstance(x, str) and x.strip()):
        return "not recorded"
    return " ".join(redact(x[:TEXT_READ_MAX]).split())[:limit]


def is_full_port_result(res) -> bool:
    return isinstance(res, dict) and res.get("schema") == fq.RESULT_SCHEMA


def section(res, redact) -> list[str]:
    """The RESULTS.md section for a stage 6 `result.json` written by the full-port quality stage. No lines for any other file."""
    if not is_full_port_result(res):
        return []
    fr, pp, bm = _d(res.get("free_run")), _d(res.get("perplexity")), _d(res.get("benchmark"))
    dec, pre, fl = _d(bm.get("decode_tok_s")), _d(bm.get("prefill_s")), _d(bm.get("floor"))
    inputs = _d(fl.get("inputs"))
    ev = "`stages/6/result.json`, `stages/6/evidence/port-quality.json`"
    lines = ["", f"## {HEADING}", "",
             f"The port is the one that passed stage {_fmt(res.get('source_stage'), 3).replace(' (measured)', '')}, run on {_fmt(res.get('chips'), 3).replace(' (measured)', '')} chips by the "
             f"supervisor's own template. Evidence: {ev}. Each figure is recomputed by the stage gate from the record. A figure is labelled `measured` "
             "or `TODO`.", "",
             "| Quality figure | Value | Label |", "|---|---|---|",
             f"| Free run: ids equal to the reference before the first difference | {_fmt(fr.get('leading_match'))} of {_fmt(fr.get('compared'))} | "
             f"{'measured' if _num(fr.get('leading_match')) else 'TODO'} |",
             f"| Teacher-forced perplexity of the reference text | {_fmt(pp.get('perplexity'))} | {'measured' if _num(pp.get('perplexity')) else 'TODO'} |",
             f"| Teacher-forced top-1 agreement with the reference | {_fmt(pp.get('top1_agreement'))} | {'measured' if _num(pp.get('top1_agreement')) else 'TODO'} |",
             f"| Top-5 logit gap difference from the reference (bar {fq.GAP_MAX}) | {_fmt(pp.get('top5_gap_diff'))} | {'measured' if _num(pp.get('top5_gap_diff')) else 'TODO'} |",
             "", f"Free-run text readable by the supervisor's floor: {'yes' if res.get('coherent') is True else 'no'}. "
             f"{_text(redact, res.get('coherent_basis'))}.", ""]
    lines += [f"### Benchmark ({_text(redact, bm.get('label'), 40)})", "",
              f"{_fmt(bm.get('warmup_runs'), 3).replace(' (measured)', '')} discarded warm-up run(s) and {_fmt(bm.get('repeats'), 3).replace(' (measured)', '')} measured runs "
              f"in one process, each one prefill and {_fmt(bm.get('steps'), 3).replace(' (measured)', '')} decode steps at batch size one. "
              "Spread is minimum, median and maximum.", "",
              "| Figure | Minimum | Median | Maximum | Label |", "|---|---|---|---|---|",
              f"| Decode speed (tok/s) | {_fmt(dec.get('min'))} | {_fmt(dec.get('median'))} | {_fmt(dec.get('max'))} | {'measured' if _num(dec.get('median')) else 'TODO'} |",
              f"| Prefill time (s) | {_fmt(pre.get('min'))} | {_fmt(pre.get('median'))} | {_fmt(pre.get('max'))} | {'measured' if _num(pre.get('median')) else 'TODO'} |",
              "", f"Model load time: {_fmt(bm.get('load_s'))} s.", "",
              "### Share of the decode floor", "",
              f"The share is the median decode speed over the floor's speed: {_fmt(bm.get('share_of_floor_median'))} "
              f"({_text(redact, bm.get('share_text'), 600)}). The speed is measured and the floor is an estimate "
              f"(orchard/decode_floor.py, {redact(str(inputs.get('board', 'board not recorded')))} at {_datasheet(inputs.get('per_die_bandwidth_gb_s'))} GB/s per die "
              f"over {_fmt(inputs.get('dies'), 3).replace(' (measured)', '')} dies). Weight dtype policy: "
              f"{redact(', '.join(f'{k} {v}' for k, v in sorted(_d(bm.get('dtype_policy')).items())) or 'not recorded')} "
              f"({_text(redact, bm.get('dtype_policy_source'))}).", ""]
    todo = [n for n in res.get("numbers", []) if isinstance(n, dict) and n.get("label") == "TODO"] if isinstance(res.get("numbers"), list) else []
    if todo:
        lines += ["### Not measured", ""] + [f"- {_text(redact, n.get('name'), 200)}: TODO." for n in todo] + [""]
    lines += ["### What these numbers do not show", ""]
    limits = res.get("limits") if isinstance(res.get("limits"), list) else []
    lines += [f"- {_text(redact, x, 500)}" for x in limits] or ["- No limits are recorded."]
    return lines


def risks(res, redact) -> list[str]:
    """The "Full port quality" lines of RISKS.md: the assumed dtype policy, a free run the rehearsal option let through, and the fixed limits."""
    if not is_full_port_result(res):
        return []
    bm = _d(res.get("benchmark"))
    out = ["## Full port quality", ""]
    if isinstance(bm.get("dtype_policy_source"), str) and bm["dtype_policy_source"].startswith("assumed"):
        out.append("- The decode floor assumes every weight is bf16, because the port declares no DTYPE_POLICY. A port that stores weights in a "
                   "smaller format has a higher floor than the one stated, so its share of the floor is overstated. Evidence: `stages/6/result.json`.")
    if res.get("coherent") is not True:
        out.append(f"- The free-run text did not pass the supervisor's readable floor, and the run's rehearsal option waived that field "
                   f"({_text(redact, res.get('coherent_basis'), 300)}). Evidence: `stages/6/result.json`.")
    out += ["- Quality and speed come from one prompt, one length and batch size one. Evidence: `stages/6/result.json`.",
            "- Perplexity is the port's perplexity of the reference's own greedy continuation, so it does not predict held-out text. Evidence: `stages/6/result.json`.",
            ""]
    return out
