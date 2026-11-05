# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 433, review of PR 444 (limit round): a refusal reason is bounded, whatever the report holds.

`problems` formats values from the report into its reasons (`matches`, `top1_agreement`, `n_tokens`, each position of `mismatches`), and the
supervisor writes the reasons verbatim into a ledger entry and into the continuation message. A malformed report whose `mismatches` holds one
9.4 MB string made an 18.9 MB feedback entry: larger than the ledger's 16 MiB read limit, so every later read of the ledger failed. Each value
shown is now cut with a count of what was cut, a list names its first few members and counts the rest, each reason has a cap, and the
supervisor bounds the number and size of the reasons it writes for every gate. The field names and the recomputed values stay in the text.

The hostile inputs are built and checked in a child process with a 2 GiB address-space limit and a deadline."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import token_check
from orchard.ledger import Ledger
from orchard.supervisor import Supervisor, bound_reasons

ROOT = Path(__file__).resolve().parent.parent

PREAMBLE = f"""
import json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
sys.path.insert(0, {str(ROOT)!r})
from orchard import token_check
REF = [1000 + 7 * i for i in range(32)]
def report(**changes):
    forced = [{{"position": k, "first_id": REF[k]}} for k in range(32)]
    r = {{"n_tokens": 32, "matches": 32, "top1_agreement": 1.0, "reference_generated_ids": list(REF), "teacher_forced": forced,
         "mismatches": [], "result_draft": {{"n_tokens": 32, "top1_agreement": 1.0}}}}
    r.update(changes)
    return r
def check(r, results=()):
    return token_check.problems(r, list(results), list(REF), "stage 2: ", required_positions=32)
def summary(reasons):
    print(json.dumps({{"count": len(reasons), "longest": max((len(x) for x in reasons), default=0), "total": sum(len(x) for x in reasons),
                      "reasons": [x[:400] for x in reasons[:6]]}}))
"""


def child(body, timeout=90):
    done = subprocess.run([sys.executable, "-I", "-c", PREAMBLE + body], capture_output=True, text=True, timeout=timeout)
    assert done.returncode == 0, done.stderr[-600:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_one_huge_position_string_gives_a_bounded_reason_that_says_how_much_was_cut():
    out = child("""
big = "x" * 20_000_000
summary(check(report(mismatches=[{"position": big}])))
""")
    assert out["total"] < 6000 and out["longest"] <= token_check.MAX_REASON_CHARS + 200
    text = " ".join(out["reasons"])
    assert "mismatches" in text and "19999920 more characters" in text


def test_a_million_odd_positions_name_a_few_and_count_the_rest_quickly():
    out = child("""
import time
t = time.time()
r = report(mismatches=[{"position": f"p{i}"} for i in range(1_000_000)])
reasons = check(r)
summary(reasons)
assert time.time() - t < 60
""")
    assert out["total"] < 6000
    assert "more" in " ".join(out["reasons"]) and "999992 more" in " ".join(out["reasons"])


def test_a_million_whole_positions_that_are_not_mismatches_are_counted_not_listed():
    out = child("""
summary(check(report(mismatches=[{"position": i} for i in range(100, 1_000_100)])))
""")
    assert out["total"] < 6000 and "more" in " ".join(out["reasons"])


@pytest.mark.parametrize("field", ["matches", "top1_agreement", "n_tokens"])
def test_a_huge_value_in_a_numeric_field_is_cut_in_the_report_the_draft_and_the_result(field):
    out = child(f"""
big = "y" * 20_000_000
r = report({field}=big)
r["result_draft"] = dict(r["result_draft"], {field}=big)
summary(check(r, [("result.json", {{{field!r}: big, "n_tokens": 32, "top1_agreement": 1.0}})]))
""")
    assert out["total"] < 6000 and out["count"] >= 1
    text = " ".join(out["reasons"])
    assert field in text and "more characters" in text
    if field in ("matches", "top1_agreement"):
        assert "32 of 32 positions match" in text                      # the recomputed value stays in the reason


def test_a_deep_or_wide_value_is_named_by_its_shape_not_copied():
    out = child("""
wide = list(range(2_000_000))
summary(check(report(matches=wide, mismatches=[{"position": wide}])))
""")
    text = " ".join(out["reasons"])
    assert out["total"] < 6000 and "a list of 2000000 items" in text


def test_an_integer_of_thousands_of_digits_does_not_raise_or_fill_the_reason():
    out = child("""
import sys
sys.set_int_max_str_digits(0)                # a value the report loader could never hand over, but the formatter must not depend on that
huge = 10 ** 100000
summary(check(report(matches=huge)))
""")
    assert out["total"] < 6000 and "digits" in " ".join(out["reasons"])


def test_the_reason_still_names_the_field_and_the_recomputed_values_for_an_ordinary_report():
    forced = [{"position": k, "first_id": 1000 + 7 * k + (1 if k in (3, 9) else 0)} for k in range(32)]
    r = {"n_tokens": 32, "matches": 32, "top1_agreement": 1.0, "reference_generated_ids": [1000 + 7 * i for i in range(32)],
         "teacher_forced": forced, "mismatches": [], "result_draft": {"n_tokens": 32, "top1_agreement": 1.0}}
    reasons = token_check.problems(r, [], [1000 + 7 * i for i in range(32)], "stage 2: ", required_positions=32)
    text = " ".join(reasons)
    assert "30 of 32 positions match" in text and "matches 32" in text and "leaves out positions 3, 9" in text


# ---- the supervisor bounds what it writes for every gate -----------------------------------------------------------------------------

def test_bound_reasons_keeps_a_short_list_unchanged():
    assert bound_reasons(["a", "b"]) == ["a", "b"]


def test_bound_reasons_cuts_each_reason_and_the_count_and_says_what_it_cut():
    big = "z" * 20_000_000
    out = bound_reasons([big] * 100)
    assert sum(len(x) for x in out) < 40_000 and len(out) <= token_check.MAX_REASONS_WRITTEN + 1
    assert any("more characters" in x for x in out[:-1])
    assert "more reasons" in out[-1] and "are not shown" in out[-1] and "100 reasons in all" in out[-1]


def test_bound_reasons_does_not_break_a_non_text_reason():
    out = bound_reasons([None, 5, ["a"], "ok"])
    assert all(isinstance(x, str) for x in out) and out[-1] == "ok"


def test_the_gate_feedback_entry_and_message_are_bounded_and_the_ledger_reads_back(tmp_path):
    big = "q" * 20_000_000
    reasons = [f"swap-check.json mismatches {big}"] * 50
    ledger = Ledger(tmp_path / "ledger.jsonl")
    seen = {}
    step = SimpleNamespace(phase="finish", continue_with=lambda text, **kw: seen.setdefault("text", text) or "continued")
    stub = SimpleNamespace(ledger=ledger)
    spec = SimpleNamespace(number=2, gate_file="result.json")
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    Supervisor._gate_feedback(stub, spec, stage_dir, step, SimpleNamespace(reasons=tuple(reasons)))
    assert len(seen["text"]) < 40_000
    assert (tmp_path / "ledger.jsonl").stat().st_size < 100_000
    assert any(e["event"] == "decision" and e["data"].get("decision") == "gate feedback" for e in ledger.read())


def test_a_reason_is_cut_at_the_cap_whatever_makes_it_long():
    """The values a reason shows are bounded one by one; the cap on the whole reason is the net under any piece that is not (here the caller's
    label)."""
    out = child("""
reasons = token_check.problems(report(matches=7), [], list(REF), "W" * 5_000_000, required_positions=32)
summary(reasons)
""")
    assert out["longest"] <= token_check.MAX_REASON_CHARS + 100


def test_the_stage_end_entries_a_failed_supervisor_stage_writes_are_bounded(tmp_path):
    """`_end_stage_body` writes the reasons into `stage_end` and a pause reason; it joined them before bounding, so 50 reasons of 20 MB built a
    1 GB string first."""
    big = "m" * 20_000_000
    ledger = Ledger(tmp_path / "ledger.jsonl")
    stub = SimpleNamespace(ledger=ledger, run_dir=tmp_path)
    spec = SimpleNamespace(number=2, name="functional decoder", harness=True, draft=None)      # a StageSpec always has `draft` (the phone-a-friend end-of-attempt report reads it)
    result = Supervisor._end_stage_body(stub, spec, tmp_path, "fail", [big] * 50, SimpleNamespace(evidence=()), False)
    assert result == "fail"
    assert (tmp_path / "ledger.jsonl").stat().st_size < 100_000
    entries = ledger.read()
    ends = [e for e in entries if e["event"] == "stage_end"]
    assert ends and len(ends[-1]["data"]["reasons"]) <= token_check.MAX_REASONS_WRITTEN + 1
    assert any("more reasons" in r for r in ends[-1]["data"]["reasons"])
