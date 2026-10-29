# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The scripts next to docs/run-logs/2026-10-28-staged-construction.md: the similarity check, the analysis tables and the
agent-loop harness (run here against a fake Telegraph; the real simulator checks the kernels, so those tests skip without it)."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import telegraph as tg
# ttlang_skill is an autouse fixture there: importing it puts a stand-in tt-lang skill directory behind ORCHARD_TTLANG_SKILL for every
# test in this file, so the loop tests do not need the operator's installed skill (it is the skill's absence that made them fail).
from test_kernelprobe import needs_sim, VENV, ttlang_skill  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "docs" / "run-logs"
SIM = LOGS / "2026-10-28-staged-construction-similarity.py"
ANALYSIS = LOGS / "2026-10-28-staged-construction-analysis.py"
LOOP = LOGS / "2026-10-28-staged-construction-agent-loop.py"
KERNELS = Path(__file__).parent / "fixtures" / "kernel-loop" / "kernels"
LIBRARY = REPO / "orchard" / "skills" / "kernel-loop-templates" / "library"


def run_sim(libdir, exdir, *kernels):
    return subprocess.run([sys.executable, str(SIM), str(libdir), str(exdir), *map(str, kernels)], capture_output=True, text=True)


# ---- similarity ---------------------------------------------------------------------------------------------------

def test_similarity_refuses_an_empty_or_missing_comparison_directory(tmp_path):
    got = run_sim(tmp_path / "nope", LIBRARY, KERNELS / "center_good.py")
    assert got.returncode != 0 and "no similarity check ran" in (got.stderr + got.stdout)
    (tmp_path / "empty").mkdir()
    assert run_sim(LIBRARY, tmp_path / "empty", KERNELS / "center_good.py").returncode != 0


def test_similarity_refuses_a_missing_or_empty_kernel(tmp_path):
    (tmp_path / "k.py").write_text("# only a comment\n")
    assert run_sim(LIBRARY, LIBRARY, tmp_path / "k.py").returncode != 0
    assert run_sim(LIBRARY, LIBRARY, tmp_path / "missing.py").returncode != 0


def test_a_copy_of_an_example_operation_is_flagged_although_the_example_has_host_code(tmp_path):
    (tmp_path / "ex").mkdir()
    (tmp_path / "ex" / "eltwise_add.py").write_text((KERNELS / "upstream_eltwise_add.py").read_text())
    got = run_sim(LIBRARY, tmp_path / "ex", KERNELS / "add_good.py")       # add_good copies upstream's operation
    assert got.returncode == 0 and "FLAG near-copy" in got.stdout, got.stdout


def test_a_copy_with_another_function_name_is_still_flagged(tmp_path):
    text = (KERNELS / "center_good.py").read_text().replace("def OPERATION", "def my_centering")
    (tmp_path / "k.py").write_text(text)
    got = run_sim(LIBRARY, KERNELS, tmp_path / "k.py")
    assert "FLAG near-copy" in got.stdout, got.stdout


def test_an_unrelated_kernel_is_not_flagged(tmp_path):
    (tmp_path / "ex").mkdir()
    (tmp_path / "ex" / "eltwise_add.py").write_text((KERNELS / "upstream_eltwise_add.py").read_text())
    got = run_sim(tmp_path / "ex", tmp_path / "ex", KERNELS / "linear_scan_raw_element.py")
    assert got.returncode == 0 and "FLAG near-copy" not in got.stdout, got.stdout


# ---- analysis -----------------------------------------------------------------------------------------------------

def analysis():
    return runpy.run_path(str(ANALYSIS), run_name="analysis")


def test_the_median_of_an_even_count_is_the_mean_of_the_two_middle_values():
    m = analysis()["median"]
    assert m([2, 10]) == 6 and m([1, 2, 9]) == 2 and m([]) == "-" and m([3]) == 3


def test_an_incomplete_experiment_has_no_verdict_and_a_complete_one_does():
    v = analysis()["verdict"]
    assert v([{}] * 0, [{}] * 5, 0, 5, 5) is None
    assert v([{}] * 10, [{}] * 9, 0, 5, 5) is None
    assert v([{}] * 10, [{}] * 10, 0, 5, 5) == (True, True, True)
    assert v([{}] * 10, [{}] * 10, 4, 5, 3) == (False, False, False)


# ---- the agent-loop harness, against a fake Telegraph ---------------------------------------------------------------

CENTER = (KERNELS / "center_good.py").read_text()


class FakeTelegraph:
    answers: dict = {}
    problems: list = []

    def __init__(self, cfg, directory, **kw):
        self.cfg = cfg

    def ask(self, problem, task, failure):
        FakeTelegraph.problems.append(problem)
        op = "center" if "-center-" in problem else "linear_scan"
        return FakeTelegraph.answers[op](problem, task)


def run_loop(tmp_path, monkeypatch, answers, attempts=1, budget="5.5", out=None):
    toml = tmp_path / "bringup.toml"
    toml.write_text('[telegraph]\nenabled = true\nendpoint = "https://example.org/v1"\nmodels = ["anthropic/claude-sonnet-5-5"]\n'
                    'max_prompt_chars = 400000\n[telegraph.prices]\n"anthropic/claude-sonnet-5-5" = [2.0, 10.0]\n')
    monkeypatch.setenv("TELEGRAPH_CONFIG", str(toml))
    monkeypatch.setattr(tg, "Telegraph", FakeTelegraph)
    FakeTelegraph.answers, FakeTelegraph.problems = answers, []
    out = out if out is not None else tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["loop", str(REPO), str(out), str(attempts), budget])
    runpy.run_path(str(LOOP), run_name="loop")
    return json.loads((Path(out) / "summary.json").read_text())


def reply(text):
    return tg.Answer(True, "```python\n" + text + "```\n", call_id="c1", model="m", tokens_in=1000, tokens_out=500, latency_s=1.0, cost_usd=0.007)


@needs_sim
def test_the_loop_prices_a_failed_answer_from_its_tokens_and_records_the_run_id_and_cache_flag(tmp_path, monkeypatch):
    empty = lambda problem, task: tg.Answer(False, reason="the answer was empty (finish_reason length)", tokens_in=49303, tokens_out=8000)
    rows = run_loop(tmp_path, monkeypatch, {"center": lambda p, t: reply(CENTER), "linear_scan": empty})
    center, scan = rows
    assert center["passed"] is True and center["turns"] == 1 and center["kernels_written"] == 1
    assert scan["cost_usd"] == pytest.approx(49303 * 2 / 1e6 + 8000 * 10 / 1e6, abs=1e-6) and scan["turns"] == 0
    assert scan["calls"][0]["sent"] is True and scan["calls"][0]["cached"] is False
    assert center["run_id"] == scan["run_id"] and center["run_id"] in FakeTelegraph.problems[0]
    assert "swept" in center


@needs_sim
def test_a_sent_request_with_unknown_cost_stops_the_run(tmp_path, monkeypatch):
    unknown = lambda problem, task: tg.Answer(False, reason="HTTP 500: server error")
    rows = run_loop(tmp_path, monkeypatch, {"center": lambda p, t: reply(CENTER), "linear_scan": unknown}, attempts=2)
    assert [r.get("skipped") is not None for r in rows] == [False, False, True, True]
    assert "cost is unknown" in rows[2]["skipped"]


@needs_sim
def test_a_local_refusal_costs_nothing_and_does_not_stop_the_run(tmp_path, monkeypatch):
    refused = lambda problem, task: tg.Answer(False, reason="the call cap for problem x is reached (6 of 6)")
    rows = run_loop(tmp_path, monkeypatch, {"center": lambda p, t: reply(CENTER), "linear_scan": refused}, attempts=2)
    assert all(r.get("skipped") is None for r in rows) and rows[1]["cost_usd"] == 0 and rows[1]["calls"][0]["sent"] is False


@needs_sim
def test_the_feedback_is_the_whole_tool_response(tmp_path, monkeypatch):
    seen = []

    def center(problem, task):
        seen.append(task)
        return reply("import ttl\n" if len(seen) == 1 else CENTER)
    run_loop(tmp_path, monkeypatch, {"center": center, "linear_scan": lambda p, t: tg.Answer(False, reason="the call cap is reached")})
    assert len(seen) == 2 and "kernel_check.py ran after this write" in seen[1] and "result of the write" in seen[1]


@needs_sim
def test_strays_are_swept_before_the_verdict_and_again_after_it(tmp_path, monkeypatch):
    from orchard import kernelprobe as kp
    order = []
    real_sweep, real_check = kp.sweep_strays, kp.authoritative_check
    monkeypatch.setattr(kp, "sweep_strays", lambda adir, **kw: order.append("sweep") or real_sweep(adir, **kw))
    monkeypatch.setattr(kp, "authoritative_check", lambda *a, **kw: order.append("verdict") or real_check(*a, **kw))
    refused = lambda p, t: tg.Answer(False, reason="the call cap is reached")
    run_loop(tmp_path, monkeypatch, {"center": lambda p, t: reply(CENTER), "linear_scan": refused})
    first = order.index("verdict")
    assert first >= 1 and order[first - 1] == "sweep" and order[first + 1] == "sweep"


# ---- the findings of issue 201 --------------------------------------------------------------------------------------

BAD = "import ttl\n"            # a kernel the check fails, so the attempt would go on to another turn
NO_USAGE = lambda text: tg.Answer(True, "```python\n" + text + "```\n", call_id="c1", model="m", tokens_in=1000, tokens_out=500, cost_usd=None)
ANSWER_CAP = lambda p, t: tg.Answer(False, reason="the call cap is reached")


@needs_sim
def test_a_successful_answer_without_usage_has_unknown_cost_and_stops_the_run_inside_the_attempt(tmp_path, monkeypatch, capsys):
    """Findings 1 and 8. The estimated token counts of such an answer are not a bill, so nothing is priced from them; the stop is
    checked inside the attempt (no second call) and the printed total says it is only the known part."""
    seen = []
    rows = run_loop(tmp_path, monkeypatch, {"center": lambda p, t: seen.append(t) or NO_USAGE(BAD), "linear_scan": ANSWER_CAP}, attempts=2)
    assert len(seen) == 1, "a second request was sent after a cost was unknown"
    first = rows[0]
    assert first["cost_known"] is False and first["calls"][0]["cost_usd"] is None
    assert all(r.get("skipped") and "cost is unknown" in r["skipped"] for r in rows[1:])
    assert "at least" in capsys.readouterr().out


@needs_sim
@pytest.mark.parametrize("reason", ["HTTP 500: capacity exhausted", "HTTP 500: the call cap was exceeded upstream",
                                    "TimeoutError: budget wait", "capacity exhausted"])
def test_an_error_from_a_sent_request_is_unknown_cost_even_when_it_contains_a_refusal_word(tmp_path, monkeypatch, reason):
    """Finding 2. 'capacity exhausted' contains 'cap'; only a local refusal that was never sent may be free."""
    crashed = lambda p, t: tg.Answer(False, reason=reason)
    rows = run_loop(tmp_path, monkeypatch, {"center": lambda p, t: reply(CENTER), "linear_scan": crashed}, attempts=2)
    assert rows[1]["calls"][0]["sent"] is True and rows[1]["cost_known"] is False
    assert "cost is unknown" in rows[2]["skipped"]


@needs_sim
def test_an_output_directory_that_already_holds_files_is_refused_and_left_alone(tmp_path, monkeypatch):
    """Finding 3."""
    out = tmp_path / "out"
    (out / "center" / "attempt-1").mkdir(parents=True)
    keep = out / "summary.json"
    keep.write_text("[]")
    with pytest.raises(SystemExit) as stop:
        run_loop(tmp_path, monkeypatch, {"center": lambda p, t: reply(CENTER), "linear_scan": ANSWER_CAP}, out=out)
    assert "fresh" in str(stop.value) and keep.read_text() == "[]" and FakeTelegraph.problems == []


@needs_sim
def test_a_non_python_fence_before_the_python_block_is_not_taken_for_the_kernel(tmp_path, monkeypatch):
    """Finding 4. The closing fence of the json block used to open a block that held the prose between the two."""
    text = "```json\n{\"a\": 1}\n```\nHere is the kernel:\n```python\n" + CENTER + "```\n"
    answer = lambda p, t: tg.Answer(True, text, call_id="c1", model="m", tokens_in=1000, tokens_out=500, cost_usd=0.007)
    center, _ = run_loop(tmp_path, monkeypatch, {"center": answer, "linear_scan": ANSWER_CAP})
    assert center["passed"] is True and center["kernels_written"] == 1


@needs_sim
def test_a_repeated_kernel_is_not_counted_as_a_written_kernel(tmp_path, monkeypatch):
    """Finding 5. write_file answers 'unchanged:' for the same text and does not run the check."""
    n = []
    center = lambda p, t: (n.append(1), reply(BAD if len(n) < 3 else CENTER))[1]
    row, _ = run_loop(tmp_path, monkeypatch, {"center": center, "linear_scan": ANSWER_CAP})
    assert row["turns"] == 3 and row["kernels_written"] == 2 and row["unchanged_writes"] == 1


@needs_sim
def test_an_interrupted_attempt_is_swept_and_its_evidence_is_already_on_disk(tmp_path, monkeypatch):
    """Findings 6 and 7. A failure in the second request still sweeps the attempt directory, keeps the finished attempt and the
    first call's record, and writes the unfinished attempt as a row of its own."""
    from orchard import kernelprobe as kp
    swept = []
    real = kp.sweep_strays
    monkeypatch.setattr(kp, "sweep_strays", lambda adir, **kw: swept.append(Path(adir).parts[-2:]) or real(adir, **kw))

    def boom(problem, task):
        raise RuntimeError("network went away")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        run_loop(tmp_path, monkeypatch, {"center": lambda p, t: reply(CENTER), "linear_scan": boom}, out=out)
    rows = json.loads((out / "summary.json").read_text())
    assert [r["op"] for r in rows] == ["center", "linear_scan"] and rows[0]["passed"] is True
    assert rows[1]["ended"].startswith("interrupted: RuntimeError")
    assert ("linear_scan", "attempt-1") in swept
    calls = [json.loads(l) for l in (out / "calls.jsonl").read_text().splitlines()]
    assert [c["op"] for c in calls] == ["center"]


@needs_sim
def test_a_relative_output_directory_still_gives_the_verdict_absolute_paths(tmp_path, monkeypatch):
    """Finding 11, already fixed in the first round (the paths are resolved at the top); this keeps it fixed."""
    monkeypatch.chdir(tmp_path)
    center, _ = run_loop(tmp_path, monkeypatch, {"center": lambda p, t: reply(CENTER), "linear_scan": ANSWER_CAP}, out=Path("rel-out"))
    assert center["passed"] is True and center["category"] is None


# ---- similarity: the alias and the empty comparison files -------------------------------------------------------------

def test_a_decorator_imported_under_another_name_is_still_the_operation(tmp_path):
    """Finding 10. The checker accepts `from ttl import operation as op` and `@op(...)`; the matcher must too."""
    alias = (KERNELS / "add_alias_function_as.py").read_text()
    host = "\n".join(f"print('host side test line {i}', {i})" for i in range(60))
    (tmp_path / "ex").mkdir()
    (tmp_path / "ex" / "eltwise_add.py").write_text(alias + "\n" + host + "\n")        # an example: the operation plus host code
    kernel = tmp_path / "k.py"
    kernel.write_text(alias.replace("def OPERATION", "def my_renamed_operation"))
    got = run_sim(tmp_path / "ex", tmp_path / "ex", kernel)
    assert got.returncode == 0 and "FLAG near-copy" in got.stdout, got.stdout


def test_a_comparison_directory_of_empty_or_comment_only_files_is_refused(tmp_path):
    """Finding 9 of the similarity list (review comment on line 58)."""
    (tmp_path / "ex").mkdir()
    (tmp_path / "ex" / "a.py").write_text("")
    (tmp_path / "ex" / "b.py").write_text("# nothing but a comment\n")
    got = run_sim(tmp_path / "ex", LIBRARY, KERNELS / "center_good.py")
    assert got.returncode != 0 and "no similarity check ran" in (got.stderr + got.stdout)


# ---- analysis: only the registered attempt ids -----------------------------------------------------------------------

def test_the_loader_ignores_attempt_ids_outside_the_registered_range(tmp_path):
    """Finding 9. Records 2 to 11 used to count as a complete experiment."""
    load = analysis()["load"]
    for k in (2, 3, 11, 12):
        d = tmp_path / f"S-center-{k}" / "center" / "attempt-1"
        d.mkdir(parents=True)
        (d / "attempt.json").write_text("{}")
    got = load(tmp_path)
    assert ("S", "center", 2) in got and ("S", "center", 3) in got
    assert ("S", "center", 11) not in got and ("S", "center", 12) not in got
