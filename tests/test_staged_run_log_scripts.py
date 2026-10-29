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
from test_kernelprobe import needs_sim, VENV  # noqa: F401

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


def run_loop(tmp_path, monkeypatch, answers, attempts=1, budget="5.5"):
    toml = tmp_path / "bringup.toml"
    toml.write_text('[telegraph]\nenabled = true\nendpoint = "https://example.org/v1"\nmodels = ["anthropic/claude-sonnet-5-5"]\n'
                    'max_prompt_chars = 400000\n[telegraph.prices]\n"anthropic/claude-sonnet-5-5" = [2.0, 10.0]\n')
    monkeypatch.setenv("TELEGRAPH_CONFIG", str(toml))
    monkeypatch.setattr(tg, "Telegraph", FakeTelegraph)
    FakeTelegraph.answers, FakeTelegraph.problems = answers, []
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["loop", str(REPO), str(out), str(attempts), budget])
    runpy.run_path(str(LOOP), run_name="loop")
    return json.loads((out / "summary.json").read_text())


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
