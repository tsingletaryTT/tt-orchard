# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The swap template's sampling probe (issue 107, step i of serve_and_compare.py). After the 32-token check it sends one greedy and two sampled
requests (same prompt, temperature 1.0, top-p 0.9, seeds that differ) to the same server and records them raw under `sampling_probe` in
swap-check.json. The fake server in tests/fake_swap_server.py has a `sampling` mode: "applied" (text depends on the seed), "ignored" (the same
text whatever the seed, as a greedy-only decode path gives) and "refuse" (a sampled request gets HTTP 400)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchard import sampling_probe as sp  # noqa: E402
from test_weights_swap_templates import report, swap  # noqa: E402,F401  (a fixture)

SESSION_LINE = ("[dflash2-serve] slot 0 session: 4 iters, 18 committed tokens, accept 3.75/7 -> 4.50 tok/iter, 21.3 tok/s in-model (incl. seed); "
                "histogram [0, 2, 0, 0, 0, 0, 1, 1]")


def requests(tmp_path):
    log = tmp_path / "requests.jsonl"
    return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


def run(swap, tmp_path, sampling="applied", **kw):
    kw.setdefault("sampling_probe", True)
    extra = {"request_log": str(tmp_path / "requests.jsonl"), "sampling": sampling, "usage_tokens": 96, "sampling_log_settle_s": 0.05,
             "probe_delay_s": 0.3}       # every probe request takes about the same time, so a few milliseconds of jitter cannot change a ratio
    extra.update(kw.pop("server_extra", {}))
    return swap["start"]("perfect", server_extra=extra, **kw)


def probe_requests(tmp_path):
    return [q["body"] for q in requests(tmp_path) if isinstance(q["body"].get("prompt"), str)]


def test_without_the_key_nothing_is_sent_and_nothing_is_recorded(swap, tmp_path):
    r = run(swap, tmp_path, sampling_probe=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sampling_probe" not in report(swap)
    assert probe_requests(tmp_path) == []


def test_the_default_config_of_the_older_tests_has_no_sampling_probe_key(swap):
    assert "sampling_probe" not in swap["cfg"]


def test_one_greedy_and_two_sampled_requests_are_sent_after_the_32_token_check(swap, tmp_path):
    r = run(swap, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    bodies = probe_requests(tmp_path)
    assert len(bodies) == 3
    greedy, a, b = bodies
    assert greedy["temperature"] == 0
    assert a["temperature"] == b["temperature"] == 1.0 and a["top_p"] == b["top_p"] == 0.9
    assert isinstance(a["seed"], int) and isinstance(b["seed"], int) and a["seed"] != b["seed"]
    assert a["prompt"] == b["prompt"] == greedy["prompt"] and a["max_tokens"] == b["max_tokens"] == greedy["max_tokens"] > 1
    log = requests(tmp_path)
    first_probe = next(i for i, q in enumerate(log) if isinstance(q["body"].get("prompt"), str))
    forced = [i for i, q in enumerate(log) if q["body"].get("max_tokens") == 1]
    assert len(forced) == 32 and max(forced) < first_probe          # the probe comes after the teacher-forced loop


def test_the_record_holds_the_raw_numbers_and_the_judge_calls_applied_sampling_full_speed(swap, tmp_path):
    r = run(swap, tmp_path, sampling="applied")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["sampling_probe"]
    assert rec["ran"] is True and rec["max_tokens"] == 96 and isinstance(rec["prompt"], str)
    for name in sp.ENTRY_NAMES:
        e = rec["requests"][name]
        assert e["accepted"] is True and e["error"] is None and e["completion_tokens"] == 96
        assert isinstance(e["seconds"], float) and e["seconds"] > 0 and e["text"]
    assert rec["requests"]["sampled_a"]["text"] != rec["requests"]["sampled_b"]["text"]
    assert sp.problems(rec) == []
    assert sp.summarize(rec)["verdict"] == "sampled-full-speed"


def test_a_server_that_ignores_the_seed_is_recorded_and_judged_ignored(swap, tmp_path):
    r = run(swap, tmp_path, sampling="ignored")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["sampling_probe"]
    assert rec["requests"]["sampled_a"]["text"] == rec["requests"]["sampled_b"]["text"]
    assert sp.summarize(rec)["verdict"] == "sampled-ignored"


def test_a_refused_sampled_request_is_recorded_with_its_error_and_the_run_goes_on(swap, tmp_path):
    r = run(swap, tmp_path, sampling="refuse")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    assert rep["result_draft"]["top1_agreement"] == 1.0                              # the 32-token measurement is saved
    rec = rep["sampling_probe"]
    assert rec["requests"]["greedy"]["accepted"] is True
    for name in ("sampled_a", "sampled_b"):
        e = rec["requests"][name]
        assert e["accepted"] is False and "HTTP 400" in e["error"] and e["text"] == ""
    assert sp.summarize(rec)["verdict"] == "sampled-refused"


def test_a_cut_off_answer_to_a_probe_request_is_that_requests_error_and_the_run_goes_on(swap, tmp_path):
    r = run(swap, tmp_path, sampling="cutoff")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["sampling_probe"]
    errors = [rec["requests"][n]["error"] for n in sp.ENTRY_NAMES if rec["requests"][n]["error"]]
    assert errors and "IncompleteRead" in errors[0]
    assert report(swap)["result_draft"]["top1_agreement"] == 1.0


def test_a_slow_sampled_request_is_slower_in_the_judges_numbers(swap, tmp_path):
    r = run(swap, tmp_path, sampling="applied", server_extra={"sampled_delay_s": 1.5})
    assert r.returncode == 0, r.stdout + r.stderr
    s = sp.summarize(report(swap)["sampling_probe"])
    assert s["verdict"] == "sampled-slower" and s["ratio"] < sp.SLOW_RATIO


def test_the_drafter_session_lines_each_request_caused_are_kept_with_that_request(swap, tmp_path):
    r = run(swap, tmp_path, server_extra={"session_line": SESSION_LINE})
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["sampling_probe"]
    for name in sp.ENTRY_NAMES:
        assert rec["requests"][name]["drafter_log"].count("[dflash2-serve]") == 1       # one line per request, not the whole log
    assert sp.summarize(rec)["greedy"]["drafter_accept_rate"] == pytest.approx(15 / 28)


def test_a_server_that_prints_no_session_line_leaves_the_log_field_empty(swap, tmp_path):
    r = run(swap, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(report(swap)["sampling_probe"]["requests"][n]["drafter_log"] == "" for n in sp.ENTRY_NAMES)
    assert sp.summarize(report(swap)["sampling_probe"])["greedy"]["drafter_accept_rate"] is None


def test_the_result_draft_is_unchanged_by_the_probe(swap, tmp_path):
    r = run(swap, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(report(swap)["result_draft"]) == {"serves", "server_ready_s", "coherent", "free_run_text", "top1_agreement",
                                                 "n_tokens", "cache_dir", "evidence"}
