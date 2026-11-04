# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 370, independent review of PR 443 (the PR is past Copilot's round limit).

1. A numeric capture with no digit in it (`GPU KV cache size: , tokens`) raised ValueError inside the boot check and ended it. No line of a
   server log may raise: every numeric slot of every pattern is fed junk, and a failure inside the log check is a recorded `not_checked`.
2. A `Loading N layers` line is only an intent, printed before the layers load. With no progress bar after it the check no longer passes.
3. The layer progress bar of a wide terminal (a body of more than 64 characters between the pipes) was not matched, so a stalled load looked
   like a log with no bar.
Also from the sweep: the arithmetic completion probe read an unbounded integer from the model's reply, and a reply that starts with more than
4,300 digits (a repetition loop does it) made `int()` raise.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import served_verify as sv

CFG64 = {"num_hidden_layers": 64}
ROOT = str(Path(__file__).resolve().parent.parent)


def inner(log, **kw):
    """The log check itself, not the wrapper that turns a failure into `not_checked`: a test that wants to know whether a line can make it raise."""
    kw.setdefault("config", CFG64)
    kw.setdefault("max_model_len", 1024)
    kw.setdefault("max_num_seqs", 2)
    kw.setdefault("block_size", 64)
    return {r["name"]: r for r in sv._check_engine_log(log, **kw)}


def run(log, **kw):
    kw.setdefault("config", CFG64)
    kw.setdefault("max_model_len", 1024)
    kw.setdefault("max_num_seqs", 1)
    return {r["name"]: r for r in sv.check_engine_log(log, **kw)}


def _loading(n):
    return f"(EngineCore pid=101) 2026-11-03 12:39:15.043 | INFO | models.x.tt.model:__init__:174 - Loading {n} transformer layers (indices=[0])\n"


def _bar(done, total, width=10):
    return (f"Loading layers:   0%|{' ' * width}| 0/{total} [00:00<?, ?it/s]\r"
            f"Loading layers: {100 * done // total:3d}%|{'#' * width}| {done}/{total} [04:23<00:00,  4.12s/it]\n")


# ---- 1. no line may raise ------------------------------------------------------------------------------------------------------------

JUNK = ["", ",", ",,", ",,,,,,,", " ", "٣", "²", "１２", "1e3", "-1", "0", "9" * 25, "9" * 5000, ",1", "1,,,", "٣,٤", "٣,٤٥", "1.5", "nan", "inf"]
TEMPLATES = [
    "GPU KV cache size: {x} tokens\n",
    "Maximum concurrency for {x} tokens per request: 1.00x\n",
    "Maximum concurrency for 8,192 tokens per request: {x}x\n",
    "# GPU blocks: {x}\n",
    "Overriding num_gpu_blocks=1 with num_gpu_blocks_override={x}\n",
    "Overriding num_gpu_blocks={x} with num_gpu_blocks_override=2\n",
    "Loading {x} transformer layers (indices=[0])\n",
    "loaded {x} layers\n",
    "layers loaded: {x}\n",
    "Loading layers:  50%|#####| {x}/64 [00:01<00:01]\n",
    "Loading layers:  50%|#####| 32/{x} [00:01<00:01]\n",
    "Loading layers: {x}%|#####| 32/64\n",
]


@pytest.mark.parametrize("template", TEMPLATES)
def test_no_numeric_slot_of_any_pattern_can_make_the_log_check_raise(template):
    for junk in JUNK:
        log = template.format(x=junk)
        for extra in ("", _loading(64) + "GPU KV cache size: 4,100 tokens\n"):
            out = inner(extra + log)
            assert set(out) == {"layers_loaded", "kv_cache"}, (junk, out)
            assert all(r["state"] in ("pass", "fail", "not_checked") for r in out.values()), (junk, out)


def test_the_reviewers_two_excerpts_are_a_result_and_not_an_exception():
    a = inner("GPU KV cache size: , tokens\n")
    assert a["kv_cache"]["state"] == "not_checked"
    b = inner("Maximum concurrency for ,, tokens per request: 1.00x\n")
    assert b["kv_cache"]["state"] == "not_checked"


def test_a_digit_free_capacity_does_not_hide_a_good_one_before_it():
    log = "GPU KV cache size: 4,100 tokens\nGPU KV cache size: , tokens\n"
    r = inner(log, max_model_len=2000)["kv_cache"]
    assert r["state"] == "pass" and r["capacity_tokens"] == 4100


def test_a_failure_inside_the_log_check_is_a_recorded_not_checked_naming_the_error_class(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(sv, "_check_engine_log", boom)
    out = sv.check_engine_log("anything", config=CFG64)
    assert [r["name"] for r in out] == ["layers_loaded", "kv_cache"]
    assert all(r["state"] == "not_checked" and "RuntimeError" in r["detail"] for r in out)


def test_the_arithmetic_probe_does_not_raise_on_a_reply_that_is_thousands_of_digits():
    good = sv._GOOD["arithmetic"][0]
    assert good("9" * 5000) is False and good("408") is True and good(" 408 apples") is True
    body = json.dumps({"choices": [{"text": "9" * 5000}]}).encode()
    assert sv.judge_completion("arithmetic", 200, body)["state"] == "fail"


# ---- 2. an intent line alone does not say the layers loaded ---------------------------------------------------------------------------

def test_an_intent_with_no_bar_after_it_is_not_a_pass_even_when_an_earlier_load_finished():
    log = _loading(32) + _bar(32, 32) + _loading(64)
    r = run(log)["layers_loaded"]
    assert r["state"] == "not_checked" and r["layers"] == 64 and r["layers_completed"] is None
    assert "intent" in r["detail"] and "no progress bar" in r["detail"]


def test_an_intent_with_no_bar_anywhere_in_the_log_is_not_a_pass_either():
    r = run(_loading(64))["layers_loaded"]
    assert r["state"] == "not_checked" and "finished loading" in r["detail"]


def test_a_wrong_count_is_still_a_fail_whatever_the_bars_say():
    assert run(_loading(32))["layers_loaded"]["state"] == "fail"


def test_a_completion_statement_without_a_bar_is_still_a_pass():
    r = run("loaded 64 layers\n")["layers_loaded"]
    assert r["state"] == "pass" and r["layers_completed"] is None


def test_an_intent_followed_by_its_finished_bar_is_still_a_pass():
    r = run(_loading(64) + _bar(64, 64))["layers_loaded"]
    assert r["state"] == "pass" and r["layers_completed"] == 64


# ---- 3. the bar of a wide terminal ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("width", [80, 200, 5000])
def test_a_stalled_bar_drawn_to_a_wide_terminal_is_a_fail(width):
    r = run(_loading(64) + _bar(32, 64, width=width))["layers_loaded"]
    assert r["state"] == "fail" and r["layers_completed"] == 32 and "32/64" in r["detail"]


@pytest.mark.parametrize("width", [80, 200, 5000])
def test_a_finished_bar_drawn_to_a_wide_terminal_is_a_pass(width):
    r = run(_loading(64) + _bar(64, 64, width=width))["layers_loaded"]
    assert r["state"] == "pass" and r["layers_completed"] == 64


HOSTILE_BAR = textwrap.dedent("""
    import resource, sys, time
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    sys.path.insert(0, sys.argv[2])
    from orchard import served_verify as sv
    kind = sys.argv[1]
    if kind == "open":                      # one opener, a 2 MB body, no closing pipe
        log = "Loading 64 transformer layers\\nLoading layers:  50%|" + "#" * (2 * 1024 * 1024) + "\\n"
    elif kind == "closed":                  # a 2 MB body with its pipes and counts
        log = "Loading 64 transformer layers\\nLoading layers:  50%|" + "#" * (2 * 1024 * 1024) + "| 32/64\\n"
    elif kind == "repeated":                # 100,000 openers on one line
        log = "Loading 64 transformer layers\\n" + "Loading layers:  5%|" * 100000 + "\\n"
    else:                                   # 100,000 openers each with a short body and no counts
        log = "Loading 64 transformer layers\\n" + "Loading layers:  5%|##|" * 100000 + "\\n"
    t = time.time()
    out = sv._check_engine_log(log, config={"num_hidden_layers": 64})
    print(out[0]["state"], round(time.time() - t, 2))
""")


@pytest.mark.parametrize("kind", ["open", "closed", "repeated", "short_bodies"])
def test_a_hostile_bar_line_is_read_in_linear_time_inside_a_limited_child(kind):
    done = subprocess.run([sys.executable, "-I", "-c", HOSTILE_BAR, kind, ROOT], capture_output=True, text=True, timeout=120,
                          env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr[-500:]
    state, seconds = done.stdout.split()
    assert float(seconds) < 15, (kind, seconds)
    assert state in ("pass", "fail", "not_checked")


def test_the_count_reader_takes_a_plain_count_with_separators_and_nothing_else():
    assert sv._int("4,100") == 4100 and sv._int("262400") == 262400 and sv._int("1,234,567") == 1234567
    for bad in ("", ",", ",,", " ", "٣", "１２", "1e3", "-1", "1.5", "9" * 21, "9" * 5000, None, 12):
        assert sv._int(bad) is None, bad


def test_a_capture_that_starts_with_a_comma_is_not_read_as_a_number():
    # `,4100` is not a count: the old pattern took it as 4,100 once the comma was dropped
    assert inner("GPU KV cache size: ,4100 tokens\n")["kv_cache"]["state"] == "not_checked"
    r = inner("Maximum concurrency for ,8192 tokens per request: 0.50x\n")["kv_cache"]
    assert r["state"] == "not_checked" and r.get("concurrency") is None


def test_a_capacity_line_whose_number_the_reader_refuses_does_not_replace_a_good_earlier_one():
    # Arabic-Indic digits match `\d` but are refused by the count reader; the line must be left out and not taken as the last report
    log = "GPU KV cache size: 4,100 tokens\nGPU KV cache size: ٣٤ tokens\n"
    r = inner(log, max_model_len=2000)["kv_cache"]
    assert r["state"] == "pass" and r["capacity_tokens"] == 4100


def test_a_concurrency_line_whose_number_the_reader_refuses_does_not_replace_a_good_earlier_one():
    log = "GPU KV cache size: 4,100 tokens\nMaximum concurrency for 1,024 tokens per request: 2.00x\nMaximum concurrency for ٣٤ tokens per request: 0.50x\n"
    r = inner(log, max_model_len=1024)["kv_cache"]
    assert r["state"] == "pass" and r["concurrency"] == 2.0 and r["tokens_per_request"] == 1024
