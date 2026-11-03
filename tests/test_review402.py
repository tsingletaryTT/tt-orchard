# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Copilot review of PR 402: one test per finding, written before the fix."""
import importlib.util
import json
from pathlib import Path

import pytest

from orchard import can, served_probes as sp, served_verify as sv
from test_served_verify import Fake
from test_served_probes import chat_reply


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.close()


# 2: the fallback sentence must identify a reference
TABLE = "| eval | this package | HF reference |\n| --- | --- | --- |\n| score corr | 0.998 |  |\n"


def card(after):
    return f"# toy\n## Expected performance\n{TABLE}{after}## Limitations\nOne limit.\n"


@pytest.mark.parametrize("after", ["### Reference\n", "Reference: TBD.\n", "Baseline: n/a\n", "Reference.\n", "The reference is unknown.\n",
                                   "### The reference is the transformers model on CPU\n",
                                   "Reference: TBD, to be filled in later by the author of this card.\n"])
def test_a_heading_or_placeholder_does_not_state_a_reference(after):
    assert can.card_problems(card(after), "catalog")


def test_a_sentence_that_names_the_reference_still_passes():
    assert can.card_problems(card("The reference is the transformers model on CPU, run on the same token ids.\n"), "catalog") == []


# 3: a probe sets its own token budget
def test_request_extra_max_tokens_does_not_change_a_probe_budget(fake):
    seen = []
    fake.post["/v1/chat/completions"] = lambda b: (seen.append(b), (200, chat_reply("408", tokens=3, reasoning_content="x")))[1]
    sp.probe_reasoning(fake.base, "m", {"max_tokens": 64})
    assert [b["max_tokens"] for b in seen] == [sp.BUDGETS[0], sp.BUDGETS[1], sp.BUDGETS[1]]
    seen.clear()
    sp.probe_code(fake.base, "m", {"max_tokens": 64})
    assert seen[0]["max_tokens"] == 256


def test_the_boot_carry_drops_max_tokens():
    assert "max_tokens" not in can._carry({"model": "m", "max_tokens": 5, "top_k": 1})
    assert can._carry({"model": "m", "top_k": 1}) == {"top_k": 1}


# 4: the prefix probe respects the context
def test_a_context_too_small_for_the_prefix_probe_is_not_checked(fake):
    fake.post["/v1/chat/completions"] = lambda b: (200, chat_reply("x", prompt_tokens=10))
    r = sp.probe_prefix_reuse(fake.base, "m", None, declared=True, max_model_len=1024)
    assert r["state"] == "not_checked" and ("POST", "/v1/chat/completions") not in fake.log


# 5: nested definitions are not the function's own body
@pytest.mark.parametrize("code", [
    "def is_prime(n):\n    def helper():\n        return 1\n    print(n)\n",
    "def is_prime(n):\n    f = lambda: 1\n    print(n)\n    class K:\n        def m(self):\n            return 2\n",
])
def test_a_return_inside_a_nested_definition_is_not_a_return_of_the_function(code):
    r = sp.check_generated_code(code)
    assert r["state"] == "fail" and "no `return`" in r["detail"], r


def test_a_while_true_inside_a_nested_definition_is_not_the_functions_loop():
    code = "def is_prime(n):\n    def spin():\n        while True:\n            pass\n    return n\n"
    assert sp.check_generated_code(code)["state"] == "pass"


# 6 and 7: batching falls back to the ratio; metrics are read with a bound
def test_an_idle_gauge_does_not_override_a_batched_ratio():
    r = sp.judge_batching(single_rate=10.0, tokens=128, seconds=3.2, n=4, max_running=0)
    assert r["state"] == "pass"
    r = sp.judge_batching(single_rate=10.0, tokens=128, seconds=12.8, n=4, max_running=1)
    assert r["state"] == "fail"


def test_the_metrics_reader_of_the_batching_probe_is_bounded(fake, monkeypatch):
    fake.metrics_body = "vllm:num_requests_running 3\n" + "x" * 100
    monkeypatch.setattr(sv, "MAX_METRICS_BYTES", 20)
    with pytest.raises(OSError):
        sp._metrics_text(fake.base + "/metrics", 2.0)


# 8: a model name with the word reasoning is not a parser line
def test_a_log_that_only_mentions_a_reasoning_model_name_is_not_checked(fake):
    fake.post["/v1/chat/completions"] = lambda b: (200, chat_reply("408", tokens=300, reasoning_content="x"))
    out = {r["name"]: r for r in sp.probe_reasoning(fake.base, "m", None, log_text="loading org/reasoning-model weights")}
    assert out["reasoning_parser"]["state"] == "not_checked"


# 9: a truncated answer is not a completed sweep
def test_a_truncated_answer_at_the_large_budget_fails_the_sweep(fake):
    fake.post["/v1/chat/completions"] = lambda b: (200, chat_reply("The result is", tokens=b["max_tokens"], finish="length", reasoning_content="x"))
    out = {r["name"]: r for r in sp.probe_reasoning(fake.base, "m", None)}
    assert out["budget_sweep"]["state"] == "fail" and out["budget_sweep"]["large_finish"] == "length"


# 10: the last matching line in log order wins
def test_the_last_layer_line_in_log_order_wins_across_formats():
    log = "layers loaded: 12\nloaded 36 layers\n"
    r = {x["name"]: x for x in sv.check_engine_log(log, config={"num_hidden_layers": 36})}
    assert r["layers_loaded"]["state"] == "pass"


# 11: a real backend warning is kept
def test_a_backend_fallback_warning_is_not_dropped_as_noise():
    line = "WARNING flash_attn is unsupported for this configuration; falling back to unfused attention"
    assert [w["text"] for w in sv.log_warnings(line)["lines"]] == [line]
    noisy = "WARNING Failed to import from vllm._C with ImportError('libcuda.so.1: cannot open shared object file')"
    assert sv.log_warnings(noisy)["lines"] == []


# 12: the container route passes its declared context
def test_the_container_template_passes_the_context_from_its_command():
    tpl = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates"
    spec = importlib.util.spec_from_file_location("sac_review", tpl / "serve_and_compare.py")
    sac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sac)

    class Proc:
        declared_max_model_len = 2048
    assert sac.live_declared_context(Proc(), tpl / "no-such-run.sh") == 2048
    run_sh = tpl.parent.parent.parent.parent / "tests" / "nonexistent-run.sh"
    assert sac.live_declared_context(object(), run_sh) is None
    assert sac.declared_max_model_len(" ".join(["docker", "run", "img", "--max_model_len", "8192"])) == 8192
