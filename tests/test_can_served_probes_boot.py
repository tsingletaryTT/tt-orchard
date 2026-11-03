# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 360 to 373 and 385: the new probes through `can boot`, against the stand-in server of tests/test_can_served_verify.py (no chip)."""
import pytest

from orchard import can
from test_can_served_verify import boot, bundle, spec_of


def run(tmp_path, fault="", **more):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return can.boot_check(bundle(tmp_path, fault), tmp_path / "work", spec_of(**more), install=False, health_timeout=30)


@pytest.mark.parametrize("fault,failed", [("bad_code", "code_parse"), ("wrong_layers", "layers_loaded"), ("small_kv", "kv_cache")])
def test_the_new_probes_each_fail_their_own_check(tmp_path, fault, failed):
    ev = boot(tmp_path, fault)
    assert not ev["passed"] and ev["checks"][failed]["ok"] is False, ev["checks"]
    assert [n for n, c in ev["checks"].items() if not c["ok"]] == [failed], ev["checks"]


def test_a_warning_in_the_log_is_recorded_and_never_fails_the_boot(tmp_path):
    ev = boot(tmp_path, "warn")
    assert ev["passed"], ev
    assert [w["text"] for w in ev["log_warnings"]["lines"]] == ["WARNING Unrecognised device name P150b; falling back to the generic profile"]


def test_the_reply_token_split_is_recorded(tmp_path):
    ev = boot(tmp_path)
    assert ev["token_split"]["chat"]["completion_tokens"] == 5 and ev["token_split"]["chat"]["basis"] == "no reasoning text"


def test_batching_tells_serialised_from_batched_when_the_can_declares_max_num_seqs_and_a_rate_is_measured(tmp_path):
    check = can.Check(expect="Paris", stream=False, min_decode_tok_s=1.0)
    ev = run(tmp_path / "a", max_num_seqs=4, check=check)
    assert ev["checks"]["batching"]["ok"] is True, ev["checks"]
    assert ev["probe_facts"]["batching"]["requests"] == 4
    ev = run(tmp_path / "b", "serialise", max_num_seqs=4, check=check)
    assert ev["checks"]["batching"]["ok"] is False and "one after another" in ev["checks"]["batching"]["detail"], ev["checks"]


def test_a_can_with_max_num_seqs_but_no_measured_rate_lists_batching_as_not_checked(tmp_path):
    ev = run(tmp_path, max_num_seqs=4)
    assert "batching" not in ev["checks"] and "batching" in [n["name"] for n in ev["not_checked"]]


def test_prefix_caching_is_timed_only_for_a_can_that_declares_it(tmp_path):
    check = can.Check(expect="Paris", stream=False, prefix_caching=True)
    ev = run(tmp_path / "a", check=check)
    assert "prefix_reuse" in ev["checks"] and ev["checks"]["prefix_reuse"]["ok"] is False      # the stand-in caches nothing
    assert "prefix_reuse" not in run(tmp_path / "b")["checks"]


def test_a_served_model_name_alias_set_only_through_extra_args_is_what_the_listing_is_judged_against(tmp_path):
    ev = run(tmp_path / "a", extra_args=("--served-model-name", "served-name"))
    assert ev["checks"]["served_model"]["ok"] is True
    ev = run(tmp_path / "b", extra_args=("--served-model-name", "alias-the-server-lacks"))
    assert ev["checks"]["served_model"]["ok"] is False and "alias-the-server-lacks" in ev["checks"]["served_model"]["detail"]


def test_quirks_are_listed_only_for_this_model_and_this_release(tmp_path):
    quirks = ({"model": "someone/toy", "server_release": "0.10.1", "note": "needs enable_thinking false"},
              {"model": "someone/toy", "server_release": "0.9.0", "note": "another release"},
              {"model": "someone/else", "server_release": "0.10.1", "note": "another model"})
    ev = run(tmp_path, check=can.Check(expect="Paris", stream=False, quirks=quirks), vllm_version="0.10.1")
    assert [q["note"] for q in ev["quirks"]] == ["needs enable_thinking false"]


def test_the_reasoning_probes_do_not_run_unless_the_can_declares_a_reasoning_model(tmp_path):
    assert "reasoning_field" not in boot(tmp_path)["checks"]


def test_a_reasoning_can_runs_the_reasoning_probes_and_the_stand_in_without_a_reasoning_field_fails_them(tmp_path):
    ev = run(tmp_path, check=can.Check(expect="Paris", stream=False, reasoning=True))
    assert ev["checks"]["reasoning_field"]["ok"] is False
    assert "reasoning_parser" in {n["name"] for n in ev["not_checked"]}                 # the stand-in's log prints no engine arguments
    assert not ev["passed"]


