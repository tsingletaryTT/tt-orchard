# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/verdict.py: the plain sentences the operator view puts above the raw numbers.

A number with no yardstick tells the operator nothing, so each verdict names the bar from the code and the earlier runs on this
machine. Two things matter most here: a number is compared only with numbers that can be compared (same chip count and package for
a boot time), and the reference check reads the evidence the template wrote instead of trusting the file an agent wrote about it.
"""
import json
from pathlib import Path

import pytest

from orchard import verdict
from orchard.defaults import SWAP_MIN_TOKENS, SWAP_TOP1_MIN


def swap(top1=1.0, n=32, ready=682.1, coherent=True):
    return {"swap": {"top1_agreement": top1, "n_tokens": n, "server_ready_s": ready, "coherent": coherent}}


def cfg(chips=2, top1=1.0, ready=132.0, package="episod/qwen3.8-27b-dflash2-p300", ok=True, reason=None):
    return {"chips": chips, "pass": ok, "top1_agreement": top1, "server_ready_s": ready, "package": package, "reason": reason}


def peer(model="org/other", *, top1=None, configs=()):
    r = {}
    if top1 is not None:
        r.update(swap(top1))
    if configs:
        r["configs"] = list(configs)
    return {"model": model, "results": r}


def by_id(items):
    return {i["id"]: i for i in items}


def test_a_run_with_no_results_has_no_verdict():
    assert verdict.build({}, []) == []


def test_a_swap_that_clears_the_bar_by_a_margin_is_good_and_names_the_bar_and_earlier_scores():
    item = by_id(verdict.build(swap(top1=1.0), [peer("a/RAG-Gate-27B", top1=0.96875), peer("b/clef", top1=0.96875)]))["swap"]
    assert item["tone"] == "good"
    assert "32 of 32" in item["headline"] and "1.00" in item["headline"]
    assert f"{SWAP_TOP1_MIN:.2f}" in item["detail"]
    assert "0.97" in item["detail"] and "RAG-Gate-27B" in item["detail"] and "clef" in item["detail"]
    assert "one prompt" in item["detail"]                                   # says what the number rests on


def test_a_swap_just_above_the_bar_is_a_warning_about_the_margin():
    item = by_id(verdict.build(swap(top1=SWAP_TOP1_MIN + 0.02), []))["swap"]
    assert item["tone"] == "warn" and "thin margin" in (item["headline"] + item["detail"])


def test_a_swap_below_the_bar_is_bad():
    item = by_id(verdict.build(swap(top1=SWAP_TOP1_MIN - 0.1), []))["swap"]
    assert item["tone"] == "bad" and "below the bar" in item["headline"]


def test_an_unreadable_swap_is_bad_whatever_its_agreement():
    assert by_id(verdict.build(swap(top1=1.0, coherent=False), []))["swap"]["tone"] == "bad"


def test_a_swap_resting_on_few_tokens_says_so():
    item = by_id(verdict.build(swap(top1=1.0, n=SWAP_MIN_TOKENS - 1), []))["swap"]
    assert item["tone"] == "warn" and str(SWAP_MIN_TOKENS - 1) in item["detail"]


def test_the_stage_2_boot_time_is_stated_and_never_ranked():
    """Stage 2's boots mix cold and warm caches, so a ranking would compare unlike things."""
    items = by_id(verdict.build(swap(ready=682.1), [peer(top1=0.97, configs=[]), {"model": "x/y", "results": swap(ready=236.0)}]))
    boot = items["swap-boot"]
    assert boot["tone"] == "info" and "682 s" in boot["headline"]
    assert "not compared" in boot["detail"].lower() and "236" not in boot["detail"]


def test_a_config_boot_is_compared_only_with_the_same_chips_and_package():
    same = peer("a/one", configs=[cfg(ready=118.0), cfg(ready=218.0)])
    other_chips = peer("a/two", configs=[cfg(chips=4, ready=9999.0)])
    other_pkg = peer("a/three", configs=[cfg(ready=1.0, package="someone/else-p300")])
    item = by_id(verdict.build({"configs": [cfg(ready=132.0)]}, [same, other_chips, other_pkg]))["config-2"]
    assert "118 s" in item["detail"] and "218 s" in item["detail"]
    assert "9,999" not in item["detail"] and "9999" not in item["detail"] and " 1 s" not in item["detail"]


def test_one_comparable_run_is_not_enough_to_call_a_boot_typical():
    item = by_id(verdict.build({"configs": [cfg(ready=132.0)]}, [peer("a/one", configs=[cfg(ready=118.0)])]))["config-2"]
    assert "not enough" in item["detail"].lower() and item["tone"] == "good"


@pytest.mark.parametrize("ready,word,tone", [(100.0, "quicker than", "good"), (170.0, "about the same as", "good"),
                                             (300.0, "slower than", "warn")])
def test_a_config_boot_is_called_quick_typical_or_slow_against_the_median_of_the_earlier_runs(ready, word, tone):
    peers = [peer("a/one", configs=[cfg(ready=150.0)]), peer("a/two", configs=[cfg(ready=160.0)]), peer("a/three", configs=[cfg(ready=170.0)])]
    item = by_id(verdict.build({"configs": [cfg(ready=ready)]}, peers))["config-2"]
    assert word in item["detail"] and item["tone"] == tone


def test_a_failed_run_of_the_package_is_not_a_yardstick_for_boot_time():
    peers = [peer("a/one", configs=[cfg(ready=100.0)]), peer("a/bad", configs=[cfg(ready=5.0, ok=False)])]
    item = by_id(verdict.build({"configs": [cfg(ready=105.0)]}, peers))["config-2"]
    assert "not enough" in item["detail"].lower() and "1 found" in item["detail"] and " 5 s" not in item["detail"]


def test_the_boot_is_judged_against_the_median_not_the_quickest_earlier_run():
    peers = [peer("a/one", configs=[cfg(ready=100.0)]), peer("a/two", configs=[cfg(ready=200.0)]), peer("a/three", configs=[cfg(ready=300.0)])]
    item = by_id(verdict.build({"configs": [cfg(ready=210.0)]}, peers))["config-2"]
    assert "about the same as" in item["detail"] and "median 200 s" in item["detail"] and item["tone"] == "good"


def test_a_failed_config_is_bad_and_carries_the_reason():
    item = by_id(verdict.build({"configs": [cfg(chips=4, ok=False, top1=None, ready=None, reason="Fabric Router Sync: Timeout")]}, []))["config-4"]
    assert item["tone"] == "bad" and "4 chips" in item["headline"] and "Fabric Router Sync" in item["detail"]


def test_a_config_that_was_not_run_is_left_out():
    assert "config-1" not in by_id(verdict.build({"configs": [cfg(chips=1, ok=None, top1=None, ready=None)]}, []))


def test_a_config_below_the_bar_is_bad_even_if_it_passed_its_own_gate():
    item = by_id(verdict.build({"configs": [cfg(top1=SWAP_TOP1_MIN - 0.2)]}, []))["config-2"]
    assert item["tone"] == "bad"


def test_speed_is_reported_as_not_measured_when_hardware_ran_and_nothing_timed_generation():
    assert by_id(verdict.build(swap(), []))["speed"]["tone"] == "info"
    assert "tokens per second" in by_id(verdict.build(swap(), []))["speed"]["headline"]
    assert "speed" not in by_id(verdict.build({"triage": {"class": "full-port"}}, []))      # nothing ran on a chip: nothing to say
    assert "speed" not in by_id(verdict.build({**swap(), "speed": {"decode_tok_s": 40.0}}, []))


def test_triage_says_a_vocabulary_trim_is_served_by_the_installed_bundle_with_changes():
    t = by_id(verdict.build({"triage": {"class": "vocab-trim", "path": "weights-only"}}, []))["triage"]
    assert t["tone"] == "info" and "vocabulary" in t["headline"] and "drafter" in t["headline"]
    assert "full port" not in t["headline"].lower() and t["detail"] == "Class vocab-trim."


def test_triage_says_what_the_class_means_for_an_unattended_run():
    assert by_id(verdict.build({"triage": {"class": "weights-only"}}, []))["triage"]["tone"] == "info"
    full = by_id(verdict.build({"triage": {"class": "unknown", "path": "full-port"}}, []))["triage"]
    assert full["tone"] == "warn" and "new model code" in full["headline"]


def test_the_worst_news_comes_first():
    res = {**swap(top1=1.0), "configs": [cfg(chips=4, ok=False, top1=None, ready=None, reason="x")], "triage": {"class": "unknown", "path": "full-port"}}
    tones = [i["tone"] for i in verdict.build(res, [])]
    assert tones == sorted(tones, key=verdict.TONES.index)
    assert tones[0] == "bad"


# ---- the reference check reads evidence, not the agent's say-so ------------------------------------------------------------------

def write_ref(run: Path, *, claimed="pass", measured="pass", missing=(), keys=851, ignore=("^model.visual.*", "^mtp.*")):
    s1 = run / "stages" / "1"
    (s1 / "evidence").mkdir(parents=True)
    (s1 / "reference.json").write_text(json.dumps({"verdict": claimed, "checks": [{"name": "loads", "pass": claimed == "pass"}]}))
    (s1 / "evidence" / "summary.json").write_text(json.dumps({"verdict": measured, "checks": {"loads": measured == "pass"}}))
    (s1 / "evidence" / "load-report.json").write_text(json.dumps(
        {"missing_keys": list(missing), "model_key_count": keys, "ignore_patterns": list(ignore)}))


def test_a_reference_the_agent_passed_but_the_template_failed_is_bad_and_says_so(tmp_path):
    write_ref(tmp_path, claimed="pass", measured="fail", missing=[f"model.layers.{i}.w" for i in range(851)], keys=851)
    ref = verdict.reference_check(tmp_path)
    item = by_id(verdict.build({}, [], ref))["reference"]
    assert item["tone"] == "bad"
    assert "reference.json says pass" in item["headline"] and "summary.json" in item["headline"]
    assert "851 of 851" in item["detail"] and "uninitialised" in item["detail"]


def test_a_reference_that_failed_its_own_checks_is_bad(tmp_path):
    write_ref(tmp_path, claimed="fail", measured="fail")
    assert by_id(verdict.build({}, [], verdict.reference_check(tmp_path)))["reference"]["tone"] == "bad"


def test_a_reference_that_loaded_every_weight_is_good(tmp_path):
    write_ref(tmp_path)
    item = by_id(verdict.build({}, [], verdict.reference_check(tmp_path)))["reference"]
    assert item["tone"] == "good" and "every weight" in item["headline"]


def test_missing_keys_under_an_ignore_pattern_do_not_count(tmp_path):
    write_ref(tmp_path, missing=["model.visual.blocks.0.w", "mtp.fc.weight"])
    assert by_id(verdict.build({}, [], verdict.reference_check(tmp_path)))["reference"]["tone"] == "good"


def test_a_passing_reference_with_missing_weights_is_a_warning(tmp_path):
    write_ref(tmp_path, claimed="pass", measured="pass", missing=["model.layers.0.w"])
    item = by_id(verdict.build({}, [], verdict.reference_check(tmp_path)))["reference"]
    assert item["tone"] == "warn" and "1 of 851" in item["detail"]


def test_no_reference_files_means_no_reference_line(tmp_path):
    assert verdict.reference_check(tmp_path) is None
    assert "reference" not in by_id(verdict.build({}, [], None))


def test_a_reference_file_that_links_outside_the_run_is_ignored(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"verdict": "fail"}))
    run = tmp_path / "run"
    write_ref(run)
    (run / "stages" / "1" / "evidence" / "summary.json").unlink()
    (run / "stages" / "1" / "evidence" / "summary.json").symlink_to(outside)
    ref = verdict.reference_check(run)
    assert ref is None or ref["measured"] is None


def test_an_oversized_reference_file_is_not_read(tmp_path):
    write_ref(tmp_path)
    # valid JSON that is over the cap, so only the cap can keep it from being read
    (tmp_path / "stages" / "1" / "evidence" / "load-report.json").write_text(
        json.dumps({"missing_keys": [], "model_key_count": 851, "ignore_patterns": [], "pad": "x" * (3 * 1024 * 1024)}))
    ref = verdict.reference_check(tmp_path)
    assert ref is None or ref["missing"] is None


# ---- finding 13: a number that is not finite is not a measurement ------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("n", [0, 32])
def test_a_non_finite_agreement_gives_no_swap_verdict_and_does_not_raise(bad, n):
    items = by_id(verdict.build(swap(top1=bad, n=n), []))
    assert "swap" not in items


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_boot_time_is_not_stated_or_ranked(bad):
    assert "swap-boot" not in by_id(verdict.build(swap(ready=bad), []))
    items = by_id(verdict.build({"configs": [cfg(top1=bad, ready=bad)]}, []))
    assert "config-2" not in items                                  # neither number is usable, so there is nothing to say


def test_a_non_finite_peer_number_is_left_out_of_the_yardstick():
    peers = [peer(top1=float("nan")), peer("org/ok", top1=0.9375)]
    detail = by_id(verdict.build(swap(top1=1.0), peers))["swap"]["detail"]
    assert "nan" not in detail and "0.94" in detail


def test_a_very_large_integer_is_still_a_number():
    assert verdict._num(10**400) is True
    assert verdict._num(float("nan")) is False and verdict._num(True) is False


# ---- a boot time is ranked only against a recorded package ---------------------------------------------------------------------------

def test_runs_with_no_recorded_package_are_not_grouped_together():
    peers = [peer(configs=[cfg(package=None, ready=r)]) for r in (100.0, 110.0, 120.0)]
    item = by_id(verdict.build({"configs": [cfg(package=None, ready=900.0)]}, peers))["config-2"]
    assert item["tone"] == "good" and "slower" not in item["detail"]
    assert "package" in item["detail"] and "not recorded" in item["detail"]
