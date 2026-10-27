# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The three recorded facts of a run (docs/superpowers/specs/2026-10-15-audio-bringup-design.md, section 3): form, serving and
modality. Stage 0 decides them, its passing `stage_end` holds them, and a ledger with none of them means a text run served by a thin
bundle, so every run from before keeps its stage table."""
import pytest

from orchard import audio_facts, classes
from orchard.audio_facts import DEFAULT_FACTS, facts_problems, serving_of
from orchard.nearest import Installed
from orchard.stages import gate_delta, run_facts
from test_stages import GOOD_DELTA, delta_file, stage, stage0_pass

AUDIO = {"form": "full", "serving": "container", "modality": "audio"}


def test_the_value_sets_are_the_documented_ones():
    assert audio_facts.FORMS == ("adapter", "full")
    assert audio_facts.SERVINGS == ("thin-bundle", "container", "none")
    assert audio_facts.MODALITIES == ("text", "audio")


def test_the_default_is_a_text_run_on_a_thin_bundle():
    assert DEFAULT_FACTS == {"form": "full", "serving": "thin-bundle", "modality": "text"}


def test_a_good_record_has_no_problems():
    assert facts_problems(AUDIO) == []
    assert facts_problems(DEFAULT_FACTS) == []


@pytest.mark.parametrize("bad", [None, [], "audio", 3, {}])
def test_a_record_that_is_not_a_full_dict_is_refused(bad):
    assert facts_problems(bad)


@pytest.mark.parametrize("key,value", [("form", "lora"), ("serving", "bundle"), ("modality", "video"), ("form", None),
                                       ("serving", True), ("modality", ["audio"])])
def test_a_value_outside_its_set_is_named(key, value):
    problems = facts_problems({**AUDIO, key: value})
    assert len(problems) == 1 and key in problems[0]


def test_an_unknown_key_is_refused():
    assert any("surprise" in p for p in facts_problems({**AUDIO, "surprise": 1}))


def test_a_missing_key_is_refused():
    partial = dict(AUDIO)
    del partial["serving"]
    assert any("serving" in p for p in facts_problems(partial))


# ---- serving_of -------------------------------------------------------------------------------

BUNDLE = Installed("org/qwen-p300", "Qwen/Qwen3.8-27B", "6", 2)
PACKAGE = Installed("org/music-pkg", "org/music-base", "5.1", 4)


def test_a_schema_6_bundle_is_a_thin_bundle():
    assert serving_of("Qwen/Qwen3.8-27B", [BUNDLE]) == "thin-bundle"


def test_a_schema_5_1_package_is_a_container():
    assert serving_of("org/music-base", [PACKAGE]) == "container"


def test_a_thin_bundle_wins_over_a_container_for_the_same_base():
    both = [Installed("org/pkg", "Qwen/Qwen3.8-27B", "5.1", 4), BUNDLE]
    assert serving_of("Qwen/Qwen3.8-27B", both) == "thin-bundle"


def test_a_base_nothing_serves_is_none():
    assert serving_of("stabilityai/stable-audio-3-small-music-base", [BUNDLE, PACKAGE]) == "none"
    assert serving_of("x/y", []) == "none"


def test_a_name_that_only_contains_the_repo_does_not_count():
    assert serving_of("org/music", [Installed("a/b", "org/music-base", "6", 1)]) == "none"


def test_a_revision_on_the_base_id_is_ignored():
    assert serving_of("Qwen/Qwen3.8-27B@" + "a" * 40, [BUNDLE]) == "thin-bundle"


# ---- the ledger reader ------------------------------------------------------------------------

def test_the_facts_are_read_from_stage_0s_passing_end(tmp_path):
    assert run_facts(stage0_pass(path="weights-only", facts=AUDIO), tmp_path) == AUDIO


def test_a_ledger_without_facts_gets_the_default(tmp_path):
    assert run_facts(stage0_pass(path="weights-only"), tmp_path) == DEFAULT_FACTS


def test_before_stage_0_passes_there_are_no_facts(tmp_path):
    assert run_facts([], tmp_path) is None


def test_a_record_that_fails_the_check_gives_none(tmp_path):
    assert run_facts(stage0_pass(path="weights-only", facts={**AUDIO, "modality": "video"}), tmp_path) is None


def test_the_ledger_wins_over_a_delta_json_edited_later(tmp_path):
    delta_file(tmp_path, {**GOOD_DELTA, "facts": {**AUDIO, "modality": "text"}})
    assert run_facts(stage0_pass(path="weights-only", facts=AUDIO), tmp_path)["modality"] == "audio"


# ---- the stage 0 gate -------------------------------------------------------------------------

def gate(tmp_path, delta):
    return gate_delta(*stage(tmp_path, 0, "delta.json", delta))


def test_a_delta_with_audio_facts_passes(tmp_path):
    assert gate(tmp_path, {**GOOD_DELTA, "class": "weights-only", "facts": AUDIO}).ok


def test_a_delta_with_bad_facts_fails_and_names_the_value(tmp_path):
    r = gate(tmp_path, {**GOOD_DELTA, "class": "weights-only", "facts": {**AUDIO, "form": "lora"}})
    assert not r.ok and any("form" in x for x in r.reasons)


def test_a_delta_without_facts_still_passes(tmp_path):
    assert gate(tmp_path, {**GOOD_DELTA}).ok


@pytest.mark.parametrize("cls", ["restricted-head", "vocab-trim", "stock-runtime", "weights+sidecar"])
def test_an_audio_run_cannot_take_a_text_only_class(tmp_path, cls):
    d = {**GOOD_DELTA, "class": cls, "path": classes.path_of(cls), "facts": AUDIO}
    r = gate(tmp_path, d)
    assert any("audio" in x for x in r.reasons)


def test_stage_0s_template_writes_form_and_modality_only(tmp_path):
    assert gate(tmp_path, {**GOOD_DELTA, "class": "weights-only", "facts": {"form": "full", "modality": "audio"}}).ok


def test_the_delta_still_needs_form_and_modality(tmp_path):
    r = gate(tmp_path, {**GOOD_DELTA, "class": "weights-only", "facts": {"modality": "audio"}})
    assert any("facts.form is missing" in x for x in r.reasons)


def test_a_missing_serving_is_refused_in_the_ledger_record_but_not_in_the_delta():
    assert facts_problems({"form": "full", "modality": "audio"})
    assert not facts_problems({"form": "full", "modality": "audio"}, required=("form", "modality"))


def test_a_text_model_may_take_any_class(tmp_path):
    d = {**GOOD_DELTA, "class": "vocab-trim", "path": "weights-only", "facts": DEFAULT_FACTS,
         "vocab_trim": {"trimmed": True, "kept_subset": True, "model_vocab": 1, "base_vocab": 2}}
    assert not any("text class" in x for x in gate(tmp_path, d).reasons)


def test_a_failed_stage_0_end_is_not_read(tmp_path):
    entries = [{"seq": 1, "event": "stage_end", "stage": 0, "data": {"result": "fail", "facts": AUDIO}}]
    assert run_facts(entries, tmp_path) is None


def test_a_list_value_is_refused_without_an_exception():
    assert facts_problems({**AUDIO, "form": ["full"]})
    assert facts_problems({**AUDIO, "form": {"a": 1}})
