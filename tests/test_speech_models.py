# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speech models (orchard/speech_models.py): what the preflight recognises, what it says, and what it leaves alone.

The facts under test come from the two Voxtral bundles' cards and the upstream configs read on 2026-11-23
(docs/run-logs/2026-11-23-voxtral-speech-models.md). The Hub answers used here are small copies of the real ones: the tags, the
architectures and the base models a real answer carried.
"""
from __future__ import annotations

import pytest

from orchard import cli, preflight as pf, speech_models as sm

MINI_UP = "mistralai/Voxtral-Mini-3B-2507"
TTS_UP = "mistralai/Voxtral-4B-TTS-2603"
MINI_TT = "apande-TT/mistralai-voxtral-mini-3b-2507-tt"
TTS_TT = "apande-TT/mistralai-voxtral-4b-tts-2603-tt"


def hub(model_id, **kw):
    base = dict(id=model_id, sha="a" * 40, private=False, gated=False, license="apache-2.0", total_bytes=10**9, files=["config.json"],
                code_files=[], has_license_file=False)
    base.update(kw)
    return pf.HubInfo(**base)


# What the real Hub answers looked like on 2026-11-23: the upstream Mini card has no pipeline tag and a transformers config; the upstream
# TTS card has a tag and no config.json; each bundle's card names its upstream model as the base model.
MINI_UP_HUB = hub(MINI_UP, architectures=["VoxtralForConditionalGeneration"], tags=["voxtral", "vllm", "en"])
TTS_UP_HUB = hub(TTS_UP, license="cc-by-nc-4.0", pipeline_tag="text-to-speech", base_models=["mistralai/Ministral-3-3B-Base-2512"],
                 files=["params.json", "consolidated.safetensors"], tags=["text-to-speech"])
MINI_TT_HUB = hub(MINI_TT, pipeline_tag="automatic-speech-recognition", base_models=[MINI_UP])
TTS_TT_HUB = hub(TTS_TT, license="cc-by-nc-4.0", pipeline_tag="text-to-speech", base_models=[TTS_UP])


# ---- task_of: how a model is recognised -------------------------------------------------------------


@pytest.mark.parametrize("tag,expected", [("automatic-speech-recognition", "speech-to-text"), ("text-to-speech", "text-to-speech")])
def test_a_pipeline_tag_names_the_task(tag, expected):
    assert sm.task_of(tag, [], []) == expected


def test_the_voxtral_architecture_names_the_task_when_the_card_has_no_pipeline_tag():
    # The upstream Mini card has no pipeline tag, so the architecture is the only evidence.
    assert sm.task_of(None, ["voxtral"], ["VoxtralForConditionalGeneration"]) == "speech-to-text"


def test_a_pipeline_tag_that_says_something_else_decides_alone():
    # A text-generation model whose config happens to name the Voxtral class is what its card says it is.
    assert sm.task_of("text-generation", [], ["VoxtralForConditionalGeneration"]) is None


def test_a_tag_in_the_tag_list_counts_only_when_there_is_no_pipeline_tag():
    assert sm.task_of(None, ["text-to-speech"], []) == "text-to-speech"
    assert sm.task_of("text-generation", ["text-to-speech"], []) is None


@pytest.mark.parametrize("tag", ["text-to-audio", "audio-to-audio"])
def test_the_stable_audio_tags_are_not_speech(tag):
    assert sm.task_of(tag, [tag], []) is None


@pytest.mark.parametrize("bad", [None, 7, [], {}, "", ["x", 3, None]])
def test_wrong_shapes_are_not_speech_and_do_not_raise(bad):
    assert sm.task_of(bad, bad, bad) is None


def test_the_four_real_answers_are_recognised():
    got = {h.id: sm.task_of(h.pipeline_tag, h.tags, h.architectures) for h in (MINI_UP_HUB, TTS_UP_HUB, MINI_TT_HUB, TTS_TT_HUB)}
    assert got == {MINI_UP: "speech-to-text", TTS_UP: "text-to-speech", MINI_TT: "speech-to-text", TTS_TT: "text-to-speech"}


# ---- the known bundles --------------------------------------------------------------------------------


def test_each_upstream_model_and_each_bundle_finds_its_bundle():
    for repo in (MINI_UP, MINI_TT):
        assert sm.bundle_for(repo).bundle == MINI_TT
    for repo in (TTS_UP, TTS_TT):
        assert sm.bundle_for(repo).bundle == TTS_TT


def test_a_revision_on_the_id_is_ignored_and_a_different_case_is_a_different_repository():
    assert sm.bundle_for(MINI_UP + "@abc123").bundle == MINI_TT
    assert sm.bundle_for(MINI_UP.lower()) is None


@pytest.mark.parametrize("bad", [None, 3, "", "voxtral", "a/b/c", ["x"]])
def test_an_id_that_is_not_a_repository_has_no_bundle(bad):
    assert sm.bundle_for(bad) is None


def test_the_records_hold_what_the_cards_say():
    mini, tts = sm.bundle_for(MINI_UP), sm.bundle_for(TTS_UP)
    assert (mini.task, tts.task) == ("speech-to-text", "text-to-speech")
    assert mini.endpoints == ("POST /v1/audio/transcriptions", "POST /v1/chat/completions")
    assert tts.endpoints == ("POST /v1/audio/speech",)
    assert mini.license == "apache-2.0" and tts.license == "cc-by-nc-4.0"
    assert tts.non_commercial and not mini.non_commercial
    assert mini.chips == tts.chips == 1


# ---- the preflight's base row ---------------------------------------------------------------------------


def row(model_id, h, **kw):
    return pf.check_base(model_id=model_id, hub=h, base_override=kw.pop("base_override", None), local_snapshot=lambda m: None,
                         installed=[], search=lambda q: [], report=None, **kw)


@pytest.mark.parametrize("model_id,h,bundle", [(MINI_UP, MINI_UP_HUB, MINI_TT), (TTS_UP, TTS_UP_HUB, TTS_TT),
                                               (MINI_TT, MINI_TT_HUB, MINI_TT), (TTS_TT, TTS_TT_HUB, TTS_TT)])
def test_a_voxtral_model_gets_a_speech_row_that_names_its_bundle(model_id, h, bundle):
    c = row(model_id, h)
    assert c.name == "base" and c.status == pf.BLOCK and c.reason == "speech-model"
    assert bundle in c.detail
    assert "speech-check" in c.detail
    assert (c.data or {}).get("speech") and (c.data or {}).get("candidates") == []
    assert "nearest-model-missing" not in str(c.reason)


def test_the_row_says_what_the_task_is_and_what_the_endpoint_is():
    mini = row(MINI_UP, MINI_UP_HUB).detail
    assert "speech-to-text" in mini and "/v1/audio/transcriptions" in mini
    tts = row(TTS_UP, TTS_UP_HUB).detail
    assert "text-to-speech" in tts and "/v1/audio/speech" in tts


def test_the_text_to_speech_row_warns_about_the_non_commercial_licence():
    assert "non-commercial" in row(TTS_UP, TTS_UP_HUB).detail
    assert "non-commercial" not in row(MINI_UP, MINI_UP_HUB).detail


def test_the_row_does_not_start_a_run_and_says_why():
    d = row(MINI_UP, MINI_UP_HUB).detail
    assert "no speech stage" in d and "does not start a run" in d


def test_a_speech_model_with_no_known_bundle_still_gets_the_speech_row():
    other = hub("someone/some-asr", pipeline_tag="automatic-speech-recognition")
    c = row("someone/some-asr", other)
    assert c.status == pf.BLOCK and c.reason == "speech-model"
    assert "No published bundle is known" in c.detail and "tt-model search" in c.detail and "speech-check" in c.detail


def test_an_operator_base_does_not_turn_the_speech_row_back_into_a_text_run():
    assert row(MINI_UP, MINI_UP_HUB, base_override=MINI_TT).reason == "speech-model"


def test_a_text_model_is_not_touched():
    c = row("org/plain", hub("org/plain", architectures=["LlamaForCausalLM"], pipeline_tag="text-generation"))
    assert c.reason != "speech-model"


def test_a_model_the_audio_track_takes_is_not_given_the_speech_row():
    # `audio` is the evidence text for a model the audio track takes. A model whose tag list names both families is the audio track's.
    both = hub("org/both", tags=["text-to-speech", "text-to-audio"])
    assert sm.task_of(both.pipeline_tag, both.tags, both.architectures) == "text-to-speech"      # alone, the tags say speech
    assert row("org/both", both).reason == "speech-model"
    assert row("org/both", both, audio="tag text-to-audio; model_config.json and no config.json at the repository root").reason != "speech-model"


def test_a_text_runtime_model_keeps_its_runtime_row(monkeypatch):
    from orchard import text_runtime
    monkeypatch.setattr(text_runtime, "runtime_for_model", lambda m: type("R", (), {"name": "stand-in", "scripts_dir": __import__("pathlib").Path("scripts/x")})())
    c = row("org/tts-on-the-text-runtime", hub("org/tts-on-the-text-runtime", pipeline_tag="text-to-speech"))
    assert c.reason != "speech-model" and c.data.get("text_runtime") == "stand-in"


def test_no_hub_answer_leaves_the_old_path():
    c = row(MINI_UP, None)
    assert c.reason != "speech-model"


def test_a_dry_classification_cannot_go_on_for_a_speech_model():
    assert "speech-model" in pf.CLASSIFICATION_BLOCKS


def test_the_bundle_help_is_not_printed_for_a_speech_row(capsys):
    import io
    out = io.StringIO()
    cli._base_help(row(MINI_UP, MINI_UP_HUB), MINI_UP, out)
    assert out.getvalue() == ""


# ---- the text in the row is plain ----------------------------------------------------------------------------


def test_the_rows_avoid_the_phrases_the_writing_rules_forbid():
    for model_id, h in ((MINI_UP, MINI_UP_HUB), (TTS_UP, TTS_UP_HUB), ("someone/some-asr", hub("someone/some-asr", pipeline_tag="automatic-speech-recognition"))):
        text = row(model_id, h).detail
        for phrase in (", not ", "rather than", "instead of", "reads as", "honest", "the one", "which is why", "This is what"):
            assert phrase not in text, (model_id, phrase)
