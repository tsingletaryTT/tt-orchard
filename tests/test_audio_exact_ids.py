# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 152: the audio preflight reads a cached model config the way stage 0 does, and compares repository ids exactly.

(1) A float that overflows (`1e999`) is refused during snapshot parsing, as NaN and Infinity already are, so such a config cannot pass audio
detection. (2) A conditioner's `repo_id` names one of the two repositories only when it equals that id exactly: same case, same owner, same name,
no prefix and no substring. The stage 0 template compares with `in (model_repo, base_repo)`; the preflight's list of component directories
that a snapshot must hold has to give the same answer."""
from __future__ import annotations

import json

import pytest

from orchard import audio_facts
from orchard import preflight as pf
from test_delta_triage_template import template_module

OWN = "stabilityai/stable-audio-3-small-music"
BASE = "stabilityai/stable-audio-3-small-music-base"


def conf(repo, sub="t5gemma-b-b-ul2"):
    return {"model_type": "diffusion_cond", "model": {"conditioning": {"configs": [{"id": "prompt", "type": "t5gemma",
                                                                                    "config": {"repo_id": repo, "subfolder": sub}}]}}}


# ---- (1) floats that overflow --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("number", ["1e999", "-1e999", "1E+999", "1.5e400", "0.1e1000"])
def test_a_float_that_overflows_is_refused_when_the_config_is_parsed(number):
    raw = ('{"model_type": "x", "model": {}, "a": ' + number + "}").encode()
    cfg, why = audio_facts.audio_config_problem(raw)
    assert cfg is None and why and "not readable JSON" in why


@pytest.mark.parametrize("number", ["1e999", "-1e999", "1.5e400"])
def test_a_nested_overflowing_float_is_refused_too(number):
    raw = ('{"model_type": "x", "model": {"a": [1, {"b": ' + number + "}]}}").encode()
    assert audio_facts.audio_config_problem(raw)[0] is None


@pytest.mark.parametrize("number", ["1e308", "-1e308", "1.5", "0.0", "1e-999", "2.5e10"])
def test_a_float_that_fits_is_kept(number):
    cfg, why = audio_facts.audio_config_problem(('{"model_type": "x", "model": {}, "a": ' + number + "}").encode())
    assert why is None and cfg["a"] == float(number)


def test_the_preflight_and_stage_0_agree_on_an_overflowing_float(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    (d / "model_config.json").write_text('{"model_type": "x", "model": {}, "a": 1e999}')
    (d / "model.safetensors").write_text("w")
    assert pf.snapshot_audio_problem(d)
    assert template_module().detect_audio(d) is False
    hub = pf.parse_hub({"id": OWN, "pipeline_tag": "text-to-audio", "tags": ["text-to-audio"],
                        "siblings": [{"rfilename": "model_config.json"}, {"rfilename": "model.safetensors"}]})
    assert pf.audio_evidence_for(hub, d) is None


def test_a_stock_runtime_config_with_an_overflowing_float_is_unreadable_and_keeps_the_row(tmp_path):
    """The stock-runtime compatibility check parses config.json files too. A 1e999 there is an unreadable config (the row is kept, as it is
    for a file that is not JSON), and is never compared as infinity against another infinity."""
    found, snap = tmp_path / "params", tmp_path / "snap"
    for d in (found, snap):
        d.mkdir()
    (found / "config.json").write_text('{"architectures": ["AForCausalLM"], "rope_theta": 1e999}')
    (snap / "config.json").write_text('{"architectures": ["BForCausalLM"], "rope_theta": 1e999}')
    hub = pf.parse_hub({"id": "a/b", "config": {"architectures": ["BForCausalLM"]}})
    # With a readable parameters file and a readable snapshot config the two architectures differ, so the row would be dropped.
    # With the overflowing float the parameters file is unreadable, and the row is kept.
    assert pf._stock_compatible(found, "a/b", hub, lambda m: snap) is True


# ---- (2) repository ids match exactly ---------------------------------------------------------------------------------------

def subs(repo, own=OWN, partners=()):
    return audio_facts.component_subfolders(conf(repo), own, partners)


def test_a_repo_id_equal_to_the_models_own_id_names_a_component_directory():
    assert subs(OWN) == ["t5gemma-b-b-ul2"]


@pytest.mark.parametrize("other", [
    "StabilityAI/stable-audio-3-small-music",            # case
    "stabilityai/Stable-Audio-3-Small-Music",
    "stabilityai/stable-audio-3-small-music-base",       # the same name with more at the end (a prefix match would accept it)
    "stabilityai/stable-audio-3-small",                  # a prefix of the id
    "stabilityai/stable-audio-3-small-musi",
    "stabilityai/other-audio",                           # same owner, another name
    "otherorg/stable-audio-3-small-music",               # same name, another owner
    "stabilityai/stable-audio-3-small-music ",           # trailing space
    "stabilityai/stable-audio-3-small-music@abc",        # a revision is not part of the id a config names
    "stabilityai",                                       # an owner alone
    "stabilityai/",
    "",
])
def test_a_repo_id_that_is_not_exactly_the_models_id_names_nothing(other):
    assert subs(other) == []


def test_the_other_repository_of_the_pair_counts_only_when_it_is_named_exactly():
    assert subs(BASE, partners=(BASE,)) == ["t5gemma-b-b-ul2"]
    assert subs("stabilityai/stable-audio-3-small-music-bas", partners=(BASE,)) == []
    assert subs("STABILITYAI/stable-audio-3-small-music-base", partners=(BASE,)) == []
    assert subs(BASE) == []                                                       # no partner given: only the model's own id


@pytest.mark.parametrize("own", [None, "", "no-slash", 7, ["a/b"]])
def test_a_model_id_that_is_not_an_owner_and_a_name_matches_nothing(own):
    assert audio_facts.component_subfolders(conf(OWN), own) == []


@pytest.mark.parametrize("both", ["no-slash", "stabilityai/", "/name", "", "a/b/c"])
def test_two_equal_texts_that_are_not_an_owner_and_a_name_are_not_a_repository(both):
    assert audio_facts.same_repository(both, both) is False
    assert audio_facts.component_subfolders(conf(both), both) == []


def _snapshot(tmp_path, repo_in_config, name="stable-audio-3-small-music"):
    d = tmp_path / name / "hub" / f"models--stabilityai--{name}" / "snapshots" / "rev"
    d.mkdir(parents=True)
    (d / "model_config.json").write_text(json.dumps(conf(repo_in_config)))
    (d / "model.safetensors").write_text("w")
    return d


def test_the_preflight_requires_the_text_encoder_directory_only_for_an_exact_repo_id(tmp_path):
    """The preflight asks for the component's config.json exactly when stage 0 would compare that directory."""
    assert pf.missing_audio_components(_snapshot(tmp_path, OWN)) == ["t5gemma-b-b-ul2/config.json"]
    assert pf.missing_audio_components(_snapshot(tmp_path / "b", "StabilityAI/stable-audio-3-small-music")) == []
    assert pf.missing_audio_components(_snapshot(tmp_path / "c", "stabilityai/stable-audio-3-small-music-x")) == []


def test_a_model_of_the_planned_pair_whose_config_names_the_other_repository_still_needs_the_directory(tmp_path):
    assert pf.missing_audio_components(_snapshot(tmp_path, BASE)) == ["t5gemma-b-b-ul2/config.json"]


def test_a_model_outside_the_planned_pair_has_no_partner_and_only_its_own_id_counts(tmp_path):
    own = "stabilityai/other-audio"
    assert pf.missing_audio_components(_snapshot(tmp_path, own, name="other-audio")) == ["t5gemma-b-b-ul2/config.json"]
    assert pf.missing_audio_components(_snapshot(tmp_path / "b", BASE, name="other-audio")) == []


def test_the_template_and_the_preflight_give_the_same_answer_for_each_repo_id(tmp_path):
    """The template compares a conditioner's repo_id with `in (model_repo, base_repo)`. Run its comparison on the same configs."""
    tm = template_module()
    for repo in (OWN, BASE, "StabilityAI/stable-audio-3-small-music", "stabilityai/stable-audio-3-small", "otherorg/x"):
        rows, _ = tm.audio_text_encoders(conf(repo), conf(repo), tmp_path, tmp_path, OWN, BASE)
        compared = bool(rows) and rows[0]["compared"]
        required = bool(audio_facts.component_subfolders(conf(repo), OWN, (BASE,)))
        assert compared == required, repo
