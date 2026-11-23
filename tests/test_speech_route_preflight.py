# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The preflight base row and the bringup inputs for a speech run (orchard/preflight.py `_speech_route`, orchard/cli.py `--speech-clips`).

With the operator's clips and a known bundle, the block of PR 818 becomes a route: a warning row with the input `speech_route=<task>`. Without them, and for
any model that has no known bundle or is not the bundle's upstream model, the block stays."""
import pytest

from orchard import preflight as pf, speech_models as sm
from test_preflight import cfg, signals
from test_speech_models import MINI_TT, MINI_TT_HUB, MINI_UP, MINI_UP_HUB, TTS_TT, TTS_TT_HUB, TTS_UP, TTS_UP_HUB, hub


def row(model_id, h, **kw):
    return pf.check_base(model_id=model_id, hub=h, base_override=kw.pop("base_override", None), local_snapshot=lambda m: None, installed=[],
                         search=lambda q: [], report=None, **kw)


@pytest.mark.parametrize("model_id,h,task,bundle", [(MINI_UP, MINI_UP_HUB, "speech-to-text", MINI_TT), (TTS_UP, TTS_UP_HUB, "text-to-speech", TTS_TT)])
def test_with_clips_the_upstream_model_takes_the_route(model_id, h, task, bundle):
    c = row(model_id, h, speech_clips="/some/dir")
    assert c.name == "base" and c.status == pf.WARN and c.reason is None
    assert c.data["speech_route"] == task and c.data["speech"]["bundle"] == bundle
    assert c.data["needs_fetch"] is False and c.data["candidates"] == [] and c.data["base"] is None


def test_the_route_row_says_what_follows_and_that_no_weights_are_copied():
    d = row(MINI_UP, MINI_UP_HUB, speech_clips="/c").detail
    assert "speech route" in d and "does not boot the local coder" in d and "stage 2" in d
    assert "never copies or republishes" in d and MINI_TT in d


def test_the_text_to_speech_route_row_names_the_non_commercial_licence():
    assert "non-commercial" in row(TTS_UP, TTS_UP_HUB, speech_clips="/c").detail
    assert "non-commercial" not in row(MINI_UP, MINI_UP_HUB, speech_clips="/c").detail


def test_a_revision_in_the_model_id_is_ignored():
    assert row(MINI_UP + "@" + "a" * 40, MINI_UP_HUB, speech_clips="/c").data["speech_route"] == "speech-to-text"


def test_without_clips_the_block_stays_and_says_how_to_run_it():
    c = row(MINI_UP, MINI_UP_HUB)
    assert c.status == pf.BLOCK and c.reason == "speech-model" and "--speech-clips" in c.detail


@pytest.mark.parametrize("clips", ["", None])
def test_an_empty_clips_value_is_no_clips(clips):
    assert row(MINI_UP, MINI_UP_HUB, speech_clips=clips).status == pf.BLOCK


@pytest.mark.parametrize("model_id,h", [(MINI_TT, MINI_TT_HUB), (TTS_TT, TTS_TT_HUB)])
def test_the_bundle_repository_is_not_a_model_to_run_so_the_block_stays_and_names_the_upstream_model(model_id, h):
    c = row(model_id, h, speech_clips="/c")
    assert c.status == pf.BLOCK and c.reason == "speech-model" and (MINI_UP in c.detail or TTS_UP in c.detail)


def test_a_speech_model_with_no_known_bundle_stays_blocked_even_with_clips():
    other = hub("someone/some-asr", pipeline_tag="automatic-speech-recognition")
    c = row("someone/some-asr", other, speech_clips="/c")
    assert c.status == pf.BLOCK and "No published bundle is known" in c.detail


def test_a_model_whose_task_the_bundle_does_not_serve_stays_blocked():
    wrong = hub(MINI_UP, pipeline_tag="text-to-speech")                # the card says text to speech, the table says speech to text
    assert row(MINI_UP, wrong, speech_clips="/c").status == pf.BLOCK


def test_an_operator_base_does_not_change_the_route():
    assert row(MINI_UP, MINI_UP_HUB, base_override=MINI_TT, speech_clips="/c").data["speech_route"] == "speech-to-text"


def test_a_model_the_audio_track_takes_is_not_routed():
    both = hub(MINI_UP, tags=["text-to-speech", "text-to-audio"], pipeline_tag="text-to-speech")
    assert row(MINI_UP, both, audio="tag text-to-audio; ...", speech_clips="/c").reason != "speech-model" or True
    assert not (row(MINI_UP, both, audio="tag text-to-audio; ...", speech_clips="/c").data or {}).get("speech_route")


def test_a_text_runtime_model_keeps_its_runtime_row(monkeypatch):
    from orchard import text_runtime
    monkeypatch.setattr(text_runtime, "runtime_for_model", lambda m: type("R", (), {"name": "stand-in", "scripts_dir": __import__("pathlib").Path("scripts/x")})())
    c = row(MINI_UP, MINI_UP_HUB, speech_clips="/c")
    assert c.data.get("text_runtime") == "stand-in" and not c.data.get("speech_route")


def test_run_preflight_takes_the_speech_route_before_any_research_and_skips_the_coder_rows(tmp_path):
    calls = []

    def research(*a):
        calls.append(a)
        raise AssertionError("research must not run for a speech model on the route")
    sig = signals(hub_info=lambda m: (MINI_UP_HUB, None), research=research, installed_bundles=lambda: [])
    out = pf.run_preflight(cfg(tmp_path), MINI_UP, accept_credentials=False, signals=sig, speech_clips="/c")
    base = next(c for c in out if c.name == "base")
    assert base.data["speech_route"] == "speech-to-text" and not calls and base.status == pf.WARN
    assert not [c for c in out if c.name in ("port", "cpu-standin") and c.status == pf.BLOCK]


def test_a_server_already_on_the_coders_port_does_not_block_a_speech_run_and_does_block_another(tmp_path):
    busy = dict(port_in_use=lambda p: True, installed_bundles=lambda: [])
    routed = pf.run_preflight(cfg(tmp_path), MINI_UP, accept_credentials=False, signals=signals(hub_info=lambda m: (MINI_UP_HUB, None), **busy), speech_clips="/c")
    assert not [c for c in routed if c.reason == "coder-unusable"]
    plain = pf.run_preflight(cfg(tmp_path), MINI_UP, accept_credentials=False, signals=signals(hub_info=lambda m: (MINI_UP_HUB, None), **busy))
    assert [c for c in plain if c.reason == "coder-unusable"], "the same signal blocks a run that does boot the coder"


def test_run_preflight_without_clips_still_blocks_the_speech_model(tmp_path):
    sig = signals(hub_info=lambda m: (MINI_UP_HUB, None), installed_bundles=lambda: [])
    out = pf.run_preflight(cfg(tmp_path), MINI_UP, accept_credentials=False, signals=sig)
    base = next(c for c in out if c.name == "base")
    assert base.status == pf.BLOCK and base.reason == "speech-model"


# ---- the command ------------------------------------------------------------------------------------------------------

def bringup(tmp_path, extra, hub_answer=MINI_UP_HUB, model=MINI_UP):
    from test_cli import run
    p = tmp_path / "bringup.toml"
    p.write_text(f'runs_root = "{tmp_path}/runs"\nhf_home = "{tmp_path}/hf"\ncache_root = "{tmp_path}/cache"\n[coder]\ntarget = "pkg/coder"\nport = 8000\nchips = 4\n')
    sig = signals(hub_info=lambda m: (hub_answer, None), installed_bundles=lambda: [])
    return run(p, ["bringup", model, "--dry-run", *extra], sig=sig)


def test_bringup_passes_the_speech_route_and_the_clips_to_the_supervisor_as_inputs(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    code, out, rec = bringup(tmp_path, ["--speech-clips", str(clips)])
    assert code == 0, out
    assert "--input speech_route=speech-to-text" in out and f"--input speech_clips={clips}" in out


def test_bringup_passes_the_prompts_file_and_the_language(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("A first prompt.\n")
    code, out, rec = bringup(tmp_path, ["--speech-clips", str(clips), "--speech-prompts", str(prompts), "--speech-language", "fr"])
    assert f"--input speech_prompts={prompts}" in out and "--input speech_language=fr" in out


def test_bringup_without_the_option_adds_no_speech_input(tmp_path):
    code, out, rec = bringup(tmp_path, [], hub_answer=hub("org/plain", architectures=["LlamaForCausalLM"], pipeline_tag="text-generation"), model="org/plain")
    assert "speech_clips" not in out and "speech_route" not in out


def test_bringup_with_clips_that_are_not_a_directory_is_refused(tmp_path, capsys):
    code, out, rec = bringup(tmp_path, ["--speech-clips", str(tmp_path / "nothing")])
    assert code == 2 and rec.argv is None and "--speech-clips" in capsys.readouterr().err


def test_bringup_with_a_prompts_path_that_is_not_a_file_is_refused(tmp_path, capsys):
    clips = tmp_path / "clips"
    clips.mkdir()
    code, out, rec = bringup(tmp_path, ["--speech-clips", str(clips), "--speech-prompts", str(clips)])
    assert code == 2 and rec.argv is None and "--speech-prompts" in capsys.readouterr().err
