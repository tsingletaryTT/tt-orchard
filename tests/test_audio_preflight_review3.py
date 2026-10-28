# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review round 3 of the audio preflight (PR 147): a failed bundle search is an unknown and never an install, the cached config must be what
stage 0's detector accepts, and component configs count toward a complete snapshot. Helpers come from test_audio_preflight."""
from __future__ import annotations

import json

import pytest

from orchard import cli
from orchard import preflight as pf
from test_audio_preflight import (AUDIO_FILES, BASE, BASE_REV, MODEL, Downloader, Pipe, Recorder, _panel, audio_base, audio_payload,  # noqa: F401
                                  base_hub, conf, full_snaps, model_config, rows, run_with, snap, world)
from test_delta_triage_template import template_module
from test_preflight import signals as plain_signals


# ---- a failed search is an unknown, never an install ----------------------------------------------------

class Search:
    """A bundle search that answers from a script: a list, None (no search could be made), or an exception to raise."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def __call__(self, query):
        self.calls += 1
        a = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        if isinstance(a, Exception):
            raise a
        return a


def bringup_base(conf, tmp_path, base, search, *, args=(), hub=None, installed=()):
    over = {"hub_info": hub} if hub else {}
    sig = world(local=full_snaps(tmp_path), search_bundles=search, installed_bundles=lambda: list(installed), **over)
    pulled = []
    rec = Recorder()
    code = cli.main(["--config", str(conf), "bringup", MODEL, "--base", base, *args], env={}, stdout=Pipe(), signals=sig,
                    supervisor_main=rec.supervisor, fetcher=rec.fetcher, puller=lambda b: pulled.append(b) or (True, ""))
    return code, pulled, rec


@pytest.mark.parametrize("answer", [None, OSError("tt not found"), RuntimeError("boom")])
def test_a_failed_search_for_an_audio_model_blocks_with_a_retryable_message_and_installs_nothing(conf, tmp_path, capsys, answer):
    code, pulled, rec = bringup_base(conf, tmp_path, "eps/audio8-p150", Search(answer))
    err = capsys.readouterr().err
    assert code == 2 and pulled == [] and rec.argv is None and "search" in err and "again" in err


def test_a_search_that_fails_once_and_then_works_never_reaches_the_puller(conf, tmp_path):
    s = Search(None, [{"name": "eps/audio8-p150", "installed": False}])
    code, pulled, rec = bringup_base(conf, tmp_path, "eps/audio8-p150", s)
    assert code == 2 and pulled == [] and s.calls == 1


def test_an_audio_model_with_a_model_id_for_base_searches_once_and_never_pulls(conf, tmp_path):
    s = Search([])
    code, pulled, rec = bringup_base(conf, tmp_path, BASE, s)
    assert code == 0 and pulled == [] and s.calls == 1


@pytest.mark.parametrize("answer", [None, [{"name": "eps/audio8-p150", "installed": False}]])
def test_when_the_hub_cannot_say_whether_the_model_is_audio_a_bundle_base_is_not_pulled(conf, tmp_path, capsys, answer):
    hub = lambda m: (None, "HTTP 500")                       # noqa: E731
    code, pulled, rec = bringup_base(conf, tmp_path, "eps/audio8-p150", Search(answer), hub=hub, args=["--dry-run"])
    assert code == 2 and pulled == [] and "refused" in capsys.readouterr().err


def test_when_the_hub_cannot_say_a_model_id_for_base_is_not_pulled(conf, tmp_path):
    hub = lambda m: (None, "HTTP 500")                       # noqa: E731
    code, pulled, rec = bringup_base(conf, tmp_path, BASE, Search([]), hub=hub, args=["--dry-run"])
    assert pulled == []


def test_a_text_model_with_a_bundle_for_base_still_installs_it_after_a_good_search(conf):
    pulled = []
    s = plain_signals(search_bundles=lambda q: [{"name": "x/clef-p150", "installed": False}])
    cli.main(["--config", str(conf), "bringup", "Cloudflare/clef", "--base", "x/clef-p150"], env={}, stdout=Pipe(), signals=s,
             supervisor_main=Recorder().supervisor, fetcher=Recorder().fetcher, puller=lambda b: pulled.append(b) or (False, "stop here"))
    assert pulled == ["x/clef-p150"]


def test_the_panel_blocks_on_a_failed_search_for_an_audio_model(conf, tmp_path, monkeypatch):
    sig = world(local=full_snaps(tmp_path), search_bundles=Search(None))
    base = next(x for x in _panel(conf, sig, monkeypatch, base="eps/audio8-p150") if x.name == "base")
    assert base.status == "block" and "again" in base.detail


# ---- the cached config must be what stage 0's detector accepts -------------------------------------------

def cfg_dir(tmp_path, text, name="c"):
    d = tmp_path / name
    d.mkdir()
    (d / "model_config.json").write_text(text)
    (d / "model.safetensors").write_text("w")
    return d


BAD_CONFIGS = {
    "empty-object": "{}", "model-not-object": '{"model_type": "x", "model": 3}', "no-model-type": '{"model": {}}',
    "type-not-string": '{"model_type": 7, "model": {}}', "not-json": "{nope", "list": "[]", "empty-file": "",
    "duplicate-key": '{"model_type": "x", "model_type": "y", "model": {}}', "nan": '{"model_type": "x", "model": {}, "a": NaN}',
    "huge-int": '{"model_type": "x", "model": {}, "a": ' + "9" * 40 + "}", "too-deep": '{"model_type": "x", "model": ' + "[" * 40 + "]" * 40 + "}",
    "too-big": '{"model_type": "x", "model": {}, "pad": "' + "a" * (1 << 20) + '"}',
}


@pytest.mark.parametrize("name", sorted(BAD_CONFIGS))
def test_a_cached_audio_config_stage_0_would_reject_is_a_problem_and_never_audio(tmp_path, name):
    d = cfg_dir(tmp_path, BAD_CONFIGS[name])
    assert pf.snapshot_audio_problem(d)
    assert template_module().detect_audio(d) is False                     # the two readers agree
    assert pf.audio_evidence_for(pf.parse_hub(audio_payload()), d) is None


def test_a_valid_cached_audio_config_is_accepted_by_both_readers(tmp_path):
    d = cfg_dir(tmp_path, json.dumps(model_config(MODEL)))
    assert pf.snapshot_audio_problem(d) is None and template_module().detect_audio(d) is True
    assert pf.audio_evidence_for(pf.parse_hub(audio_payload()), d)


def test_a_missing_or_unreadable_config_is_a_problem(tmp_path):
    assert pf.snapshot_audio_problem(tmp_path / "nowhere")
    d = tmp_path / "d"
    d.mkdir()
    (d / "model_config.json").mkdir()                                       # a directory where the file should be
    assert pf.snapshot_audio_problem(d)


def test_the_config_reader_stops_at_its_size_limit(tmp_path, monkeypatch):
    d = cfg_dir(tmp_path, BAD_CONFIGS["too-big"])
    reads = []
    real = open

    def spy(path, *a, **k):
        f = real(path, *a, **k)
        if "model_config.json" not in str(path):
            return f

        class Spy:
            def read(self, n=-1):
                reads.append(n)
                return f.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *e):
                f.close()
                return False
        return Spy()
    monkeypatch.setattr("builtins.open", spy)
    assert pf.snapshot_audio_problem(d)
    assert reads and all(isinstance(n, int) and 0 < n <= (1 << 20) + 1 for n in reads)


def test_a_cached_snapshot_with_a_bad_config_is_not_complete(tmp_path):
    assert pf.snapshot_complete(cfg_dir(tmp_path, "{}")) is False


def test_a_cached_snapshot_needs_its_text_encoder_config(tmp_path):
    d = cfg_dir(tmp_path, json.dumps(model_config(MODEL)), name="models--stabilityai--stable-audio-3-small-music")
    sub = d / "t5gemma-b-b-ul2"
    sub.mkdir()
    (sub / "model.safetensors").write_text("w")
    assert pf.snapshot_complete(d) is False                                 # the text encoder's config.json is missing
    (sub / "config.json").write_text("{}")
    assert pf.snapshot_complete(d) is True


def test_a_text_encoder_that_loads_from_another_organisation_is_not_required_locally(tmp_path):
    d = cfg_dir(tmp_path, json.dumps(model_config("google/t5gemma-b-b-ul2")), name="models--stabilityai--stable-audio-3-small-music")
    assert pf.snapshot_complete(d) is True


@pytest.mark.parametrize("sub", ["../escape", "/abs", 7, ""])
def test_a_hostile_subfolder_is_never_required_or_followed(tmp_path, sub):
    d = cfg_dir(tmp_path, json.dumps(model_config(MODEL, sub)), name="models--stabilityai--stable-audio-3-small-music")
    assert pf.snapshot_complete(d) is True


def test_the_model_with_a_bad_cached_config_leaves_the_audio_path(tmp_path):
    snaps = full_snaps(tmp_path)
    (snaps[MODEL] / "model_config.json").write_text("{}")
    assert rows(tmp_path, world(local=snaps))["base"].reason == "nearest-model-missing"


def test_a_base_with_a_bad_cached_config_leaves_the_audio_path(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV, names=AUDIO_FILES + ["svd_bases.pt"])
    (s / "model_config.json").write_text("{}")
    c = audio_base(base_info=(base_hub(gated=False), None), local={BASE: s})
    assert c.status == "block" and c.reason == "nearest-model-missing"


def test_a_downloaded_model_whose_config_stage_0_would_reject_stops_the_run(conf, tmp_path, capsys):
    local = {MODEL: full_snaps(tmp_path)[MODEL]}
    code, out, rec = run_with(conf, ["bringup", MODEL], Downloader(tmp_path, config={}), world(local=local, base_gated=False))
    assert code == 2 and rec.argv is None and "model_config.json" in capsys.readouterr().err


def test_a_downloaded_audio_pair_with_good_configs_runs(conf, tmp_path):
    local = {MODEL: full_snaps(tmp_path)[MODEL]}
    code, out, rec = run_with(conf, ["bringup", MODEL], Downloader(tmp_path), world(local=local, base_gated=False))
    assert code == 0 and rec.argv is not None


# ---- component configs ---------------------------------------------------------------------------------------

NO_ENCODER_CONFIG = [n for n in AUDIO_FILES + ["svd_bases.pt"] if n != "t5gemma-b-b-ul2/config.json"]


def test_a_cached_base_missing_a_listed_component_config_is_fetched_when_public(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV, names=NO_ENCODER_CONFIG)
    c = audio_base(base_info=(base_hub(gated=False), None), local={BASE: s})
    assert c.data["needs_fetch"] is True and "t5gemma-b-b-ul2/config.json" in c.detail


def test_a_cached_base_missing_a_listed_component_config_blocks_when_gated(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV, names=NO_ENCODER_CONFIG)
    c = audio_base(local={BASE: s})
    assert (c.status, c.reason) == ("block", "credentials-needed")


def test_a_cached_base_with_every_listed_file_is_complete(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV, names=AUDIO_FILES + ["svd_bases.pt"])
    assert audio_base(base_info=(base_hub(gated=False), None), local={BASE: s}).data["needs_fetch"] is False
