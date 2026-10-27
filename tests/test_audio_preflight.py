# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The preflight for an audio model (plan docs/superpowers/plans/2026-10-28-audio-bringup.md, slice A4).

A Stable Audio style model passes the `base` row without an installed bundle, because stage 0 compares the two repositories directly and
nothing can serve an audio model on a chip yet. The test model is stabilityai/stable-audio-3-small-music and its base. Everything runs on
injected signals and tiny fake snapshots: no network, no token, no chip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import audio_facts, bringup_config, cli, fetch
from orchard import preflight as pf
from orchard.nearest import Installed
from test_cli import Pipe, Recorder, conf, run  # noqa: F401  (conf is a fixture)
from test_delta_triage_template import template_module
from test_preflight import cfg, signals as plain_signals

MODEL = "stabilityai/stable-audio-3-small-music"
BASE = "stabilityai/stable-audio-3-small-music-base"
REV, BASE_REV = "8bdfb53c6b0f" + "0" * 28, "4249e9179cd6" + "0" * 28
AUDIO_FILES = ["model.safetensors", "model_config.json", "LICENSE.md", "t5gemma-b-b-ul2/config.json",
               "t5gemma-b-b-ul2/model.safetensors"]
GB = 1_000_000_000


def audio_payload(model_id=MODEL, *, sha=REV, gated="manual", files=AUDIO_FILES, **over):
    sib = [{"rfilename": f, "size": GB} for f in files]
    card = {"license": "other"}
    if model_id == MODEL:
        card["base_model"] = BASE
    return {"id": model_id, "private": False, "gated": gated, "sha": sha, "pipeline_tag": "text-to-audio",
            "tags": ["text-to-audio", "stable-audio-3"], "cardData": card, "siblings": sib, **over}


def snap(root: Path, repo: str, rev: str, names=("model.safetensors", "model_config.json")) -> Path:
    org, name = repo.split("/")
    d = root / "hub" / f"models--{org}--{name}" / "snapshots" / rev
    d.mkdir(parents=True)
    for n in names:
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        (d / n).write_text("{}" if n.endswith(".json") else "w")
    return d


# ---- the shared rule ---------------------------------------------------------------------------------

@pytest.mark.parametrize("names,expected", [
    (["model.safetensors", "model_config.json"], True),
    (["model.safetensors", "model_config.json", "t5/config.json"], True),         # a component's config is not the root's
    (["model.safetensors", "model_config.json", "config.json"], False),           # both config files: unclear
    (["model.safetensors", "config.json"], False),
    (["model_config.json"], False),                                                # no top-level weights
    (["sub/model.safetensors", "model_config.json"], False),
    ([], False),
    (["model.safetensors", "model_config.json", 7, None, {"a": 1}], True),         # non-strings are ignored
    ([7, None], False),
    (None, False),
    ("model_config.json", False),                                                  # a string is not a list of names
    ({"model_config.json": 1}, False),
])
def test_the_layout_rule_reads_the_root_of_the_file_list(names, expected):
    assert audio_facts.audio_layout(names) is expected


@pytest.mark.parametrize("pipeline,tags,expected", [
    ("text-to-audio", [], True), ("audio-to-audio", [], True),
    ("text-to-speech", ["text-to-audio"], False),       # a pipeline tag that is present decides alone
    ("text-generation", ["text-to-audio"], False),
    (None, ["text-to-audio"], True), (None, ["audio-to-audio", "x"], True),
    (None, ["text-to-speech"], False), (None, [], False), (None, None, False),
    (["text-to-audio"], ["text-to-audio"], True),       # a pipeline tag that is not a string is absent: the tags decide
    (7, ["text-to-audio"], True), (7, [], False),
    ("Text-To-Audio", [], False), ("text-to-audio ", [], False), ("", [], False),
    (None, "text-to-audio", False),                      # tags must be a list
    (None, [7, None, ["text-to-audio"]], False),
])
def test_the_tag_rule_lists_exactly_text_to_audio_and_audio_to_audio(pipeline, tags, expected):
    assert audio_facts.audio_tag(pipeline, tags) is expected


def test_the_documented_tag_set_is_exactly_the_two_in_scope():
    assert audio_facts.AUDIO_PIPELINE_TAGS == ("text-to-audio", "audio-to-audio")


def test_a_huge_tag_or_name_is_only_compared():
    huge = "x" * 5_000_000
    assert audio_facts.audio_tag(huge, [huge]) is False
    assert audio_facts.audio_layout([huge, "model_config.json", "model.safetensors"]) is True


def test_evidence_needs_a_tag_and_a_layout_and_every_source_must_agree():
    ok = ["model.safetensors", "model_config.json"]
    assert audio_facts.audio_evidence("text-to-audio", [], ok)
    assert audio_facts.audio_evidence("text-to-audio", [], ok, ok)
    assert not audio_facts.audio_evidence("text-to-audio", [], ok, ["config.json", "model.safetensors"])   # the hub and the disk disagree
    assert audio_facts.audio_evidence("text-to-audio", [], ok, [])                                        # an empty list holds no evidence
    assert not audio_facts.audio_evidence("text-generation", [], ok)
    assert not audio_facts.audio_evidence("text-to-audio", [])
    assert not audio_facts.audio_evidence("text-to-audio", [], [], [])
    assert not audio_facts.audio_evidence("text-to-audio", [], ["config.json", "model.safetensors"])


def test_the_evidence_text_names_the_tag_and_the_layout():
    text = audio_facts.audio_evidence("text-to-audio", [], ["model.safetensors", "model_config.json"])
    assert "text-to-audio" in text and "model_config.json" in text and "config.json" in text
    assert "audio-to-audio" in audio_facts.audio_evidence(None, ["audio-to-audio"], ["model.safetensors", "model_config.json"])


VARIANTS = {"audio": ["model_config.json", "model.safetensors"],
            "with-config": ["model_config.json", "config.json", "model.safetensors"],
            "no-model-config": ["config.json", "model.safetensors"], "no-weights": ["model_config.json"],
            "nested-weights": ["model_config.json", "sub/model.safetensors"],
            "model-config-only-nested": ["sub/model_config.json", "model.safetensors"]}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_layout_rule_and_stage_0s_detector_give_the_same_answer(tmp_path, variant):
    """Stage 0's template runs as a script in the run directory and cannot import orchard, so it carries its own copy of the rule. Both
    copies are pinned to the same snapshots here (the template also reads the config's content, which these fixtures make valid)."""
    d = tmp_path / variant
    d.mkdir()
    for n in VARIANTS[variant]:
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        (d / n).write_text(json.dumps({"model_type": "diffusion_cond", "model": {}}) if n.endswith(".json") else "w")
    expected = template_module().detect_audio(d)
    assert audio_facts.audio_layout([p.name for p in d.iterdir()]) is expected
    assert audio_facts.audio_layout(pf.snapshot_root_names(d)) is expected


# ---- the hub's answer --------------------------------------------------------------------------------

def test_the_hub_payload_gives_the_pipeline_tag_and_the_tags():
    h = pf.parse_hub(audio_payload())
    assert h.pipeline_tag == "text-to-audio" and "stable-audio-3" in h.tags and h.base_models == [BASE]


@pytest.mark.parametrize("bad", [["text-to-audio"], 7, {"a": 1}, None, True, ""])
def test_a_pipeline_tag_that_is_not_a_string_is_no_pipeline_tag(bad):
    assert pf.parse_hub(audio_payload(pipeline_tag=bad)).pipeline_tag is None


@pytest.mark.parametrize("bad", ["text-to-audio", 7, {"a": 1}, None, [1, None, ["x"], "ok"]])
def test_tags_that_are_not_a_list_of_strings_keep_only_the_strings(bad):
    t = pf.parse_hub(audio_payload(tags=bad)).tags
    assert t == (["ok"] if isinstance(bad, list) else [])


@pytest.mark.parametrize("sib", [None, [], "model.safetensors", [None, 3, "x", {"rfilename": 7}, {"rfilename": "a.safetensors", "size": "big"}]])
def test_a_hostile_file_list_gives_an_empty_or_filtered_list_and_no_exception(sib):
    h = pf.parse_hub(audio_payload(siblings=sib))
    assert h.files in ([], ["a.safetensors"]) and h.total_bytes >= 0


def test_a_payload_with_no_tags_or_pipeline_tag_is_no_audio_evidence():
    p = audio_payload()
    del p["tags"], p["pipeline_tag"]
    h = pf.parse_hub(p)
    assert (h.pipeline_tag, h.tags) == (None, [])
    assert not audio_facts.audio_evidence(h.pipeline_tag, h.tags, h.files)


# ---- the base row for an audio model -----------------------------------------------------------------

EVIDENCE = "pipeline tag text-to-audio; model_config.json and no config.json at the repository root"


def base_hub(**over):
    return pf.parse_hub(audio_payload(BASE, sha=BASE_REV, files=AUDIO_FILES + ["svd_bases.pt"], **over))


def audio_base(*, hub=None, base_info=None, local=None, report=None, installed=(), **kw):
    hub = hub or pf.parse_hub(audio_payload())
    info = base_info if base_info is not None else (base_hub(), None)
    return pf.check_base(model_id=MODEL, hub=hub, base_override=kw.pop("base_override", None), installed=list(installed),
                         local_snapshot=lambda m: (local or {}).get(m), search=None, report=report, audio=EVIDENCE,
                         base_hub_info=lambda m: info, **kw)


def test_an_audio_model_needs_no_bundle_and_the_row_is_a_warning_that_says_why():
    c = audio_base(base_info=(base_hub(gated=False), None))
    assert c.status == "warn" and c.reason is None
    for part in (BASE, "audio", "No installed bundle is needed", "audio-path-unbuilt", "stage 0", "text-to-audio"):
        assert part in c.detail, part
    assert c.data["base"] == BASE and c.data["needs_fetch"] is True and c.data["snapshot"] is None
    assert c.data["bundle"] is None and c.data["candidates"] == [] and c.data["audio"] is True
    assert c.data["base_gb"] == pytest.approx(6.0)     # six listed files of 1 GB


def test_the_row_points_to_the_unblock_text():
    assert "BLOCKED.md" in audio_base(base_info=(base_hub(gated=False), None)).detail


def test_a_local_base_snapshot_is_handed_on_and_nothing_is_downloaded(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV)
    c = audio_base(base_info=(base_hub(gated=False), None), local={BASE: s})
    assert c.status == "warn" and c.data["snapshot"] == s and c.data["needs_fetch"] is False and "local" in c.detail


def test_the_base_is_the_override_when_one_is_given():
    other = "stabilityai/other-base"
    c = audio_base(base_override=other, base_info=(base_hub(gated=False), None))
    assert c.data["base"] == other and other in c.detail


def test_a_gated_base_that_is_not_on_disk_blocks_and_is_never_fetched():
    c = audio_base()          # the base is gated and nothing is local
    assert (c.status, c.reason) == ("block", "credentials-needed") and BASE in c.detail and "Download it yourself" in c.detail
    assert c.data["needs_fetch"] is False and c.data["audio"] is True


def test_a_gated_base_that_is_complete_on_disk_passes(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV, names=AUDIO_FILES + ["svd_bases.pt"])
    c = audio_base(local={BASE: s})
    assert c.status == "warn" and "gated" in c.detail and BASE_REV[:12] in c.detail


def test_a_gated_base_missing_a_listed_weight_file_blocks(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV)         # no t5gemma weights
    c = audio_base(local={BASE: s})
    assert (c.status, c.reason) == ("block", "credentials-needed")


def test_a_public_base_that_is_not_on_disk_is_downloaded():
    c = audio_base(base_info=(base_hub(gated=False), None))
    assert c.status == "warn" and c.data["needs_fetch"] is True and "will be downloaded" in c.detail


def test_a_private_base_blocks_even_when_local(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV)
    c = audio_base(base_info=(base_hub(private=True, gated=False), None), local={BASE: s})
    assert (c.status, c.reason) == ("block", "credentials-needed")


def test_a_base_the_hub_cannot_answer_for_and_no_local_copy_blocks():
    c = audio_base(base_info=(None, "HTTP 404"))
    assert (c.status, c.reason) == ("block", "model-unavailable") and "HTTP 404" in c.detail


def test_a_base_the_hub_cannot_answer_for_but_present_on_disk_warns(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV)
    c = audio_base(base_info=(None, "URLError"), local={BASE: s})
    assert c.status == "warn" and c.data["snapshot"] == s and "could not be reached" in c.detail


def test_a_lookup_that_raises_is_a_clear_block_and_not_an_exception():
    def boom(m):
        raise OSError("down")
    c = pf.check_base(model_id=MODEL, hub=pf.parse_hub(audio_payload()), base_override=None, installed=[], local_snapshot=lambda m: None,
                      search=None, audio=EVIDENCE, base_hub_info=boom)
    assert c.status == "block" and c.reason == "model-unavailable"


@pytest.mark.parametrize("answer", [("not a HubInfo", None), (None, None), "junk", 7])
def test_a_lookup_that_returns_junk_is_a_clear_block_and_not_an_exception(answer):
    c = pf.check_base(model_id=MODEL, hub=pf.parse_hub(audio_payload()), base_override=None, installed=[], local_snapshot=lambda m: None,
                      search=None, audio=EVIDENCE, base_hub_info=lambda m: answer)
    assert c.status == "block" and c.reason == "model-unavailable"


def test_a_base_that_is_not_in_the_audio_layout_leaves_the_audio_path():
    text_base = pf.parse_hub(audio_payload(BASE, files=["model.safetensors", "config.json"]))
    c = audio_base(base_info=(text_base, None))
    assert c.status == "block" and c.reason == "nearest-model-missing"          # today's row


def test_a_local_base_in_the_text_layout_leaves_the_audio_path(tmp_path):
    s = snap(tmp_path / "hf", BASE, BASE_REV, names=("model.safetensors", "config.json"))
    c = audio_base(base_info=(None, "x"), local={BASE: s})
    assert c.status == "block" and c.reason == "nearest-model-missing"


def test_an_inferred_base_waits_for_the_operator():
    report = {"selected": {"repo": BASE, "confidence": "inferred"}, "candidates": []}
    hub = pf.parse_hub(audio_payload(cardData={"license": "other"}))
    c = audio_base(hub=hub, report=report, base_info=(base_hub(gated=False), None))
    assert (c.status, c.reason) == ("block", "inferred-base-unconfirmed") and "--accept-inferred-base" in c.detail
    ok = audio_base(hub=hub, report=report, accept_inferred=True, base_info=(base_hub(gated=False), None))
    assert ok.status == "warn" and ok.data["base"] == BASE


def test_a_stated_selection_needs_no_confirmation():
    report = {"selected": {"repo": BASE, "confidence": "stated"}, "candidates": []}
    c = audio_base(hub=pf.parse_hub(audio_payload(cardData={"license": "other"})), report=report, base_info=(base_hub(gated=False), None))
    assert c.status == "warn"


def test_with_no_base_at_all_the_old_warning_stays():
    c = audio_base(hub=pf.parse_hub(audio_payload(cardData={"license": "other"})))
    assert c.status == "warn" and "names no base_model" in c.detail and c.data["base"] is None


def test_a_text_model_keeps_the_old_block_when_no_bundle_serves_its_base():
    c = pf.check_base(model_id=MODEL, hub=pf.parse_hub(audio_payload()), base_override=None, installed=[], local_snapshot=lambda m: None,
                      search=None)
    assert (c.status, c.reason) == ("block", "nearest-model-missing")


def test_an_installed_bundle_that_serves_the_base_is_not_used_for_audio():
    c = audio_base(installed=[Installed("x/y", BASE, "6", 1)], base_info=(base_hub(gated=False), None))
    assert c.data["bundle"] is None and c.data["audio"] is True


# ---- the whole preflight -----------------------------------------------------------------------------

def world(*, local=None, model_gated="manual", base_gated="manual", pipeline="text-to-audio", **over):
    def hub_info(m):
        if m == MODEL:
            return pf.parse_hub(audio_payload(gated=model_gated, pipeline_tag=pipeline)), None
        if m == BASE:
            return pf.parse_hub(audio_payload(BASE, sha=BASE_REV, gated=base_gated, files=AUDIO_FILES + ["svd_bases.pt"],
                                              pipeline_tag=pipeline)), None
        return None, "HTTP 404"
    base = dict(hub_info=hub_info, local_snapshot=lambda m: (local or {}).get(m), installed_bundles=lambda: [], search_bundles=lambda q: [])
    return plain_signals(**{**base, **over})


def full_snaps(tmp_path):
    hf = tmp_path / "hf"
    return {MODEL: snap(hf, MODEL, REV, names=AUDIO_FILES), BASE: snap(hf, BASE, BASE_REV, names=AUDIO_FILES + ["svd_bases.pt"])}


def rows(tmp_path, sig):
    return {c.name: c for c in pf.run_preflight(cfg(tmp_path), MODEL, accept_credentials=True, signals=sig)}


def test_the_two_downloaded_gated_models_pass_the_whole_preflight(tmp_path):
    by = rows(tmp_path, world(local=full_snaps(tmp_path)))
    assert not [n for n, c in by.items() if c.status == "block"], {n: c.detail for n, c in by.items() if c.status == "block"}
    assert by["base"].status == "warn" and by["hub"].status == "warn"


def test_a_text_to_speech_model_is_not_audio_for_this_rule(tmp_path):
    by = rows(tmp_path, world(local=full_snaps(tmp_path), pipeline="text-to-speech"))
    assert by["base"].reason == "nearest-model-missing"


def test_a_model_whose_hub_lookup_failed_is_not_audio_even_with_a_local_audio_layout(tmp_path):
    sig = world(local=full_snaps(tmp_path), hub_info=lambda m: (None, "HTTP 500"))
    by = rows(tmp_path, sig)
    assert not (by["base"].data or {}).get("audio")


def test_a_gated_base_that_was_never_downloaded_blocks_the_whole_preflight(tmp_path):
    by = rows(tmp_path, world(local={MODEL: full_snaps(tmp_path)[MODEL]}))
    assert by["base"].reason == "credentials-needed"


def test_the_disk_row_counts_a_base_that_must_be_fetched_even_when_the_model_is_local(tmp_path):
    local = {MODEL: full_snaps(tmp_path)[MODEL]}
    by = rows(tmp_path, world(local=local, base_gated=False, free_gb=lambda p: 43.0))       # margin 40 GB + the base's 6 GB
    assert by["base"].status == "warn" and by["base"].data["needs_fetch"] is True
    assert by["disk"].status == "block" and by["disk"].reason == "disk-full" and "needs 46" in by["disk"].detail


def test_the_disk_row_sums_both_models_when_neither_is_local(tmp_path):
    by = rows(tmp_path, world(base_gated=False, model_gated=False, free_gb=lambda p: 45.0))
    assert by["disk"].status == "block" and "needs 51" in by["disk"].detail        # 40 + 5 (model) + 6 (base)


def test_a_text_model_with_a_local_snapshot_and_a_base_to_fetch_also_counts_the_base(tmp_path):
    from test_preflight import CLEF_PAYLOAD
    big = dict(CLEF_PAYLOAD, siblings=[{"rfilename": "LICENSE", "size": 50 * 10**9}], cardData={"license": "x", "base_model": "o/base"})
    s = plain_signals(hub_info=lambda m: (pf.parse_hub(big), None), free_gb=lambda p: 80.0,
                      local_snapshot=lambda m: Path("/s") if m == "Cloudflare/clef" else None,
                      installed_bundles=lambda: [Installed("b/b", "o/base", "6", 1)])
    by = {c.name: c for c in pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=True, signals=s)}
    assert by["disk"].status == "block" and "needs 90" in by["disk"].detail


# ---- the command ------------------------------------------------------------------------------------

def test_a_dry_run_for_the_audio_pair_passes_and_hands_the_base_snapshot_to_the_supervisor(conf, tmp_path):
    snaps = full_snaps(tmp_path)
    code, out, rec = run(conf, ["bringup", MODEL, "--dry-run"], sig=world(local=snaps))
    flat = " ".join(out.replace("║", " ").split())
    assert code == 0 and rec.argv is None and rec.fetched == []
    assert f"base={snaps[BASE]}" in flat.replace("-- input", "--input") or str(snaps[BASE]) in flat
    assert "audio-path-unbuilt" in flat


def test_the_real_run_passes_the_base_snapshot_to_the_supervisor(conf, tmp_path):
    snaps = full_snaps(tmp_path)
    code, out, rec = run(conf, ["bringup", MODEL], sig=world(local=snaps))
    assert code == 0 and f"base={snaps[BASE]}" in rec.argv and rec.fetched == []


def test_a_public_base_that_is_missing_is_downloaded_at_the_hub_revision(conf, tmp_path):
    local = {MODEL: full_snaps(tmp_path)[MODEL]}
    code, out, rec = run(conf, ["bringup", MODEL], sig=world(local=local, base_gated=False))
    assert code == 0 and [f[0] for f in rec.fetched] == [BASE] and rec.fetched[0][1] == BASE_REV
    assert f"base={fetch.snapshot_dir(tmp_path / 'hf', BASE, BASE_REV)}" in rec.argv


def test_a_gated_base_that_is_missing_stops_the_command_and_downloads_nothing(conf, tmp_path):
    local = {MODEL: full_snaps(tmp_path)[MODEL]}
    code, out, rec = run(conf, ["bringup", MODEL], sig=world(local=local))
    assert code == 2 and rec.argv is None and rec.fetched == [] and "credentials-needed" in out
    assert "install a bundle" not in out         # the bundle help does not apply to an audio model


def test_base_naming_an_installed_bundle_is_refused_for_audio(conf, tmp_path, capsys):
    s = world(local=full_snaps(tmp_path), installed_bundles=lambda: [Installed("eps/whisper-p150", "openai/whisper", "6", 1)])
    code, out, rec = run(conf, ["bringup", MODEL, "--base", "eps/whisper-p150"], sig=s)
    err = capsys.readouterr().err
    assert code == 2 and rec.argv is None and "audio" in err and "bundle" in err and "model id" in err


def test_base_naming_a_published_bundle_is_refused_for_audio_and_never_pulled(conf, tmp_path, capsys):
    s = world(local=full_snaps(tmp_path), search_bundles=lambda q: [{"name": "eps/audio8-p150", "installed": False}])
    pulled = []
    code = cli.main(["--config", str(conf), "bringup", MODEL, "--base", "eps/audio8-p150"], env={}, stdout=Pipe(), signals=s,
                    supervisor_main=Recorder().supervisor, fetcher=Recorder().fetcher, puller=lambda b: pulled.append(b) or (True, ""))
    assert code == 2 and pulled == [] and "bundle" in capsys.readouterr().err


def test_base_naming_a_model_id_is_accepted_for_audio(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", MODEL, "--base", BASE, "--dry-run"], sig=world(local=full_snaps(tmp_path)))
    assert code == 0 and BASE in out


def test_base_naming_a_bundle_is_still_accepted_for_a_text_model(conf):
    s = plain_signals(installed_bundles=lambda: [Installed("x/clef-p150", "Cloudflare/clef", "6", 1)], local_snapshot=lambda m: Path("/s"))
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--base", "x/clef-p150", "--dry-run"], sig=s)
    assert code == 0


# ---- the web page's preflight panel --------------------------------------------------------------------

def _panel(conf, sig, monkeypatch, base=None):
    monkeypatch.setattr(pf, "default_signals", lambda cfg_, research_options=None: sig)
    return cli._ui_preflight(bringup_config.load(conf))(MODEL, base)


def test_the_panel_and_the_command_agree_on_every_row(conf, tmp_path, monkeypatch):
    sig = world(local=full_snaps(tmp_path))
    panel = _panel(conf, sig, monkeypatch)
    code, out, rec = run(conf, ["bringup", MODEL, "--dry-run"], sig=sig)
    flat = " ".join(out.replace("║", " ").split())
    assert not pf.blocked(panel) and code == 0
    for row in panel:
        assert row.name in out
    base = next(x for x in panel if x.name == "base")
    assert base.status == "warn" and base.data["audio"] is True
    assert " ".join(base.detail.split())[:80] in flat


def test_the_panel_and_the_command_agree_when_the_base_is_gated_and_missing(conf, tmp_path, monkeypatch):
    sig = world(local={MODEL: full_snaps(tmp_path)[MODEL]})
    panel = _panel(conf, sig, monkeypatch)
    code, out, rec = run(conf, ["bringup", MODEL, "--dry-run"], sig=sig)
    assert [c.reason for c in pf.blocked(panel)] == ["credentials-needed"] and code == 2


def test_the_panel_refuses_a_bundle_for_base_on_an_audio_model(conf, tmp_path, monkeypatch):
    sig = world(local=full_snaps(tmp_path), installed_bundles=lambda: [Installed("eps/whisper-p150", "openai/whisper", "6", 1)])
    base = next(x for x in _panel(conf, sig, monkeypatch, base="eps/whisper-p150") if x.name == "base")
    assert (base.status, base.reason) == ("block", "nearest-model-missing") and "model id" in base.detail


# ---- reading the real sources safely -------------------------------------------------------------------

def test_a_broken_link_at_the_top_of_a_snapshot_is_not_a_name(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "model_config.json").write_text("{}")
    (d / "config.json").symlink_to(tmp_path / "nowhere")          # a link whose blob is gone does not count as a config.json
    (d / "model.safetensors").write_text("w")
    assert sorted(pf.snapshot_root_names(d)) == ["model.safetensors", "model_config.json"]


def test_an_unreadable_snapshot_directory_gives_no_names(tmp_path):
    assert pf.snapshot_root_names(tmp_path / "missing") == []


class _Resp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return self.body


@pytest.mark.parametrize("body", [b"[]", b'"x"', b"7", b"null", b"{not json"])
def test_the_hub_lookup_turns_an_answer_that_is_not_an_object_into_an_error_text(tmp_path, monkeypatch, body):
    monkeypatch.setattr(pf.urllib.request, "urlopen", lambda req, timeout=None: _Resp(body))
    hub, err = pf.default_signals(cfg(tmp_path)).hub_info(MODEL)
    assert hub is None and isinstance(err, str) and err


def test_the_hub_lookup_sends_no_credentials(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(pf.urllib.request, "urlopen", lambda req, timeout=None: seen.append(dict(req.header_items())) or _Resp(json.dumps(audio_payload()).encode()))
    hub, err = pf.default_signals(cfg(tmp_path)).hub_info(MODEL)
    assert err is None and hub.pipeline_tag == "text-to-audio"
    assert not any("authorization" in k.lower() for k in seen[0])
