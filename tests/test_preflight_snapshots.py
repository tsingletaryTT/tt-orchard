# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 141: how the preflight judges a local snapshot and picks one.

Three gaps from the review of PR 120: (1) the Hub's weight index files (`model.safetensors.index.json`, `pytorch_model.bin.index.json`) are
required files, so a snapshot that has every shard and no index is not complete and the hub row for a gated model blocks; (2) a shard map
that is empty, has a null or empty-string name, or names something that is not a file does not pass the completeness check; (3) with two
cached revisions the complete one is chosen even when a partial one sorts later."""
from __future__ import annotations

import json

import pytest

from orchard import preflight as pf
from test_preflight import FakeResponse, cfg, payload, real


def _snap(tmp_path, rev="aaaa1111", repo="org--model", files=("config.json", "model-00001-of-00002.safetensors",
                                                                "model-00002-of-00002.safetensors")):
    d = tmp_path / "hf" / "hub" / f"models--{repo}" / "snapshots" / rev
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text("{}" if f.endswith(".json") else "w")
    return d


def _index(snap, weight_map, name="model.safetensors.index.json"):
    (snap / name).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


# ---- (1) weight indexes are required files ----------------------------------------------------------------------------

def _hub(files, **over):
    sib = [{"rfilename": f, "size": 1} for f in files] + [{"rfilename": "LICENSE.md", "size": 1}]
    return pf.parse_hub(payload(**{"siblings": sib, **over}))


@pytest.mark.parametrize("index", ["model.safetensors.index.json", "pytorch_model.bin.index.json", "sub/model.safetensors.index.json"])
def test_a_hub_listed_weight_index_is_a_required_file(tmp_path, index):
    snap = _snap(tmp_path)
    hub = _hub(["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", index])
    assert pf._missing_weights(hub, snap) == [index]
    assert pf._missing_listed(hub, snap) == [index]


def test_a_gated_model_with_every_shard_and_no_index_blocks(tmp_path):
    snap = _snap(tmp_path)
    hub = _hub(["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"],
               gated="manual")
    assert pf.check_hub(hub, None, local=True, local_path=snap).reason == "credentials-needed"
    _index(snap, {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"})
    assert pf.check_hub(hub, None, local=True, local_path=snap).status == "warn"


def test_a_file_that_only_looks_like_an_index_is_not_one(tmp_path):
    snap = _snap(tmp_path)
    hub = _hub(["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "notes.index.json",
                "model.safetensors.index.json.bak"])
    assert pf._missing_weights(hub, snap) == []


# ---- (2) a shard map has to be a non-empty map of names of files --------------------------------------------------------

def _complete(tmp_path):
    snap = _snap(tmp_path)
    _index(snap, {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"})
    return snap


def test_a_good_index_passes(tmp_path):
    assert pf.snapshot_complete(_complete(tmp_path)) is True


@pytest.mark.parametrize("weight_map", [
    {}, None, [], "model.safetensors", 3,
    {"a": ""}, {"a": None}, {"a": 7}, {"a": ["model-00001-of-00002.safetensors"]},
    {"a": "model-00001-of-00002.safetensors", "b": ""},
    {"a": "model-00001-of-00002.safetensors", "b": None},
    {"a": "   "},
    {"a": "../outside.safetensors"}, {"a": "/etc/passwd"}, {"a": "sub/../model-00001-of-00002.safetensors"},
])
def test_a_malformed_shard_map_is_not_complete(tmp_path, weight_map):
    snap = _snap(tmp_path)
    _index(snap, weight_map)
    assert pf.snapshot_complete(snap) is False


def test_a_shard_that_is_a_directory_is_not_a_file(tmp_path):
    snap = _snap(tmp_path)
    (snap / "shard-dir.safetensors").mkdir()
    _index(snap, {"a": "shard-dir.safetensors", "b": "model-00001-of-00002.safetensors"})
    assert pf.snapshot_complete(snap) is False


def test_a_shard_that_is_a_link_to_nowhere_is_not_a_file(tmp_path):
    snap = _snap(tmp_path)
    (snap / "gone.safetensors").symlink_to(tmp_path / "nowhere")
    _index(snap, {"a": "gone.safetensors"})
    assert pf.snapshot_complete(snap) is False


@pytest.mark.parametrize("body", ['{"weight_map": {"a": "model-00001-of-00002.safetensors"}, "weight_map": {}}', '{"weight_map": NaN}',
                                  '{"weight_map": 1e999}', "[]", "3", ""])
def test_an_index_that_is_not_strict_json_with_a_map_is_not_complete(tmp_path, body):
    snap = _snap(tmp_path)
    (snap / "model.safetensors.index.json").write_text(body)
    assert pf.snapshot_complete(snap) is False


def test_an_index_that_is_too_large_is_not_read_and_is_not_complete(tmp_path):
    snap = _snap(tmp_path)
    (snap / "model.safetensors.index.json").write_text('{"weight_map": {"a": "' + "x" * (pf.INDEX_MAX_BYTES + 10) + '"}}')
    assert pf.snapshot_complete(snap) is False


def test_an_index_that_is_a_link_into_a_blob_is_read(tmp_path):
    """A Hugging Face snapshot holds each file as a link into the cache's blobs directory."""
    snap = _snap(tmp_path)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "idx").write_text(json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}}))
    (snap / "model.safetensors.index.json").symlink_to(blobs / "idx")
    assert pf.snapshot_complete(snap) is True


def test_a_pytorch_bin_index_is_checked_like_the_safetensors_one(tmp_path):
    snap = _snap(tmp_path, files=("config.json", "pytorch_model-00001-of-00002.bin"))
    _index(snap, {}, name="pytorch_model.bin.index.json")
    assert pf.snapshot_complete(snap) is False
    _index(snap, {"a": "pytorch_model-00001-of-00002.bin"}, name="pytorch_model.bin.index.json")
    assert pf.snapshot_complete(snap) is True


# ---- (3) the complete revision is chosen over a partial one that sorts later ----------------------------------------------

FILES = ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"]


def _two_revisions(tmp_path, partial_name="zzzz9999", complete_name="aaaa1111"):
    complete = _snap(tmp_path, rev=complete_name)
    _index(complete, {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"})
    partial = _snap(tmp_path, rev=partial_name, files=("config.json", "model-00001-of-00002.safetensors"))   # config and a root weight only
    return complete, partial


def _signals_with_hub(tmp_path, monkeypatch, sha, files=FILES):
    body = json.dumps(payload(id="org/model", sha=sha, siblings=[{"rfilename": f, "size": 1} for f in files] + [{"rfilename": "LICENSE", "size": 1}]))
    monkeypatch.setattr(pf.urllib.request, "urlopen", lambda req, timeout: FakeResponse(body.encode()))
    s = real(tmp_path)
    hub, err = s.hub_info("org/model")
    assert err is None
    return s, hub


def test_the_complete_revision_is_chosen_over_a_partial_one_that_sorts_later(tmp_path, monkeypatch):
    complete, partial = _two_revisions(tmp_path)
    assert partial.name > complete.name
    s, _ = _signals_with_hub(tmp_path, monkeypatch, sha="ffff0000")
    assert s.local_snapshot("org/model") == complete


def test_the_complete_revision_is_chosen_when_the_partial_one_is_the_hubs_current_one(tmp_path, monkeypatch):
    complete, partial = _two_revisions(tmp_path)
    s, _ = _signals_with_hub(tmp_path, monkeypatch, sha=partial.name)
    assert s.local_snapshot("org/model") == complete


def test_among_complete_revisions_the_hubs_current_one_is_chosen(tmp_path, monkeypatch):
    a, b = _snap(tmp_path, rev="aaaa1111"), _snap(tmp_path, rev="zzzz9999")
    for d in (a, b):
        _index(d, {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"})
    s, _ = _signals_with_hub(tmp_path, monkeypatch, sha="aaaa1111")
    assert s.local_snapshot("org/model") == a


def test_with_no_hub_answer_the_last_complete_revision_by_name_is_used_as_before(tmp_path):
    for rev in ("aaaa1111", "mmmm5555"):
        _index(_snap(tmp_path, rev=rev), {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"})
    assert real(tmp_path).local_snapshot("org/model").name == "mmmm5555"


def test_choose_snapshot_returns_none_when_no_revision_has_every_listed_weight_and_index(tmp_path):
    """Review of PR 191: the least incomplete revision used to be returned, and the missing index was then never fetched for a public model."""
    one, two = _snap(tmp_path, rev="aaaa1111", files=("config.json", "model-00001-of-00002.safetensors")), \
        _snap(tmp_path, rev="bbbb2222", files=("config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"))
    hub = _hub(FILES)                                           # lists the index too, which neither revision has
    assert pf.choose_snapshot(hub, [one, two]) is None
    assert pf.choose_snapshot(hub, []) is None
    assert pf.choose_snapshot(None, [one, two]) == two          # no Hub answer: the last by name


def test_a_revision_missing_only_a_component_config_wins_over_one_missing_a_weight(tmp_path):
    """Review of PR 191: both lack one listed file, so the ranking alone could not tell them apart and the weight-complete copy could lose to a tie."""
    files = ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "sub/config.json"]
    hub = _hub(files, sha="bbbb2222")                           # the Hub's own revision is the one that lacks a weight
    weights_ok = _snap(tmp_path, rev="aaaa1111")                # both shards, no sub/config.json
    lacks_weight = _snap(tmp_path, rev="bbbb2222", files=("config.json", "model-00001-of-00002.safetensors", "sub/config.json"))
    assert pf._missing_listed(hub, weights_ok) == ["sub/config.json"] and pf._missing_listed(hub, lacks_weight) == ["model-00002-of-00002.safetensors"]
    assert pf.choose_snapshot(hub, [weights_ok, lacks_weight]) == weights_ok
    assert pf.choose_snapshot(hub, [lacks_weight]) is None


def test_with_the_index_present_the_fuller_revision_is_chosen_among_incomplete_ones(tmp_path):
    one = _snap(tmp_path, rev="aaaa1111", files=("config.json", "model-00001-of-00002.safetensors"))
    two = _snap(tmp_path, rev="bbbb2222")
    _index(two, {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"})
    assert pf.choose_snapshot(_hub(FILES), [one, two]) == two


def test_a_public_model_whose_cached_revisions_all_lack_the_listed_index_is_not_local_so_it_is_fetched(tmp_path, monkeypatch):
    _snap(tmp_path, rev="aaaa1111")                              # every shard, no index
    s, hub = _signals_with_hub(tmp_path, monkeypatch, sha="ffff0000")
    assert s.local_snapshot("org/model") is None


def test_the_run_uses_the_chosen_revision_for_the_hub_row(tmp_path, monkeypatch):
    complete, partial = _two_revisions(tmp_path)
    s, hub = _signals_with_hub(tmp_path, monkeypatch, sha="ffff0000")
    hub.gated = True
    # the gated hub row names the revision it found on disk: the complete one, and the partial one that sorts later is not named
    c = pf.check_hub(hub, None, local=True, local_path=s.local_snapshot("org/model"))
    assert c.status == "warn" and complete.name in c.detail and partial.name not in c.detail


def test_the_audio_base_row_asks_the_hub_before_it_looks_at_the_cache():
    """The real local_snapshot uses the Hub's answer it already has, so for the base model the hub lookup must come first."""
    from test_audio_preflight import BASE, EVIDENCE, MODEL, audio_payload, base_hub
    calls = []

    def local(m):
        calls.append(("cache", m))
        return None

    def hub(m):
        calls.append(("hub", m))
        return base_hub(gated=False), None
    pf.check_base(model_id=MODEL, hub=pf.parse_hub(audio_payload()), base_override=None, installed=[], search=None, audio=EVIDENCE,
                  local_snapshot=local, base_hub_info=hub)
    assert calls.index(("hub", BASE)) < calls.index(("cache", BASE))


def test_the_complete_base_revision_is_chosen_over_a_partial_one_for_the_audio_base(tmp_path, monkeypatch):
    """A public audio base with a complete older revision and a partial newer one is not downloaded again."""
    from test_audio_preflight import AUDIO_FILES, BASE, BASE_REV, EVIDENCE, MODEL, audio_payload, base_hub, snap
    hf = tmp_path / "hf"
    complete = snap(hf, BASE, "aaaa" + "0" * 36, names=AUDIO_FILES + ["svd_bases.pt"])
    # partial: passes the completeness check (a config, a root weight, the text encoder's config) but lacks a weight the Hub lists; sorts later
    snap(hf, BASE, "zzzz" + "0" * 36, names=["model.safetensors", "model_config.json", "t5gemma-b-b-ul2/config.json"])
    body = json.dumps(audio_payload(BASE, sha=BASE_REV, gated=False, files=AUDIO_FILES + ["svd_bases.pt"]))
    monkeypatch.setattr(pf.urllib.request, "urlopen", lambda req, timeout: FakeResponse(body.encode()))
    s = real(tmp_path)
    c = pf.check_base(model_id=MODEL, hub=pf.parse_hub(audio_payload()), base_override=None, installed=[], search=None, audio=EVIDENCE,
                      local_snapshot=s.local_snapshot, base_hub_info=s.hub_info)
    assert c.data["snapshot"] == complete and c.data["needs_fetch"] is False
