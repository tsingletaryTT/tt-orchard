# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Finding the model to base a bring-up on (orchard/nearest.py and the `base` preflight check).

A new model is compared with a model that already runs on Tenstorrent hardware. The humanizer run died
because its base, google/gemma-4-12B, was not on disk and no installed bundle served it, and nothing said so
until three coder boots and 105 agent turns later. These tests pin the early answer."""
import json
import subprocess
from pathlib import Path

import pytest

from orchard import nearest
from orchard import preflight as pf
from orchard.nearest import Installed

GEMMA = "google/gemma-4-12B"


def hub(**over):
    base = {"id": "jialinyyzz/humanizer", "private": False, "gated": False, "sha": "c07c06c8",
            "cardData": {"license": "apache-2.0", "base_model": GEMMA},
            "config": {"architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified"},
            "siblings": [{"rfilename": "LICENSE", "size": 1000}, {"rfilename": "config.json", "size": 1000},
                         {"rfilename": "model.safetensors", "size": 24_000_000_000}]}
    base.update(over)
    return pf.parse_hub(base)


def installed(*pairs):
    return [Installed(name=n, weights_repo=w, schema="6", chips=2) for n, w in pairs]


QWEN_BUNDLE = installed(("episod/qwen3.8-27b-dflash2-p300", "Qwen/Qwen3.8-27B"))


# ---- reading what the hub says about a model's base ------------------------------------------

def test_the_card_base_model_and_architecture_are_read():
    h = hub()
    assert h.base_models == [GEMMA] and h.architectures == ["Gemma4UnifiedForConditionalGeneration"]


def test_a_base_model_given_as_a_string_or_a_list_or_missing():
    assert hub(cardData={"license": "mit", "base_model": GEMMA}).base_models == [GEMMA]
    assert hub(cardData={"license": "mit", "base_model": ["a/b", "c/d"]}).base_models == ["a/b", "c/d"]
    assert hub(cardData={"license": "mit"}).base_models == []
    assert hub(cardData={"license": "mit", "base_model": 7}).base_models == []


def test_a_base_model_that_is_the_model_itself_is_ignored():
    assert hub(cardData={"license": "mit", "base_model": "jialinyyzz/humanizer"}).base_models == []


# ---- what to search for -----------------------------------------------------------------------

@pytest.mark.parametrize("model, query", [("google/gemma-4-12B", "gemma"), ("Qwen/Qwen3.8-27B", "qwen"),
                                          ("meta-llama/Llama-3.2-1B", "llama"), ("org/7b-chat", "7b-chat")])
def test_the_search_query_is_the_model_family(model, query):
    assert nearest.family_query(model) == query


# ---- the installed bundles --------------------------------------------------------------------

def make_bundle(root: Path, org, name, repo, schema="6", chips=2):
    d = root / org / name
    d.mkdir(parents=True)
    (d / "tt_kernel_manifest.json").write_text(json.dumps(
        {"schema_version": schema, "name": name, "weights": {"repo_id": repo, "revision": "r"},
         "device_count": chips}))


def test_installed_bundles_are_read_from_their_manifests(tmp_path):
    make_bundle(tmp_path, "episod", "qwen3.8-27b-dflash2-p300", "Qwen/Qwen3.8-27B")
    make_bundle(tmp_path, "stisiTT", "gemma-4-12b-it-p150", "google/gemma-4-12B-it", schema="5.1", chips=1)
    (tmp_path / "episod" / "broken").mkdir()
    (tmp_path / "episod" / "broken" / "tt_kernel_manifest.json").write_text("not json")
    got = {i.name: i for i in nearest.read_installed(tmp_path)}
    assert set(got) == {"episod/qwen3.8-27b-dflash2-p300", "stisiTT/gemma-4-12b-it-p150"}
    assert got["stisiTT/gemma-4-12b-it-p150"] == Installed("stisiTT/gemma-4-12b-it-p150", "google/gemma-4-12B-it", "5.1", 1)


def test_a_missing_models_root_is_an_empty_list(tmp_path):
    assert nearest.read_installed(tmp_path / "nope") == []


# ---- the search -------------------------------------------------------------------------------

TT_JSON = json.dumps({"query": "gemma", "bundles": [
    {"name": "stisiTT/gemma-4-12b-it-p150", "installed": False},
    {"name": "tt-hous/gemma-4-26B-A4B-it_p150", "installed": True}]})
TT_MODEL_JSON = json.dumps([{"id": "stisiTT/gemma-4-12b-it-p150", "private": False}])


def test_both_search_outputs_are_understood():
    assert nearest.parse_search(TT_JSON) == [{"name": "stisiTT/gemma-4-12b-it-p150", "installed": False},
                                              {"name": "tt-hous/gemma-4-26B-A4B-it_p150", "installed": True}]
    assert nearest.parse_search(TT_MODEL_JSON) == [{"name": "stisiTT/gemma-4-12b-it-p150", "installed": None}]


@pytest.mark.parametrize("text", ["", "not json", "{}", "[1, 2]", '{"bundles": "x"}'])
def test_an_unreadable_search_is_none_not_a_crash(text):
    assert nearest.parse_search(text) is None or nearest.parse_search(text) == []


class Run:
    def __init__(self, outputs):
        self.outputs, self.argvs = outputs, []

    def __call__(self, argv, **kw):
        self.argvs.append(argv)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return subprocess.CompletedProcess(argv, out[0], out[1], out[2])


def which_all(name):
    return f"/bin/{name}"


def test_the_official_cli_is_asked_first():
    run = Run([(0, TT_JSON, "")])
    got = nearest.search("gemma", run=run, which=which_all)
    assert run.argvs[0][:4] == ["/bin/tt", "model", "search", "gemma"] and "--json" in run.argvs[0]
    assert [b["name"] for b in got][0] == "stisiTT/gemma-4-12b-it-p150"


def test_tt_model_is_the_fallback_when_tt_is_missing_or_fails():
    run = Run([(1, "", "boom"), (0, TT_MODEL_JSON, "")])
    got = nearest.search("gemma", run=run, which=which_all)
    assert run.argvs[1][:3] == ["/bin/tt-model", "search", "gemma"] and got[0]["name"] == "stisiTT/gemma-4-12b-it-p150"
    assert nearest.search("gemma", run=Run([]), which=lambda n: None) is None


def test_a_search_that_times_out_is_none():
    run = Run([subprocess.TimeoutExpired("tt", 1), subprocess.TimeoutExpired("tt-model", 1)])
    assert nearest.search("gemma", run=run, which=which_all) is None


def test_pull_runs_tt_model_pull_and_reports_failure_text():
    run = Run([(0, "installed", "")])
    assert nearest.pull("stisiTT/gemma-4-12b-it-p150", run=run, which=which_all) == (True, "")
    assert run.argvs[0] == ["/bin/tt-model", "pull", "stisiTT/gemma-4-12b-it-p150"]
    ok, why = nearest.pull("x/y", run=Run([(1, "", "no such bundle\n")]), which=which_all)
    assert not ok and "no such bundle" in why
    assert nearest.pull("x/y", run=Run([]), which=lambda n: None)[0] is False


# ---- the base check ---------------------------------------------------------------------------

def check(h=None, *, override=None, local=None, inst=(), search=None, model="jialinyyzz/humanizer"):
    return pf.check_base(model_id=model, hub=h if h is not None else hub(), base_override=override,
                         local_snapshot=lambda m: local, installed=list(inst), search=search)


def test_a_model_whose_base_has_a_bundle_and_a_snapshot_passes():
    c = check(hub(cardData={"license": "mit", "base_model": "Qwen/Qwen3.8-27B"}), local=Path("/s"), inst=QWEN_BUNDLE)
    assert c.status == "ok" and "episod/qwen3.8-27b-dflash2-p300" in c.detail
    assert c.data["base"] == "Qwen/Qwen3.8-27B" and c.data["snapshot"] == Path("/s") and not c.data["needs_fetch"]


def test_a_missing_base_snapshot_with_a_bundle_is_fetched_not_blocked():
    c = check(hub(cardData={"license": "mit", "base_model": "Qwen/Qwen3.8-27B"}), local=None, inst=QWEN_BUNDLE)
    assert c.status == "ok" and c.data["needs_fetch"] and "will be downloaded" in c.detail


def test_no_bundle_for_the_base_blocks_and_names_the_fix():
    c = check(search=lambda q: [{"name": "stisiTT/gemma-4-12b-it-p150", "installed": False}], inst=QWEN_BUNDLE)
    assert c.status == "block" and c.reason == "nearest-model-missing"
    assert GEMMA in c.detail and "stisiTT/gemma-4-12b-it-p150" in c.detail and "--base" in c.detail
    assert c.data["candidates"] == [{"name": "stisiTT/gemma-4-12b-it-p150", "installed": False}]


def test_the_search_is_for_the_family_of_the_base():
    asked = []
    check(search=lambda q: asked.append(q) or [], inst=[])
    assert asked == ["gemma"]


def test_installed_bundles_of_the_family_are_listed_first_and_others_left_out():
    inst = installed(("tt-hous/gemma-4-26B-it", "google/gemma-4-26B-it")) + QWEN_BUNDLE
    c = check(search=lambda q: [{"name": "stisiTT/gemma-4-12b-it-p150", "installed": False}], inst=inst)
    assert [x["name"] for x in c.data["candidates"]] == ["tt-hous/gemma-4-26B-it", "stisiTT/gemma-4-12b-it-p150"]
    assert c.data["candidates"][0]["installed"] is True


def test_the_model_itself_is_never_offered_as_its_own_base():
    c = check(search=lambda q: [{"name": "jialinyyzz/humanizer", "installed": False}], inst=[])
    assert c.data["candidates"] == []


def test_a_search_that_failed_still_blocks_and_says_so():
    c = check(search=lambda q: None, inst=[])
    assert c.status == "block" and "search" in c.detail.lower() and c.data["candidates"] == []


def test_without_a_search_signal_it_still_blocks():
    c = check(search=None, inst=[])
    assert c.status == "block" and c.reason == "nearest-model-missing"


def test_a_card_with_no_base_model_is_a_warning_with_a_way_forward():
    c = check(hub(cardData={"license": "mit"}))
    assert c.status == "warn" and "--base" in c.detail and c.data["base"] is None


def test_an_override_replaces_the_cards_base():
    c = check(override="Qwen/Qwen3.8-27B", local=Path("/s"), inst=QWEN_BUNDLE)
    assert c.status == "ok" and c.data["base"] == "Qwen/Qwen3.8-27B"


def test_a_v5_1_container_bundle_is_a_warning_because_the_swap_template_wants_a_thin_bundle():
    inst = [Installed("stisiTT/gemma-4-12b-it-p150", GEMMA, "5.1", 1)]
    c = check(local=Path("/s"), inst=inst)
    assert c.status == "warn" and "5.1" in c.detail and "thin" in c.detail


def test_one_thin_bundle_among_containers_is_enough():
    inst = [Installed("a/b-c", GEMMA, "5.1", 1), Installed("a/b-t", GEMMA, "6", 2)]
    assert check(local=Path("/s"), inst=inst).status == "ok"


# ---- inside run_preflight ---------------------------------------------------------------------

def test_the_base_check_is_part_of_the_preflight_and_adds_the_download_to_the_disk_need(tmp_path):
    from test_preflight import cfg, signals
    seen = {}

    def hub_info(m):
        seen.setdefault("asked", []).append(m)
        if m == "Qwen/Qwen3.8-27B":
            return pf.parse_hub({"id": m, "sha": "b" * 40, "cardData": {"license": "apache-2.0"},
                                 "siblings": [{"rfilename": "config.json", "size": 1},
                                              {"rfilename": "w.safetensors", "size": 50_000_000_000}]}), None
        return pf.parse_hub({"id": m, "sha": "a" * 40, "cardData": {"license": "mit", "base_model": "Qwen/Qwen3.8-27B"},
                             "siblings": [{"rfilename": "config.json", "size": 1}, {"rfilename": "LICENSE", "size": 1},
                                          {"rfilename": "w.safetensors", "size": 1_000_000_000}]}), None
    s = signals(hub_info=hub_info, installed_bundles=lambda: QWEN_BUNDLE, search_bundles=lambda q: [],
                free_gb=lambda p: 60.0)
    out = pf.run_preflight(cfg(tmp_path), "org/new", accept_credentials=False, signals=s)
    names = [c.name for c in out]
    assert names[0] == "hub" and "base" in names
    base = next(c for c in out if c.name == "base")
    assert base.status == "ok" and base.data["needs_fetch"]
    disk = next(c for c in out if c.name == "disk")
    assert disk.status == "block"          # 51 GB of downloads plus the margin do not fit in 60 GB


def test_a_failing_base_signal_is_a_block_not_a_crash(tmp_path):
    from test_preflight import cfg, signals

    def boom():
        raise RuntimeError("manifest unreadable")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(installed_bundles=boom))
    base = next(c for c in out if c.name == "base")
    assert base.status == "block" and "manifest unreadable" in base.detail
