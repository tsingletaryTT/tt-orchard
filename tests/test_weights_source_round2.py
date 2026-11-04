# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 442, Copilot round 2 (issue 73): five findings about how the weights-source check is wired.

1. The dry run of the swap's run.sh edit must not redirect the process-wide stderr (the web UI runs preflights on request threads).
2. A recorded hard-coded container that is not installed must not be offered as a search result, a ranked choice or a closest bundle.
3. The inferred-base confirmation must not offer a refused package that the research report lists.
4. Stage 2's and stage 4's selectors must apply the same eligibility as the preflight, so a refused bundle is never chosen over a usable
   container.
5. The recorded verdict is about container packages. A v6 bundle of the same name is judged by its own run.sh.
Nothing here needs a chip, a container or the network."""
import json
import sys
import threading
from pathlib import Path

import pytest

from orchard import preflight as pf
from orchard import swap_draft as sd
from orchard import weights_source as ws
from orchard.nearest import Installed

BASE = "Qwen/Base-7B"
HARD = "org/hard-coded-pkg"
OPEN = "org/probed-open-pkg"
RECORDS = {
    HARD: ws.Record(HARD, ws.HARD_CODED, "the server log shows 'Fetching 28 files' while 32 of 32 tokens matched", 73),
    OPEN: ws.Record(OPEN, ws.HONOURS, "an isolated, empty /hf stayed empty over a full test", 73),
}

GOOD = f'''#!/usr/bin/env bash
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PYBIN="$HERE/venv/bin/python"
export HF_MODEL="${{HF_MODEL:-{BASE}}}"
"$PYBIN" "$HERE/prepare_model_dir.py"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "$HERE/model-dir" --max_num_seqs 4 "$@")
exec "${{CMD[@]}}"
'''
BAD = GOOD.replace(' --model "$HERE/model-dir"', "")          # no --model argument: the swap cannot rewrite it


def container(name, repo=BASE, chips=2):
    return Installed(name, repo, "5.1", chips)


def v6(tmp_path, name, text, repo=BASE, chips=2):
    d = tmp_path / name.replace("/", "__")
    d.mkdir(exist_ok=True)
    if text is not None:
        (d / "run.sh").write_text(text)
    return Installed(name, repo, "6", chips, str(d))


def hub(base_model=BASE):
    card = {"license": "apache-2.0"}
    if base_model:
        card["base_model"] = base_model
    return pf.parse_hub({"id": "someone/fine-tune", "private": False, "gated": False, "sha": "c07c06c8", "cardData": card,
                         "config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
                         "siblings": [{"rfilename": "config.json", "size": 1000},
                                      {"rfilename": "model.safetensors", "size": 14_000_000_000}]})


def check(inst, *, search=None, report=None, base_model=BASE):
    return pf.check_base(model_id="someone/fine-tune", hub=hub(base_model), base_override=None, local_snapshot=lambda m: Path("/s"),
                         installed=list(inst), search=search, report=report, weights_records=RECORDS)


# ---- 1. the dry run leaves the process's stderr alone ----------------------------------------------------------------------------

def test_the_dry_run_does_not_redirect_the_process_stderr_while_it_edits(monkeypatch):
    real = ws._template()
    marker = open("/dev/null", "w")
    monkeypatch.setattr(sys, "stderr", marker)
    seen = []

    class Proxy:
        def __getattr__(self, name):
            return getattr(real, name)

        def edit_run_script(self, *a, **k):
            seen.append(sys.stderr is marker)
            return real.edit_run_script(*a, **k)

    monkeypatch.setattr(ws, "_template", lambda: Proxy())
    assert ws.run_sh_problem(BAD, BASE) and ws.run_sh_problem(GOOD, BASE) is None
    assert seen == [True, True] and sys.stderr is marker


def test_overlapping_dry_runs_keep_each_callers_message_and_leave_stderr_in_place(monkeypatch):
    """The order that broke a redirect of the process-wide stderr: A starts its edit, B starts its edit, A ends first, B ends second.
    The order is forced with events, so the test does not depend on thread timing."""
    real = ws._template()
    marker = open("/dev/null", "w")
    monkeypatch.setattr(sys, "stderr", marker)
    a_inside, b_inside, a_done = threading.Event(), threading.Event(), threading.Event()
    got = {}

    class Proxy:
        def __getattr__(self, name):
            return getattr(real, name)

        def edit_run_script(self, *a, **k):
            if threading.current_thread().name == "A":
                a_inside.set()
                assert b_inside.wait(timeout=30)
            else:
                b_inside.set()
                assert a_done.wait(timeout=30)
            return real.edit_run_script(*a, **k)

    monkeypatch.setattr(ws, "_template", lambda: Proxy())

    def run_a():
        got["A"] = ws.run_sh_problem(BAD, BASE)
        a_done.set()

    def run_b():
        assert a_inside.wait(timeout=30)
        got["B"] = ws.run_sh_problem(BAD, BASE)

    threads = [threading.Thread(target=run_a, name="A"), threading.Thread(target=run_b, name="B")]
    [t.start() for t in threads]
    [t.join(timeout=60) for t in threads]
    assert not any(t.is_alive() for t in threads)
    assert all(got.get(k) and "--model" in got[k] for k in "AB"), got
    assert sys.stderr is marker


def test_the_swap_script_still_says_a_failure_on_stderr_when_it_runs_on_its_own(capsys):
    with pytest.raises(SystemExit) as stop:
        ws._template().fail("expected something else")
    assert stop.value.code == 2 and "prepare_swap: expected something else" in capsys.readouterr().err


# ---- 2. a recorded hard-coded container that is not installed is not offered ----------------------------------------------------

def test_a_hard_coded_container_that_is_not_installed_is_not_a_search_result_or_candidate():
    c = check([], search=lambda family: [{"name": HARD, "installed": False}, {"name": "org/searched", "installed": False}])
    assert c.status == "block" and [x["name"] for x in c.data["candidates"]] == ["org/searched"]
    assert HARD not in c.detail


def test_a_hard_coded_container_that_is_not_installed_is_not_a_ranked_choice():
    report = {"bundle_choices": [{"name": HARD, "class": "weights-only", "confidence": "stated"},
                                 {"name": "org/some-bundle", "class": "weights-only", "confidence": "stated"}]}
    c = check([], report=report)
    assert [x["name"] for x in c.data["candidates"]] == ["org/some-bundle"] and HARD not in c.detail


def test_when_every_ranked_choice_is_unoffered_the_candidates_are_listed_and_not_an_empty_ranking():
    report = {"bundle_choices": [{"name": HARD, "class": "weights-only", "confidence": "stated"}]}
    c = check([], report=report, search=lambda family: [{"name": "org/searched", "installed": False}])
    assert "ranked bundles:" not in c.detail and "org/searched" in c.detail
    assert [x["name"] for x in c.data["candidates"]] == ["org/searched"]


def test_a_hard_coded_container_is_not_a_closest_bundle_when_the_card_names_no_base():
    report = {"bundle_choices": [{"name": HARD, "class": "weights-only", "confidence": "weak"},
                                 {"name": "org/some-bundle", "class": "weights-only", "confidence": "weak"}]}
    c = check([], report=report, base_model=None)
    assert "org/some-bundle" in c.detail and HARD not in c.detail


def test_an_installed_v6_bundle_with_a_recorded_name_stays_a_candidate_to_be_judged_by_its_own_script(tmp_path):
    other = v6(tmp_path, HARD, GOOD.replace(BASE, "Qwen/Other-7B"), repo="Qwen/Other-7B")   # a patched replacement; it does not serve this base
    report = {"bundle_choices": [{"name": HARD, "class": "weights-only", "confidence": "stated"}]}
    c = check([other], report=report)
    assert c.status == "block" and [x["name"] for x in c.data["candidates"]] == [HARD]


# ---- 3. the inferred-base confirmation offers no refused package ------------------------------------------------------------------

def test_the_inferred_base_confirmation_does_not_offer_a_refused_package_from_the_report():
    sel = {"repo": BASE, "bundle": OPEN, "confidence": "inferred", "class": "weights-only"}
    report = {"model": "someone/fine-tune", "selected": sel, "candidates": [{"repo": BASE, "class_estimate": {"class": "weights-only"}}],
              "bundle_choices": [{"name": HARD, "repo": BASE, "installed": True, "schema": "5.1", "class": "weights-only", "confidence": "inferred", "group": 0},
                                 {"name": OPEN, "repo": BASE, "installed": True, "schema": "5.1", "class": "weights-only", "confidence": "inferred", "group": 0},
                                 {"name": "org/elsewhere", "repo": "x/y", "installed": False, "schema": "5.1", "class": "full-port", "confidence": "weak", "group": 2}],
              "ports": [], "evidence": [], "unavailable": [], "offline": False, "stated_bases": [], "generated_at": "t"}
    c = check([container(HARD), container(OPEN)], report=report, base_model=None)
    assert c.reason == "inferred-base-unconfirmed" and c.data["inferred"] is True
    names = [x["name"] for x in c.data["candidates"]]
    assert HARD not in names and names[0] == OPEN and "org/elsewhere" in names


# ---- 5. the recorded verdict is about container packages -------------------------------------------------------------------------

def test_a_v6_bundle_with_a_recorded_container_name_is_judged_by_its_own_run_sh(tmp_path):
    patched = v6(tmp_path, HARD, GOOD)
    usable, refused = ws.judge([patched], BASE, RECORDS)
    assert [i.name for i in usable] == [HARD] and refused == []


def test_a_v6_bundle_with_a_recorded_container_name_is_still_refused_when_its_script_fails(tmp_path):
    broken = v6(tmp_path, HARD, BAD)
    usable, refused = ws.judge([broken], BASE, RECORDS)
    assert usable == [] and "cannot take swapped weights" in refused[0][1] and "hard-codes" not in refused[0][1]


def test_the_container_of_that_name_is_still_refused():
    usable, refused = ws.judge([container(HARD)], BASE, RECORDS)
    assert usable == [] and "hard-codes its model repository" in refused[0][1]


# ---- 4. stages 2 and 4 choose among the packages the preflight accepted ----------------------------------------------------------

def _bundle_dir(root, name, repo, chips, run_sh):
    d = Path(root) / name
    d.mkdir(parents=True)
    (d / "tt_kernel_manifest.json").write_text(json.dumps({"schema_version": "6", "device_count": chips, "weights": {"repo_id": repo}}))
    if run_sh is not None:
        (d / "run.sh").write_text(run_sh)
    return d


def _containers(home, entries):
    root = Path(home) / ".cache" / "tt-model"
    out = {}
    for name, (repo, chips) in entries.items():
        mf = root / "pulled" / name.replace("/", "__") / "tt_kernel_manifest.json"
        mf.parent.mkdir(parents=True)
        mf.write_text(json.dumps({"schema_version": "5.1", "device_count": chips, "weights": {"repo_id": repo}}))
        out[name] = {"container": True, "image": f"tt-model/{name.split('/')[1]}:abc", "profile": "default", "profiles": ["default"],
                     "manifest": str(mf)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "installed.json").write_text(json.dumps(out))
    return Path(home)


NEAREST = "Qwen/Qwen3.8-27B"
MODEL = "Altworld/Hemmingway-1@1a5f363a"
GOOD_NEAREST = GOOD.replace(BASE, NEAREST)
BAD_NEAREST = BAD.replace(BASE, NEAREST)


@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "runs" / "altworld--hemmingway-1"
    (run_dir / "stages" / "0").mkdir(parents=True)
    (run_dir / "stages" / "0" / "delta.json").write_text(json.dumps(
        {"model": MODEL, "nearest_model": NEAREST + "@1d4bf0f2", "path": "weights-only"}))
    base = tmp_path / "hf" / "hub" / "models--Qwen--Qwen3.8-27B" / "snapshots" / "1d4bf0f2"
    new = tmp_path / "hf" / "hub" / "models--Altworld--Hemmingway-1" / "snapshots" / "1a5f363a"
    base.mkdir(parents=True)
    new.mkdir(parents=True)
    return dict(run_dir=run_dir, tt_model_root=tmp_path / "tt-model" / "models", cache_root=tmp_path / "cache",
                hf_home=tmp_path / "hf", inputs={"base": str(base), "model": str(new)})


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setattr(ws, "_shipped", lambda: dict(RECORDS))
    return RECORDS


def test_stage_2_takes_the_usable_container_when_the_only_bundle_cannot_take_swapped_weights(run, tmp_path, records):
    _bundle_dir(run["tt_model_root"], "org/bad-bundle", NEAREST, 2, BAD_NEAREST)
    home = _containers(tmp_path / "op", {OPEN: (NEAREST, 2)})
    cfg, problems = sd.draft(**run, chips=2, operator_home=home)
    assert problems == [] and cfg["package"] == OPEN and "bundle_dir" not in cfg


def test_stage_2_with_a_refused_bundle_and_no_other_package_says_nothing_serves_it(run, tmp_path, records):
    _bundle_dir(run["tt_model_root"], "org/bad-bundle", NEAREST, 2, BAD_NEAREST)
    cfg, problems = sd.draft(**run, chips=2, operator_home=_containers(tmp_path / "op", {}))
    assert cfg is None and any("no installed" in p for p in problems)


def test_stage_2_does_not_take_a_hard_coded_container(run, tmp_path, records):
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 2)})
    cfg, problems = sd.draft(**run, chips=2, operator_home=home)
    assert cfg is None and any("no installed" in p for p in problems)


def test_stage_2_keeps_a_bundle_whose_run_sh_can_take_swapped_weights(run, tmp_path, records):
    d = _bundle_dir(run["tt_model_root"], "org/good-bundle", NEAREST, 2, GOOD_NEAREST)
    cfg, problems = sd.draft(**run, chips=2, operator_home=_containers(tmp_path / "op", {OPEN: (NEAREST, 2)}))
    assert problems == [] and cfg["bundle_dir"] == str(d)


def test_stage_2_names_a_broken_record_file_instead_of_choosing_without_it(run, tmp_path, monkeypatch):
    def broken():
        raise ValueError("weights_source record: schema must be 1, got 2")
    monkeypatch.setattr(ws, "_shipped", broken)
    _bundle_dir(run["tt_model_root"], "org/good-bundle", NEAREST, 2, GOOD_NEAREST)
    cfg, problems = sd.draft(**run, chips=2, operator_home=_containers(tmp_path / "op", {}))
    assert cfg is None and any("schema must be 1" in p for p in problems)


def test_stage_4_leaves_a_refused_bundle_out_and_does_not_take_a_hard_coded_container(run, tmp_path, records):
    _bundle_dir(run["tt_model_root"], "org/bad-bundle", NEAREST, 2, BAD_NEAREST)
    good = _bundle_dir(run["tt_model_root"], "org/good-bundle-p150", NEAREST, 1, GOOD_NEAREST)
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 4)})
    configs, tests, notes = sd.draft_stage4(run_dir=run["run_dir"], tt_model_root=run["tt_model_root"], cache_root=run["cache_root"],
                                            hf_home=run["hf_home"], operator_home=home, inputs=run["inputs"], counts=(1, 2, 4))
    assert sorted(configs) == [1] and configs[1]["bundle_dir"] == str(good)
    assert any("2-chip" in n for n in notes) and any("4-chip" in n for n in notes)


def test_stage_4_takes_the_usable_container_for_a_count_whose_bundle_is_refused(run, tmp_path, records):
    _bundle_dir(run["tt_model_root"], "org/bad-bundle", NEAREST, 2, BAD_NEAREST)
    home = _containers(tmp_path / "op", {OPEN: (NEAREST, 2)})
    configs, tests, notes = sd.draft_stage4(run_dir=run["run_dir"], tt_model_root=run["tt_model_root"], cache_root=run["cache_root"],
                                            hf_home=run["hf_home"], operator_home=home, inputs=run["inputs"], counts=(2,))
    assert sorted(configs) == [2] and configs[2]["package"] == OPEN
