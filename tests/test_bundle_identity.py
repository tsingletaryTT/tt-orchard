# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The bundle the operator names is the bundle the run uses (Track C, item C3 of the Qwen4 drop-in plan).

`--base <bundle>` used to keep only the bundle's weights repository. On this box two bundles serve `Qwen/Qwen3.8-27B` on four chips
(the Qwen3.8 bundle and iteration 0 of the Qwen4 bundle), so the base check reported whichever sorted first and stage 2 left the
choice to the agent. Now the preflight keeps the named bundle, the run records its directory as the `bundle` input (stage 0 reads its
orchard-capabilities.json from there), and stages 2 and 4 choose by chip count among the bundle's family only: the v6 bundles that serve
the same weights repository and ship the same capabilities file (or both none). A weights repository that bundles of different
families serve, or that two bundles serve on the same chip count, is refused when no bundle is named."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from orchard import nearest
from orchard import preflight as pf
from orchard import swap_draft as sd
from orchard.nearest import Installed
from swap_run_sh import valid_run_sh

REPO = "Qwen/Qwen3.8-27B"
CAPS = json.dumps({"schema": 1, "config_keys": {"num_hidden_layers": {"type": "int"}}})
QWEN38 = "mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin"
QWEN4_4 = "mando2222/qwen4-27b-dense-v6-thin"
QWEN4_2 = "mando2222/qwen4-27b-dense-p300-v6-thin"


def bundle(root: Path, name: str, repo: str, chips: int, caps: str | None = None, schema="6") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "tt_kernel_manifest.json").write_text(json.dumps({"schema_version": schema, "device_count": chips, "weights": {"repo_id": repo}}))
    (d / "run.sh").write_text(valid_run_sh(repo))
    if caps is not None:
        (d / nearest.CAPABILITIES_FILE).write_text(caps)
    return d


def inst(root: Path, name: str) -> Installed:
    return next(b for b in nearest.read_installed(root) if b.name == name)


# ---- nearest: the capabilities digest, the family, the ambiguity ------------------------------------------------------------

def test_the_capabilities_digest_is_the_files_sha256_or_none(tmp_path):
    a = bundle(tmp_path, "a/a", REPO, 4, CAPS)
    b = bundle(tmp_path, "b/b", REPO, 4)
    assert nearest.capabilities_digest(str(a)) == hashlib.sha256(CAPS.encode()).hexdigest()
    assert nearest.capabilities_digest(str(b)) is None
    assert nearest.capabilities_digest("") is None


def test_an_unreadable_capabilities_file_has_a_digest_no_other_bundle_shares(tmp_path):
    a, b = bundle(tmp_path, "a/a", REPO, 4), bundle(tmp_path, "b/b", REPO, 2)
    real = tmp_path / "caps.json"
    real.write_text(CAPS)
    os.symlink(real, a / nearest.CAPABILITIES_FILE)
    os.symlink(real, b / nearest.CAPABILITIES_FILE)
    da, db = nearest.capabilities_digest(str(a)), nearest.capabilities_digest(str(b))
    assert da is not None and db is not None and da != db


def test_no_directory_means_no_capabilities_even_with_a_stray_file_in_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / nearest.CAPABILITIES_FILE).write_text(CAPS)
    monkeypatch.chdir(tmp_path)
    assert nearest.capabilities_digest("") is None


def test_a_path_that_cannot_be_checked_gets_a_digest_of_its_own():
    a, b = nearest.capabilities_digest("a\0b"), nearest.capabilities_digest("c\0d")
    assert a and b and a != b


def test_the_family_is_the_bundles_with_the_same_repo_and_the_same_capabilities(tmp_path):
    bundle(tmp_path, QWEN38, REPO, 2)
    bundle(tmp_path, QWEN4_4, REPO, 4, CAPS)
    bundle(tmp_path, QWEN4_2, REPO, 2, CAPS)
    bundle(tmp_path, "x/other-caps", REPO, 1, CAPS.replace("int", "number"))
    bundle(tmp_path, "y/other-repo", "Qwen/Other", 2, CAPS)
    every = nearest.read_installed(tmp_path)
    assert sorted(b.name for b in nearest.bundle_family(inst(tmp_path, QWEN4_4), every)) == [QWEN4_2, QWEN4_4]
    assert [b.name for b in nearest.bundle_family(inst(tmp_path, QWEN38), every)] == [QWEN38]


def test_a_container_package_is_never_in_a_bundles_family(tmp_path):
    bundle(tmp_path, QWEN38, REPO, 4)
    box = Installed("tt-hous/qwen3.8-27b", REPO, "5.1", 2)
    assert [b.name for b in nearest.bundle_family(inst(tmp_path, QWEN38), nearest.read_installed(tmp_path) + [box])] == [QWEN38]


def test_ambiguity(tmp_path):
    q38, q4, q42 = bundle(tmp_path, QWEN38, REPO, 4), bundle(tmp_path, QWEN4_4, REPO, 4, CAPS), bundle(tmp_path, QWEN4_2, REPO, 2, CAPS)
    every = nearest.read_installed(tmp_path)
    by = {b.name: b for b in every}
    assert nearest.ambiguity([by[QWEN4_4], by[QWEN4_2]]) is None                 # one family, one bundle per chip count
    assert nearest.ambiguity([by[QWEN38]]) is None
    why = nearest.ambiguity([by[QWEN38], by[QWEN4_4]])
    assert why and QWEN38 in why and QWEN4_4 in why
    assert nearest.ambiguity([by[QWEN38], by[QWEN4_2]])                         # different families on different chip counts
    same = [Installed("a/x", REPO, "6", 4, str(q38)), Installed("b/y", REPO, "6", 4, str(q38))]
    assert "4 chips" in nearest.ambiguity(same)
    box = Installed("tt-hous/qwen3.8-27b", REPO, "5.1", 4)
    assert nearest.ambiguity([by[QWEN38], box]) is None                          # a container package is a fallback, not a rival
    del q4, q42


# ---- the base check ------------------------------------------------------------------------------------------------------------

def check(installed, *, base_bundle=None, override=REPO):
    return pf.check_base(model_id="Qwen/Qwen4-27B", hub=None, base_override=override, local_snapshot=lambda m: Path("/snap"),
                         installed=installed, search=None, base_bundle=base_bundle)


def test_two_bundles_of_different_families_on_one_repo_block_until_one_is_named(tmp_path):
    bundle(tmp_path, QWEN38, REPO, 4)
    bundle(tmp_path, QWEN4_4, REPO, 4, CAPS)
    every = nearest.read_installed(tmp_path)
    c = check(every)
    assert c.status == pf.BLOCK and c.reason == "nearest-model-missing"
    assert QWEN38 in c.detail and QWEN4_4 in c.detail and "--base" in c.detail
    assert sorted(x["name"] for x in c.data["candidates"]) == [QWEN38, QWEN4_4]
    assert all(x["installed"] is True for x in c.data["candidates"])


def test_the_named_bundle_is_kept_with_its_directory(tmp_path):
    bundle(tmp_path, QWEN38, REPO, 4)
    d = bundle(tmp_path, QWEN4_4, REPO, 4, CAPS)
    c = check(nearest.read_installed(tmp_path), base_bundle=QWEN4_4)
    assert c.status == pf.OK, c.detail
    assert c.data["bundle"] == QWEN4_4 and c.data["bundle_dir"] == str(d) and c.data["bundle_named"] is True
    assert QWEN4_4 in c.detail


def test_one_family_on_two_chip_counts_needs_no_name(tmp_path):
    d4 = bundle(tmp_path, QWEN4_4, REPO, 4, CAPS)
    bundle(tmp_path, QWEN4_2, REPO, 2, CAPS)
    c = check(nearest.read_installed(tmp_path))
    assert c.status == pf.OK, c.detail
    assert c.data["bundle_named"] is False and c.data["bundle_dir"] in (str(d4), str(tmp_path / QWEN4_2))


def test_a_single_bundle_records_its_directory(tmp_path):
    d = bundle(tmp_path, QWEN38, REPO, 4)
    c = check(nearest.read_installed(tmp_path))
    assert c.status == pf.OK and c.data["bundle"] == QWEN38 and c.data["bundle_dir"] == str(d)


def test_a_container_package_alone_records_no_directory(tmp_path):
    c = check([Installed("tt-hous/qwen3.8-27b", REPO, "5.1", 4)])
    assert c.status == pf.WARN and c.data["bundle_dir"] is None


def test_a_named_bundle_that_does_not_serve_the_base_blocks(tmp_path):
    bundle(tmp_path, QWEN38, REPO, 4)
    c = check(nearest.read_installed(tmp_path), base_bundle="nobody/else")
    assert c.status == pf.BLOCK and "nobody/else" in c.detail


# ---- the command: resolve, preflight, inputs -----------------------------------------------------------------------------------

def test_resolve_base_bundle_keeps_the_bundle_name():
    from orchard import cli
    from types import SimpleNamespace
    sig = SimpleNamespace(installed_bundles=lambda: [Installed(QWEN4_4, REPO, "6", 4, "/m/q4")], search_bundles=None)
    assert cli.resolve_base_bundle(QWEN4_4, sig, None, None, None) == (REPO, QWEN4_4, None)
    assert cli.resolve_base_bundle(REPO, sig, None, None, None) == (REPO, None, None)
    assert cli.resolve_base_bundle("Org/Model", sig, None, None, None, dry_run=True) == ("Org/Model", None, None)
    assert cli.resolve_base(QWEN4_4, sig, None, None, None) == (REPO, None)


@pytest.fixture
def world(tmp_path):
    import test_cli
    root = tmp_path / "models"
    q38 = bundle(root, QWEN38, REPO, 4)
    q4 = bundle(root, QWEN4_4, REPO, 4, CAPS)
    w = test_cli.BaseWorld(installed=nearest.read_installed(root), local={REPO: tmp_path / "base-snap", "jialinyyzz/humanizer": tmp_path / "snap"})
    return {"w": w, "q38": q38, "q4": q4, "tmp": tmp_path}


def bringup(conf, args, w):
    import test_cli
    rec = test_cli.Recorder()
    out = test_cli.Pipe()
    from orchard import cli
    code = cli.main(["--config", str(conf), "bringup", "jialinyyzz/humanizer", *args], env={}, stdout=out, signals=w.signals(),
                    supervisor_main=rec.supervisor, fetcher=rec.fetcher, chooser=None, puller=w.puller)
    return code, out.getvalue(), rec


@pytest.fixture
def conf(tmp_path):
    p = tmp_path / "bringup.toml"
    p.write_text(f'runs_root = "{tmp_path}/runs"\nhf_home = "{tmp_path}/hf"\ncache_root = "{tmp_path}/cache"\n'
                 '[coder]\ntarget = "pkg/coder"\nport = 8000\nchips = 4\n')
    return p


def test_naming_the_bundle_records_its_directory_as_the_runs_bundle_input(conf, world):
    code, out, rec = bringup(conf, ["--base", QWEN4_4], world["w"])
    assert code == 0, out
    assert f"bundle={world['q4']}" in rec.argv
    assert f"base={world['tmp'] / 'base-snap'}" in rec.argv


def test_naming_only_the_repo_shared_by_two_families_is_refused_with_both_bundles(conf, world):
    code, out, rec = bringup(conf, ["--base", REPO], world["w"])
    assert code == 2 and rec.argv is None
    assert f"--base {QWEN38}" in out and f"--base {QWEN4_4}" in out


def test_a_resumed_run_keeps_the_bundle_it_started_with(conf, world, tmp_path):
    from orchard.ledger import Ledger
    from orchard import bringup_config
    cfg = bringup_config.load(conf)
    run_dir = bringup_config.run_dir(cfg, "jialinyyzz/humanizer")
    run_dir.mkdir(parents=True)
    with Ledger(run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="jialinyyzz/humanizer", inputs={"model": "/m", "bundle": str(world["q4"])})
    code, out, rec = bringup(conf, ["--base", QWEN38], world["w"])
    assert code == 0, out
    assert f"bundle={world['q4']}" in rec.argv and f"bundle={world['q38']}" not in rec.argv


# ---- stages 2 and 4 choose within the family ----------------------------------------------------------------------------------

@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "runs" / "qwen--qwen4-27b"
    (run_dir / "stages" / "0").mkdir(parents=True)
    (run_dir / "stages" / "0" / "delta.json").write_text(json.dumps(
        {"model": "Qwen/Qwen4-27B@1a5f363a", "nearest_model": REPO + "@1d4bf0f2", "path": "weights-only"}))
    base, new = tmp_path / "hf" / "base", tmp_path / "hf" / "new"
    base.mkdir(parents=True)
    new.mkdir(parents=True)
    return dict(run_dir=run_dir, tt_model_root=tmp_path / "models", cache_root=tmp_path / "cache", hf_home=tmp_path / "hf",
                inputs={"base": str(base), "model": str(new)}, chips=2)


def test_without_a_bundle_input_stage_2_chooses_as_before(run):
    b = bundle(run["tt_model_root"], QWEN38, REPO, 2)
    cfg, problems = sd.draft(**run)
    assert problems == [] and cfg["bundle_dir"] == str(b)


def test_the_two_chip_stage_never_falls_back_to_another_familys_bundle(run):
    bundle(run["tt_model_root"], QWEN38, REPO, 2)
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    cfg, problems = sd.draft(**{**run, "inputs": {**run["inputs"], "bundle": str(q4)}})
    assert cfg is None and any(QWEN4_4 in p and "2 chips" in p for p in problems), problems


def test_the_two_chip_sibling_with_the_same_capabilities_is_chosen(run):
    bundle(run["tt_model_root"], QWEN38, REPO, 2)
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    q42 = bundle(run["tt_model_root"], QWEN4_2, REPO, 2, CAPS)
    cfg, problems = sd.draft(**{**run, "inputs": {**run["inputs"], "bundle": str(q4)}})
    assert problems == [] and cfg["bundle_dir"] == str(q42)


def test_a_sibling_with_other_capabilities_is_not_chosen(run):
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    bundle(run["tt_model_root"], QWEN4_2, REPO, 2, CAPS.replace("int", "number"))
    cfg, problems = sd.draft(**{**run, "inputs": {**run["inputs"], "bundle": str(q4)}})
    assert cfg is None and problems


def test_the_named_bundle_wins_on_its_own_chip_count(run):
    bundle(run["tt_model_root"], QWEN38, REPO, 2)
    q4 = bundle(run["tt_model_root"], QWEN4_2, REPO, 2, CAPS)
    cfg, problems = sd.draft(**{**run, "inputs": {**run["inputs"], "bundle": str(q4)}})
    assert problems == [] and cfg["bundle_dir"] == str(q4)


def test_a_bundle_input_that_is_not_installed_or_serves_another_model_is_a_problem(run, tmp_path):
    bundle(run["tt_model_root"], QWEN38, REPO, 2)
    cfg, problems = sd.draft(**{**run, "inputs": {**run["inputs"], "bundle": str(tmp_path / "nowhere")}})
    assert cfg is None and any("not an installed bundle" in p for p in problems)
    other = bundle(run["tt_model_root"], "x/other", "Qwen/Other", 2)
    cfg, problems = sd.draft(**{**run, "inputs": {**run["inputs"], "bundle": str(other)}})
    assert cfg is None and any("Qwen/Other" in p for p in problems)


def test_a_named_bundle_with_capabilities_never_falls_back_to_a_container_package(run, tmp_path, monkeypatch):
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    monkeypatch.setattr(sd, "installed_containers", lambda home: [{"name": "tt-hous/qwen3.8", "repo": REPO, "chips": 2, "profile": "p300"}])
    cfg, problems = sd.draft(**{**run, "inputs": {**run["inputs"], "bundle": str(q4)}}, operator_home=tmp_path)
    assert cfg is None and problems


def test_stage_4_drafts_each_count_from_the_family(run, tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "installed_containers", lambda home: [])
    bundle(run["tt_model_root"], QWEN38, REPO, 2)
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    q42 = bundle(run["tt_model_root"], QWEN4_2, REPO, 2, CAPS)
    configs, tests, notes = sd.draft_stage4(run_dir=run["run_dir"], tt_model_root=run["tt_model_root"], cache_root=run["cache_root"],
                                            hf_home=run["hf_home"], operator_home=tmp_path, counts=[2, 4],
                                            inputs={**run["inputs"], "bundle": str(q4)})
    assert notes == [] and configs[2]["bundle_dir"] == str(q42) and configs[4]["bundle_dir"] == str(q4)
    (run["tt_model_root"] / QWEN4_2 / nearest.CAPABILITIES_FILE).unlink()
    configs, tests, notes = sd.draft_stage4(run_dir=run["run_dir"], tt_model_root=run["tt_model_root"], cache_root=run["cache_root"],
                                            hf_home=run["hf_home"], operator_home=tmp_path, counts=[2, 4],
                                            inputs={**run["inputs"], "bundle": str(q4)})
    assert 2 not in configs and any("2-chip" in n for n in notes), notes


def test_an_agent_written_config_that_names_a_bundle_outside_the_family_is_refused(run, tmp_path):
    q38 = bundle(run["tt_model_root"], QWEN38, REPO, 2)
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    stage = run["run_dir"] / "stages" / "2"
    stage.mkdir(parents=True)
    (stage / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": str(q38)}))
    assert sd.nearest_problem(run["run_dir"], stage / "swap_config.json", tt_model_root=run["tt_model_root"]) is None
    msg = sd.nearest_problem(run["run_dir"], stage / "swap_config.json", tt_model_root=run["tt_model_root"], bundle=str(q4))
    assert msg and QWEN4_4 in msg
    (stage / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": str(q4)}))
    assert sd.nearest_problem(run["run_dir"], stage / "swap_config.json", tt_model_root=run["tt_model_root"], bundle=str(q4)) is None


def test_the_supervisor_checks_an_agent_written_config_against_the_runs_bundle(run, tmp_path, monkeypatch):
    """_read_test (stage 2) and the stage 4 plan read the run's `bundle` input, the bundle stage 0 judged."""
    from orchard.supervisor import NOTE_KEYS, Supervisor
    q38 = bundle(run["tt_model_root"], QWEN38, REPO, 2)
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    stage = run["run_dir"] / "stages" / "2"
    stage.mkdir(parents=True)
    (stage / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": str(q38)}))
    (stage / "hw_test.json").write_text(json.dumps({"command": "true", "deadline_s": 60}))
    (stage / "handoff.json").write_text(json.dumps({k: "x" for k in NOTE_KEYS}))
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir = run["run_dir"]
    sup.paths = type("P", (), {"tt_model_root": run["tt_model_root"], "operator_home": tmp_path / "op"})()
    sup.inputs = {"bundle": str(q4)}
    _, problems = sup._read_test(stage)
    assert any(QWEN4_4 in m for m in problems), problems
    sup.inputs = {}
    _, problems = sup._read_test(stage)
    assert not any(QWEN4_4 in m for m in problems), problems
    stage4 = run["run_dir"] / "stages" / "4"
    (stage4 / "configs" / "2").mkdir(parents=True)
    (stage4 / "configs" / "2" / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": str(q38)}))
    import orchard.supervisor as sup_mod
    monkeypatch.setattr(sup_mod, "read_plan", lambda *a, **k: ([type("T", (), {"chips": 2})()], []))
    sup.required_chips, sup.home = None, tmp_path
    sup.inputs = {"bundle": str(q4)}
    got = sup._read_plan(type("S", (), {"boards": 2, "budget_s": 60})(), stage4)
    assert any(QWEN4_4 in m for m in got[1]), got
