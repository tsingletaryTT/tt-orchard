# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7 with `graft_mtp`: the package serves a model that has no mtp.* tensors with the speculative drafter on.

The package ships no weights. Its prepare_model_dir.py adds the nearest model's mtp.* tensors to the
served model directory at each start, from that model's pinned snapshot, as prepare_swap.py does in
stage 2. These tests use real (tiny) safetensors files, so the bytes the package builds are compared with
the bytes prepare_swap.py builds."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prepare_swap_graft import (BACKBONE, MTP, SHARD, put_link, read_header,  # noqa: E402
                                     tensor_bytes, write_safetensors)

from orchard.package import (PackageError, finish, prepare_verify, read_run, serving_env,  # noqa: E402
                             serving_problems, stage_all)
from orchard.stages import gate_package  # noqa: E402
from package_fakes import (BASE, BASE_REV, DRAFTER, NEW, NEW_REV, SOURCE_EXTRA, fake_bin,  # noqa: E402
                           fake_boot_result, make_clef_like, make_run, make_source, write)

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
TEMPLATES = Path(__file__).resolve().parent.parent / "orchard" / "package_templates"
SWAP_TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates" / "prepare_swap.py"
SHARD_FILE = "model-00001-of-00001.safetensors"
DEVICE_SAMPLING = '{"tt": {"l1_small_size": 24576, "sample_on_device_mode": "decode_only"}}'


def real_file(snap: Path, name: str, tensors: dict) -> None:
    """Replace a snapshot file (a link to a text blob) by a real safetensors file, keeping the link."""
    write_safetensors(Path(os.path.realpath(snap / name)), tensors)


@pytest.fixture
def world(tmp_path, monkeypatch, stub_tools):
    """A run shaped like AREX-2's: stage 2 served with the drafter off and host sampling, the model has no
    mtp.* tensors, the nearest model's snapshot has three, in one shard beside lm_head."""
    models = tmp_path / "models"
    src = make_source(models, env={"ARCH_NAME": "blackhole", "QWEN36_DRAFTER": "dflash2",
                                   "DFLASH_WEIGHTS": f"{DRAFTER}@" + "d" * 40, "QWEN36_DFLASH_TP": "1"})
    (src / "run.sh").write_text((src / "run.sh").read_text().replace(
        '{"tt": {"l1_small_size": 24576}}', DEVICE_SAMPLING))
    r = make_run(tmp_path, src)
    make_clef_like(r["run"], src)
    s2 = r["run"] / "stages/2/run.sh"
    s2.write_text(s2.read_text().replace(', "sample_on_device_mode": "decode_only"', ""))
    base = r["hf_op"] / "hub" / "models--Qwen--Qwen3.8-27B" / "snapshots" / BASE_REV
    real_file(base, SHARD_FILE, {**BACKBONE, **MTP})
    real_file(r["snapshot"], SHARD_FILE, {"model.layers.0.weight": ("BF16", [4, 4]), **BACKBONE})
    cfg = json.loads((r["run"] / "stages/2/swap_config.json").read_text())
    write(r["run"] / "stages/2/swap_config.json", {**cfg, "base_snapshot": str(base)})
    delta = json.loads((r["run"] / "stages/0/delta.json").read_text())
    write(r["run"] / "stages/0/delta.json", {**delta, "mtp": {"nearest_tensors": len(MTP), "new_tensors": 0}})
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    return {**r, "models": models, "src": src, "env": env, "tmp": tmp_path, "base": base}


def stage(w, **kw):
    stage_dir = w["run"] / "stages/7"
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir, stage_all(w["run"], stage_dir, namespace="episod", models_root=w["models"], env=w["env"], **kw)


def finished(w):
    stage_dir, package = stage(w, graft_mtp=True)
    fake_boot_result(stage_dir)
    finish(w["run"], stage_dir)
    return stage_dir, w["run"] / package["profiles"][0]["dir"]


# ---- staging ----------------------------------------------------------------------------------------

def test_a_grafted_package_keeps_the_drafter_and_on_device_sampling(world):
    _, package = stage(world, graft_mtp=True)
    pkg = world["run"] / package["profiles"][0]["dir"]
    run_sh = (pkg / "run.sh").read_text()
    m = json.loads((pkg / "tt_kernel_manifest.json").read_text())
    assert "sample_on_device_mode" in run_sh
    assert m["env"]["QWEN36_DRAFTER"] == "dflash2" and "DFLASH_WEIGHTS" in m["env"]


def test_the_package_records_the_graft_in_graft_json_and_in_package_json(world):
    _, package = stage(world, graft_mtp=True)
    pkg = world["run"] / package["profiles"][0]["dir"]
    graft = json.loads((pkg / "graft.json").read_text())
    assert package["mtp_graft"] == graft
    assert graft["repo"] == BASE and graft["revision"] == BASE_REV
    assert graft["file"] == SHARD_FILE and graft["shard"] == SHARD
    assert graft["tensors"] == sorted(MTP)
    assert len(graft["sha256"]) == 64 and graft["bytes"] > 0
    assert "/home/" not in json.dumps(graft) and str(world["tmp"]) not in json.dumps(graft)


def test_the_recorded_sha256_is_the_one_prepare_swap_computes(world, tmp_path):
    _, package = stage(world, graft_mtp=True)
    graft = json.loads((world["run"] / package["profiles"][0]["dir"] / "graft.json").read_text())
    spec = importlib.util.spec_from_file_location("ps", SWAP_TEMPLATE)
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    plan = ps.plan_mtp_graft(world["base"], world["snapshot"])
    digest, total = ps.write_graft_shard(plan, tmp_path / "x.safetensors")
    assert (graft["sha256"], graft["bytes"]) == (digest, total)


def test_without_the_option_the_package_is_the_drafter_off_package(world):
    _, package = stage(world)
    pkg = world["run"] / package["profiles"][0]["dir"]
    assert not (pkg / "graft.json").exists() and "mtp_graft" not in package
    assert "sample_on_device_mode" not in (pkg / "run.sh").read_text()
    assert json.loads((pkg / "tt_kernel_manifest.json").read_text())["env"]["QWEN36_DRAFTER"] == ""


def test_a_run_whose_stage_2_grafted_gives_a_grafted_package_without_the_option(world):
    # Stage 2 served with the graft: its run.sh keeps the drafter, and its record says so.
    stage_2_with_the_real_graft(world)
    _, package = stage(world)
    assert package["mtp_graft"]["file"] == SHARD_FILE


def stage_2_with_the_real_graft(w):
    """Stage 2's model-dir as prepare_swap.py itself builds it with `graft_mtp`: the graft shard is a regular file
    next to the links to the model's own shards. The script runs in the run's stage 2 directory."""
    s2 = w["run"] / "stages/2"
    for name in ("config.json", "preprocessor_config.json"):          # the files prepare_swap copies from the nearest model
        put_link(w["base"], name, "{}")
    shutil.copy(SWAP_TEMPLATE, s2 / "prepare_swap.py")
    cfg = json.loads((s2 / "swap_config.json").read_text())
    write(s2 / "swap_config.json", {**cfg, "graft_mtp": True})
    done = subprocess.run([sys.executable, str(s2 / "prepare_swap.py")], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (s2 / "model-dir" / SHARD).is_symlink() and (s2 / "model-dir" / SHARD).is_file()
    (s2 / "run.sh").write_text((w["src"] / "run.sh").read_text())      # stage 2 served with the drafter on


def test_a_run_whose_stage_2_really_grafted_is_read_and_packaged(world):
    # The graft shard is a new regular file, so it links into no Hugging Face snapshot. Only the validated
    # shard is exempt from the backbone provenance check.
    stage_2_with_the_real_graft(world)
    facts = read_run(world["run"])
    assert facts.stage_2_grafted and facts.graft["file"] == SHARD_FILE
    _, package = stage(world)
    assert package["mtp_graft"]["sha256"] == json.loads(
        (world["run"] / "stages/2/model-dir-sources.json").read_text())["mtp_graft"]["sha256"]


def graft_record_path(w):
    return w["run"] / "stages/2/model-dir-sources.json"


def edit_graft_record(w, **changes):
    rec = json.loads(graft_record_path(w).read_text())
    rec["mtp_graft"].update(changes)
    graft_record_path(w).write_text(json.dumps(rec))


def test_stage_2s_graft_shard_must_be_the_one_the_record_describes(world):
    stage_2_with_the_real_graft(world)
    shard = world["run"] / "stages/2/model-dir" / SHARD
    with open(shard, "r+b") as f:                       # one byte of the tensors changed
        f.seek(-1, os.SEEK_END)
        f.write(b"\xff")
    with pytest.raises(PackageError, match="does not match the sha256"):
        read_run(world["run"])


def test_a_shard_and_a_record_that_agree_with_each_other_but_not_with_the_nearest_model_are_refused(world):
    stage_2_with_the_real_graft(world)
    shard = world["run"] / "stages/2/model-dir" / SHARD
    write_safetensors(shard, {"mtp.fc.weight": ("BF16", [4, 8])})        # some other bytes, recorded as the graft
    edit_graft_record(world, sha256=hashlib.sha256(shard.read_bytes()).hexdigest())
    with pytest.raises(PackageError, match="not what the nearest model's snapshot"):
        read_run(world["run"])


def test_a_graft_record_for_another_revision_of_the_nearest_model_is_refused(world):
    stage_2_with_the_real_graft(world)
    edit_graft_record(world, base_revision="e" * 40)
    with pytest.raises(PackageError, match="base_revision"):
        read_run(world["run"])


@pytest.mark.parametrize("changes, reason", [({"file": "other.safetensors"}, "must be"),
                                             ({"sha256": "abc"}, "64 hex")])
def test_a_malformed_graft_record_is_refused(world, changes, reason):
    stage_2_with_the_real_graft(world)
    edit_graft_record(world, **changes)
    with pytest.raises(PackageError, match=reason):
        read_run(world["run"])


def test_a_graft_record_that_is_not_an_object_exempts_nothing(world):
    stage_2_with_the_real_graft(world)
    graft_record_path(world).write_text(json.dumps({"schema": 1, "mtp_graft": ["not", "a", "record"]}))
    with pytest.raises(PackageError, match="links into no Hugging Face snapshot"):
        read_run(world["run"])


def test_a_graft_record_with_a_new_snapshot_that_is_not_pinned_says_so(world):
    stage_2_with_the_real_graft(world)
    cfg = json.loads((world["run"] / "stages/2/swap_config.json").read_text())
    write(world["run"] / "stages/2/swap_config.json", {**cfg, "new_snapshot": str(world["tmp"] / "not-a-snapshot")})
    with pytest.raises(PackageError, match="is not a pinned snapshot directory"):
        read_run(world["run"])


def test_a_new_snapshot_named_by_a_commit_but_missing_is_refused_before_the_weights_are_read(world):
    stage_2_with_the_real_graft(world)
    cfg = json.loads((world["run"] / "stages/2/swap_config.json").read_text())
    write(world["run"] / "stages/2/swap_config.json", {**cfg, "new_snapshot": str(world["tmp"] / ("c" * 40))})
    with pytest.raises(PackageError, match="is not a pinned snapshot directory"):
        read_run(world["run"])


def test_a_graft_shard_that_is_a_link_is_refused(world, tmp_path):
    stage_2_with_the_real_graft(world)
    shard = world["run"] / "stages/2/model-dir" / SHARD
    real = tmp_path / "elsewhere.safetensors"
    shutil.copy(shard, real)
    shard.unlink()
    shard.symlink_to(real)
    with pytest.raises(PackageError, match="not the regular file"):
        read_run(world["run"])


def test_only_the_graft_shard_is_exempt_from_the_provenance_check(world, tmp_path):
    stage_2_with_the_real_graft(world)
    write_safetensors(world["run"] / "stages/2/model-dir/model-extra.safetensors", dict(BACKBONE))
    with pytest.raises(PackageError, match="model-extra.safetensors links into no Hugging Face snapshot"):
        read_run(world["run"])


def test_a_graft_shard_without_a_graft_record_is_not_exempt(world):
    stage_2_with_the_real_graft(world)
    graft_record_path(world).write_text(json.dumps({"schema": 1}))        # the record that says it was grafted is gone
    with pytest.raises(PackageError, match="links into no Hugging Face snapshot"):
        read_run(world["run"])


def test_a_package_that_adds_the_graft_at_stage_7_still_needs_every_stage_2_weight_to_link_into_a_snapshot(world):
    write_safetensors(world["run"] / "stages/2/model-dir" / SHARD, dict(MTP))      # a stray file, no stage 2 record
    with pytest.raises(PackageError, match="links into no Hugging Face snapshot"):
        read_run(world["run"], graft_mtp=True)


@pytest.mark.parametrize("what, change, reason", [
    ("the model already has mtp tensors", lambda w: real_file(w["snapshot"], SHARD_FILE, {**BACKBONE, "mtp.fc.weight": ("BF16", [4, 8])}),
     "already has mtp."),
    ("the nearest model has none", lambda w: real_file(w["base"], SHARD_FILE, dict(BACKBONE)), "no mtp."),
    ("the model's weights cannot be read", lambda w: Path(os.path.realpath(w["snapshot"] / SHARD_FILE)).write_text("text"),
     "cannot tell"),
    ("the model ships a file with the graft shard's name",
     lambda w: put_link(w["snapshot"], SHARD, b"ships its own"), "ships a file named"),
])
def test_a_graft_that_cannot_be_made_stops_staging(world, what, change, reason):
    change(world)
    with pytest.raises(PackageError, match=reason):
        stage(world, graft_mtp=True)
    assert not (world["run"] / "stages/7/package").exists() or not any((world["run"] / "stages/7/package").iterdir())


def test_a_graft_needs_the_nearest_models_pinned_snapshot(world):
    cfg = json.loads((world["run"] / "stages/2/swap_config.json").read_text())
    cfg.pop("base_snapshot")
    write(world["run"] / "stages/2/swap_config.json", cfg)
    with pytest.raises(PackageError, match="base_snapshot|nearest model's snapshot"):
        stage(world, graft_mtp=True)


def test_the_boot_check_can_reach_the_nearest_models_shard(world):
    stage_dir, package = stage(world, graft_mtp=True)
    cfg = json.loads((stage_dir / "hw_test.json").read_text())
    assert BASE in package["hf_linked"]
    assert (stage_dir / "verify/hf/hub/models--Qwen--Qwen3.8-27B").is_symlink()
    assert cfg["command"]


# ---- serving_problems ---------------------------------------------------------------------------------

DRAFTER_ON = 'export QWEN36_DRAFTER="dflash2"\n' + f'--additional-config \'{DEVICE_SAMPLING}\''


def test_a_drafter_on_package_for_a_model_without_mtp_tensors_and_without_a_graft_is_refused():
    problems = serving_problems(DRAFTER_ON, {"QWEN36_DRAFTER": "dflash2"}, drafter_off=False, host_sampling=False,
                                model_has_mtp=False, graft=False)
    assert any("graft.json" in p and "mtp" in p for p in problems), problems


def test_a_drafter_on_package_with_a_graft_or_with_mtp_tensors_is_accepted():
    env = {"QWEN36_DRAFTER": "dflash2"}
    assert serving_problems(DRAFTER_ON, env, drafter_off=False, host_sampling=False, model_has_mtp=False, graft=True) == []
    assert serving_problems(DRAFTER_ON, env, drafter_off=False, host_sampling=False, model_has_mtp=True, graft=False) == []
    assert serving_problems(DRAFTER_ON, env, drafter_off=False, host_sampling=False, model_has_mtp=None, graft=False) == []


def test_a_graft_with_the_drafter_off_is_refused():
    problems = serving_problems('export QWEN36_DRAFTER=""\n', {"QWEN36_DRAFTER": ""}, drafter_off=False,
                                host_sampling=False, model_has_mtp=False, graft=True)
    assert any("drafter" in p and "graft" in p for p in problems), problems


def test_the_manifest_decides_when_run_sh_does_not_set_the_drafter():
    on = {"QWEN36_DRAFTER": "dflash2"}
    assert serving_problems("", on, drafter_off=False, host_sampling=False, model_has_mtp=False, graft=False)
    assert serving_problems("", {"QWEN36_DRAFTER": ""}, drafter_off=False, host_sampling=False, model_has_mtp=False, graft=True)
    assert serving_problems("", on, drafter_off=False, host_sampling=False, model_has_mtp=False, graft=True) == []


def test_the_last_drafter_export_in_run_sh_decides():
    text = 'export QWEN36_DRAFTER="dflash2"\nexport QWEN36_DRAFTER=""\n'
    assert serving_problems(text, {}, drafter_off=False, host_sampling=False, model_has_mtp=False, graft=True)
    assert serving_problems('export QWEN36_DRAFTER=""\nexport QWEN36_DRAFTER="dflash2"\n', {}, drafter_off=False,
                            host_sampling=False, model_has_mtp=False, graft=False)


def test_the_drafter_environment_is_kept_for_a_graft():
    env = {"QWEN36_DRAFTER": "dflash2", "DFLASH_WEIGHTS": f"{DRAFTER}@x"}
    assert serving_env(env, drafter_off=False) == env


# ---- the card and the gate -----------------------------------------------------------------------------

def test_the_card_says_what_was_grafted_and_what_was_not_measured(world):
    _, pkg = finished(world)
    card = (pkg / "README.md").read_text()
    flat = " ".join(card.split())
    assert "mtp.* tensors of Qwen/Qwen3.8-27B" in flat and "added at each start" in flat
    assert "Speculative decoding uses the drafter" in card and "Speculative decoding is off" not in card
    assert "Sampling runs on the host" not in card
    assert "acceptance rate" in card.split("## Not measured")[1]
    assert "`graft.json`" in card


def test_the_card_names_the_count_the_revision_the_file_and_the_limit(world):
    _, pkg = finished(world)
    flat = " ".join((pkg / "README.md").read_text().split())
    assert f"reads the {len(MTP)} mtp.* tensors of {BASE}" in flat
    assert f"revision `{BASE_REV}`, file `{SHARD_FILE}`" in flat
    assert f"`{SHARD}`" in flat
    limits = flat.split("## Limitations")[1].split("## Risks")[0]
    assert "How often the drafter's guesses are accepted on this model was not measured" in limits


def test_the_card_says_the_stage_2_rows_were_measured_before_the_graft(world):
    _, pkg = finished(world)
    flat = " ".join((pkg / "README.md").read_text().split())
    assert "stage 2 and stage 4 numbers" in flat and "without the graft" in flat


def test_a_package_staged_from_a_graft_in_stage_2_does_not_say_that(world):
    stage_2_with_the_real_graft(world)
    _, pkg = finished(world)
    assert "without the graft" not in " ".join((pkg / "README.md").read_text().split())


def test_the_gate_passes_a_finished_grafted_package(world):
    stage_dir, _ = finished(world)
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_the_gate_fails_when_graft_json_is_gone(world):
    stage_dir, pkg = finished(world)
    (pkg / "graft.json").unlink()
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any("graft.json is missing" in r for r in reasons), reasons


def test_the_gate_fails_a_package_whose_prepare_script_does_not_read_graft_json(world):
    stage_dir, pkg = finished(world)
    script = pkg / "prepare_model_dir.py"
    script.write_text(script.read_text().replace("graft.json", "nothing.json"))
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any("prepare_model_dir.py does not read graft.json" in r for r in reasons), reasons


def test_the_gate_fails_a_graft_package_whose_drafter_was_switched_off(world):
    stage_dir, pkg = finished(world)
    run_sh = pkg / "run.sh"
    run_sh.write_text(run_sh.read_text() + '\nexport QWEN36_DRAFTER=""\n')
    m = json.loads((pkg / "tt_kernel_manifest.json").read_text())
    m["env"]["QWEN36_DRAFTER"] = ""
    (pkg / "tt_kernel_manifest.json").write_text(json.dumps(m))
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any("drafter" in r and "graft" in r for r in reasons), reasons


@pytest.mark.parametrize("key, value, reason", [
    ("repo", "someone/else", "repo"), ("revision", "abc", "40-character commit"), ("revision", "f" * 40, "stage 2's"),
    ("sha256", "xyz", "sha256"), ("tensors", [], "tensors"), ("shard", "other.safetensors", "shard"), ("file", "", "file")])
def test_the_gate_fails_a_graft_json_that_does_not_describe_the_graft(world, key, value, reason):
    stage_dir, pkg = finished(world)
    graft = json.loads((pkg / "graft.json").read_text())
    graft[key] = value
    (pkg / "graft.json").write_text(json.dumps(graft))
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any("graft.json" in r and reason in r for r in reasons), reasons


@pytest.mark.parametrize("phrase", [f"mtp.* tensors of {BASE}", "added at each start", "`graft.json`"])
def test_the_gate_fails_a_card_that_does_not_mention_the_graft(world, phrase):
    stage_dir, pkg = finished(world)
    card = pkg / "README.md"
    card.write_text(card.read_text().replace(phrase, "present"))
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any("card" in r and "graft" in r and phrase in r for r in reasons), reasons


def test_a_drafter_on_package_without_graft_json_is_refused_by_the_gate(world):
    # Stage 2 served with the drafter on, so only the rule about a model without mtp.* tensors is left to fire.
    stage_dir, pkg = finished(world)
    (pkg / "graft.json").unlink()
    package = json.loads((stage_dir / "package.json").read_text())
    package.pop("mtp_graft")
    (stage_dir / "package.json").write_text(json.dumps(package))
    (world["run"] / "stages/2/run.sh").write_text((world["src"] / "run.sh").read_text())
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any("no mtp.* tensors" in r and "graft.json" in r for r in reasons), reasons


@pytest.mark.parametrize("delta, expected", [
    ({"mtp": {"new_tensors": 0}}, False), ({"mtp": {"new_tensors": 3}}, True), ({"mtp": {"new_tensors": True}}, None),
    ({"mtp": {"new_tensors": "3"}}, None), ({"mtp": {}}, None), ({"mtp": 3}, None), ({}, None), (None, None)])
def test_stage_0s_mtp_finding_is_read_strictly(delta, expected):
    from orchard.stages import delta_model_has_mtp
    assert delta_model_has_mtp(delta) is expected


def test_a_model_that_ships_its_own_mtp_tensors_is_packaged_with_the_drafter_on_and_no_graft(world):
    run = world["run"]
    delta = json.loads((run / "stages/0/delta.json").read_text())
    write(run / "stages/0/delta.json", {**delta, "mtp": {"nearest_tensors": 3, "new_tensors": 3}})
    (run / "stages/2/run.sh").write_text((world["src"] / "run.sh").read_text())
    stage_dir, package = stage(world)
    pkg = run / package["profiles"][0]["dir"]
    fake_boot_result(stage_dir)
    finish(run, stage_dir)
    assert not (pkg / "graft.json").exists() and "mtp_graft" not in package
    assert json.loads((pkg / "tt_kernel_manifest.json").read_text())["env"]["QWEN36_DRAFTER"] == "dflash2"
    g = gate_package(stage_dir, run)
    assert g.ok, g.reasons


# ---- the package's prepare_model_dir.py -----------------------------------------------------------------

@pytest.fixture
def installed(world, tmp_path):
    """A copy of the staged package where prepare_model_dir.py can run, and a hub holding both models."""
    _, package = stage(world, graft_mtp=True)
    pkg = tmp_path / "pkg"
    shutil.copytree(world["run"] / package["profiles"][0]["dir"], pkg, symlinks=True)
    hub = tmp_path / "hub"
    hub.mkdir()
    for root in (world["hf_run"], world["hf_op"]):
        for d in (root / "hub").iterdir():
            if d.name.startswith("models--") and not (hub / d.name).exists():
                (hub / d.name).symlink_to(d)
    return {"pkg": pkg, "hub": hub, **world}


def prepare(inst, *, offline=True, extra_env=None, pythonpath=None):
    env = dict(os.environ, HF_HUB_CACHE=str(inst["hub"]))
    env.pop("HF_HOME", None)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, str(inst["pkg"] / "prepare_model_dir.py")], capture_output=True,
                          text=True, timeout=60, env=env)


def test_the_package_builds_the_same_shard_prepare_swap_does(installed, tmp_path):
    r = prepare(installed)
    assert r.returncode == 0, r.stdout + r.stderr
    shard = installed["pkg"] / "model-dir" / SHARD
    assert shard.is_file() and not shard.is_symlink()
    graft = json.loads((installed["pkg"] / "graft.json").read_text())
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == graft["sha256"]
    src = installed["base"] / SHARD_FILE
    for name in MTP:
        assert tensor_bytes(shard, name) == tensor_bytes(src, name)
    assert {k for k in read_header(shard) if k != "__metadata__"} == set(MTP)


def test_the_package_gives_the_model_dir_its_own_index_that_lists_the_grafted_tensors(installed):
    # The model ships an index; the package links it, and the graft must not write through that link.
    snap = installed["snapshot"]
    blob = Path(os.path.realpath(snap / "model-00001-of-00001.safetensors")).parent / "blob-index"
    blob.write_text(json.dumps({"metadata": {"total_size": 10}, "weight_map": {"model.layers.0.weight": SHARD_FILE}}))
    (snap / "model.safetensors.index.json").symlink_to(os.path.relpath(blob, snap))
    before = blob.read_bytes()
    r = prepare(installed)
    assert r.returncode == 0, r.stdout + r.stderr
    index = installed["pkg"] / "model-dir" / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    assert not index.is_symlink() and blob.read_bytes() == before
    assert all(data["weight_map"][k] == SHARD for k in MTP) and data["weight_map"]["model.layers.0.weight"] == SHARD_FILE
    assert data["metadata"]["total_size"] > 10          # the grafted bytes are counted


def test_a_second_start_builds_the_model_dir_again(installed):
    assert prepare(installed).returncode == 0
    r = prepare(installed)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (installed["pkg"] / "model-dir" / SHARD).is_file()


def test_a_shard_that_is_not_the_one_that_was_tested_stops_the_start(installed):
    graft = json.loads((installed["pkg"] / "graft.json").read_text())
    graft["sha256"] = "0" * 64
    (installed["pkg"] / "graft.json").write_text(json.dumps(graft))
    r = prepare(installed)
    assert r.returncode == 2 and "sha256" in r.stderr
    assert not (installed["pkg"] / "model-dir").exists()          # a model-dir without its head would not start


def test_offline_and_without_the_nearest_models_shard_the_start_stops_and_says_what_to_download(installed):
    Path(os.path.realpath(installed["base"] / SHARD_FILE)).unlink()
    r = prepare(installed)
    assert r.returncode == 2
    assert BASE in r.stderr and SHARD_FILE in r.stderr and "HF_HUB_OFFLINE" in r.stderr and BASE_REV in r.stderr


def test_a_graft_json_that_is_not_a_graft_record_stops_the_start(installed):
    (installed["pkg"] / "graft.json").write_text(json.dumps({"repo": "x"}))
    r = prepare(installed)
    assert r.returncode == 2 and "graft record" in r.stderr
    (installed["pkg"] / "graft.json").write_text("{")
    assert prepare(installed).returncode == 2


@pytest.mark.parametrize("key, value", [("revision", "abc"), ("tensors", []), ("repo", "noslash"), ("file", ""), ("shard", ""),
                                        ("sha256", 5)])
def test_each_field_of_graft_json_is_checked_before_the_start_goes_on(installed, key, value):
    graft = json.loads((installed["pkg"] / "graft.json").read_text())
    graft[key] = value
    (installed["pkg"] / "graft.json").write_text(json.dumps(graft))
    r = prepare(installed)
    assert r.returncode == 2 and "graft record" in r.stderr, r.stderr


def test_a_shard_cut_short_stops_the_start(installed):
    blob = Path(os.path.realpath(installed["base"] / SHARD_FILE))
    blob.write_bytes(blob.read_bytes()[:-5])
    r = prepare(installed)
    assert r.returncode == 2 and "does not match the bytes" in r.stderr


def test_a_missing_tensor_stops_the_start(installed):
    real_file(installed["base"], SHARD_FILE, dict(BACKBONE))
    r = prepare(installed)
    assert r.returncode == 2 and "mtp.fc.weight" in r.stderr


def test_a_tensor_that_is_not_bf16_stops_the_start(installed):
    real_file(installed["base"], SHARD_FILE, {**BACKBONE, **{**MTP, "mtp.norm.weight": ("F32", [4])}})
    r = prepare(installed)
    assert r.returncode == 2 and "mtp.norm.weight is F32; the graft copies BF16 tensors only" in r.stderr


def test_online_the_shard_is_downloaded_alone_and_the_rest_of_the_model_is_not(installed, tmp_path):
    real = Path(os.path.realpath(installed["base"] / SHARD_FILE))
    keep = tmp_path / "keep.safetensors"
    shutil.copy(real, keep)
    (installed["hub"] / "models--Qwen--Qwen3.8-27B").unlink()          # the hub link, not the cache behind it
    fake = tmp_path / "fake"
    (fake / "huggingface_hub").mkdir(parents=True)
    log = tmp_path / "hf-calls.json"
    (fake / "huggingface_hub" / "__init__.py").write_text(
        "import json, os\n"
        f"def hf_hub_download(**kw):\n"
        f"    open({str(log)!r}, 'a').write(json.dumps(kw) + '\\n')\n"
        f"    return {str(keep)!r}\n"
        "def snapshot_download(**kw):\n"
        "    raise SystemExit('snapshot_download must not be used for the nearest model')\n")
    r = prepare(installed, offline=False, pythonpath=fake)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [{"repo_id": BASE, "filename": SHARD_FILE, "revision": BASE_REV}]
    assert (installed["pkg"] / "model-dir" / SHARD).is_file()


def test_the_package_without_graft_json_builds_what_it_always_built(world, tmp_path):
    _, package = stage(world)
    pkg = tmp_path / "plain"
    shutil.copytree(world["run"] / package["profiles"][0]["dir"], pkg, symlinks=True)
    hub = tmp_path / "hub2"
    hub.mkdir()
    for root in (world["hf_run"], world["hf_op"]):
        for d in (root / "hub").iterdir():
            (hub / d.name).symlink_to(d)
    r = subprocess.run([sys.executable, str(pkg / "prepare_model_dir.py")], capture_output=True, text=True,
                       env=dict(os.environ, HF_HUB_CACHE=str(hub), HF_HUB_OFFLINE="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (pkg / "model-dir" / SHARD).exists()
    assert "mtp.*" not in r.stdout


def test_stage_2_and_the_package_build_byte_identical_shards(installed, tmp_path):
    stage = tmp_path / "stage2"
    stage.mkdir()
    shutil.copy(SWAP_TEMPLATE, stage)
    (stage / "swap_config.json").write_text(json.dumps({
        "nearest_model_id": BASE, "base_snapshot": str(installed["base"]), "new_snapshot": str(installed["snapshot"]),
        "package": "episod/some-container", "graft_mtp": True}))
    r = subprocess.run([sys.executable, str(stage / "prepare_swap.py")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert prepare(installed).returncode == 0
    assert (stage / "model-dir" / SHARD).read_bytes() == (installed["pkg"] / "model-dir" / SHARD).read_bytes()


def test_the_readme_documents_the_flag():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "- `--graft-mtp`:" in readme and "graft.json" in readme


def put_models_own_file_named_like_the_shard(inst, tmp_path):
    """The model's snapshot ships a link to a blob named like the graft shard (backbone tensors, no mtp.*)."""
    mine = tmp_path / "ships.safetensors"
    write_safetensors(mine, dict(BACKBONE))
    return put_link(inst["snapshot"], SHARD, mine.read_bytes())


def test_the_package_never_writes_through_a_models_own_file_with_the_shards_name(installed, tmp_path):
    # The launcher links every *.safetensors of the model into model-dir; opening the shard's destination for
    # writing would follow that link and truncate the model's cached blob.
    blob = put_models_own_file_named_like_the_shard(installed, tmp_path)
    before = blob.read_bytes()
    r = prepare(installed)
    assert r.returncode == 2, r.stdout + r.stderr
    assert SHARD in r.stderr and "name" in r.stderr
    assert blob.read_bytes() == before
    assert not (installed["pkg"] / "model-dir").exists()


def test_the_package_refuses_a_shard_destination_that_already_exists(installed):
    # A file in base_config/ with the shard's name is copied into model-dir before the graft is written.
    (installed["pkg"] / "base_config" / SHARD).write_bytes(b"in the way")
    r = prepare(installed)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "already exists" in r.stderr
