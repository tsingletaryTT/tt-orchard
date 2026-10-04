"""Stage 7, part 1: read the run, find the source bundles, run package-thin with --out only (plan 5)."""
import json
import os
import re
import socket
import subprocess
import time

import pytest

from orchard.package import (PackageError, Profile, assert_no_publish, bundle_name, find_sources,
                             load_source, plan_profiles, read_run, run_logged, stage_profile,
                             thin_argv, weights_wiring_problems)
from orchard.scrub import scrub_package
from package_fakes import (BASE, NEW, NEW_REV, SOURCE_EXTRA, calls, fake_bin, make_run, make_source,
                           write)

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def world(tmp_path, monkeypatch, stub_tools):
    """Installed bundles (2-chip and 1-chip of the base model, one of another model), a finished
    run, and the fake tt-model first on PATH, ahead of the stubs."""
    models = tmp_path / "models"
    src2 = make_source(models)
    src1 = make_source(models, name="qwen3.8-27b-dflash2-p150", chips=1, mesh="P150")
    make_source(models, name="other-model-p300", weights="Other/Model")
    r = make_run(tmp_path, src2)
    fb, log = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    return {**r, "models": models, "src2": src2, "src1": src1, "log": log, "stubs": stub_tools,
            "tmp": tmp_path}


def test_the_run_facts_come_from_stages_0_2_and_4(world):
    f = read_run(world["run"])
    assert (f.model_id, f.revision, f.nearest_model) == (NEW, NEW_REV, BASE)
    assert f.license_id == "cc-by-nc-4.0"
    assert f.passing_chips == frozenset({1, 2, 4})
    assert f.source.path == world["src2"] and f.source.chips == 2
    assert [p.name for p in f.base_config] == ["config.json", "preprocessor_config.json"]
    assert f.hf_home == world["hf_op"]


@pytest.mark.parametrize("breakage, words", [
    (lambda w: write(w["run"] / "stages/0/delta.json", {"model": NEW, "nearest_model": BASE,
                                                         "path": "full-port"}), "weights-only"),
    (lambda w: write(w["run"] / "stages/2/result.json", {**json.loads(
        (w["run"] / "stages/2/result.json").read_text()), "top1_agreement": 0.5}), "top1_agreement"),
    (lambda w: (w["snapshot"] / "README.md").unlink(), "license"),
    (lambda w: (w["run"] / "stages/2/model-dir/config.json").unlink(), "config.json"),
    (lambda w: write(w["src2"] / "tt_kernel_manifest.json", {**json.loads(
        (w["src2"] / "tt_kernel_manifest.json").read_text()),
        "weights": {"repo_id": "Other/Model", "revision": "0" * 40}}), "Other/Model"),
])
def test_a_run_stage_7_cannot_package_is_refused_with_the_reason(world, breakage, words):
    breakage(world)
    with pytest.raises(PackageError, match=words):
        read_run(world["run"])


def swap_check(w, **changes):
    path = w["run"] / "stages/2/evidence/swap-check.json"
    write(path, {**json.loads(path.read_text()), **changes})


def relink_weights(w, rev):
    """Point model-dir's weight link at another revision's snapshot of the new model."""
    other = w["snapshot"].parent / rev
    other.mkdir()
    (other / "model-00001-of-00001.safetensors").write_text("other weights")
    link = w["run"] / "stages/2/model-dir/model-00001-of-00001.safetensors"
    link.unlink()
    link.symlink_to(other / "model-00001-of-00001.safetensors")


def test_a_short_label_passes_when_the_served_links_match_stage_0(world):
    # Run 3's live case: stage 0 names '<repo>@<rev>', stage 2's label is '<repo>' alone.
    assert json.loads((world["run"] / "stages/0/delta.json").read_text())["model"] == f"{NEW}@{NEW_REV}"
    f = read_run(world["run"])
    assert (f.model_id, f.revision) == (NEW, NEW_REV)


@pytest.mark.parametrize("breakage, words", [
    (lambda w: swap_check(w, new_model_id="Someone/Else"), "Someone/Else"),
    (lambda w: swap_check(w, new_model_id=f"{NEW}@{'0' * 40}"), "0" * 40),
    (lambda w: relink_weights(w, "e" * 40), "e" * 40),
    (lambda w: swap_check(w, model_dir=None), "model_dir"),
    (lambda w: write(w["run"] / "stages/0/delta.json", {
        **json.loads((w["run"] / "stages/0/delta.json").read_text()), "model": NEW}), "names no revision"),
])
def test_stage_7_refuses_weights_stage_0_does_not_name(world, breakage, words):
    breakage(world)
    with pytest.raises(PackageError, match=re.escape(words)):
        read_run(world["run"])


def test_only_installed_v6_bundles_of_the_nearest_model_are_sources(world):
    container = world["models"] / "mando" / "x-p300x2"
    write(container / "tt_kernel_manifest.json", {"schema_version": "5.1", "name": "x-p300x2"})
    found = find_sources(world["models"], entry_cls=load_source(world["src2"]).entry_cls,
                         nearest_model=BASE)
    assert [(s.name, s.chips) for s in found] == [("qwen3.8-27b-dflash2-p150", 1),
                                                  ("qwen3.8-27b-dflash2-p300", 2)]


def test_profiles_are_the_stage_2_bundle_plus_others_stage_4_passed(world):
    f = read_run(world["run"])
    profiles, skipped = plan_profiles(f, [load_source(world["src1"]), f.source])
    assert [(p.chips, p.required, p.source.name) for p in profiles] == [
        (1, False, "qwen3.8-27b-dflash2-p150"), (2, True, "qwen3.8-27b-dflash2-p300")]
    assert skipped == [{"chips": 4, "reason": f"no v6 bundle of {BASE} for 4 chips is installed"}]


def test_the_required_profile_needs_a_passing_stage_4_configuration(world):
    write(world["run"] / "stages/4/result.json", {"configs": [{"chips": 1, "pass": True}]})
    with pytest.raises(PackageError, match="no passing 2-chip"):
        plan_profiles(read_run(world["run"]), [])


def test_the_package_thin_call_names_the_new_weights_and_stages_only(world, tmp_path):
    src = load_source(world["src2"])
    argv = thin_argv(src, model_id=NEW, revision=NEW_REV, name="hemmingway-1-p300", out=tmp_path / "o")
    assert argv[:2] == ["tt-model", "package-thin"] and argv[-2:] == ["--out", str(tmp_path / "o")]
    assert argv[argv.index("--weights") + 1] == NEW and argv[argv.index("--weights-revision") + 1] == NEW_REV
    assert "--env" in argv and f"DFLASH_WEIGHTS={src.manifest['env']['DFLASH_WEIGHTS']}" in argv
    assert_no_publish(argv)
    assert bundle_name(NEW, src.name) == "hemmingway-1-p300"


@pytest.mark.parametrize("change, words", [
    (lambda a: a[:2] + ["episod/hemmingway-1-p300"] + a[2:], "positional"),
    (lambda a: a + ["--public"], "--public"),
    (lambda a: a + ["--publish"], "--publish"),
    (lambda a: ["tt-model", "push"] + a[2:], "only `tt-model package-thin`"),
    (lambda a: a[:-2], "--out"),
    (lambda a: a + ["--name"], "--name"),
])
def test_any_call_that_could_upload_is_refused(world, tmp_path, change, words):
    argv = thin_argv(load_source(world["src2"]), model_id=NEW, revision=NEW_REV, name="n",
                     out=tmp_path / "o")
    with pytest.raises(PackageError, match=words):
        assert_no_publish(change(argv))


def staged(world, profile_source=None, required=True):
    f = read_run(world["run"])
    src = profile_source or f.source
    out = world["run"] / "stages/7/package" / bundle_name(NEW, src.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    return f, stage_profile(Profile(src.chips, src, required), f, out), out


def test_a_staged_profile_is_wired_scrubbed_and_built_by_one_package_thin_call(world):
    f, rec, out = staged(world)
    assert rec == {"chips": 2, "name": "hemmingway-1-p300", "required": True,
                   "source": "qwen3.8-27b-dflash2-p300", "mesh": "P150x2",
                   "source_manifest_sha256": rec["source_manifest_sha256"]}
    text = (out / "run.sh").read_text()
    assert weights_wiring_problems(text, nearest_model=BASE) == []
    assert f' {SOURCE_EXTRA} "$@")' in text
    m = json.loads((out / "tt_kernel_manifest.json").read_text())
    assert m["producer"]["hostname"] == "redacted"
    assert m["weights"]["repo_id"] == NEW and m["weights"]["revision"] == NEW_REV
    assert sorted(p.name for p in (out / "base_config").iterdir()) == ["config.json",
                                                                       "preprocessor_config.json"]
    for name in ("run.sh", "install.sh", "prepare_model_dir.py"):
        assert os.access(out / name, os.X_OK), name
    assert scrub_package(out, hostname=socket.gethostname(), namespace="episod") == []
    # One tt-model call, package-thin with --out; no stub (hf, git, gh, docker, curl ...) ran.
    assert [c[:1] + c[-2:] for c in calls(world["log"])] == [["package-thin", "--out", str(out)]]
    assert world["stubs"].calls() == []


def test_an_optional_profile_is_staged_from_its_own_bundle(world):
    _, rec, out = staged(world, load_source(world["src1"]), required=False)
    assert (rec["chips"], rec["name"], rec["required"]) == (1, "hemmingway-1-p150", False)
    assert json.loads((out / "tt_kernel_manifest.json").read_text())["mesh"]["topology"] == "P150"


def test_a_wheel_that_differs_from_the_source_bundle_is_refused(world, monkeypatch, tmp_path):
    other = tmp_path / "other-ttnn"
    other.mkdir()
    (other / "ttnn-0.79.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"PK a different build")
    src = load_source(world["src2"])
    real = thin_argv

    def swapped(*a, **k):
        argv = real(*a, **k)
        i = argv.index("--models-wheel")
        argv[i + 1] = str(other / "ttnn-0.79.0-cp312-cp312-linux_x86_64.whl")
        return argv
    monkeypatch.setattr("orchard.package.thin_argv", swapped)
    with pytest.raises(PackageError, match="not byte-identical"):
        staged(world, src)


def test_a_failed_package_thin_is_reported_with_its_output(world, monkeypatch):
    monkeypatch.setenv("FAKE_TT_MODEL_FAIL", "1")
    with pytest.raises(PackageError, match="exited 3(.|\n)*SFPI"):
        staged(world)


def test_run_logged_kills_the_whole_session_on_a_timeout(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "slow.sh"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nwait\n")
    script.chmod(0o755)
    t0 = time.monotonic()
    assert run_logged([script], log=tmp_path / "slow.log", timeout=1) is None
    assert time.monotonic() - t0 < 10
    child = int(pid_file.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_run_logged_kills_the_session_when_interrupted(tmp_path, monkeypatch):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "slow.sh"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nwait\n")
    script.chmod(0o755)
    real_wait = subprocess.Popen.wait

    def interrupted(self, timeout=None):
        if timeout is not None:
            while not pid_file.exists():
                time.sleep(0.05)
            raise KeyboardInterrupt
        return real_wait(self)
    monkeypatch.setattr(subprocess.Popen, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_logged([script], log=tmp_path / "slow.log", timeout=30)
    monkeypatch.undo()
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_the_packaging_module_has_no_upload_code():
    # The only external program package.py starts on its own is the argv that assert_no_publish
    # accepts; it imports no hub client and names no upload command outside the publish text.
    import orchard.package as pkg
    src = open(pkg.__file__, encoding="utf-8").read()
    assert not re.search(r"^\s*(?:import|from)\s+(?:huggingface_hub|tt_kernel)\b", src, re.M)
    assert "subprocess.run(" not in src and "os.system" not in src
