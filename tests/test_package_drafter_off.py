"""Stage 7 for a model without mtp.* tensors (Clef): the package serves the way stage 2 and 4 tested it.

Stage 2's run.sh copy has the speculative drafter off and no on-device sampling. The staged package
must match it, name no drafter repo, and say on its card that the sidecar head is not served."""
import json
import os
import sys
from pathlib import Path

import pytest

from orchard.package import (PackageError, drop_device_sampling, finish, read_run, serving_env,
                             serving_problems, stage_all, thin_argv)
from orchard.stages import gate_package
from package_fakes import (DRAFTER, SIDECAR, SOURCE_EXTRA, fake_bin, fake_boot_result, make_clef_like,
                           make_run, make_source, write)

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
ON_DEVICE = '--additional-config \'{"tt": {"l1_small_size": 24576, "sample_on_device_mode": "decode_only"}}\''


def test_device_sampling_is_dropped_from_the_one_additional_config():
    out = drop_device_sampling(f"{ON_DEVICE} --max-num-batched-tokens 65536")
    assert json.loads(out.split("'")[1]) == {"tt": {"l1_small_size": 24576}}
    assert out.endswith("--max-num-batched-tokens 65536")


def test_extra_args_without_the_key_or_the_option_are_unchanged():
    assert drop_device_sampling(SOURCE_EXTRA) == SOURCE_EXTRA
    assert drop_device_sampling("--no-async-scheduling") == "--no-async-scheduling"


def test_two_additional_configs_are_refused():
    with pytest.raises(PackageError, match="additional-config"):
        drop_device_sampling(f"{ON_DEVICE} {ON_DEVICE}")


def test_the_drafter_environment_is_cleared_only_when_it_is_off():
    env = {"ARCH_NAME": "blackhole", "QWEN36_DRAFTER": "dflash2", "DFLASH_WEIGHTS": f"{DRAFTER}@x",
           "QWEN36_DFLASH_TP": "1", "QWEN36_DFLASH_BLOCK": "8"}
    assert serving_env(env, drafter_off=False) == env
    assert serving_env(env, drafter_off=True) == {"ARCH_NAME": "blackhole", "QWEN36_DRAFTER": ""}


@pytest.fixture
def world(tmp_path, monkeypatch, stub_tools):
    models = tmp_path / "models"
    src = make_source(models, env={"ARCH_NAME": "blackhole", "QWEN36_DRAFTER": "dflash2",
                                   "DFLASH_WEIGHTS": f"{DRAFTER}@" + "d" * 40, "QWEN36_DFLASH_TP": "1"})
    # The source's run.sh carries on-device sampling, as the 27B bundle does.
    (src / "run.sh").write_text((src / "run.sh").read_text().replace(
        '{\\"tt\\": {\\"l1_small_size\\": 24576}}', "X").replace(
        "{\"tt\": {\"l1_small_size\": 24576}}", "{\"tt\": {\"l1_small_size\": 24576, "
                                               "\"sample_on_device_mode\": \"decode_only\"}}"))
    r = make_run(tmp_path, src)
    make_clef_like(r["run"], src)
    # Stage 2 served without on-device sampling.
    s2 = r["run"] / "stages/2/run.sh"
    s2.write_text(s2.read_text().replace(', "sample_on_device_mode": "decode_only"', ""))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    return {**r, "models": models, "src": src, "env": env, "tmp": tmp_path}


def staged(w):
    stage = w["run"] / "stages/7"
    stage.mkdir(parents=True)
    stage_all(w["run"], stage, namespace="episod", models_root=w["models"], env=w["env"])
    fake_boot_result(stage)
    finish(w["run"], stage)
    return stage, stage / "package/hemmingway-1-p300"


def test_the_run_facts_say_what_stage_2_served(world):
    f = read_run(world["run"])
    assert (f.drafter_off, f.host_sampling) == (True, True)
    assert [s["file"] for s in f.sidecars] == [SIDECAR["file"]]


def test_a_run_whose_stage_2_kept_the_drafter_keeps_it(tmp_path, monkeypatch, stub_tools):
    src = make_source(tmp_path / "models")
    r = make_run(tmp_path, src)
    assert (read_run(r["run"]).drafter_off, read_run(r["run"]).host_sampling) == (False, False)


def test_thin_argv_passes_the_cleared_environment(world):
    f = read_run(world["run"])
    argv = thin_argv(f.source, model_id=f.model_id, revision=f.revision, name="n",
                     out=world["tmp"] / "o", drafter_off=True)
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]
    assert "QWEN36_DRAFTER=" in envs
    assert not [e for e in envs if e.startswith(("DFLASH_WEIGHTS", "QWEN36_DFLASH"))]


def test_the_staged_package_matches_what_stage_2_served(world):
    stage, pkg = staged(world)
    run_sh = (pkg / "run.sh").read_text()
    m = json.loads((pkg / "tt_kernel_manifest.json").read_text())
    assert "sample_on_device_mode" not in run_sh and "DFLASH" not in run_sh
    assert 'QWEN36_DRAFTER=""' in run_sh or "QWEN36_DRAFTER=\n" in run_sh
    assert m["env"]["QWEN36_DRAFTER"] == "" and "DFLASH_WEIGHTS" not in m["env"]
    assert serving_problems(run_sh, m["env"], drafter_off=True, host_sampling=True) == []
    g = gate_package(stage, world["run"])
    assert g.ok, g.reasons


def test_the_card_names_no_drafter_and_says_the_head_is_not_served(world):
    _, pkg = staged(world)
    card = (pkg / "README.md").read_text()
    assert DRAFTER not in card and "Speculative decoding uses" not in card
    assert "speculative decoding is off" in card.lower()
    assert "joint_head.safetensors" in card and "not served" in card.lower()
    assert "host" in card.lower()


def test_the_gate_fails_a_package_that_kept_the_drafter(world):
    stage, pkg = staged(world)
    run_sh = pkg / "run.sh"
    run_sh.write_text(run_sh.read_text() + '\nexport QWEN36_DRAFTER="dflash2"\n')
    reasons = gate_package(stage, world["run"]).reasons
    assert any("drafter" in r.lower() for r in reasons), reasons


def test_the_gate_fails_a_package_that_kept_device_sampling(world):
    stage, pkg = staged(world)
    run_sh = pkg / "run.sh"
    run_sh.write_text(run_sh.read_text().replace('"l1_small_size": 24576',
                                                 '"l1_small_size": 24576, "sample_on_device_mode": "decode_only"'))
    reasons = gate_package(stage, world["run"]).reasons
    assert any("sample_on_device_mode" in r for r in reasons), reasons


def test_the_gate_fails_a_card_that_leaves_out_the_sidecar(world):
    stage, pkg = staged(world)
    card = pkg / "README.md"
    card.write_text(card.read_text().replace("joint_head.safetensors", "the head"))
    reasons = gate_package(stage, world["run"]).reasons
    assert any("sidecar" in r.lower() for r in reasons), reasons


def test_a_manifest_that_turns_the_drafter_on_is_a_problem():
    run_sh = 'export QWEN36_DRAFTER=""\n'
    assert serving_problems(run_sh, {"QWEN36_DRAFTER": "dflash2"}, drafter_off=True, host_sampling=False)
    assert serving_problems(run_sh, {"QWEN36_DRAFTER": ""}, drafter_off=True, host_sampling=False) == []
    assert serving_problems(run_sh, {"QWEN36_DRAFTER": "dflash2"}, drafter_off=False,
                            host_sampling=False) == []


def test_the_last_drafter_line_of_stage_2s_script_decides(tmp_path):
    from orchard.package import stage_2_serving
    cmd = "CMD=(x --additional-config '{\"tt\": {}}')\n"
    for text, want in ((f'export QWEN36_DRAFTER="dflash2"\nexport QWEN36_DRAFTER=""\n{cmd}', True),
                       (f'export QWEN36_DRAFTER=""\nexport QWEN36_DRAFTER="dflash2"\n{cmd}', False)):
        write(tmp_path / "stages/2/run.sh", text)
        assert stage_2_serving(tmp_path) == (want, True)


def test_staging_refuses_a_package_that_still_serves_with_the_drafter(world, monkeypatch):
    import orchard.package as package
    monkeypatch.setattr(package, "serving_env", lambda env, drafter_off: dict(env))
    stage = world["run"] / "stages/7"
    stage.mkdir(parents=True)
    with pytest.raises(PackageError, match="drafter"):
        stage_all(world["run"], stage, namespace="episod", models_root=world["models"], env=world["env"])


# ---- the evidence the card cites ships with the package -----------------------------------------

def cited(card: str) -> list[str]:
    from orchard.package_card import card_evidence_paths
    return card_evidence_paths(card)


def test_every_file_a_measured_number_cites_ships_under_evidence(world):
    _, pkg = staged(world)
    paths = cited((pkg / "README.md").read_text())
    assert "stages/2/result.json" in paths and "stages/7/verify/evidence/verify.json" in paths
    for p in paths:
        assert (pkg / "evidence" / p).is_file(), p
    assert not (world["models"].parent / "x").exists()


def test_the_copied_evidence_has_the_run_and_home_paths_replaced(world, monkeypatch):
    run = str(world["run"])
    _, pkg = staged(world)
    text = (pkg / "evidence/stages/2/evidence/swap-check.json").read_text()
    assert run not in text and "<RUN_DIR>" in text


def test_hostname_and_home_are_replaced_in_the_copy(world, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "gethostname", lambda: "tt-secret-box")
    log = world["run"] / "stages/2/evidence/server.log"
    log.write_text(f"ready on tt-secret-box\nweights in /home/someone/.cache/x and {os.path.expanduser('~')}/y\n")
    _, pkg = staged(world)
    text = (pkg / "evidence/stages/2/evidence/server.log").read_text()
    assert "tt-secret-box" not in text and "/home/someone" not in text and os.path.expanduser("~") not in text
    assert "<HOST>" in text and "<HOME>" in text


def test_a_token_in_evidence_fails_the_gate(world):
    (world["run"] / "stages/2/evidence/server.log").write_text("hf_" + "a" * 36 + "\n")
    stage = world["run"] / "stages/7"
    stage.mkdir(parents=True)
    stage_all(world["run"], stage, namespace="episod", models_root=world["models"], env=world["env"])
    fake_boot_result(stage)
    finish(world["run"], stage)
    reasons = gate_package(stage, world["run"]).reasons
    assert any("token" in r and "evidence/stages/2/evidence/server.log" in r for r in reasons), reasons


def test_evidence_that_is_too_large_is_refused(world, monkeypatch):
    import orchard.package as package
    monkeypatch.setattr(package, "EVIDENCE_MAX_BYTES", 10)
    stage = world["run"] / "stages/7"
    stage.mkdir(parents=True)
    stage_all(world["run"], stage, namespace="episod", models_root=world["models"], env=world["env"])
    fake_boot_result(stage)
    with pytest.raises(PackageError, match="too large"):
        finish(world["run"], stage)


def test_the_gate_fails_when_a_cited_evidence_file_is_missing(world):
    stage, pkg = staged(world)
    (pkg / "evidence/stages/2/result.json").unlink()
    reasons = gate_package(stage, world["run"]).reasons
    assert any("evidence/stages/2/result.json" in r for r in reasons), reasons


def test_the_card_says_where_the_evidence_is(world):
    _, pkg = staged(world)
    card = (pkg / "README.md").read_text()
    assert "`evidence/`" in card and "relative to the run directory" not in card


def test_a_home_directory_outside_slash_home_is_replaced(world, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: "/srv/people/bob" if p == "~" else p)
    (world["run"] / "stages/2/evidence/server.log").write_text("cache in /srv/people/bob/.cache\n")
    _, pkg = staged(world)
    text = (pkg / "evidence/stages/2/evidence/server.log").read_text()
    assert "/srv/people/bob" not in text and "<HOME>/.cache" in text


def test_a_card_that_cites_a_file_outside_the_run_is_refused(world, tmp_path):
    from orchard.package import copy_evidence
    (world["tmp"] / "outside.txt").write_text("x")
    card = ("## Expected performance\n\n| Number | Value | Label | Evidence |\n|---|---|---|---|\n"
            "| n | 1 fraction | measured | `../outside.txt` |\n\n## Limitations\n")
    out = world["tmp"] / "pkg"
    out.mkdir()
    with pytest.raises(PackageError, match="not a file inside the run"):
        copy_evidence(read_run(world["run"]), out, card)


def test_an_installed_bundle_path_loses_its_namespace(world):
    (world["run"] / "stages/2/evidence/server.log").write_text(
        f"{os.path.expanduser('~')}/.cache/tt-model/models/episod/qwen3.8-27b-dflash2-p300/.python/x.py:279\n")
    stage, pkg = staged(world)
    text = (pkg / "evidence/stages/2/evidence/server.log").read_text()
    assert "episod" not in text and "<TT_MODEL_MODELS>/qwen3.8-27b-dflash2-p300/.python/x.py" in text
    assert gate_package(stage, world["run"]).ok
