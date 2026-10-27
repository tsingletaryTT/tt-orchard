# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/skills/weights-swap-templates/sampling_trial.py: the script the supervisor runs under a lease to serve the drafter-off variant and run the stage 2
check on it. prepare_swap.py and serve_and_compare.py are stubs in a fake checkout, so the tests cover the script's own work: the order, the config it
builds, the files it copies and what it writes when a step fails. No server starts and no chip is used."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates" / "sampling_trial.py"
RESULT = "trial-result.json"

STUB_PREPARE = '''import json, os, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
cfg = json.loads((here / "swap_config.json").read_text())
behave = json.loads(pathlib.Path(cfg["behavior"]).read_text())
pathlib.Path(cfg["calls_log"]).open("a").write(json.dumps({"tool": "prepare", "cwd": str(here), "pgrp": os.getpgrp(), "config": cfg}) + "\\n")
if behave.get("prepare_exit"):
    print("prepare_swap: swap_config.json has drafter off and graft_mtp true"); sys.exit(behave["prepare_exit"])
(here / "model-dir").mkdir()
(here / "run.sh").write_text("#!/bin/bash\\n")
(here / "model-dir-sources.json").write_text("{}")
'''

STUB_SERVE = '''import json, os, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
cfg = json.loads((here / "swap_config.json").read_text())
behave = json.loads(pathlib.Path(cfg["behavior"]).read_text())
pathlib.Path(cfg["calls_log"]).open("a").write(json.dumps({"tool": "serve", "cwd": str(here), "pgrp": os.getpgrp(),
                                                            "suite_beside": (here / "prompt_suite.json").exists()}) + "\\n")
if behave.get("serve_exit"):
    print("serve_and_compare: the server exited with code 1 before it was healthy"); sys.exit(behave["serve_exit"])
(here / "evidence").mkdir()
draft = {"serves": True, "server_ready_s": 124.0, "coherent": True, "top1_agreement": 0.96875, "n_tokens": 32}
(here / "evidence" / "swap-check.json").write_text(json.dumps({"result_draft": draft, "serving": {"drafter": "off"}}))
(here / "evidence" / "server.log").write_text("log")
'''


@pytest.fixture
def rig(tmp_path):
    orchard = tmp_path / "checkout"
    t = orchard / "orchard" / "skills" / "weights-swap-templates"
    t.mkdir(parents=True)
    (t / "prepare_swap.py").write_text(STUB_PREPARE)
    (t / "serve_and_compare.py").write_text(STUB_SERVE)
    (t / "prompt_suite.json").write_text("{}")
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    stage.mkdir(parents=True)
    log, behavior = orchard / "calls.log", orchard / "behavior.json"
    cfg = {"run_dir": str(run), "new_model_id": "Org/Model@" + "b" * 40, "hf_home": "/hf", "port": 8100,
           "tt_cache": str(tmp_path / "cache" / "tt_cache"), "nearest_model_id": "Qwen/Qwen3.8-27B", "prompt_suite": True,
           "sampling_probe": False, "calls_log": str(log), "behavior": str(behavior)}
    (stage / "swap_config.json").write_text(json.dumps(cfg))
    trial = stage / "sampling-trial"
    trial.mkdir()
    (trial / "sampling_trial.py").write_text(SCRIPT.read_text())
    (trial / "trial_config.json").write_text(json.dumps({"stage_dir": str(stage), "orchard_dir": str(orchard)}))
    r = type("Rig", (), {})()
    r.tmp, r.run_dir, r.stage, r.trial, r.orchard, r.cfg, r.log, r.behavior, r.templates = tmp_path, run, stage, trial, orchard, cfg, log, behavior, t
    r.set = lambda **b: behavior.write_text(json.dumps(b))
    r.set()
    r.run = lambda: subprocess.run([sys.executable, str(trial / "sampling_trial.py")], capture_output=True, text=True, timeout=60)
    r.calls = lambda: [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    r.result = lambda: json.loads((trial / RESULT).read_text())
    return r


def test_the_script_builds_the_variant_then_checks_it_and_measures_nothing_else(rig):
    p = rig.run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert [c["tool"] for c in rig.calls()] == ["prepare", "serve"]
    res = rig.result()
    assert [s["name"] for s in res["steps"]] == ["prepare", "serve"]
    v = res["variant"]
    assert v["served"] is True and v["coherent"] is True and v["top1_agreement"] == 0.96875 and v["n_tokens"] == 32
    assert v["server_ready_s"] == 124.0 and v["failure"] is None
    assert res["new_model_id"] == rig.cfg["new_model_id"] and res["label"] == "measured"


def test_the_variant_config_has_the_drafter_off_the_probe_on_and_its_own_cache(rig):
    rig.run()
    prep = next(c for c in rig.calls() if c["tool"] == "prepare")
    cfg = prep["config"]
    assert prep["cwd"] == str(rig.trial / "variant")
    assert cfg["drafter"] == "off" and cfg["sampling_probe"] is True
    assert cfg["tt_cache"] == rig.cfg["tt_cache"] + "-nodraft" and cfg["tt_cache"] != rig.cfg["tt_cache"]
    for key in ("new_model_id", "nearest_model_id", "hf_home", "port", "run_dir", "prompt_suite"):
        assert cfg[key] == rig.cfg[key], key
    assert (rig.trial / "variant" / "swap_config.json").is_file()


def test_a_stage_config_that_grafts_is_built_without_the_graft_because_prepare_swap_refuses_both(rig):
    (rig.stage / "swap_config.json").write_text(json.dumps({**rig.cfg, "graft_mtp": True}))
    rig.run()
    prep = next(c for c in rig.calls() if c["tool"] == "prepare")
    assert "graft_mtp" not in prep["config"] and prep["config"]["drafter"] == "off"


def test_the_stage_own_swap_config_is_left_alone(rig):
    before = (rig.stage / "swap_config.json").read_text()
    rig.run()
    assert (rig.stage / "swap_config.json").read_text() == before


def test_the_shipped_scripts_come_from_the_checkout_and_the_suite_only_when_the_config_asks(rig):
    rig.run()
    v = rig.trial / "variant"
    assert (v / "prepare_swap.py").read_text() == STUB_PREPARE and (v / "serve_and_compare.py").read_text() == STUB_SERVE
    assert (v / "prompt_suite.json").is_file()
    assert [c["suite_beside"] for c in rig.calls() if c["tool"] == "serve"] == [True]


def test_no_suite_file_is_copied_when_the_config_does_not_ask(rig):
    (rig.stage / "swap_config.json").write_text(json.dumps({**rig.cfg, "prompt_suite": False}))
    rig.run()
    assert not (rig.trial / "variant" / "prompt_suite.json").exists()


def test_a_prepare_step_that_refuses_is_a_result_not_a_crash(rig):
    rig.set(prepare_exit=2)
    p = rig.run()
    assert p.returncode == 0
    v = rig.result()["variant"]
    assert v["served"] is False and "prepare_swap.py refused" in v["failure"] and "exit 2" in v["failure"]
    assert [c["tool"] for c in rig.calls()] == ["prepare"]


def test_a_server_that_does_not_start_is_a_result_with_the_failure_text(rig):
    rig.set(serve_exit=1)
    assert rig.run().returncode == 0
    v = rig.result()["variant"]
    assert v["served"] is False and "code 1" in v["failure"] and "before it was healthy" in v["failure"]
    assert v["coherent"] is None and v["top1_agreement"] is None


def test_the_result_names_the_variants_tensor_cache_and_the_evidence_files_that_exist(rig):
    rig.run()
    res = rig.result()
    assert res["variant"]["tt_cache"] == rig.cfg["tt_cache"] + "-nodraft"
    assert res["evidence"] == sorted(["stages/2/sampling-trial/variant/evidence/swap-check.json",
                                      "stages/2/sampling-trial/variant/evidence/server.log",
                                      "stages/2/sampling-trial/variant/model-dir-sources.json"])


def test_the_evidence_list_names_only_files_that_exist(rig):
    rig.set(prepare_exit=2)
    rig.run()
    assert rig.result()["evidence"] == []


def test_the_children_stay_in_the_scripts_process_group_so_the_supervisors_stop_reaches_them(rig):
    rig.run()
    import os
    groups = {c["pgrp"] for c in rig.calls()}
    assert groups == {os.getpgrp()}          # the script was started without a new session, so its children share the group the test started it in


def test_a_stale_result_and_variant_directory_are_removed_before_the_run(rig):
    (rig.trial / RESULT).write_text('{"stale": true}')
    (rig.trial / "variant").mkdir()
    (rig.trial / "variant" / "old.txt").write_text("old")
    rig.set(prepare_exit=2)
    rig.run()
    assert "stale" not in rig.result() and not (rig.trial / "variant" / "old.txt").exists()


def test_the_result_is_written_whole_with_no_temporary_file_left(rig):
    rig.run()
    assert not list(rig.trial.glob("*.tmp"))


@pytest.mark.parametrize("missing", ["trial_config.json"])
def test_a_missing_trial_config_is_exit_2_and_no_result(rig, missing):
    (rig.trial / missing).unlink()
    p = rig.run()
    assert p.returncode == 2 and not (rig.trial / RESULT).exists()


@pytest.mark.parametrize("key", ["new_model_id", "hf_home", "port", "tt_cache", "run_dir"])
def test_a_stage_config_without_a_key_is_exit_2_and_no_result(rig, key):
    cfg = {k: v for k, v in rig.cfg.items() if k != key}
    (rig.stage / "swap_config.json").write_text(json.dumps(cfg))
    p = rig.run()
    assert p.returncode == 2 and key in p.stderr and not (rig.trial / RESULT).exists()


def test_a_missing_shipped_script_is_exit_2(rig):
    (rig.templates / "serve_and_compare.py").unlink()
    p = rig.run()
    assert p.returncode == 2 and "serve_and_compare.py" in p.stderr


def test_the_script_imports_nothing_from_orchard():
    import ast
    tree = ast.parse(SCRIPT.read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names] + \
            [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m.split(".")[0] == "orchard"]


def test_the_result_feeds_the_sampling_rule_with_no_adapter(rig):
    """The script and orchard/sampling_choice.py share the schema: trial-result.json's `variant.served` is what `variant_side` reads."""
    from orchard import sampling_choice as sc
    rig.set(serve_exit=1)
    rig.run()
    side = sc.variant_side(rig.stage)
    assert side is not None and side["served"] is False
