# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The graft trial's variant sends the prompt suite when the stage does (PR 49 review, finding 3).

The drafted stage 2 config says `"prompt_suite": true`, and the trial copies that config into its variant directory. The variant
holds only the files the trial ships, so a missing prompt_suite.json made the real `suite_preflight()` exit 2, and every eligible
trial quietly kept the safe configuration without measuring the graft. These tests use the real serve_and_compare.py preflight,
and not a stub serving script: a stub never looks for the file."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

TEMPLATES = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates"

STUB_PREPARE = '''import pathlib
here = pathlib.Path(__file__).resolve().parent
(here / "model-dir").mkdir()
(here / "run.sh").write_text("#!/bin/bash\\n")
'''


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rig(tmp_path):
    """A fake checkout whose templates are the real serve_and_compare.py and prompt_suite.json and a prepare_swap.py stub, and a trial
    directory holding the real graft_trial.py."""
    orchard = tmp_path / "checkout"
    templates = orchard / "orchard" / "skills" / "weights-swap-templates"
    templates.mkdir(parents=True)
    shutil.copy(TEMPLATES / "serve_and_compare.py", templates)
    shutil.copy(TEMPLATES / "prompt_suite.json", templates)
    (templates / "prepare_swap.py").write_text(STUB_PREPARE)
    (orchard / "orchard" / "measure_serving.py").write_text("")
    trial = tmp_path / "stage" / "graft-trial"
    trial.mkdir(parents=True)
    shutil.copy(TEMPLATES / "graft_trial.py", trial)
    mod = load(trial / "graft_trial.py", f"graft_trial_under_test_{id(tmp_path)}")
    r = type("Rig", (), {})()
    r.tmp, r.orchard, r.templates, r.trial, r.mod = tmp_path, orchard, templates, trial, mod
    r.cfg = {"run_dir": str(tmp_path / "run"), "new_model_id": "Org/Model@" + "b" * 40, "nearest_model_id": "Qwen/Qwen3.8-27B",
             "hf_home": "/hf", "port": 8100, "tt_cache": str(tmp_path / "cache" / "tt_cache"), "prompt_suite": True}
    r.build = lambda cfg=None: mod.build_variant(orchard, cfg or r.cfg, tmp_path / "cache" / "tt_cache-graft", [])
    return r


def real_preflight(variant: Path):
    """What serve_and_compare.py does first when it starts in the variant directory: the real function, from the variant's own copy."""
    mod = load(variant / "serve_and_compare.py", f"variant_serve_and_compare_{abs(hash(str(variant)))}")
    return mod.suite_preflight(json.loads((variant / "swap_config.json").read_text()))


def test_a_variant_built_for_a_config_with_the_suite_passes_the_real_preflight(rig):
    variant, problem = rig.build()
    assert problem is None and variant is not None
    setup = real_preflight(variant)                      # exits 2 when prompt_suite.json is not next to the script
    assert setup is not None and setup["suite"]["prompts"]


def test_the_suite_file_in_the_variant_is_the_shipped_one_and_not_the_stage_copy(rig):
    stage_copy = rig.trial.parent / "prompt_suite.json"
    stage_copy.write_text(json.dumps({"prompts": [{"id": "edited", "text": "x", "messages": [{"role": "user", "content": "x"}]}]}))
    variant, _ = rig.build()
    assert (variant / "prompt_suite.json").read_bytes() == (TEMPLATES / "prompt_suite.json").read_bytes()


def test_a_config_without_the_suite_gets_a_variant_without_the_file(rig):
    cfg = {k: v for k, v in rig.cfg.items() if k != "prompt_suite"}
    variant, _ = rig.build(cfg)
    assert not (variant / "prompt_suite.json").exists()


def test_a_config_that_sets_the_suite_to_false_does_not_ship_it(rig):
    variant, _ = rig.build({**rig.cfg, "prompt_suite": False})
    assert not (variant / "prompt_suite.json").exists()


def test_the_variants_swap_config_keeps_the_suite_key_and_the_other_files_are_still_shipped(rig):
    variant, _ = rig.build()
    cfg = json.loads((variant / "swap_config.json").read_text())
    assert cfg["prompt_suite"] is True and cfg["graft_mtp"] is True
    for name in ("prepare_swap.py", "serve_and_compare.py"):
        assert (variant / name).read_bytes() == (rig.templates / name).read_bytes()


def test_the_script_exits_2_and_names_the_suite_file_when_the_checkout_lacks_it(rig):
    import subprocess
    stage = rig.trial.parent
    (stage / "model-dir").mkdir()
    (stage / "run.sh").write_text("#!/bin/bash\n")
    (stage / "swap_config.json").write_text(json.dumps(rig.cfg))
    (rig.trial / "trial_config.json").write_text(json.dumps({"stage_dir": str(stage), "orchard_dir": str(rig.orchard)}))
    (rig.templates / "prompt_suite.json").unlink()
    r = subprocess.run([sys.executable, str(rig.trial / "graft_trial.py")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "prompt_suite.json" in r.stderr
    assert not (rig.trial / "trial-result.json").exists()


def test_the_script_does_not_ask_for_the_suite_file_when_the_config_does_not_use_it(rig):
    import subprocess
    stage = rig.trial.parent
    (stage / "model-dir").mkdir()
    (stage / "run.sh").write_text("#!/bin/bash\n")
    cfg = {k: v for k, v in rig.cfg.items() if k != "prompt_suite"}
    (stage / "swap_config.json").write_text(json.dumps(cfg))
    (rig.trial / "trial_config.json").write_text(json.dumps({"stage_dir": str(stage), "orchard_dir": str(rig.orchard)}))
    (rig.templates / "prompt_suite.json").unlink()
    r = subprocess.run([sys.executable, str(rig.trial / "graft_trial.py")], capture_output=True, text=True, timeout=60)
    assert "prompt_suite.json" not in r.stderr


@pytest.mark.parametrize("value, shipped_", [(True, True), (1, True), ("yes", True), (False, False), (0, False), (None, False), ("", False)])
def test_the_variant_gets_the_suite_file_exactly_when_the_preflight_would_look_for_it(rig, value, shipped_):
    # serve_and_compare.suite_preflight() starts the suite when `cfg.get("prompt_suite")` is truthy; the trial follows the same rule.
    variant, _ = rig.build({**rig.cfg, "prompt_suite": value})
    assert (variant / "prompt_suite.json").exists() is shipped_
    if shipped_:
        assert real_preflight(variant) is not None
