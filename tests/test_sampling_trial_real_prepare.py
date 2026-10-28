# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The drafter-off trial script (sampling_trial.py) with the real prepare_swap.py (issue 107). The script's own tests use stubs for both swap scripts, so
they cannot show that the config it writes is one prepare_swap.py accepts and acts on. Here prepare_swap.py is the shipped file and only the server step is
a stub: the variant directory the script builds must hold a run.sh with the drafter off and no on-device sampling, for a model that has its own mtp.*
tensors (the case that needs the variant), and the stage's own directory must be left alone."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prepare_swap_drafter_key import WITH_MTP, set_cfg  # noqa: E402
from test_weights_swap_templates import TEMPLATES, drafter_lines, new_index, prep, with_drafter  # noqa: E402,F401

SCRIPT = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates" / "sampling_trial.py"

STUB_SERVE = '''import json, pathlib
here = pathlib.Path(__file__).resolve().parent
(here / "evidence").mkdir()
(here / "evidence" / "swap-check.json").write_text(json.dumps({"result_draft": {"serves": True, "coherent": True, "top1_agreement": 0.96875, "n_tokens": 32,
                                                                "server_ready_s": 100.0}}))
'''


@pytest.fixture
def trial(prep, tmp_path):
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace(
        "--max_model_len", "--additional-config '{\"tt\": {\"sample_on_device_mode\": \"decode_only\"}}' --max_model_len", 1))
    with_drafter(prep)
    new_index(prep, WITH_MTP)
    set_cfg(prep, prompt_suite=False)
    checkout = tmp_path / "checkout"
    templates = checkout / "orchard" / "skills" / "weights-swap-templates"
    templates.mkdir(parents=True)
    shutil.copy(TEMPLATES / "prepare_swap.py", templates / "prepare_swap.py")
    (templates / "serve_and_compare.py").write_text(STUB_SERVE)
    stage = prep["stage"]
    d = stage / "sampling-trial"
    d.mkdir()
    shutil.copy(SCRIPT, d / "sampling_trial.py")
    (d / "trial_config.json").write_text(json.dumps({"stage_dir": str(stage), "orchard_dir": str(checkout)}))
    # Stage 2's own build, as the supervisor's first test leaves it: the drafter on.
    subprocess.run([sys.executable, str(stage / "prepare_swap.py")], check=True, capture_output=True, timeout=60)
    return {**prep, "dir": d}


def test_the_variant_the_script_builds_has_the_drafter_off_and_the_host_sampling(trial):
    assert drafter_lines(trial) == ['export QWEN36_DRAFTER="dflash2"']                       # stage 2's own run.sh: drafter on
    p = subprocess.run([sys.executable, str(trial["dir"] / "sampling_trial.py")], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    variant = trial["dir"] / "variant"
    text = (variant / "run.sh").read_text()
    assert 'export QWEN36_DRAFTER=""' in text and "dflash2" not in text
    assert "sample_on_device_mode" not in text
    assert (variant / "model-dir" / "config.json").is_file()
    result = json.loads((trial["dir"] / "trial-result.json").read_text())
    assert result["variant"]["served"] is True and result["steps"][0]["returncode"] == 0


def test_the_stage_own_build_is_not_touched_by_the_trial(trial):
    before = {p.name: p.read_bytes() for p in (trial["stage"] / "run.sh", trial["stage"] / "swap_config.json")}
    subprocess.run([sys.executable, str(trial["dir"] / "sampling_trial.py")], capture_output=True, text=True, timeout=60)
    assert {p.name: p.read_bytes() for p in (trial["stage"] / "run.sh", trial["stage"] / "swap_config.json")} == before
    assert drafter_lines(trial) == ['export QWEN36_DRAFTER="dflash2"']


def test_a_stage_that_grafted_is_built_without_the_graft_and_prepare_swap_accepts_it(trial):
    set_cfg(trial, graft_mtp=True)
    p = subprocess.run([sys.executable, str(trial["dir"] / "sampling_trial.py")], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    result = json.loads((trial["dir"] / "trial-result.json").read_text())
    assert result["variant"]["served"] is True, result["variant"]
    assert not (trial["dir"] / "variant" / "model-dir" / "model-mtp-from-base.safetensors").exists()
