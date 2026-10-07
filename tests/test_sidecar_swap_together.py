# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The two prepare scripts of the weights-sidecar-check skill, run in ONE stage directory.

The skill runs prepare_swap.py and then prepare_parity.py in `stages/2`. Each had its own tests and each
built `stages/2/model-dir`, so the second wiped the first's directory: the swap server then had no
preprocessor_config.json and died in seconds on Cloudflare/clef (2026-10-06), while the parity half ran
fine. Nothing had ever run both in one directory. Each script now owns its own directory, and this test
runs them together in both orders."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SWAP = REPO / "orchard" / "skills" / "weights-swap-templates"
PARITY = REPO / "orchard" / "skills" / "sidecar-parity-templates"
NEAREST = "Qwen/Qwen3.8-27B"
RUN_SH = f"""#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{VENV:-$HERE/venv}}"
PYBIN="$VENV/bin/python"
export HF_MODEL="${{HF_MODEL:-{NEAREST}}}"
export QWEN36_DRAFTER="dflash2"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{NEAREST}" --max_num_seqs 4 "$@")
exec "${{CMD[@]}}"
"""


def snapshot(root: Path, files: dict, rev: str) -> Path:
    blobs, snap = root / "blobs", root / "snapshots" / rev
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, content in files.items():
        (blobs / f"b-{name}").write_text(content)
        (snap / name).symlink_to(os.path.relpath(blobs / f"b-{name}", snap))
    return snap


@pytest.fixture
def stage(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    stage.mkdir(parents=True)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run.sh").write_text(RUN_SH)
    base = snapshot(tmp_path / "base", {"config.json": '{"base": true}', "preprocessor_config.json": '{"pre": 1}',
                                        "video_preprocessor_config.json": '{"vid": 1}'}, "1" * 40)
    new = snapshot(tmp_path / "new", {
        "config.json": '{"new": true}', "tokenizer.json": "{}", "tokenizer_config.json": "{}",
        "chat_template.jinja": "x", "generation_config.json": "{}", "model.safetensors.index.json": "{}",
        "model-00001-of-00002.safetensors": "w1", "model-00002-of-00002.safetensors": "w2",
        "joint_head.safetensors": "head", "joint_head_config.json": "{}", "joint_schema_model.py": "# c"},
        "2" * 40)
    for src in (SWAP / "prepare_swap.py", PARITY / "prepare_parity.py"):
        shutil.copy(src, stage)
    (stage / "swap_config.json").write_text(json.dumps({
        "run_dir": str(run), "bundle_dir": str(bundle), "nearest_model_id": NEAREST, "base_snapshot": str(base),
        "new_snapshot": str(new), "new_model_id": "Cloudflare/clef", "tt_cache": str(tmp_path / "c1"),
        "hf_home": str(tmp_path / "hf"), "port": 8100}))
    (stage / "parity_config.json").write_text(json.dumps({
        "run_dir": str(run), "bundle_dir": str(bundle), "model_snapshot": str(new),
        "tt_cache": str(tmp_path / "c2")}))
    return stage


def run(stage: Path, script: str):
    return subprocess.run([sys.executable, str(stage / script)], cwd=stage, capture_output=True, text=True,
                          timeout=60)


def listing(d: Path) -> dict:
    """name -> (is link, resolved target or content hash)"""
    out = {}
    for p in sorted(d.iterdir()):
        out[p.name] = os.path.realpath(p) if p.is_symlink() else hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.mark.parametrize("order", [("prepare_swap.py", "prepare_parity.py"), ("prepare_parity.py", "prepare_swap.py")])
def test_both_prepare_scripts_leave_both_directories_and_both_launchers_in_place(stage, order):
    for script in order:
        r = run(stage, script)
        assert r.returncode == 0, r.stderr
    swap_dir, parity_dir = stage / "model-dir", stage / "parity-model-dir"
    assert swap_dir.is_dir() and parity_dir.is_dir() and swap_dir != parity_dir
    assert (stage / "run.sh").is_file() and (stage / "parity-run.sh").is_file()


@pytest.mark.parametrize("order", [("prepare_swap.py", "prepare_parity.py"), ("prepare_parity.py", "prepare_swap.py")])
def test_the_swap_directory_still_has_the_processor_files_after_the_parity_script_ran(stage, order):
    for script in order:
        assert run(stage, script).returncode == 0
    swap = {p.name for p in (stage / "model-dir").iterdir()}
    assert {"preprocessor_config.json", "video_preprocessor_config.json", "config.json"} <= swap


def test_running_the_parity_script_does_not_change_the_swap_directory_at_all(stage):
    assert run(stage, "prepare_swap.py").returncode == 0
    before = listing(stage / "model-dir")
    assert run(stage, "prepare_parity.py").returncode == 0
    assert listing(stage / "model-dir") == before


def test_the_parity_directory_has_the_new_models_config_and_no_sidecar(stage):
    assert run(stage, "prepare_parity.py").returncode == 0
    names = {p.name for p in (stage / "parity-model-dir").iterdir()}
    assert "config.json" in names and "joint_head.safetensors" not in names and "joint_schema_model.py" not in names
    assert json.loads((stage / "parity-model-dir" / "config.json").read_text()) == {"new": True}


def test_the_parity_launcher_points_the_weights_variables_at_the_parity_directory(stage):
    assert run(stage, "prepare_parity.py").returncode == 0
    text = (stage / "parity-run.sh").read_text()
    assert f'MODEL_WEIGHTS_DIR="{stage / "parity-model-dir"}"' in text
    assert f'"{stage / "model-dir"}"' not in text


def test_hidden_parity_loads_the_parity_directory(stage):
    import importlib.util
    spec = importlib.util.spec_from_file_location("hp_under_test", PARITY / "hidden_parity.py")
    src = (PARITY / "hidden_parity.py").read_text()
    assert 'HERE_DIR / "parity-model-dir"' in src and 'HERE_DIR / "model-dir"' not in src
