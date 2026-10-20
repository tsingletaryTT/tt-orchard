# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""prepare_restricted_head.py builds the launcher and the model directory the parity check runs in.

The launcher is the nearest bundle's run.sh with two changes that matter: it runs restricted_head_parity.py, and it points the TT runtime at a
directory that holds only the backbone (so the readout and the model's code cannot be read by the loader). These tests use a fake bundle.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIR = REPO / "orchard" / "skills" / "restricted-head-templates"
PARITY = REPO / "orchard" / "skills" / "sidecar-parity-templates" / "prepare_parity.py"
RUN_SH = '#!/bin/bash\nHERE="$(cd "$(dirname "$0")" && pwd)"\nexport QWEN36_DRAFTER="/x/drafter"\nPYBIN="$HERE/venv/bin/python"\nCMD=(vllm serve)\nexec "${CMD[@]}"\n'


def build(tmp_path, run_sh=RUN_SH, *, shards=True, copy_builder=True):
    stage = tmp_path / "stage"
    stage.mkdir()
    shutil.copy(DIR / "prepare_restricted_head.py", stage)
    if copy_builder:
        shutil.copy(PARITY, stage)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run.sh").write_text(run_sh)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text("{}")
    (snap / "readout.safetensors").write_bytes(b"x")
    (snap / "source").mkdir()
    (snap / "source" / "model.py").write_text("raise SystemExit('must not be linked')\n")
    if shards:
        (snap / "model-00001-of-00002.safetensors").write_bytes(b"a")
        (snap / "model-00002-of-00002.safetensors").write_bytes(b"b")
    (stage / "restricted_head_config.json").write_text(json.dumps({"bundle_dir": str(bundle), "model_snapshot": str(snap), "tt_cache": str(tmp_path / "cache")}))
    r = subprocess.run([sys.executable, str(stage / "prepare_restricted_head.py")], capture_output=True, text=True)
    return stage, r


def test_the_launcher_runs_the_parity_script_with_the_drafter_off_and_the_clean_model_dir(tmp_path):
    stage, r = build(tmp_path)
    assert r.returncode == 0, r.stderr
    sh = (stage / "parity-run.sh").read_text()
    assert f'exec "$PYBIN" "{stage / "restricted_head_parity.py"}"' in sh
    assert sh.count("exec ") == 1 and 'export QWEN36_DRAFTER=""' in sh and "/x/drafter" not in sh
    md = stage / "parity-model-dir"
    assert f'export MODEL_WEIGHTS_DIR="{md}"' in sh and f'export TT_CACHE_PATH="{tmp_path / "cache"}"' in sh
    assert (stage / "parity-run.sh").stat().st_mode & 0o111


def test_the_model_dir_holds_the_backbone_and_not_the_readout_or_the_code(tmp_path):
    stage, r = build(tmp_path)
    names = sorted(p.name for p in (stage / "parity-model-dir").iterdir())
    assert names == ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                     "model.safetensors.index.json", "tokenizer.json"]
    assert (stage / "parity-model-dir" / "model-00001-of-00002.safetensors").is_symlink()
    assert not (stage / "parity-model-dir" / "config.json").is_symlink()          # the config is a copy


def test_a_run_script_without_exactly_one_exec_line_writes_nothing(tmp_path):
    stage, r = build(tmp_path, run_sh=RUN_SH.replace('exec "${CMD[@]}"\n', ""))
    assert r.returncode == 2 and "exec" in r.stderr
    assert not (stage / "parity-run.sh").exists() and not (stage / "parity-model-dir").exists()


def test_a_snapshot_with_no_shards_writes_nothing(tmp_path):
    stage, r = build(tmp_path, shards=False)
    assert r.returncode == 2 and "model-*.safetensors" in r.stderr
    assert not (stage / "parity-model-dir").exists()


def test_the_builder_is_found_in_the_skills_tree_when_it_was_not_copied(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    # the script's own directory is the real templates directory's sibling layout: run it in place with its config beside it
    work = tmp_path / "work" / "restricted-head-templates"
    work.mkdir(parents=True)
    shutil.copy(DIR / "prepare_restricted_head.py", work)
    shutil.copytree(PARITY.parent, tmp_path / "work" / "sidecar-parity-templates")
    bundle, snap = tmp_path / "bundle", tmp_path / "snap"
    bundle.mkdir(), snap.mkdir()
    (bundle / "run.sh").write_text(RUN_SH)
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "model-00001-of-00001.safetensors").write_bytes(b"a")
    (work / "restricted_head_config.json").write_text(json.dumps({"bundle_dir": str(bundle), "model_snapshot": str(snap), "tt_cache": str(tmp_path / "c")}))
    r = subprocess.run([sys.executable, str(work / "prepare_restricted_head.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (work / "parity-model-dir" / "config.json").exists() and (work / "parity-run.sh").exists()     # next to this script, not to the builder
