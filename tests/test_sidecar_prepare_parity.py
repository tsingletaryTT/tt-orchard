# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""prepare_parity.py (orchard/skills/sidecar-parity-templates/).

It builds, in a stage directory, the launcher for the sidecar parity check (a copy of the nearest
bundle's run.sh with edits) and a clean model directory. Everything runs on fakes in tmp directories:
no device is opened and nothing is imported from tt-metal."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "sidecar-parity-templates"
NEAREST = "Qwen/Qwen3.8-27B"

BUNDLE_RUN_SH = f"""#!/usr/bin/env bash
# Serve this model on TT hardware.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{VENV:-$HERE/venv}}"
PYBIN="$VENV/bin/python"
export HF_HOME="${{HF_HOME:-$HERE/.hf}}"
export TT_CACHE_PATH="${{TT_CACHE_PATH:-$HERE/.tt_cache}}"
export HF_MODEL="${{HF_MODEL:-{NEAREST}}}"
export QWEN36_DRAFTER="dflash2"
export DFLASH_WEIGHTS="incoai/Qwen3.8-27B-DFlash2@dedf8df"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{NEAREST}" --max_num_seqs 4 "$@")
exec "${{CMD[@]}}"
"""


def make_snapshot(root: Path, files: dict) -> Path:
    snap = root / "snapshots" / ("1" * 40)
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, content in files.items():
        (blobs / f"b-{name}").write_text(content)
        (snap / name).symlink_to(os.path.relpath(blobs / f"b-{name}", snap))
    return snap


NEW_FILES = {"config.json": '{"model_type": "qwen3_5"}', "tokenizer.json": "{}", "tokenizer_config.json": "{}",
             "chat_template.jinja": "x", "generation_config.json": "{}",
             "model.safetensors.index.json": "{}", "model-00001-of-00002.safetensors": "w1",
             "model-00002-of-00002.safetensors": "w2",
             "joint_head.safetensors": "head", "joint_head_config.json": "{}", "joint_schema_model.py": "# code",
             "README.md": "# readme", "LICENSE": "apache"}


@pytest.fixture
def stage(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    shutil.copy(TEMPLATES / "prepare_parity.py", stage)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run.sh").write_text(BUNDLE_RUN_SH)
    new = make_snapshot(tmp_path / "new", NEW_FILES)
    cfg = {"run_dir": str(tmp_path), "bundle_dir": str(bundle), "model_snapshot": str(new),
           "tt_cache": str(tmp_path / "cache")}
    (stage / "parity_config.json").write_text(json.dumps(cfg))
    return stage


def prepare(stage):
    return subprocess.run([sys.executable, str(stage / "prepare_parity.py")], cwd=stage, capture_output=True,
                          text=True)


def edit_bundle(stage, fn):
    cfg = json.loads((stage / "parity_config.json").read_text())
    run = Path(cfg["bundle_dir"]) / "run.sh"
    run.write_text(fn(run.read_text()))


# ---- the launcher ------------------------------------------------------------------------------

def test_the_launcher_is_a_copy_of_run_sh_with_the_last_line_replaced_by_the_parity_script(stage):
    done = prepare(stage)
    assert done.returncode == 0, done.stderr
    text = (stage / "parity-run.sh").read_text()
    assert 'exec "${CMD[@]}"' not in text
    assert text.rstrip().splitlines()[-1] == f'exec "$PYBIN" "{stage / "hidden_parity.py"}" "$@"'
    assert os.access(stage / "parity-run.sh", os.X_OK)


def test_the_launcher_still_finds_the_bundles_venv_and_model_code(stage):
    prepare(stage)
    bundle = json.loads((stage / "parity_config.json").read_text())["bundle_dir"]
    text = (stage / "parity-run.sh").read_text()
    assert f'HERE="{bundle}"' in text and text.count("HERE=") == 1


def test_no_drafter_is_asked_for(stage):
    prepare(stage)
    text = (stage / "parity-run.sh").read_text()
    lines = [l.strip() for l in text.splitlines() if "QWEN36_DRAFTER=" in l]
    assert lines and all(l == 'export QWEN36_DRAFTER=""' for l in lines), lines


def test_both_weight_variables_name_the_model_dir(stage):
    prepare(stage)
    text = (stage / "parity-run.sh").read_text()
    md = stage / "parity-model-dir"
    assert f'export HF_MODEL="{md}"' in text and f'export MODEL_WEIGHTS_DIR="{md}"' in text


def test_the_caches_and_the_offline_flag_are_set_explicitly(stage):
    prepare(stage)
    cache = json.loads((stage / "parity_config.json").read_text())["tt_cache"]
    text = (stage / "parity-run.sh").read_text()
    assert f'export TT_CACHE_PATH="{cache}"' in text and f'export TT_CACHE_HOME="{cache}"' in text
    assert "export HF_HUB_OFFLINE=1" in text
    # They come after the bundle's own lines, so they win.
    assert text.index('export TT_CACHE_PATH="' + cache) > text.index("${TT_CACHE_PATH:-")


def test_the_rest_of_the_bundle_environment_is_kept(stage):
    prepare(stage)
    text = (stage / "parity-run.sh").read_text()
    assert 'PYBIN="$VENV/bin/python"' in text and "DFLASH_WEIGHTS" in text


@pytest.mark.parametrize("name,fn,needle", [
    ("no HERE", lambda t: t.replace("HERE=", "THERE="), "HERE="),
    ("two HERE", lambda t: t + '\nHERE="/x"\n', "HERE="),
    ("no exec", lambda t: t.replace('exec "${CMD[@]}"', "echo done"), "exec"),
    ("two exec", lambda t: t + '\nexec "${CMD[@]}"\n', "exec"),
    ("two drafter lines", lambda t: t + '\nexport QWEN36_DRAFTER="x"\n', "QWEN36_DRAFTER"),
])
def test_an_edit_that_does_not_happen_exactly_once_exits_2_and_writes_nothing(stage, name, fn, needle):
    edit_bundle(stage, fn)
    done = prepare(stage)
    assert done.returncode == 2 and needle in done.stderr, name
    assert not (stage / "parity-run.sh").exists() and not (stage / "parity-model-dir").exists()


def test_an_absent_drafter_line_is_reported_and_not_an_error(stage):
    edit_bundle(stage, lambda t: t.replace('export QWEN36_DRAFTER="dflash2"\n', ""))
    done = prepare(stage)
    assert done.returncode == 0 and "QWEN36_DRAFTER" in done.stdout
    assert 'export QWEN36_DRAFTER=""' in (stage / "parity-run.sh").read_text()


# ---- the model dir -----------------------------------------------------------------------------

def test_the_model_dir_links_the_backbone_and_leaves_the_sidecar_and_code_out(stage):
    assert prepare(stage).returncode == 0
    names = sorted(p.name for p in (stage / "parity-model-dir").iterdir())
    assert "model-00001-of-00002.safetensors" in names and "tokenizer.json" in names
    assert "config.json" in names and "model.safetensors.index.json" in names
    for left_out in ("joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py", "README.md"):
        assert left_out not in names


def test_weight_links_are_resolved_to_the_blobs(stage):
    prepare(stage)
    link = stage / "parity-model-dir" / "model-00001-of-00002.safetensors"
    assert link.is_symlink() and os.path.isabs(os.readlink(link)) and "snapshots" not in os.readlink(link)
    assert link.read_text() == "w1"


def test_the_config_is_the_new_models_own(stage):
    prepare(stage)
    assert json.loads((stage / "parity-model-dir" / "config.json").read_text()) == {"model_type": "qwen3_5"}


def test_a_second_run_rebuilds_the_model_dir_from_scratch(stage):
    prepare(stage)
    (stage / "parity-model-dir" / "stray").write_text("x")
    prepare(stage)
    assert not (stage / "parity-model-dir" / "stray").exists()


def test_a_symlinked_model_dir_is_refused(stage, tmp_path):
    (stage / "parity-model-dir").symlink_to(tmp_path)
    done = prepare(stage)
    assert done.returncode == 2 and "symlink" in done.stderr


@pytest.mark.parametrize("missing", ["tokenizer.json", "config.json"])
def test_a_required_backbone_file_that_is_missing_exits_2(stage, missing):
    cfg = json.loads((stage / "parity_config.json").read_text())
    (Path(cfg["model_snapshot"]) / missing).unlink()
    done = prepare(stage)
    assert done.returncode == 2 and missing in done.stderr


def test_a_snapshot_without_weight_shards_exits_2(stage):
    cfg = json.loads((stage / "parity_config.json").read_text())
    for p in Path(cfg["model_snapshot"]).glob("model-*.safetensors"):
        p.unlink()
    done = prepare(stage)
    assert done.returncode == 2 and "safetensors" in done.stderr


@pytest.mark.parametrize("key", ["bundle_dir", "model_snapshot", "tt_cache"])
def test_a_missing_config_key_is_named(stage, key):
    cfg = json.loads((stage / "parity_config.json").read_text())
    del cfg[key]
    (stage / "parity_config.json").write_text(json.dumps(cfg))
    done = prepare(stage)
    assert done.returncode == 2 and key in done.stderr


def test_a_missing_config_file_exits_2(stage):
    (stage / "parity_config.json").unlink()
    assert prepare(stage).returncode == 2
