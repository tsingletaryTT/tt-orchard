# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The run.sh edits for a weights-only package, and the model-dir script it runs (plan 5)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard.package import (EXEC_LINE, PREPARE_LINE, WIRING_LINES, PackageError, extra_args_from,
                             splice_extra_args, weights_wiring_problems, wire_weights)

NEW, BASE = "Altworld/Hemmingway-1", "Qwen/Qwen3.8-27B"
NEW_REV, BASE_REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXTRA = ("--additional-config '{\"tt\": {\"l1_small_size\": 24576, \"fabric_config\": \"FABRIC_1D\"}}' "
         "--max-num-batched-tokens 65536 --enable-auto-tool-choice --tool-call-parser qwen3_coder "
         "--reasoning_parser qwen3 --no-async-scheduling")
TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "package_templates" / "prepare_model_dir.py"


def generated_run_sh(weights, rev, extra=""):
    """The lines of a package-thin run.sh that the edits touch, in the generated order
    (copied from the staged qwen3.8-27b-dflash2-p300 bundle, 2026-09-30)."""
    tail = f" {extra}" if extra else ""
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'VENV="${VENV:-$HERE/venv}"\nPYBIN="$VENV/bin/python"\n'
        'export HF_HOME="${HF_HOME:-$HERE/.hf}"\n'
        f'export HF_MODEL="${{HF_MODEL:-{weights}}}"\n'
        f'export TT_MODEL_WEIGHTS_REVISION="${{TT_MODEL_WEIGHTS_REVISION:-{rev}}}"\n'
        'export DFLASH_WEIGHTS="incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4"\n'
        f'CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{weights}" --max_num_seqs 4 '
        f'--block_size 64 --revision {rev} --tokenizer-revision {rev} --max_model_len 262144{tail} "$@")\n'
        'if [ "${TT_MODEL_PRINT:-0}" = "1" ]; then\n'
        "  printf 'HF_MODEL=%s\\n' \"${HF_MODEL:-}\"\n  exit 0\nfi\n"
        'exec "${CMD[@]}"\n')


def test_the_extra_arguments_are_read_from_the_source_bundle():
    assert extra_args_from(generated_run_sh(BASE, BASE_REV, EXTRA)) == EXTRA
    assert extra_args_from(generated_run_sh(BASE, BASE_REV)) == ""


def test_a_source_run_sh_without_the_expected_command_is_refused():
    with pytest.raises(PackageError, match="--max_model_len"):
        extra_args_from(generated_run_sh(BASE, BASE_REV).replace("--max_model_len 262144", ""))


def test_the_extra_arguments_go_before_the_passed_through_ones():
    text = splice_extra_args(generated_run_sh(NEW, NEW_REV), EXTRA)
    assert f'--max_model_len 262144 {EXTRA} "$@")\n' in text
    assert extra_args_from(text) == EXTRA


def test_wiring_points_every_weights_setting_at_the_model_dir():
    text = wire_weights(generated_run_sh(NEW, NEW_REV, EXTRA), NEW)
    for line in WIRING_LINES:
        assert f"\n{line}\n" in text
    assert '--model "$HERE/model-dir" --max_num_seqs 4' in text
    assert "--revision" not in text and "--tokenizer-revision" not in text
    assert text.index(PREPARE_LINE) < text.index(EXEC_LINE)
    assert NEW not in text.split("CMD=(")[1]          # the command names the model-dir only
    assert weights_wiring_problems(text, nearest_model=BASE) == []


def test_the_unwired_generated_script_has_wiring_problems():
    problems = weights_wiring_problems(generated_run_sh(NEW, NEW_REV), nearest_model=BASE)
    assert any("MODEL_WEIGHTS_DIR" in p for p in problems)
    assert any("prepare_model_dir.py" in p for p in problems)


def test_a_script_that_names_the_base_model_has_a_wiring_problem():
    text = wire_weights(generated_run_sh(NEW, NEW_REV), NEW) + f'export X="{BASE}"\n'
    assert weights_wiring_problems(text, nearest_model=BASE) == [
        f"run.sh names the nearest model {BASE}"]


def test_a_second_weights_setting_has_a_wiring_problem():
    text = wire_weights(generated_run_sh(NEW, NEW_REV), NEW).replace(
        EXEC_LINE, f'export MODEL_WEIGHTS_DIR="/somewhere/else"\n{EXEC_LINE}')
    assert any("MODEL_WEIGHTS_DIR" in p for p in weights_wiring_problems(text, nearest_model=BASE))


@pytest.mark.parametrize("edit, named", [
    (lambda s: s.replace(f'--model "{NEW}"', '--model "Other/Model"'), "--model"),
    (lambda s: s.replace(f"--revision {NEW_REV} ", ""), "--revision"),
    (lambda s: s.replace('export HF_MODEL="${HF_MODEL:-' + NEW + '}"\n', ""), "HF_MODEL"),
    (lambda s: s.replace('exec "${CMD[@]}"\n', ""), "exec"),
])
def test_an_edit_that_does_not_apply_exactly_once_is_refused(edit, named):
    with pytest.raises(PackageError, match=named):
        wire_weights(edit(generated_run_sh(NEW, NEW_REV)), NEW)


# ---- prepare_model_dir.py, the script the staged run.sh runs before vLLM -------------------------

def hf_snapshot(hf_home, repo, rev, files):
    """An HF-cache-shaped snapshot under hf_home/hub: files are links into blobs/."""
    org, name = repo.split("/")
    root = hf_home / "hub" / f"models--{org}--{name}"
    blobs, snap = root / "blobs", root / "snapshots" / rev
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True)
    for fname, content in files.items():
        (blobs / f"b-{fname}").write_text(content)
        (snap / fname).symlink_to(os.path.relpath(blobs / f"b-{fname}", snap))
    return snap


@pytest.fixture
def bundle(tmp_path):
    b = tmp_path / "bundle"
    (b / "base_config").mkdir(parents=True)
    (b / "base_config" / "config.json").write_text('{"architectures": ["Qwen3_5ForConditionalGeneration"]}')
    (b / "base_config" / "preprocessor_config.json").write_text("{}")
    (b / "tt_kernel_manifest.json").write_text(json.dumps(
        {"schema_version": "6", "weights": {"repo_id": NEW, "revision": NEW_REV}}))
    (b / "prepare_model_dir.py").write_text(TEMPLATE.read_text())
    return b


def run_prepare(bundle, hf_home, offline=True, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "HF_HUB_CACHE")}
    env["HF_HOME"] = str(hf_home)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run([sys.executable, str(bundle / "prepare_model_dir.py")], env=env,
                          capture_output=True, text=True, timeout=60)


NEW_FILES = {"config.json": '{"architectures": ["Qwen3_5ForCausalLM"]}', "tokenizer.json": "{}",
             "tokenizer_config.json": "{}", "generation_config.json": "{}",
             "model.safetensors.index.json": "{}", "model-00001-of-00002.safetensors": "w1",
             "model-00002-of-00002.safetensors": "w2"}


def test_the_model_dir_has_the_base_config_and_the_new_weights(bundle, tmp_path):
    hf = tmp_path / "hf"
    snap = hf_snapshot(hf, NEW, NEW_REV, NEW_FILES)
    r = run_prepare(bundle, hf)
    assert r.returncode == 0, r.stdout + r.stderr
    md = bundle / "model-dir"
    assert json.loads((md / "config.json").read_text())["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert not (md / "config.json").is_symlink()
    for name in ("tokenizer.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        assert os.readlink(md / name) == os.path.realpath(snap / name)
    assert (md / ".weights").read_text() == f"{NEW}@{NEW_REV}"
    assert run_prepare(bundle, hf).returncode == 0              # a second start rebuilds it


def test_a_missing_snapshot_offline_exits_2_and_names_it(bundle, tmp_path):
    r = run_prepare(bundle, tmp_path / "empty-hf")
    assert r.returncode == 2
    assert NEW in r.stderr and NEW_REV in r.stderr
    assert not (bundle / "model-dir").exists()


def test_a_missing_snapshot_online_is_downloaded_at_the_pinned_revision(bundle, tmp_path):
    hf = tmp_path / "hf"
    fake = tmp_path / "fakepkgs" / "huggingface_hub"
    fake.mkdir(parents=True)
    calls = tmp_path / "calls.json"
    (fake / "__init__.py").write_text(
        "import json, os, pathlib\n"
        "def snapshot_download(repo_id, revision):\n"
        f"    pathlib.Path({str(calls)!r}).write_text(json.dumps([repo_id, revision]))\n"
        "    snap = pathlib.Path(os.environ['HF_HOME']) / 'hub' / 'got' / revision\n"
        "    snap.mkdir(parents=True)\n"
        "    (snap / 'tokenizer.json').write_text('{}')\n"
        "    (snap / 'model-00001-of-00001.safetensors').write_text('w')\n"
        "    return str(snap)\n")
    r = run_prepare(bundle, hf, offline=False, pythonpath=fake.parent)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(calls.read_text()) == [NEW, NEW_REV]
    assert (bundle / "model-dir" / "model-00001-of-00001.safetensors").is_symlink()


def test_a_model_dir_the_script_did_not_build_is_left_alone(bundle, tmp_path):
    hf = tmp_path / "hf"
    hf_snapshot(hf, NEW, NEW_REV, NEW_FILES)
    (bundle / "model-dir").mkdir()
    (bundle / "model-dir" / "mine.txt").write_text("the operator's file")
    r = run_prepare(bundle, hf)
    assert r.returncode == 2 and "not built by this script" in r.stderr
    assert (bundle / "model-dir" / "mine.txt").read_text() == "the operator's file"


def test_an_unpinned_weights_revision_is_refused(bundle, tmp_path):
    (bundle / "tt_kernel_manifest.json").write_text(json.dumps(
        {"weights": {"repo_id": NEW, "revision": "main"}}))
    r = run_prepare(bundle, tmp_path / "hf")
    assert r.returncode == 2 and "40-character" in r.stderr
