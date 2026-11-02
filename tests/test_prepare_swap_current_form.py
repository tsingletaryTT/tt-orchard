# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""prepare_swap.py with a current-generation bundle (the SenseNova-6.8-Code2Video run, 2026-11-01).

A current bundle (episod/arex-2-p300, episod/qwen3.5-35b-a3b-blackhole-p300) has no `--model "<nearest id>"` in its run.sh. Its
command is `--model "$HERE/model-dir"`, and run.sh builds that directory at start by running the bundle's prepare_model_dir.py.
That script reads the weights repo and revision from tt_kernel_manifest.json (it does not read the environment) and copies every file of
the bundle's base_config/ into the directory. HF_MODEL and TT_MODEL_WEIGHTS_REVISION only feed a download check whose result
run.sh does not use. A copy of that run.sh therefore cannot be pointed at other weights by an environment variable. It has to
stop running prepare_model_dir.py and name the swap's own model-dir in its command and in the two variables that choose the
weights directory.

The fixture below has the shape of the real run.sh (same line forms, same order) and none of its text that does not matter here."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weights_swap_templates import (NEAREST, TEMPLATES, make_snapshot, new_index, put,  # noqa: E402
                                         run_prepare)

BASE_CONFIG_FILES = {"config.json": '{"architectures": ["BundleArch"], "text_config": {"vocab_size": 100}}',
                     "generation_config.json": '{"eos_token_id": [7, 8], "temperature": 1.0}',
                     "preprocessor_config.json": '{"bundle": "pre"}',
                     "video_preprocessor_config.json": '{"bundle": "video"}'}
NEW_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
             "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

CURRENT_RUN_SH = """#!/usr/bin/env bash
# Serve this model on TT hardware. Assumes ./install.sh has been run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/venv}"
PYBIN="$VENV/bin/python"
export HF_HOME="${HF_HOME:-$HERE/.hf}"                  # HF weights + hub cache
export HF_MODEL="${HF_MODEL:-Org/Nearest}"
export TT_MODEL_WEIGHTS_REVISION="${TT_MODEL_WEIGHTS_REVISION:-d4e3502f92d9e889031c2d04e387ab2eb3520268}"
export QWEN36_DRAFTER="dflash2"
export DFLASH_WEIGHTS="incoai/drafter@dedf8df68adfb1afeaf7b7480c0a0243108177b4"
CMD=("$PYBIN" "$HERE/record_args.py" --model "$HERE/model-dir" --max_num_seqs 4 --additional-config '{"tt": {"l1_small_size": 24576, "sample_on_device_mode": "decode_only"}}' --max-num-batched-tokens 65536 "$@")
# TT_MODEL_PRINT=1 (set by `tt-model serve --print`) echoes the fully-resolved command+env
if [ "${TT_MODEL_PRINT:-0}" = "1" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
# Resolve pinned weights before vLLM opens the device.
resolved_hf_model="$("$PYBIN" - "$HF_MODEL" "${TT_MODEL_WEIGHTS_REVISION:-}" "${TT_AUXILIARY_WEIGHTS:-}" '{"allow_patterns": null, "ignore_patterns": null}' <<'PY'
import sys
from pathlib import Path

spec = sys.argv[1]
if Path(spec).expanduser().is_dir():
    print(Path(spec).resolve())
    raise SystemExit
# The real script calls huggingface_hub.snapshot_download here.
raise SystemExit("would download " + spec)
PY
)"
export HF_MODEL="$HERE/model-dir"
export MODEL_WEIGHTS_DIR="$HERE/model-dir"
"$PYBIN" "$HERE/prepare_model_dir.py"
exec "${CMD[@]}"
"""

# Stand-ins for the bundle's own files. The first records how the server would have been started; the second is the
# script the swap must not run (it would rebuild the bundle's model-dir from the manifest's weights).
RECORD_ARGS = ("import json, os, sys\n"
               "json.dump({'argv': sys.argv[1:], 'env': {k: os.environ.get(k) for k in ('HF_MODEL', 'MODEL_WEIGHTS_DIR')}},\n"
               "          open(os.environ['RECORD_FILE'], 'w'))\n")
PREPARE_MODEL_DIR = "import pathlib\n(pathlib.Path(__file__).resolve().parent / 'prepare_model_dir_ran').write_text('x')\n"


@pytest.fixture
def cur(tmp_path):
    """A stage dir with prepare_swap.py, a current-form bundle (run.sh, base_config/, graft.json) and two snapshots."""
    bundle = tmp_path / "bundle"
    (bundle / "base_config").mkdir(parents=True)
    (bundle / "venv" / "bin").mkdir(parents=True)
    (bundle / "venv" / "bin" / "python").symlink_to(sys.executable)
    (bundle / "run.sh").write_text(CURRENT_RUN_SH)
    (bundle / "record_args.py").write_text(RECORD_ARGS)
    (bundle / "prepare_model_dir.py").write_text(PREPARE_MODEL_DIR)
    (bundle / "graft.json").write_text('{"repo": "Org/Nearest"}')
    for name, text in BASE_CONFIG_FILES.items():
        (bundle / "base_config" / name).write_text(text)
    base = make_snapshot(tmp_path / "base", {"config.json": '{"snapshot": "base"}', "preprocessor_config.json": '{"snapshot": "pre"}'})
    new = make_snapshot(tmp_path / "new", {"config.json": '{"snapshot": "new"}', **{n: f"new {n}" for n in NEW_FILES}})
    stage = tmp_path / "run" / "stages" / "2"
    stage.mkdir(parents=True)
    shutil.copy(TEMPLATES / "prepare_swap.py", stage)
    cfg = {"run_dir": str(tmp_path / "run"), "bundle_dir": str(bundle), "nearest_model_id": "Org/Nearest",
           "base_snapshot": str(base), "new_snapshot": str(new), "new_model_id": "Altworld/Hemmingway-1",
           "tt_cache": str(tmp_path / "cache"), "hf_home": str(tmp_path / "hf"), "port": 8100}
    (stage / "swap_config.json").write_text(json.dumps(cfg))
    return {"stage": stage, "bundle": bundle, "base": base, "new": new, "tmp": tmp_path}


def run_text(cur) -> str:
    return (cur["stage"] / "run.sh").read_text()


def sources(cur) -> dict:
    return json.loads((cur["stage"] / "model-dir-sources.json").read_text())


def bundle_listing(bundle: Path) -> list[str]:
    return sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*") if "venv" not in p.parts)


# ---- the run script ---------------------------------------------------------------------------------------------

def test_a_current_bundle_prepares(cur):
    r = run_prepare(cur["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (cur["stage"] / "run.sh").is_file() and (cur["stage"] / "model-dir").is_dir()


def test_the_command_serves_the_swaps_model_dir_and_the_bundles_own_is_never_named(cur):
    assert run_prepare(cur["stage"]).returncode == 0
    text, md = run_text(cur), str(cur["stage"] / "model-dir")
    assert f"--model {md} " in text and text.count("--model ") == 1
    assert "$HERE/model-dir" not in text and "${HERE}/model-dir" not in text
    assert f'HERE="{cur["bundle"]}"\n' in text and "BASH_SOURCE" not in text


def test_both_variables_that_choose_the_weights_directory_name_the_swaps_model_dir(cur):
    assert run_prepare(cur["stage"]).returncode == 0
    text, md = run_text(cur), str(cur["stage"] / "model-dir")
    assert text.count(f'export HF_MODEL="{md}"\n') == 2          # the default line and the line after the download check
    assert text.count(f'export MODEL_WEIGHTS_DIR="{md}"\n') == 1
    assert "Org/Nearest" not in text


def test_the_bundles_prepare_model_dir_step_is_removed(cur):
    r = run_prepare(cur["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    text = run_text(cur)
    assert "prepare_model_dir.py" in text and '"$PYBIN" "$HERE/prepare_model_dir.py"' not in text   # only a comment names it
    assert [l for l in text.splitlines() if l.strip().startswith("#") and "prepare_model_dir.py" in l]
    assert "prepare_model_dir.py" in r.stdout


def test_only_the_edited_lines_differ_from_the_bundles_run_script(cur):
    assert run_prepare(cur["stage"]).returncode == 0
    old, new = CURRENT_RUN_SH.splitlines(), run_text(cur).splitlines()
    assert len(old) == len(new)
    changed = [a.split("=")[0].replace("export ", "") if a.startswith(("HERE", "export")) else a[:8]
               for a, b in zip(old, new) if a != b]
    assert sorted(changed) == sorted(["HERE", "HF_MODEL", "HF_MODEL", "MODEL_WEIGHTS_DIR", "CMD=(\"$P", '"$PYBIN"'])


def test_running_the_edited_script_starts_the_server_on_the_swap_model_dir_and_leaves_the_bundle_alone(cur):
    assert run_prepare(cur["stage"]).returncode == 0
    before = bundle_listing(cur["bundle"])
    record = cur["tmp"] / "record.json"
    env = {k: v for k, v in os.environ.items() if k not in ("HF_MODEL", "MODEL_WEIGHTS_DIR")}
    env |= {"RECORD_FILE": str(record), "HF_MODEL": "Some/Other", "MODEL_WEIGHTS_DIR": "/nowhere"}
    r = subprocess.run(["bash", str(cur["stage"] / "run.sh"), "--port", "8123"], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    md = str(cur["stage"] / "model-dir")
    seen = json.loads(record.read_text())
    assert seen["argv"][seen["argv"].index("--model") + 1] == md and seen["argv"][-2:] == ["--port", "8123"]
    assert seen["env"] == {"HF_MODEL": md, "MODEL_WEIGHTS_DIR": md}
    assert not (cur["bundle"] / "prepare_model_dir_ran").exists()          # the swap did not run the bundle's script
    assert not (cur["bundle"] / "model-dir").exists()                       # and the bundle gained no model-dir
    assert bundle_listing(cur["bundle"]) == before


def test_the_unedited_script_would_have_run_the_bundles_step_the_check_above_depends_on(cur):
    """The bundle's own run.sh runs prepare_model_dir.py, so the previous test's sentinel can be written."""
    record = cur["tmp"] / "record.json"
    env = dict(os.environ) | {"RECORD_FILE": str(record), "HF_MODEL": str(cur["new"])}
    r = subprocess.run(["bash", str(cur["bundle"] / "run.sh")], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (cur["bundle"] / "prepare_model_dir_ran").exists()


# ---- the model directory ----------------------------------------------------------------------------------------

def test_configuration_files_come_from_the_bundles_base_config_not_the_nearest_snapshot(cur):
    assert run_prepare(cur["stage"]).returncode == 0
    md = cur["stage"] / "model-dir"
    for name in ("config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        assert (md / name).is_file() and not (md / name).is_symlink()
        assert (md / name).read_text() == BASE_CONFIG_FILES[name]


def test_the_new_models_files_replace_a_base_config_copy_of_the_same_name(cur):
    assert run_prepare(cur["stage"]).returncode == 0
    md = cur["stage"] / "model-dir"
    link = md / "generation_config.json"
    assert link.is_symlink() and link.read_text() == "new generation_config.json"
    for name in NEW_FILES:
        assert (md / name).is_symlink() and os.readlink(md / name) == os.path.realpath(cur["new"] / name)
    assert sorted(p.name for p in md.iterdir()) == sorted(NEW_FILES + ["config.json", "preprocessor_config.json",
                                                                       "video_preprocessor_config.json"])


def test_a_model_without_a_generation_config_gets_the_bundles_base_config_one(cur):
    (cur["new"] / "generation_config.json").unlink()
    r = run_prepare(cur["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    gen = cur["stage"] / "model-dir" / "generation_config.json"
    assert gen.is_file() and not gen.is_symlink() and gen.read_text() == BASE_CONFIG_FILES["generation_config.json"]
    s = sources(cur)
    assert s["sampling_defaults_from"] == "base" and s["eos_token_ids"] == [7, 8]
    entry = {f["name"]: f for f in s["files"]}["generation_config.json"]
    assert entry["source"] == "base" and entry["how"] == "copy" and entry["reason"] == "new_snapshot has none"
    assert "in neither snapshot" not in r.stdout


def test_the_record_says_where_the_configuration_files_came_from(cur):
    assert run_prepare(cur["stage"]).returncode == 0
    s = sources(cur)
    assert s["config_from"] == "bundle base_config/"
    by = {f["name"]: f for f in s["files"]}
    assert by["config.json"] == {"name": "config.json", "source": "base", "how": "copy"}
    assert by["tokenizer.json"]["source"] == "model" and by["generation_config.json"]["source"] == "model"
    assert s["weights"] == {"source": "model", "how": "link", "count": 2}


def test_the_vocabulary_check_reads_the_bundles_config_when_that_is_what_is_served(cur):
    put(cur["new"], "config.json", json.dumps({"text_config": {"vocab_size": 60}}))
    r = run_prepare(cur["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((cur["stage"] / "model-dir" / "config.json").read_text())["text_config"]["vocab_size"] == 60
    assert sources(cur)["vocab"] == {"model": 60, "base": 100, "trimmed": True}
    assert 'export QWEN36_DRAFTER=""' in run_text(cur)


def test_the_run_twice_gives_the_same_result(cur):
    assert run_prepare(cur["stage"]).returncode == 0
    first = run_text(cur)
    assert run_prepare(cur["stage"]).returncode == 0
    assert run_text(cur) == first


# ---- the drafter and the bundle's graft ----------------------------------------------------------------------------

def test_a_model_without_mtp_tensors_gets_the_drafter_and_device_sampling_cleared(cur):
    new_index(cur, ["model.embed_tokens.weight", "lm_head.weight"])
    r = run_prepare(cur["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    text = run_text(cur)
    assert 'export QWEN36_DRAFTER=""\n' in text and "sample_on_device_mode" not in text


def test_the_bundles_graft_is_named_as_not_applied(cur):
    new_index(cur, ["model.embed_tokens.weight", "lm_head.weight"])
    r = run_prepare(cur["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "graft.json" in r.stdout and "graft_mtp" in r.stdout


# ---- failures: exit 2, a message that names what is wrong, nothing built ------------------------------------------

def broken(cur, edit, named: list[str]):
    (cur["bundle"] / "run.sh").write_text(edit(CURRENT_RUN_SH))
    r = run_prepare(cur["stage"])
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    for n in named:
        assert n in out, (n, out)
    assert not (cur["stage"] / "run.sh").exists() and not (cur["stage"] / "model-dir").exists()
    return out


def test_a_command_that_names_neither_form_exits_2_and_names_both(cur):
    broken(cur, lambda s: s.replace('--model "$HERE/model-dir"', '--model "Other/Model"'),
           ["--model", '"Org/Nearest"', "$HERE/model-dir"])


def test_a_bundle_run_script_in_both_forms_at_once_exits_2(cur):
    broken(cur, lambda s: s.replace('--max_num_seqs 4', '--model "Org/Nearest" --max_num_seqs 4'), ["--model", "both"])


def test_two_model_arguments_in_the_current_form_exit_2(cur):
    broken(cur, lambda s: s.replace('--max_num_seqs 4', '--model "$HERE/model-dir" --max_num_seqs 4'), ["--model", "found 2"])


def test_a_current_bundle_without_the_prepare_model_dir_line_exits_2(cur):
    broken(cur, lambda s: s.replace('"$PYBIN" "$HERE/prepare_model_dir.py"\n', ""), ["prepare_model_dir.py", "found 0"])


def test_two_prepare_model_dir_lines_exit_2(cur):
    broken(cur, lambda s: s.replace('exec "${CMD[@]}"', '"$PYBIN" "$HERE/prepare_model_dir.py"\nexec "${CMD[@]}"'),
           ["prepare_model_dir.py", "found 2"])


def test_another_reference_to_the_bundles_model_dir_exits_2(cur):
    broken(cur, lambda s: s.replace("exec ", 'ls "$HERE/model-dir" >/dev/null\nexec '), ["model-dir", "bundle's own"])


def test_a_current_bundle_without_a_base_config_exits_2(cur):
    shutil.rmtree(cur["bundle"] / "base_config")
    r = run_prepare(cur["stage"])
    assert r.returncode == 2 and "base_config" in r.stdout + r.stderr
    assert not (cur["stage"] / "run.sh").exists() and not (cur["stage"] / "model-dir").exists()


def test_a_base_config_without_a_config_json_exits_2(cur):
    (cur["bundle"] / "base_config" / "config.json").unlink()
    r = run_prepare(cur["stage"])
    assert r.returncode == 2 and "config.json" in r.stdout + r.stderr
    assert not (cur["stage"] / "model-dir").exists()


# ---- the older form keeps working -----------------------------------------------------------------------------------

def test_the_older_form_still_copies_its_configuration_from_the_nearest_snapshot(cur):
    old = ('#!/usr/bin/env bash\nset -euo pipefail\nHERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
           f'export HF_MODEL="${{HF_MODEL:-{NEAREST}}}"\nexport MODEL_WEIGHTS_DIR="/bundle/model"\n'
           f'CMD=(python -m vllm.entrypoints.openai.api_server --model "{NEAREST}" "$@")\nexec "${{CMD[@]}}"\n')
    shutil.rmtree(cur["bundle"] / "base_config")           # an older bundle has none
    (cur["bundle"] / "run.sh").write_text(old)
    cfg = json.loads((cur["stage"] / "swap_config.json").read_text()) | {"nearest_model_id": NEAREST}
    (cur["stage"] / "swap_config.json").write_text(json.dumps(cfg))
    r = run_prepare(cur["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    md = cur["stage"] / "model-dir"
    assert (md / "config.json").read_text() == '{"snapshot": "base"}'
    assert sources(cur)["config_from"] == "nearest model's snapshot"
    text, mds = run_text(cur), str(md)
    assert f"--model {mds}" in text and f'export MODEL_WEIGHTS_DIR="{mds}"\n' in text
