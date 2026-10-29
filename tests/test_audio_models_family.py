# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Stable Audio 3 family beyond small-music: the planners know small-sfx and medium (and their bases), each pair shares one prompt, the stage 2 config
names the prompt of each run and asks for the decoder on the chip, and the hardware test passes both on to `chip_generate.py`."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from orchard import audio_plan as ap
from orchard import audio_runtime as ar

TEMPLATES = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates"
PAIRS = [("stabilityai/stable-audio-3-small-music", "stabilityai/stable-audio-3-small-music-base"),
         ("stabilityai/stable-audio-3-small-sfx", "stabilityai/stable-audio-3-small-sfx-base"),
         ("stabilityai/stable-audio-3-medium", "stabilityai/stable-audio-3-medium-base")]


@pytest.mark.parametrize("model, base", PAIRS)
def test_every_pair_is_known_and_runs_each_side_with_its_own_recipe(model, base):
    m, b = ap.library_name(model), ap.library_name(base)
    assert m and b and m != b
    assert (ap.settings_for(m)["steps"], ap.settings_for(m)["cfg_scale"], ap.settings_for(m)["sampler"]) == (8, 1.0, "pingpong")
    assert (ap.settings_for(b)["steps"], ap.settings_for(b)["cfg_scale"], ap.settings_for(b)["sampler"]) == (50, 7.0, "euler")


@pytest.mark.parametrize("model, base", PAIRS)
def test_a_model_and_its_base_share_the_prompt_the_control_compares(model, base):
    assert ap.settings_for(ap.library_name(model))["prompt"] == ap.settings_for(ap.library_name(base))["prompt"]


def test_the_sound_effects_pair_uses_a_sound_effects_prompt_and_the_music_models_keep_theirs():
    assert ap.settings_for("small-sfx")["prompt"] == ap.SFX_PROMPT != ap.PROMPT
    assert ap.settings_for("small-music")["prompt"] == ap.settings_for("medium")["prompt"] == ap.PROMPT


class FakePaths:
    src_dir = "/src"


def test_the_stage_2_config_names_each_runs_prompt_and_the_decoder_choice(tmp_path):
    plan = ap.chip_plan("stabilityai/stable-audio-3-small-sfx", "stabilityai/stable-audio-3-small-sfx-base")
    path = ap.write_chip_config(tmp_path, FakePaths(), "/hf", plan, repeats=2, warmup=1, chip_decoder=True)
    cfg = json.loads(path.read_text())
    assert cfg["chip_decoder"] is True
    assert {r["prompt"] for r in cfg["runs"]} == {ap.SFX_PROMPT}
    again = json.loads(ap.write_chip_config(tmp_path, FakePaths(), "/hf", plan).read_text())
    assert again["chip_decoder"] is False


def load_chip_test(tmp_path):
    d = tmp_path / "tpl"
    d.mkdir()
    shutil.copy(TEMPLATES / "audio_chip_test.py", d / "audio_chip_test.py")
    (d / "chip_generate.py").write_text(
        "import json, sys\nfrom pathlib import Path\nargv = sys.argv[1:]\nout = Path(argv[argv.index('--out') + 1])\nout.parent.mkdir(parents=True, exist_ok=True)\n"
        "open(out.parent / 'calls.jsonl', 'a').write(json.dumps(argv) + '\\n')\nout.mkdir()\n(out / 'manifest.json').write_text('{}')\n")
    spec = importlib.util.spec_from_file_location("audio_chip_test_family", d / "audio_chip_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_config(tmp_path, **over):
    mod = load_chip_test(tmp_path)
    runs = [{"name": "m", "model": "m", "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong", **over.pop("run", {})}]
    cfg = {"prompt": "default prompt", "duration": 10.0, "seed": 1, "sample_size": 100, "src_dir": str(tmp_path), "hf_home": str(tmp_path), "out_dir": "chip",
           "runs": runs, **over}
    path = tmp_path / "stage" / "audio_chip_config.json"
    path.parent.mkdir()
    path.write_text(json.dumps(cfg))
    assert mod.main(["--config", str(path)]) == 0
    return json.loads((tmp_path / "stage" / "chip" / "calls.jsonl").read_text().splitlines()[0])


def test_the_hardware_test_passes_the_runs_own_prompt_and_the_decoder_flag(tmp_path):
    argv = run_config(tmp_path, run={"prompt": "its own"}, chip_decoder=True)
    assert argv[argv.index("--prompt") + 1] == "its own" and "--chip-decoder" in argv


def test_without_a_run_prompt_or_the_flag_nothing_changes(tmp_path):
    argv = run_config(tmp_path)
    assert argv[argv.index("--prompt") + 1] == "default prompt" and "--chip-decoder" not in argv


def test_the_runtime_module_checks_the_decoder_of_a_model_directory(tmp_path):
    good = {"model": {"pretransform": {"config": {"decoder": {"type": "taae_v2", "config": {
        "out_channels": 512, "channels": 128, "c_mults": [6], "strides": [16], "latent_dim": 256, "transformer_depths": [6], "sinusoidal_blocks": [0],
        "differential": True, "mapping_style": "none", "dim_heads": 64, "variable_stride": True, "dyt": True, "chunk_size": 32, "conv_mapping": True,
        "chunk_midpoint_shift": True, "mask_noise": 0.01}}}}}}
    (tmp_path / "model_config.json").write_text(json.dumps(good))
    assert ar.decoder_problems(tmp_path) == []
    good["model"]["pretransform"]["config"]["decoder"]["config"]["differential"] = False
    (tmp_path / "model_config.json").write_text(json.dumps(good))
    assert ar.decoder_problems(tmp_path)
    (tmp_path / "model_config.json").write_text("not json")
    assert ar.decoder_problems(tmp_path)
