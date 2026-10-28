# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The configuration guard of the Stable Audio 3 chip path (`config_problems` in orchard/skills/audio-chip-templates/chip_dit.py).

The chip code implements one network configuration. A model whose `model_config.json` differs in a setting the chip code relies on, or carries a
setting the code has never seen, must be refused before a chip is opened."""
import copy
import glob
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates" / "chip_dit.py"
spec = importlib.util.spec_from_file_location("chip_dit_config_under_test", MODULE)
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)

REAL_CONFIG = {"model": {"diffusion": {
    "type": "dit", "diffusion_objective": "rf_denoiser",
    "config": {"io_channels": 256, "embed_dim": 1024, "depth": 20, "num_heads": 16, "cond_token_dim": 768, "global_cond_dim": 768,
               "local_add_cond_dim": 257, "global_cond_type": "adaLN", "timestep_features_type": "expo", "timestep_features_logsnr": False,
               "attn_kwargs": {"qk_norm": "rms", "differential": False}, "norm_type": "rms_norm", "norm_kwargs": {"force_fp32": True},
               "ff_kwargs": {"mult": 4.0}, "num_memory_tokens": 64}}}}


def with_change(path, value):
    c = copy.deepcopy(REAL_CONFIG)
    node = c
    for part in path[:-1]:
        node = node[part]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return c


def test_the_real_small_music_config_is_supported():
    assert cd.config_problems(REAL_CONFIG) == []


def test_both_objectives_of_the_model_pair_are_supported():
    assert cd.config_problems(with_change(["model", "diffusion", "diffusion_objective"], "rectified_flow")) == []


@pytest.mark.parametrize("path, value", [
    (["model", "diffusion", "type"], "dit_v2"),
    (["model", "diffusion", "diffusion_objective"], "v"),
    (["model", "diffusion", "config", "global_cond_type"], "prepend"),
    (["model", "diffusion", "config", "attn_kwargs", "differential"], True),
    (["model", "diffusion", "config", "attn_kwargs", "qk_norm"], "l2"),
    (["model", "diffusion", "config", "norm_type"], "layer_norm"),
    (["model", "diffusion", "config", "num_memory_tokens"], 32),
    (["model", "diffusion", "config", "num_heads"], 8),
    (["model", "diffusion", "config", "ff_kwargs", "mult"], 2.0),
    (["model", "diffusion", "config", "timestep_features_logsnr"], True),
    (["model", "diffusion", "config", "conformer"], True),                      # a key the path has never seen
    (["model", "diffusion", "config", "sliding_window"], [128, 128]),
    (["model", "diffusion", "config", "embed_dim"], KeyError),                  # a required key missing
    (["model", "diffusion", "config", "embed_dim"], 1000),                      # head dimension 62
])
def test_a_config_the_path_does_not_implement_is_refused(path, value):
    assert cd.config_problems(with_change(path, value)), f"{path} = {value!r} was accepted"


def test_a_config_without_the_diffusion_section_is_refused_and_does_not_crash():
    assert cd.config_problems({}) and cd.config_problems({"model": 3}) and cd.config_problems({"model": {"diffusion": []}})


def test_the_cached_small_music_configs_are_supported_when_present():
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--stabilityai--stable-audio-3-small-music*/snapshots/*/model_config.json"))
    if not hits:
        pytest.skip("the model is not in the local cache")
    for h in hits:
        assert cd.config_problems(json.loads(Path(h).read_text())) == [], h


def test_a_head_count_other_than_16_is_refused_even_when_the_head_dimension_is_64():
    """embed 512 with 8 heads has head dimension 64, so only the head-count rule can refuse it."""
    c = with_change(["model", "diffusion", "config", "embed_dim"], 512)
    c["model"]["diffusion"]["config"]["num_heads"] = 8
    problems = cd.config_problems(c)
    assert any("num_heads is 8" in p for p in problems) and not any("head dimension" in p for p in problems)


def test_a_head_dimension_other_than_64_is_refused_even_when_the_head_count_is_16():
    problems = cd.config_problems(with_change(["model", "diffusion", "config", "embed_dim"], 2048))
    assert any("head dimension is 128" in p for p in problems) and not any("num_heads" in p for p in problems)


@pytest.mark.parametrize("config", [3, [], "x", None])
def test_a_diffusion_config_that_is_not_an_object_is_refused(config):
    problems = cd.config_problems({"model": {"diffusion": {"type": "dit", "diffusion_objective": "rf_denoiser", "config": config}}})
    assert problems == ["model.diffusion and its config must be objects"]
