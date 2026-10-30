# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stock_config.json the supervisor drafts for stage 2 of a stock-runtime run (orchard/stock_runtime.py, draft_config), and the
supervisor's step that writes it and copies the template. Every fact in the file is one the run already holds."""
import json
from pathlib import Path

import pytest

from orchard import stock_runtime as sr

PARAMS = "/host/tt-metal/models/tt_transformers/model_params/Llama-3.1-8B-Instruct"


def delta(**stock):
    return {"model": "unsloth/Llama-3.1-8B-Instruct@abc", "class": "stock-runtime",
            "stock_runtime": {"name": "Llama-3.1-8B-Instruct", "params_b": 8.03, **stock}}


def draft(tmp_path, chips=2, **over):
    args = dict(run_dir=tmp_path / "run", delta=delta(), inputs={"model": "/hf/snap", "stock_params": PARAMS}, chips=chips,
                cache_root=tmp_path / "cache")
    args.update(over)
    return sr.draft_config(**args)


def test_a_two_chip_stage_drafts_the_whole_config(tmp_path):
    cfg, problems = draft(tmp_path)
    assert problems == [] and cfg == {
        "run_dir": str(tmp_path / "run"), "model_id": "unsloth/Llama-3.1-8B-Instruct@abc", "model_name": "Llama-3.1-8B-Instruct",
        "model_snapshot": "/hf/snap", "tt_metal_dir": "/host/tt-metal", "scratch_dir": str(tmp_path / "run" / "tt-metal-scratch"),
        "mesh_device": "P300", "chips": 2, "optimizations": "accuracy",
        "tt_cache": str(tmp_path / "cache" / "stock" / "llama-3.1-8b-instruct" / "P300"), "deadline_s": sr.STOCK_DEADLINE_S,
        "disable_tensor_cache": True}


def test_one_chip_is_p150(tmp_path):
    cfg, _ = draft(tmp_path, chips=1)
    assert cfg["mesh_device"] == "P150" and cfg["chips"] == 1 and cfg["tt_cache"].endswith("/P150")


def test_four_chips_are_p150x4(tmp_path):
    cfg, _ = draft(tmp_path, chips=4)
    assert cfg["mesh_device"] == "P150x4"


@pytest.mark.parametrize("chips", [0, 3, 5, True, None, "2"])
def test_a_chip_count_with_no_board_name_is_a_problem(tmp_path, chips):
    cfg, problems = draft(tmp_path, chips=chips)
    assert cfg is None and problems and "chip" in problems[0]


@pytest.mark.parametrize("drop", ["model", "stock_params"])
def test_a_missing_input_is_a_problem_and_nothing_is_drafted(tmp_path, drop):
    inputs = {"model": "/hf/snap", "stock_params": PARAMS}
    del inputs[drop]
    cfg, problems = draft(tmp_path, inputs=inputs)
    assert cfg is None and any(drop in p for p in problems)


def test_a_delta_that_is_not_the_stock_class_is_a_problem(tmp_path):
    cfg, problems = draft(tmp_path, delta={**delta(), "class": "weights-only"})
    assert cfg is None and problems and "stock-runtime" in problems[0]
    cfg, problems = draft(tmp_path, delta=None)
    assert cfg is None and problems


def test_a_params_directory_that_is_not_under_a_tt_metal_tree_is_a_problem(tmp_path):
    cfg, problems = draft(tmp_path, inputs={"model": "/hf/snap", "stock_params": "/x/y"})
    assert cfg is None and any("model_params" in p for p in problems)


def test_the_tt_metal_directory_is_three_levels_above_the_models_name_and_the_name_is_the_last_part(tmp_path):
    cfg, _ = draft(tmp_path, inputs={"model": "/hf/snap", "stock_params": "/a/b/models/tt_transformers/model_params/Qwen2.5-7B-Instruct"},
                   delta=delta(name="Qwen2.5-7B-Instruct"))
    assert cfg["tt_metal_dir"] == "/a/b" and cfg["model_name"] == "Qwen2.5-7B-Instruct"
    assert cfg["tt_cache"].endswith("/stock/qwen2.5-7b-instruct/P300")


def test_the_config_holds_no_token_and_every_path_is_absolute(tmp_path):
    cfg, _ = draft(tmp_path)
    for key in ("run_dir", "model_snapshot", "tt_metal_dir", "scratch_dir", "tt_cache"):
        assert Path(cfg[key]).is_absolute()
    assert not [k for k in cfg if "token" in k or "secret" in k or "key" in k]


def test_the_draft_turns_the_tensor_cache_off(tmp_path):
    """ttnn 0.80.0 on this box never returns from load_tensor_flatbuffer for a cached tensor of about 35 MB or more (found 2026-10-29, Llama-3.1-8B), and
    tt-transformers reaches it on the second KV-cache as_tensor. The draft says to run with no cache files."""
    cfg, _ = draft(tmp_path, chips=1)
    assert cfg["disable_tensor_cache"] is True
