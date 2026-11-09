# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The drafter is switched off by the kind of drafter the bundle uses, and its repo leaves the run script and the package (Track C,
item C6 of the Qwen4 drop-in plan).

The Qwen3.8 bundles serve with `QWEN36_DRAFTER="dflash2"`: a separate drafter checkpoint (DFLASH_WEIGHTS, for example
incoai/Qwen3.8-27B-DFlash2) that reads the hidden states of the model it was trained on at fixed layers (its `target_layer_ids`). It
works for a fine-tune of the nearest model and for no other architecture. The earlier rule switched the drafter off only when the new
model had no `mtp.*` tensors, so a Qwen4 release that ships an MTP head would have kept a Qwen3.8 drafter on. `QWEN36_DRAFTER="mtp"`
uses the model's own MTP head and keeps the earlier rule. With the drafter off, the drafter's entry also leaves `TT_AUXILIARY_WEIGHTS`
(the run script downloads every entry before vLLM starts, and the package card names every repo in the environment as the drafter)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weights_swap_templates import DRAFTER_LINE, new_index, prep, put, run_prepare  # noqa: E402,F401

from orchard import package

QWEN38 = {"model_type": "qwen3_5_text", "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
          "head_dim": 256, "intermediate_size": 17408, "vocab_size": 248320, "layer_types": ["linear_attention"] * 3 + ["full_attention"]}
AUX = "incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4"
MTP_INDEX = ["model.embed_tokens.weight", "mtp.fc.weight", "mtp.layers.0.mlp.up_proj.weight"]


def bundle_with(prep, kind="dflash2", aux=AUX, dflash=AUX):
    run = prep["bundle"] / "run.sh"
    lines = f'export QWEN36_DRAFTER="{kind}"\n'
    if dflash is not None:
        lines += f'export DFLASH_WEIGHTS="{dflash}"\n'
    if aux is not None:
        lines += f'export TT_AUXILIARY_WEIGHTS="{aux}"\n'
    run.write_text(run.read_text().replace("CMD=(", lines + "CMD=(", 1))


def configs(prep, new: dict, base: dict = QWEN38, nested_base: bool = True):
    put(prep["base"], "config.json", json.dumps({"model_type": "qwen3_5", "text_config": base} if nested_base else base))
    put(prep["new"], "config.json", json.dumps(new))


def line(prep, name: str) -> list[str]:
    return [ln for ln in (prep["stage"] / "run.sh").read_text().splitlines() if f"{name}=" in ln and ln.lstrip().startswith("export")]


# ---- the decision ---------------------------------------------------------------------------------------------------------------

def test_a_fine_tune_with_an_mtp_head_keeps_the_dflash2_drafter(prep):
    bundle_with(prep)
    configs(prep, {**QWEN38, "architectures": ["Qwen3_5ForCausalLM"]})
    new_index(prep, MTP_INDEX)
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stderr
    assert line(prep, "QWEN36_DRAFTER") == ['export QWEN36_DRAFTER="dflash2"']
    assert line(prep, "TT_AUXILIARY_WEIGHTS") == [f'export TT_AUXILIARY_WEIGHTS="{AUX}"']


@pytest.mark.parametrize("change", [{"hidden_size": 4096}, {"num_hidden_layers": 20}, {"model_type": "qwen4_exp_text"},
                                    {"num_key_value_heads": 2}, {"intermediate_size": 14336}, {"head_dim": 128},
                                    {"num_attention_heads": 16}, {"layer_types": ["full_attention"] * 4}])
def test_another_architecture_switches_the_dflash2_drafter_off_even_with_an_mtp_head(prep, change):
    bundle_with(prep)
    configs(prep, {**QWEN38, **change})
    new_index(prep, MTP_INDEX)
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stderr
    assert line(prep, "QWEN36_DRAFTER") == ['export QWEN36_DRAFTER=""']
    key = next(iter(change))
    assert "DFlash2" in r.stdout and key in r.stdout
    assert line(prep, "TT_AUXILIARY_WEIGHTS") == []


def test_a_text_model_type_suffix_alone_is_not_another_architecture(prep):
    bundle_with(prep)
    configs(prep, {**QWEN38, "model_type": "qwen3_5"})
    new_index(prep, MTP_INDEX)
    assert run_prepare(prep["stage"]).returncode == 0
    assert line(prep, "QWEN36_DRAFTER") == ['export QWEN36_DRAFTER="dflash2"']


def test_head_dim_left_out_is_not_taken_from_the_widths(prep):
    """Qwen3.8 states head_dim 256 while hidden_size / num_attention_heads is 213, so a value worked out from the widths would call a
    fine-tune that leaves head_dim out another architecture."""
    bundle_with(prep)
    configs(prep, {k: v for k, v in QWEN38.items() if k != "head_dim"})
    new_index(prep, MTP_INDEX)
    assert run_prepare(prep["stage"]).returncode == 0
    assert line(prep, "QWEN36_DRAFTER") == ['export QWEN36_DRAFTER="dflash2"']


def test_the_last_drafter_line_is_the_drafter_the_bundle_uses(prep):
    bundle_with(prep, kind="dflash2")
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace('export QWEN36_DRAFTER="dflash2"', 'export QWEN36_DRAFTER="mtp"\nexport QWEN36_DRAFTER="dflash2"', 1))
    configs(prep, {**QWEN38, "num_hidden_layers": 20})
    new_index(prep, MTP_INDEX)
    r = run_prepare(prep["stage"])
    assert r.returncode == 2 and "QWEN36_DRAFTER" in r.stderr        # the DFlash2 drafter must go off, and two lines cannot be edited


def test_a_key_only_one_config_states_is_not_a_difference(prep):
    bundle_with(prep)
    configs(prep, {k: v for k, v in QWEN38.items() if k != "vocab_size"})
    new_index(prep, MTP_INDEX)
    assert run_prepare(prep["stage"]).returncode == 0
    assert line(prep, "QWEN36_DRAFTER") == ['export QWEN36_DRAFTER="dflash2"']


def test_the_mtp_drafter_uses_the_models_own_head_and_keeps_the_earlier_rule(prep):
    bundle_with(prep, kind="mtp", aux=None, dflash=None)
    configs(prep, {**QWEN38, "num_hidden_layers": 20})
    new_index(prep, MTP_INDEX)
    assert run_prepare(prep["stage"]).returncode == 0
    assert line(prep, "QWEN36_DRAFTER") == ['export QWEN36_DRAFTER="mtp"']
    new_index(prep, ["model.embed_tokens.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    assert line(prep, "QWEN36_DRAFTER") == ['export QWEN36_DRAFTER=""']


def test_configs_that_cannot_be_read_switch_a_dflash2_drafter_off(prep):
    bundle_with(prep)
    put(prep["new"], "config.json", "{not json")
    new_index(prep, MTP_INDEX)
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stderr
    assert line(prep, "QWEN36_DRAFTER") == ['export QWEN36_DRAFTER=""']


# ---- the auxiliary weights ------------------------------------------------------------------------------------------------------

def test_without_mtp_tensors_the_drafters_auxiliary_entry_goes_and_others_stay(prep):
    other = "acme/vision-tower@" + "1" * 40
    bundle_with(prep, aux=f"{AUX},{other}")
    configs(prep, QWEN38)
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stderr
    assert line(prep, "TT_AUXILIARY_WEIGHTS") == [f'export TT_AUXILIARY_WEIGHTS="{other}"']
    assert "TT_AUXILIARY_WEIGHTS" in r.stdout


def test_an_auxiliary_entry_with_another_revision_of_the_drafter_repo_goes_too(prep):
    bundle_with(prep, aux="incoai/Qwen3.8-27B-DFlash2", dflash=AUX)
    configs(prep, QWEN38)
    new_index(prep, ["model.embed_tokens.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    assert line(prep, "TT_AUXILIARY_WEIGHTS") == []


def test_without_a_drafter_weights_line_the_auxiliary_weights_are_left_and_reported(prep):
    bundle_with(prep, dflash=None)
    configs(prep, QWEN38)
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stderr
    assert line(prep, "TT_AUXILIARY_WEIGHTS") == [f'export TT_AUXILIARY_WEIGHTS="{AUX}"']
    assert "TT_AUXILIARY_WEIGHTS" in r.stdout and "DFLASH_WEIGHTS" in r.stdout


def test_two_auxiliary_lines_are_an_error(prep):
    bundle_with(prep)
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace("CMD=(", f'export TT_AUXILIARY_WEIGHTS="{AUX}"\nCMD=(', 1))
    configs(prep, QWEN38)
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 2 and "TT_AUXILIARY_WEIGHTS" in r.stderr


def test_with_the_drafter_on_the_auxiliary_weights_stay(prep):
    bundle_with(prep)
    configs(prep, QWEN38)
    new_index(prep, MTP_INDEX)
    assert run_prepare(prep["stage"]).returncode == 0
    assert line(prep, "TT_AUXILIARY_WEIGHTS") == [f'export TT_AUXILIARY_WEIGHTS="{AUX}"']


# ---- the package ---------------------------------------------------------------------------------------------------------------

ENV = {"QWEN36_DRAFTER": "dflash2", "DFLASH_WEIGHTS": AUX, "QWEN36_DFLASH_BLOCK": "8", "TT_AUXILIARY_WEIGHTS": AUX,
       "QWEN_SDPA_KV_DTYPE": "bf4"}


def test_a_package_with_the_drafter_off_names_no_drafter_repo():
    env = package.serving_env(ENV, drafter_off=True)
    assert package.aux_repos(env, nearest_model="Qwen/Qwen3.8-27B") == []
    assert "TT_AUXILIARY_WEIGHTS" not in env and env["QWEN36_DRAFTER"] == "" and env["QWEN_SDPA_KV_DTYPE"] == "bf4"


def test_a_package_with_the_drafter_off_carries_no_auxiliary_weights_at_all():
    """The card names every repo in the package environment as the drafter (package.aux_repos), so a drafter-off package carries none."""
    other = "acme/vision-tower@" + "1" * 40
    env = package.serving_env({**ENV, "TT_AUXILIARY_WEIGHTS": f"{AUX}, {other}"}, drafter_off=True)
    assert "TT_AUXILIARY_WEIGHTS" not in env


def test_with_the_drafter_on_the_package_environment_is_unchanged():
    assert package.serving_env(ENV, drafter_off=False) == ENV


def test_the_package_check_refuses_a_drafter_repo_left_in_the_auxiliary_weights():
    env = {k: v for k, v in package.serving_env(ENV, drafter_off=True).items()}
    assert package.serving_problems("", env, drafter_off=True, host_sampling=False) == []
    left = {**env, "TT_AUXILIARY_WEIGHTS": AUX}
    problems = package.serving_problems("", left, drafter_off=True, host_sampling=False)
    assert any("incoai/Qwen3.8-27B-DFlash2" in p for p in problems), problems
    in_run_sh = f'export TT_AUXILIARY_WEIGHTS="{AUX}"\n'
    problems = package.serving_problems(in_run_sh, env, drafter_off=True, host_sampling=False)
    assert any("TT_AUXILIARY_WEIGHTS" in p for p in problems), problems
    assert package.serving_problems(in_run_sh, left, drafter_off=False, host_sampling=False) == []
