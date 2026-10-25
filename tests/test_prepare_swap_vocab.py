# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""prepare_swap.py with a model whose vocabulary is smaller than the nearest model's (issue 56).

ProprietaryLegal/Qwen3.8-27B-CJK-Trim is Qwen3.8-27B with 66,533 vocabulary entries removed: the embedding and lm_head have
181,760 rows against 248,320 and every other tensor is the base's. On 2026-10-24 the base's 2-chip bundle served it only after
two changes to what this script builds. With the base's config.json (vocab_size 248320) the engine failed at load with a size
mismatch. With the model's own vocab_size the weights loaded and the speculative drafter's warm-up then raised IndexError,
because the drafter embeds token ids of the base's vocabulary. With the drafter off (and the host sampling) the model served
and agreed with the CPU reference on 30 of 32 tokens."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weights_swap_templates import (DRAFTER_LINE, drafter_lines, new_index, prep, put,  # noqa: E402,F401
                                         run_prepare, with_drafter)

BASE_CFG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"vocab_size": 248320, "hidden_size": 5120}}
MODEL_CFG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"vocab_size": 181760, "hidden_size": 5120}}
WITH_MTP = ["model.embed_tokens.weight", "lm_head.weight", "mtp.fc.weight", "model.layers.0.mlp.up_proj.weight"]


def configs(prep, base=BASE_CFG, model=MODEL_CFG):
    put(prep["base"], "config.json", json.dumps(base))
    put(prep["new"], "config.json", json.dumps(model))


def served_config(prep):
    return json.loads((prep["stage"] / "model-dir" / "config.json").read_text())


def sources(prep):
    return json.loads((prep["stage"] / "model-dir-sources.json").read_text())


def test_a_smaller_vocabulary_is_written_into_the_served_config(prep):
    configs(prep)
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    cfg = served_config(prep)
    assert cfg["text_config"]["vocab_size"] == 181760
    assert cfg["text_config"]["hidden_size"] == 5120 and cfg["architectures"] == BASE_CFG["architectures"]


def test_the_served_config_stays_a_real_file_and_the_base_snapshot_is_untouched(prep):
    configs(prep)
    before = (prep["base"] / "config.json").read_text()
    assert run_prepare(prep["stage"]).returncode == 0
    served = prep["stage"] / "model-dir" / "config.json"
    assert served.is_file() and not served.is_symlink()
    assert (prep["base"] / "config.json").read_text() == before
    assert json.loads((prep["base"] / "config.json").resolve().read_text())["text_config"]["vocab_size"] == 248320


def test_a_top_level_vocab_size_is_written_too_when_the_base_config_has_one(prep):
    configs(prep, base={"vocab_size": 248320, "text_config": {"vocab_size": 248320}},
            model={"text_config": {"vocab_size": 181760}})
    assert run_prepare(prep["stage"]).returncode == 0
    cfg = served_config(prep)
    assert cfg["vocab_size"] == 181760 and cfg["text_config"]["vocab_size"] == 181760


def test_a_model_config_that_keeps_vocab_at_the_top_level_is_read(prep):
    configs(prep, model={"vocab_size": 181760})
    assert run_prepare(prep["stage"]).returncode == 0
    assert served_config(prep)["text_config"]["vocab_size"] == 181760


def test_the_drafter_is_cleared_for_a_smaller_vocabulary_even_with_mtp_tensors(prep):
    configs(prep)
    with_drafter(prep)
    new_index(prep, WITH_MTP)
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert drafter_lines(prep) == ['export QWEN36_DRAFTER=""']
    assert "vocabulary" in r.stdout.lower() and "drafter" in r.stdout.lower()


def test_on_device_sampling_is_dropped_with_the_drafter_for_a_smaller_vocabulary(prep):
    configs(prep)
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace(
        "--max_model_len", "--additional-config '{\"tt\": {\"l1_small_size\": 24576, \"sample_on_device_mode\": \"decode_only\"}}' --max_model_len", 1))
    new_index(prep, WITH_MTP)
    assert run_prepare(prep["stage"]).returncode == 0
    assert "sample_on_device_mode" not in (prep["stage"] / "run.sh").read_text()


def test_the_same_vocabulary_changes_neither_the_config_nor_the_drafter(prep):
    configs(prep, model=BASE_CFG)
    with_drafter(prep)
    new_index(prep, WITH_MTP)
    assert run_prepare(prep["stage"]).returncode == 0
    assert served_config(prep) == BASE_CFG
    assert drafter_lines(prep) == [DRAFTER_LINE.strip(), 'export DFLASH_WEIGHTS="incoai/x@abc"'][:1]
    assert sources(prep)["vocab"] == {"model": 248320, "base": 248320, "trimmed": False}


def test_the_record_says_what_was_written(prep):
    configs(prep)
    assert run_prepare(prep["stage"]).returncode == 0
    assert sources(prep)["vocab"] == {"model": 181760, "base": 248320, "trimmed": True}


def test_a_model_config_with_no_vocab_size_changes_nothing(prep):
    configs(prep, model={"architectures": ["X"]})
    with_drafter(prep)
    new_index(prep, WITH_MTP)
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert served_config(prep) == BASE_CFG
    assert drafter_lines(prep)[0] == DRAFTER_LINE.strip()
    assert sources(prep)["vocab"] == {"model": None, "base": 248320, "trimmed": None}


def test_a_vocab_size_that_is_not_a_whole_number_is_ignored(prep):
    for bad in (True, "181760", 1.5, -3):
        configs(prep, model={"vocab_size": bad})
        assert run_prepare(prep["stage"]).returncode == 0
        assert served_config(prep) == BASE_CFG, bad
        assert sources(prep)["vocab"]["trimmed"] is None, bad


def test_a_larger_vocabulary_is_written_too(prep):
    configs(prep, model={"text_config": {"vocab_size": 250000}})
    assert run_prepare(prep["stage"]).returncode == 0
    assert served_config(prep)["text_config"]["vocab_size"] == 250000
    assert sources(prep)["vocab"]["trimmed"] is True


def test_grafting_the_nearest_models_head_into_a_different_vocabulary_is_refused(prep):
    configs(prep)
    cfg = json.loads((prep["stage"] / "swap_config.json").read_text())
    cfg["graft_mtp"] = True
    (prep["stage"] / "swap_config.json").write_text(json.dumps(cfg))
    r = run_prepare(prep["stage"])
    assert r.returncode == 2
    assert "vocabulary" in (r.stdout + r.stderr).lower()
    assert not (prep["stage"] / "model-dir").exists() and not (prep["stage"] / "run.sh").exists()


def test_a_container_run_with_no_bundle_still_gets_the_model_vocab_in_its_config(prep):
    configs(prep)
    cfg = json.loads((prep["stage"] / "swap_config.json").read_text())
    del cfg["bundle_dir"]
    cfg["package"] = "org/pkg"
    (prep["stage"] / "swap_config.json").write_text(json.dumps(cfg))
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert served_config(prep)["text_config"]["vocab_size"] == 181760
    assert sources(prep)["vocab"]["trimmed"] is True


def test_a_text_config_vocab_size_that_is_not_valid_is_not_replaced_by_the_top_level_one(prep):
    configs(prep, model={"vocab_size": 181760, "text_config": {"vocab_size": True}})
    assert run_prepare(prep["stage"]).returncode == 0
    assert served_config(prep) == BASE_CFG
    assert sources(prep)["vocab"] == {"model": None, "base": 248320, "trimmed": None}


def test_a_nearest_model_config_with_no_vocab_size_means_nothing_is_changed(prep):
    configs(prep, base={"architectures": ["X"]})
    with_drafter(prep)
    new_index(prep, WITH_MTP)
    assert run_prepare(prep["stage"]).returncode == 0
    assert served_config(prep) == {"architectures": ["X"]}
    assert drafter_lines(prep)[0] == DRAFTER_LINE.strip()
    assert sources(prep)["vocab"] == {"model": 181760, "base": None, "trimmed": None}


def test_the_key_is_written_only_where_the_base_config_has_it(prep):
    configs(prep)
    assert run_prepare(prep["stage"]).returncode == 0
    assert "vocab_size" not in served_config(prep)
