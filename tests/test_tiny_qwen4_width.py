# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The width presets of the tiny Qwen4 rehearsal checkpoint (scripts/tiny_qwen4/make_checkpoint.py, `--width`, `--set`, `--max-gb`).
PROVISIONAL: the full-size dimensions are the public readings in docs/superpowers/specs/2026-11-02-tiny-qwen4-rehearsal-design.md,
the weights are random.

What this file pins:
  - the default output (`--width tiny`, no `--set`) is byte-identical to what the script wrote before the presets existed;
  - the `27b` preset and every override keep the widths a 2-die and a 4-die mesh can tile, and a bad width is refused with the key named;
  - the size estimate is printed and checked against `--max-gb` and the free space before anything is written.

The config, override and size-rule tests need no torch. The tests that count parameters or build a checkpoint use the reference
interpreter (ORCHARD_REFERENCE_PYTHON) and skip without it. No device, no ttnn, no network; checkpoints go to pytest temp directories."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from orchard import hazards as hz
from orchard import model_config as mc
from test_ops_inventory_stub import PY, needs_real_forward

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "tiny_qwen4" / "make_checkpoint.py"
FLASH = REPO / "tests" / "fixtures" / "model-configs" / "qwen3.8-flash-next.json"
spec = importlib.util.spec_from_file_location("make_checkpoint_width", SCRIPT)
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)

# sha256 of what the script wrote for `--seed 11 --no-reference` BEFORE the width options existed (torch 2.14.1+cpu, transformers 5.19.0,
# tokenizers 0.23.2). config.json, generation_config.json and prompts.json come from this repository's code alone. The weights and the
# tokenizer files also depend on those three packages, so they are compared only when the installed versions are the pinned ones.
GOLDEN_PURE = {
    "config.json": "dd09e6e8bc3c7032923635d687acda8e37ad0eba32fa185b812090077b7ba895",
    "generation_config.json": "5885bd07e4d7f1594c590d2c8c6f7157671b813f9682166ac74263e546f6dd54",
    "prompts.json": "e4a099f549f297cafea76dbefae7463c1a194b8d532fe844d8b4e14c6ce1a224",
}
GOLDEN_STACK = {
    "model.safetensors": "a6aaf206443045ba8b07030de01172f41edebd9686eb1affda9faf53a3a39a4c",
    "tokenizer.json": "69a379b8017577ef95013153771575c7c34a1fddcf07a41b525837d02668cfdd",
    "tokenizer_config.json": "9ac2a64442cce9ca46ee62d99e12b9973e367c7213f67645c3bc71da7b8c26fd",
}
GOLDEN_STACK_VERSIONS = ("2.14.1+cpu", "5.19.0", "0.23.2")

SCALED_27B = {
    "hidden_size": 5120, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "hc_lowrank": 640, "indexer_budget": 2048, "indexer_head_dim": 128, "max_position_embeddings": 4096,
}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_cli(*args, py=None):
    return subprocess.run([py or PY, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


# ---- the default is unchanged (no torch) ------------------------------------------------------------------------

def test_the_default_width_is_tiny_and_no_option_changes_the_built_config():
    assert mk.WIDTHS == ("tiny", "27b")
    plain = mk.build_config("dense", 4)
    assert mk.build_config("dense", 4, width="tiny") == plain
    assert mk.build_config("dense", 4, width="tiny", overrides={}) == plain
    assert mk.build_config("dense", 4, width="tiny", overrides=None) == plain


def test_the_default_config_prompts_and_generation_config_are_byte_identical_to_the_script_before_the_presets(tmp_path):
    cfg = mk.build_config("dense", 4)
    mk.write_json(tmp_path / "config.json", {**cfg, "seed": 11})
    mk.write_json(tmp_path / "generation_config.json", {"do_sample": False, "eos_token_id": mk.SPECIAL_IDS["<eos>"],
                                                        "pad_token_id": mk.SPECIAL_IDS["<pad>"], "bos_token_id": mk.SPECIAL_IDS["<bos>"]})
    mk.write_json(tmp_path / "prompts.json", {"prompts": mk.make_prompts(11),
                                              "note": "token ids in the byte-token range; no tokenizer is needed"})
    for name, digest in GOLDEN_PURE.items():
        assert sha(tmp_path / name) == digest, name


def test_the_tiny_default_is_not_held_to_the_scaled_width_rules():
    """The tiny model has hidden size 64 and two linear key heads. The tile rules are for the scaled configs; they must not reject today's."""
    cfg = mk.build_config("dense", 4)
    assert cfg["hidden_size"] == 64 and cfg["linear_num_key_heads"] == 2
    with pytest.raises(ValueError):
        mk.check_widths(cfg)                                 # the rules would reject it, so build_config must not apply them by default


# ---- the 27b preset (no torch) -----------------------------------------------------------------------------------

def test_the_27b_preset_has_the_full_size_widths_in_four_layers():
    cfg = mk.build_config("dense", 4, width="27b")
    for key, value in SCALED_27B.items():
        assert cfg[key] == value, key
    assert cfg["intermediate_size"] == 14336 and cfg["hidden_act"] == "silu"
    assert cfg["num_hidden_layers"] == 4 and cfg["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
    assert not [k for k in mk.EXPERT_KEYS + ("mlp_only_layers",) if k in cfg], "a dense 27b config must carry no expert key"
    assert cfg["hc_count"] == 4 and cfg["indexer_n_heads"] == 4 and cfg["indexer_kv_heads"] == 1
    assert cfg["ngram_vocab_size_base"] <= 10_000            # the n-gram table stays small: the full-size one is 102 GB


def test_the_27b_preset_is_the_tiny_config_with_only_the_listed_widths_changed():
    tiny, big = mk.build_config("dense", 4), mk.build_config("dense", 4, width="27b")
    changed = {k for k in big if big[k] != tiny[k]}
    assert changed == set(SCALED_27B) | {"intermediate_size"}, changed ^ (set(SCALED_27B) | {"intermediate_size"})


@pytest.mark.parametrize("variant", ["moe", "mixed"])
def test_the_27b_moe_and_mixed_configs_carry_the_full_size_expert_widths(variant):
    cfg = mk.build_config(variant, 4, width="27b")
    assert (cfg["num_experts"], cfg["num_experts_per_tok"]) == (512, 10)
    assert cfg["moe_intermediate_size"] == 640 == cfg["shared_expert_intermediate_size"]
    assert ("mlp_only_layers" in cfg) == (variant == "mixed")
    if variant == "moe":
        assert "intermediate_size" not in cfg                 # a MoE-only config has no dense width to scale (and must not gain one)


@pytest.mark.parametrize("variant", ["dense", "moe", "mixed"])
@pytest.mark.parametrize("dies", [2, 4])
def test_no_stage_0_division_hazard_fires_on_the_27b_config_at_two_or_four_dies(tmp_path, variant, dies):
    p = tmp_path / f"{variant}.json"
    p.write_text(json.dumps(mk.build_config(variant, 4, width="27b")))
    scan = hz.scan_model_hazards(mc.read_model_config(p), mc.read_model_config(FLASH), dies=dies)
    bad = [h.name for h in scan.hazards if "not_divisible" in h.name or "not_whole_tiles" in h.name or "tile" in h.name]
    assert bad == [], bad


# ---- overrides (no torch) ----------------------------------------------------------------------------------------

def test_an_override_applies_after_the_preset_and_the_result_is_checked():
    cfg = mk.build_config("dense", 4, width="27b", overrides={"hidden_size": 2560, "intermediate_size": 7168})
    assert cfg["hidden_size"] == 2560 and cfg["intermediate_size"] == 7168 and cfg["head_dim"] == 256
    cfg = mk.build_config("dense", 4, width="tiny", overrides=dict(hidden_size=256, intermediate_size=512, vocab_size=512,
                                                                   linear_num_key_heads=4, linear_num_value_heads=12))
    assert cfg["vocab_size"] == 512 and cfg["hidden_size"] == 256 and cfg["head_dim"] == 32
    with pytest.raises(ValueError, match="hidden_size"):               # one key on the tiny base leaves the rest of it off the tiles
        mk.build_config("dense", 4, width="tiny", overrides={"vocab_size": 512})


@pytest.mark.parametrize("key", ["num_hidden_layers", "model_type", "ple_layer_ids", "layer_types", "tie_word_embeddings", "bogus", "seed"])
def test_a_key_that_is_not_a_width_is_refused_by_name(key):
    with pytest.raises(ValueError, match=key):
        mk.build_config("dense", 4, width="27b", overrides={key: 64})


@pytest.mark.parametrize("key", ["num_experts", "moe_intermediate_size", "shared_expert_intermediate_size", "num_experts_per_tok"])
def test_an_expert_override_on_a_dense_config_is_refused_because_transformers_would_then_build_a_moe_layer(key):
    with pytest.raises(ValueError, match=key):
        mk.build_config("dense", 4, width="tiny", overrides={key: 64})


def test_a_dense_width_override_on_a_moe_only_config_is_refused():
    with pytest.raises(ValueError, match="intermediate_size"):
        mk.build_config("moe", 4, width="tiny", overrides={"intermediate_size": 256})


@pytest.mark.parametrize("value", [2.5, "5120", True, False, None, [1], {"a": 1}, 0, -32, float("nan"), float("inf")])
def test_an_override_value_must_be_a_positive_whole_number(value):
    with pytest.raises(ValueError, match="hidden_size"):
        mk.build_config("dense", 4, width="tiny", overrides={"hidden_size": value})


@pytest.mark.parametrize("key,value", [("hidden_size", 10 ** 20), ("vocab_size", 1 << 21), ("head_dim", (1 << 20) + 32),
                                       ("num_experts", 4097), ("num_attention_heads", 1 << 22)])
def test_an_override_above_the_limit_is_refused_before_a_model_with_that_shape_is_built(key, value):
    """10**20 columns tile (it is a multiple of 128) and would reach the meta-device count; the limit stops it first (a size, bounded before arithmetic)."""
    variant = "moe" if key == "num_experts" else "dense"
    with pytest.raises(ValueError, match=f"override {key}: .* above the limit"):
        mk.build_config(variant, 4, width="27b", overrides={key: value})


def test_the_limits_themselves_are_allowed_where_the_width_rules_pass():
    cfg = mk.build_config("moe", 4, width="27b", overrides={"num_experts": 4096})
    assert cfg["num_experts"] == 4096
    cfg = mk.build_config("dense", 4, width="27b", overrides={"vocab_size": 1 << 20})
    assert cfg["vocab_size"] == 1 << 20


@pytest.mark.parametrize("key,value", [("hidden_size", 5120.0), ("head_dim", 256.0), ("indexer_kv_heads", True), ("indexer_n_heads", 4.0)])
def test_an_override_of_the_wrong_type_is_refused_even_when_its_number_would_tile(key, value):
    """5120.0 and True tile as numbers and would pass every width rule, then reach config.json as `5120.0` or `true`, which transformers
    takes for a different type. The type check has to refuse them before the width rules see them."""
    with pytest.raises(ValueError, match=f"override {key}"):
        mk.build_config("dense", 4, width="27b", overrides={key: value})


@pytest.mark.parametrize("overrides,fragment", [
    ({"hidden_size": 5100}, "hidden_size"),                       # not a multiple of 32 x 4 dies
    ({"hidden_size": 2592}, "hidden_size"),                       # a multiple of 32 and not of 128
    ({"intermediate_size": 14340}, "intermediate_size"),
    ({"intermediate_size": 14400}, "intermediate_size"),          # 14400 = 112.5 x 128
    ({"vocab_size": 248321}, "vocab_size"),
    ({"vocab_size": 248352}, "vocab_size"),                       # a multiple of 32 and not of 128
    ({"vocab_size": 128}, "vocab_size"),                          # tiles, but too small to hold the 260 special and byte tokens
    ({"vocab_size": 256}, "vocab_size"),
    ({"head_dim": 250}, "head_dim"),
    ({"head_dim": 16}, "head_dim"),
    ({"linear_key_head_dim": 100}, "linear_key_head_dim"),
    ({"linear_value_head_dim": 48}, "linear_value_head_dim"),
    ({"indexer_head_dim": 100}, "indexer_head_dim"),
    ({"num_attention_heads": 22}, "num_attention_heads"),         # 22 does not divide by 4 dies
    ({"num_attention_heads": 26}, "num_attention_heads"),
    ({"num_attention_heads": 22, "num_key_value_heads": 2}, "num_attention_heads"),   # a multiple of the 2 KV heads, so only the 4-die rule fires
    ({"num_attention_heads": 26, "num_key_value_heads": 1}, "num_attention_heads"),
    ({"num_key_value_heads": 3}, "num_key_value_heads"),          # neither divides nor is divided by 4
    ({"num_key_value_heads": 8, "num_attention_heads": 28}, "num_attention_heads"),   # 28 heads do not divide by 8 kv heads
    ({"linear_num_key_heads": 6}, "linear_num_key_heads"),
    ({"linear_num_value_heads": 50}, "linear_num_value_heads"),
    ({"linear_num_key_heads": 16, "linear_num_value_heads": 40}, "linear_num_value_heads"),   # 40 is not a multiple of 16 key heads
    ({"indexer_n_heads": 2}, "indexer_n_heads"),                  # 2 heads trip qsa_heads_not_divisible_by_dies
    ({"indexer_n_heads": 6}, "indexer_n_heads"),
    ({"indexer_kv_heads": 3}, "indexer_kv_heads"),                # neither divides nor is divided by 4 dies (review of PR 434, round 1)
    ({"indexer_kv_heads": 6}, "indexer_kv_heads"),
    ({"indexer_kv_heads": 5}, "indexer_kv_heads"),
])
def test_a_width_a_two_or_four_die_mesh_cannot_tile_is_refused_with_the_key_named(overrides, fragment):
    with pytest.raises(ValueError, match=fragment):
        mk.build_config("dense", 4, width="27b", overrides=overrides)


@pytest.mark.parametrize("kv", [1, 2, 4, 8, 12, 16])
def test_an_indexer_kv_head_count_that_splits_or_replicates_over_the_dies_is_accepted(kv):
    """The same rule as num_key_value_heads: a count that divides by the 4 dies splits, a count that divides 4 replicates."""
    assert mk.build_config("dense", 4, width="27b", overrides={"indexer_kv_heads": kv})["indexer_kv_heads"] == kv


def test_the_indexer_kv_head_rule_names_the_two_ways_out():
    with pytest.raises(ValueError) as err:
        mk.build_config("dense", 4, width="27b", overrides={"indexer_kv_heads": 3})
    assert "indexer_kv_heads 3" in str(err.value) and "divides" in str(err.value) and "replication" in str(err.value)


def test_the_moe_widths_are_checked_when_the_variant_has_them():
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        mk.build_config("moe", 4, width="27b", overrides={"moe_intermediate_size": 650})
    with pytest.raises(ValueError, match="shared_expert_intermediate_size"):
        mk.build_config("mixed", 4, width="27b", overrides={"shared_expert_intermediate_size": 650})
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        mk.build_config("moe", 4, width="27b", overrides={"num_experts_per_tok": 600})


def test_an_unknown_width_is_refused():
    with pytest.raises(ValueError, match="width"):
        mk.build_config("dense", 4, width="70b")


def test_set_options_parse_to_whole_numbers_and_refuse_a_bad_form():
    assert mk.parse_overrides(["hidden_size=2560", "vocab_size=512"]) == {"hidden_size": 2560, "vocab_size": 512}
    assert mk.parse_overrides([]) == {} and mk.parse_overrides(None) == {}
    for bad in (["hidden_size"], ["=5"], ["hidden_size=abc"], ["hidden_size=2.5"], ["hidden_size=5120", "hidden_size=2560"], ["a=b=c"], ["hidden_size=1e3"]):
        with pytest.raises(ValueError):
            mk.parse_overrides(bad)


# ---- the size rule (no torch) ------------------------------------------------------------------------------------

GB = 10 ** 9


def test_the_estimate_over_max_gb_is_refused_and_the_message_gives_both_numbers():
    with pytest.raises(mk.SizeRefused) as err:
        mk.check_size(13 * GB, 12, 10_000 * GB)
    assert "13.0 GB" in str(err.value) and "--max-gb 12" in str(err.value)
    mk.check_size(12 * GB, 12, 10_000 * GB)                  # exactly the limit is allowed
    mk.check_size(int(8.3 * GB), 12, 10_000 * GB)


def test_the_estimate_over_the_free_space_minus_the_margin_is_refused():
    assert mk.SPACE_MARGIN_GB == 40 and mk.DEFAULT_MAX_GB == 12
    with pytest.raises(mk.SizeRefused) as err:
        mk.check_size(8 * GB, 100, 47 * GB)                  # 47 - 40 = 7 GB usable
    assert "free" in str(err.value) and "40 GB" in str(err.value) and "7.0 GB" in str(err.value)
    mk.check_size(7 * GB, 100, 47 * GB)                      # fits exactly
    with pytest.raises(mk.SizeRefused):
        mk.check_size(1, 100, 40 * GB)                       # no usable space at all
    with pytest.raises(mk.SizeRefused):
        mk.check_size(1, 100, 10 * GB)                       # free space below the margin must not go negative and pass


def test_the_size_check_refuses_a_bad_limit_or_a_bad_estimate():
    for bad_limit in (0, -1, float("nan"), float("inf"), True, "12", None):
        with pytest.raises(ValueError, match="max-gb"):
            mk.check_size(GB, bad_limit, 10_000 * GB)
    for bad_size in (-1, float("nan"), True, "1", None, 1.5):
        with pytest.raises(ValueError, match="estimate"):
            mk.check_size(bad_size, 12, 10_000 * GB)


def test_free_bytes_reads_the_nearest_existing_parent(tmp_path):
    assert mk.free_bytes(tmp_path / "not" / "yet" / "made") > 0


def fake_torch_stack(monkeypatch, estimate_total, free):
    """Replace the stack, the estimate and the free-space read, so the size rule can be driven without torch and the run stops at the
    first write when the rule lets it through."""
    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop()
    monkeypatch.setattr(mk, "import_stack", lambda: (None, None))
    monkeypatch.setattr(mk, "estimate_bytes", lambda torch, transformers, cfg, **k: {"weights": estimate_total, "tokenizer": 0,
                                                                                    "total": estimate_total, "parameters": 0})
    monkeypatch.setattr(mk, "free_bytes", lambda path: free)
    monkeypatch.setattr(mk, "write_weights", stop)
    return Stop


def test_the_size_rule_refuses_before_the_output_directory_exists(tmp_path, monkeypatch, capsys):
    fake_torch_stack(monkeypatch, 30 * GB, 10_000 * GB)
    out = tmp_path / "big"
    assert mk.main(["--out", str(out), "--width", "27b"]) == 2
    err = capsys.readouterr().err
    assert "30.0 GB" in err and "--max-gb 12" in err
    assert not out.exists()


def test_a_raised_max_gb_lets_the_run_through_to_the_first_write(tmp_path, monkeypatch):
    stop = fake_torch_stack(monkeypatch, 30 * GB, 10_000 * GB)
    with pytest.raises(stop):
        mk.main(["--out", str(tmp_path / "big"), "--width", "27b", "--max-gb", "40"])


def test_the_free_space_margin_refuses_even_under_max_gb(tmp_path, monkeypatch, capsys):
    fake_torch_stack(monkeypatch, 8 * GB, 45 * GB)
    assert mk.main(["--out", str(tmp_path / "big"), "--width", "27b", "--max-gb", "40"]) == 2
    assert "free space" in capsys.readouterr().err and not (tmp_path / "big").exists()


class FakeTorch:
    """Just what write_weights touches: no_grad, bfloat16."""
    bfloat16 = "bf16"

    class no_grad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False


class FakeModel:
    """A model whose save_pretrained writes model.safetensors into the directory it is given, and records where that was."""
    saved_to: list = []
    shard_size = None

    def parameters(self):
        return []

    def to(self, dtype):
        return self

    def save_pretrained(self, directory, safe_serialization=True, max_shard_size=None):
        FakeModel.saved_to.append(Path(directory))
        FakeModel.shard_size = max_shard_size
        (Path(directory) / "model.safetensors").write_bytes(b"weights")
        (Path(directory) / "config.json").write_text("{}")


def test_the_weights_are_written_beside_the_output_and_moved_not_copied(tmp_path, monkeypatch):
    """Review of PR 434, round 1: `save_pretrained` wrote to the system temp directory and the file was copied into `out`, so a
    checkpoint needed its size twice on one filesystem, and the temp filesystem was never checked at all. The size rule budgets the
    filesystem of `out` once, so the only temporary copy allowed is one on that filesystem that is renamed into place."""
    FakeModel.saved_to = []
    monkeypatch.setattr(mk, "build_model", lambda *a, **k: FakeModel())

    def no_copy(*a, **k):
        raise AssertionError("the weights were copied; a rename keeps one copy on the checked filesystem")
    monkeypatch.setattr(mk.shutil, "copy", no_copy)
    monkeypatch.setattr(mk.shutil, "copyfile", no_copy)
    monkeypatch.setattr(mk.shutil, "copy2", no_copy)
    out = tmp_path / "ckpt"
    out.mkdir()
    mk.write_weights(FakeTorch, None, {}, 1, out, 1.0)
    assert (out / "model.safetensors").read_bytes() == b"weights"
    assert len(FakeModel.saved_to) == 1 and FakeModel.saved_to[0].parent == out         # the filesystem the size rule checked
    assert sorted(p.name for p in out.iterdir()) == ["model.safetensors"]                 # the temporary directory is gone
    assert FakeModel.shard_size >= 1 << 40                                                # one file: the default 50 GB shard size split a bigger checkpoint


def test_a_failed_save_leaves_no_temporary_directory_in_the_output(tmp_path, monkeypatch):
    class Boom(FakeModel):
        def save_pretrained(self, directory, safe_serialization=True, max_shard_size=None):
            (Path(directory) / "model.safetensors").write_bytes(b"half")
            raise OSError(28, "No space left on device")
    monkeypatch.setattr(mk, "build_model", lambda *a, **k: Boom())
    out = tmp_path / "ckpt"
    out.mkdir()
    with pytest.raises(OSError):
        mk.write_weights(FakeTorch, None, {}, 1, out, 1.0)
    assert list(out.iterdir()) == []


def test_the_estimate_is_printed_before_anything_is_written(tmp_path, monkeypatch, capsys):
    stop = fake_torch_stack(monkeypatch, 8 * GB, 10_000 * GB)
    with pytest.raises(stop):
        mk.main(["--out", str(tmp_path / "ok"), "--width", "27b"])
    assert "estimated size 8.0 GB" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "abc"])
def test_a_bad_max_gb_is_refused_before_any_work(tmp_path, monkeypatch, capsys, value):
    fake_torch_stack(monkeypatch, 1, 10_000 * GB)
    if value == "abc":
        with pytest.raises(SystemExit):                        # argparse's own type error
            mk.main(["--out", str(tmp_path / "x"), "--max-gb", value])
    else:
        assert mk.main(["--out", str(tmp_path / "x"), "--max-gb", value]) == 2
    assert not (tmp_path / "x").exists()


def test_a_bad_set_option_is_refused_before_any_work(tmp_path, monkeypatch, capsys):
    fake_torch_stack(monkeypatch, 1, 10_000 * GB)
    assert mk.main(["--out", str(tmp_path / "x"), "--set", "hidden_size=100"]) == 2     # 100 is not a multiple of 128
    assert "hidden_size" in capsys.readouterr().err and not (tmp_path / "x").exists()


def fake_versions(monkeypatch):
    """write_manifest imports torch and transformers only to record their versions; a stand-in module is enough."""
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.0-test"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="5.0-test"))


def test_the_manifest_records_the_27b_width_the_overrides_and_both_sizes(tmp_path, monkeypatch):
    """The checkpoint tests above all use the tiny base, so the width a manifest records is only exercised by a width other than `tiny` here
    (a real 27b checkpoint is 8 GB, too large for a unit test)."""
    fake_versions(monkeypatch)
    over = {"hidden_size": 2560}
    cfg = mk.build_config("dense", 4, width="27b", overrides=over)
    (tmp_path / "model.safetensors").write_bytes(b"w" * 100)
    (tmp_path / "config.json").write_text(json.dumps({**cfg, "seed": 5}))
    man = mk.write_manifest(tmp_path, cfg, "dense", 5, 4, 1.0, width="27b", overrides=over, estimate={"weights": 90, "total": 120})
    assert man["width"] == "27b" and man["overrides"] == over and man["dimensions"]["hidden_size"] == 2560
    config_bytes = (tmp_path / "config.json").stat().st_size
    assert man["size"] == {"estimated_weights_bytes": 90, "estimated_total_bytes": 120, "actual_weights_bytes": 100,
                           "actual_total_bytes": 100 + config_bytes}
    assert json.loads((tmp_path / "tiny-qwen4-manifest.json").read_text()) == man
    assert set(man["files"]) == {"model.safetensors", "config.json"}


def test_a_refresh_keeps_a_27b_manifests_width_overrides_and_estimate(tmp_path, monkeypatch):
    fake_versions(monkeypatch)
    over = {"hidden_size": 2560}
    cfg = mk.build_config("dense", 4, width="27b", overrides=over)
    (tmp_path / "model.safetensors").write_bytes(b"w" * 100)
    (tmp_path / "config.json").write_text(json.dumps({**cfg, "seed": 5}))
    mk.write_manifest(tmp_path, cfg, "dense", 5, 4, 1.0, width="27b", overrides=over, estimate={"weights": 90, "total": 120})
    (tmp_path / "extra.json").write_text("{}")                                  # a file a reference run would add
    after = mk.refresh_manifest(tmp_path)
    assert after["width"] == "27b" and after["overrides"] == over and after["seed"] == 5
    assert after["size"]["estimated_weights_bytes"] == 90 and after["size"]["estimated_total_bytes"] == 120
    assert "extra.json" in after["files"]


def test_a_snapshot_made_before_the_width_options_is_refreshed_without_inventing_a_width(tmp_path, monkeypatch):
    fake_versions(monkeypatch)
    cfg = mk.build_config("dense", 4)
    (tmp_path / "model.safetensors").write_bytes(b"w" * 10)
    (tmp_path / "config.json").write_text(json.dumps({**cfg, "seed": 5}))
    (tmp_path / "tiny-qwen4-manifest.json").write_text(json.dumps({"variant": "dense", "seed": 5, "layers": 4, "weight_scale": 1.0}))
    after = mk.refresh_manifest(tmp_path)
    assert after["width"] == "unknown" and after["overrides"] == {}
    assert after["size"]["estimated_weights_bytes"] is None and after["size"]["actual_weights_bytes"] == 10


# ---- the estimate and the checkpoint (reference interpreter) ----------------------------------------------------

SMALL_SCALED = ("--set", "hidden_size=256", "--set", "intermediate_size=512", "--set", "vocab_size=512",
                "--set", "linear_num_key_heads=4", "--set", "linear_num_value_heads=12")


@pytest.fixture(scope="module")
def tiny_default(tmp_path_factory):
    out = tmp_path_factory.mktemp("width") / "default"
    done = run_cli("--out", out, "--seed", 11, "--no-reference")
    assert done.returncode == 0, (done.stdout, done.stderr)
    return out


@needs_real_forward
def test_the_default_checkpoint_is_byte_identical_to_the_script_before_the_presets(tiny_default):
    for name, digest in GOLDEN_PURE.items():
        assert sha(tiny_default / name) == digest, name
    versions = subprocess.run([PY, "-c", "import torch,transformers,tokenizers;print(torch.__version__,transformers.__version__,tokenizers.__version__)"],
                              capture_output=True, text=True, check=True).stdout.split()
    if tuple(versions) != GOLDEN_STACK_VERSIONS:
        pytest.skip(f"the weights and tokenizer hashes were taken with {GOLDEN_STACK_VERSIONS}; this stack is {tuple(versions)}")
    for name, digest in GOLDEN_STACK.items():
        assert sha(tiny_default / name) == digest, name


@needs_real_forward
def test_the_default_manifest_records_the_width_the_overrides_and_both_sizes(tiny_default):
    man = json.loads((tiny_default / "tiny-qwen4-manifest.json").read_text())
    assert man["width"] == "tiny" and man["overrides"] == {} and "random weights" in man["provisional"]      # the `provisional` key stays
    size = man["size"]
    assert set(size) == {"estimated_weights_bytes", "estimated_total_bytes", "actual_weights_bytes", "actual_total_bytes"}
    assert size["actual_weights_bytes"] == (tiny_default / "model.safetensors").stat().st_size
    others = sum(p.stat().st_size for p in tiny_default.rglob("*") if p.is_file() and p.name != "tiny-qwen4-manifest.json")
    assert size["actual_total_bytes"] == others
    assert man["dimensions"] == mk.build_config("dense", 4)


@needs_real_forward
def test_the_estimate_is_within_two_percent_of_the_weights_file_for_the_default(tiny_default):
    man = json.loads((tiny_default / "tiny-qwen4-manifest.json").read_text())["size"]
    assert abs(man["estimated_weights_bytes"] - man["actual_weights_bytes"]) <= 0.02 * man["actual_weights_bytes"], man


@needs_real_forward
def test_the_estimate_counts_parameters_without_allocating_them():
    """The 27b config has 4.1 billion parameters; counting them must not allocate 8 GB (the meta device)."""
    code = ("import sys,importlib.util,resource;s=importlib.util.spec_from_file_location('m',sys.argv[1]);m=importlib.util.module_from_spec(s);"
            "s.loader.exec_module(m);t,x=m.import_stack();cfg=m.build_config('dense',4,width='27b');e=m.estimate_bytes(t,x,cfg);"
            "print(e['weights'],e['total'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)")
    done = subprocess.run([PY, "-c", code, str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    weights, total, rss_kb = map(int, done.stdout.split()[-3:])
    assert 7.5e9 < weights < 9e9 and total >= weights, (weights, total)         # about 4.12 billion parameters in bf16
    assert rss_kb < 3_000_000, rss_kb                                              # under 3 GB resident: nothing was materialised


@needs_real_forward
def test_the_real_estimate_refuses_the_27b_moe_before_writing(tmp_path):
    out = tmp_path / "moe27"
    done = run_cli("--out", out, "--variant", "moe", "--width", "27b")
    assert done.returncode == 2 and "--max-gb 12" in done.stderr, (done.stdout, done.stderr)
    assert not out.exists()


@needs_real_forward
def test_the_real_estimate_refuses_the_27b_dense_under_a_low_max_gb(tmp_path):
    out = tmp_path / "dense27"
    done = run_cli("--out", out, "--width", "27b", "--max-gb", 4)
    assert done.returncode == 2 and "--max-gb 4" in done.stderr and "estimated size 8." in done.stdout, (done.stdout, done.stderr)
    assert not out.exists()


@pytest.fixture(scope="module")
def small_scaled(tmp_path_factory):
    out = tmp_path_factory.mktemp("width") / "scaled"
    done = run_cli("--out", out, "--seed", 5, "--n-new", 4, *SMALL_SCALED)
    assert done.returncode == 0, (done.stdout, done.stderr)
    return out


@needs_real_forward
def test_a_scaled_checkpoint_builds_loads_clean_and_records_a_reference(small_scaled):
    cfg = json.loads((small_scaled / "config.json").read_text())
    assert cfg["hidden_size"] == 256 and cfg["intermediate_size"] == 512 and cfg["vocab_size"] == 512 and cfg["linear_num_key_heads"] == 4
    load = json.loads((small_scaled / "reference" / "load-report.json").read_text())
    assert load["pass"] and not load["missing_keys"] and not load["unexpected_keys"] and not load["mismatched_keys"]
    ref = json.loads((small_scaled / "reference" / "reference-ids.json").read_text())
    assert ref["n_new"] == 4 and all(len(p["generated_ids"]) == 4 for p in ref["prompts"])


@needs_real_forward
def test_a_scaled_checkpoints_tokenizer_is_sized_to_its_vocabulary(small_scaled):
    code = ("import sys;from transformers import AutoTokenizer;t=AutoTokenizer.from_pretrained(sys.argv[1]);print(len(t),"
            "t.decode(t.encode('Hello, world.',add_special_tokens=False)))")
    size, text = subprocess.run([PY, "-c", code, str(small_scaled)], capture_output=True, text=True, check=True).stdout.strip().split(" ", 1)
    assert int(size) == 512 and text == "Hello, world."


@needs_real_forward
def test_a_scaled_manifest_records_the_overrides_the_dimensions_and_an_estimate_near_the_file(small_scaled):
    man = json.loads((small_scaled / "tiny-qwen4-manifest.json").read_text())
    assert man["width"] == "tiny" and man["overrides"] == {"hidden_size": 256, "intermediate_size": 512, "vocab_size": 512,
                                                           "linear_num_key_heads": 4, "linear_num_value_heads": 12}
    assert man["dimensions"]["hidden_size"] == 256 and man["layers"] == 4
    size = man["size"]
    assert abs(size["estimated_weights_bytes"] - size["actual_weights_bytes"]) <= 0.02 * size["actual_weights_bytes"], size
    assert size["estimated_total_bytes"] >= size["estimated_weights_bytes"]


@needs_real_forward
def test_a_reference_refresh_keeps_the_width_the_overrides_and_the_estimate(tmp_path):
    out = tmp_path / "bare"
    assert run_cli("--out", out, "--seed", 5, "--no-reference", *SMALL_SCALED).returncode == 0
    before = json.loads((out / "tiny-qwen4-manifest.json").read_text())
    assert run_cli("--reference-only", out, "--n-new", 4).returncode == 0
    after = json.loads((out / "tiny-qwen4-manifest.json").read_text())
    assert after["width"] == before["width"] and after["overrides"] == before["overrides"]
    assert after["size"]["estimated_weights_bytes"] == before["size"]["estimated_weights_bytes"]
    assert after["size"]["actual_total_bytes"] > before["size"]["actual_total_bytes"]       # the reference files are now on disk


@needs_real_forward
def test_a_bad_width_through_the_cli_is_refused_with_exit_2_and_writes_nothing(tmp_path):
    out = tmp_path / "bad"
    done = run_cli("--out", out, "--set", "hidden_size=100")
    assert done.returncode == 2 and "hidden_size" in done.stderr, (done.stdout, done.stderr)
    assert not out.exists()
