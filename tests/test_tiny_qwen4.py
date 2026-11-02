# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The tiny Qwen4-shaped rehearsal checkpoint (scripts/tiny_qwen4/make_checkpoint.py, design in
docs/superpowers/specs/2026-11-02-tiny-qwen4-rehearsal-design.md). PROVISIONAL: the shape comes from public readings of an
unreleased model, the weights are random.

The config, prompt and refusal tests need no torch. The tests that build a checkpoint or run the CPU reference use the reference
interpreter (ORCHARD_REFERENCE_PYTHON; transformers 5.19.0 has qwen4_exp) and skip without it. Nothing here opens a device, imports
ttnn or ttml, or touches the network. Checkpoints are generated into pytest temp directories and never into the repository."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from orchard import hazards as hz
from orchard import model_config as mc
from test_ops_inventory_stub import PY, needs_stack

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "tiny_qwen4" / "make_checkpoint.py"
FLASH = REPO / "tests" / "fixtures" / "model-configs" / "qwen3.8-flash-next.json"
spec = importlib.util.spec_from_file_location("make_checkpoint", SCRIPT)
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)

EXPERT_KEYS = ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size")
SCHEDULE_KEYS = ("mlp_only_layers", "decoder_sparse_step")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_cli(*args, py=None):
    return subprocess.run([py or PY, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


# ---- the config (no torch) -------------------------------------------------------------------------------------

@pytest.mark.parametrize("layers", [4, 8, 12])
def test_the_layer_schedule_is_three_linear_then_one_full_attention(layers):
    for variant in mk.VARIANTS:
        types = mk.build_config(variant, layers)["layer_types"]
        assert types == (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)


@pytest.mark.parametrize("layers", [0, 1, 3, 5, 6, -4, True, 4.0, "4"])
def test_a_layer_count_that_is_not_a_positive_multiple_of_four_is_refused(layers):
    with pytest.raises(ValueError, match="multiple of 4"):
        mk.build_config("dense", layers)


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="variant"):
        mk.build_config("sparse", 4)


def test_the_dense_config_has_no_expert_or_schedule_key_at_all():
    """transformers fills a missing num_experts with 512 and builds the wrong model without an error, and the shim reads the raw
    config, so the dense config must carry none of these keys (not even a zero)."""
    cfg = mk.build_config("dense", 4)
    assert not [k for k in EXPERT_KEYS + SCHEDULE_KEYS if k in cfg], cfg
    assert cfg["intermediate_size"] == 256 and cfg["hidden_act"] == "silu"


def test_the_moe_and_mixed_configs_carry_every_expert_key_and_the_mixed_one_names_its_dense_layer():
    moe, mixed = mk.build_config("moe", 4), mk.build_config("mixed", 4)
    for cfg in (moe, mixed):
        assert all(isinstance(cfg[k], int) and cfg[k] > 0 for k in EXPERT_KEYS), cfg
    assert not [k for k in SCHEDULE_KEYS if k in moe]
    assert mixed["mlp_only_layers"] == [0] and mixed["intermediate_size"] == 256


def test_the_config_is_flat_qwen4_exp_text_with_the_class_that_reads_it():
    cfg = mk.build_config("dense", 4)
    assert cfg["model_type"] == "qwen4_exp_text" and cfg["architectures"] == ["Qwen4ExpForCausalLM"]
    assert "text_config" not in cfg and "auto_map" not in cfg


def test_hyper_connection_settings_are_written_explicitly_and_are_not_a_borrowed_default():
    """hc_lowrank is 640 at full size and 320 in Flash-Next; the tiny model has its own value, written down."""
    cfg = mk.build_config("dense", 4)
    assert cfg["hc_count"] == 4
    assert isinstance(cfg["hc_lowrank"], int) and cfg["hc_lowrank"] not in (320, 640)
    assert cfg["hc_lowrank"] * 8 == cfg["hidden_size"]                      # the 1/8 ratio of the full-size reading


def test_the_sparse_indexer_and_ngram_keys_exist_because_the_full_attention_layer_does_not_build_without_them():
    cfg = mk.build_config("dense", 4)
    for key in ("indexer_budget", "indexer_compress_ratio", "indexer_head_dim", "indexer_kv_heads", "indexer_n_heads",
                "ngram_size", "ngram_vocab_size_base", "heads_per_ngram", "split_ngram_parts", "ple_layer_ids", "ple_embed_dim"):
        assert isinstance(cfg[key], (int, list)), key
    assert cfg["ngram_vocab_size_base"] <= 10_000          # a small table: the full-size one is 20 million


def write_cfg(tmp_path, variant, layers=4) -> Path:
    p = tmp_path / f"{variant}.json"
    p.write_text(json.dumps(mk.build_config(variant, layers)))
    return p


@pytest.mark.parametrize("variant", ["dense", "moe", "mixed"])
@pytest.mark.parametrize("dies", [2, 4])
def test_no_stage_0_division_hazard_fires_at_two_or_four_dies(tmp_path, variant, dies):
    scan = hz.scan_model_hazards(mc.read_model_config(write_cfg(tmp_path, variant)), mc.read_model_config(FLASH), dies=dies)
    bad = [h.name for h in scan.hazards if "not_divisible" in h.name or "not_whole_tiles" in h.name or "tile" in h.name]
    assert bad == [], bad


def test_stage_0_reads_the_hyper_connection_values_and_flags_the_dense_shape(tmp_path):
    flash = mc.read_model_config(FLASH)
    dense = hz.scan_model_hazards(mc.read_model_config(write_cfg(tmp_path, "dense")), flash, dies=2)
    names = {h.name for h in dense.hazards}
    assert {"dense_model_nearest_is_moe", "dense_qwen4_not_flash_next_shape"} <= names
    assert dense.facts["hyper_connections"]["hc_count"] == 4 and dense.facts["hyper_connections"]["hc_lowrank"] == 8
    assert dense.facts["hyper_connections"]["width"] == 4 * 64
    moe = hz.scan_model_hazards(mc.read_model_config(write_cfg(tmp_path, "moe")), flash, dies=2)
    assert not {"dense_model_nearest_is_moe", "dense_qwen4_not_flash_next_shape"} & {h.name for h in moe.hazards}


# ---- the prompts (no torch) ------------------------------------------------------------------------------------

def test_prompts_are_seeded_have_the_stated_lengths_and_stay_inside_the_byte_token_range():
    a, b, c = mk.make_prompts(7), mk.make_prompts(7), mk.make_prompts(8)
    assert a == b and a != c
    assert [len(p) for p in a] == [8, 16, 33]                   # 33 is not a multiple of the delta-net chunk size
    for prompt in a + c:
        assert all(isinstance(i, int) and mk.FIRST_BYTE_ID <= i < mk.FIRST_BYTE_ID + 256 for i in prompt)
    assert mk.FIRST_BYTE_ID == 4 and mk.FIRST_BYTE_ID > max(mk.SPECIAL_IDS.values())


# ---- where the output may go (no torch) ------------------------------------------------------------------------

def test_an_output_directory_inside_the_repository_is_refused_before_anything_is_written(tmp_path):
    inside = REPO / "tmp-tiny-qwen4-refusal-test"
    done = subprocess.run(["python3", str(SCRIPT), "--out", str(inside)], capture_output=True, text=True)
    assert done.returncode == 2 and "repository" in done.stderr, done.stderr
    assert not inside.exists()
    with pytest.raises(SystemExit):
        mk.check_output_dir(REPO / "docs" / "x")


def test_a_non_empty_output_directory_is_refused_so_two_checkpoints_never_mix(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    done = subprocess.run(["python3", str(SCRIPT), "--out", str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 2 and "not empty" in done.stderr, done.stderr


# ---- the checkpoint and the CPU reference (reference interpreter) ----------------------------------------------

@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny") / "dense"
    done = run_cli("--out", out, "--variant", "dense", "--seed", 11)
    assert done.returncode == 0, (done.stdout, done.stderr)
    return out


@needs_stack
def test_a_dense_checkpoint_has_the_expected_files_and_a_manifest_whose_hashes_match(dense):
    files = {p.name for p in dense.iterdir()}
    assert {"config.json", "generation_config.json", "model.safetensors", "prompts.json", "tokenizer.json",
            "tokenizer_config.json", "tiny-qwen4-manifest.json", "reference"} <= files
    man = json.loads((dense / "tiny-qwen4-manifest.json").read_text())
    assert man["seed"] == 11 and man["variant"] == "dense" and man["layers"] == 4
    assert man["shim_sha256"] == sha(REPO / "orchard" / "skills" / "reference-gate-templates" / "qwen4_dense.py")
    assert set(man["files"]) >= {"config.json", "model.safetensors", "prompts.json", "reference/generated-ids.json"}
    for rel, digest in man["files"].items():
        assert sha(dense / rel) == digest, rel


@needs_stack
def test_the_written_config_is_the_built_config(dense):
    assert json.loads((dense / "config.json").read_text()) == {**mk.build_config("dense", 4), "seed": 11}


@needs_stack
def test_the_safetensors_file_holds_bf16_floats_and_the_hybrid_layer_keys(dense):
    code = ("import json,sys,struct;f=open(sys.argv[1],'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));"
            "h.pop('__metadata__',None);print(json.dumps({k:v['dtype'] for k,v in h.items()}))")
    dtypes = json.loads(subprocess.run([PY, "-c", code, str(dense / "model.safetensors")], capture_output=True, text=True,
                                       check=True).stdout)
    floats = {d for k, d in dtypes.items() if "layer_multipliers" not in k and "vocab_sizes" not in k and "offsets" not in k}
    assert floats == {"BF16"}, floats
    names = " ".join(dtypes)
    for part in ("linear_attn.in_proj_qkv", "self_attn.indexer.index_qk_proj", "attn_hyper_connection.hc_norm", "ple.ple_embedding",
                 "hyper_connection_mixer", "mlp.gate_proj", "lm_head"):
        assert part in names, part
    assert "mlp.experts" not in names and "shared_expert" not in names        # dense: no expert tensors


@needs_stack
def test_the_reference_records_32_greedy_tokens_for_each_prompt_through_a_clean_load(dense):
    ref = json.loads((dense / "reference" / "reference-ids.json").read_text())
    prompts = json.loads((dense / "prompts.json").read_text())["prompts"]
    assert ref["n_new"] == 32 and [p["prompt_ids"] for p in ref["prompts"]] == prompts
    for p in ref["prompts"]:
        assert len(p["generated_ids"]) == 32 and p["first_token_ok"] and p["appended_ok"]
        assert len(set(p["generated_ids"])) >= 4
        assert len(p["steps"]) == 32 and len(p["steps"][0]["top5"]) == 5 and p["steps"][0]["gap"] >= 0
    load = json.loads((dense / "reference" / "load-report.json").read_text())
    assert load["pass"] and not load["missing_keys"] and not load["unexpected_keys"] and not load["mismatched_keys"]
    assert load["dense_qwen4"]["layers"] == [0, 1, 2, 3]                   # the shim built every layer dense
    first = json.loads((dense / "reference" / "generated-ids.json").read_text())
    assert first["generated_ids"] == ref["prompts"][0]["generated_ids"] and first["complete"] is True
    assert json.loads((dense / "reference" / "prompt-ids.json").read_text())["prompt_ids"] == prompts[0]


@needs_stack
def test_the_same_seed_gives_the_same_bytes_and_the_same_reference_ids_and_a_different_seed_does_not(dense, tmp_path):
    again = tmp_path / "again"
    assert run_cli("--out", again, "--variant", "dense", "--seed", 11).returncode == 0
    assert sha(again / "model.safetensors") == sha(dense / "model.safetensors")
    assert sha(again / "reference" / "reference-ids.json") == sha(dense / "reference" / "reference-ids.json")
    other = tmp_path / "other"
    assert run_cli("--out", other, "--variant", "dense", "--seed", 12, "--no-reference").returncode == 0
    assert sha(other / "model.safetensors") != sha(dense / "model.safetensors")


@needs_stack
@pytest.mark.parametrize("variant,layer_kinds", [("moe", {"Qwen4ExpTextSparseMoeBlock"}), ("mixed", {"Qwen4ExpTextMLP", "Qwen4ExpTextSparseMoeBlock"})])
def test_the_moe_and_mixed_variants_load_clean_and_record_a_reference(tmp_path, variant, layer_kinds):
    out = tmp_path / variant
    done = run_cli("--out", out, "--variant", variant, "--n-new", 8)      # the MoE fallback on the CPU is slow
    assert done.returncode == 0, (done.stdout, done.stderr)
    load = json.loads((out / "reference" / "load-report.json").read_text())
    assert load["pass"] and not load["missing_keys"] and not load["unexpected_keys"]
    assert set(load["layer_blocks"]) == layer_kinds
    if variant == "moe":
        assert "dense_qwen4" not in load                                        # no dense layer: the shim is not used
    else:
        assert load["dense_qwen4"]["layers"] == [0]


@needs_stack
def test_the_reference_refuses_a_checkpoint_that_is_missing_a_tensor(dense, tmp_path):
    """A load with a missing key would give a reference from random fill-in weights, and every port would 'agree' with the wrong ids."""
    broken = tmp_path / "broken"
    shutil.copytree(dense, broken, ignore=shutil.ignore_patterns("reference", "tiny-qwen4-manifest.json"))
    code = ("import sys;from safetensors.torch import load_file,save_file;d=load_file(sys.argv[1]);"
            "d.pop('model.layers.1.mlp.down_proj.weight');save_file(d,sys.argv[1],metadata={'format':'pt'})")
    subprocess.run([PY, "-c", code, str(broken / "model.safetensors")], check=True)
    done = run_cli("--reference-only", broken, "--n-new", 8)
    assert done.returncode == 3 and "missing" in done.stderr, (done.stdout, done.stderr)
    assert not (broken / "reference" / "generated-ids.json").exists()


@needs_stack
def test_the_reference_refuses_a_degenerate_decode(tmp_path):
    """With every weight scaled to zero the model emits one constant token; a constant reference lets any wrong port pass."""
    out = tmp_path / "zero"
    done = run_cli("--out", out, "--variant", "dense", "--weight-scale", 0, "--n-new", 8)
    assert done.returncode == 3 and ("degenerate" in done.stderr or "non-finite" in done.stderr), (done.stdout, done.stderr)
    assert not (out / "reference" / "generated-ids.json").exists()


@needs_stack
def test_stage_1s_own_template_runs_on_the_checkpoint_and_passes(dense, tmp_path):
    """The claim in the design: with the tokenizer present, reference_gate.py runs unmodified on the snapshot."""
    stage = tmp_path / "run" / "stages" / "1"
    stage.mkdir(parents=True)
    tmpl = REPO / "orchard" / "skills" / "reference-gate-templates"
    for name in ("reference_gate.py", "qwen4_dense.py"):
        shutil.copy(tmpl / name, stage / name)
    (stage / "reference_config.json").write_text(json.dumps(
        {"run_dir": str(tmp_path / "run"), "model_snapshot": str(dense), "python": PY}))
    done = subprocess.run([PY, str(stage / "reference_gate.py")], capture_output=True, text=True, cwd=stage)
    assert done.returncode == 0, (done.stdout[-1500:], done.stderr[-1500:])
    ref = json.loads((stage / "reference.json").read_text())
    assert ref["verdict"] == "pass", [c for c in ref["checks"] if not c["pass"]]
    load = json.loads((stage / "evidence" / "load-report.json").read_text())
    assert load["dense_qwen4"]["layers"] == [0, 1, 2, 3]


@needs_stack
def test_the_tokenizer_round_trips_and_its_size_equals_the_config_vocabulary(dense):
    code = ("import sys,json;from transformers import AutoTokenizer;t=AutoTokenizer.from_pretrained(sys.argv[1]);"
            "s=['Hello, world.','caf\\u00e9 \\u4f60\\u597d \\U0001F600','a\\tb\\r\\n  c  '];"
            "print(json.dumps([len(t),[t.decode(t.encode(x,add_special_tokens=False))==x for x in s]]))")
    size, trips = json.loads(subprocess.run([PY, "-c", code, str(dense)], capture_output=True, text=True, check=True).stdout)
    assert size == 384 == json.loads((dense / "config.json").read_text())["vocab_size"] and all(trips)


@needs_stack
def test_no_tokenizer_files_are_written_with_the_option_off_and_the_prompts_still_exist(tmp_path):
    out = tmp_path / "bare"
    assert run_cli("--out", out, "--variant", "dense", "--no-tokenizer", "--no-reference").returncode == 0
    assert not (out / "tokenizer.json").exists() and (out / "prompts.json").exists()
    assert not (out / "reference").exists()


@needs_stack
def test_a_failed_reference_refresh_removes_the_old_ids_so_they_cannot_be_read_as_results(dense, tmp_path):
    """PR 356 review. A refresh that fails must not leave an earlier `generated-ids.json` that still says complete."""
    broken = tmp_path / "broken"
    shutil.copytree(dense, broken)
    assert (broken / "reference" / "generated-ids.json").exists()
    code = ("import sys;from safetensors.torch import load_file,save_file;d=load_file(sys.argv[1]);"
            "d.pop('model.layers.1.mlp.down_proj.weight');save_file(d,sys.argv[1],metadata={'format':'pt'})")
    subprocess.run([PY, "-c", code, str(broken / "model.safetensors")], check=True)
    assert run_cli("--reference-only", broken, "--n-new", 8).returncode == 3
    left = sorted(p.name for p in (broken / "reference").iterdir()) if (broken / "reference").exists() else []
    assert not {"generated-ids.json", "prompt-ids.json", "reference-ids.json"} & set(left), left


@needs_stack
def test_the_greedy_token_is_torchs_argmax_even_when_the_top_logits_tie():
    """PR 356 review. bf16 logits tie often. Stage 1 takes `torch.argmax`; `torch.topk` can order the tied entries differently, which
    would give a reference that stage 1's decoder would not reproduce."""
    code = ("import sys,importlib.util,torch;s=importlib.util.spec_from_file_location('m',sys.argv[1]);m=importlib.util.module_from_spec(s);"
            "s.loader.exec_module(m);torch.manual_seed(1);bad=0\n"
            "for _ in range(200):\n"
            "    x=(torch.randn(384)*2).to(torch.bfloat16).float();x[torch.randperm(384)[:6]]=x.max()\n"
            "    bad+=m.greedy_token(torch,x)!=int(torch.argmax(x))\n"
            "print(bad)")
    done = subprocess.run([PY, "-c", code, str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.strip() == "0", (done.stdout, done.stderr)


@needs_stack
def test_a_reference_run_on_an_existing_snapshot_refreshes_the_manifest(tmp_path):
    """PR 356 review. `--reference-only` adds or replaces files the manifest covers, so the manifest is rewritten to match the disk."""
    out = tmp_path / "bare"
    assert run_cli("--out", out, "--variant", "dense", "--seed", 5, "--no-reference").returncode == 0
    before = json.loads((out / "tiny-qwen4-manifest.json").read_text())
    assert not [f for f in before["files"] if f.startswith("reference/")]
    assert run_cli("--reference-only", out, "--n-new", 8).returncode == 0
    after = json.loads((out / "tiny-qwen4-manifest.json").read_text())
    assert after["seed"] == 5 and after["variant"] == "dense" and after["layers"] == 4
    assert {"reference/generated-ids.json", "reference/reference-ids.json", "reference/load-report.json"} <= set(after["files"])
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "tiny-qwen4-manifest.json"}
    assert set(after["files"]) == on_disk
    for rel, digest in after["files"].items():
        assert sha(out / rel) == digest, rel
