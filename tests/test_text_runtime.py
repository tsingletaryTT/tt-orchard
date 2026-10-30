# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The adapter boundary of the text-model route (orchard/text_runtime.py) and the MiniCPM3-4B adapter."""
import json
from pathlib import Path

import pytest

from orchard import text_runtime as rt
from test_supervisor_text_runtime import CONFIG

M = rt.get("minicpm3")


def ctx(tmp_path, **kw):
    base = dict(python="/py", model_dir=str(tmp_path / "m"), weights=str(tmp_path / "w.safetensors"), out_dir=str(tmp_path / "out"), ref_dir=str(tmp_path / "s1" / "reference"),
                prompt_file=str(tmp_path / "prompt.txt"), ppl_parquet="/data/test.parquet")
    base.update(kw)
    return rt.TextContext(**base)


def model_dir(tmp_path, cfg=CONFIG):
    d = tmp_path / "m"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


# ---- the registry ---------------------------------------------------------------------------------------------------------

def test_a_repository_id_finds_its_runtime_whatever_the_revision():
    assert rt.runtime_for_model("openbmb/MiniCPM3-4B") is M and rt.runtime_for_model("openbmb/MiniCPM3-4B@" + "a" * 40) is M
    assert rt.runtime_for_model("someone/other") is None and rt.runtime_for_model(None) is None


def test_get_takes_only_a_registered_name():
    assert rt.get("minicpm3") is M and rt.get("nope") is None and rt.get(None) is None and rt.get(["minicpm3"]) is None


def test_the_default_prompt_is_the_reference_scripts_prompt():
    import sys
    sys.path.insert(0, str(M.scripts_dir))
    pytest.importorskip("transformers")
    import make_reference
    assert make_reference.PROMPT == rt.DEFAULT_PROMPT


# ---- the boundary ---------------------------------------------------------------------------------------------------------

def test_the_minicpm3_adapter_keeps_the_boundary(tmp_path):
    assert rt.adapter_problems(M, ctx(tmp_path)) == []


def test_an_adapter_that_breaks_the_boundary_is_named(tmp_path):
    class Bad(rt.TextRuntime):
        name, package_name, license_id = "bad", "bad", "mit"
        model_ids = ("a/b",)
        variants = (rt.Variant("Bad Name", (), "x"), rt.Variant("Bad Name", (), "y"))
        shipped_files = ("nonesuch.py",)
        scripts_dir = tmp_path
        context_limit = 0

        def reference_argv(self, c):
            return "not a list"
        ppl_reference_argv = agree_argv = ppl_chip_argv = bench_argv = lambda self, c, *a: ["python"]

        def dtype_policy(self, v):
            return []
    problems = rt.adapter_problems(Bad(), ctx(tmp_path))
    text = " ".join(problems)
    for needle in ("unique lowercase words", "nonesuch.py", "context_limit", "reference_argv must be a non-empty list", "ppl_chip_argv must start with", "dtype_policy"):
        assert needle in text, needle


def test_two_variants_with_one_name_are_named(tmp_path):
    class Dup(rt.TextRuntime):
        name, package_name, license_id, model_ids, context_limit = "d", "d", "mit", ("a/b",), 5
        variants = (rt.Variant("same", (), "x"), rt.Variant("same", (), "y"))
    assert any("unique" in p for p in rt.adapter_problems(Dup()))


def test_a_subclass_that_leaves_a_method_out_fails_loudly(tmp_path):
    class Partial(rt.TextRuntime):
        name = "partial"
    with pytest.raises(NotImplementedError):
        Partial().reference_argv(ctx(tmp_path))


def test_an_adapter_with_no_variants_is_refused():
    class Empty(rt.TextRuntime):
        name, package_name, license_id, model_ids, context_limit = "e", "e", "mit", ("a/b",), 5
    assert rt.adapter_problems(Empty()) == ["variants is empty"]


# ---- the MiniCPM3 commands ------------------------------------------------------------------------------------------------------

def test_the_reference_command_writes_under_the_contexts_directory(tmp_path):
    argv = M.reference_argv(ctx(tmp_path))
    assert argv[1].endswith("make_reference.py") and argv[argv.index("--out-dir") + 1] == str(tmp_path / "out") and "--prompt-file" in argv
    assert argv[argv.index("--decode") + 1] == str(rt.DECODE_TOKENS)


def test_the_chip_agreement_command_follows_the_float32_reference(tmp_path):
    argv = M.agree_argv(ctx(tmp_path), M.variant("bf16"), "/o")
    assert argv[argv.index("--ref") + 1] == str(tmp_path / "s1" / "reference" / "float32") and argv[argv.index("--out") + 1] == "/o"
    assert "run_chip.py" in argv[1] and argv[2] == "agree" and "--opt" in argv
    assert "--mlp-dtype" in M.agree_argv(ctx(tmp_path), M.variant("mlp8"), "/o") and "--mlp-dtype" not in argv


def test_the_perplexity_commands_use_only_flags_the_script_takes(tmp_path):
    argv = M.ppl_chip_argv(ctx(tmp_path), M.variant("mlp8"), "/o.json")
    assert argv[2] == "chip" and "--upd" not in argv and argv[argv.index("--mlp-dtype") + 1] == "bfp8"
    assert argv[argv.index("--ref") + 1] == str(tmp_path / "s1" / "ppl_reference.json")
    ref = M.ppl_reference_argv(ctx(tmp_path), "/r.json")
    assert ref[2] == "ref" and ref[ref.index("--parquet") + 1] == "/data/test.parquet" and ref[ref.index("--tokens") + 1] == str(rt.PPL_TOKENS)


def test_the_bench_command_traces_and_repeats(tmp_path):
    argv = M.bench_argv(ctx(tmp_path), M.variant("bf16"), "/o.json", 64, 3)
    assert argv[2] == "bench" and "--trace" in argv and "--batched-prefill" in argv
    assert argv[argv.index("--steps") + 1] == "64" and argv[argv.index("--repeats") + 1] == "3"


def test_each_stage_gets_its_commands_for_each_variant(tmp_path):
    c2 = rt.chip_config_commands(M, ctx(tmp_path), 2)
    assert [x["name"] for x in c2] == ["agree_bf16", "ppl_bf16", "agree_mlp8", "ppl_mlp8"]
    assert c2[0]["outputs"] == ["chip/bf16/manifest.json"] and c2[1]["outputs"] == ["ppl_chip_bf16.json"]
    c6 = rt.chip_config_commands(M, ctx(tmp_path), 6)
    assert [x["name"] for x in c6] == ["bench_bf16", "bench_mlp8"] and all("--repeats" in x["argv"] for x in c6)


def test_the_chip_test_command_is_one_command_in_the_stage_directory(tmp_path):
    command = rt.chip_test_command(rt.TextPaths("/py"), tmp_path / "stage")
    assert command.startswith("/py ") and "text_chip_test.py" in command and command.endswith("text_chip_config.json")


# ---- the configuration and the weights ------------------------------------------------------------------------------------------------

def test_the_real_configuration_is_accepted(tmp_path):
    assert M.config_problems(model_dir(tmp_path)) == []


@pytest.mark.parametrize("change,needle", [({"model_type": "llama"}, "model_type"), ({"kv_lora_rank": None}, "kv_lora_rank"),
                                           ({"num_key_value_heads": 8}, "key-value heads")])
def test_a_configuration_the_port_does_not_cover_is_refused(tmp_path, change, needle):
    cfg = dict(CONFIG)
    for k, v in change.items():
        cfg.pop(k) if v is None else cfg.update({k: v})
    assert needle in " ".join(M.config_problems(model_dir(tmp_path, cfg)))


def test_a_directory_with_no_config_is_a_problem_and_not_a_crash(tmp_path):
    assert "config.json cannot be read" in M.config_problems(tmp_path / "nowhere")[0]


def test_weights_in_the_directory_are_used_in_place_and_otherwise_converted_into_the_cache(tmp_path):
    d = model_dir(tmp_path)
    cache = tmp_path / "cache"
    w = M.weights_path(d, cache)
    assert w.parent == cache and w.name.startswith("m-") and w.name.endswith(".safetensors")
    (d / "model.safetensors").write_bytes(b"x")
    assert M.weights_path(d, cache) == d.resolve() / "model.safetensors"


def test_a_weights_file_that_exists_needs_no_conversion(tmp_path):
    w = tmp_path / "w.safetensors"
    assert M.prepare_argv(ctx(tmp_path))[1].endswith("convert_weights.py")
    w.write_bytes(b"x")
    assert M.prepare_argv(ctx(tmp_path)) is None


def test_the_dtype_policy_follows_the_variant():
    assert M.dtype_policy(M.variant("bf16")) == {"default": "bf16", "dense_mlp": "bf16"}
    assert M.dtype_policy(M.variant("mlp8")) == {"default": "bf16", "dense_mlp": "bfp8"}
    with pytest.raises(rt.TextStageError):
        M.variant("nope")


# ---- inputs ---------------------------------------------------------------------------------------------------------------------

def test_the_perplexity_window_is_found_in_the_hugging_face_cache_or_given(tmp_path):
    f = tmp_path / "hub" / "datasets--Salesforce--wikitext" / "snapshots" / "abc" / "wikitext-2-raw-v1" / "test-00000-of-00001.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    assert rt.find_ppl_parquet(tmp_path) == str(f)
    assert rt.find_ppl_parquet(tmp_path, str(f)) == str(f)
    assert rt.find_ppl_parquet(tmp_path, str(tmp_path / "missing.parquet")) is None
    assert rt.find_ppl_parquet(tmp_path / "empty") is None


def test_the_card_license_is_read_from_the_front_matter(tmp_path):
    d = model_dir(tmp_path)
    (d / "README.md").write_text("---\nlicense: apache-2.0\nlanguage:\n- en\n---\nlicense: not this one\n")
    assert rt.card_license(d) == "apache-2.0"
    (d / "README.md").write_text("---\nlicense_name: other\nlicense: mit\n---\n")
    assert rt.card_license(d) == "mit"
    (d / "README.md").write_text("no front matter\nlicense: mit\n")
    assert rt.card_license(d) is None and rt.card_license(tmp_path / "none") is None


def test_the_script_hashes_cover_every_shipped_file():
    h = rt.script_hashes(M)
    assert set(h) == set(M.shipped_files) and all(len(v) == 64 for v in h.values())


def test_the_interpreter_check_names_the_variable(tmp_path):
    assert "ORCHARD_TEXT_PYTHON" in rt.environment_problems(rt.TextPaths(str(tmp_path / "nope")))[0]
    assert rt.environment_problems(rt.TextPaths("/bin/sh")) == []
