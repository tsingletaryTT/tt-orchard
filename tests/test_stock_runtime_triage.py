# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 of a stock-runtime run (orchard/stock_runtime.py): the model's config compared with the tt-transformers `model_params`
config, and its tensor names and shapes compared with the layout tt-transformers loads. The measurement decides the class:
equal and loadable is `stock-runtime`, anything else is `full-port` with the reasons."""
import json
import struct
from pathlib import Path

import pytest

from orchard import stock_runtime as sr
from orchard.stages import gate_delta_stock

LLAMA = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
         "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100,
         "rope_theta": 500000.0, "rope_scaling": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                                  "original_max_position_embeddings": 8192, "rope_type": "llama3"},
         "tie_word_embeddings": False, "hidden_act": "silu", "rms_norm_eps": 1e-05, "attention_bias": False,
         "mlp_bias": False, "max_position_embeddings": 131072, "torch_dtype": "bfloat16"}


def tensors_for(cfg, *, qwen=False, tie=None):
    h, ff, v = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or h // nh
    t = {"model.embed_tokens.weight": ("BF16", [v, h]), "model.norm.weight": ("BF16", [h])}
    if not (cfg.get("tie_word_embeddings") if tie is None else tie):
        t["lm_head.weight"] = ("BF16", [v, h])
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        t[p + "self_attn.q_proj.weight"] = ("BF16", [nh * hd, h])
        t[p + "self_attn.k_proj.weight"] = ("BF16", [nkv * hd, h])
        t[p + "self_attn.v_proj.weight"] = ("BF16", [nkv * hd, h])
        t[p + "self_attn.o_proj.weight"] = ("BF16", [h, nh * hd])
        if qwen:
            t[p + "self_attn.q_proj.bias"] = ("BF16", [nh * hd])
            t[p + "self_attn.k_proj.bias"] = ("BF16", [nkv * hd])
            t[p + "self_attn.v_proj.bias"] = ("BF16", [nkv * hd])
        t[p + "mlp.gate_proj.weight"] = ("BF16", [ff, h])
        t[p + "mlp.up_proj.weight"] = ("BF16", [ff, h])
        t[p + "mlp.down_proj.weight"] = ("BF16", [h, ff])
        t[p + "input_layernorm.weight"] = ("BF16", [h])
        t[p + "post_attention_layernorm.weight"] = ("BF16", [h])
    return t


def write_st(path: Path, tensors: dict):
    """A safetensors file with a real header and no tensor bytes (the triage reads headers only)."""
    header, off = {"__metadata__": {"format": "pt"}}, 0
    sizes = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1}
    for name, (dtype, shape) in tensors.items():
        n = sizes[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def snapshot(tmp_path, cfg=None, tensors=None, *, split=False, name="snap"):
    snap = tmp_path / name
    snap.mkdir(exist_ok=True)
    (snap / "config.json").write_text(json.dumps(LLAMA if cfg is None else cfg))
    t = tensors_for(LLAMA if cfg is None else cfg) if tensors is None else tensors
    if split:
        items = list(t.items())
        a, b = dict(items[: len(items) // 2]), dict(items[len(items) // 2:])
        write_st(snap / "model-00001-of-00002.safetensors", a)
        write_st(snap / "model-00002-of-00002.safetensors", b)
        (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            **{k: "model-00001-of-00002.safetensors" for k in a}, **{k: "model-00002-of-00002.safetensors" for k in b}}}))
    else:
        write_st(snap / "model.safetensors", t)
    return snap


def params(tmp_path, cfg=None, name="Llama-Test-1B-Instruct"):
    d = tmp_path / "tt-metal" / "models" / "tt_transformers" / "model_params" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(LLAMA if cfg is None else cfg))
    return d


def run(tmp_path, snap, pdir, **kw):
    out = tmp_path / "run"
    stage_dir = out / "stages" / "0"
    stage_dir.mkdir(parents=True, exist_ok=True)
    return sr.triage(stage_dir=stage_dir, run_dir=out, model_id="org/Llama-Test-1B-Instruct", snapshot=snap, params_dir=pdir,
                     support=kw.get("support", {"via": "model_params", "matched": "org/Llama-Test-1B-Instruct",
                                                "boards": ["N150"], "blackhole": False}),
                     tt_metal=kw.get("tt_metal", {"tag": "v0.80.0", "dirty": False}))


def test_an_equal_config_and_a_loadable_layout_is_stock_runtime(tmp_path):
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path))
    assert d["class"] == "stock-runtime" and d["path"] == "stock-runtime"
    rec = d["stock_runtime"]
    assert rec["config_equal"] is True and rec["keys_differing"] == [] and rec["layout_problems"] == []
    assert rec["name"] == "Llama-Test-1B-Instruct" and rec["layers"] == 2
    assert d["nearest_model"] == "tt-transformers:Llama-Test-1B-Instruct" and d["model"] == "org/Llama-Test-1B-Instruct"


def test_the_delta_passes_the_stock_gate_and_names_its_evidence(tmp_path):
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path))
    g = gate_delta_stock(tmp_path / "run" / "stages" / "0", tmp_path / "run")
    assert g.ok, g.reasons
    for rel in ("stages/0/evidence/stock-config-compare.json", "stages/0/evidence/stock-layout.json"):
        assert (tmp_path / "run" / rel).is_file()
        assert rel in {e for item in d["differences"] for e in item["evidence"]}


def test_a_sharded_checkpoint_is_read_through_its_index(tmp_path):
    d = run(tmp_path, snapshot(tmp_path, split=True), params(tmp_path))
    assert d["class"] == "stock-runtime"


@pytest.mark.parametrize("key,value", [("hidden_size", 128), ("num_hidden_layers", 3), ("num_attention_heads", 8),
                                       ("num_key_value_heads", 4), ("intermediate_size", 256), ("vocab_size", 101),
                                       ("rope_theta", 10000.0), ("tie_word_embeddings", True), ("hidden_act", "gelu"),
                                       ("rms_norm_eps", 1e-6), ("model_type", "mistral"), ("attention_bias", True),
                                       ("architectures", ["MistralForCausalLM"])])
def test_a_decisive_config_difference_makes_it_a_full_port_and_names_the_key(tmp_path, key, value):
    cfg = {**LLAMA, key: value}
    d = run(tmp_path, snapshot(tmp_path, cfg=cfg, tensors=tensors_for(LLAMA)), params(tmp_path))
    assert d["class"] == "full-port" and d["path"] == "full-port"
    assert key in d["stock_runtime"]["keys_differing"] and d["stock_runtime"]["config_equal"] is False
    assert any(key in r for r in d["path_reasons"])
    assert gate_delta_stock(tmp_path / "run" / "stages" / "0", tmp_path / "run").ok


def test_a_context_length_or_dtype_difference_alone_does_not_change_the_class(tmp_path):
    cfg = {**LLAMA, "max_position_embeddings": 4096, "torch_dtype": "float16", "transformers_version": "4.50"}
    d = run(tmp_path, snapshot(tmp_path, cfg=cfg, tensors=tensors_for(LLAMA)), params(tmp_path))
    assert d["class"] == "stock-runtime"


def test_a_rope_scaling_value_that_differs_is_a_difference(tmp_path):
    cfg = {**LLAMA, "rope_scaling": {**LLAMA["rope_scaling"], "factor": 4.0}}
    d = run(tmp_path, snapshot(tmp_path, cfg=cfg, tensors=tensors_for(LLAMA)), params(tmp_path))
    assert d["class"] == "full-port" and "rope_scaling" in d["stock_runtime"]["keys_differing"]


def test_the_two_spellings_of_the_rope_type_are_the_same_value(tmp_path):
    cfg = {**LLAMA, "rope_scaling": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                     "original_max_position_embeddings": 8192, "type": "llama3"}}
    d = run(tmp_path, snapshot(tmp_path, cfg=cfg, tensors=tensors_for(LLAMA)), params(tmp_path))
    assert d["class"] == "stock-runtime"


def test_an_absent_head_dim_equals_hidden_over_heads_and_a_different_one_differs(tmp_path):
    same = {**LLAMA, "head_dim": 16}          # 64 / 4
    assert run(tmp_path, snapshot(tmp_path, cfg=same, tensors=tensors_for(same)), params(tmp_path))["class"] == "stock-runtime"
    other = {**LLAMA, "head_dim": 32}
    d = run(tmp_path, snapshot(tmp_path, cfg=other, tensors=tensors_for(other)), params(tmp_path))
    assert "head_dim" in d["stock_runtime"]["keys_differing"]


def test_absent_defaults_are_equal_to_their_explicit_value(tmp_path):
    cfg = {k: v for k, v in LLAMA.items() if k not in ("attention_bias", "mlp_bias", "tie_word_embeddings")}
    d = run(tmp_path, snapshot(tmp_path, cfg=cfg, tensors=tensors_for(LLAMA)), params(tmp_path))
    assert d["class"] == "stock-runtime"


# ---- the layout ---------------------------------------------------------------------------------------------

def test_a_missing_layer_tensor_is_a_layout_problem(tmp_path):
    t = tensors_for(LLAMA)
    del t["model.layers.1.mlp.up_proj.weight"]
    d = run(tmp_path, snapshot(tmp_path, tensors=t), params(tmp_path))
    assert d["class"] == "full-port" and any("model.layers.1.mlp.up_proj.weight" in p for p in d["stock_runtime"]["layout_problems"])


def test_a_missing_lm_head_is_a_layout_problem_unless_the_embeddings_are_tied(tmp_path):
    t = tensors_for(LLAMA)
    del t["lm_head.weight"]
    d = run(tmp_path, snapshot(tmp_path, tensors=t), params(tmp_path))
    assert d["class"] == "full-port" and any("lm_head" in p for p in d["stock_runtime"]["layout_problems"])
    tied = {**LLAMA, "tie_word_embeddings": True}
    d = run(tmp_path, snapshot(tmp_path, cfg=tied, tensors=tensors_for(tied)), params(tmp_path, cfg=tied))
    assert d["class"] == "stock-runtime"


def test_an_extra_text_tensor_is_a_layout_problem(tmp_path):
    t = tensors_for(LLAMA)
    t["model.layers.0.self_attn.extra_gate.weight"] = ("BF16", [4])
    d = run(tmp_path, snapshot(tmp_path, tensors=t), params(tmp_path))
    assert d["class"] == "full-port" and any("extra_gate" in p for p in d["stock_runtime"]["layout_problems"])


def test_a_layer_beyond_the_configured_count_is_a_layout_problem(tmp_path):
    t = tensors_for({**LLAMA, "num_hidden_layers": 3})
    d = run(tmp_path, snapshot(tmp_path, tensors=t), params(tmp_path))
    assert d["class"] == "full-port" and any("model.layers.2" in p for p in d["stock_runtime"]["layout_problems"])


def test_a_wrong_shape_is_a_layout_problem(tmp_path):
    t = tensors_for(LLAMA)
    t["model.layers.0.self_attn.k_proj.weight"] = ("BF16", [99, 64])
    d = run(tmp_path, snapshot(tmp_path, tensors=t), params(tmp_path))
    assert d["class"] == "full-port" and any("k_proj" in p and "shape" in p for p in d["stock_runtime"]["layout_problems"])


def test_a_quantized_dtype_is_a_layout_problem(tmp_path):
    t = tensors_for(LLAMA)
    t["model.layers.0.mlp.up_proj.weight"] = ("I8", [128, 64])
    d = run(tmp_path, snapshot(tmp_path, tensors=t), params(tmp_path))
    assert d["class"] == "full-port" and any("I8" in p for p in d["stock_runtime"]["layout_problems"])


def test_qwen2_layout_needs_the_attention_biases(tmp_path):
    q = {**LLAMA, "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "attention_bias": True}
    good = run(tmp_path, snapshot(tmp_path, cfg=q, tensors=tensors_for(q, qwen=True)), params(tmp_path, cfg=q))
    assert good["class"] == "stock-runtime"
    t = tensors_for(q, qwen=True)
    del t["model.layers.0.self_attn.q_proj.bias"]
    bad = run(tmp_path, snapshot(tmp_path, cfg=q, tensors=t, name="s2"), params(tmp_path, cfg=q))
    assert bad["class"] == "full-port" and any("q_proj.bias" in p for p in bad["stock_runtime"]["layout_problems"])


def test_an_architecture_tt_transformers_layout_does_not_cover_is_a_full_port(tmp_path):
    m = {**LLAMA, "architectures": ["MambaForCausalLM"], "model_type": "mamba"}
    d = run(tmp_path, snapshot(tmp_path, cfg=m, tensors=tensors_for(LLAMA)), params(tmp_path, cfg=m))
    assert d["class"] == "full-port" and any("MambaForCausalLM" in p for p in d["stock_runtime"]["layout_problems"])


def test_a_snapshot_with_no_weights_or_no_config_is_a_full_port_with_a_reason_and_not_a_crash(tmp_path):
    snap = tmp_path / "empty"
    snap.mkdir()
    d = run(tmp_path, snap, params(tmp_path))
    assert d["class"] == "full-port" and d["path_reasons"]
    snap2 = snapshot(tmp_path, name="s3")
    (snap2 / "config.json").unlink()
    d = run(tmp_path, snap2, params(tmp_path))
    assert d["class"] == "full-port" and d["path_reasons"]


def test_an_unreadable_header_is_a_reason(tmp_path):
    snap = snapshot(tmp_path)
    (snap / "model.safetensors").write_bytes(b"\xff" * 4)
    d = run(tmp_path, snap, params(tmp_path))
    assert d["class"] == "full-port" and any("header" in r or "safetensors" in r for r in d["path_reasons"])


def test_a_header_that_claims_an_enormous_size_is_refused(tmp_path):
    snap = snapshot(tmp_path)
    (snap / "model.safetensors").write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    d = run(tmp_path, snap, params(tmp_path))
    assert d["class"] == "full-port"


def test_the_parameter_count_comes_from_the_shapes(tmp_path):
    t = tensors_for(LLAMA)
    expect = 0
    for _dtype, shape in t.values():
        n = 1
        for s in shape:
            n *= s
        expect += n
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path))
    assert d["stock_runtime"]["params_b"] == pytest.approx(expect / 1e9, rel=1e-3, abs=1e-9)
    assert d["stock_runtime"]["params_b"] > 0


# ---- what the record says about the support and the host tree --------------------------------------------------

def test_the_record_carries_the_research_facts_and_the_tt_metal_tree(tmp_path):
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path),
            support={"via": "model_params", "matched": "meta-llama/Llama-3.1-8B-Instruct", "boards": ["N150", "P150"], "blackhole": True},
            tt_metal={"tag": "v0.80.0", "dirty": True})
    rec = d["stock_runtime"]
    assert rec["matched"] == "meta-llama/Llama-3.1-8B-Instruct" and rec["boards"] == ["N150", "P150"] and rec["blackhole"] is True
    assert rec["tt_metal"] == {"tag": "v0.80.0", "dirty": True}
    assert rec["params_dir"] == "models/tt_transformers/model_params/Llama-Test-1B-Instruct"        # no machine path


def test_a_readme_that_names_no_blackhole_board_is_a_hazard_and_not_a_block(tmp_path):
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path),
            support={"via": "model_params", "matched": "x/y", "boards": ["N150"], "blackhole": False})
    assert d["class"] == "stock-runtime"
    assert any("Blackhole" in h["finding"] for h in d["hazards"])
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path),
            support={"via": "model_params", "matched": "x/y", "boards": ["P150"], "blackhole": True})
    assert not any("no Blackhole" in h["finding"] for h in d["hazards"])


def test_a_dirty_tt_metal_tree_is_a_hazard(tmp_path):
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path), tt_metal={"tag": "v0.80.0", "dirty": True})
    assert any("uncommitted" in h["finding"] for h in d["hazards"])


def test_the_delta_has_no_home_path_or_host_name(tmp_path):
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path))
    text = json.dumps(d) + (tmp_path / "run" / "stages" / "0" / "evidence" / "stock-layout.json").read_text()
    assert str(tmp_path) not in text and "/home/" not in text


# ---- finding the parameters directory and the support row ------------------------------------------------------

def test_find_params_needs_a_config_in_the_directory(tmp_path):
    base = tmp_path / "tt-metal"
    p = params(tmp_path)
    assert sr.find_params(base, "Llama-Test-1B-Instruct") == p
    assert sr.find_params(base, "Nope") is None
    assert sr.find_params(base, "../etc") is None and sr.find_params(base, "a/b") is None and sr.find_params(base, "") is None
    (p / "config.json").unlink()
    assert sr.find_params(base, "Llama-Test-1B-Instruct") is None


def test_support_row_reads_the_research_report_and_prefers_the_params_match(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "research").mkdir(parents=True)
    rows = [{"name": "Llama 3.1 8B", "via": "readme", "matched": "unsloth/Llama-3.1-8B-Instruct", "boards": ["P150"], "blackhole": True},
            {"name": "Llama-3.1-8B-Instruct", "via": "model_params", "matched": "unsloth/Llama-3.1-8B-Instruct", "boards": [],
             "blackhole": None}]
    (run_dir / "research" / "base-candidates.json").write_text(json.dumps({"tt_transformers": rows}))
    got = sr.support_row(run_dir, "Llama-3.1-8B-Instruct")
    assert got["via"] == "model_params" and got["matched"] == "unsloth/Llama-3.1-8B-Instruct"
    # the README row's boards are merged in, because the params row has none
    assert got["boards"] == ["P150"] and got["blackhole"] is True


def test_support_row_with_no_report_gives_an_empty_row(tmp_path):
    got = sr.support_row(tmp_path / "run", "X")
    assert got == {"via": None, "matched": None, "boards": [], "blackhole": None}
    (tmp_path / "run" / "research").mkdir(parents=True)
    (tmp_path / "run" / "research" / "base-candidates.json").write_text("{")
    assert sr.support_row(tmp_path / "run", "X")["via"] is None


# ---- cases the first mutation run showed were not pinned -----------------------------------------------------------

def test_support_row_takes_the_row_of_the_asked_name(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "research").mkdir(parents=True)
    rows = [{"name": "Other-7B", "via": "model_params", "matched": "a/other", "boards": [], "blackhole": None},
            {"name": "Llama-3.1-8B-Instruct", "via": "model_params", "matched": "a/llama", "boards": [], "blackhole": None}]
    (run_dir / "research" / "base-candidates.json").write_text(json.dumps({"tt_transformers": rows}))
    assert sr.support_row(run_dir, "Llama-3.1-8B-Instruct")["matched"] == "a/llama"
    assert sr.support_row(run_dir, "Nope")["via"] is None


def test_support_row_without_a_readme_row_keeps_the_params_rows_boards(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "research").mkdir(parents=True)
    rows = [{"name": "X-1B", "via": "model_params", "matched": "a/x", "boards": ["P150"], "blackhole": True}]
    (run_dir / "research" / "base-candidates.json").write_text(json.dumps({"tt_transformers": rows}))
    got = sr.support_row(run_dir, "X-1B")
    assert got["boards"] == ["P150"] and got["blackhole"] is True


def test_rope_fields_the_runtime_does_not_read_are_not_compared(tmp_path):
    cfg = {**LLAMA, "rope_scaling": {**LLAMA["rope_scaling"], "beta_fast": 32}}
    d = run(tmp_path, snapshot(tmp_path, cfg=cfg, tensors=tensors_for(LLAMA)), params(tmp_path))
    assert d["class"] == "stock-runtime"


def test_a_header_cut_short_is_a_reason(tmp_path):
    snap = snapshot(tmp_path)
    blob = (snap / "model.safetensors").read_bytes()
    (snap / "model.safetensors").write_bytes(blob[: len(blob) // 2])
    d = run(tmp_path, snap, params(tmp_path))
    assert d["class"] == "full-port" and any("cut short" in r for r in d["path_reasons"])


def test_unknown_boards_are_a_hazard_like_wormhole_only_boards(tmp_path):
    d = run(tmp_path, snapshot(tmp_path), params(tmp_path), support={"via": None, "matched": None, "boards": [], "blackhole": None})
    assert any("no Blackhole board is named" in h["finding"] and "no board" in h["finding"] for h in d["hazards"])


def test_the_tt_metal_tree_facts_come_from_describe_and_status(monkeypatch):
    calls = []

    class Done:
        def __init__(self, out, rc=0):
            self.stdout, self.returncode = out, rc

    def fake(argv, **kw):
        calls.append(argv)
        return Done("v0.80.0\n") if argv[3] == "describe" else Done(" M models/x.py\n")
    monkeypatch.setattr(sr.subprocess, "run", fake)
    assert sr.tt_metal_tree("/t") == {"tag": "v0.80.0", "dirty": True}
    assert all(c[:3] == ["git", "-C", "/t"] for c in calls) and {c[3] for c in calls} == {"describe", "status"}
    monkeypatch.setattr(sr.subprocess, "run", lambda argv, **kw: Done("v0.80.0\n" if argv[3] == "describe" else ""))
    assert sr.tt_metal_tree("/t") == {"tag": "v0.80.0", "dirty": False}


def test_the_tt_metal_tree_facts_are_none_when_the_tool_cannot_answer(monkeypatch):
    class Done:
        stdout, returncode = "", 128
    monkeypatch.setattr(sr.subprocess, "run", lambda argv, **kw: Done())
    assert sr.tt_metal_tree("/t") == {"tag": None, "dirty": None}

    def boom(argv, **kw):
        raise OSError("no such program")
    monkeypatch.setattr(sr.subprocess, "run", boom)
    assert sr.tt_metal_tree("/t") == {"tag": None, "dirty": None}


def test_support_row_takes_the_readme_boards_of_the_same_matched_model_only(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "research").mkdir(parents=True)
    rows = [{"name": "Llama 3.1 8B", "via": "readme", "matched": "m/eight", "boards": ["P150"], "blackhole": True},
            {"name": "Llama-3.2-1B-Instruct", "via": "model_params", "matched": "m/one", "boards": [], "blackhole": None},
            {"name": "Llama 3.2 1B", "via": "readme", "matched": "m/one", "boards": ["N150"], "blackhole": False}]
    (run_dir / "research" / "base-candidates.json").write_text(json.dumps({"tt_transformers": rows}))
    got = sr.support_row(run_dir, "Llama-3.2-1B-Instruct")
    assert got["boards"] == ["N150"] and got["blackhole"] is False
    (run_dir / "research" / "base-candidates.json").write_text(json.dumps({"tt_transformers": [rows[0], rows[1]]}))
    got = sr.support_row(run_dir, "Llama-3.2-1B-Instruct")
    assert got["boards"] == [] and got["blackhole"] is None
