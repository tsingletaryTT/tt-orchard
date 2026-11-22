# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Layer-type rope and the sliding window in the generated attention (issue 790).

`microsoft/Aion-1.0-Instruct-2.3B` gives `rope_parameters` keyed by layer type (`full_attention`: YaRN with theta 32000; `sliding_attention`: the default
type with theta 32000) and lists `layer_types` of `sliding_attention` and `full_attention` with `sliding_window` 2048. The generator took that dict as one flat
rope block. It found no `rope_type` (so it read `default`) and no `rope_theta` (so it fell back to 10000.0), and it wrote causal attention over the whole
prefix for every layer. A port built that way scored top-1 0.875 and fidelity 9.249 at 51 tokens and lost the passkey at 8,192 tokens.

Step 1 (this file's first section): the generator refuses what it does not write. Step 2 and step 3 replace the refusal with per-layer-type rotary tables
and a sliding-window mask; their tests are in the later sections of this file and compare the generated port with a torch forward at a length above the window.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import (STD_CONFIG, STD_EXCERPTS, headers_of, import_port, std_construction, std_generate, tiny_qwen4,
                                write_checkpoint)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def aion_config() -> dict:
    """Aion's own config.json (every key as the model card ships it)."""
    return json.loads((FIXTURES / "aion-config.json").read_text(encoding="utf-8"))


def tiny_aion_config(*, window=8, original=16) -> dict:
    """The small standard decoder with Aion's structure: layer types, a sliding window, and `rope_parameters` keyed by layer type (no top-level rope_theta)."""
    cfg = copy.deepcopy(STD_CONFIG)
    cfg.pop("rope_theta")
    cfg.update({"model_type": "aion_instruct", "num_hidden_layers": 3, "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
                "sliding_window": window,
                "rope_parameters": {"full_attention": {"rope_type": "yarn", "rope_theta": 32000, "factor": 16.0, "beta_fast": 32.0, "beta_slow": 1.0,
                                                       "original_max_position_embeddings": original},
                                    "sliding_attention": {"rope_type": "default", "rope_theta": 32000}}})
    return cfg


def generate_with_weights(cfg):
    """(text, plan, weights) for the small decoder with this config's layer count and head sizes."""
    weights = _weights(cfg)
    n = cfg["num_hidden_layers"]
    construction = std_construction()
    for entry in construction["classes"]:        # the construction lists the layer count of each module group
        for variant in entry["variants"]:
            if variant["count"] == 2 and any("{i}" in p for p in variant["paths"]):
                variant["count"] = n
    text, p = pt.generate(config=cfg, construction=construction, tensors=headers_of(weights), excerpts=STD_EXCERPTS)
    return text, p, weights


def generate(cfg):
    text, p, _ = generate_with_weights(cfg)
    return text, p


def _weights(cfg, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    h, nh, nkv, hd, inter, v = (cfg[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size"))

    def r(*shape, scale=0.2):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    w = {"model.embed_tokens.weight": r(v, h, scale=1.0), "model.norm.weight": r(h, scale=1.0) + 1, "lm_head.weight": r(v, h, scale=0.5)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        w.update({f"{p}.input_layernorm.weight": r(h, scale=0.3) + 1, f"{p}.post_attention_layernorm.weight": r(h, scale=0.3) + 1,
                  f"{p}.self_attn.q_proj.weight": r(nh * hd, h), f"{p}.self_attn.k_proj.weight": r(nkv * hd, h),
                  f"{p}.self_attn.v_proj.weight": r(nkv * hd, h), f"{p}.self_attn.o_proj.weight": r(h, nh * hd),
                  f"{p}.self_attn.q_norm.weight": r(hd, scale=0.3) + 1, f"{p}.self_attn.k_norm.weight": r(hd, scale=0.3) + 1,
                  f"{p}.mlp.gate_proj.weight": r(inter, h), f"{p}.mlp.up_proj.weight": r(inter, h), f"{p}.mlp.down_proj.weight": r(h, inter)})
    return w


# ---- step 1: the generator refuses what it does not write ------------------------------------------------------------------------------------

def test_aions_real_config_is_keyed_by_layer_type_and_has_a_sliding_window():
    section = aion_config()
    assert pt.rope_keyed_by_layer_type(section) is True
    assert pt.sliding_window_of(section) == 2048


def test_a_flat_rope_block_is_not_keyed_by_layer_type():
    assert pt.rope_keyed_by_layer_type({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}) is False
    assert pt.rope_keyed_by_layer_type({"rope_theta": 1e6}) is False
    assert pt.rope_keyed_by_layer_type({"rope_parameters": {}}) is False


def test_a_block_with_plain_values_and_one_nested_dict_is_still_a_flat_block():
    """Only a block whose every value is a dict is keyed by layer type: a flat block may carry one nested setting."""
    assert pt.rope_keyed_by_layer_type({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6, "extra": {"a": 1}}}) is False


def test_a_sliding_window_counts_only_when_a_layer_is_marked_sliding():
    assert pt.sliding_window_of({"sliding_window": 4096, "layer_types": ["full_attention", "linear_attention"]}) is None
    assert pt.sliding_window_of({"sliding_window": 4096}) is None
    assert pt.sliding_window_of({"sliding_window": None, "layer_types": ["sliding_attention"]}) is None
    assert pt.sliding_window_of({"sliding_window": 8, "layer_types": ["sliding_attention", "full_attention"]}) == 8


def test_the_generator_never_writes_theta_10000_for_a_rope_block_it_cannot_read():
    """The failure of the Aion run: a layer-type-keyed block became `theta: 10000.0` with no refusal."""
    text, p = generate(tiny_aion_config())
    assert "10000.0" not in text.replace("1e-06", ""), "the generated port holds the fallback theta"
    assert p.rotary is None


def test_a_layer_type_keyed_rope_block_is_generated_with_one_block_per_layer_type():
    """Step 1 refused this shape; step 2 writes it (tables per layer type). The theta of the fallback never appears."""
    cfg = tiny_aion_config()
    cfg["sliding_window"] = None                                # only the rope keying is under test here
    text, p = generate(cfg)
    assert p.stubs == [] and p.attention is not None
    assert sorted(p.attention["rope"]) == ["full_attention", "sliding_attention"] and "theta" not in p.attention
    assert "10000.0" not in text


def test_a_sliding_window_with_a_flat_rope_block_is_generated_with_the_window_and_one_table_pair():
    """The window does not need a block per layer type: a flat rope block keeps one table pair, and the layer types only choose the mask."""
    cfg = tiny_aion_config()
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 32000}
    text, p = generate(cfg)
    assert p.stubs == [] and p.attention is not None
    assert p.attention["window"] == 8 and p.attention["theta"] == 32000.0 and "rope" not in p.attention
    assert "cos, sin = cos[layer_type], sin[layer_type]" not in text


def test_a_flat_rope_model_with_a_sliding_window_matches_a_banded_torch_forward(monkeypatch, tmp_path):
    cfg = window_config(8)
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 32000}
    ids = [(5 * i + 1) % 50 for i in range(40)]
    mod, model, weights, got = run_port(monkeypatch, tmp_path, cfg, ids)
    keyed = dict(cfg, rope_parameters={t: dict(cfg["rope_parameters"]) for t in set(cfg["layer_types"])})      # the reference looks a block up by layer type
    want = torch_forward(weights, keyed, ids)
    assert torch.allclose(got, want, atol=2e-3, rtol=2e-3), (got - want).abs().max()


# ---- step 2: one rotary table pair per layer type ---------------------------------------------------------------------------------------------

def reference_rope(cfg_section: dict, layer_type: str, hd: int):
    """(inverse frequencies, attention scaling) for one layer type, from transformers' own functions on a config class that holds layer-type rope blocks
    (the Gemma 3 text config carries `rope_parameters` keyed by layer type, as Aion's does). The default type is Aion's `compute_default_rope_parameters`."""
    from transformers import Gemma3TextConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    block = cfg_section["rope_parameters"][layer_type]
    if block["rope_type"] == "default":
        return 1.0 / (block["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd)), 1.0
    original = block["original_max_position_embeddings"]
    cfg = Gemma3TextConfig(head_dim=hd, hidden_size=hd * cfg_section["num_attention_heads"], num_attention_heads=cfg_section["num_attention_heads"],
                           num_key_value_heads=cfg_section["num_key_value_heads"], num_hidden_layers=len(cfg_section["layer_types"]),
                           max_position_embeddings=int(original * block["factor"]), layer_types=cfg_section["layer_types"],
                           rope_parameters=copy.deepcopy(cfg_section["rope_parameters"]), sliding_window=cfg_section.get("sliding_window"))
    return ROPE_INIT_FUNCTIONS[block["rope_type"]](cfg, layer_type=layer_type)


def aion_rope_config(*, hd=128, original=2048, window=None) -> dict:
    """Aion's rope blocks and layer pattern at a head size of `hd` (the real model has 128), small enough to generate and run on the CPU stand-in."""
    cfg = tiny_aion_config(window=window if window else 8, original=original)
    cfg.update({"head_dim": hd, "num_attention_heads": 1, "num_key_value_heads": 1})
    if not window:
        cfg["sliding_window"] = None
    return cfg


def model_object():
    return {"n_devices": 1, "mesh_device": fake.MeshDevice(1), "ck": None, "dev": {}, "host": {}, "ids": {}}


@pytest.mark.parametrize("n", [64, 2048, 4096, 8192])
@pytest.mark.parametrize("hd", [16, 128])
def test_the_generated_rope_tables_match_transformers_for_each_layer_type(monkeypatch, tmp_path, hd, n):
    """Aion's own rope blocks (YaRN with factor 16 over 2048 original positions for the full layers; the default type for the sliding layers, both with theta
    32000), at the real head size of 128 and at a small one, and at lengths up to 8,192 where YaRN has long left the original range."""
    cfg = aion_rope_config(hd=hd, window=None)
    cfg["rope_parameters"]["full_attention"]["original_max_position_embeddings"] = 2048
    text, p = generate(cfg)
    assert p.attention is not None and p.stubs == [], [s.reason for s in p.stubs]
    mod = import_port(monkeypatch, text, tmp_path)
    cos, sin = mod._rope_tables(model_object(), n)
    assert sorted(cos) == sorted(sin) == ["full_attention", "sliding_attention"]
    for layer_type in ("full_attention", "sliding_attention"):
        inv, scaling = reference_rope(cfg, layer_type, hd)
        emb = torch.cat([torch.arange(n).float()[:, None] * inv[None]] * 2, dim=-1)
        want_cos, want_sin = (emb.cos() * scaling).to(torch.bfloat16).float(), (emb.sin() * scaling).to(torch.bfloat16).float()
        got_cos, got_sin = cos[layer_type].t.float().reshape(n, hd), sin[layer_type].t.float().reshape(n, hd)
        assert torch.allclose(got_cos, want_cos, atol=0.01), (layer_type, (got_cos - want_cos).abs().max())
        assert torch.allclose(got_sin, want_sin, atol=0.01), (layer_type, (got_sin - want_sin).abs().max())
    assert not torch.allclose(cos["full_attention"].t.float(), cos["sliding_attention"].t.float(), atol=0.01), "the two layer types got one table"


@pytest.mark.parametrize("options", [{"attention_factor": 1.5}, {"mscale": 1.2, "mscale_all_dim": 0.9}, {"beta_fast": 16.0, "beta_slow": 2.0},
                                     {"beta_fast": 64.0, "beta_slow": 0.5, "factor": 4.0}])
def test_the_optional_yarn_settings_follow_transformers(monkeypatch, tmp_path, options):
    cfg = aion_rope_config(hd=64, window=None)
    cfg["rope_parameters"]["full_attention"].update(options)
    text, p = generate(cfg)
    assert p.attention is not None, [s.reason for s in p.stubs]
    mod = import_port(monkeypatch, text, tmp_path)
    n = 4096
    cos, sin = mod._rope_tables(model_object(), n)
    inv, scaling = reference_rope(cfg, "full_attention", 64)
    emb = torch.cat([torch.arange(n).float()[:, None] * inv[None]] * 2, dim=-1)
    assert torch.allclose(cos["full_attention"].t.float().reshape(n, 64), (emb.cos() * scaling).to(torch.bfloat16).float(), atol=0.01), options
    assert torch.allclose(sin["full_attention"].t.float().reshape(n, 64), (emb.sin() * scaling).to(torch.bfloat16).float(), atol=0.01), options


def test_a_yarn_option_that_is_not_a_positive_number_leaves_a_named_stub():
    for key, value in (("attention_factor", "high"), ("beta_fast", -1.0), ("mscale", float("nan"))):
        cfg = aion_rope_config(hd=16, window=None)
        cfg["rope_parameters"]["full_attention"][key] = value
        text, p = generate(cfg)
        reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
        assert p.attention is None and reasons and all(key in r for r in reasons), (key, reasons)


def test_the_yarn_attention_scaling_is_applied_to_the_full_attention_tables_only(monkeypatch, tmp_path):
    cfg = aion_rope_config(hd=16, window=None)
    text, p = generate(cfg)
    mod = import_port(monkeypatch, text, tmp_path)
    cos, _sin = mod._rope_tables(model_object(), 32)
    scaling = reference_rope(cfg, "full_attention", 16)[1]
    assert scaling > 1.2                                                         # 0.1 * ln(16) + 1
    assert float(cos["full_attention"].t.float().abs().max()) > 1.2              # position 0 has cos 1 times the scaling
    assert float(cos["sliding_attention"].t.float().abs().max()) <= 1.0 + 1e-2


def test_the_attention_picks_its_tables_by_the_layer_type_of_its_layer():
    text, p = generate(aion_rope_config(hd=16, window=None))
    assert p.attention["layer_types"] == ["sliding_attention", "sliding_attention", "full_attention"]
    assert p.attention["rope"]["full_attention"]["kind"] == "yarn" and p.attention["rope"]["sliding_attention"]["kind"] == "default"
    assert "theta" not in p.attention                                              # no single theta: every layer type holds its own
    assert 'ATTENTION["layer_types"][' in text and "_rope_inv_freq" in text
    from orchard import thin_repair
    for helper in ("_rope_inv_freq", "_rope_tables"):
        assert thin_repair._holds_generated_mark(text, helper), helper             # a failure inside a table helper is never a repair target


@pytest.mark.parametrize("what, mutate, reason", [
    ("an unsupported rope type", lambda c: c["rope_parameters"]["full_attention"].update(rope_type="longrope"), "longrope"),
    ("a yarn block with no factor", lambda c: c["rope_parameters"]["full_attention"].pop("factor"), "factor"),
    ("a yarn block with no original length", lambda c: c["rope_parameters"]["full_attention"].pop("original_max_position_embeddings"), "original_max_position_embeddings"),
    ("a layer type with no rope block", lambda c: c["rope_parameters"].pop("sliding_attention"), "sliding_attention"),
    ("a block with no theta", lambda c: c["rope_parameters"]["sliding_attention"].pop("rope_theta"), "rope_theta"),
    ("a partial rotary factor", lambda c: c["rope_parameters"]["sliding_attention"].update(partial_rotary_factor=0.5), "partial_rotary_factor"),
    ("a layer list that is shorter than the layers", lambda c: c.update(layer_types=c["layer_types"][:2]), "layer_types"),
    ("a layer type that is not a string", lambda c: c.update(layer_types=["sliding_attention", 3, "full_attention"]), "layer_types"),
])
def test_a_layer_keyed_rope_block_the_tables_cannot_follow_leaves_a_named_stub(what, mutate, reason):
    cfg = aion_rope_config(hd=16, window=None)
    mutate(cfg)
    text, p = generate(cfg)
    assert p.attention is None, what
    reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
    assert reasons and all(reason in r for r in reasons), (what, reasons)


def test_a_flat_rope_model_is_generated_byte_for_byte_as_before():
    """The standard small decoder and the tiny Qwen4 rehearsal model. The hashes moved once, when the generator began to write the one line
    `TOKEN_CHOICE = "host-row"` (tests/test_port_template_token_choice.py holds the hashes from before that line and checks the text is the
    same without it)."""
    assert hashlib.sha256(std_generate()[0].encode()).hexdigest() == "6efab8c58df94df3e5872c33ef536685ec7954d4b9a1346f732436c83e3f9ccf"
    assert hashlib.sha256(tiny_qwen4()[0].encode()).hexdigest() == "aa1dceb6c5204afce761ecba00ec2b0816d14736062f1baae86164738cf2c02d"


# ---- the generated port against a torch forward at a length above the YaRN original length -----------------------------------------------------

def torch_forward(weights: dict, cfg: dict, ids: list) -> torch.Tensor:
    """A float32 forward of the small decoder in the Aion form: per-layer-type rotary tables (cos and sin rounded to bf16 as the port uploads them),
    per-head RMSNorm on q and k, and, for a layer marked sliding_attention with a window, attention over the last `window` positions only."""
    nh, nkv, hd, eps = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    w = {k: t.float() for k, t in weights.items()}
    bf = lambda t: t.to(torch.bfloat16).float()
    rms = lambda x, g: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
    s = len(ids)
    tables = {}
    for layer_type in sorted(set(cfg["layer_types"])):
        inv, scaling = reference_rope(cfg, layer_type, hd)
        emb = torch.cat([torch.arange(s).float()[:, None] * inv[None]] * 2, dim=-1)
        tables[layer_type] = (bf(emb.cos() * scaling)[None], bf(emb.sin() * scaling)[None])
    rot = lambda t: torch.cat((-t[..., hd // 2:], t[..., : hd // 2]), -1)
    x = w["model.embed_tokens.weight"][torch.tensor(ids)]
    window = cfg.get("sliding_window")
    for i in range(cfg["num_hidden_layers"]):
        layer_type = cfg["layer_types"][i]
        cos, sin = tables[layer_type]
        p = f"model.layers.{i}"
        h = rms(x, w[f"{p}.input_layernorm.weight"])
        q = (h @ w[f"{p}.self_attn.q_proj.weight"].T).view(s, nh, hd)
        k = (h @ w[f"{p}.self_attn.k_proj.weight"].T).view(s, nkv, hd)
        v = (h @ w[f"{p}.self_attn.v_proj.weight"].T).view(s, nkv, hd).transpose(0, 1)
        q = rms(q, w[f"{p}.self_attn.q_norm.weight"]).transpose(0, 1)
        k = rms(k, w[f"{p}.self_attn.k_norm.weight"]).transpose(0, 1)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        k, v = k.repeat_interleave(nh // nkv, 0), v.repeat_interleave(nh // nkv, 0)
        sc = (q @ k.transpose(-2, -1)) * hd ** -0.5
        pos = torch.arange(s)
        banned = pos[None, :] > pos[:, None]
        if window and layer_type == "sliding_attention":
            banned = banned | (pos[None, :] <= pos[:, None] - window)
        a = (torch.softmax(sc.masked_fill(banned, float("-inf")), -1) @ v).transpose(0, 1).reshape(s, nh * hd)
        x = x + a @ w[f"{p}.self_attn.o_proj.weight"].T
        h = rms(x, w[f"{p}.post_attention_layernorm.weight"])
        x = x + (torch.nn.functional.silu(h @ w[f"{p}.mlp.gate_proj.weight"].T) * (h @ w[f"{p}.mlp.up_proj.weight"].T)) @ w[f"{p}.mlp.down_proj.weight"].T
    return (rms(x, w["model.norm.weight"]) @ w["lm_head.weight"].T)[-1]


def run_port(monkeypatch, tmp_path, cfg, ids, *, n_devices=1):
    text, p, weights = generate_with_weights(cfg)
    assert p.attention is not None and p.stubs == [], [s.reason for s in p.stubs]
    mod = import_port(monkeypatch, text, tmp_path)
    model = mod.load(str(write_checkpoint(tmp_path, weights)), fake.MeshDevice(n_devices))
    return mod, model, weights, mod.prefill(model, ids)


def test_the_port_matches_a_torch_forward_above_the_yarn_original_length_with_per_layer_type_rope(monkeypatch, tmp_path):
    """48 tokens, original length 16: the full layers are past the point where YaRN changes the frequencies. With sliding_window None every layer attends
    over the whole prefix, so this isolates the rotary tables. Before this change the generator refused (a stub) for the keyed block."""
    cfg = aion_rope_config(hd=16, original=16, window=None)
    ids = [(7 * i + 3) % 50 for i in range(48)]
    mod, model, weights, got = run_port(monkeypatch, tmp_path, cfg, ids)
    want = torch_forward(weights, cfg, ids)
    assert torch.allclose(got, want, atol=2e-3, rtol=2e-3), (got - want).abs().max()
    assert int(got.argmax()) == int(want.argmax())
    nxt = mod.decode_step(model, int(got.argmax()), len(ids))
    assert torch.allclose(nxt, torch_forward(weights, cfg, ids + [int(got.argmax())]), atol=2e-3, rtol=2e-3)


def test_the_one_table_the_old_generator_wrote_would_have_failed_that_comparison(monkeypatch, tmp_path):
    """The comparison can tell the two apart: the same weights with every layer on the sliding type's table (what a flat read of the block gave, apart from
    the fallback theta) differ from the reference's per-type result by far more than the tolerance."""
    cfg = aion_rope_config(hd=16, original=16, window=None)
    ids = [(7 * i + 3) % 50 for i in range(48)]
    weights = _weights(cfg)
    flat = copy.deepcopy(cfg)
    flat["layer_types"] = ["sliding_attention"] * 3
    right, wrong = torch_forward(weights, cfg, ids), torch_forward(weights, flat, ids)
    assert not torch.allclose(right, wrong, atol=2e-3, rtol=2e-3), (right - wrong).abs().max()


# ---- step 3: the sliding window ---------------------------------------------------------------------------------------------------------------

def window_config(window=8):
    return aion_rope_config(hd=16, original=16, window=window)


def test_a_sliding_window_is_generated_and_recorded():
    text, p = generate(window_config(8))
    assert p.stubs == [] and p.attention is not None and p.attention["window"] == 8
    assert "_window_mask" in text and "attn_mask=_window_mask(model, seq)" in text
    from orchard import thin_repair
    assert thin_repair._holds_generated_mark(text, "_window_mask")     # a failure inside the mask helper is never a repair target
    flat, _ = generate(aion_rope_config(hd=16, original=16, window=None))
    assert "_window_mask" not in flat and "window" not in generate(aion_rope_config(hd=16, original=16, window=None))[1].attention


@pytest.mark.parametrize("n_devices", [1, 2])
def test_the_port_matches_a_banded_torch_forward_above_the_window_and_above_the_yarn_length(monkeypatch, tmp_path, n_devices):
    """48 tokens against a window of 8 and a yarn original length of 16: the first two layers (sliding) see only the last 8 positions, the third (full) sees
    all of them, and each type has its own rotary table. Before this change the generator refused the window (a stub)."""
    cfg = window_config(8)
    ids = [(7 * i + 3) % 50 for i in range(48)]
    mod, model, weights, got = run_port(monkeypatch, tmp_path, cfg, ids, n_devices=n_devices)
    want = torch_forward(weights, cfg, ids)
    assert torch.allclose(got, want, atol=2e-3, rtol=2e-3), (got - want).abs().max()
    assert int(got.argmax()) == int(want.argmax())
    for step in range(2):                                              # decode runs the whole sequence again: the window must hold at every length
        tok = int(got.argmax())
        got = mod.decode_step(model, tok, len(ids))
        ids = ids + [tok]
        want = torch_forward(weights, cfg, ids)
        assert torch.allclose(got, want, atol=2e-3, rtol=2e-3), (step, (got - want).abs().max())


def test_the_comparison_can_tell_a_window_from_none():
    cfg = window_config(8)
    ids = [(7 * i + 3) % 50 for i in range(48)]
    weights = _weights(cfg)
    open_cfg = dict(cfg, sliding_window=None)
    assert not torch.allclose(torch_forward(weights, cfg, ids), torch_forward(weights, open_cfg, ids), atol=2e-3, rtol=2e-3)


def test_each_sliding_layer_gets_the_mask_and_each_full_layer_the_causal_flag(monkeypatch, tmp_path):
    cfg = window_config(8)
    text, p, weights = generate_with_weights(cfg)
    mod = import_port(monkeypatch, text, tmp_path)
    import sys
    ttnn = sys.modules["ttnn"]
    seen = []
    real = ttnn.transformer.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        seen.append((kw.get("attn_mask") is not None, kw.get("is_causal", True)))
        return real(q, k, v, **kw)

    monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", spy)
    model = mod.load(str(write_checkpoint(tmp_path, weights)), fake.MeshDevice(1))
    mod.prefill(model, list(range(1, 41)))                              # 40 tokens pad to 64, above the window of 8
    assert seen == [(True, False), (True, False), (False, True)]        # layers: sliding, sliding, full
    seen.clear()
    mod.prefill(model, [1, 2, 3, 4, 5])                                 # padded to 32: still above the window of 8
    assert seen == [(True, False), (True, False), (False, True)]


def test_a_sequence_inside_the_window_keeps_the_causal_call(monkeypatch, tmp_path):
    cfg = window_config(64)
    text, p, weights = generate_with_weights(cfg)
    mod = import_port(monkeypatch, text, tmp_path)
    import sys
    ttnn = sys.modules["ttnn"]
    seen = []
    real = ttnn.transformer.scaled_dot_product_attention
    monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", lambda q, k, v, **kw: (seen.append((kw.get("attn_mask") is not None, kw.get("is_causal", True))), real(q, k, v, **kw))[1])
    model = mod.load(str(write_checkpoint(tmp_path, weights)), fake.MeshDevice(1))
    mod.prefill(model, list(range(1, 41)))                              # 40 tokens pad to 64, which the window of 64 covers
    assert seen == [(False, True)] * 3


def test_the_window_mask_allows_a_query_and_the_window_minus_one_positions_before_it(monkeypatch, tmp_path):
    text, p = generate(window_config(8))
    mod = import_port(monkeypatch, text, tmp_path)
    model = model_object()
    mask = mod._window_mask(model, 32).t.float().reshape(32, 32)
    for i in (0, 5, 7, 8, 20, 31):
        allowed = {j for j in range(32) if float(mask[i, j]) == 0.0}
        assert allowed == set(range(max(0, i - 7), i + 1)), (i, sorted(allowed))
    assert float(mask.min()) == float(torch.tensor(-30000.0).to(torch.bfloat16))          # bf16 holds -30000 as -29952, as the other generated masks do
    assert set(mask.unique().tolist()) == {0.0, float(torch.tensor(-30000.0).to(torch.bfloat16))}
    assert mod._window_mask(model, 32) is model["window_mask"][1]       # the same length reuses the mask
    again = mod._window_mask(model, 64)
    assert model["window_mask"][0] == 64 and again is model["window_mask"][1]


def test_close_frees_the_window_mask(monkeypatch, tmp_path):
    cfg = window_config(8)
    mod, model, weights, _ = run_port(monkeypatch, tmp_path, cfg, list(range(1, 41)))
    assert model.get("window_mask") is not None
    mod.close(model)
    assert model.get("window_mask") is None and model["dev"] == {}


@pytest.mark.parametrize("window", ["8", 8.5, True, -3, 0, [], {}, "x\0y", 10 ** 20, 10 ** 400, -(10 ** 400), float("inf"), float("nan")])
def test_a_sliding_window_that_is_not_a_positive_whole_number_leaves_a_named_stub(window):
    """A layer marked sliding with a window the generator cannot read must not turn into full attention without a word."""
    cfg = window_config(8)
    cfg["sliding_window"] = window
    text, p = generate(cfg)
    reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
    assert p.attention is None and reasons and all("sliding_window" in r for r in reasons), (window, reasons)


def test_a_missing_or_null_sliding_window_means_full_attention_as_in_the_reference():
    cfg = window_config(8)
    cfg["sliding_window"] = None
    assert generate(cfg)[1].attention is not None
    cfg.pop("sliding_window")
    assert generate(cfg)[1].attention is not None


def test_a_new_padded_length_frees_the_previous_window_mask(monkeypatch, tmp_path):
    text, p = generate(window_config(8))
    mod = import_port(monkeypatch, text, tmp_path)
    import sys
    ttnn = sys.modules["ttnn"]
    freed = []
    real = ttnn.deallocate
    monkeypatch.setattr(ttnn, "deallocate", lambda t: (freed.append(t), real(t))[1])
    model = model_object()
    first = mod._window_mask(model, 32)
    assert freed == []
    mod._window_mask(model, 32)
    assert freed == []                                                  # the same length keeps its mask
    mod._window_mask(model, 64)
    assert freed == [first]                                             # a new length frees the old one


@pytest.mark.parametrize("layer_types", [["sliding_attention", "full_attention"], ["sliding_attention", 3, "full_attention"]])
def test_a_sliding_window_with_layer_types_that_do_not_cover_the_layers_leaves_a_named_stub(layer_types):
    cfg = window_config(8)
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 32000}
    cfg["layer_types"] = layer_types
    text, p = generate(cfg)
    reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
    assert p.attention is None, layer_types
    assert reasons and all("sliding" in r or "layer_types" in r for r in reasons), (layer_types, reasons)


# ---- hostile values in a layer-keyed rope block (checklist item 1): a named stub, never an exception ------------------------------------------------

HOSTILE = [[], {}, "", "x\0y", 10 ** 400, -(10 ** 400), True, float("nan"), float("inf"), -1, 0]


@pytest.mark.parametrize("key", ["rope_theta", "factor", "original_max_position_embeddings"])
@pytest.mark.parametrize("value", HOSTILE + [None], ids=lambda v: repr(v)[:14])
def test_a_hostile_required_value_in_a_layer_keyed_block_is_a_named_stub(key, value):
    cfg = aion_rope_config(hd=16, window=None)
    cfg["rope_parameters"]["full_attention"][key] = value
    text, p = generate(cfg)
    reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
    assert p.attention is None and reasons and all(key in r for r in reasons), (key, value, reasons)


@pytest.mark.parametrize("key", ["beta_fast", "beta_slow", "attention_factor", "mscale", "mscale_all_dim"])
@pytest.mark.parametrize("value", HOSTILE, ids=lambda v: repr(v)[:14])
def test_a_hostile_optional_value_in_a_layer_keyed_block_is_a_named_stub(key, value):
    cfg = aion_rope_config(hd=16, window=None)
    cfg["rope_parameters"]["full_attention"][key] = value
    text, p = generate(cfg)
    reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
    assert p.attention is None and reasons and all(key in r for r in reasons), (key, value, reasons)


@pytest.mark.parametrize("value", [[], {}, "", "x\0y", 10 ** 400, True, float("nan"), 0.5, 2])
def test_a_partial_rotary_factor_other_than_one_in_a_layer_keyed_block_is_a_named_stub(value):
    cfg = aion_rope_config(hd=16, window=None)
    cfg["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = value
    text, p = generate(cfg)
    reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
    assert p.attention is None and reasons and all("partial_rotary_factor" in r for r in reasons), (value, reasons)


@pytest.mark.parametrize("value", [None, [], {}, "", "x\0y", 10 ** 400, True, 3.5])
def test_a_hostile_rope_type_or_layer_type_name_is_a_named_stub(value):
    cfg = aion_rope_config(hd=16, window=None)
    cfg["rope_parameters"]["full_attention"]["rope_type"] = value
    text, p = generate(cfg)
    assert p.attention is None and p.stubs
    cfg = aion_rope_config(hd=16, window=None)
    cfg["layer_types"] = ["sliding_attention", value, "full_attention"]
    text, p = generate(cfg)
    assert p.attention is None and p.stubs


def test_a_flat_block_with_an_integer_too_large_for_a_float_does_not_crash_the_plan_for_a_layer_keyed_config():
    cfg = aion_rope_config(hd=16, window=None)
    cfg["rope_parameters"]["sliding_attention"]["rope_theta"] = 10 ** 400
    text, p = generate(cfg)
    assert p.attention is None


@pytest.mark.parametrize("window", [1, 2, 31, 40, 64])
def test_the_port_matches_a_banded_torch_forward_for_small_and_large_windows(monkeypatch, tmp_path, window):
    """40 tokens pad to 64. A window of 1 sees only the query itself; 31 and 40 sit under the padded length (the mask is used); 64 covers it (the causal call)."""
    cfg = window_config(window)
    ids = [(11 * i + 2) % 50 for i in range(40)]
    mod, model, weights, got = run_port(monkeypatch, tmp_path, cfg, ids)
    want = torch_forward(weights, cfg, ids)
    assert torch.allclose(got, want, atol=2e-3, rtol=2e-3), (window, (got - want).abs().max())


# ---- a config value is never echoed as it came: the port must still parse ---------------------------------------------------------------------

INJECTION = ['\'"""\nraise SystemExit(7)\n"""', "\\", "a'b\"c", "x" * 10_000, "line\nbreak", "tab\tx"]


def assert_a_clean_port(text, p):
    import ast
    tree = ast.parse(text)                                                       # a SyntaxError here is the failure
    assert not any(isinstance(n, ast.Raise) and "SystemExit" in ast.unparse(n) for n in ast.walk(tree)), "hostile text became code"
    assert "SystemExit" not in "\n".join(s.reason for s in p.stubs)
    assert all(len(s.reason) < 600 for s in p.stubs)


@pytest.mark.parametrize("value", INJECTION, ids=lambda v: repr(v)[:12])
def test_a_hostile_string_for_sliding_window_leaves_a_port_that_parses(value):
    cfg = window_config(8)
    cfg["sliding_window"] = value
    text, p = generate(cfg)
    assert p.attention is None and p.stubs
    assert_a_clean_port(text, p)


@pytest.mark.parametrize("value", INJECTION, ids=lambda v: repr(v)[:12])
def test_a_hostile_rope_type_leaves_a_port_that_parses(value):
    cfg = aion_rope_config(hd=16, window=None)
    cfg["rope_parameters"]["full_attention"]["rope_type"] = value
    text, p = generate(cfg)
    assert p.attention is None and p.stubs
    assert_a_clean_port(text, p)
    flat = aion_rope_config(hd=16, window=None)
    flat["rope_parameters"] = {"rope_type": value, "rope_theta": 1000.0}
    text, p = generate(flat)
    assert_a_clean_port(text, p)


@pytest.mark.parametrize("value", INJECTION, ids=lambda v: repr(v)[:12])
def test_a_hostile_layer_type_name_leaves_a_port_that_parses(value):
    cfg = aion_rope_config(hd=16, window=None)
    cfg["layer_types"] = ["sliding_attention", value, "full_attention"]
    text, p = generate(cfg)                                                     # no block for that name: a stub that does not echo it
    assert p.attention is None and p.stubs
    assert_a_clean_port(text, p)
    cfg["rope_parameters"][value] = dict(cfg["rope_parameters"]["full_attention"])
    text, p = generate(cfg)                                                     # with a block: generated, the name a repr literal in ATTENTION
    assert_a_clean_port(text, p)


def test_shown_prints_plain_names_and_small_numbers_and_names_the_type_of_anything_else():
    assert pt._shown("full_attention") == "'full_attention'" and pt._shown(8) == "8" and pt._shown(0.5) == "0.5" and pt._shown(None) == "None"
    assert pt._shown(True) == "True"
    assert pt._shown([1]) == "<a list>" and pt._shown({"a": 1}) == "<a dict>" and pt._shown(10 ** 400) == "<a int>"
    assert pt._shown(float("nan")) == "<a float>" and pt._shown(float("inf")) == "<a float>"
    assert pt._shown('a"b') == "<a string that is not a plain name>" and pt._shown("x" * 41) == "<a string that is not a plain name>"
    assert pt._shown("x" * 40) == repr("x" * 40) and pt._shown("") == "<a string that is not a plain name>"
