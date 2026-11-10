# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The rotary helpers of the port generator (issue 696, slice 4): `_rope_tables(model, n)` and `_apply_rotary(model, x, cos, sin)` are written whenever the
config gives the rotary settings and the plan has an attention block (generated or a stub), so that a coder-written attention stub calls them and does
not invent a rotary table (arm E13: `model['host']['cos_cached']`) or a partial-rotary slice and concat (arm E7).

Layout: `x` is [1, heads, S, head_dim] (the layout the reference applies rotary in); the tables are [1, 1, S, head_dim] bf16 on the device. The rotary part
is the first `rot = int(head_dim * partial_rotary_factor)` columns of each head; the tables hold cos and sin there and 1 and 0 in the other columns, so
the rotation is `x * cos + (x @ R) * sin` with R the constant matrix of rotate-half on the first `rot` columns (zero elsewhere). That is one matmul and two
elementwise ops, with no narrow slice and no concat. The comparison runs the generated helpers through the torch-backed fake ttnn (tests/fake_ttnn_compute.py)
against the transformers `Qwen4ExpTextRotaryEmbedding` and `apply_rotary_pos_emb` at tiny sizes. It shows that the helpers compute the reference's values; it
does not show that the device ops run on a board. The reference builds three position streams (a multimodal layout); for text the three are equal, and the
helpers cover text only."""
from __future__ import annotations

import ast
import copy

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import _load, import_port, tiny_qwen4, std_generate

TABLES_TEST = "tests/test_port_template_rotary.py::test_rope_tables_match_the_reference_rotary_embedding"


def _config(factor=None, mutate=None):
    cfg = copy.deepcopy(_load("tiny-qwen4-config.json"))
    sec = cfg.get("text_config", cfg)
    if factor is not None:
        sec["partial_rotary_factor"] = factor
    if mutate:
        mutate(sec)
    return cfg


def _generate(cfg):
    return pt.generate(config=cfg, construction=_load("tiny-qwen4-construction.json"), tensors=_load("tiny-qwen4-tensors.json"),
                       inventory=_load("tiny-qwen4-ops-inventory.json"),
                       excerpts="class Qwen4ExpTextRMSNorm(nn.Module):\n    def forward(self, x):\n        output = output * (1.0 + self.weight.float())\n")


def _model():
    return {"n_devices": 1, "mesh_device": fake.MeshDevice(1), "ck": None, "dev": {}, "host": {}, "ids": {}}


def _reference(factor, n):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    cfg = Qwen4ExpTextConfig(head_dim=32, hidden_size=64, num_attention_heads=8, num_key_value_heads=2, partial_rotary_factor=factor,
                             max_position_embeddings=512)
    rot = m.Qwen4ExpTextRotaryEmbedding(cfg)
    cos, sin = rot(torch.zeros(1, n, 32), torch.arange(n).unsqueeze(0))
    return m, cos.float(), sin.float()


# ---- the plan and the text ------------------------------------------------------------------------------------------------------------------

def test_the_tiny_qwen4_port_has_both_rotary_helpers_and_the_constants():
    text, p = tiny_qwen4()
    assert p.rotary == {"head_dim": 32, "rot": 8, "theta": 10000.0}
    names = [n.name for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
    assert names.count("_rope_tables") == 1 and names.count("_apply_rotary") == 1
    assert "ROTARY = {" in text and "_rope_tables" in pt.FIXED_HELPERS and "_apply_rotary" in pt.FIXED_HELPERS
    assert pt.stub_lines(text) != [] and pt.sheet_problems(text) == []


def test_a_standard_decoder_keeps_its_one_rope_tables_and_gains_apply_rotary():
    text, p = std_generate()
    names = [n.name for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
    assert names.count("_rope_tables") == 1 and names.count("_apply_rotary") == 1 and p.rotary and p.rotary["rot"] == p.rotary["head_dim"]


@pytest.mark.parametrize("why,mutate", [
    ("no head size", lambda s: (s.pop("head_dim", None), s.pop("num_attention_heads", None))),
    ("a rope type that is not the default", lambda s: s.__setitem__("rope_parameters", {"rope_type": "yarn", "rope_theta": 10000.0})),
    ("an odd rotary width", lambda s: s.__setitem__("partial_rotary_factor", 0.1)),
    ("a factor above one", lambda s: s.__setitem__("partial_rotary_factor", 1.5)),
    ("a factor that is not a number", lambda s: s.__setitem__("partial_rotary_factor", "quarter")),
])
def test_without_usable_rotary_settings_the_helpers_are_not_written_and_the_plan_says_why(why, mutate):
    try:
        text, p = _generate(_config(mutate=mutate))
    except pt.PlanError:
        return                                                  # a config the plan cannot read at all also writes no helper
    assert p.rotary is None, why
    assert "def _apply_rotary" not in text and "def _rope_tables" not in text
    assert any("rotary" in n for n in p.notes), p.notes


# ---- the helpers against the transformers reference -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [0.25, 1.0])
def test_rope_tables_match_the_reference_rotary_embedding(monkeypatch, tmp_path, factor):
    text, p = _generate(_config(factor=factor))
    mod = import_port(monkeypatch, text, tmp_path)
    n = 32
    _m, cos, sin = _reference(factor, n)
    gcos, gsin = (t.t.float().reshape(n, 32) for t in mod._rope_tables(_model(), n))
    rot = p.rotary["rot"]
    assert rot == int(32 * factor)
    assert torch.allclose(gcos[:, :rot], cos[0], atol=1e-2) and torch.allclose(gsin[:, :rot], sin[0], atol=1e-2)
    assert (gcos[:, rot:] == 1).all() and (gsin[:, rot:] == 0).all()
    assert cos[0].abs().max() > 0.9 and sin[0].abs().max() > 0.5            # the reference tables are not trivially 1 and 0


@pytest.mark.parametrize("factor", [0.25, 1.0])
def test_apply_rotary_matches_the_reference_apply_function(monkeypatch, tmp_path, factor):
    text, _p = _generate(_config(factor=factor))
    mod = import_port(monkeypatch, text, tmp_path)
    n, heads = 32, 4
    m, cos, sin = _reference(factor, n)
    torch.manual_seed(0)
    x = torch.randn(1, heads, n, 32).to(torch.bfloat16).float()
    model = _model()
    gcos, gsin = mod._rope_tables(model, n)
    got = mod._apply_rotary(model, fake.Tensor(x, layout="TILE"), gcos, gsin).t.float()
    want = m.apply_rotary_pos_emb(x, None, cos, sin)
    assert got.shape == want.shape == (1, heads, n, 32)
    assert (got - want).abs().max() < 0.05 * want.abs().max()
    if factor < 1:
        assert torch.equal(got[..., 8:], x[..., 8:])               # the columns past the rotary part pass through unchanged
        assert not torch.allclose(got[..., :8], x[..., :8], atol=1e-2)   # and the rotary part does change


# ---- mutations of the emitted text: each must turn a comparison red ----------------------------------------------------------------------

def _run(monkeypatch, tmp_path, old, new, test):
    text, _p = _generate(_config(factor=0.25))
    monkeypatch.setattr(pt, "render", lambda plan: text.replace(old, new))
    test(monkeypatch, tmp_path, 0.25)


TABLE_MUTATIONS = {
    "the frequencies are spread over the whole head": ("dtype=torch.float32) / rot))", "dtype=torch.float32) / hd))"),
    "the passthrough columns of cos are not one": ("torch.ones(n, hd - rot)", "torch.zeros(n, hd - rot)"),
    "the passthrough columns of sin are not zero": ("torch.zeros(n, hd - rot)", "torch.ones(n, hd - rot)"),
}
APPLY_MUTATIONS = {
    "the rotation matrix has the wrong sign": ("rot_m[i + half, i] = -1.0", "rot_m[i + half, i] = 1.0"),
    "the rotation pairs columns half a head apart": ("half = rot // 2", "half = hd // 2"),
    "the sin term is dropped": ("return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))", "return ttnn.multiply(x, cos)"),
    "the cos term is dropped": ("return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))", "return ttnn.multiply(rotated, sin)"),
}


@pytest.mark.parametrize("name", sorted(TABLE_MUTATIONS))
def test_a_mutated_table_turns_the_table_comparison_red(monkeypatch, tmp_path, name):
    old, new = TABLE_MUTATIONS[name]
    assert old in _generate(_config(factor=0.25))[0], "the mutation does not apply: " + old
    with pytest.raises(AssertionError):
        _run(monkeypatch, tmp_path, old, new, test_rope_tables_match_the_reference_rotary_embedding)


@pytest.mark.parametrize("name", sorted(APPLY_MUTATIONS))
def test_a_mutated_apply_turns_the_apply_comparison_red(monkeypatch, tmp_path, name):
    old, new = APPLY_MUTATIONS[name]
    assert old in _generate(_config(factor=0.25))[0], "the mutation does not apply: " + old
    with pytest.raises(AssertionError):
        _run(monkeypatch, tmp_path, old, new, test_apply_rotary_matches_the_reference_apply_function)


# ---- the helpers are written only for a plan that has an attention block --------------------------------------------------------------------

def test_has_attention_counts_a_generated_block_and_an_attention_stub_and_not_other_stubs():
    st = lambda fn, cls: pt.Stub(function=fn, path="p", cls=cls, reason="r")
    assert pt._has_attention({"heads": 1}, []) is True
    assert pt._has_attention(None, [st("_C_self_attn", "SomeBlock")]) is True
    assert pt._has_attention(None, [st("_X_block", "Qwen4ExpTextAttention")]) is True
    assert pt._has_attention(None, [st("_layer_C", "Qwen4ExpTextDecoderLayer"), st("_B_ple", "Qwen4ExpTextPLELayer")]) is False
    assert pt._has_attention(None, []) is False


def test_a_plan_without_an_attention_block_writes_no_rotary_helpers(monkeypatch):
    monkeypatch.setattr(pt, "_has_attention", lambda settings, stubs: False)
    text, p = tiny_qwen4()
    assert p.rotary is None and "def _apply_rotary" not in text and "def _rope_tables" not in text and "ROTARY" not in text


def test_a_partial_rotary_factor_inside_rope_parameters_keeps_the_standard_attention_a_stub():
    """The transformers config moves `partial_rotary_factor` into `rope_parameters`; the attention plan read only the top level, so it would have
    written a whole-head rotary for such a model."""
    from test_port_template import STD_CONFIG
    cfg = copy.deepcopy(STD_CONFIG)
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}
    _text, p = std_generate(config=cfg)
    assert p.attention is None
    assert any("partial_rotary_factor is 0.5" in s.reason for s in p.stubs)
