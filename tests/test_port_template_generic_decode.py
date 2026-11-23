# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The cached decode of the generic generated port (issue 759 for the dense, grouped-key, sliding-window case).

The generator wrote `decode_step` as "append the token and run the whole sequence again" for every port whose layers are the standard pre-norm attention and MLP
(Aion: 21 sliding and 7 full layers). The first cached decode (PR 762) covers the composed Qwen4 layers only. Here every attention layer keeps its keys and
values (after the per-head norm and the rotary), a sliding layer keeps only the last `window` rows, and `decode_step` runs one row through the layers.

Every comparison runs on the torch-backed fake ttnn (tests/fake_ttnn_compute.py): the cached path against (a) the same port run over the whole sequence, which
is the reference path and is kept, and (b) a plain torch forward. Nothing here shows that the ops run on a board.
"""
from __future__ import annotations

import copy

import pytest
import torch

import fake_ttnn_compute as fake
from orchard import port_template as pt
from test_port_template import STD_CONFIG, import_port, write_checkpoint
from test_port_template_layer_rope import aion_rope_config, generate_with_weights, torch_forward

_NAMES = iter(range(10 ** 6))


def grouped(cfg):
    """Four query heads sharing two key heads: a grouped-key layout in which the head order matters (the small config has one key head)."""
    cfg = copy.deepcopy(cfg)
    cfg.update({"num_attention_heads": 4, "num_key_value_heads": 2})
    return cfg


def aion_like(window=8):
    """Aion's structure on the small decoder: layer types, a window of `window` positions, rope keyed by layer type with YaRN on the full layers."""
    return grouped(aion_rope_config(hd=16, original=16, window=window))


def plain_qwen():
    """The plain Qwen-like decoder: one rope table, no window, two query heads sharing one key head."""
    return grouped(STD_CONFIG)


def build(monkeypatch, tmp_path, cfg, mutate=None, n_devices=1):
    """(module, cached model, reference model, weights): the same port loaded twice. The reference model is never given a cache, so it runs the whole sequence."""
    text, p, weights = generate_with_weights(cfg)
    assert p.stubs == [] and p.attention is not None
    for old, new in ([mutate] if mutate and isinstance(mutate[0], str) else mutate or []):          # one (old, new) pair or a list of pairs
        assert old in text, f"the generated text no longer contains {old!r}: update the mutation"
        text = text.replace(old, new, 1)
    mod = import_port(monkeypatch, text, tmp_path, name=f"generic_decode_{next(_NAMES)}")
    snap = str(write_checkpoint(tmp_path, weights))
    return mod, mod.load(snap, fake.MeshDevice(n_devices)), mod.load(snap, fake.MeshDevice(n_devices)), weights, p, text


def ids_of(n, mul=7, add=3):
    return [(mul * i + add) % 49 + 1 for i in range(n)]


def rows_kept(model):
    """{layer index: rows of kept keys} of a model that has run a prompt."""
    return {int(name.rsplit(".", 2)[-2]): s["k"].shape[2] for name, s in model["cache"]["state"].items()}


def run_steps(mod, model, ref, ids, n0, *, tol=0.05, window=None, types=None):
    """Prefill ids[:n0], then decode the rest one token at a time; every step's logits against the whole-sequence run and its argmax. With `window` and `types`
    (the layer types) the rows each layer holds after the prompt are checked too: a sliding layer holds the last `window`, a full layer all n0."""
    got = mod.prefill(model, ids[:n0])
    if window:
        assert rows_kept(model) == {i: (min(n0, window) if t == "sliding_attention" else n0) for i, t in enumerate(types)}, rows_kept(model)
    want = mod._forward(ref, ids[:n0])
    assert torch.allclose(got, want, atol=1e-3)
    for pos in range(n0, len(ids)):
        got = mod.decode_step(model, ids[pos], pos)
        want = mod._forward(ref, ids[: pos + 1])
        scale = want.abs().max().item()
        assert (got - want).abs().max().item() <= tol * scale, (pos, (got - want).abs().max().item(), scale)
        assert int(got.argmax()) == int(want.argmax()), pos
        assert model["cache"]["n"] == pos + 1
        if window:                     # the masks are shared by the layers and only those of the latest position are held: the sliding layers have one length, the full layers another
            assert model["step_masks"][0] == pos and len(model["step_masks"][1]) <= 2, (pos, sorted(model["step_masks"][1]))


# ---- the plan says when a port keeps a cache --------------------------------------------------------------------------------------------------

def test_a_standard_port_with_generated_attention_keeps_a_cache_and_writes_the_step_function():
    text, p, _ = generate_with_weights(aion_like())
    assert p.cached_decode and "def _attention_step(" in text and "def _rope_row(" in text and "def _forward_step(" in text
    flat, pf, _ = generate_with_weights(plain_qwen())
    assert pf.cached_decode and "def _attention_step(" in flat


def test_a_port_with_a_stub_keeps_the_whole_sequence_decode():
    from test_port_template import std_construction, std_generate, std_weights
    weights = std_weights()
    text, p = std_generate(weights=weights, construction=std_construction(attention_children={"q_proj": "Linear", "k_proj": "Linear", "v_proj": "Linear",
                                                                                              "o_proj": "Linear", "q_norm": "Qwen3RMSNorm", "k_norm": "Qwen3RMSNorm",
                                                                                              "odd": "WeirdThing"}))
    assert not p.cached_decode
    assert "def _attention_step(" not in text and "run the whole sequence again" in text


def test_the_step_functions_are_never_a_repair_target():
    from orchard import thin_repair
    text, _, _ = generate_with_weights(aion_like())
    for name in ("_attention_step", "_rope_row"):
        assert thin_repair._holds_generated_mark(text, name), name


# ---- the cached decode gives what the whole-sequence run gives ------------------------------------------------------------------------------

def test_the_aion_like_port_decodes_sixteen_steps_past_the_window_like_the_full_rerun(monkeypatch, tmp_path):
    """A prompt of 20 tokens and 20 steps against a window of 8 and a YaRN original length of 16: the sliding layers must drop old rows and the full layers must not,
    and each layer type rotates with its own table."""
    mod, model, ref, weights, _p, _t = build(monkeypatch, tmp_path, aion_like(8))
    ids = ids_of(40)
    cfg = aion_like(8)
    run_steps(mod, model, ref, ids, 20, window=8, types=cfg["layer_types"])
    got = mod.decode_step(model, 5, 40)
    want = torch_forward(weights, cfg, ids + [5])
    assert torch.allclose(got, want, atol=2e-3, rtol=2e-3), (got - want).abs().max()


def test_a_plain_qwen_like_port_decodes_sixteen_steps_like_the_full_rerun(monkeypatch, tmp_path):
    mod, model, ref, weights, _p, _t = build(monkeypatch, tmp_path, plain_qwen())
    ids = ids_of(30)
    run_steps(mod, model, ref, ids, 12)


def test_a_sliding_layer_keeps_only_the_window_and_a_full_layer_keeps_every_row(monkeypatch, tmp_path):
    mod, model, ref, _w, _p, _t = build(monkeypatch, tmp_path, aion_like(8))
    ids = ids_of(30)
    mod.prefill(model, ids[:20])
    for pos in range(20, 30):
        mod.decode_step(model, ids[pos], pos)
    state = model["cache"]["state"]
    rows = {name.rsplit(".", 2)[-2]: s["k"].shape[2] for name, s in state.items()}
    assert rows == {"0": 8, "1": 8, "2": 30}, rows


def test_the_steps_never_run_the_whole_sequence(monkeypatch, tmp_path):
    mod, model, ref, _w, _p, _t = build(monkeypatch, tmp_path, aion_like(8))
    ids = ids_of(26)
    mod.prefill(model, ids[:20])
    monkeypatch.setattr(mod, "_forward", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a step ran the whole sequence")))
    for pos in range(20, 26):
        mod.decode_step(model, ids[pos], pos)


def test_a_step_at_another_position_runs_the_prompt_again_and_the_next_step_is_cached(monkeypatch, tmp_path):
    mod, model, ref, _w, _p, _t = build(monkeypatch, tmp_path, aion_like(8))
    ids = ids_of(30)
    mod.prefill(model, ids[:14])
    mod.decode_step(model, ids[14], 14)
    mod.decode_step(model, ids[15], 15)
    back = mod.decode_step(model, ids[10], 10)                 # a rewind: the history is cut back to 10 tokens and the new token is the eleventh
    assert torch.allclose(back, mod._forward(ref, ids[:10] + [ids[10]]), atol=1e-3)
    assert model["cache"]["n"] == 11
    nxt = mod.decode_step(model, ids[11], 11)
    want = mod._forward(ref, ids[:12])
    assert (nxt - want).abs().max().item() <= 0.05 * want.abs().max().item()


def test_a_new_prompt_after_a_decode_starts_the_cache_afresh(monkeypatch, tmp_path):
    mod, model, ref, _w, _p, _t = build(monkeypatch, tmp_path, aion_like(8))
    ids = ids_of(24)
    mod.prefill(model, ids[:12])
    mod.decode_step(model, ids[12], 12)
    run_steps(mod, model, ref, ids_of(24, mul=5, add=1), 10)


@pytest.mark.parametrize("n_devices", [2])
def test_the_cached_decode_on_a_mesh_gives_the_same_logits(monkeypatch, tmp_path, n_devices):
    mod, model, ref, _w, _p, _t = build(monkeypatch, tmp_path, aion_like(8), n_devices=n_devices)
    run_steps(mod, model, ref, ids_of(28), 18)


def test_close_frees_the_cached_tensors(monkeypatch, tmp_path):
    mod, model, ref, _w, _p, _t = build(monkeypatch, tmp_path, aion_like(8))
    ids = ids_of(24)
    mod.prefill(model, ids[:16])
    mod.decode_step(model, ids[16], 16)
    held = [t for s in model["cache"]["state"].values() for t in s.values()]
    assert held
    freed = []
    monkeypatch.setattr(mod.ttnn, "deallocate", lambda t: freed.append(id(t)), raising=False)
    mod.close(model)
    assert all(id(t) in freed for t in held)


# ---- each guard turns the comparison red --------------------------------------------------------------------------------------------------------

MUTATIONS = {
    "the window keeps one row too many": ('T > ATTENTION["window"]:', 'T > ATTENTION["window"] + 1:'),
    "the window cuts the values one row too short": ("values = ttnn.slice(full_values, (0, 0, full_values.shape[2] - T, 0)", "values = ttnn.slice(full_values, (0, 0, full_values.shape[2] - T + 1, 0)"),
    "a step rotates at position zero": ("_rope_row(model, layer_type, n)", "_rope_row(model, layer_type, 0)"),
    "a step rotates at the position of the last row": ("_rope_row(model, layer_type, n)", "_rope_row(model, layer_type, n - 1)"),
    "the new key is put before the cached keys": ('keys = ttnn.concat([state["k"], k], dim=2)', 'keys = ttnn.concat([k, state["k"]], dim=2)'),
    "the new value is put before the cached values": ('values = ttnn.concat([state["v"], v], dim=2)', 'values = ttnn.concat([v, state["v"]], dim=2)'),
    "the new key and value are not kept": ('state["k"], state["v"] = keys, values', "pass"),
    "the prompt keeps the padding rows": ('state["k"] = ttnn.slice(k, (0, 0, first, 0), (1, ATTENTION["kv_heads"], n, ATTENTION["head_dim"]))',
                                          'state["k"] = ttnn.slice(k, (0, 0, first, 0), (1, ATTENTION["kv_heads"], n + 1, ATTENTION["head_dim"]))'),
    "the prompt keeps the keys from before the rotary": [("    k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)\n",
                                                          "    k_raw = k\n    k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)\n"),
                                                         ('state["k"] = ttnn.slice(k, (0, 0, first, 0)', 'state["k"] = ttnn.slice(k_raw, (0, 0, first, 0)')],
    "a sliding layer keeps the first rows of the prompt, not the last": ('first = n - min(n, ATTENTION["window"])', "first = 0"),
    "the padding columns are not masked": ("row = torch.where(torch.arange(total) < T, 0.0, -30000.0)", "row = torch.zeros(total)"),
    "the masks of earlier positions are never freed": ("if held is None or held[0] != n:", 'if held is None or held[0] == -1:'),
    "the key heads are taken as many as the query heads": ('ttnn.reshape(_linear(model, prefix + ".k_proj", x), (1, 1, kv, hd))', 'ttnn.reshape(_linear(model, prefix + ".k_proj", x), (1, 1, heads, hd))'),
    "the per-head norm of the query is skipped on a step": ('q = ttnn.reshape(ttnn.rms_norm(ttnn.reshape(q, (1, 1, heads, hd)), epsilon=ATTENTION["norm_eps"], weight=model["dev"][prefix + ".q_norm.weight"]), (1, heads, 1, hd))',
                                                            "pass"),
    "the per-head norm of the key is skipped on a step": ('k = ttnn.reshape(ttnn.rms_norm(ttnn.reshape(k, (1, 1, kv, hd)), epsilon=ATTENTION["norm_eps"], weight=model["dev"][prefix + ".k_norm.weight"]), (1, kv, 1, hd))',
                                                          "pass"),
    "the position is not advanced after a step": ('cache["n"] = position + 1', 'cache["n"] = position'),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_mutation_of_the_cache_turns_the_decode_comparison_red(monkeypatch, tmp_path, name):
    cfg = aion_like(8)
    mod, model, ref, _w, _p, _t = build(monkeypatch, tmp_path, cfg, mutate=MUTATIONS[name])
    ids = ids_of(40)
    with pytest.raises((AssertionError, RuntimeError, KeyError, TypeError, ValueError, IndexError)):
        run_steps(mod, model, ref, ids, 20, window=8, types=cfg["layer_types"])
