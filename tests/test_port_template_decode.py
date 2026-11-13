# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The cached decode of the generated Qwen4 port (issue 759). The port used to run the whole sequence again for every new token; with a cache each block keeps what
the next token needs (the recurrent state and the convolution tail of the gated delta net, the keys, values and pooled index keys of the sparse attention, the
convolution tail of the PLE block) and `decode_step` runs one row. Every comparison here is against the same port run over the whole sequence, on the torch-backed
fake ttnn (tests/fake_ttnn_compute.py): the cached path must give what the full re-run gives. The fake's recurrent op is a torch stand-in; nothing here shows that
the device ops run on a board."""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

import fake_ttnn_compute as fake
from test_port_template import _load, import_port, tiny_qwen4

HIDDEN = 64


_NAMES = iter(range(10 ** 6))


def tiny_model(monkeypatch, tmp_path, seed=1, mutate=None):
    """The generated tiny Qwen4 port, loaded with random weights through the fake: (module, model, snapshot directory). `mutate` is (old, new): the text is changed before import."""
    text, _plan = tiny_qwen4()
    if mutate:
        assert mutate[0] in text, f"the generated text no longer contains {mutate[0]!r}: update the mutation"
        text = text.replace(*mutate, 1)
    mod = import_port(monkeypatch, text, tmp_path, name=f"decode_port_{next(_NAMES)}")
    g = torch.Generator().manual_seed(seed)
    weights = {}
    for t in _load("tiny-qwen4-tensors.json"):
        if t["dtype"] == "BF16":
            weights[t["name"]] = (torch.randn(*t["shape"], generator=g) * 0.1).to(torch.bfloat16)
        else:
            weights[t["name"]] = torch.arange(int(torch.tensor(t["shape"]).prod()), dtype=torch.int64).reshape(t["shape"])
        if t["name"].endswith("ple_embedding.ngram_heads_vocab_sizes"):
            weights[t["name"]] = torch.tensor([1009, 1013, 1019, 1021])
        elif t["name"].endswith("ple_embedding.ngram_heads_offsets"):
            weights[t["name"]] = torch.tensor([0, 1009, 2022, 3041])
        elif t["name"].endswith("ple_embedding.layer_multipliers"):
            weights[t["name"]] = torch.tensor([3, 5, 7])
    snap = tmp_path / f"snap{seed}"
    snap.mkdir()
    save_file(weights, str(snap / "model.safetensors"))
    return mod, mod.load(str(snap), fake.MeshDevice(1)), str(snap)


def host_rows(t, rows):
    return t.t.float().reshape(-1, t.t.shape[-1])[:rows]


def test_the_gated_delta_net_step_gives_the_row_the_full_run_gives(monkeypatch, tmp_path):
    mod, model, _snap = tiny_model(monkeypatch, tmp_path)
    g = torch.Generator().manual_seed(7)
    n = 37                                                     # the prompt: not a multiple of 32, so the padding rows matter
    x_all = (torch.randn(1, 1, 96, HIDDEN, generator=g) * 0.5).to(torch.bfloat16)
    up = lambda rows: fake.Tensor(x_all[:, :, rows].float(), replicated=True, layout="TILE")
    model["cache"] = None
    full = mod._A_linear_attn(model, 0, up(slice(0, 64)), list(range(64)))
    state_model = {"mode": "prefill", "n": n, "state": {}}
    model["cache"] = state_model
    mod._A_linear_attn(model, 0, up(slice(0, 64)), list(range(64)))
    for pos in range(n, n + 6):
        state_model["mode"], state_model["n"] = "step", pos
        got = mod._A_linear_attn(model, 0, up(slice(pos, pos + 1)), list(range(pos + 1)))
        model["cache"] = None
        want = mod._A_linear_attn(model, 0, up(slice(0, 64)), list(range(64))) if pos < 64 else None
        model["cache"] = state_model
        assert torch.allclose(got.t.float().reshape(-1, HIDDEN)[0], want.t.float().reshape(-1, HIDDEN)[pos], atol=5e-2, rtol=5e-2), pos


def test_the_cached_decode_gives_the_logits_the_full_rerun_gives(monkeypatch, tmp_path):
    mod, model, snap = tiny_model(monkeypatch, tmp_path)
    ref = mod.load(snap, fake.MeshDevice(1))                    # a second copy of the model: it never has a cache, so it runs the whole sequence
    ids = [(7 * i + 3) % 90 + 1 for i in range(26)]
    n0 = 11
    got = mod.prefill(model, ids[:n0])
    want = mod._forward(ref, ids[:n0])
    assert torch.allclose(got, want, atol=1e-3)
    for pos in range(n0, 26):
        got = mod.decode_step(model, ids[pos], pos)
        want = mod._forward(ref, ids[: pos + 1])
        scale = want.abs().max().item()
        assert (got - want).abs().max().item() <= 0.05 * scale, (pos, (got - want).abs().max().item(), scale)
        assert int(got.argmax()) == int(want.argmax()), pos
        assert model["cache"]["n"] == pos + 1 and fake.COUNTS["fused_recurrent_gated_delta_rule"] >= 1, "every step ran from the kept state"


def test_a_step_at_another_position_runs_the_prompt_again_and_the_next_step_is_cached(monkeypatch, tmp_path):
    mod, model, snap = tiny_model(monkeypatch, tmp_path)
    ref = mod.load(snap, fake.MeshDevice(1))
    ids = [(5 * i + 2) % 90 + 1 for i in range(20)]
    mod.prefill(model, ids[:10])
    mod.decode_step(model, ids[10], 10)
    mod.decode_step(model, ids[11], 11)
    back = mod.decode_step(model, ids[8], 8)                    # a rewind: the history is cut back to 8 tokens and the new token is the ninth
    assert torch.allclose(back, mod._forward(ref, ids[:8] + [ids[8]]), atol=1e-3)
    assert model["cache"]["n"] == 9
    nxt = mod.decode_step(model, ids[9], 9)
    want = mod._forward(ref, ids[:8] + [ids[8], ids[9]])
    assert (nxt - want).abs().max().item() <= 0.05 * want.abs().max().item()


MUTATIONS = {
    "padding rows change the recurrent state": ("g, beta = ttnn.multiply(ttnn.reshape(g, (1, seq, hv)), keep), ttnn.multiply(ttnn.reshape(beta, (1, seq, hv)), keep)",
                                                 "g, beta = ttnn.reshape(g, (1, seq, hv)), ttnn.reshape(beta, (1, seq, hv))"),
    "the recurrent state is not kept": ('state["rec"] = final', "pass"),
    "the convolution tail is cut from the wrong rows after the prompt": ('state["tail"] = ttnn.slice(xpad, (0, 0, n, 0), (1, 1, n + kernel - 1, channels))',
                                                                       'state["tail"] = ttnn.slice(xpad, (0, 0, n - 1, 0), (1, 1, n + kernel - 2, channels))'),
    "the convolution tail is not advanced by a step": ('state["tail"] = ttnn.slice(xpad, (0, 0, 1, 0), (1, 1, kernel, channels))',
                                                         'state["tail"] = ttnn.slice(xpad, (0, 0, 0, 0), (1, 1, kernel - 1, channels))'),
    "the PLE convolution tail is not advanced by a step": ('state["tail"] = ttnn.slice(xpad, (0, 0, 1, 0), (1, 1, 1 + pad, channels))',
                                                            'state["tail"] = ttnn.slice(xpad, (0, 0, 0, 0), (1, 1, pad, channels))'),
    "the PLE takes the first row of the embeddings, not the last": ("[:, -seq:]          # decoding one token", "[:, :seq]          # decoding one token"),
    "the new key and value are not added to the cache": ('state["k"], state["v"] = ttnn.concat([state["k"], k], dim=2), ttnn.concat([state["v"], v], dim=2)',
                                                          'state["k"], state["v"] = state["k"], state["v"]'),
    "a step rotates the query at position 0": ("torch.arange(n, n + 1) if mode == \"step\" else torch.arange(seq), hd)", "torch.arange(1) if mode == \"step\" else torch.arange(seq), hd)"),
    "the tokens after the last complete block are hidden": ("visible[nb * ratio: pos + 1] = 1.0", "visible[nb * ratio: pos] = 1.0"),
    "a finished block is not cleared from the partial keys": ('state["partial"] = state["partial"][:0]', "pass"),
    "a step picks the lowest scoring blocks": ("values, indices = ttnn.topk(scores, k, dim=-1, largest=True, sorted=True, stable=True)\n        for b in",
                                                "values, indices = ttnn.topk(scores, k, dim=-1, largest=False, sorted=True, stable=True)\n        for b in"),
    "the position is not advanced after a step": ('cache["n"] = position + 1', 'cache["n"] = position'),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_mutation_of_the_cache_turns_the_decode_comparison_red(monkeypatch, tmp_path, name):
    mod, model, snap = tiny_model(monkeypatch, tmp_path, mutate=MUTATIONS[name])
    ref = mod.load(snap, fake.MeshDevice(1))
    ids = [(7 * i + 3) % 90 + 1 for i in range(26)]
    mod.prefill(model, ids[:11])
    with pytest.raises((AssertionError, RuntimeError, KeyError, TypeError, ValueError)):
        for pos in range(11, 26):
            got = mod.decode_step(model, ids[pos], pos)
            want = mod._forward(ref, ids[: pos + 1])
            assert (got - want).abs().max().item() <= 0.05 * want.abs().max().item() and int(got.argmax()) == int(want.argmax())
            assert model["cache"]["n"] == pos + 1
