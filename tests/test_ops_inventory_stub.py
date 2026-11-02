# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ops-inventory template's `--stub-data-dependent` option. The Qwen4-style sparse attention indexer picks tokens with a
Python loop over query positions and `torch.nonzero`, which torch.export cannot trace. The stub swaps in a static version
with the same ops so the layer's ops appear in the inventory. Run end to end on a tiny config on the meta device."""
from __future__ import annotations

import pytest

import subprocess

from test_ops_inventory_template import PY, run_template

def _has_qwen4() -> bool:
    return PY is not None and subprocess.run(
        [PY, "-c", "from transformers.models.qwen4_exp import modeling_qwen4_exp"], capture_output=True).returncode == 0


needs_stack = pytest.mark.skipif(not _has_qwen4(), reason=(
    "SKIPPED: no interpreter with torch and a transformers that has qwen4_exp (5.19.0 or later; set "
    "ORCHARD_REFERENCE_PYTHON), so the indexer stub was not run end to end"))

_TEXT = {
    "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 128, "layer_types": ["linear_attention"] * 3 + ["full_attention"], "num_experts": 8,
        "num_experts_per_tok": 2, "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.25, "max_position_embeddings": 256,
        "tie_word_embeddings": False, "indexer_budget": 16, "indexer_compress_ratio": 4, "indexer_head_dim": 16,
        "indexer_kv_heads": 1, "indexer_n_heads": 2, "hc_count": 4, "hc_lowrank": 8, "ngram_size": 3,
        "ngram_vocab_size_base": 1000, "heads_per_ngram": 2, "split_ngram_parts": 4, "ple_layer_ids": [2],
        "ple_embed_dim": 64, "ple_conv_kernel_size": 4, "output_gate_type": "sigmoid", "eos_token_id": 1}
# the text tower alone (a flat config, as Qwen4ExpForCausalLM reads it), the form the study's traces used
TINY_QWEN4 = {"model_type": "qwen4_exp_text", "architectures": ["Qwen4ExpForCausalLM"], **_TEXT}


def op_names(raw):
    return {o["op"] if "op" in o else o.get("name") for o in raw["ops"]}


@needs_stack
def test_without_the_stub_the_trace_stops_at_the_indexer(tmp_path):
    done, raw = run_template(tmp_path, TINY_QWEN4, "--layers", "4", "--seq", "16")
    assert done.returncode == 3 and raw["failures"][0]["stage"] == "export", (done.stderr, raw)


@needs_stack
def test_with_the_stub_the_trace_passes_the_indexer_and_says_what_it_replaced(tmp_path):
    done, raw = run_template(tmp_path, TINY_QWEN4, "--layers", "4", "--seq", "16", "--stub-data-dependent")
    assert done.returncode == 0 and raw["failures"] == [], (done.stderr, raw["failures"])
    assert raw["stubbed"] == [{"module": "Qwen4ExpTextQSAIndexer", "count": 1,
                               "note": raw["stubbed"][0]["note"]}]
    assert "op mix" in raw["stubbed"][0]["note"]


@needs_stack
def test_the_stubbed_indexer_keeps_the_ops_the_real_one_uses(tmp_path):
    """The real indexer's tensor ops: a linear, two norms, rotary, a mean over key blocks, a matmul, relu, a sum over heads,
    a topk and a scatter-style mask build."""
    done, raw = run_template(tmp_path, TINY_QWEN4, "--layers", "4", "--seq", "16", "--stub-data-dependent")
    names = " ".join(str(o) for o in raw["ops"])
    for needed in ("topk", "relu", "mean", "cat"):
        assert needed in names, needed


@needs_stack
def test_the_stub_is_opt_in(tmp_path):
    done, raw = run_template(tmp_path, TINY_QWEN4, "--layers", "2", "--seq", "16")
    assert "stubbed" not in raw


PARITY = r'''
import importlib.util, json, sys, torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as m
spec = importlib.util.spec_from_file_location("tpl", sys.argv[1])
tpl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tpl)
cfg = Qwen4ExpTextConfig(**json.loads(sys.argv[2]))
torch.manual_seed(1)
idx = m.Qwen4ExpTextQSAIndexer(cfg, 0).eval()
with torch.no_grad():
    for p in idx.parameters():
        p.copy_(torch.randn_like(p) * 0.5)
B, L = 1, 32
h = torch.randn(B, L, cfg.hidden_size)
rot = m.Qwen4ExpTextRotaryEmbedding(cfg)
pos = torch.arange(L).unsqueeze(0)
cos, sin = rot(h, pos)
causal = torch.tril(torch.ones(L, L, dtype=torch.bool)).view(1, 1, L, L)
with torch.no_grad():
    try:
        real = idx(h, (cos, sin), causal, None)
    except RuntimeError as exc:      # some torch builds refuse the real indexer's int32 scatter index
        print(json.dumps({"real_unrunnable": str(exc)[:200]})); sys.exit(0)
    import types
    idx.forward = types.MethodType(tpl.qsa_indexer_stub, idx)
    stub = idx(h, (cos, sin), causal, None)
print(json.dumps({"shape_equal": list(real.shape) == list(stub.shape),
                  "last_row_equal": bool(torch.equal(real[0, 0, -1], stub[0, 0, -1])),
                  "last_row_selected": int(real[0, 0, -1].sum())}))
'''


@needs_stack
def test_the_stub_picks_the_same_blocks_as_the_real_indexer_where_every_token_is_visible():
    """The stub ignores visibility, so the two agree on the last query, which sees every token. A stub that selected the
    wrong blocks (a wrong pool, scale or topk) would differ here."""
    import json

    from test_ops_inventory_template import TEMPLATE

    done = subprocess.run([PY, "-c", PARITY, str(TEMPLATE), json.dumps(_TEXT)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-600:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    if "real_unrunnable" in got:
        pytest.skip(f"SKIPPED: this torch cannot run the real indexer ({got['real_unrunnable']}), so there is nothing to compare with")
    assert got["shape_equal"] and got["last_row_equal"], got
    assert 0 < got["last_row_selected"] < 32, got          # a real selection, not everything and not nothing
