# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/expert_skew_stream.py: the layer-by-layer skew run gives the counts of a whole-model run (issue 739 step 2).

The reference here is a whole-model forward pass with a hook on every router, so the two paths share no loop. The checkpoint is a tiny random
Qwen3-Next model (four layers, three of them linear attention, one full attention, eight experts, top two) saved the way a real one is, with one
tensor per expert."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("expert_skew_stream", SCRIPTS / "expert_skew_stream.py")
ess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ess)


def tiny_config():
    from transformers import Qwen3NextConfig
    return Qwen3NextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        full_attention_interval=4, max_position_embeddings=256, rope_theta=10000.0, partial_rotary_factor=0.25, norm_topk_prob=True,
        decoder_sparse_step=1, mlp_only_layers=[], tie_word_embeddings=False)


def write_checkpoint(path: Path, per_expert: bool = True):
    """Random weights, saved as safetensors. per_expert=True rewrites the fused expert tensors as the one-tensor-per-expert layout of real checkpoints."""
    from safetensors.torch import save_file
    from transformers import AutoConfig, Qwen3NextForCausalLM
    # The model is built from the config as it reads back from disk: that is the config the script under test sees, and a config that went
    # through save and load can differ from the config in memory (the rotary settings did).
    path.mkdir()
    tiny_config().save_pretrained(path)
    torch.manual_seed(0)
    model = Qwen3NextForCausalLM(AutoConfig.from_pretrained(path)).eval()
    for p in model.parameters():                       # weights in bfloat16; buffers such as the rotary frequencies stay float32, as in a loaded model
        p.data = p.data.to(torch.bfloat16)
    tensors = {}
    for k, v in model.state_dict().items():
        if per_expert and k.endswith("mlp.experts.gate_up_proj"):
            base = k[:-len("gate_up_proj")]
            half = v.shape[1] // 2
            for e in range(v.shape[0]):
                tensors[f"{base}{e}.gate_proj.weight"] = v[e, :half].contiguous()
                tensors[f"{base}{e}.up_proj.weight"] = v[e, half:].contiguous()
        elif per_expert and k.endswith("mlp.experts.down_proj"):
            base = k[:-len("down_proj")]
            for e in range(v.shape[0]):
                tensors[f"{base}{e}.down_proj.weight"] = v[e].contiguous()
        else:
            tensors[k] = v.contiguous()
    save_file(tensors, str(path / "model.safetensors"))
    return model


def whole_model_counts(model, chunks):
    """Reference: the full model forward, a hook on every router, the same chunks."""
    cfg = model.config
    counts = [[0] * cfg.num_experts for _ in range(cfg.num_hidden_layers)]
    hooks = []
    for li, layer in enumerate(model.model.layers):
        def hook(m, i, out, li=li):
            for e in out[2].reshape(-1).tolist():
                counts[li][e] += 1
        hooks.append(layer.mlp.gate.register_forward_hook(hook))
    with torch.no_grad():
        for c in chunks:
            model(torch.tensor([c]))
    for h in hooks:
        h.remove()
    return counts


CHUNKS = [[(7 * i + 3 * j) % 128 for j in range(24)] for i in range(3)]


@pytest.mark.parametrize("per_expert", [True, False])
def test_the_layer_by_layer_counts_equal_a_whole_model_run(tmp_path, per_expert):
    model = write_checkpoint(tmp_path / "ckpt", per_expert=per_expert)
    want = whole_model_counts(model, CHUNKS)
    got = ess.stream_counts(tmp_path / "ckpt", {"t": CHUNKS})["t"]
    assert got == want
    assert sum(sum(layer) for layer in got) == len(CHUNKS) * 24 * 2 * 4          # chunks x tokens x top_k x layers


def test_two_sources_share_one_pass_and_keep_separate_counts(tmp_path):
    model = write_checkpoint(tmp_path / "ckpt")
    other = [[(11 * i + j) % 128 for j in range(24)] for i in range(2)]
    got = ess.stream_counts(tmp_path / "ckpt", {"a": CHUNKS, "b": other})
    assert got["a"] == whole_model_counts(model, CHUNKS) and got["b"] == whole_model_counts(model, other)
    assert got["a"] != got["b"]


def test_a_checkpoint_that_does_not_match_the_module_is_refused(tmp_path):
    from safetensors.torch import load_file, save_file
    write_checkpoint(tmp_path / "ckpt")
    f = tmp_path / "ckpt" / "model.safetensors"
    tensors = load_file(str(f))
    del tensors["model.layers.1.mlp.gate.weight"]
    save_file(tensors, str(f))
    with pytest.raises(ValueError, match="layer 1"):
        ess.stream_counts(tmp_path / "ckpt", {"t": CHUNKS})


def test_the_result_has_the_keys_tier_plan_and_expert_skew_use(tmp_path):
    from transformers import AutoConfig
    write_checkpoint(tmp_path / "ckpt")
    counts = ess.stream_counts(tmp_path / "ckpt", {"t": CHUNKS})["t"]
    r = ess.result(tmp_path / "ckpt", "tiny-moe", "t", CHUNKS, counts, AutoConfig.from_pretrained(tmp_path / "ckpt"), "test")
    assert {"model", "tokens", "experts", "top_k", "moe_layers", "mean", "layers", "counts", "note"} <= set(r)
    assert r["model"] == "tiny-moe" and r["snapshot"] == "ckpt"
    assert r["tokens"] == 72 and r["experts"] == 8 and r["top_k"] == 2 and r["moe_layers"] == 4
    assert set(r["mean"]) == {"never_picked", "top_10pct_share", "top_25pct_share", "top_50pct_share", "entropy_vs_max"}
    json.dumps(r)                                                                  # serialisable as it is


def test_text_too_short_for_one_chunk_is_refused():
    with pytest.raises(ValueError, match="only 5 tokens"):
        ess.to_chunks([1, 2, 3, 4, 5], 8)


def test_a_short_tail_is_dropped_like_the_whole_model_script():
    assert ess.to_chunks(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]
