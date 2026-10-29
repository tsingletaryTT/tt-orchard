# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The plain-torch MiniCPM3 reference in scripts/minicpm3/ref_model.py computes what the transformers-native MiniCPM3 class computes.

A tiny random model is built with the native class, its weights are saved as one safetensors file, and both implementations run the same token ids. The reference is the
CPU side of the MiniCPM3-4B port (PR 210); this keeps that claim checked without the 8 GB weights. Skipped when the installed transformers has no `minicpm3` model.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")
if importlib.util.find_spec("transformers.models.minicpm3") is None:
    pytest.skip("this transformers has no minicpm3 model", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "minicpm3"))
import ref_model  # noqa: E402

DIMS = dict(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32,
            kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, scale_emb=4.0, scale_depth=1.4, dim_model_base=16, rms_norm_eps=1e-5,
            max_position_embeddings=256)
FACTORS = [1.1, 1.4, 2.0, 3.5]                       # one factor per rotary frequency: qk_rope_head_dim / 2 of them


def native_model():
    from transformers import AutoModelForCausalLM, MiniCPM3Config
    cfg = MiniCPM3Config(**DIMS, rope_parameters={"rope_type": "longrope", "factor": 1.0, "rope_theta": 10000.0, "short_factor": FACTORS, "long_factor": FACTORS,
                                                  "original_max_position_embeddings": 256})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).float().eval()
    return model


def test_reference_matches_the_native_class_on_a_tiny_random_model(tmp_path):
    from safetensors.torch import save_file
    model = native_model()
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items() if k.startswith("model.") and "rotary" not in k}
    save_file(state, str(tmp_path / "w.safetensors"))
    cfg = dict(DIMS, rope_scaling={"type": "longrope", "short_factor": FACTORS, "long_factor": FACTORS, "original_max_position_embeddings": 256})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    ref = ref_model.RefModel(str(tmp_path), str(tmp_path / "w.safetensors"), dtype=torch.float32, max_len=64)
    ids = torch.tensor([[1, 5, 9, 33, 70, 2, 8, 40, 41, 3, 17, 90]])
    with torch.no_grad():
        want = model(ids).logits[0]
    got = ref.forward(ids)[0]
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-4, rel
    # one token at a time with the cache gives the same last-position logits as the whole sequence
    ref.reset()
    for p in range(ids.shape[1]):
        step = ref.forward(ids[:, p:p + 1], start=p)
    assert ((step[0, -1] - want[-1]).norm() / want[-1].norm()).item() < 1e-4


def test_the_reference_tests_are_not_vacuous_a_wrong_scale_is_caught(tmp_path):
    """The comparison must fail when the reference is wrong: change `scale_emb` in the reference's config only."""
    from safetensors.torch import save_file
    model = native_model()
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items() if k.startswith("model.") and "rotary" not in k}
    save_file(state, str(tmp_path / "w.safetensors"))
    cfg = dict(DIMS, scale_emb=DIMS["scale_emb"] * 1.1,
               rope_scaling={"type": "longrope", "short_factor": FACTORS, "long_factor": FACTORS, "original_max_position_embeddings": 256})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    ref = ref_model.RefModel(str(tmp_path), str(tmp_path / "w.safetensors"), dtype=torch.float32, max_len=64)
    ids = torch.tensor([[1, 5, 9, 33, 70, 2, 8, 40]])
    with torch.no_grad():
        want = model(ids).logits[0]
    assert ((ref.forward(ids)[0] - want).norm() / want.norm()).item() > 1e-3
    assert math.isfinite(want.sum().item())
