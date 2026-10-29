# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the transformers-native MiniCPM3 model from a model directory and one safetensors weights file.

The directory holds the repository's `config.json` (the 4.41-era layout, with `rope_scaling`). The native class in transformers 5.x reads `rope_parameters`, so
the config is converted in memory. No second directory and no copy of the weights is needed.
"""
import json


def native_config(cfg: dict) -> dict:
    """Return a copy of a repository config with `rope_parameters` filled from `rope_scaling` (the native class reads the former)."""
    cfg = dict(cfg)
    if "rope_parameters" not in cfg:
        rs = dict(cfg.get("rope_scaling") or {})
        rs.setdefault("rope_type", rs.pop("type", "longrope"))
        rs.pop("type", None)
        rs.setdefault("factor", 1.0)
        rs.setdefault("rope_theta", cfg.get("rope_theta", 10000.0))
        cfg["rope_parameters"] = rs
    cfg.pop("rope_scaling", None)
    return cfg


def build_native_model(model_dir, weights_file, dtype):
    """The native MiniCPM3 causal LM with the weights from `weights_file`, in `dtype`, in eval mode."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, MiniCPM3Config
    cfg = native_config(json.load(open(f"{model_dir}/config.json")))
    keep = {k: v for k, v in cfg.items() if k not in ("architectures", "auto_map", "model_type", "torch_dtype", "transformers_version")}
    model = AutoModelForCausalLM.from_config(MiniCPM3Config(**keep))
    state = load_file(weights_file)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # lm_head is tied to the embedding table; the rotary buffers are computed, not stored
    missing = [k for k in missing if k != "lm_head.weight" and "rotary" not in k]
    if missing or unexpected:
        raise ValueError(f"weights do not match the native model: missing {missing[:5]}, unexpected {list(unexpected)[:5]}")
    model.tie_weights()
    return model.to(dtype).eval()
