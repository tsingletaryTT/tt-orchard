# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Recording a text model's run in the layout `orchard/audio_reference.py` reads (issue 214, item 1; issue 206).

`first_divergence`, `noise_floor` and the stage gates read recorded runs in the audio layout. That layout asks for eight settings written in audio words.
This module writes a text run's own settings once and maps them to those eight, so a runtime script does not repeat the mapping:

    text setting          stored as          note
    prompt                prompt             the prompt text (the run is also keyed by its token ids, see `prompt_sha256`)
    prompt_tokens         duration           the number of prompt token ids
    decode_tokens         steps              the number of greedy decode steps recorded
    sampler               sampler            always "greedy" (teacher-forced runs follow the reference's token path)
    dtype                 dtype              the dtype the run computed in
    (none)                cfg_scale          0.0
    (none)                seed               0
    (none)                sample_size        0

The stored names are the audio ones, so existing comparisons keep working. `text_settings(run)` returns the text names back.

Array names (kind `layer` or `other`), the contract every text runtime's recorder follows:

    layer/layer_NN          the residual stream after block NN, every prompt position, shape [prompt_tokens, hidden]
    layer/layer_NN.attn     the stream after the attention step of block NN (optional; used to find the first operation that leaves)
    other/dec0_layer_NN     the stream after block NN for the first decode step
    other/prefill_logits    the logits of the last prompt position
    other/dec_logits_K      the logits of decode step K
    other/final_norm        the stream after the final norm (optional)
    other/tokens            the greedy token path recorded by the float32 reference, one value per decode step

Nothing here imports torch or ttnn. It needs numpy through `audio_reference`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from orchard import audio_reference as ar

TOKENS = "tokens"
PREFILL_LOGITS = "prefill_logits"


def dec_logits(k: int) -> str:
    return f"dec_logits_{k}"


def block_name(i: int) -> str:
    return f"layer_{i:02d}"


def text_settings(prompt: str, prompt_tokens: int, decode_tokens: int, dtype: str, *, device: str | None = None) -> dict:
    """The eight settings of the audio layout for a text run (see the table above), plus `device` when given."""
    for name, value in (("prompt_tokens", prompt_tokens), ("decode_tokens", decode_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ar.ReferenceError(f"{name} must be a whole number of at least 0")
    settings = {"prompt": prompt, "duration": prompt_tokens, "steps": decode_tokens, "cfg_scale": 0.0, "sampler": "greedy", "seed": 0, "dtype": dtype,
                "sample_size": 0}
    if device:
        settings["device"] = device
    return settings


def settings_view(settings: dict) -> dict:
    """The text names of a recorded run's settings."""
    return {"prompt": settings.get("prompt"), "prompt_tokens": settings.get("duration"), "decode_tokens": settings.get("steps"),
            "sampler": settings.get("sampler"), "dtype": settings.get("dtype"), "device": settings.get("device")}


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def make_recorder(out, *, model_id: str, config_path, weights_bytes: int, runtime: str, prompt: str, prompt_tokens: int, decode_tokens: int, dtype: str,
                  device: str | None = None, revision: str = "local-snapshot") -> ar.Recorder:
    """An `audio_reference.Recorder` for a text run. `config_path` is the model's `config.json` (its sha256 identifies the configuration);
    `runtime` names the code that made the run (the library entry of the manifest)."""
    cfg_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    model = {"name": model_id, "revision": revision, "config_sha256": cfg_sha, "weights_bytes": str(int(weights_bytes))}
    library = {"name": runtime, "commit": "n/a"}
    return ar.Recorder(out, model=model, library=library, settings=text_settings(prompt, prompt_tokens, decode_tokens, dtype, device=device))
