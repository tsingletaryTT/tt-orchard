# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The one place that says how the Stable Audio 3 text encoder (T5Gemma) runs on the host (issue 162, operator decision 2026-11-06).

The library loads the encoder itself (`T5GemmaConditioner`, `T5GemmaEncoderModel.from_pretrained`) and gives it no dtype or attention argument.
Two things then depend on the installed transformers:

  * the checkpoint is stored in bfloat16 and `from_pretrained` keeps the stored dtype, so a run that is labelled float32 had a bfloat16 encoder;
  * transformers 5.19.0 picks `sdpa` attention for this encoder and 5.21.0 picks `eager` (5.21.0 refuses `sdpa` for it).

Two bfloat16 runs that differ only in attention differ by 1.3 to 1.9 percent in the prompt embeddings, which was the 1.7 percent between two CPU
environments in issue 162 (docs/run-logs/2026-11-02-t5gemma-environment-floor.md). With the encoder in float32 and `eager` attention in both
environments the embeddings agree exactly. So every run that builds prompt embeddings, the CPU reference and the chip run's host encoder, pins
the encoder to these two values. `apply` puts them on a loaded model, `observed` reads back what the model really has (the manifest records that, not
the constants), and `cache_entry_name` and `settings_problems` make an old record (bfloat16, or an attention implementation nobody chose) visible:
the reference cache looks under another entry name, and a comparison refuses two runs whose encoder settings differ.

Runs made before the pin carry no encoder fields. They stay as recorded and are read as "unrecorded" (`state`).
"""
from __future__ import annotations

ENCODER_DTYPE = "float32"
ENCODER_ATTN = "eager"
# The manifest settings keys that hold the encoder setting a run really had.
DTYPE_KEY = "encoder_dtype"
ATTN_KEY = "encoder_attn"
ENCODER_KEYS = (DTYPE_KEY, ATTN_KEY)
PINNED = {DTYPE_KEY: ENCODER_DTYPE, ATTN_KEY: ENCODER_ATTN}
# One line that says where a quoted number was measured. Anything that prints a number from a run made with the pin carries it.
LABEL = f"text encoder: {ENCODER_DTYPE}, {ENCODER_ATTN} attention (issue 162)"


class EncoderPinError(Exception):
    """The model has no T5Gemma encoder to pin, or its encoder is not in the pinned state."""


def _encoders(model) -> list:
    """The T5Gemma encoder modules of a library StableAudioModel (`model.model.conditioner.conditioners`). The conditioner keeps its encoder in
    `__dict__["model"]` when gradients are off, so it is not found by walking the module's parameters."""
    conditioners = getattr(getattr(getattr(model, "model", None), "conditioner", None), "conditioners", None)
    if conditioners is None:
        return []
    out = []
    for cond in list(conditioners.values()):
        if type(cond).__name__ == "T5GemmaConditioner" and getattr(cond, "model", None) is not None:
            out.append(cond.model)
    return out


def _set_attention(encoder, name: str) -> None:
    """Switch every config of the encoder to attention implementation `name`. Attention looks the implementation up in the config on each call, so a
    switch after loading takes effect without a reload."""
    setter = getattr(encoder, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter(name)
        except Exception:                      # an older or newer transformers: the config below is the fallback
            pass
    seen = set()
    for module in encoder.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            try:
                cfg._attn_implementation = name
            except Exception:                  # a read-only config: observed() reports what the model has
                pass


def apply(model) -> dict:
    """Put the pinned dtype and attention implementation on every T5Gemma encoder of `model` and return `observed(model)`. Raises EncoderPinError when there
    is no encoder or when one still differs from the pin afterwards (a transformers version that ignored the change must not record the pin)."""
    encoders = _encoders(model)
    if not encoders:
        raise EncoderPinError("the model has no T5Gemma text encoder to pin")
    for encoder in encoders:
        encoder.float()                        # exact: the checkpoint's bfloat16 values are all representable in float32
        _set_attention(encoder, ENCODER_ATTN)
    return require(model)


def _read(encoder) -> tuple:
    params = list(encoder.parameters())
    dtypes = {str(p.dtype).removeprefix("torch.") for p in params}
    dtype = dtypes.pop() if len(dtypes) == 1 else "mixed" if dtypes else None
    return dtype, getattr(getattr(encoder, "config", None), "_attn_implementation", None)


def observed(model) -> dict:
    """{"encoder_dtype", "encoder_attn"} as the loaded model has them. Raises EncoderPinError when there is no encoder or the encoders disagree."""
    encoders = _encoders(model)
    if not encoders:
        raise EncoderPinError("the model has no T5Gemma text encoder to read")
    reads = {_read(e) for e in encoders}
    if len(reads) != 1:
        raise EncoderPinError(f"the model's text encoders differ from each other: {sorted(map(str, reads))}")
    dtype, attn = reads.pop()
    return {DTYPE_KEY: dtype, ATTN_KEY: attn}


def require(model) -> dict:
    """`observed(model)`, refused when it is not the pin."""
    seen = observed(model)
    if seen != PINNED:
        raise EncoderPinError(f"the text encoder is {seen[DTYPE_KEY]} with {seen[ATTN_KEY]} attention; the pin is {ENCODER_DTYPE} with {ENCODER_ATTN} "
                              "(issue 162)")
    return seen


def cache_entry_name(model_name: str, *, uses_encoder: bool = True) -> str:
    """The reference cache entry for library model `model_name` made with the pinned encoder. An entry made before the pin is `<model_name>`
    and is never read or replaced by a Stable Audio run; it stays on disk, unused. A model that has no T5Gemma encoder (`uses_encoder` false, an
    ACE-Step profile) keeps the plain name."""
    return f"{model_name}.enc-{ENCODER_DTYPE}-{ENCODER_ATTN}" if uses_encoder else model_name


def state(settings) -> str:
    """"pinned", "other" (recorded, but not the pin) or "unrecorded" (a run made before the pin) for a run's manifest settings."""
    if not isinstance(settings, dict) or not any(k in settings for k in ENCODER_KEYS):
        return "unrecorded"
    return "pinned" if all(settings.get(k) == v for k, v in PINNED.items()) else "other"


def settings_problems(a, b) -> list[str]:
    """One problem for each encoder setting that differs between two runs' settings (a missing field differs from a recorded one). Two runs that
    both predate the pin have no fields and no problem; `state` says they are unrecorded."""
    a = a if isinstance(a, dict) else {}
    b = b if isinstance(b, dict) else {}
    return [f"settings.{k} differs: {a.get(k)!r} and {b.get(k)!r} (the text encoder ran differently; issue 162)" for k in ENCODER_KEYS if a.get(k) != b.get(k)]
