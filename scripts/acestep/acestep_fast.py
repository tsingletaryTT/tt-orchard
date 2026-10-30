# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Request-level savings for ACE-Step 1.5 turbo on the chip that leave every output unchanged (issue 254, chip-path work of 2026-10-30).

The profile in `docs/run-logs/2026-10-30-acestep-chip-path.md` shows where a 30 s request spends its time outside the chip. Three of the pieces are repeated or
constant work, so they are removed with an exact cache and no change of arithmetic:

  cond_cache    ACE-Step's handler calls `model.prepare_condition` (the condition encoder: text projector, lyric encoder, timbre encoder) once, and `generate_audio`
                calls it again with the same arguments. The second call is answered from the first. One entry, compared tensor by tensor, cleared at the start of every request (`new_request`), so a benchmark that repeats one prompt measures the request and not the cache.
  timbre_cache  For text to music the reference audio is always the first 750 frames of the silence latent, so the timbre encoder (four transformer layers on the CPU)
                returns the same tensor on every request. Cached by the bytes of its input, at most `entries` inputs.
  kv_on_chip    (chip step, see acestep_chip.py) the cross-attention keys and values of a request are made on the chip.
  text_on_chip  (chip step) the Qwen3 text encoder runs on the chip.
  lyric_on_chip (chip step) the lyric encoder runs on the chip.
  gc_freeze     The handler calls `gc.collect()` twice per request. With the models loaded, a collection walks every object in the process (about 0.4 s each).
                `gc.freeze()` after loading moves the loaded objects out of the collector's view; collection still runs and still frees what a request makes.

The steps are named so a measurement can switch each one on and off: `parse_steps("cond_cache,gc_freeze")`.
"""
from __future__ import annotations

import gc
import hashlib

import torch

CHIP_STEPS = ("kv_on_chip", "text_on_chip", "lyric_on_chip")           # pieces moved onto the chip by acestep_chip.py; they change arithmetic (bfloat16), so each has its own bars
STEPS = ("cond_cache", "timbre_cache", "gc_freeze") + CHIP_STEPS


def parse_steps(text: str) -> tuple:
    """"all", "none" or a comma-separated list of step names; an unknown name is an error (a misspelt step would silently measure nothing)."""
    text = (text or "none").strip()
    if text == "all":
        return STEPS
    if text == "none":
        return ()
    names = tuple(n.strip() for n in text.split(",") if n.strip())
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise ValueError(f"unknown fast step(s): {', '.join(unknown)}; the steps are {', '.join(STEPS)}")
    return names


def _same(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and a.device == b.device and bool(torch.equal(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def _clone(x):
    return x.clone() if torch.is_tensor(x) else x


def memoize_prepare_condition(model):
    """Makes a repeat of the last `model.prepare_condition` call (all keyword arguments equal, tensors compared by value) return that call's result."""
    original = model.prepare_condition
    last = {}

    def prepare_condition(**kw):
        given = {k: v for k, v in kw.items() if v is not None}     # an argument passed as None is the same as one left out
        if last and last["keys"] == set(given) and all(_same(last["args"][k], v) for k, v in given.items()):
            return tuple(_clone(x) for x in last["result"])
        result = original(**kw)
        last["keys"] = set(given)
        last["args"] = {k: _clone(v) for k, v in given.items()}
        last["result"] = tuple(_clone(x) for x in result)
        return result

    prepare_condition.reset = last.clear
    model.prepare_condition = prepare_condition
    return model


def new_request(handler):
    """Called at the start of each request: the memo of `prepare_condition` holds one request's call, never the next request's."""
    reset = getattr(handler.model.prepare_condition, "reset", None)
    if reset is not None:
        reset()


def _digest(*tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(f"{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def cache_timbre(timbre_encoder, entries: int = 4):
    """Caches `timbre_encoder.forward(refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask)` by the bytes of its two inputs."""
    original = timbre_encoder.forward
    cache = {}

    def forward(refer_audio_acoustic_hidden_states_packed=None, refer_audio_order_mask=None, *a, **kw):
        if a or kw or refer_audio_acoustic_hidden_states_packed is None or refer_audio_order_mask is None:
            return original(refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask, *a, **kw)
        key = _digest(refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask)
        if key not in cache:
            while len(cache) >= entries:
                cache.pop(next(iter(cache)))
            out = original(refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask)
            cache[key] = tuple(_clone(x) for x in out)
            return out
        return tuple(_clone(x) for x in cache[key])

    timbre_encoder.forward = forward
    return timbre_encoder


def freeze_gc():
    """Call once after the models are loaded."""
    gc.collect()
    gc.freeze()


def apply(handler, steps, worker=None) -> tuple:
    """Installs the named steps on a loaded ACE-Step handler. Returns the steps installed."""
    steps = tuple(steps)
    chip_steps = [s for s in steps if s in CHIP_STEPS]
    if chip_steps and worker is None:
        raise ValueError(f"{', '.join(chip_steps)} run on the chip and need a chip session (device=chip)")
    if chip_steps:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import acestep_chip
        if "kv_on_chip" in steps:
            acestep_chip.install_kv_on_chip(worker)
        if "text_on_chip" in steps:
            acestep_chip.install_text_on_chip(worker, handler)
        if "lyric_on_chip" in steps:
            acestep_chip.install_lyric_on_chip(worker, handler)
    if "timbre_cache" in steps:
        cache_timbre(handler.model.encoder.timbre_encoder)
    if "cond_cache" in steps:
        memoize_prepare_condition(handler.model)
    if "gc_freeze" in steps:
        freeze_gc()
    return steps
