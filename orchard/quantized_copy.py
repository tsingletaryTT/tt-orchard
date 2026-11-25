# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Say that a model is a quantized copy of a supported one (issue 63).

A quantized checkpoint (FP8, NVFP4, GPTQ, AWQ) stores weights and scales in integer or block formats. Stage 0's tensor comparison finds tensors that the
nearest model does not have and dtypes it does not use, and the class is `full-port`. No quantized checkpoint has been served on a Tenstorrent chip with this
harness, so the class stays `full-port` until a hardware test of one format says what the runtime accepts, converts or refuses. What can be said without a
board is what stage 0 saw. This module reads that from stage 0's own evidence (`evidence/tensor-compare.json`) and the model's `quantization_config` and gives a
hint for the pause text and the ledger. It never changes the class or the path, and every read is bounded and never raises."""
from __future__ import annotations

import json
import re
from pathlib import Path

from orchard import safeio

EVIDENCE_MAX = 64 * 1024 * 1024        # tensor-compare.json lists tensor names that only one model has; a large model's is some megabytes
CONFIG_MAX = 4 * 1024 * 1024
DELTA_MAX = 16 * 1024 * 1024
FORMAT_MAX = 40
SIGNAL_CHARS = 200
LABELS = {"fp8": "FP8", "nvfp4": "NVFP4", "mxfp8": "MXFP8", "gptq": "GPTQ", "awq": "AWQ", "compressed-tensors": "compressed-tensors"}
AN = ("fp8", "nvfp4", "mxfp8", "awq")             # formats whose name is said with a vowel sound: "an FP8 copy"
FLOAT8 = ("F8_E4M3", "F8_E5M2", "F8_E8M0")            # safetensors dtypes of 8-bit float weights and block scales
_METHOD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_NAME_SUFFIX = (("gptq", (".qweight", ".qzeros", ".g_idx")), ("fp8", (".weight_scale_inv",)), ("nvfp4", (".weight_scale_2", ".input_scale")))


def _method_format(config) -> tuple[str | None, str | None]:
    """(format, signal) from a model's `quantization_config`, or (None, None). The method is a short clean word; anything else is no signal."""
    if not isinstance(config, dict):
        return None, None
    method = config.get("quant_method")
    if not isinstance(method, str) or not method.strip() or method != method.strip() or not _METHOD.fullmatch(method):
        return None, None
    low = method.lower()
    algo = str(config.get("quant_algo") or "").lower()
    if low in ("modelopt", "nvfp4") and (low == "nvfp4" or "nvfp4" in algo):
        return "nvfp4", f"quantization_config says {method}" + (f" ({config.get('quant_algo')})" if algo else "")
    if low == "modelopt" and "fp8" in algo:
        return "fp8", f"quantization_config says {method} ({config.get('quant_algo')})"
    return low[:FORMAT_MAX], f"quantization_config says {method[:FORMAT_MAX]}"


def detect(*, model_dtypes, base_dtypes, only_in_model, quant_config, nearest_model) -> dict | None:
    """{"format", "nearest", "signals"} when the model looks like a quantized copy of the nearest model, else None. The model must show a signal the nearest model
    does not: a `quantization_config`, tensors that only it has and that are scales or packed weights, or an 8-bit float dtype. A nearest model that is quantized
    in the same way gives None. The order of the signals: the config, then the tensor names, then the dtypes."""
    md = model_dtypes if isinstance(model_dtypes, dict) else {}
    bd = base_dtypes if isinstance(base_dtypes, dict) else {}
    names = [n for n in (only_in_model if isinstance(only_in_model, list) else []) if isinstance(n, str)]
    base_f8 = any(k in bd for k in FLOAT8)
    fmt, signals = None, []
    cfg_fmt, cfg_signal = _method_format(quant_config)
    if cfg_fmt:
        fmt = cfg_fmt
        signals.append(cfg_signal)
    named: dict[str, list[str]] = {}
    for n in names:
        for kind, suffixes in _NAME_SUFFIX:
            if n.endswith(suffixes):
                named.setdefault(kind, []).append(n)
    f8 = [k for k in FLOAT8 if k in md and k not in bd]
    if named:
        kind = max(named, key=lambda k: len(named[k]))
        if kind == "fp8" and "nvfp4" in named:
            kind = "nvfp4"
        if kind == "fp8" and not f8 and base_f8:
            kind = None             # the nearest model has 8-bit float weights and scales too
        if kind:
            fmt = fmt or kind
            signals.append(f"{len(named[kind])} tensors only the model has end in {named[kind][0].rsplit('.', 1)[-1]}, for example {named[kind][0][:80]}")
    if f8 and not (base_f8 and not named and not cfg_fmt):
        fmt = fmt or "fp8"
        signals.append(f"{sum(md[k] for k in f8 if isinstance(md.get(k), int))} tensors in {'/'.join(f8)} that the nearest model does not use")
    if not fmt:
        return None
    return {"format": fmt, "nearest": str(nearest_model)[:200], "signals": [s[:SIGNAL_CHARS] for s in signals]}


def sentence(copy: dict) -> str:
    """The sentence for the pause text. It holds none of the words the status reader uses to give a pause another kind."""
    label = LABELS.get(copy["format"], copy["format"])
    seen = "; ".join(copy.get("signals") or [])
    return (f"The model looks like {'an' if copy['format'] in AN else 'a'} {label} quantized copy of {copy['nearest']}" + (f" (stage 0 saw: {seen})" if seen else "")
            + ". No quantized checkpoint has been served on a Tenstorrent chip by this harness, so the class stays full port until a hardware test of that format "
              "says what the runtime accepts, converts or refuses (issue 63)")


def _read(path, limit, *, snapshot=None):
    """A JSON object from `path`, or None. `snapshot` is for a file of the model's snapshot, where Hugging Face stores each file as a link into its blobs
    directory: the link is followed only when it ends inside the snapshot or its cache's blobs directory (`safeio.resolved_inside`, issue 494). Files of the run
    directory are never followed: a link there is refused."""
    try:
        if snapshot is not None:
            path = safeio.resolved_inside(path, snapshot)
            if path is None:
                return None
        return safeio.read_json(path, max_bytes=limit, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError):
        return None


def from_run(run_dir, model_snapshot) -> dict | None:
    """`detect` on a run: stage 0's tensor comparison, its delta.json's nearest model and the model snapshot's config.json. None when the evidence cannot be read,
    which includes a link, a file over its bound and a file that is not an object. A missing config or delta only loses a signal."""
    stage = Path(run_dir) / "stages" / "0"
    tensors = _read(stage / "evidence" / "tensor-compare.json", EVIDENCE_MAX)
    if tensors is None:
        return None
    model, base = tensors.get("model"), tensors.get("base")
    sets = tensors.get("raw_name_sets")
    delta = _read(stage / "delta.json", DELTA_MAX) or {}
    config = _read(Path(model_snapshot) / "config.json", CONFIG_MAX, snapshot=model_snapshot) if model_snapshot else None
    nearest = delta.get("nearest_model")
    return detect(model_dtypes=model.get("dtype_histogram") if isinstance(model, dict) else None,
                  base_dtypes=base.get("dtype_histogram") if isinstance(base, dict) else None,
                  only_in_model=sets.get("only_in_model") if isinstance(sets, dict) else None,
                  quant_config=config.get("quantization_config") if isinstance(config, dict) else None,
                  nearest_model=str(nearest).split("@", 1)[0] if isinstance(nearest, str) and nearest else "the nearest model")
