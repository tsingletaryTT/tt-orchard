# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Print the effective construction of the reference model: what each module's constructor resolved to (issue 541).

Run by the supervisor under the reference interpreter (the interpreter that has torch and transformers), on the CPU, with no device:

    python3 -I effective_construction.py SNAPSHOT_DIR

Why it exists. The reference excerpts (orchard/port_excerpts.py) print each class as its signature and its `forward`. A setting that a
module's constructor chooses is not in either. In run 18 of the tiny Qwen4 rehearsal the gated norm's constructor was called with
`activation=config.output_gate_type or config.hidden_act`, the config said `sigmoid`, and the only line the coder could see was the class
default `activation: str = "silu"`. The port used silu and scored 0.5625 for five repair rounds. This script instantiates the reference
on the `meta` device (so no weights are allocated) and reads the settings off the constructed modules, which are the resolved values.

What it prints. One line that starts with MARK and holds a JSON object:

    {"model_class": ..., "modules_total": N, "classes": [{"class": ..., "module": ..., "signature": [{"name", "default"}...],
     "variants": [{"count": n, "paths": [...], "attrs": {...}}...], "variants_left_out": k}...], "classes_left_out": j}

`signature` lists the constructor parameters that have a default. `attrs` holds the scalar attributes of the constructed module (numbers,
strings, booleans, short lists of them, and the name of a function-valued attribute such as `act_fn`) plus `children`, the class names of
its named child modules. Per-layer index attributes (`layer_idx`) are left out so that the modules of every layer fall in one variant.
The output is sorted, so two runs on the same checkpoint print the same bytes. The rendering and the comparison with the defaults are the
supervisor's (orchard/port_construction.py); this file only reads.

`describe` takes any object with `named_modules()`, so the tests run it on fakes without torch."""
import inspect
import math
import re
import sys

MARK = "ORCHARD_EFFECTIVE_CONSTRUCTION"
MAX_CLASSES = 80                # classes kept; the rest are counted
MAX_VARIANTS = 6                # distinct attribute sets kept per class; the rest are counted
MAX_PATHS = 3                   # example module paths kept per variant
MAX_ATTRS = 40                  # attributes kept per module
MAX_STR = 80
MAX_ITEMS = 8                   # elements kept of a tuple or list attribute
MAX_CHILDREN = 16
MAX_PARAMS = 40
MAX_MODULES = 200_000           # modules read; a model with more is cut and says so
PER_INSTANCE = frozenset({"layer_idx", "layer_id", "layer_index", "layer_number", "block_idx", "block_id", "idx", "index", "training"})
_NUMBER = re.compile(r"(?<=\.)\d+(?=\.|$)")
_NOT_SET = object()


def scalar(value):
    """The JSON-ready form of an attribute value, or _NOT_SET when it is not a setting (a tensor, a module, a config object)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value[:MAX_STR] if isinstance(value, str) else value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_ITEMS:
            return _NOT_SET
        items = [scalar(v) for v in value]
        return _NOT_SET if any(i is _NOT_SET or isinstance(i, list) for i in items) else items
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    if inspect.isfunction(value) or inspect.isbuiltin(value) or inspect.isclass(value):
        return "<callable " + (getattr(value, "__qualname__", None) or getattr(value, "__name__", "?")) + ">"
    return _NOT_SET


def attributes(module) -> dict:
    """The settings of one constructed module, in sorted key order."""
    out = {}
    try:
        items = sorted(vars(module).items(), key=lambda kv: str(kv[0]))
    except TypeError:
        return out
    for key, value in items:
        if not isinstance(key, str) or key.startswith("_") or key in PER_INSTANCE:
            continue
        shown = scalar(value)
        if shown is not _NOT_SET:
            out[key] = shown
        if len(out) >= MAX_ATTRS:
            break
    kids = {}
    try:
        modules = getattr(module, "_modules", None) or {}
        for name, child in modules.items():
            if child is not None and isinstance(name, str) and not name.isdigit() and len(kids) < MAX_CHILDREN:
                kids[name] = type(child).__qualname__
    except Exception:
        kids = {}
    if kids:
        out["children"] = dict(sorted(kids.items()))
    return out


def signature(cls) -> list:
    """The constructor parameters of `cls` that have a default, with the default in the same form `attributes` uses."""
    try:
        params = inspect.signature(cls.__init__).parameters.values()
    except (TypeError, ValueError):
        return []
    out = []
    for p in params:
        if p.name == "self" or p.default is inspect.Parameter.empty or len(out) >= MAX_PARAMS:
            continue
        shown = scalar(p.default)
        out.append({"name": p.name, "default": None if shown is _NOT_SET else shown, "shown": shown is not _NOT_SET})
    return out


def pattern(path: str) -> str:
    """A module path with each layer number written {i}: `model.layers.3.mlp` -> `model.layers.{i}.mlp`."""
    return _NUMBER.sub("{i}", path)


def describe(model) -> dict:
    groups = {}
    total = 0
    for path, module in model.named_modules():
        total += 1
        if total > MAX_MODULES:
            break
        cls = type(module)
        key = (cls.__module__, cls.__qualname__)
        entry = groups.setdefault(key, {"cls": cls, "variants": {}})
        attrs = attributes(module)
        sig = repr(sorted(attrs.items(), key=lambda kv: kv[0]))
        v = entry["variants"].setdefault(sig, {"count": 0, "paths": [], "attrs": attrs})
        v["count"] += 1
        shown = pattern(path or "<model>")
        if shown not in v["paths"] and len(v["paths"]) < MAX_PATHS:
            v["paths"].append(shown)
    classes = []
    for (mod_name, qual), entry in sorted(groups.items()):
        variants = sorted(entry["variants"].values(), key=lambda v: (-v["count"], repr(v["attrs"])))
        classes.append({"class": qual, "module": mod_name, "signature": signature(entry["cls"]),
                        "variants": variants[:MAX_VARIANTS], "variants_left_out": max(0, len(variants) - MAX_VARIANTS)})
    kept = classes[:MAX_CLASSES]
    return {"model_class": type(model).__qualname__, "modules_total": total, "classes": kept,
            "classes_left_out": max(0, len(classes) - len(kept)), "modules_cut": total > MAX_MODULES}


def build_model(snapshot: str):
    """The reference model for the snapshot's config on the meta device (no weights are read or allocated)."""
    import torch
    import transformers
    config = transformers.AutoConfig.from_pretrained(snapshot, trust_remote_code=False)
    names = [a for a in (getattr(config, "architectures", None) or []) if isinstance(a, str)]
    last = None
    for factory in [*(getattr(transformers, a, None) for a in names), transformers.AutoModelForCausalLM, transformers.AutoModel]:
        if factory is None:
            continue
        try:
            with torch.device("meta"):
                if hasattr(factory, "from_config"):             # an Auto class
                    return factory.from_config(config)
                return factory(config)                          # the architecture class the config names
        except Exception as exc:        # the next factory is tried; the last error is what the caller reports
            last = exc
    raise RuntimeError(f"no reference class could be built from the config: {type(last).__name__}: {str(last)[:200]}")


def main(argv) -> int:
    import json
    if len(argv) != 2:
        print("usage: effective_construction.py SNAPSHOT_DIR", file=sys.stderr)
        return 2
    print(MARK + " " + json.dumps(describe(build_model(argv[1])), sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
