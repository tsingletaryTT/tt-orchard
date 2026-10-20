#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Trace a model's forward pass on the meta device and write the ATen ops it uses.

`tt-orchard inventory` runs this with the reference interpreter (torch and transformers must import):

    python ops_inventory.py --model-dir DIR --out raw-ops.json [--layers 2] [--seq 8] [--class NAME]

DIR needs only config.json. No weights are read and nothing is downloaded: the model is built on the meta device, so a
27B model traces in seconds. The graph comes from torch.export(strict=False) on one prefill-sized input
(input_ids of shape (1, SEQ)) with use_cache=False. Without use_cache=False the export fails on models that return a
DynamicCache. Decoding-time cache and state updates are not traced. A recurrent model's slow reference path unrolls the
recurrence over the SEQ positions, so its scan ops do appear.

What it will not do: import a repo's own modeling code. A config.json with `auto_map` is written as `skipped`
(exit 0), and so is an architecture the installed transformers does not have, or a name that is not a model class
(a subclass of PreTrainedModel): a downloaded config.json must not choose which function of transformers runs.
The traced class is the class config.json names, unless --class says otherwise (for example MoshiForCausalLM, the text model inside MoshiForConditionalGeneration,
which takes audio inputs this script does not build). The output says which class was traced.

Exit codes: 0 written (including a `skipped` file), 2 config.json missing, 3 the build or the export failed (the file
is written with the failure), 4 torch or transformers does not import.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = 1
LAYER_KEYS = ("num_hidden_layers", "n_layer", "num_layers")
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.\d+(?:\.|$)")
MAX_MODULES = 5
NOTE = ("prefill graph from torch.export(strict=False, use_cache=False) on the meta device; "
        "cache and state updates of decoding are not traced. Bodies of higher-order operations (for example "
        "wrap_with_set_grad_enabled) were traversed, so the ops inside them are counted; the wrapper itself is not an op")


def write(path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reduce_layers(config, wanted: int):
    """(layers in the config, layers traced). Sets the first recognised layer-count attribute to at most `wanted`.
    (None, None) when the config has none; the model is then traced as built."""
    for key in LAYER_KEYS:
        value = getattr(config, key, None)
        if isinstance(value, int) and not isinstance(value, bool):
            traced = min(wanted, value)
            try:
                setattr(config, key, traced)
            except Exception:                 # a read-only property: trace at full depth and say so
                return value, value
            return value, traced
    return None, None


def sample_of(val):
    """Output shapes and dtype of one graph node, or None."""
    items = val if isinstance(val, (list, tuple)) else [val]
    shapes, dtype = [], None
    for v in items:
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            shapes.append(list(v.shape))
            dtype = dtype or str(v.dtype)
    return {"output_shapes": shapes, "dtype": dtype} if shapes else None


def module_path(node) -> str:
    stack = node.meta.get("nn_module_stack") or {}
    if not stack:
        return ""
    last = list(stack.values())[-1]
    return last[0] if isinstance(last, (tuple, list)) and last else str(last)


def graph_modules(ep):
    """Every GraphModule reachable from the exported program's top graph, once each. torch.export puts the ops that run
    inside a higher-order operation (wrap_with_set_grad_enabled and similar) in a sub-GraphModule that
    `ep.graph.nodes` does not enter."""
    import torch
    seen, found = set(), []
    for mod in ep.graph_module.modules():
        if isinstance(mod, torch.fx.GraphModule) and id(mod) not in seen:
            seen.add(id(mod))
            found.append(mod)
    return found


def aggregate(nodes):
    """(ops, python_nodes, higher_order_nodes) for call_function nodes.

    ops: one record per ATen op overload. python_nodes: {str(target): count} for call_function nodes whose target is
    not an OpOverload (operator.getitem is tuple indexing in the graph and is no operation). higher_order_nodes:
    {name: count} for HigherOrderOperator wrappers; each owns one body, and the body's ops are counted by the caller
    walking every GraphModule. Neither of the last two appears in `ops`."""
    import torch
    ops: dict[str, dict] = {}
    python_nodes: dict[str, int] = {}
    hops: dict[str, int] = {}
    for node in nodes:
        if node.op != "call_function":
            continue
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            name = node.target.name()
            hops[name] = hops.get(name, 0) + 1
            continue
        if not isinstance(node.target, torch._ops.OpOverload):
            python_nodes[str(node.target)] = python_nodes.get(str(node.target), 0) + 1
            continue
        rec = ops.setdefault(str(node.target), {"op": str(node.target), "count": 0, "count_in_layers": 0,
                                                "sample": None, "modules": []})
        rec["count"] += 1
        path = module_path(node)
        if LAYER_RE.search(path):
            rec["count_in_layers"] += 1
        if rec["sample"] is None:
            rec["sample"] = sample_of(node.meta.get("val"))
        if path and path not in rec["modules"] and len(rec["modules"]) < MAX_MODULES:
            rec["modules"].append(path)
    return sorted(ops.values(), key=lambda r: (-r["count"], r["op"])), python_nodes, hops


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--class", dest="cls")
    a = ap.parse_args(argv)
    mdir = Path(a.model_dir)
    cfg_path = mdir / "config.json"
    if not cfg_path.is_file():
        print(f"error: {cfg_path} does not exist", file=sys.stderr)
        return 2
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = {"schema": SCHEMA, "model_dir": str(mdir), "architectures": raw_cfg.get("architectures") or [],
            "seq_len": a.seq, "note": NOTE, "failures": []}
    if "auto_map" in raw_cfg:                                   # remote code: never imported
        write(a.out, {**base, "skipped": "needs remote code: config.json has auto_map"})
        return 0
    if not isinstance(base["architectures"], list):
        write(a.out, {**base, "skipped": f"architectures in config.json is not a list: {base['architectures']!r:.200}"})
        return 0
    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    base.update(torch=torch.__version__, transformers=transformers.__version__)
    name = a.cls or (base["architectures"][0] if base["architectures"] else None)
    if name is not None and not isinstance(name, str):
        write(a.out, {**base, "skipped": f"the architecture name {name!r} is not a string"})
        return 0
    try:
        # transformers is a lazy module: a lookup can raise when the model's backend fails to import
        cls = getattr(transformers, name, None) if name else None
    except Exception as exc:
        write(a.out, {**base, "skipped": f"looking up {name!r} in transformers failed: "
                                         f"{type(exc).__name__}: {str(exc)[:300]}"})
        return 0
    if cls is None:
        write(a.out, {**base, "skipped": f"class {name!r} is not in the installed transformers {transformers.__version__}"})
        return 0
    # config.json comes from the Hub, so the name it gives must not choose which public callable of transformers
    # runs (a function such as `pipeline`, or a class such as AutoTokenizer). Only a model class is built.
    try:
        model_base = transformers.PreTrainedModel
    except Exception as exc:
        write(a.out, {**base, "skipped": f"reading transformers.PreTrainedModel failed: "
                                         f"{type(exc).__name__}: {str(exc)[:300]}"})
        return 0
    if not (isinstance(cls, type) and isinstance(model_base, type) and issubclass(cls, model_base)):
        write(a.out, {**base, "skipped": f"{name!r} is not a transformers model class (a subclass of PreTrainedModel); "
                                         "nothing was built"})
        return 0
    base["traced_class"] = name
    try:
        config = transformers.AutoConfig.from_pretrained(str(mdir), trust_remote_code=False)
        base["layers_in_config"], base["layers_traced"] = reduce_layers(config, a.layers)
        with torch.device("meta"):
            model = cls(config).eval()
        ids = torch.zeros(1, a.seq, dtype=torch.long, device="meta")
    except Exception as exc:
        write(a.out, {**base, "ops": [], "failures": [{"stage": "build", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}]})
        return 3
    try:
        ep = torch.export.export(model, (), {"input_ids": ids, "use_cache": False}, strict=False)
    except Exception as exc:
        write(a.out, {**base, "ops": [], "failures": [{"stage": "export", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}]})
        return 3
    nodes = [n for gm in graph_modules(ep) for n in gm.graph.nodes]
    ops, python_nodes, hops = aggregate(nodes)
    write(a.out, {**base, "ops": ops, "python_nodes": python_nodes, "higher_order_nodes": hops})
    return 0


if __name__ == "__main__":
    sys.exit(main())
