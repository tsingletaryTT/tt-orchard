# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fusion port candidates: which fused decode ops of the nearest bundle a new model could reuse (issue 184).

A published fused-decode bundle ships its fused ops as `ttnn.generic_op` programs in Python model code. Each op checks the
model's dimensions with `assert` statements and reads its tuning from module constants and environment switches. This module
reads those checks from the bundle's source, evaluates them against a new model's config, and sorts each op into one of
three classes:

  applies as is                  every check passes and no tuned constant depends on a dimension that differs
  applies with a changed constant  a check on the Python side fails, or a constant tuned for the bundle's dimension must be re-set
  needs new code                 a check that guards kernel code fails, or the bundle has no such op for a mixer the model uses

An op in the third class is written down as a gap. Nothing here attempts it.

Reading the bundle is read-only and never runs its code: files are opened without following links, limited in size and in
count, parsed with `ast`, and the constraints are evaluated by a small interpreter that accepts arithmetic, comparisons and
names only (no calls, no attributes other than `self.x` assigned earlier in the same function). A term it cannot evaluate is
listed with value `None` and does not change the class.

The per-op table `OPS` is the part that knows the bundle's layout. It names the module, the class or function that holds the
checks, the environment switch that turns the op on, how each variable in a check maps to a model dimension, and which checks
guard kernel code. Add a row when a bundle ships a new fused op.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field

from orchard import hazards as hz
from orchard import model_config as mc
from orchard import safeio

MAX_SOURCE_BYTES = 1 << 20        # one source file; the real fused-op modules are 100 to 1,500 lines
MAX_FILES = 20_000                # directory entries looked at
MAX_DEPTH = 12
MAX_TOTAL_BYTES = 64 << 20        # all source read for the env switch search
MAX_CONSTRAINTS = 200

AS_IS = "applies as is"
CHANGED = "applies with a changed constant"
NEW_CODE = "needs new code"
NOT_USED = "not used by this model"
STATUSES = (AS_IS, CHANGED, NEW_CODE, NOT_USED)

TILE = 32
FOUR_DIES_SOURCE = ("the coder bundle's model card: the fabric layer boundary runs only on four dies "
                    "(docs/run-logs/2026-10-28-coder-fused-decode.md)")


class FusionPortError(ValueError):
    """A config or a bundle directory could not be used. The message names the problem and never holds file content."""


# ---------------------------------------------------------------------------------------------------- the op table

@dataclass(frozen=True)
class Tunable:
    """A constant the bundle tuned for its own dimensions. `kind` is "env" (an `os.environ.get(NAME, default)` anywhere in the
    bundle) or "const" (a module constant of the op's module)."""
    name: str
    kind: str
    depends: tuple
    why: str


@dataclass(frozen=True)
class OpSpec:
    name: str
    family: str                    # router, moe decode, mixer, layer boundary, sampler
    module: str                    # path suffix of the module that holds the op
    scope: str                     # "class:NAME" (its __init__) or "func:NAME" (its return expression)
    gate: str | None               # environment switch that turns the op on
    bindings: dict                 # variable in a check -> quantity (see quantities())
    kernel_side: object = ()       # "all", or the texts of the checks that guard kernel code
    mixer: str | None = None       # layer type the op serves; the op is not used by a model without it
    tunables: tuple = ()
    extra: tuple = ()              # (text, source, kernel_side) checks the code does not state


OPS = (
    OpSpec("router_topk", "router", "moe_decode/topk.py", "class:RouterTopK", "QWEN35_ROUTER_TOPK_FUSED",
           {"num_experts": "routed_experts", "k": "top_k"}, kernel_side="all",
           tunables=(Tunable("QWEN35_ROUTER_TOPK_ROUTE_K", "env", ("routed_experts", "top_k"),
                             "the large-k route was measured at 512 experts and top-k 10"),)),
    OpSpec("moe_decode", "moe decode", "moe_decode/fused.py", "func:supported", "QWEN35_MOE_FUSED_DECODE",
           {"num_experts": "routed_experts", "hidden": "hidden", "inter_local": "moe_inter_per_die"}, kernel_side="all"),
    OpSpec("gdn_decode", "mixer (gated delta net)", "gdn_decode/ops.py", "class:GdnDecode", "QWEN35_GDN_FUSED_DECODE",
           {"num_k_heads": "linear_k_heads_per_die", "num_v_heads": "linear_v_heads_per_die", "head_k_dim": "linear_key_dim",
            "head_v_dim": "linear_value_dim", "proj_dim": "gdn_proj_dim"}, kernel_side="all", mixer="linear_attention"),
    OpSpec("attn_decode", "mixer (gated attention)", "attn_decode/ops.py", "class:AttnDecodePrep", "QWEN35_ATTN_DECODE_OPS",
           {"num_heads": "heads_per_die", "num_kv_heads": "kv_heads_per_die", "head_dim": "head_dim", "rope_dim": "rope_dim"},
           kernel_side="all", mixer="full_attention"),
    OpSpec("sampler_rows_topk", "sampler", "decode_sampling/ops.py", "class:RowsTopK", "QWEN35_ROWS_TOPK",
           {"width": "vocab_per_die"}, kernel_side=("width % 32 == 0", "k == 32"),
           tunables=(Tunable("GRID", "const", ("vocab_per_die",), "the scan grid was sized for the bundle's vocabulary per die"),)),
    OpSpec("fabric_boundary", "layer boundary", "fabric_boundary/ops.py", "class:FabricBoundary", "QWEN35_FABRIC_BOUNDARY",
           {"hidden": "hidden"}, kernel_side="all",
           extra=(("dies == 4", FOUR_DIES_SOURCE, True),)),
    OpSpec("addnorm", "layer boundary", "addnorm/ops.py", "class:AddNorm", "QWEN35_ADDNORM_DECODE",
           {"hidden": "hidden"}, kernel_side=()),
)
# Defaults for names a check uses that no binding supplies and the function signature does not default.
COMMON_NAMES = {"TILE": TILE, "banks": 8}


# ----------------------------------------------------------------------------------------- the expression interpreter

class _Unknown(Exception):
    """An expression the interpreter will not or cannot evaluate."""


_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b, ast.LShift: lambda a, b: a << b if 0 <= b <= 64 else _unknown(),
        ast.RShift: lambda a, b: a >> b}
_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}


def _unknown():
    raise _Unknown()


def _ev(node, env):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, str)) or node.value is None:
            return node.value
        _unknown()
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        _unknown()
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self" and f"self.{node.attr}" in env:
            return env[f"self.{node.attr}"]
        _unknown()
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_ev(e, env) for e in node.elts]
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not v
        if isinstance(node.op, ast.USub):
            return -v
        _unknown()
    if isinstance(node, ast.BinOp):
        f = _BIN.get(type(node.op))
        if f is None:
            _unknown()
        a, b = _ev(node.left, env), _ev(node.right, env)
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            _unknown()
        try:
            return f(a, b)
        except ZeroDivisionError:
            _unknown()
    if isinstance(node, ast.IfExp):
        return _ev(node.body, env) if _ev(node.test, env) else _ev(node.orelse, env)
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        for op, right_node in zip(node.ops, node.comparators):
            right = _ev(right_node, env)
            f = _CMP.get(type(op))
            if f is None:
                _unknown()
            if not f(left, right):
                return False
            left = right
        return True
    _unknown()


def eval_constraint(text: str, env: dict):
    """True or False for the expression `text` under the names in `env`; None when it uses anything unknown (a name with no
    value, a call, an attribute, a type the interpreter does not handle). Never executes code."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
        v = _ev(tree.body, env)
    except (_Unknown, SyntaxError, ValueError, RecursionError, TypeError, MemoryError):
        return None
    return bool(v) if isinstance(v, (bool, int, float)) else None


def split_conjuncts(text: str) -> list[str]:
    """The top-level `and` terms of an expression, each as source text."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return [text.strip()]
    terms = node.values if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) else [node]
    return [ast.unparse(t) for t in terms]


# ------------------------------------------------------------------------------------------------------ reading files

def _scan(root: str) -> dict:
    """Index the `.py` files below `root`: {relative path: absolute path}. Links are not followed and are counted."""
    index, links, seen = {}, 0, 0
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            seen += 1
            if seen > MAX_FILES:
                return {"files": index, "links": links, "truncated": True}
            try:
                if e.is_symlink():
                    links += 1
                elif e.is_dir(follow_symlinks=False):
                    if depth < MAX_DEPTH:
                        stack.append((e.path, depth + 1))
                elif e.is_file(follow_symlinks=False) and e.name.endswith(".py"):
                    index[os.path.relpath(e.path, root)] = e.path
            except OSError:
                continue
    return {"files": index, "links": links, "truncated": False}


def _read_source(path: str):
    """(text, None) or (None, reason). The reason holds no file content."""
    try:
        raw = safeio.read_bytes(path, max_bytes=MAX_SOURCE_BYTES)
    except safeio.TooLarge:
        return None, f"larger than {MAX_SOURCE_BYTES} bytes"
    except safeio.SafeReadError as exc:
        return None, f"not readable ({exc.reason})"
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "not UTF-8 text"


def _module_constants(tree: ast.Module) -> dict:
    out = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            try:
                v = _ev(stmt.value, {**COMMON_NAMES, **out})
            except _Unknown:
                continue
            if isinstance(v, (int, float, bool, str, list)):
                out[stmt.targets[0].id] = v
    return out


def _find_scope(tree: ast.Module, scope: str):
    """(function node, kind) for "class:NAME" (its __init__) or "func:NAME"."""
    kind, _, name = scope.partition(":")
    for stmt in tree.body:
        if kind == "class" and isinstance(stmt, ast.ClassDef) and stmt.name == name:
            for item in stmt.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
        if kind == "func" and isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            return stmt
    return None


def _signature_defaults(fn: ast.FunctionDef) -> dict:
    out = {}
    args = fn.args
    pos = args.args
    for a, d in zip(pos[len(pos) - len(args.defaults):], args.defaults):
        try:
            v = _ev(d, {})
        except _Unknown:
            continue
        if isinstance(v, (int, float, bool)):
            out[a.arg] = v
    return out


def _target_name(t):
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
        return f"self.{t.attr}"
    return None


def _constraints(fn: ast.FunctionDef):
    """(constraints, prelude, defaults). Each constraint is {"text", "prelude"} where prelude is the number of assignments before
    it; `prelude` is the ordered list of (target, source text) assignments made in the function body before its end."""
    prelude, cons = [], []
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and _target_name(stmt.targets[0]):
            prelude.append((_target_name(stmt.targets[0]), ast.unparse(stmt.value)))
        elif isinstance(stmt, ast.Assert):
            for t in split_conjuncts(ast.unparse(stmt.test)):
                cons.append({"text": t, "n_prelude": len(prelude)})
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            for t in split_conjuncts(ast.unparse(stmt.value)):
                cons.append({"text": t, "n_prelude": len(prelude)})
        if len(cons) > MAX_CONSTRAINTS:
            break
    return cons[:MAX_CONSTRAINTS], prelude, _signature_defaults(fn)


_ENV_RE = r"""environ\.get\(\s*["']%s["']\s*,\s*["']([^"']*)["']"""


def _env_default(texts: dict, name: str):
    """The distinct defaults of `os.environ.get("NAME", "default")` found in the source texts, in file order."""
    rx = re.compile(_ENV_RE % re.escape(name))
    found = []
    for rel in sorted(texts):
        for m in rx.finditer(texts[rel]):
            if m.group(1) not in found:
                found.append(m.group(1))
    return found


def read_bundle_inventory(bundle_dir) -> dict:
    """The fused ops found in the code under `bundle_dir`: one entry per row of OPS, with the checks the code states.

    Read-only; the code is parsed and never run. Raises FusionPortError only when `bundle_dir` is not a directory. A module that
    is missing, ambiguous, too large or unparseable is reported on its op with `found` false and a `note`."""
    root = os.fspath(bundle_dir)
    if not os.path.isdir(root) or os.path.islink(root):
        raise FusionPortError("bundle directory is not a directory")
    scan = _scan(root)
    index = scan["files"]
    notes = []
    if scan["truncated"]:
        notes.append(f"the directory listing stopped at {MAX_FILES} entries")
    if scan["links"]:
        notes.append(f"{scan['links']} symbolic link(s) were not followed")
    texts, total = {}, 0
    for rel, path in sorted(index.items()):                    # source text for the environment switch search
        text, why = _read_source(path)
        if text is None:
            continue
        total += len(text)
        if total > MAX_TOTAL_BYTES:
            notes.append("source search stopped at the size limit")
            break
        texts[rel] = text
    ops = []
    for spec in OPS:
        ops.append(_inventory_op(spec, index, texts))
    return {"bundle_dir_files": len(index), "ops": ops, "notes": notes}


def _inventory_op(spec: OpSpec, index: dict, texts: dict) -> dict:
    parts = spec.module.split("/")
    matches = sorted(rel for rel in index if rel.split(os.sep)[-len(parts):] == parts)
    base = {"name": spec.name, "family": spec.family, "module": spec.module, "found": False, "gate": None, "constraints": [],
            "constants": {}, "tunables": [], "note": ""}
    if not matches:
        base["note"] = f"no file ending in {spec.module}"
        return base
    if len(matches) > 1:
        base["note"] = f"more than one file ends in {spec.module}"
        return base
    rel = matches[0]
    text, why = _read_source(index[rel])
    if text is None:
        base["note"] = f"{spec.module} is {why}"
        return base
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        base["note"] = f"{spec.module} did not parse"
        return base
    fn = _find_scope(tree, spec.scope)
    if fn is None:
        base["note"] = f"{spec.module} has no {spec.scope}"
        return base
    cons, prelude, defaults = _constraints(fn)
    consts = _module_constants(tree)
    base.update(found=True, path=rel, constants=consts, _prelude=prelude, _defaults=defaults)
    base["constraints"] = [{"text": c["text"], "n_prelude": c["n_prelude"]} for c in cons]
    if spec.gate:
        defaults_found = _env_default(texts, spec.gate)
        base["gate"] = {"env": spec.gate, "code_default": defaults_found[0] if defaults_found else None}
        if len(defaults_found) > 1:
            base["gate"]["other_defaults"] = defaults_found[1:]
    for t in spec.tunables:
        if t.kind == "env":
            v = _env_default(texts, t.name)
            if v:
                base["tunables"].append({"name": t.name, "value": v[0], "depends_on": list(t.depends), "why": t.why})
        elif t.name in consts:
            base["tunables"].append({"name": t.name, "value": consts[t.name], "depends_on": list(t.depends), "why": t.why})
    return base


# ------------------------------------------------------------------------------------------------- model quantities

def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def quantities(config, dies: int) -> dict:
    """The numbers the checks use, per die where the bundle shards by tensor parallelism (heads and expert width are split over
    `dies`, rounded up, as the bundle's loader does; vocabulary is split over dies). A number the model does not have is None."""
    d = mc.dims(config)
    cfg = mc.text_config(config)
    lin = d.linear or {}
    rot = cfg.get("partial_rotary_factor")
    if rot is None and isinstance(cfg.get("rope_parameters"), dict):
        rot = cfg["rope_parameters"].get("partial_rotary_factor")
    rope_dim = int(d.head_dim * (rot if isinstance(rot, (int, float)) else 1.0)) if d.head_dim else None
    kh, vh = lin.get("key_heads"), lin.get("value_heads")
    # model_config.dims reports the linear-attention key and value widths summed over heads; the op checks the per-head width
    kd = lin["key_dim"] // kh if kh and lin.get("key_dim") and lin["key_dim"] % kh == 0 else None
    vd = lin["value_dim"] // vh if vh and lin.get("value_dim") and lin["value_dim"] % vh == 0 else None
    q = {
        "dies": dies, "hidden": d.hidden, "routed_experts": d.routed_experts or None, "top_k": d.top_k or None,
        "moe_inter_per_die": _ceil_div(d.moe_intermediate, dies) if d.moe_intermediate else None,
        "heads_per_die": _ceil_div(d.heads, dies), "kv_heads_per_die": _ceil_div(d.kv_heads, dies), "head_dim": d.head_dim,
        "rope_dim": rope_dim, "vocab_per_die": _ceil_div(d.vocab, dies),
        "linear_k_heads_per_die": _ceil_div(kh, dies) if kh else None, "linear_v_heads_per_die": _ceil_div(vh, dies) if vh else None,
        "linear_key_dim": kd, "linear_value_dim": vd,
    }
    if kh and vh and kd and vd:
        nk, nv = q["linear_k_heads_per_die"], q["linear_v_heads_per_die"]
        qkv = 2 * nk * kd + nv * vd
        q["gdn_proj_dim"] = qkv + nv * vd + 2 * TILE            # [q | k | v | z | a | b], a and b padded to one tile each
    else:
        q["gdn_proj_dim"] = None
    return q


def _mixers_of(config) -> set:
    return set(mc.dims(config).layer_types)


def _kernel_side(spec: OpSpec, text: str) -> bool:
    return spec.kernel_side == "all" or text in spec.kernel_side


def _env_for(spec: OpSpec, op: dict, q: dict) -> dict:
    env = dict(COMMON_NAMES)
    env.update(op.get("constants", {}))
    env.update(op.get("_defaults", {}))
    for var, quantity in spec.bindings.items():
        if q.get(quantity) is not None:
            env[var] = q[quantity]
    return env


def _evaluate_op(spec: OpSpec, op: dict, q: dict) -> list:
    """Each of the op's checks with its value for quantities `q`. Assignments before a check are replayed first."""
    base = _env_for(spec, op, q)
    out = []
    for c in op["constraints"]:
        env = dict(base)
        for target, src in op["_prelude"][: c["n_prelude"]]:
            try:
                env[target] = _ev(ast.parse(src, mode="eval").body, env)
            except (_Unknown, SyntaxError, ValueError, RecursionError, TypeError):
                env.pop(target, None)
        out.append({"text": c["text"], "value": eval_constraint(c["text"], env), "kernel_side": _kernel_side(spec, c["text"]),
                    "source": f"assert in {spec.module}"})
    for text, source, kside in spec.extra:
        out.append({"text": text, "value": eval_constraint(text, {"dies": q["dies"]}), "kernel_side": kside, "source": source})
    return out


def _classify(spec, op, model_q, bundle_q, model_mixers):
    """(status, reasons, constraints, port) for one op."""
    if spec.mixer and spec.mixer not in model_mixers:
        return NOT_USED, [f"the model has no {spec.mixer} layers"], [], []
    if not op["found"]:
        return NEW_CODE, [f"the model needs this op and it is not in the bundle ({op['note']})"], [], []
    cons = _evaluate_op(spec, op, model_q)
    reasons, port = [], []
    failing = [c for c in cons if c["value"] is False]
    kernel = [c for c in failing if c["kernel_side"]]
    for c in kernel:
        reasons.append(f"check `{c['text']}` fails and guards kernel code ({c['source']})")
    if kernel:
        return NEW_CODE, reasons, cons, port
    for c in failing:
        reasons.append(f"check `{c['text']}` fails on the Python side: change the constant it names")
        port.append(c["text"])
    for t in op["tunables"]:
        if bundle_q is None:
            continue
        differ = [k for k in t["depends_on"] if model_q.get(k) != bundle_q.get(k)]
        if differ:
            vals = ", ".join(f"{k} {bundle_q.get(k)} -> {model_q.get(k)}" for k in differ)
            reasons.append(f"{t['name']} (bundle value {t['value']}) {t['why']}; this model differs: {vals}")
            port.append(t["name"])
    if reasons:
        return CHANGED, reasons, cons, port
    return AS_IS, ["every check that could be evaluated passes"], cons, port


def candidates(model_config, bundle_dir, *, bundle_config=None, dies: int = 2) -> dict:
    """The candidate list: for each fused op of the bundle in `bundle_dir`, its status for `model_config` (a config dict).

    `bundle_config` is the config of the model the bundle was built for; without it the constants tuned for the bundle's
    dimensions cannot be compared and the report says so. `dies` is the number of dies the model is split over (2 for one p300
    board). Raises FusionPortError for an unusable config or directory."""
    if isinstance(dies, bool) or not isinstance(dies, int) or dies < 1:
        raise FusionPortError("dies must be a whole number of at least 1")
    try:
        model_q = quantities(model_config, dies)
        bundle_q = quantities(bundle_config, dies) if bundle_config is not None else None
        mixers = _mixers_of(model_config)
        scan = hz.scan_model_hazards(model_config, bundle_config, dies=dies)
    except (mc.ModelConfigError, hz.HazardInputError) as exc:
        raise FusionPortError(str(exc)) from exc
    inv = read_bundle_inventory(bundle_dir)
    notes = list(inv["notes"])
    if bundle_q is None:
        notes.append("no bundle config was given, so constants tuned for the bundle's dimensions were not compared with the model's")
    ops, gaps, port = [], [], []
    by_name = {s.name: s for s in OPS}
    for op in inv["ops"]:
        spec = by_name[op["name"]]
        status, reasons, cons, ported = _classify(spec, op, model_q, bundle_q, mixers)
        quantity_names = sorted({q for v, q in spec.bindings.items() if any(v in c["text"] for c in op["constraints"])}
                                | {k for t in op["tunables"] for k in t["depends_on"]})
        row = {"name": spec.name, "family": spec.family, "status": status, "reasons": reasons, "module": spec.module,
               "found": op["found"], "gate": op["gate"], "depends_on": quantity_names, "constraints": cons,
               "tunables": op["tunables"],
               "values": {k: {"model": model_q.get(k), "bundle": bundle_q.get(k) if bundle_q else None} for k in quantity_names}}
        ops.append(row)
        if status == NEW_CODE:
            gaps.append({"op": spec.name, "family": spec.family, "reasons": reasons})
        if status == CHANGED:
            port.append({"op": spec.name, "constants": ported})
    counts = {s: sum(1 for o in ops if o["status"] == s) for s in STATUSES}
    return {"label": "prediction from the config and the bundle's source; nothing was run on a chip", "dies": dies, "ops": ops,
            "counts": counts, "gaps": gaps, "port": port, "notes": notes,
            "hazards": [h.as_dict() for h in scan.hazards]}


# --------------------------------------------------------------------------------------------------------- the command

def render_table(report: dict) -> str:
    """The report as plain text: one row per op, then the gaps and the notes. No right-hand border."""
    rows = [("op", "family", "status", "depends on")]
    for o in report["ops"]:
        rows.append((o["name"], o["family"], o["status"], ", ".join(o["depends_on"]) or "-"))
    w = [max(len(r[i]) for r in rows) for i in range(3)]
    lines = [f"{r[0]:<{w[0]}}  {r[1]:<{w[1]}}  {r[2]:<{w[2]}}  {r[3]}" for r in rows]
    lines.append("")
    for o in report["ops"]:
        for why in o["reasons"]:
            lines.append(f"{o['name']}: {why}")
    c = report["counts"]
    lines.append("")
    lines.append("; ".join(f"{c[s]} {s}" for s in STATUSES))
    for g in report["gaps"]:
        lines.append(f"gap (needs new code, not attempted): {g['op']}")
    if report["hazards"]:
        lines.append("hazards: " + ", ".join(h["name"] for h in report["hazards"]))
    for n in report["notes"]:
        lines.append(f"note: {n}")
    lines.append(report["label"])
    return "\n".join(lines)


def main(argv=None, out=None) -> int:
    """`tt-orchard fusion-candidates MODEL_CONFIG BUNDLE_DIR [--bundle-config FILE] [--dies N] [--json]`. Exit 2 on a bad input."""
    import argparse
    ap = argparse.ArgumentParser(prog="tt-orchard fusion-candidates",
                                 description="List the fused ops of a bundle's code and say which a new model could reuse.")
    ap.add_argument("model_config", help="the new model's config.json")
    ap.add_argument("bundle_dir", help="a directory holding the bundle's Python code (read only, never run)")
    ap.add_argument("--bundle-config", help="config.json of the model the bundle was built for (default: BUNDLE_DIR/config.json if present)")
    ap.add_argument("--dies", type=int, default=2, help="dies the model is split over (default 2, one p300 board)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    try:
        model = mc.read_model_config(os.path.realpath(a.model_config))     # a Hugging Face cache snapshot file is a link
        bpath = a.bundle_config or os.path.join(a.bundle_dir, "config.json")
        bundle_cfg = mc.read_model_config(os.path.realpath(bpath)) if (a.bundle_config or os.path.isfile(bpath)) else None
        report = candidates(model, a.bundle_dir, bundle_config=bundle_cfg, dies=a.dies)
    except (FusionPortError, mc.ModelConfigError) as exc:
        print(f"fusion-candidates: {exc}", file=out)
        return 2
    print(json.dumps(report, indent=2) if a.json else render_table(report), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
