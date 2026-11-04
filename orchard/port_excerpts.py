# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Reference excerpts for the stage 2 coder: the source of the layer classes its model uses, in one file.

Why this exists (issue 427). In three rehearsal runs the stage 2 coder spent its first attempt reading, and in the third it paged through
the installed `transformers` modeling module for its model type with `sed` and never wrote the port. The facts file
(`orchard/port_facts.py`) gives config and tensor facts. It does not give the code of the layers the port has to copy. This module reads
the modeling module of the model type, picks the classes the config names, and writes each class's constructor signature, its `forward`
method and the small same-file helpers that `forward` calls to `stages/2/reference_excerpts.md`, so the coder can write a first
`port_model.py` from one file.

How it reads. The module text is read with `safeio.read_bytes` (a size cap, no links in the last component, no FIFO block, regular
files only) and parsed with `ast`. Nothing is imported or run, so the module's top level never executes. A module that does not parse,
holds a null byte, is nested deeper than the parser allows, is not found, or is too large gives an empty result with a reason, and the
coder works without the file as it did before this module existed. `for_root` never raises: this step is advisory.

Bounds. A class is at most MAX_CLASS_BYTES of code (signature, `forward`, helpers), the file is at most MAX_TOTAL_BYTES, a helper is at
most HELPER_BYTES, and at most MAX_CLASSES classes are considered. Everything cut or left out is listed, in the file's header and in the
result's `left_out`. A class name longer than MAX_KEY_CHARS characters is not copied anywhere.

Selection. Which classes: the architecture class named in the config's `architectures`, the family prefix taken from it (or from the model
type), and within that prefix the decoder layer, the attention or mixer class of each layer type the config lists
(`linear_attention` picks a delta-net or mixer class, `full_attention` and `sliding_attention` pick the attention class), then the MLP or
mixture-of-experts block, the norm, the rotary embedding and the model and head classes, in that order. The order matters because the
total cap drops the last ones first."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from orchard import port_facts, safeio

FILE_NAME = "reference_excerpts.md"
SOURCE_MAX = 1 << 20              # a modeling module is tens to a few hundred kilobytes; this bound stops a hostile file
MAX_CLASS_BYTES = 8 * 1024        # code per class: signature, forward and helpers
MAX_TOTAL_BYTES = 60 * 1024       # the whole file, header included
HEADER_RESERVE = 4 * 1024         # room kept in the total for the file's header and its left-out list
SIGNATURE_BYTES = 1500            # a constructor signature longer than this is cut
HELPER_BYTES = 2 * 1024           # a same-file helper function larger than this is listed and not copied
MAX_HELPERS_PER_CLASS = 6
HELPER_DEPTH = 2                  # helpers of helpers: apply_rotary_pos_emb calls rotate_half
MAX_CLASSES = 24
MAX_LEFT_OUT_LISTED = 20
LEFT_OUT_CHARS = 220
SUMMARY_CLASSES = 12              # what the facts file names; the file itself holds the full list
SUMMARY_LEFT_OUT = 6
SUMMARY_CHARS = 100
MAX_LAYER_KINDS = 64

_FAMILY_SUFFIXES = ("ForConditionalGeneration", "ForCausalLM", "ForSequenceClassification", "ForTokenClassification",
                    "ForQuestionAnswering", "TextModel", "Model")
_NAME = re.compile(rf"[A-Za-z_][A-Za-z0-9_]{{0,{port_facts.MAX_KEY_CHARS - 1}}}")
_SKIP = re.compile(r"(PreTrainedModel|Config)$")
_NON_TEXT = re.compile(r"(Vision|Audio|Image|Video|Speech)")      # a text port does not need these classes unless the config has a section for them
_FIXED_RULES = (
    (re.compile(r"DecoderLayer$"), "decoder layer"),
)
_AFTER_LAYER_RULES = (
    (re.compile(r"(MLP|FeedForward)$"), "feed-forward block"),
    (re.compile(r"(SparseMoeBlock|MoeBlock|MoE|Experts|Router)$"), "mixture-of-experts part"),
    (re.compile(r"Norm$"), "normalisation"),
    (re.compile(r"RotaryEmbedding$"), "rotary embedding"),
    (re.compile(r"(TextModel|Model)$"), "model body"),
)
_KIND_RULES = (            # a layer type string (lower case) -> the class names it uses
    (re.compile(r"linear|delta|mamba|ssm|recurrent|conv"), re.compile(r"(DeltaNet|LinearAttention|Mamba|Mixer|Ssm|SSM|Recurrent|Conv)\w*$")),
    (re.compile(r"full|sliding|global|local|attention|chunk"), re.compile(r"Attention$")),
)


@dataclass
class Excerpts:
    """The result: `text` is the file's content ('' when nothing was written), `classes` the class names in it, `left_out` what was cut or
    skipped (one short sentence each), `reason` why `text` is empty (else '')."""
    text: str = ""
    classes: list = field(default_factory=list)
    left_out: list = field(default_factory=list)
    reason: str = ""

    def summary(self) -> dict:
        """What the facts file says about the excerpts: small and bounded whatever the file holds."""
        return {"file": FILE_NAME, "bytes": len(self.text.encode("utf-8")),
                "classes": [c for c in self.classes if _NAME.fullmatch(c)][:SUMMARY_CLASSES],
                "classes_total": len(self.classes),
                "left_out": [x[:SUMMARY_CHARS] for x in self.left_out[:SUMMARY_LEFT_OUT]],
                "left_out_total": len(self.left_out)}


def _camel(model_type: str) -> str:
    return "".join(p.capitalize() for p in model_type.split("_") if p)


def _strings(value, limit: int) -> list:
    """The short strings of a config list (at most `limit` distinct ones, in order); anything else gives []."""
    if not isinstance(value, list):
        return []
    out: list = []
    for x in value:
        if isinstance(x, str) and 0 < len(x) <= port_facts.MAX_STR and x not in out:
            out.append(x)
            if len(out) >= limit:
                break
    return out


def _config_names(config: dict) -> tuple:
    """(architecture names, layer type strings, model types) the config names, each a bounded list of short strings."""
    nested = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    archs = [a for a in _strings(config.get("architectures"), 8) if _NAME.fullmatch(a)]
    kinds = _strings(nested.get("layer_types"), MAX_LAYER_KINDS) or _strings(config.get("layer_types"), MAX_LAYER_KINDS)
    types = [m for m in (config.get("model_type"), nested.get("model_type")) if isinstance(m, str) and port_facts._SHOWN_MODEL_TYPE.fullmatch(m)]
    return archs, [k.lower() for k in kinds], types


def _prefix(archs: list, types: list, names) -> str | None:
    """The family prefix of the class names (`Qwen4Exp` for `Qwen4ExpForCausalLM`): from an architecture class that is in the module, else from
    the model type. None when no class name starts with either."""
    candidates = []
    for a in archs:
        for suffix in _FAMILY_SUFFIXES:
            if a.endswith(suffix) and len(a) > len(suffix):
                candidates.append(a[: -len(suffix)])
                break
    for t in types:
        base = t
        head, _, tail = t.rpartition("_")
        if head and tail in port_facts.SUB_CONFIG_SUFFIXES:
            base = head
        candidates.append(_camel(base))
    return next((c for c in candidates if any(n.startswith(c) for n in names)), None)


def _select_all(classes: dict, archs: list, kinds: list, types: list, with_other_sections: bool = False) -> list:
    """[(class name, why)] in priority order, every match. Classes for a vision or audio part are left out unless the config has such a
    section (`with_other_sections`). `build` applies MAX_CLASSES and lists the classes beyond it as left out."""
    prefix = _prefix(archs, types, classes)
    pool = [n for n in classes if _NAME.fullmatch(n) and not _SKIP.search(n) and (prefix is None or n.startswith(prefix))
            and (with_other_sections or not _NON_TEXT.search(n))]
    rules = list(_FIXED_RULES)
    seen_kind_rules: set = set()
    for kind in kinds or ["full_attention"]:
        for kind_re, class_re in _KIND_RULES:
            if kind_re.search(kind) and class_re.pattern not in seen_kind_rules:
                seen_kind_rules.add(class_re.pattern)
                rules.append((class_re, f"used by layer type {kind}"))
                break
    rules.extend(_AFTER_LAYER_RULES)
    out: list = []
    for rule, why in rules:
        for n in pool:
            if rule.search(n) and n not in (c for c, _ in out):
                out.append((n, why))
    for a in archs:                                    # the architecture class last: its forward is the generic wrapper
        if a in classes and a not in (c for c, _ in out):
            out.append((a, "architecture named in the config"))
    return out


def _select(classes: dict, archs: list, kinds: list, types: list, with_other_sections: bool = False) -> list:
    """The first MAX_CLASSES of `_select_all`."""
    return _select_all(classes, archs, kinds, types, with_other_sections)[:MAX_CLASSES]


def _top_level(tree: ast.Module) -> tuple:
    """(classes, functions) defined at the top level of the module, first definition of a name wins."""
    classes: dict = {}
    funcs: dict = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.setdefault(node.name, node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, node)
    return classes, funcs


def _start(node) -> int:
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _clean(text: str) -> str:
    """Source text for a fenced block: a triple backtick in the source cannot close the fence."""
    return text.replace("```", "` ` `")


def _lines(lines: list, first: int, last: int) -> str:
    return "\n".join(lines[first - 1:last])


def _method(cls: ast.ClassDef, name: str):
    return next((n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)


def _helpers(node, funcs: dict) -> list:
    """Names of same-file top-level functions that `node` calls or references, then those they reference, to HELPER_DEPTH levels."""
    found: list = []
    frontier = [node]
    for _ in range(HELPER_DEPTH):
        nxt = []
        for n in frontier:
            for sub in ast.walk(n):
                if isinstance(sub, ast.Name) and sub.id in funcs and sub.id not in found and funcs[sub.id] is not node:
                    found.append(sub.id)
                    nxt.append(funcs[sub.id])
        frontier = nxt
    return found


def _cut_lines(text: str, budget: int) -> tuple:
    """(the whole lines of `text` that fit in `budget` bytes, whether anything was cut)."""
    if len(text.encode("utf-8")) <= budget:
        return text, False
    kept, used = [], 0
    for line in text.split("\n"):
        cost = len(line.encode("utf-8")) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept), True


def _section(name: str, why: str, cls: ast.ClassDef, funcs: dict, lines: list, helpers_done: set, left_out: list) -> str:
    """One class's section: header, then a python block with the class line, the constructor signature, `forward` and its helpers,
    held to MAX_CLASS_BYTES of code. A cut is a comment inside the block and an entry in `left_out`."""
    budget = MAX_CLASS_BYTES
    parts: list = []
    head = _lines(lines, cls.lineno, cls.lineno)
    init, fwd = _method(cls, "__init__"), _method(cls, "forward")
    block = [head]
    if init is not None and init.body:
        sig = _lines(lines, _start(init), max(init.body[0].lineno - 1, init.lineno))
        sig, cut = _cut_lines(sig, SIGNATURE_BYTES)
        block.append(sig + ("\n        # [cut: signature continues]" if cut else ""))
        block.append("        ...")
        if cut:
            left_out.append(f"{name}.__init__: signature cut at {SIGNATURE_BYTES} bytes")
    parts.append("\n".join(block))
    budget -= len("\n".join(block).encode("utf-8"))
    used_helpers: list = []
    if fwd is not None:
        src = _lines(lines, _start(fwd), fwd.end_lineno)
        total = len(src.encode("utf-8"))
        src, cut = _cut_lines(src, max(budget, 0))
        if cut:
            shown = len(src.encode("utf-8"))
            src += f"\n        # [cut: forward continues; {shown} of {total} bytes shown]"
            left_out.append(f"{name}.forward: cut at {shown} of {total} bytes (per-class cap {MAX_CLASS_BYTES}); "
                            f"the whole method is at source lines {fwd.lineno}-{fwd.end_lineno} of the modeling module")
        parts.append(src)
        budget -= len(src.encode("utf-8"))
        every = _helpers(fwd, funcs)
        for h in every[MAX_HELPERS_PER_CLASS:]:
            if h not in helpers_done:
                left_out.append(f"helper {h} (used by {name}): not included, more than {MAX_HELPERS_PER_CLASS} helpers; "
                                f"source lines {funcs[h].lineno}-{funcs[h].end_lineno} of the modeling module")
        for h in every[:MAX_HELPERS_PER_CLASS]:
            if h in helpers_done:
                continue
            hsrc = _lines(lines, _start(funcs[h]), funcs[h].end_lineno)
            size = len(hsrc.encode("utf-8"))
            if size > HELPER_BYTES:
                left_out.append(f"helper {h} (used by {name}): {size} bytes, over the helper cap of {HELPER_BYTES}; "
                                f"source lines {funcs[h].lineno}-{funcs[h].end_lineno} of the modeling module")
                helpers_done.add(h)
            elif size + 2 > budget:
                left_out.append(f"helper {h} (used by {name}): {size} bytes, no room left in the per-class cap of {MAX_CLASS_BYTES}; "
                                f"source lines {funcs[h].lineno}-{funcs[h].end_lineno} of the modeling module")
            else:
                used_helpers.append(hsrc)
                helpers_done.add(h)
                budget -= size + 2
    else:
        left_out.append(f"{name}: no forward method of its own in this class")
    body = "\n\n".join(parts + used_helpers)
    return f"## {name}\n\nWhy: {why}. Source lines {cls.lineno}-{cls.end_lineno}.\n\n```python\n{_clean(body)}\n```\n"


def build(config, source) -> Excerpts:
    """The excerpts for a config from the text of the reference modeling module. Never raises; an empty result carries its reason."""
    if not isinstance(config, dict):
        return Excerpts(reason="the config is not an object")
    if not isinstance(source, str) or not source.strip():
        return Excerpts(reason="the reference modeling module was not found or could not be read")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return Excerpts(reason="the reference modeling module could not be parsed")
    archs, kinds, types = _config_names(config)
    classes, funcs = _top_level(tree)
    with_other = any(isinstance(config.get(k), dict) for k in ("vision_config", "audio_config"))
    every_match = _select_all(classes, archs, kinds, types, with_other)
    chosen = every_match[:MAX_CLASSES]
    if not chosen:
        return Excerpts(reason="no class of the reference modeling module matches the config's architecture or layer types")
    lines = source.splitlines()
    left_out: list = []
    for n, _ in every_match[MAX_CLASSES:]:
        src = classes[n]
        left_out.append(f"{n}: not included, more than {MAX_CLASSES} matching classes; source lines {src.lineno}-{src.end_lineno} of the modeling module")
    sections: list = []
    names: list = []
    helpers_done: set = set()
    total = 0
    room = MAX_TOTAL_BYTES - HEADER_RESERVE
    for name, why in chosen:
        mark = len(left_out)
        trial = set(helpers_done)            # a rejected section must not mark its helpers as shown
        sec = _section(name, why, classes[name], funcs, lines, trial, left_out)
        size = len(sec.encode("utf-8")) + 1
        if total + size > room:
            del left_out[mark:]
            left_out.append(f"{name}: not included, the total cap of {MAX_TOTAL_BYTES} bytes would be passed")
            continue
        helpers_done = trial
        total += size
        sections.append(sec)
        names.append(name)
    def render() -> str:
        """The file text with a header that describes the sections kept now and everything left out so far."""
        shown = left_out[:MAX_LEFT_OUT_LISTED]
        header = ["# Reference excerpts", "",
                  "Source text of the classes this model uses, copied from the installed `transformers` modeling module for its model type. "
                  "Each class shows its constructor signature, its `forward` method and the small helper functions `forward` calls. "
                  "Nothing here was run. Write your first `port_model.py` from this file before you read anything else.", ""]
        if shown:
            header += ["Left out or cut:"] + [f"- {_clean(x)[:LEFT_OUT_CHARS]}" for x in shown]
            if len(left_out) > len(shown):
                header.append(f"- and {len(left_out) - len(shown)} more")
            header.append("")
        return "\n".join(header) + "\n" + "\n".join(sections)

    text = render()
    while sections and len(text.encode("utf-8")) > MAX_TOTAL_BYTES:   # the reserve was too small for this header: cut sections from the end
        left_out.append(f"{names.pop()}: not included, the total cap of {MAX_TOTAL_BYTES} bytes would be passed")
        sections.pop()
        text = render()                                             # rebuilt, so the header lists the class just removed
    return Excerpts(text=text, classes=names, left_out=left_out)


def for_root(root, config) -> Excerpts:
    """The excerpts for `config` from the modeling module under the transformers directory `root` (None: no reference). Never raises."""
    try:
        if root is None or not isinstance(config, dict):
            return Excerpts(reason="there is no reference interpreter or config to read the modeling module from")
        _, _, types = _config_names(config)
        source = next((s for s in (port_facts.reference_source(root, m, kind="modeling") for m in types) if s), None)
        return build(config, source)
    except Exception as exc:         # advisory step: whatever goes wrong, the coder works without the file; the message may quote the source
        return Excerpts(reason=f"{exc.__class__.__name__}: an unexpected error while reading the reference modeling module")


def write(stage_dir, text: str) -> Path:
    path = Path(stage_dir) / FILE_NAME
    safeio.atomic_write(path, text, mode=0o644)
    return path
