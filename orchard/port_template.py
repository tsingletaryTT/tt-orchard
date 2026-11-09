# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A generated first port for stage 2 of a full port (issue 677): the standard blocks written, the new blocks left as marked stubs.

Why. Stage 2 asks the coder for the whole port from a skeleton whose four functions raise NotImplementedError. In the rehearsal runs
the coder's first version uploaded the weights and then computed on the host in torch (the Coder-Next arm of the cp4 audition failed the
device-ops gate on seven checks), or it spent its first fifteen turns reading before it wrote a line. This module writes the parts of a
decoder port that are the same for every model from data the harness already has, so every attempt starts from a program that makes real
`ttnn.linear` calls, and the coder's work is the blocks that are new.

Inputs (all read on the CPU, none from a chip):
- the checkpoint's config (`config.json` as a dict; the `text_config` section when there is one),
- the effective construction (the JSON object `effective_construction.py` prints: module classes, their resolved settings and children),
- the checkpoint's tensors (name, dtype and shape; `residency.read_headers` or a list of dicts),
- the verified TTNN cheat sheet (`op_cheats`): every call the generated code makes is an entry or appears in an entry's usage line,
- optionally stage 0's ops inventory (an aten op labelled `candidate` or `host` under a module makes that block a stub) and the
  reference excerpts (to see whether a norm class adds 1 to its weight).

What it writes (`generate`): a complete `load` (every float tensor uploaded, Linear weights transposed once and TILE bf16, embedding
tables ROW_MAJOR, norm weights as [1, 1, 1, dim], a mesh mapper when the device is a mesh of more than one chip), a `prefill` that runs
the embedding, RMSNorm, the gated MLP, ordinary attention (rotary in the HF layout and causal SDPA) and `lm_head` through TTNN calls, a
`decode_step` that re-runs the whole sequence (the idiom of the port that passed, no cache), and `close`. Every block it cannot write is a
named function whose body is a `# AGENT:` comment and a NotImplementedError that names the block and the reference class. The generator
does not know the data flow of a layer with children it does not recognise (a hyper-connection, a gated delta-net, a per-layer
embedding), so such a layer's function is a stub too, and it lists the generated helpers the coder can call from it.

The CPU check (`check_port`) runs before any board lease: some file under the port directory must make a reachable ttnn matmul-family
call (the same static analysis as the port check, `orchard/ttnn_calls.py`), the files must parse, and no `# AGENT:` stub may remain at the
end of the stage. `sheet_problems` lists every `ttnn.<name>` a source uses that the verified sheet does not cover.

This module is not on the supervisor's path: the supervisor still copies the plain skeleton. The sample generated port for the tiny Qwen4
rehearsal model is in docs/run-logs/2026-11-10-port-generator-sample.md.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from orchard import op_cheats, safeio, ttnn_calls

STUB_MARK = "# AGENT:"
GENERATED_MARK = "# generated-by: orchard/port_template.py"
SOURCE_MAX = 2 * 1024 * 1024            # one port source file is read to this size (the port check's own bound)
MAX_PORT_FILES = 64                     # files under stages/2/port/ the check reads; a directory with more is refused
INPUT_MAX = 16 * 1024 * 1024            # a construction, inventory, config or excerpts file
MAX_TENSORS = 1_000_000                 # the header reader's own bound
MAX_LAYERS = 10_000
MAX_SHAPE_DIM = 2 ** 40
STANDARD_LAYER = frozenset({"input_layernorm", "post_attention_layernorm", "self_attn", "mlp"})
MODEL_CHILDREN = frozenset({"embed_tokens", "layers", "norm", "rotary_emb"})   # the text model's children a standard flow has
MLP_CORE = frozenset({"gate_proj", "up_proj", "down_proj"})
ATTN_CORE = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
ATTN_ALL = ATTN_CORE | {"q_norm", "k_norm"}
FLAGGED_STATUSES = ("candidate", "host")     # an aten op with one of these labels under a module makes the block a stub
FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_NUMBER = re.compile(r"(?<=\.)\d{1,6}(?=\.|$)")
_FIRST_NUMBER = re.compile(r"^(?P<head>(?:[A-Za-z_][A-Za-z0-9_]*\.)+)(?P<num>\d{1,6})(?P<tail>\.[A-Za-z_].*)$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLUS_ONE = re.compile(r"\(\s*1(?:\.0)?\s*\+\s*self\.weight")
_SAME_WAY = re.compile(r"\b([a-z_]+(?:, [a-z_]+)*) and ([a-z_]+) are called the same way")


class PlanError(Exception):
    """An input the generator cannot plan from. The message names the input and never quotes its content."""


# ---- inputs ---------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple


def as_tensors(items) -> list[Tensor]:
    """`items` (dicts with name, dtype and shape, or objects with those attributes, such as residency.Tensor) as Tensor records.
    A name that is not a dotted identifier chain, a dtype that is not a short upper-case word or a shape with a bad dimension is refused."""
    if not isinstance(items, (list, tuple)) or len(items) > MAX_TENSORS:
        raise PlanError("tensors must be a list of at most a million records")
    out = []
    for item in items:
        get = item.get if isinstance(item, dict) else (lambda k, _i=item: getattr(_i, k, None))
        name, dtype, shape = get("name"), get("dtype"), get("shape")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*", name) or len(name) > 200:
            raise PlanError("a tensor name is not a dotted identifier chain of at most 200 characters")
        if not isinstance(dtype, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", dtype):
            raise PlanError(f"tensor {name}: the dtype is not a short upper-case word")
        if not isinstance(shape, (list, tuple)) or len(shape) > 8 or not all(type(d) is int and 0 <= d <= MAX_SHAPE_DIM for d in shape):
            raise PlanError(f"tensor {name}: the shape is not a list of at most 8 whole numbers")
        out.append(Tensor(name, dtype, tuple(shape)))
    return out


def text_section(config) -> dict:
    """The config section that describes the text model: `text_config` when there is one, else the config itself."""
    if not isinstance(config, dict):
        raise PlanError("the config is not an object")
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _whole(section: dict, key: str, default=None):
    v = section.get(key, default)
    return v if type(v) is int and 0 < v <= MAX_SHAPE_DIM else default


def _number(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf") else None


# ---- the construction: which class is at which module path -------------------------------------------------------------------------------

class Construction:
    """What `effective_construction.describe` said: for a module path pattern (`model.layers.{i}.mlp`), the class and its settings.

    A class's variants list up to three example paths each, and a child map names the class of each named child, so a path that no
    variant lists is resolved through its parent. A path that several variants of one class list (every decoder layer is `model.layers.{i}`)
    is resolved by the children the checkpoint shows for it (`resolve(path, observed=...)`)."""

    def __init__(self, data):
        if not isinstance(data, dict) or not isinstance(data.get("classes"), list):
            raise PlanError("the construction is not the object effective_construction.py prints (no `classes` list)")
        self.model_class = data.get("model_class") if isinstance(data.get("model_class"), str) else "<model>"
        self.classes: dict[str, list] = {}           # class name -> its variants (attrs dicts)
        self.by_path: dict[str, list] = {}           # path pattern -> [(class, attrs)]
        for c in data["classes"]:
            if not isinstance(c, dict) or not isinstance(c.get("class"), str) or not isinstance(c.get("variants"), list):
                continue
            name = c["class"]
            for v in c["variants"]:
                attrs = v.get("attrs") if isinstance(v, dict) and isinstance(v.get("attrs"), dict) else {}
                self.classes.setdefault(name, []).append(attrs)
                for p in (v.get("paths") if isinstance(v, dict) and isinstance(v.get("paths"), list) else []):
                    if isinstance(p, str):
                        self.by_path.setdefault(p, []).append((name, attrs))

    @staticmethod
    def children(attrs: dict) -> dict:
        kids = attrs.get("children")
        return {k: v for k, v in kids.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(kids, dict) else {}

    def resolve(self, path: str, observed: set | None = None) -> tuple[str | None, dict]:
        """(class name, attrs) at the module path pattern `path`, or (None, {}) when the construction does not say. With several
        candidates, `observed` (the child names the checkpoint's tensors show under the path) picks the variant whose children fit best."""
        found = list(self.by_path.get(path, []))
        if not found:
            parent, _, last = path.rpartition(".")
            cls, attrs = self.resolve(parent) if parent else (self.by_path.get("<model>") or [(None, {})])[0]   # a top-level name is a child of the root module
            kid = self.children(attrs).get(last) if cls else None
            if kid is None:
                return (None, {})
            found = [(kid, a) for a in self.classes.get(kid, [{}])]
        if len(found) == 1 or observed is None:
            return found[0]
        def fit(item):
            kids = set(self.children(item[1]))
            return (-len(kids & observed), len(kids - observed) + len(observed - kids))
        return sorted(found, key=fit)[0]


# ---- the plan --------------------------------------------------------------------------------------------------------------------------

@dataclass
class Stub:
    function: str          # the generated function's name
    path: str              # module path pattern
    cls: str               # reference class
    reason: str


@dataclass
class Layer:
    key: str                               # variant letter
    indices: list                          # layer numbers with this variant
    children: dict                         # child name -> class
    cls: str
    standard: bool
    reason: str = ""                       # why the layer is a stub
    blocks: dict = field(default_factory=dict)      # child name -> ("mlp" | "attention" | "rms_norm" | "stub", details)


@dataclass
class Plan:
    model_class: str
    text_path: str                         # "model"
    layers_path: str                       # "model.layers"
    n_layers: int
    hidden: int
    vocab: int
    pad_id: int
    config: dict                           # the fields the generated file carries
    embed_path: str | None
    final_norm: tuple | None               # (path, eps)
    lm_head_path: str
    tied: bool
    prelude_reason: str | None             # a stub between the embedding and the first layer, when the model has children the flow does not know
    epilogue: list                         # (child, class) of the text model's extra children, or [] when a final norm stands there
    layers: list                           # Layer, in key order
    placements: dict                       # tensor name pattern -> role
    norm_eps: dict                         # module path pattern -> eps
    plus_one_norms: set                    # norm classes whose forward multiplies by (1 + weight)
    attention: dict | None                 # settings of the generated attention, when one is generated
    stubs: list                            # Stub
    notes: list

    def summary(self) -> dict:
        """What the ledger or a run log records: the generated blocks and the stubs, without any file content."""
        return {"model_class": self.model_class, "layers": self.n_layers,
                "generated": sorted({"embedding"} | {b[0] for l in self.layers for b in l.blocks.values() if b[0] != "stub"}
                                    | ({"final_norm"} if self.final_norm else set()) | {"lm_head"}
                                    | ({"layer_flow"} if any(l.standard for l in self.layers) else set())),
                "layer_variants": [{"key": l.key, "indices": l.indices, "standard": l.standard, "reason": l.reason} for l in self.layers],
                "stubs": [{"function": s.function, "path": s.path, "class": s.cls, "reason": s.reason} for s in self.stubs],
                "notes": list(self.notes)}


def _flagged_modules(inventory) -> dict:
    """{module path pattern: "aten.op (status)"} for every module an op labelled candidate or host lists in stage 0's inventory.
    The inventory lists at most a few modules per op, so a module that is not flagged may still run such an op; the plan says so."""
    out: dict = {}
    ops = inventory.get("ops") if isinstance(inventory, dict) else None
    for op in ops if isinstance(ops, list) else []:
        if not isinstance(op, dict) or op.get("status") not in FLAGGED_STATUSES:
            continue
        name = op.get("op") if isinstance(op.get("op"), str) else "?"
        for m in op.get("modules") if isinstance(op.get("modules"), list) else []:
            if isinstance(m, str) and len(m) <= 200:
                out.setdefault(_NUMBER.sub("{i}", m), f"{name[:80]} ({op['status']})")
    return out


def _flag_for(flagged: dict, path: str) -> str | None:
    """The inventory flag that touches the block at `path`: an op under the path itself or under one of its descendants."""
    for m, why in flagged.items():
        if m == path or m.startswith(path + "."):
            return f"stage 0 labelled {why} under {m}"
    return None


def _plus_one_classes(excerpts, norm_classes) -> set:
    """The norm classes whose excerpt multiplies the normalised value by (1 + self.weight)."""
    out = set()
    if not isinstance(excerpts, str):
        return out
    for cls in norm_classes:
        start = excerpts.find(f"class {cls}(")
        if start < 0:
            continue
        end = excerpts.find("\nclass ", start + 1)
        if _PLUS_ONE.search(excerpts[start: end if end > 0 else len(excerpts)]):
            out.add(cls)
    return out


def _is_rms_norm(cls, attrs) -> bool:
    return isinstance(cls, str) and "RMSNorm" in cls and "Gated" not in cls and attrs.get("group_size") is None


def _eps(attrs: dict, default: float) -> float:
    for key in ("eps", "variance_epsilon", "epsilon", "layer_norm_eps", "norm_eps"):
        v = _number(attrs.get(key))
        if v is not None and 0 < v < 1:
            return float(v)
    return default


def plan(*, config, construction, tensors, inventory=None, excerpts=None) -> Plan:
    """The plan of the generated port. Raises PlanError when the inputs do not describe a decoder with an embedding, numbered layers and a head."""
    section = text_section(config)
    con = construction if isinstance(construction, Construction) else Construction(construction)
    ts = as_tensors(tensors)
    names = {t.name for t in ts}
    by_pattern: dict[str, Tensor] = {}
    for t in ts:
        by_pattern.setdefault(_NUMBER.sub("{i}", t.name), t)
    # The layer stack: the dominant `prefix.N.rest` pattern among the tensor names (the first number in a name is the layer number).
    heads: dict = {}
    for t in ts:
        m = _FIRST_NUMBER.match(t.name)
        if m:
            heads.setdefault(m.group("head"), set()).add(int(m.group("num")))
    if not heads:
        raise PlanError("no tensor name holds a layer number, so there is no layer stack to plan")
    layers_head = max(heads, key=lambda h: (sum(1 for t in ts if t.name.startswith(h)), -len(h)))
    layers_path = layers_head.rstrip(".")
    text_path = layers_path.rpartition(".")[0]
    indices = sorted(heads[layers_head])
    n_layers = _whole(section, "num_hidden_layers") or (len(section["layer_types"]) if isinstance(section.get("layer_types"), list) else None) or len(indices)
    if n_layers > MAX_LAYERS or indices != list(range(len(indices))):
        raise PlanError("the layer numbers in the tensor names are not 0..N-1")
    if len(indices) < n_layers:
        raise PlanError(f"the config says {n_layers} layers and the checkpoint holds tensors for {len(indices)}")
    # The embedding, the head and the final norm.
    text_cls, text_attrs = con.resolve(text_path)
    text_kids = con.children(text_attrs)
    embed_path = next((f"{text_path}.{k}" for k, v in sorted(text_kids.items()) if v == "Embedding" and f"{text_path}.{k}.weight" in names), None)
    if embed_path is None:
        embed_path = next((n.rsplit(".", 1)[0] for n in sorted(names) if n.startswith(text_path + ".") and "embed" in n and n.count(".") == 2), None)
    if embed_path is None:
        raise PlanError("no embedding table was found among the text model's direct children")
    embed_shape = by_pattern[embed_path + ".weight"].shape
    if len(embed_shape) != 2:
        raise PlanError("the embedding table is not 2-D")
    vocab, hidden = embed_shape
    hidden = _whole(section, "hidden_size") or hidden
    tied = "lm_head.weight" not in names
    if tied and section.get("tie_word_embeddings") is False and config.get("tie_word_embeddings") is False:
        raise PlanError("the checkpoint has no lm_head.weight and the config says the embeddings are not tied")
    lm_head_path = "lm_head"
    norm_classes = {c for c in con.classes if _is_rms_norm(c, {})}
    plus_one = _plus_one_classes(excerpts, norm_classes)
    flagged = _flagged_modules(inventory)
    notes: list = []
    if excerpts is None and norm_classes:
        notes.append("no reference excerpts were given, so norm weights are used as stored (a class that multiplies by 1 + weight needs the excerpts)")
    if inventory is None:
        notes.append("no ops inventory was given, so no block was set aside for an op labelled candidate or host")
    else:
        notes.append("the inventory lists at most a few modules per op, so a block it did not flag may still run an op labelled candidate or host")
    stubs: list = []
    placements: dict = {}
    norm_eps: dict = {}
    default_eps = _number(section.get("rms_norm_eps")) or 1e-6

    def place(path: str, cls: str | None, attrs: dict) -> None:
        """Placement roles for the tensors of the module at `path`, by its class."""
        w = by_pattern.get(path + ".weight")
        if cls == "Linear" and w is not None and len(w.shape) == 2:
            placements[path + ".weight"] = "linear_weight"
            if path + ".bias" in by_pattern:
                placements[path + ".bias"] = "linear_bias"
        elif cls == "Embedding" and w is not None and len(w.shape) == 2:
            placements[path + ".weight"] = "embedding"
        elif _is_rms_norm(cls, attrs) and w is not None and len(w.shape) == 1:
            placements[path + ".weight"] = "norm_plus_one" if cls in plus_one else "norm"
            norm_eps[path] = _eps(attrs, default_eps)

    def module_paths():
        seen = set()
        for p in sorted(by_pattern):
            path = p.rsplit(".", 1)[0]
            if path not in seen:
                seen.add(path)
                yield path

    for path in module_paths():
        cls, attrs = con.resolve(path)
        place(path, cls, attrs)
    for p, t in by_pattern.items():
        placements.setdefault(p, "tile" if t.dtype in FLOAT_DTYPES else "host")
    if tied:
        placements["lm_head.weight"] = "linear_weight"      # uploaded from the embedding table in load
    # The final norm or the model-level stubs.
    final_norm = None
    extra = {k: v for k, v in text_kids.items() if k not in MODEL_CHILDREN}
    norm_path = f"{text_path}.norm"
    if norm_path + ".weight" in by_pattern and placements.get(norm_path + ".weight", "").startswith("norm"):
        final_norm = (norm_path, norm_eps[norm_path])
    elif norm_path + ".weight" in by_pattern:
        cls, _ = con.resolve(norm_path)
        extra["norm"] = cls or "?"
    prelude_reason = None
    if extra:
        prelude_reason = (f"{text_cls or 'the text model'} has children the generated flow does not know: "
                          + ", ".join(f"{k} ({v})" for k, v in sorted(extra.items())))
    if text_path in flagged:          # an op labelled candidate or host in the text model's own forward, between the blocks; a flag under a child goes to that child's block
        flag = f"stage 0 labelled {flagged[text_path]} in {text_path} itself"
        prelude_reason = (prelude_reason + "; " if prelude_reason else "") + flag
    epilogue = sorted((k, v) for k, v in extra.items())
    # The layers: one variant per set of children the tensors show.
    per_layer: dict = {}
    for t in ts:
        m = _FIRST_NUMBER.match(t.name)
        if m and m.group("head") == layers_head:
            per_layer.setdefault(int(m.group("num")), set()).add(m.group("tail").lstrip(".").split(".", 1)[0])
    variants: dict = {}
    for i in range(n_layers):
        variants.setdefault(tuple(sorted(per_layer.get(i, ()))), []).append(i)
    layers: list = []
    attention_settings = None
    for n, (kids, idx) in enumerate(sorted(variants.items(), key=lambda kv: kv[1][0])):
        key = chr(ord("A") + n) if n < 26 else f"V{n}"
        lp = f"{layers_path}.{{i}}"
        cls, attrs = con.resolve(lp, observed=set(kids))
        kid_classes = con.children(attrs)
        children = {k: kid_classes.get(k, "?") for k in kids}
        layer = Layer(key, idx, children, cls or "?", standard=False)
        for kid in kids:
            kp = f"{lp}.{kid}"
            kcls, kattrs = con.resolve(kp)
            kcls = kcls or children[kid]
            kkids = con.children(kattrs)
            flag = _flag_for(flagged, kp)
            if kcls == "Linear":
                layer.blocks[kid] = ("linear", {})
            elif _is_rms_norm(kcls, kattrs) and kp + ".weight" in by_pattern and len(by_pattern[kp + ".weight"].shape) == 1 and not flag:
                layer.blocks[kid] = ("rms_norm", {"eps": norm_eps.get(kp, default_eps)})
            elif MLP_CORE <= set(kkids) and set(kkids) - MLP_CORE <= {"act_fn"} and not flag \
                    and ("SiLU" in kkids.get("act_fn", "") or (kkids.get("act_fn") is None and section.get("hidden_act") == "silu")) \
                    and all(f"{kp}.{p}.weight" in by_pattern for p in MLP_CORE):
                layer.blocks[kid] = ("mlp", {})
            elif ATTN_CORE <= set(kkids) and set(kkids) <= ATTN_ALL and not flag:
                settings, why = _attention_plan(section, kp, kcls, kattrs, kkids, by_pattern, con, norm_eps, default_eps, hidden)
                if settings is None:
                    layer.blocks[kid] = ("stub", {"cls": kcls, "reason": why})
                else:
                    layer.blocks[kid] = ("attention", settings)
                    attention_settings = settings
            else:
                why = flag or ("a " + kcls + " block the generator does not write" if kcls != "?" else "a block the construction does not describe")
                if MLP_CORE <= set(kkids) and not flag:
                    why = f"{kcls}: its activation is not silu (the sheet has silu only)"
                elif ATTN_CORE <= set(kkids) and not flag:
                    beyond = ", ".join(f"{k} ({v})" for k, v in sorted(kkids.items()) if k not in ATTN_ALL)
                    why = f"{kcls}: attention with children beyond q, k, v, o and their norms: {beyond}"
                layer.blocks[kid] = ("stub", {"cls": kcls, "reason": why})
        own_flag = f"stage 0 labelled {flagged[lp]} in the layer itself" if lp in flagged else None
        standard = set(kids) == STANDARD_LAYER and layer.blocks.get("self_attn", ("",))[0] == "attention" \
            and layer.blocks.get("mlp", ("",))[0] == "mlp" and layer.blocks["input_layernorm"][0] == "rms_norm" \
            and layer.blocks["post_attention_layernorm"][0] == "rms_norm" and not own_flag
        layer.standard = standard
        if not standard:
            parts = [f"{k} ({children[k]})" for k, v in sorted(layer.blocks.items()) if v[0] == "stub"]
            layer.reason = (own_flag or (f"{cls or 'the layer class'}'s children are not the standard pre-norm set (input_layernorm, self_attn, "
                            "post_attention_layernorm, mlp)" if set(kids) != STANDARD_LAYER else "a child block is a stub")) \
                + ("; stubs: " + ", ".join(parts) if parts else "")
            stubs.append(Stub(f"_layer_{key}", lp, cls or "?", layer.reason))
            for k, v in sorted(layer.blocks.items()):
                if v[0] == "stub":
                    stubs.append(Stub(f"_{key}_{k}", f"{lp}.{k}", v[1]["cls"], v[1]["reason"]))
        layers.append(layer)
    if prelude_reason:
        stubs.insert(0, Stub("_model_prelude", text_path, text_cls or "?", prelude_reason))
    for k, v in epilogue:
        stubs.append(Stub(f"_model_{k}", f"{text_path}.{k}", v, f"a {v} block at the end of the text model that the generator does not write"))
    pad = section.get("pad_token_id", config.get("pad_token_id"))
    small = {k: v for k, v in section.items() if isinstance(v, (int, float, str, bool)) and not isinstance(v, bool) or isinstance(v, bool)}
    small = {k: v for k, v in small.items() if isinstance(k, str) and _IDENT.fullmatch(k) and (not isinstance(v, str) or len(v) <= 80)}
    return Plan(model_class=con.model_class, text_path=text_path, layers_path=layers_path, n_layers=n_layers, hidden=hidden, vocab=vocab,
                pad_id=pad if type(pad) is int and 0 <= pad < vocab else 0, config=small, embed_path=embed_path, final_norm=final_norm,
                lm_head_path=lm_head_path, tied=tied, prelude_reason=prelude_reason, epilogue=epilogue, layers=layers, placements=placements,
                norm_eps=norm_eps, plus_one_norms=plus_one, attention=attention_settings, stubs=stubs, notes=notes)


def _attention_plan(section, kp, kcls, kattrs, kkids, by_pattern, con, norm_eps, default_eps, hidden):
    """(settings, None) for an attention block the generator writes (q, k, v and o projections, optional per-head RMSNorm on q and k,
    rotary in the HF layout over the whole head, causal SDPA), else (None, why)."""
    heads = _whole(section, "num_attention_heads")
    kv = _whole(section, "num_key_value_heads") or heads
    head_dim = _whole(section, "head_dim") or (hidden // heads if heads else None)
    if not heads or not head_dim:
        return None, f"{kcls}: the config does not give num_attention_heads and head_dim"
    q = by_pattern.get(kp + ".q_proj.weight")
    k = by_pattern.get(kp + ".k_proj.weight")
    v = by_pattern.get(kp + ".v_proj.weight")
    o = by_pattern.get(kp + ".o_proj.weight")
    if not all(t is not None and len(t.shape) == 2 for t in (q, k, v, o)):
        return None, f"{kcls}: a projection weight is missing or not 2-D"
    if q.shape != (heads * head_dim, hidden) or k.shape != (kv * head_dim, hidden) or v.shape != (kv * head_dim, hidden) or o.shape != (hidden, heads * head_dim):
        return None, (f"{kcls}: the projection shapes are not [heads * head_dim, hidden] for q, [kv_heads * head_dim, hidden] for k and v and "
                      f"[hidden, heads * head_dim] for o (q_proj is {list(q.shape)}), so the head split is not the plain one")
    factor = _number(section.get("partial_rotary_factor"))
    if factor is not None and factor != 1:
        return None, f"{kcls}: partial_rotary_factor is {factor}; the generated rotary rotates the whole head"
    rope = section.get("rope_parameters") if isinstance(section.get("rope_parameters"), dict) else section.get("rope_scaling")
    rope_type = rope.get("rope_type", rope.get("type", "default")) if isinstance(rope, dict) else "default"
    if rope_type != "default":
        return None, f"{kcls}: rope_type is not default"
    theta = _number((rope or {}).get("rope_theta")) if isinstance(rope, dict) else None
    theta = theta or _number(section.get("rope_theta")) or 10000.0
    for n in ("q_norm", "k_norm"):
        if n in kkids:
            ncls, nattrs = con.resolve(f"{kp}.{n}")
            w = by_pattern.get(f"{kp}.{n}.weight")
            if not _is_rms_norm(ncls, nattrs) or w is None or w.shape != (head_dim,):
                return None, f"{kcls}: {n} is not a plain RMSNorm over head_dim"
    scaling = _number(kattrs.get("scaling"))
    return {"heads": heads, "kv_heads": kv, "head_dim": head_dim, "theta": float(theta), "scale": float(scaling) if scaling else float(head_dim) ** -0.5,
            "q_norm": "q_norm" in kkids, "k_norm": "k_norm" in kkids,
            "norm_eps": norm_eps.get(f"{kp}.q_norm", default_eps)}, None


# ---- rendering -------------------------------------------------------------------------------------------------------------------------

def _lit(x) -> str:
    return repr(x)


def _stub_function(name: str, args: str, stub: Stub, extra: list | None = None) -> list[str]:
    """A function whose body is the stub marker and a NotImplementedError that names the block and its reference class."""
    lines = [f"def {name}({args}):",
             f'    """{stub.cls} at {stub.path}. Not generated: {stub.reason}."""',
             f"    {STUB_MARK} write {stub.cls} ({stub.path}) here from its forward in stages/2/reference_excerpts.md, then remove this marker."]
    for e in extra or []:
        lines.append(f"    # {e}")
    lines.append(f"    raise NotImplementedError({_lit(f'{name}: {stub.cls} at {stub.path} is not written yet')})")
    return lines + ["", ""]


def render(p: Plan) -> str:
    """The text of the generated port_model.py."""
    L: list[str] = []
    stub_lines = "\n".join(f"- {s.function}: {s.cls} at {s.path}: {s.reason}" for s in p.stubs) or "- none"
    L += ["# SPDX-License-Identifier: Apache-2.0", "# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.", GENERATED_MARK,
          f'"""Generated first port of {p.model_class} for stage 2 of a full port (orchard/port_template.py).',
          "",
          "The standard blocks are written through TTNN calls from the verified cheat sheet (stages/2/cheat_sheet.md): weight upload in",
          "load, the embedding, RMSNorm, the gated MLP, ordinary attention and lm_head. Each function whose body starts with `# AGENT:` is a",
          "block the generator does not write: fill it in from the reference class named in stages/2/reference_excerpts.md and remove the",
          "marker. The CPU check refuses a port that keeps a marker.",
          "", "Stubs:", stub_lines, "",
          "Notes:", *[f"- {n}" for n in p.notes], '"""',
          "from __future__ import annotations", "", "import glob", "import os", "import re", "",
          "import torch", "import ttnn", "from safetensors import safe_open", "",
          f"CONFIG = {_lit(p.config)}",
          f"N_LAYERS = {p.n_layers}", f"HIDDEN = {p.hidden}", f"VOCAB = {p.vocab}", f"PAD_ID = {p.pad_id}",
          "PAD_TO = 32                       # the sequence is padded to a tile multiple; the logits are read at the last real position",
          f"LAYER_VARIANT = {_lit(dict(sorted((i, l.key) for l in p.layers for i in l.indices)))}",
          "# Every tensor pattern of the checkpoint and how load() places it: linear_weight (transposed once, TILE bf16), linear_bias, embedding",
          "# (ROW_MAJOR bf16 table), norm and norm_plus_one ([1, 1, 1, dim] TILE bf16), tile (any other float tensor), host (integer tensors stay in torch).",
          "PLACEMENT = {"]
    for k, v in sorted(p.placements.items()):
        L.append(f"    {_lit(k)}: {_lit(v)},")
    L += ["}", f"NORM_EPS = {_lit(dict(sorted(p.norm_eps.items())))}", "_NUMBER = re.compile(r\"(?<=\\.)\\d+(?=\\.|$)\")", "", ""]
    L += ['def read_tensors(model_dir):',
          '    """Every tensor of the checkpoint as {name: torch tensor}, from one file or from shards."""',
          "    out = {}",
          '    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):',
          '        with safe_open(path, framework="pt") as f:',
          "            for name in f.keys():",
          "                out[name] = f.get_tensor(name)",
          "    return out", "", "",
          "def _pattern(name):", '    return _NUMBER.sub("{i}", name)', "", "",
          "def _n_devices(mesh_device):",
          '    """How many chips the device the check opened holds (1 for ttnn.open_device, N for a mesh)."""',
          "    try:",
          "        return int(mesh_device.get_num_devices())",
          "    except Exception:          # noqa: BLE001 - a device object without the method is one chip",
          "        return 1", "", "",
          "def _upload(model, t, layout):",
          '    """A host tensor onto the device as bf16 in `layout`; replicated on every chip of a mesh."""',
          '    if model["n_devices"] > 1:',
          '        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=model["mesh_device"], memory_config=ttnn.DRAM_MEMORY_CONFIG,',
          '                               mesh_mapper=ttnn.ReplicateTensorToMesh(model["mesh_device"]))',
          '    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=model["mesh_device"], memory_config=ttnn.DRAM_MEMORY_CONFIG)', "", "",
          "def _upload_ids(model, ids):",
          '    """Token ids as a uint32 ROW_MAJOR [1, S] tensor on the device."""',
          "    ids_t = torch.tensor([ids], dtype=torch.int32)",
          '    if model["n_devices"] > 1:',
          '        return ttnn.from_torch(ids_t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model["mesh_device"],',
          '                               mesh_mapper=ttnn.ReplicateTensorToMesh(model["mesh_device"]))',
          '    return ttnn.from_torch(ids_t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model["mesh_device"])', "", "",
          "def _to_host(model, t):",
          '    """A device tensor back to torch. On a mesh the composer joins one copy per chip along dim 0; the first copy is kept."""',
          '    if model["n_devices"] > 1:',
          '        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(model["mesh_device"], dim=0))[:1]',
          "    return ttnn.to_torch(t)", "", "",
          "def _linear(model, prefix, x):",
          '    """x [1, 1, S, in] @ W (+ bias) for the Linear module at `prefix`; W was uploaded as [in, out] in load."""',
          '    w = model["dev"][prefix + ".weight"]',
          '    b = model["dev"].get(prefix + ".bias")',
          "    if b is not None:",
          '        return ttnn.linear(x, w, bias=b, dtype=ttnn.bfloat16, compute_kernel_config=model["ck"], memory_config=ttnn.DRAM_MEMORY_CONFIG)',
          '    return ttnn.linear(x, w, dtype=ttnn.bfloat16, compute_kernel_config=model["ck"], memory_config=ttnn.DRAM_MEMORY_CONFIG)', "", "",
          "def _rms_norm(model, prefix, x):",
          '    """RMSNorm over the last dim with the weight at `prefix` (uploaded as [1, 1, 1, dim]; a class that adds 1 to its weight had it added at load)."""',
          '    return ttnn.rms_norm(x, epsilon=NORM_EPS[_pattern(prefix)], weight=model["dev"][prefix + ".weight"])', "", "",
          "def _embedding(model, prefix, ids):",
          '    """The embedding table at `prefix` looked up for `ids` (a list of ints), as [1, 1, S, HIDDEN] TILE bf16."""',
          "    ids_dev = _upload_ids(model, ids)",
          '    x = ttnn.embedding(ids_dev, model["dev"][prefix + ".weight"], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)',
          "    return ttnn.reshape(x, (1, 1, len(ids), HIDDEN))", "", "",
          "def _mlp(model, prefix, x):",
          '    """down(silu(gate(x)) * up(x)) for the gated MLP at `prefix`."""',
          '    gate = _linear(model, prefix + ".gate_proj", x)',
          '    up = _linear(model, prefix + ".up_proj", x)',
          "    h = ttnn.multiply(ttnn.silu(gate), up)",
          '    y = _linear(model, prefix + ".down_proj", h)',
          "    for t in (gate, up, h):",
          "        ttnn.deallocate(t)",
          "    return y", "", ""]
    if p.attention:
        a = p.attention
        L += [f"ATTENTION = {_lit(a)}", "", "",
              "def _rope_tables(model, n):",
              '    """cos and sin for positions 0..n-1 in the HF layout (freqs repeated twice), [1, 1, n, head_dim] TILE bf16 on the device."""',
              '    hd = ATTENTION["head_dim"]',
              '    inv_freq = 1.0 / (ATTENTION["theta"] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))',
              "    freqs = torch.arange(n, dtype=torch.float32)[:, None] * inv_freq[None, :]",
              "    emb = torch.cat([freqs, freqs], dim=-1)",
              "    cos = _upload(model, emb.cos().reshape(1, 1, n, hd).to(torch.bfloat16), ttnn.TILE_LAYOUT)",
              "    sin = _upload(model, emb.sin().reshape(1, 1, n, hd).to(torch.bfloat16), ttnn.TILE_LAYOUT)",
              "    return cos, sin", "", "",
              "def _attention(model, prefix, x, cos, sin):",
              '    """Ordinary causal attention for the block at `prefix`: q, k, v projections, the head split, per-head RMSNorm on q and k when the',
              '    reference has one, rotary in the HF layout, SDPA with grouped keys, the head join and the output projection."""',
              '    q = _linear(model, prefix + ".q_proj", x)',
              '    k = _linear(model, prefix + ".k_proj", x)',
              '    v = _linear(model, prefix + ".v_proj", x)',
              "    xqkv = ttnn.concat([q, k, v], dim=-1)",
              '    q, k, v = ttnn.experimental.nlp_create_qkv_heads(xqkv, num_heads=ATTENTION["heads"], num_kv_heads=ATTENTION["kv_heads"],',
              "                                                     transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)"]
        if a["q_norm"]:
            L.append('    q = ttnn.rms_norm(q, epsilon=ATTENTION["norm_eps"], weight=model["dev"][prefix + ".q_norm.weight"])')
        if a["k_norm"]:
            L.append('    k = ttnn.rms_norm(k, epsilon=ATTENTION["norm_eps"], weight=model["dev"][prefix + ".k_norm.weight"])')
        L += ["    q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)",
              "    k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)",
              '    o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=ATTENTION["scale"])',
              "    o = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)",
              '    return _linear(model, prefix + ".o_proj", o)', "", ""]
    stubs = {s.function: s for s in p.stubs}
    if p.prelude_reason:
        L += _stub_function("_model_prelude", "model, x, ids", stubs["_model_prelude"],
                            [f"The reference's forward may change the embedding before the first layer; x is [1, 1, S, HIDDEN] TILE bf16 from _embedding."])
    for k, v in p.epilogue:
        L += _stub_function(f"_model_{k}", "model, x", stubs[f"_model_{k}"])
    for layer in p.layers:
        lp = f"{p.layers_path}.{{i}}"
        if layer.standard:
            L += [f"def _layer_{layer.key}(model, i, x, ids, cos, sin):",
                  f'    """Layers {layer.indices}: {layer.cls}, pre-norm attention and MLP with residual adds."""',
                  f'    p = f"{p.layers_path}.{{i}}"',
                  '    h = _rms_norm(model, p + ".input_layernorm", x)',
                  '    x = ttnn.add(x, _attention(model, p + ".self_attn", h, cos, sin))',
                  '    h = _rms_norm(model, p + ".post_attention_layernorm", x)',
                  '    return ttnn.add(x, _mlp(model, p + ".mlp", h))', "", ""]
            continue
        for kid, (kind, details) in sorted(layer.blocks.items()):
            if kind == "stub":
                L += _stub_function(f"_{layer.key}_{kid}", "model, i, x, ids", stubs[f"_{layer.key}_{kid}"],
                                    [f"The Linear children of this block are uploaded by load() and can be called as _linear(model, f\"{lp}.{kid}.<name>\", x)."])
        parts = []
        for kid, (kind, details) in sorted(layer.blocks.items()):
            call = {"mlp": f'_mlp(model, p + ".{kid}", x)', "rms_norm": f'_rms_norm(model, p + ".{kid}", x)',
                    "attention": f'_attention(model, p + ".{kid}", x, cos, sin)', "linear": f'_linear(model, p + ".{kid}", x)',
                    "stub": f"_{layer.key}_{kid}(model, i, x, ids)"}[kind]
            parts.append(f"{kid} ({layer.children[kid]}): {call}")
        L += _stub_function(f"_layer_{layer.key}", "model, i, x, ids, cos, sin", stubs[f"_layer_{layer.key}"],
                            [f"Layers {layer.indices}. Parts, each a generated helper or a stub above; compose them as {layer.cls}.forward does:",
                             *[f"  {part}" for part in parts], f'p = f"{p.layers_path}.{{i}}"'])
    # load, prefill, decode_step, close
    L += ["def load(model_dir, mesh_device):",
          '    """Read the checkpoint and place every tensor on the device as PLACEMENT says; return the model object the other functions take."""',
          '    model = {"mesh_device": mesh_device, "n_devices": _n_devices(mesh_device), "dev": {}, "host": {}, "ids": []}',
          '    model["ck"] = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)',
          '    dev, host = model["dev"], model["host"]',
          "    tensors = read_tensors(model_dir)          # a local: the model object keeps no host copy of an uploaded weight",
          "    for name, t in tensors.items():",
          '        role = PLACEMENT.get(_pattern(name), "tile" if t.is_floating_point() else "host")',
          '        if role == "linear_weight":',
          "            dev[name] = _upload(model, t.to(torch.bfloat16).t().contiguous(), ttnn.TILE_LAYOUT)",
          '        elif role == "linear_bias" or role == "norm":',
          "            dev[name] = _upload(model, t.to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)",
          '        elif role == "norm_plus_one":',
          "            dev[name] = _upload(model, (t.float() + 1.0).to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)",
          '        elif role == "embedding":',
          "            dev[name] = _upload(model, t.to(torch.bfloat16).contiguous(), ttnn.ROW_MAJOR_LAYOUT)",
          '        elif role == "tile":',
          "            dev[name] = _upload(model, t.to(torch.bfloat16).reshape(1, 1, 1, -1) if t.dim() == 1 else t.to(torch.bfloat16).contiguous(), ttnn.TILE_LAYOUT)",
          "        else:",
          "            host[name] = t",
          f'    if "{p.lm_head_path}.weight" not in dev:         # tied embeddings: the head is the embedding table, uploaded once more as a Linear weight',
          f'        dev["{p.lm_head_path}.weight"] = _upload(model, tensors["{p.embed_path}.weight"].to(torch.bfloat16).t().contiguous(), ttnn.TILE_LAYOUT)',
          "    return model", "", "",
          "def _forward(model, ids):",
          '    """Logits for the token after the last of `ids`, as a float32 torch vector of VOCAB values."""',
          "    n = len(ids)",
          "    padded = list(ids) + [PAD_ID] * (-n % PAD_TO)",
          f'    x = _embedding(model, "{p.embed_path}", padded)']
    if p.attention:
        L.append("    cos, sin = _rope_tables(model, len(padded))")
    else:
        L.append("    cos = sin = None")
    if p.prelude_reason:
        L.append("    x = _model_prelude(model, x, padded)")
    L += ["    for i in range(N_LAYERS):",
          "        x = LAYERS[LAYER_VARIANT[i]](model, i, x, padded, cos, sin)"]
    if p.final_norm:
        L.append(f'    x = _rms_norm(model, "{p.final_norm[0]}", x)')
    for k, _ in p.epilogue:
        L.append(f"    x = _model_{k}(model, x)")
    L += [f'    logits = _linear(model, "{p.lm_head_path}", x)',
          "    out = _to_host(model, logits).float()",
          "    return out.reshape(-1, out.shape[-1])[n - 1, :VOCAB]", "", "",
          f"LAYERS = {{{', '.join(f'{_lit(l.key)}: _layer_{l.key}' for l in p.layers)}}}", "", "",
          "def prefill(model, ids):",
          '    """Start the request afresh: run the prompt and return the logits for the next token."""',
          '    model["ids"] = [int(t) for t in ids]',
          '    return _forward(model, model["ids"])', "", "",
          "def decode_step(model, token_id, position):",
          '    """Append `token_id` at `position` and run the whole sequence again (no cache: the same device calls as prefill)."""',
          '    model["ids"] = model["ids"][:position] + [int(token_id)]',
          '    return _forward(model, model["ids"])', "", "",
          "def close(model):",
          '    """Free every device tensor load took. The check closes the device itself."""',
          '    for t in list(model["dev"].values()):',
          "        try:",
          "            ttnn.deallocate(t)",
          "        except Exception:      # noqa: BLE001 - a failed free must not stop the run",
          "            pass",
          '    model["dev"].clear()',
          '    model["host"].clear()', ""]
    return "\n".join(L)


def generate(*, config, construction, tensors, inventory=None, excerpts=None) -> tuple[str, Plan]:
    """(the port_model.py text, its plan)."""
    p = plan(config=config, construction=construction, tensors=tensors, inventory=inventory, excerpts=excerpts)
    return render(p), p


# ---- the CPU check -----------------------------------------------------------------------------------------------------------------------

def stub_lines(source: str) -> list[tuple]:
    """(line number, text) of every comment line that starts with the stub marker. A string that holds the marker is not a stub."""
    out = []
    for n, line in enumerate(source.split("\n"), 1):
        if line.strip().startswith(STUB_MARK):
            out.append((n, line.strip()[:160]))
    return out


def check_sources(sources: dict) -> dict:
    """The CPU check over {file name: source text}: {"ok", "problems", "matmul_calls", "stubs": [(file, line, text)], "parse_errors"}.
    A port passes when some file makes a reachable ttnn matmul-family call, every file parses and no stub marker remains."""
    calls, errors, stubs = 0, [], []
    for name in sorted(sources):
        found = ttnn_calls.static_calls(sources[name])
        if found["error"]:
            errors.append(f"{name}: {found['error']}")
        calls += sum(v for k, v in found["calls"].items() if ttnn_calls.is_matmul(k) and ttnn_calls.is_compute(k))
        stubs += [(name, n, text) for n, text in stub_lines(sources[name])]
    problems = list(errors)
    if not sources:
        problems.append("the port directory holds no Python file")
    if sources and calls == 0:
        problems.append("no file makes a reachable ttnn matmul-family call (ttnn.linear, ttnn.matmul and the like), so the check would fail "
                        "before the device is touched")
    if stubs:
        problems.append(f"{len(stubs)} {STUB_MARK} stub(s) remain: " + "; ".join(f"{f}:{n}" for f, n, _ in stubs[:12]) + (" ..." if len(stubs) > 12 else ""))
    return {"ok": not problems, "problems": problems, "matmul_calls": calls, "stubs": stubs, "parse_errors": errors}


def port_sources(port_dir, *, suffix: str = ".py") -> dict:
    """{relative path: text} of the regular files ending in `suffix` under `port_dir`, read beneath one pinned descriptor (a link is never
    followed, a FIFO or a directory in a file's place is refused, each read is bounded, `__pycache__` is skipped). Raises
    safeio.SafeReadError when the root cannot be pinned, the walk stops early, a file cannot be read or more than MAX_PORT_FILES match."""
    sources: dict = {}
    with safeio.pinned_root(port_dir) as pin:
        found = safeio.scan_beneath(pin, max_depth=4, max_entries=MAX_PORT_FILES * 4)
        for entry in found.entries:
            if entry.kind != "file" or not entry.parts[-1].endswith(suffix) or "__pycache__" in entry.parts:
                continue
            if len(sources) >= MAX_PORT_FILES:
                raise safeio.TooLarge(f"more than {MAX_PORT_FILES} files under the port directory")
            sources["/".join(entry.parts)] = safeio.read_beneath(pin.fd, entry.parts, max_bytes=SOURCE_MAX).decode("utf-8", "replace")
        if found.incomplete:
            raise safeio.IncompleteScan("the port directory walk stopped early (too deep or too many entries)")
    return sources


def check_port(port_dir) -> dict:
    """`check_sources` over the `.py` files `port_sources` reads; a directory it refuses is one problem that names the refusal."""
    try:
        sources = port_sources(port_dir)
    except (safeio.SafeReadError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", None) or getattr(exc, "strerror", None) or exc.__class__.__name__
        return {"ok": False, "problems": [f"the port directory cannot be read: {str(reason)[:200]}"], "matmul_calls": 0, "stubs": [], "parse_errors": []}
    return check_sources(sources)


def covered_names(sheet: dict) -> dict:
    """{dotted ttnn name: the entry that covers it, or None}: every entry, every `ttnn.<name>` an entry's usage line mentions, and the ops an
    entry's notes say are "called the same way" (the sheet's multiply entry names add, subtract and silu that way)."""
    out: dict = {}
    for e in op_cheats.entries(sheet):
        out[e["name"]] = e
        for m in re.finditer(r"\bttnn(?:\.[A-Za-z_][A-Za-z0-9_]*)+", e["usage"]):
            out.setdefault(m.group(0), None)
        m = _SAME_WAY.search(e["notes"])
        if m:
            for leaf in [*m.group(1).split(", "), m.group(2)]:
                out.setdefault(f"ttnn.{leaf}", e)
    return out


def sheet_problems(source: str, sheet: dict | None = None) -> list[str]:
    """Every `ttnn.<name>` the source calls or reads that the verified sheet does not cover, and every keyword a call passes that the
    covering entry's signature does not list. An empty list means each TTNN name in the source has a verified form."""
    sheet = sheet if sheet is not None else op_cheats.load_sheet(op_cheats.TTNN_SHEET)
    covered = covered_names(sheet)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return [f"the source does not parse: {type(exc).__name__}"]
    problems: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name and name.startswith("ttnn."):
                entry = covered.get(name)
                if name not in covered:
                    problems.append(f"line {node.lineno}: {name} is not in the verified sheet")
                elif entry is not None:
                    allowed = op_cheats.signature_names(entry["signature"])
                    for kw in node.keywords:
                        if kw.arg and kw.arg not in allowed:
                            problems.append(f"line {node.lineno}: {name} passes {kw.arg!r}, which the sheet's signature does not list")
        elif isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name and name.startswith("ttnn.") and name not in covered and not any(c.startswith(name + ".") for c in covered):
                problems.append(f"line {node.lineno}: {name} is not in the verified sheet")
    return sorted(set(problems), key=lambda s: (int(s.split()[1].rstrip(":")) if s.startswith("line") else 0, s))


def _dotted(node) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


# ---- reading the inputs from files ----------------------------------------------------------------------------------------------------------

def read_construction(path) -> dict:
    """The construction JSON from a file that holds either the object itself or the line effective_construction.py prints (MARK and the object)."""
    text = safeio.read_bytes(path, max_bytes=INPUT_MAX).decode("utf-8", "replace")
    line = next((ln for ln in text.split("\n") if ln.startswith("ORCHARD_EFFECTIVE_CONSTRUCTION ")), text)
    if line.startswith("ORCHARD_EFFECTIVE_CONSTRUCTION "):
        line = line.split(" ", 1)[1]
    data = safeio.parse_json_bytes(line.encode("utf-8"), max_bytes=INPUT_MAX)
    if not isinstance(data, dict):
        raise PlanError("the construction file does not hold a JSON object")
    return data


def generate_from_files(*, snapshot, construction, inventory=None, excerpts=None) -> tuple[str, Plan]:
    """The generated port for the checkpoint at `snapshot` (its config.json and safetensors headers), the construction file, and the optional
    inventory JSON and excerpts Markdown."""
    from orchard import residency
    snap = Path(snapshot)
    config = safeio.read_json(snap / "config.json", max_bytes=INPUT_MAX, expect=dict)
    headers = residency.read_headers(snap)
    inv = safeio.read_json(inventory, max_bytes=INPUT_MAX, expect=dict) if inventory else None
    exc = safeio.read_bytes(excerpts, max_bytes=INPUT_MAX).decode("utf-8", "replace") if excerpts else None
    return generate(config=config, construction=read_construction(construction), tensors=list(headers.tensors), inventory=inv, excerpts=exc)


def main(argv) -> int:
    """`python3 -m orchard.port_template generate --snapshot DIR --construction FILE [--inventory FILE] [--excerpts FILE] --out FILE`, or
    `python3 -m orchard.port_template check PORT_DIR`. Not a tt-orchard subcommand: the generator is a prototype off the supervisor's path."""
    import argparse
    import json
    import sys
    ap = argparse.ArgumentParser(prog="python3 -m orchard.port_template")
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--snapshot", required=True)
    g.add_argument("--construction", required=True)
    g.add_argument("--inventory")
    g.add_argument("--excerpts")
    g.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("port_dir")
    args = ap.parse_args(argv)
    if args.command == "generate":
        try:
            text, p = generate_from_files(snapshot=args.snapshot, construction=args.construction, inventory=args.inventory, excerpts=args.excerpts)
        except (PlanError, safeio.SafeReadError, OSError, ValueError) as exc:
            print(f"cannot generate: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            return 1
        safeio.atomic_write(args.out, text.encode("utf-8"), mode=0o644)
        print(json.dumps(p.summary(), indent=2))
        return 0
    result = check_port(args.port_dir)
    print(json.dumps({k: v for k, v in result.items()}, indent=2, default=list))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
