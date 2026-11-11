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
`decode_step` that re-runs the whole sequence (the idiom of the port that passed, no cache), and `close`. When the config gives the default
rotary settings and the plan has an attention block or an attention stub, it also writes `ROTARY`, `_rope_tables` and `_apply_rotary` (partial
rotary included), so that an attention stub calls them. Every block it cannot write is a
named function whose body is a `# AGENT:` comment and a NotImplementedError that names the block and the reference class. The generator
does not know the data flow of a layer with children it does not recognise (an indexer, and any per-layer embedding, gated delta net or
hyper-connection that a registered family does not prove), so such a layer's function is a stub too, and it lists the generated helpers and blocks the coder can call from it.
A layer whose parts are all generated blocks (or whose attention block is the one remaining stub) is written whole by `_compose_layer`.
`Construction.resolve` takes the width of a module's weight vector, so that a norm whose path the construction's cut `paths` list leaves out is not given the
group size of another variant that cannot divide its width.

Block families (issue 696). A family of blocks that the generator writes whole is one entry in `BLOCK_FAMILIES`: a recognizer over the
construction and the tensor shapes, an emitter that returns the function text, the module-level helpers it calls, and the id of a test that
compares the generated block with the transformers module at tiny sizes (`register_block_family` refuses an entry without that id). A function a
family wrote has `GENERATED_BLOCK_MARK` as its first comment line, so the thin repair never sends it to the coder as a repair target. The first
entry is the hyper-connection of the Qwen4 family (the prelude that repeats the embedding into the streams, the per-layer connections before
attention and the MLP, the injection after each block, and the mixer after the last layer). The second is the gated delta net (linear attention: the
fused projection, a depthwise causal convolution built from shifted multiply-adds, the decay and beta, `ttnn.transformer.chunk_gated_delta_rule` in its
flat form, the gated norm and the output projection); its load roles are `neg_exp` (minus exp of `A_log`) and `conv_taps` (one device tensor per
convolution tap). The third is the per-layer embedding (PLE: hashed n-gram ids and the table lookup on the host with the table kept in `model["host"]`, then a gate,
group norms and a dilated depthwise convolution on the device). A layer of the Qwen4 family is composed from these (the PLE add, the two connections, the sublayer, the
two injections; issue 696, slice 3). The fifth is the indexed attention (`Qwen4ExpTextAttention` with a `Qwen4ExpTextQSAIndexer` child; issue 696, slice 5): the
block selection runs on the device with the op forms probes 6 to 9 ran on a board, four small steps run on the host (see the comment above `QSA_REFERENCE_TEST`), and
the tie rule of `ttnn.topk(stable=True)` (the lowest index among tied scores) is the block's rule. With it, the tiny Qwen4 plan has no stub. The standard blocks above are not entries. The
recognizer returns None when it cannot prove the shapes, and the block is then a stub as before. The ttnn calls of a family's code are
checked against the verified sheet like the rest. Whether the families' ops (a width-1 slice and a column-broadcast multiply in the injection; `ttnn.softplus`, the row-major concat of the
convolution's left padding and the delta rule op at these head counts) run on a chip is not shown by the CPU test: the first board check of a run that uses them is that evidence.

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
GENERATED_BLOCK_MARK = "# GENERATED (not a stub):"      # the first comment line of a function a block family wrote; a failure inside it is never a coder repair target
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

def _group_size_fits(attrs: dict, width: int) -> bool:
    """True when a norm with these attributes can have a weight vector of `width` values: no group size, or one that divides the width."""
    g = attrs.get("group_size")
    return g is None or (type(g) is int and 0 < g <= width and width % g == 0)


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

    def resolve(self, path: str, observed: set | None = None, width: int | None = None) -> tuple[str | None, dict]:
        """(class name, attrs) at the module path pattern `path`, or (None, {}) when the construction does not say. With several
        candidates, `observed` (the child names the checkpoint's tensors show under the path) picks the variant whose children fit best.
        When the construction does not list the path (its `paths` lists are cut) every variant of the class is a candidate, and `width` (the length of
        the module's weight vector) removes the variants that cannot be: a group size that does not divide the width (arm E14's port facts: the
        indexer's `k_layernorm`, width 32, was given the hyper-connection norm's group size 64, so it was taken for a grouped norm and had no epsilon)."""
        found = list(self.by_path.get(path, []))
        if not found:
            parent, _, last = path.rpartition(".")
            cls, attrs = self.resolve(parent) if parent else (self.by_path.get("<model>") or [(None, {})])[0]   # a top-level name is a child of the root module
            kid = self.children(attrs).get(last) if cls else None
            if kid is None:
                return (None, {})
            found = [(kid, a) for a in self.classes.get(kid, [{}])]
            if width is not None:
                fits = [f for f in found if _group_size_fits(f[1], width)]
                found = fits or found
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
    blocks: dict = field(default_factory=dict)      # child name -> ("mlp" | "attention" | "rms_norm" | "linear" | "stub" | a registered family name, details)
    composed: dict | None = None                    # set when the layer is written whole from generated blocks (`_compose_layer`): the streams, the sublayer and whether a PLE block runs first


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
    model_blocks: dict = field(default_factory=dict)   # text-model child name -> (family name, settings), a block a registered family writes after the layers
    prelude_family: tuple | None = None                # (family name, settings) when a family writes the step between the embedding and the first layer
    rotary: dict | None = None                         # {head_dim, rot, theta} when the rotary helpers are written (issue 696, slice 4)

    def summary(self) -> dict:
        """What the ledger or a run log records: the generated blocks and the stubs, without any file content."""
        return {"model_class": self.model_class, "layers": self.n_layers,
                "generated": sorted({"embedding"} | {b[0] for l in self.layers for b in l.blocks.values() if b[0] != "stub"}
                                    | {b[0] for b in self.model_blocks.values()}
                                    | ({"final_norm"} if self.final_norm else set()) | {"lm_head"}
                                    | ({"layer_flow"} if any(l.standard for l in self.layers) else set())
                                    | ({"layer_composition"} if any(l.composed for l in self.layers) else set())),
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


# ---- block families --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    """What a family's recognizer is shown: one module of the construction, with the checkpoint's tensors."""
    path: str              # module path pattern, such as model.layers.{i}.attn_hyper_connection
    cls: str               # the reference class
    attrs: dict            # the construction's attributes of this module
    kids: dict             # child name -> class
    by_pattern: dict       # tensor name pattern -> Tensor
    con: "Construction"
    plus_one: frozenset    # norm classes whose forward multiplies by 1 + weight (from the reference excerpts)
    where: str             # "layer" (a child of a decoder layer) or "model" (a child of the text model)
    rotary: dict | None = None   # the plan's rotary settings {head_dim, rot, theta} (from the config), or None when the config gives none the generator can use


@dataclass(frozen=True)
class BlockFamily:
    """One family of blocks the generator writes whole, held as data so a later family is an entry and a test.

    recognize(site) -> settings dict, or None when the module is not of this family or the shapes cannot be proved (the block then stays a
        stub: no silent default). A settings dict may hold "placements": {tensor pattern: role}, which the plan applies.
    emit(function, site_path, settings, where) -> the lines of the function `function` (the first comment line is GENERATED_BLOCK_MARK).
    helpers() -> the module-level lines the family's functions call, written once when any block of the family is used.
    prelude(settings) -> the lines of `_model_prelude` when this family also writes the step before the first layer, else None.
    reference_test: the pytest node id of the test that compares the generated block with the transformers module at tiny sizes.
    describe(function, settings) -> one line for the layer stub: the call and what it returns.
    model_level: the family may also stand as a child of the text model (after the layers).
    needs_rotary: the family's functions call `_apply_rotary` and read `ROTARY`, so the plan writes the rotary helpers when a block of this family is used."""
    name: str
    recognize: object
    emit: object
    reference_test: str
    describe: object
    helpers: object = None
    prelude: object = None
    model_level: bool = False
    needs_rotary: bool = False


BLOCK_FAMILIES: dict = {}


def register_block_family(family: BlockFamily) -> None:
    """Add a family. A family without a reference test is refused: a generated block with no comparison against the reference is a claim with no check."""
    if not isinstance(family.reference_test, str) or "::" not in family.reference_test:
        raise ValueError(f"family {family.name!r}: reference_test must be a pytest node id (path::test)")
    if family.name in BLOCK_FAMILIES:
        raise ValueError(f"family {family.name!r} is already registered")
    BLOCK_FAMILIES[family.name] = family


def _recognize_family(site: Site):
    """(family, settings) of the first registered family that recognizes the module, or None."""
    for fam in BLOCK_FAMILIES.values():
        if site.where == "model" and not fam.model_level:
            continue
        settings = fam.recognize(site)
        if settings is not None:
            return fam, settings
    return None


# The hyper-connections of the Qwen4 family: a gated residual that mixes `hc` residual streams into one block input and injects the block's
# output back into each stream. The equations are the transformers `Qwen4ExpTextGatedResidual` (op probe 4 matched them on one chip at PCC 0.9999):
#   normed = group RMSNorm of x (groups of `hidden`) times (1 + hc_norm weight)
#   gate   = sigmoid(up(silu(down(normed) / hc)))          mixed = mean over the hc streams of gate * normed
#   inject = 2 * sigmoid(block_inject(normed) / hc)        (not present for the mixer after the last layer)
#   after the block: x = x + (block_out per stream * inject per stream)
_HC_NEED = frozenset({"hc_norm", "input_mix_weight_down", "input_mix_weight_up"})


def _recognize_hyper_connection(site: Site):
    kids = set(site.kids)
    if not _HC_NEED <= kids or kids - _HC_NEED - {"block_inject_weight"}:
        return None
    hc, hidden = site.attrs.get("hc_count"), site.attrs.get("hidden_size")
    if type(hc) is not int or type(hidden) is not int or hc < 1 or hidden < 32 or hidden % 32:
        return None          # a slice of a stream must start on a tile
    norm_cls, norm_attrs = site.con.resolve(site.path + ".hc_norm")
    if not isinstance(norm_cls, str) or "RMSNorm" not in norm_cls or "Gated" in norm_cls or norm_attrs.get("group_size") != hidden:
        return None
    eps = _eps(norm_attrs, -1.0)
    if eps <= 0:
        return None          # the construction gave no epsilon: no default is invented
    n = hc * hidden

    def shape(kid):
        t = site.by_pattern.get(f"{site.path}.{kid}.weight")
        return t.shape if t is not None else None
    low = shape("input_mix_weight_down")
    if shape("hc_norm") != (n,) or not low or len(low) != 2 or low[1] != n or shape("input_mix_weight_up") != (n, low[0]):
        return None
    inject = "block_inject_weight" in kids
    if inject and shape("block_inject_weight") != (hc, n):
        return None
    role = "norm_plus_one" if norm_cls in site.plus_one else "norm"
    return {"hc": hc, "hidden": hidden, "eps": eps, "inject": inject, "placements": {f"{site.path}.hc_norm.weight": role}}


def _hc_args(settings: dict) -> str:
    return f"hc={settings['hc']}, hidden={settings['hidden']}, eps={settings['eps']!r}, inject={settings['inject']}"


def _emit_hyper_connection(function: str, site_path: str, settings: dict, where: str) -> list:
    if where == "model":
        return [f"def {function}(model, x):",
                f'    """The mixer after the last layer ({site_path}): the streams [1, 1, S, HC * HIDDEN] mixed into [1, 1, S, HIDDEN]."""',
                f"    {GENERATED_BLOCK_MARK} hyper_connection",
                f'    return _hyper_connection(model, "{site_path}", x, {_hc_args(settings)})[0]', "", ""]
    return [f"def {function}(model, i, x, ids):",
            f'    """{site_path}: returns (mixed, hyper_input, injection_weights) for the block that follows; see _hyper_connection_inject."""',
            f"    {GENERATED_BLOCK_MARK} hyper_connection",
            f'    return _hyper_connection(model, f"{site_path}", x, {_hc_args(settings)})', "", ""]


def _emit_hc_prelude(settings: dict) -> list:
    return ["def _model_prelude(model, x, ids):",
            '    """The embedding [1, 1, S, HIDDEN] repeated HC times along the last dimension: the residual streams the layers read and return."""',
            f"    {GENERATED_BLOCK_MARK} hyper_connection",
            f"    return ttnn.concat([x] * {settings['hc']}, dim=-1)", "", ""]


def _hc_helpers() -> list:
    return ["def _hyper_connection(model, prefix, x, hc, hidden, eps, inject):",
            '    """The gated residual at `prefix` on x [1, 1, S, hc * hidden]: (mixed [1, 1, S, hidden], x, injection weights [1, 1, S, hc] or None)."""',
            f"    {GENERATED_BLOCK_MARK} hyper_connection",
            "    seq = x.shape[2]",
            "    n = hc * hidden",
            "    normed = ttnn.reshape(ttnn.rms_norm(ttnn.reshape(x, (1, 1, seq * hc, hidden)), epsilon=eps), (1, 1, seq, n))",
            '    normed = ttnn.multiply(normed, model["dev"][prefix + ".hc_norm.weight"])',
            '    mix = ttnn.silu(ttnn.multiply(_linear(model, prefix + ".input_mix_weight_down", normed), 1.0 / hc))',
            '    gate = ttnn.sigmoid(_linear(model, prefix + ".input_mix_weight_up", mix))',
            "    prod = ttnn.multiply(gate, normed)",
            "    mixed = ttnn.slice(prod, (0, 0, 0, 0), (1, 1, seq, hidden))",
            "    for j in range(1, hc):",
            "        mixed = ttnn.add(mixed, ttnn.slice(prod, (0, 0, 0, j * hidden), (1, 1, seq, (j + 1) * hidden)))",
            "    mixed = ttnn.multiply(mixed, 1.0 / hc)",
            "    if not inject:",
            "        return mixed, x, None",
            '    weights = ttnn.multiply(ttnn.sigmoid(ttnn.multiply(_linear(model, prefix + ".block_inject_weight", normed), 1.0 / hc)), 2.0)',
            "    return mixed, x, weights", "", "",
            "def _hyper_connection_inject(block_out, hyper_input, weights, hc, hidden):",
            '    """hyper_input + block_out scaled by each stream\'s injection weight, the streams joined on the last dimension: [1, 1, S, hc * hidden]."""',
            f"    {GENERATED_BLOCK_MARK} hyper_connection",
            "    seq = block_out.shape[2]",
            "    parts = [ttnn.multiply(block_out, ttnn.slice(weights, (0, 0, 0, j), (1, 1, seq, j + 1))) for j in range(hc)]",
            "    return ttnn.add(hyper_input, ttnn.concat(parts, dim=-1))", "", ""]


def _describe_hyper_connection(function: str, settings: dict) -> str:
    call = f"{function}(model, i, x, ids)"
    if not settings["inject"]:
        return f"{call}"
    return (f"{call} returns (mixed, hyper_input, injection_weights): run the block on `mixed`, then "
            f"x = _hyper_connection_inject(block_out, hyper_input, injection_weights, {settings['hc']}, {settings['hidden']})")


register_block_family(BlockFamily(
    name="hyper_connection", recognize=_recognize_hyper_connection, emit=_emit_hyper_connection,
    reference_test="tests/test_port_template_registry.py::test_generated_hyper_connections_match_the_transformers_module",
    describe=_describe_hyper_connection, helpers=_hc_helpers, prelude=_emit_hc_prelude, model_level=True))


# The gated delta net of the Qwen4 family (linear attention; the Qwen3.5 and Olmo hybrids hold the same projections). The equations are the transformers
# `Qwen4ExpTextGatedDeltaNet.forward` without a cache (prefill):
#   qkv = in_proj_qkv(x); z = in_proj_z(x); b = in_proj_b(x); a = in_proj_a(x)
#   qkv = silu(causal depthwise convolution of qkv over the sequence, kernel K, one weight row per channel and tap)
#   q, k, v = split(qkv); beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
#   o = the gated delta rule over (q, k, v, g, beta) with the key heads shared across the value heads (repeat_interleave); q and k are L2 normed
#   o = RMSNorm(o over each value head, weight) * act(z); out = out_proj(o)         act is the norm class's `activation` (sigmoid or silu)
# The delta rule is `ttnn.transformer.chunk_gated_delta_rule` in its flat (rank 3, token-major) form with `num_key_heads`; op probe 2 matched it with the
# reference at the Flash-Next head counts (16 key, 48 value heads, width 128) and at 2 key / 4 value heads of width 64. The convolution is four shifted
# multiply-adds because `ttnn.conv1d` was wrong for a dilated depthwise convolution (probe 1); this convolution has no dilation, and the shifted form is what
# probe 1 validated. `-exp(A_log)` is computed at load (role `neg_exp`) and each convolution tap is its own device tensor (role `conv_taps`).
_GDN_KIDS = frozenset({"conv1d", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "norm", "out_proj"})
_GDN_GATES = ("sigmoid", "silu")


def _recognize_gated_delta_net(site: Site):
    if site.where != "layer" or set(site.kids) != _GDN_KIDS:
        return None
    a = site.attrs
    hk, hv, dk, dv, kernel, hidden = (a.get(k) for k in ("num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim", "conv_kernel_size", "hidden_size"))
    if any(type(v) is not int or v < 1 for v in (hk, hv, dk, dv, kernel, hidden)):
        return None
    if hv % hk or dk % 32 or dv % 32 or hidden % 32:
        return None          # the op maps value heads to key heads in whole groups, and a slice of the fused projection must start on a tile
    if a.get("activation") != "silu":
        return None          # the convolution's activation: silu is the only one the generator writes
    norm_cls, norm_attrs = site.con.resolve(site.path + ".norm")
    gate = norm_attrs.get("activation")
    if not isinstance(norm_cls, str) or "RMSNormGated" not in norm_cls or gate not in _GDN_GATES:
        return None
    eps = _eps(norm_attrs, -1.0)
    if eps <= 0:
        return None          # the construction gave no epsilon: no default is invented
    key_dim, value_dim = hk * dk, hv * dv
    conv_dim = 2 * key_dim + value_dim

    def shape(name):
        t = site.by_pattern.get(f"{site.path}.{name}")
        return t.shape if t is not None and t.dtype in FLOAT_DTYPES else None
    if (shape("in_proj_qkv.weight") != (conv_dim, hidden) or shape("in_proj_z.weight") != (value_dim, hidden)
            or shape("in_proj_b.weight") != (hv, hidden) or shape("in_proj_a.weight") != (hv, hidden)
            or shape("out_proj.weight") != (hidden, value_dim) or shape("conv1d.weight") != (conv_dim, 1, kernel)
            or shape("norm.weight") != (dv,) or shape("A_log") != (hv,) or shape("dt_bias") != (hv,)):
        return None
    return {"hk": hk, "hv": hv, "dk": dk, "dv": dv, "kernel": kernel, "eps": eps, "gate": gate, "hidden": hidden,
            "placements": {f"{site.path}.norm.weight": "norm", f"{site.path}.A_log": "neg_exp", f"{site.path}.dt_bias": "tile",
                           f"{site.path}.conv1d.weight": "conv_taps"}}


def _emit_gated_delta_net(function: str, site_path: str, settings: dict, where: str) -> list:
    s = settings
    return [f"def {function}(model, i, x, ids):",
            f'    """{site_path}: the gated delta net on x [1, 1, S, {s["hidden"]}] (prefill, no cache); returns the block output [1, 1, S, {s["hidden"]}], without the residual add."""',
            f"    {GENERATED_BLOCK_MARK} gated_delta_net",
            f'    return _gated_delta_net(model, f"{site_path}", x, hk={s["hk"]}, hv={s["hv"]}, dk={s["dk"]}, dv={s["dv"]}, kernel={s["kernel"]}, eps={s["eps"]!r}, gate="{s["gate"]}")',
            "", ""]


def _gdn_helpers() -> list:
    return ["def _causal_conv(model, name, x, kernel, channels):",
            '    """The depthwise causal convolution over the sequence of x [1, 1, S, channels], then silu: output row t is the sum over taps j of tap j (uploaded by load as `name.tap<j>`) times input row t - (kernel - 1 - j), with zero rows before the first."""',
            f"    {GENERATED_BLOCK_MARK} gated_delta_net",
            "    seq = x.shape[2]",
            "    xpad = ttnn.concat([_upload(model, torch.zeros(1, 1, kernel - 1, channels, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT), ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)], dim=2)",
            "    acc = None",
            "    for j in range(kernel):",
            "        piece = ttnn.to_layout(ttnn.slice(xpad, (0, 0, j, 0), (1, 1, j + seq, channels)), ttnn.TILE_LAYOUT)",
            '        term = ttnn.multiply(piece, model["dev"][f"{name}.tap{j}"])',
            "        acc = term if acc is None else ttnn.add(acc, term)",
            "    return ttnn.silu(acc)", "", "",
            "def _gated_delta_net(model, prefix, x, hk, hv, dk, dv, kernel, eps, gate):",
            '    """The gated delta net at `prefix` on x [1, 1, S, hidden] (S a multiple of 32, no cache): the block output [1, 1, S, hidden]."""',
            f"    {GENERATED_BLOCK_MARK} gated_delta_net",
            "    seq = x.shape[2]",
            "    key_dim, value_dim = hk * dk, hv * dv",
            "    conv_dim = 2 * key_dim + value_dim",
            '    qkv = _linear(model, prefix + ".in_proj_qkv", x)',
            '    z = _linear(model, prefix + ".in_proj_z", x)',
            '    b = _linear(model, prefix + ".in_proj_b", x)',
            '    a = _linear(model, prefix + ".in_proj_a", x)',
            '    qkv = _causal_conv(model, prefix + ".conv1d.weight", qkv, kernel, conv_dim)',
            "    beta = ttnn.sigmoid(b)",
            '    g = ttnn.multiply(ttnn.softplus(ttnn.add(a, model["dev"][prefix + ".dt_bias"])), model["dev"][prefix + ".A_log"])',
            "    q = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, seq, key_dim)), (1, seq, key_dim))",
            "    k = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, key_dim), (1, 1, seq, 2 * key_dim)), (1, seq, key_dim))",
            "    v = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, 2 * key_dim), (1, 1, seq, conv_dim)), (1, seq, value_dim))",
            "    g, beta = ttnn.reshape(g, (1, seq, hv)), ttnn.reshape(beta, (1, seq, hv))",
            "    o = ttnn.transformer.chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32, num_key_heads=hk)[0]          # [1, S, hv, dv], fp32, ROW_MAJOR",
            "    o = ttnn.typecast(ttnn.to_layout(ttnn.reshape(o, (1, 1, seq * hv, dv)), ttnn.TILE_LAYOUT), ttnn.bfloat16)",
            '    o = ttnn.rms_norm(o, epsilon=eps, weight=model["dev"][prefix + ".norm.weight"])',
            "    z = ttnn.reshape(z, (1, 1, seq * hv, dv))",
            '    if gate == "sigmoid":',
            "        z = ttnn.sigmoid(z)",
            "    else:",
            "        z = ttnn.silu(z)",
            "    o = ttnn.multiply(o, z)",
            '    return _linear(model, prefix + ".out_proj", ttnn.reshape(o, (1, 1, seq, value_dim)))', "", ""]


def _describe_gated_delta_net(function: str, settings: dict) -> str:
    return (f"{function}(model, i, mixed, ids) returns the block output [1, 1, S, {settings['hidden']}] (no residual add; the block's input is the "
            "mixed input of the connection before it)")


register_block_family(BlockFamily(
    name="gated_delta_net", recognize=_recognize_gated_delta_net, emit=_emit_gated_delta_net, reference_test=(
        "tests/test_port_template_gdn.py::test_generated_gated_delta_net_matches_the_transformers_module"),
    describe=_describe_gated_delta_net, helpers=_gdn_helpers))


# The per-layer embedding block (PLE) of the Qwen4 family: `Qwen4ExpTextPLELayer.forward` without a cache (prefill). The layer adds its output to the
# residual streams before the attention connection. The equations:
#   ids -> n-gram ids: the previous `ngram_size - 1` tokens (the context starts as eos, and a token's context restarts after an eos), shifted copies of the token
#          history, each head's id = ((t0 * m0) xor (t1 * m1) ...) % head_vocab_size + head_offset, all in int64 (m is the checkpoint's `layer_multipliers`)
#   emb  = the table rows of those ids, the heads joined: [S, ngram_heads * head_dim] = [S, E]
#   key  = group RMSNorm(key_proj(emb)) per stream; value = value_proj(emb); query = group RMSNorm(x) per stream
#   gate = sum over the stream's width of key * query / sqrt(hidden); gate = sign(gate) * sqrt(max(|gate|, 1e-6))
#   gated = sigmoid(gate) * value (the value is shared by the streams); out = gated + silu(dilated depthwise causal convolution of group RMSNorm(gated))
# The hashing and the table lookup run on the HOST (`_ple_ngram_embeddings`): the ids are integers, and the table is large (the real model's is over 100 GB), so
# the table stays in model["host"] as the checkpoint's row shards and is never uploaded. The embeddings (a [S, E] slab) are uploaded once per call. The projections,
# the norms, the gate and the convolution run on the device. The dilated convolution is shifted multiply-adds (probe 1: `ttnn.conv1d` was wrong for a dilated
# depthwise convolution). The gate needs `ttnn.sum`, `abs`, `sign`, `sqrt` and `clamp`; none of them has run on a chip in this repository.
PLE_REFERENCE_TEST = "tests/test_port_template_ple.py::test_generated_ple_matches_the_transformers_module"
_PLE_KIDS = frozenset({"ple_embedding", "key_proj", "value_proj", "norm_key", "norm_query", "norm_conv", "conv1d"})
_PLE_TABLE = re.compile(r"\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")


def _recognize_ple(site: Site):
    if site.where != "layer" or set(site.kids) != _PLE_KIDS:
        return None
    hc, hidden, state_len = (site.attrs.get(k) for k in ("hc_count", "hidden_size", "short_conv_state_len"))
    if any(type(v) is not int or v < 1 for v in (hc, hidden, state_len)) or hidden % 32:
        return None          # a slice of a stream must start on a tile
    n = hc * hidden
    emb_cls, emb = site.con.resolve(site.path + ".ple_embedding")
    ngram, per, eos, ctx, heads = (emb.get(k) for k in ("ngram_size", "heads_per_ngram", "eos_token_id", "context_len", "ngram_heads"))
    if any(type(v) is not int or v < 1 for v in (ngram, per, eos if type(eos) is int and eos >= 0 else 0, heads)) or ngram < 2 or ctx != ngram - 1 or heads != (ngram - 1) * per:
        return None
    total = emb.get("total_vocab_size")
    if type(total) is not int or total < 1:
        return None
    eps = None
    for norm in ("norm_key", "norm_query", "norm_conv"):
        cls, attrs = site.con.resolve(f"{site.path}.{norm}")
        if not isinstance(cls, str) or "RMSNorm" not in cls or "Gated" in cls or attrs.get("group_size") != hidden:
            return None
        e = _eps(attrs, -1.0)
        if e <= 0 or (eps is not None and e != eps):
            return None      # no epsilon in the construction, or the three norms disagree: no default is invented
        eps = e

    def shape(name, dtypes=FLOAT_DTYPES):
        t = site.by_pattern.get(f"{site.path}.{name}")
        return t.shape if t is not None and t.dtype in dtypes else None
    conv = shape("conv1d.weight")
    if conv is None or len(conv) != 3 or conv[0] != n or conv[1] != 1 or conv[2] < 1 or state_len != (conv[2] - 1) * ngram:
        return None          # depthwise, one channel per stream element, and the state length of the reference's dilation (= ngram_size)
    kernel = conv[2]
    key = shape("key_proj.weight")
    if key is None or len(key) != 2 or key[0] != n or key[1] % heads or shape("value_proj.weight") != (hidden, key[1]):
        return None
    head_dim = key[1] // heads
    if any(shape(f"{k}.weight") != (n,) for k in ("norm_key", "norm_query", "norm_conv")):
        return None
    ints = {"I64", "I32"}
    if (shape("ple_embedding.layer_multipliers", ints) != (ngram,) or shape("ple_embedding.ngram_heads_vocab_sizes", ints) != (heads,)
            or shape("ple_embedding.ngram_heads_offsets", ints) != (heads,)):
        return None
    prefix = site.path + ".ple_embedding.ngram_embedding.shard_"
    shards = sorted((int(m.group(1)), name) for name in site.by_pattern for m in [_PLE_TABLE.search(name)] if m and name.startswith(prefix))
    if not shards or [k for k, _ in shards] != list(range(len(shards))):
        return None          # the table is the checkpoint's row shards, numbered from 0 with no gap
    rows = 0
    for _k, name in shards:
        t = site.by_pattern[name]
        if t.dtype not in FLOAT_DTYPES or len(t.shape) != 2 or t.shape[1] != head_dim:
            return None
        rows += t.shape[0]
    if rows < total:
        return None          # the head offsets would point past the table
    names = [f".ple_embedding.ngram_embedding.shard_{k}.weight" for k, _ in shards]
    norm_cls, _ = site.con.resolve(site.path + ".norm_key")
    role = "norm_plus_one" if norm_cls in site.plus_one else "norm"
    placements = {f"{site.path}.{norm}.weight": role for norm in ("norm_key", "norm_query", "norm_conv")}
    placements[f"{site.path}.conv1d.weight"] = "conv_taps"
    placements.update({name: "host" for _k, name in shards})
    return {"hc": hc, "hidden": hidden, "eps": eps, "kernel": kernel, "dilation": ngram, "ngram": ngram, "per": per, "eos": eos, "tables": tuple(names),
            "placements": placements}


def _emit_ple(function: str, site_path: str, settings: dict, where: str) -> list:
    s = settings
    n = s["hc"] * s["hidden"]
    return [f"def {function}(model, i, x, ids):",
            f'    """{site_path}: the per-layer embedding on the streams x [1, 1, S, {n}] and the token ids (a list of S ints); returns [1, 1, S, {n}], to be ADDED to x '
            'before the attention connection."""',
            f"    {GENERATED_BLOCK_MARK} ple",
            f'    return _ple(model, f"{site_path}", x, ids, hc={s["hc"]}, hidden={s["hidden"]}, eps={s["eps"]!r}, kernel={s["kernel"]}, dilation={s["dilation"]}, '
            f'ngram_size={s["ngram"]}, heads_per_ngram={s["per"]}, eos={s["eos"]}, tables={s["tables"]!r})', "", ""]


def _ple_helpers() -> list:
    mark = f"    {GENERATED_BLOCK_MARK} ple"
    return ["def _ple_shift(token_ids, shift, eos):",
            '    """The token history moved right by `shift`; a position whose context reaches back past an eos token (or the start) gets eos."""',
            mark,
            "    if shift == 0:",
            "        return token_ids",
            "    batch, seq = token_ids.shape",
            "    positions = torch.arange(seq, dtype=torch.long)",
            "    eos_positions = torch.where(token_ids == eos, positions, -1)",
            "    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values",
            "    previous_eos = torch.cat([eos_positions.new_full((batch, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)",
            "    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)",
            "    source_positions = positions - shift",
            "    shifted = token_ids.gather(dim=1, index=source_positions.clamp_min(0).unsqueeze(0).expand(batch, -1))",
            "    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)",
            "    return torch.where(valid, shifted, token_ids.new_full((), eos))", "", "",
            "def _ple_ngram_embeddings(model, prefix, ids, ngram_size, heads_per_ngram, eos, tables):",
            '    """The hashed n-gram embeddings of the token ids as a host tensor [1, S, ngram_heads * head_dim]. Host code: integer hashing and a lookup in the table shards, which stay in model["host"]."""',
            mark,
            "    host = model[\"host\"]",
            "    ids_t = torch.tensor([[int(t) for t in ids]], dtype=torch.long)",
            "    history = torch.cat([torch.full((1, ngram_size - 1), eos, dtype=torch.long), ids_t], dim=-1)",
            "    shifted = [_ple_shift(history, shift, eos) for shift in range(ngram_size)]",
            '    mult = host[prefix + ".ple_embedding.layer_multipliers"].long()',
            '    sizes = host[prefix + ".ple_embedding.ngram_heads_vocab_sizes"].long()',
            '    offsets = host[prefix + ".ple_embedding.ngram_heads_offsets"].long()',
            "    blocks = []",
            "    for ngram in range(2, ngram_size + 1):",
            "        start = (ngram - 2) * heads_per_ngram",
            "        end = start + heads_per_ngram",
            "        mixed = shifted[0] * mult[0]",
            "        for pos in range(1, ngram):",
            "            mixed = torch.bitwise_xor(mixed, shifted[pos] * mult[pos])",
            "        vocab, off = sizes[start:end], offsets[start:end]",
            "        blocks.append(torch.remainder(mixed.unsqueeze(-1), vocab.view(1, 1, -1)) + off.view(1, 1, -1))",
            "    ngram_ids = torch.cat(blocks, dim=-1)[:, -ids_t.shape[1]:]",
            "    flat = ngram_ids.reshape(-1)",
            "    first = host[prefix + tables[0]]",
            "    out = torch.zeros(flat.shape[0], first.shape[1], dtype=first.dtype)",
            "    lo = 0",
            "    for name in tables:",
            "        part = host[prefix + name]",
            "        hit = (flat >= lo) & (flat < lo + part.shape[0])",
            "        out[hit] = part[flat[hit] - lo]",
            "        lo += part.shape[0]",
            "    return out.reshape(1, ids_t.shape[1], -1)", "", "",
            "def _group_rms_norm(model, name, x, hc, hidden, eps):",
            '    """RMS norm over each stream\'s `hidden` values of x [1, 1, S, hc * hidden], times the weight at `name`."""',
            mark,
            "    seq = x.shape[2]",
            "    normed = ttnn.reshape(ttnn.rms_norm(ttnn.reshape(x, (1, 1, seq * hc, hidden)), epsilon=eps), (1, 1, seq, hc * hidden))",
            '    return ttnn.multiply(normed, model["dev"][name + ".weight"])', "", "",
            "def _dilated_causal_conv(model, name, x, kernel, dilation, channels):",
            '    """The depthwise causal convolution of x [1, 1, S, channels] with dilation, then silu: output row t is the sum over taps j of tap j (uploaded by load as `name.tap<j>`) times row t - (kernel - 1 - j) * dilation."""',
            mark,
            "    seq = x.shape[2]",
            "    pad = (kernel - 1) * dilation",
            "    xpad = ttnn.concat([_upload(model, torch.zeros(1, 1, pad, channels, dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT), ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)], dim=2)",
            "    acc = None",
            "    for j in range(kernel):",
            "        piece = ttnn.to_layout(ttnn.slice(xpad, (0, 0, j * dilation, 0), (1, 1, j * dilation + seq, channels)), ttnn.TILE_LAYOUT)",
            '        term = ttnn.multiply(piece, model["dev"][f"{name}.tap{j}"])',
            "        acc = term if acc is None else ttnn.add(acc, term)",
            "    return ttnn.silu(acc)", "", "",
            "def _ple(model, prefix, x, ids, hc, hidden, eps, kernel, dilation, ngram_size, heads_per_ngram, eos, tables):",
            '    """The per-layer embedding at `prefix` on the streams x [1, 1, S, hc * hidden]: the tensor to add to x, [1, 1, S, hc * hidden]."""',
            mark,
            "    seq = x.shape[2]",
            "    n = hc * hidden",
            "    emb = _ple_ngram_embeddings(model, prefix, ids, ngram_size, heads_per_ngram, eos, tables)",
            "    e = _upload(model, emb.reshape(1, 1, seq, -1).to(torch.bfloat16), ttnn.TILE_LAYOUT)",
            '    key = _group_rms_norm(model, prefix + ".norm_key", _linear(model, prefix + ".key_proj", e), hc, hidden, eps)',
            '    value = _linear(model, prefix + ".value_proj", e)',
            '    query = _group_rms_norm(model, prefix + ".norm_query", x, hc, hidden, eps)',
            "    prod = ttnn.multiply(key, query)",
            "    gated_parts = []",
            "    for j in range(hc):",
            "        s = ttnn.sum(ttnn.slice(prod, (0, 0, 0, j * hidden), (1, 1, seq, (j + 1) * hidden)), dim=3, keepdim=True)",
            "        s = ttnn.multiply(s, hidden ** -0.5)",
            "        mag = ttnn.sqrt(ttnn.clamp(ttnn.abs(s), min=1e-6))",
            "        gate = ttnn.sigmoid(ttnn.multiply(ttnn.sign(s), mag))",
            "        gated_parts.append(ttnn.multiply(value, gate))",
            "    gated = ttnn.concat(gated_parts, dim=-1)",
            '    normed = _group_rms_norm(model, prefix + ".norm_conv", gated, hc, hidden, eps)',
            '    return ttnn.add(gated, _dilated_causal_conv(model, prefix + ".conv1d.weight", normed, kernel, dilation, n))', "", ""]


def _describe_ple(function: str, settings: dict) -> str:
    return (f"{function}(model, i, x, ids) returns [1, 1, S, {settings['hc'] * settings['hidden']}], to be ADDED to x before the attention connection "
            "(x = ttnn.add(x, ...)); ids is the padded token list")


register_block_family(BlockFamily(
    name="ple", recognize=_recognize_ple, emit=_emit_ple, reference_test=PLE_REFERENCE_TEST, describe=_describe_ple, helpers=_ple_helpers))


# The indexed attention of the Qwen4 family: `Qwen4ExpTextAttention` with its `Qwen4ExpTextQSAIndexer` (prefill, no cache). The reference forward:
#   indexer: qk = index_qk_proj(x); q = RMSNorm(split into index_n_heads heads), rotary at the query position; the index keys are the token keys (one head);
#            for each query, the visible tokens before its last incomplete block form complete blocks of `ratio` tokens; each block's key is the mean of its token keys,
#            RMSNorm'ed and rotated at the block's first position; score(block) = sum over index heads of relu(q . block key) / sqrt(index_dim); the
#            `budget / ratio` best complete blocks are kept, with every visible token after the last complete block (the tail); the mask is those tokens.
#   attention: q and the output gate come from q_proj ([q | gate] per head), RMSNorm per head on q and k, rotary on q and k (the rotary part of each head),
#              grouped-key attention under (causal AND selected), joined heads times sigmoid(gate), o_proj.
# The device ops are the forms probes 6 to 9 ran on a board (PCC 0.9998 or better): the q/gate split by reshape and slice, the head permute, the per-head norm on
# [1, 1, heads * S, hd], the rotary helpers, the score matmul of [1, heads, S, D] with a [1, 1, D, nb] operand, relu, sum over heads, `ttnn.topk(stable=True)`, the
# mask built as selection @ expansion + tail, and scaled_dot_product_attention with an additive mask and native grouped KV. Four steps run on the host, on small
# tensors, because the probes validated the device op and not the chain into it: the block mean of the index keys (the reference takes a float mean too), the
# transposed pooled keys (probe 9 built that operand on the host), a one-hot matrix of the selected blocks built from the top-k indices (the reference loop is host code), and
# the constant masks (validity, expansion, tail), built once per prompt length.
# TIE RULE: scores tie often (relu gives exact zeros on random weights), and torch.topk's order among tied scores is not the lowest index (probe 7, real shapes: 509 of 2557
# rows differ). The block uses `stable=True`, which picks the lowest index among tied scores, on every row of a board run. On a model whose scores do not tie
# the two agree. The reference comparison runs the reference with the same rule, and states how far the unmodified reference is.
QSA_REFERENCE_TEST = "tests/test_port_template_qsa.py::test_generated_qsa_attention_matches_the_transformers_module_with_the_lowest_index_tie_rule"
_QSA_KIDS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "indexer"})
_QSA_INDEXER_KIDS = frozenset({"index_qk_proj", "q_layernorm", "k_layernorm"})


def _recognize_qsa_attention(site: Site):
    if site.where != "layer" or set(site.kids) != _QSA_KIDS or site.rotary is None:
        return None
    hd, groups, scaling = (site.attrs.get(k) for k in ("head_dim", "num_key_value_groups", "scaling"))
    if type(hd) is not int or hd < 32 or hd % 32 or type(groups) is not int or groups < 1 or site.attrs.get("is_causal") is not True:
        return None
    if not isinstance(scaling, (int, float)) or abs(scaling - hd ** -0.5) > 1e-6 * hd ** -0.5 or site.rotary["head_dim"] != hd:
        return None
    ind_cls, ind = site.con.resolve(site.path + ".indexer")
    if set(site.con.children(ind)) != _QSA_INDEXER_KIDS:
        return None
    heads_i, kv_i, dim, ratio, budget, topk = (ind.get(k) for k in ("index_n_heads", "index_kv_heads", "index_head_dim", "compress_ratio", "token_budget", "block_topk"))
    if any(type(v) is not int or v < 1 for v in (heads_i, kv_i, dim, ratio, budget, topk)):
        return None
    if kv_i != 1 or dim < 32 or dim % 32 or dim < site.rotary["rot"] or ratio < 2 or 32 % ratio or budget % ratio or topk != budget // ratio:
        return None          # one index key head, a whole number of tiles wide and at least as wide as the rotary part (the reference rotates the first `rot` columns of an index vector with the attention's tables; Flash-Next has index_head_dim 128 and head_dim 256); a prompt padded to 32 tokens then holds whole blocks

    def shape(name, dtypes=FLOAT_DTYPES):
        t = site.by_pattern.get(f"{site.path}.{name}")
        return t.shape if t is not None and t.dtype in dtypes else None
    q, k, v, o = (shape(f"{n}.weight") for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    if any(s is None or len(s) != 2 for s in (q, k, v, o)):
        return None
    hidden = q[1]
    if q[0] % (2 * hd):
        return None
    heads, kv = q[0] // (2 * hd), k[0] // hd
    if heads != kv * groups or k != (kv * hd, hidden) or v != k or o != (hidden, heads * hd) or shape("indexer.index_qk_proj.weight") != ((heads_i + kv_i) * dim, hidden):
        return None
    if any(f"{site.path}.{n}.bias" in site.by_pattern for n in ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj")):
        return None          # the block writes no bias
    for norm, width in (("q_norm", hd), ("k_norm", hd), ("indexer.q_layernorm", dim), ("indexer.k_layernorm", dim)):
        cls, attrs = site.con.resolve(f"{site.path}.{norm}", width=width)
        if not isinstance(cls, str) or not _is_rms_norm(cls, attrs) or _eps(attrs, -1.0) <= 0 or shape(f"{norm}.weight") != (width,):
            return None      # no epsilon in the construction: no default is invented
    return {"heads": heads, "kv_heads": kv, "hd": hd, "index_heads": heads_i, "index_dim": dim, "ratio": ratio, "budget": budget, "hidden": hidden}


def _emit_qsa_attention(function: str, site_path: str, s: dict, where: str) -> list:
    return [f"def {function}(model, i, x, ids):",
            f'    """{site_path}: the indexed attention on x [1, 1, S, {s["hidden"]}] (prefill, no cache; S a multiple of {s["ratio"]}); returns the block output [1, 1, S, {s["hidden"]}], '
            'without the residual add."""',
            f"    {GENERATED_BLOCK_MARK} qsa_attention",
            f'    return _qsa_attention(model, f"{site_path}", x, heads={s["heads"]}, kv_heads={s["kv_heads"]}, hd={s["hd"]}, index_heads={s["index_heads"]}, '
            f'index_dim={s["index_dim"]}, ratio={s["ratio"]}, budget={s["budget"]})', "", ""]


_QSA_HELPERS = '''def _qsa_rope_tables(model, positions, width):
    """cos and sin [1, 1, n, width] TILE bf16 for the given positions (a torch tensor of ints), the way _rope_tables builds them: ROTARY["rot"] rotating columns, then 1 and 0. `width` is the head width of the vectors they are applied to (the attention's head_dim, or the indexer's index_head_dim)."""
    # MARK
    hd, rot = width, ROTARY["rot"]
    inv = 1.0 / (ROTARY["theta"] ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    n = positions.shape[0]
    cos = torch.cat([emb.cos(), torch.ones(n, hd - rot)], dim=-1).reshape(1, 1, n, hd)
    sin = torch.cat([emb.sin(), torch.zeros(n, hd - rot)], dim=-1).reshape(1, 1, n, hd)
    return _upload(model, cos.to(torch.bfloat16), ttnn.TILE_LAYOUT), _upload(model, sin.to(torch.bfloat16), ttnn.TILE_LAYOUT)


def _qsa_apply_rotary(model, x, cos, sin):
    """Rotary on x [1, heads, S, width] with the tables of `_qsa_rope_tables(model, positions, width)`: the first ROTARY["rot"] columns of each head rotate (rotate-half through a constant matrix, as `_apply_rotary` does) and the rest pass through. `width` comes from x, so it serves index vectors that are narrower or wider than the attention head."""
    # MARK
    heads, seq, width, rot = x.shape[1], x.shape[2], x.shape[3], ROTARY["rot"]
    if len(x.shape) != 4 or x.shape[0] != 1 or rot > width or tuple(cos.shape) != (1, 1, seq, width) or tuple(sin.shape) != tuple(cos.shape):
        raise ValueError(f"_qsa_apply_rotary needs x [1, heads, S, width] with width >= {rot} and tables [1, 1, S, width] (from _qsa_rope_tables(model, positions, width)); got x {list(x.shape)}, cos {list(cos.shape)}")
    half = rot // 2
    rot_m = torch.zeros(width, width)
    for i in range(half):
        rot_m[i, i + half] = 1.0
        rot_m[i + half, i] = -1.0
    r = _upload(model, rot_m.to(torch.bfloat16), ttnn.TILE_LAYOUT)
    rotated = ttnn.linear(ttnn.reshape(x, (1, 1, heads * seq, width)), r, dtype=ttnn.bfloat16, compute_kernel_config=model["ck"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rotated = ttnn.reshape(rotated, (1, heads, seq, width))
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))


def _qsa_masks(model, seq, ratio):
    """The constants of the block selection for a prompt of `seq` tokens, built on the host and uploaded once per (seq, ratio): `valid` [S, nb] (block b is complete
    for query q when its last token is at or before q), `bias` (0 where valid, -1 where not: scores are never below 0, so an invalid block ranks last), `expand`
    [nb, S] (block to its tokens) and `tail` [S, S] (the visible tokens after the last complete block)."""
    # MARK
    cache = model.setdefault("qsa_cache", {})
    if (seq, ratio) in cache:
        return cache[(seq, ratio)]
    nb = seq // ratio
    q = torch.arange(seq)
    valid = ((torch.arange(nb) + 1) * ratio - 1)[None, :] <= q[:, None]
    tail = (q[None, :] >= (((q + 1) // ratio) * ratio)[:, None]) & (q[None, :] <= q[:, None])
    expand = (q[None, :] // ratio) == torch.arange(nb)[:, None]
    up = lambda m: _upload(model, m.to(torch.float32).reshape(1, 1, *m.shape).to(torch.bfloat16), ttnn.TILE_LAYOUT)
    cache[(seq, ratio)] = {"valid_host": valid, "valid": up(valid), "bias": up(valid.to(torch.float32) - 1.0), "expand": up(expand), "tail": up(tail)}
    return cache[(seq, ratio)]


def _qsa_split_heads(x, heads, hd):
    """x [1, 1, S, heads * hd] to [1, heads, S, hd]: ttnn.reshape to [1, S, heads, hd], then ttnn.permute."""
    # MARK
    return ttnn.permute(ttnn.reshape(x, (1, x.shape[2], heads, hd)), (0, 2, 1, 3))


def _qsa_norm_heads(model, prefix, y):
    """The per-head RMS norm of y [1, heads, S, hd] with the weight at `prefix`: rows of width hd, as in probe 9 (head_norm)."""
    # MARK
    h, seq, hd = y.shape[1], y.shape[2], y.shape[3]
    return ttnn.reshape(_rms_norm(model, prefix, ttnn.reshape(y, (1, 1, h * seq, hd))), (1, h, seq, hd))


def _qsa_selection_mask(model, prefix, x, index_heads, index_dim, ratio, budget):
    """The additive attention mask [1, 1, S, S] (0 where a query may attend, -30000 elsewhere) from the indexer at `prefix`: causal AND (selected blocks OR tail)."""
    # MARK
    seq = x.shape[2]
    nb = seq // ratio
    k = min(budget // ratio, nb)
    cos, sin = _qsa_rope_tables(model, torch.arange(seq), index_dim)
    ip = _linear(model, prefix + ".index_qk_proj", x)
    q = _qsa_split_heads(ttnn.slice(ip, (0, 0, 0, 0), (1, 1, seq, index_heads * index_dim)), index_heads, index_dim)
    q = _qsa_apply_rotary(model, _qsa_norm_heads(model, prefix + ".q_layernorm", q), cos, sin)
    keys = _to_host(model, ttnn.slice(ip, (0, 0, 0, index_heads * index_dim), (1, 1, seq, (index_heads + 1) * index_dim))).float()
    pooled = keys.reshape(nb, ratio, index_dim).mean(dim=1).to(torch.bfloat16)
    pooled = _rms_norm(model, prefix + ".k_layernorm", _upload(model, pooled.reshape(1, 1, nb, index_dim), ttnn.TILE_LAYOUT))
    block_cos, block_sin = _qsa_rope_tables(model, torch.arange(nb) * ratio, index_dim)
    pooled = _to_host(model, _qsa_apply_rotary(model, pooled, block_cos, block_sin)).float().reshape(nb, index_dim)
    pk = _upload(model, pooled.t().reshape(1, 1, index_dim, nb).to(torch.bfloat16), ttnn.TILE_LAYOUT)
    scores = ttnn.multiply(ttnn.sum(ttnn.relu(ttnn.matmul(q, pk)), dim=1, keepdim=True), index_dim ** -0.5)
    const = _qsa_masks(model, seq, ratio)
    scores = ttnn.add(ttnn.multiply(scores, const["valid"]), const["bias"])
    values, indices = ttnn.topk(scores, k, dim=-1, largest=True, sorted=True, stable=True)
    chosen = _to_host(model, indices).reshape(seq, -1)[:, :k].long()
    selected = torch.zeros(seq, nb).scatter_(1, chosen, 1.0) * const["valid_host"].to(torch.float32)
    selected = _upload(model, selected.reshape(1, 1, seq, nb).to(torch.bfloat16), ttnn.TILE_LAYOUT)
    visible = ttnn.add(ttnn.matmul(selected, const["expand"]), const["tail"])
    return ttnn.multiply(ttnn.subtract(visible, 1.0), 30000.0)


def _qsa_attention(model, prefix, x, heads, kv_heads, hd, index_heads, index_dim, ratio, budget):
    """The indexed attention at `prefix` on x [1, 1, S, hidden] (S a multiple of `ratio`): the block output [1, 1, S, hidden]."""
    # MARK
    seq = x.shape[2]
    if len(x.shape) != 4 or x.shape[0] != 1 or x.shape[1] != 1 or seq % ratio:
        raise ValueError(f"_qsa_attention needs x [1, 1, S, hidden] with S a multiple of {ratio}; got {list(x.shape)}")
    qg = ttnn.reshape(_linear(model, prefix + ".q_proj", x), (1, seq, heads, 2 * hd))
    q = ttnn.permute(ttnn.slice(qg, (0, 0, 0, 0), (1, seq, heads, hd)), (0, 2, 1, 3))
    gate = ttnn.reshape(ttnn.slice(qg, (0, 0, 0, hd), (1, seq, heads, 2 * hd)), (1, 1, seq, heads * hd))
    k = _qsa_norm_heads(model, prefix + ".k_norm", _qsa_split_heads(_linear(model, prefix + ".k_proj", x), kv_heads, hd))
    v = _qsa_split_heads(_linear(model, prefix + ".v_proj", x), kv_heads, hd)
    cos, sin = _qsa_rope_tables(model, torch.arange(seq), hd)
    q = _apply_rotary(model, _qsa_norm_heads(model, prefix + ".q_norm", q), cos, sin)
    k = _apply_rotary(model, k, cos, sin)
    mask = _qsa_selection_mask(model, prefix + ".indexer", x, index_heads, index_dim, ratio, budget)
    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=hd ** -0.5)
    joined = ttnn.reshape(ttnn.permute(out, (0, 2, 1, 3)), (1, 1, seq, heads * hd))
    return _linear(model, prefix + ".o_proj", ttnn.multiply(joined, ttnn.sigmoid(gate)))


'''


def _qsa_helpers() -> list:
    """The helper functions as lines; each function's first comment line is the generated-block mark."""
    mark = f"    {GENERATED_BLOCK_MARK} qsa_attention"
    return [mark if l.strip() == "# MARK" else l for l in _QSA_HELPERS.rstrip("\n").split("\n")] + ["", ""]


def _describe_qsa_attention(function: str, settings: dict) -> str:
    return (f"{function}(model, i, mixed, ids) returns the block output [1, 1, S, {settings['hidden']}] (no residual add; the block's input is the "
            "connection's mixed stream)")


register_block_family(BlockFamily(
    name="qsa_attention", recognize=_recognize_qsa_attention, emit=_emit_qsa_attention, reference_test=QSA_REFERENCE_TEST, describe=_describe_qsa_attention,
    helpers=_qsa_helpers, needs_rotary=True))


# A layer of the Qwen4 family written whole from generated blocks. The reference forward (`Qwen4ExpTextDecoderLayer.forward`, no cache):
#   if the layer has a PLE block:  x = x + ple(x, ids)
#   mixed, hyper_input, w = attn_hyper_connection(x);  x = hyper_input + inject(linear_attn(mixed), w)
#   mixed, hyper_input, w = mlp_hyper_connection(x);   x = hyper_input + inject(mlp(mixed), w)
# The layer is composed only when every part is a generated block of the expected kind (a layer with a stub part stays a stub). Stage 0 flags the layer with
# `aten.flatten.using_ints` (the reference's `injection.flatten(-2)`, which `_hyper_connection_inject` writes with a concat); the inventory records one op per
# module and lists few modules per op, so that one flag is accepted and any other flag on the layer keeps it a stub. The reference test of the layer family is
# the check that the composition is right.
LAYER_COMPOSITION_TEST = "tests/test_port_template_layer.py::test_generated_layers_match_the_transformers_decoder_layer"
_COMPOSABLE_FLAG = "aten.flatten.using_ints (candidate)"


def _compose_layer(layer: Layer, attrs: dict, flag: str | None):
    """The composition settings of a layer built from generated parts, or None. The sublayer is the gated delta net of a linear-attention layer, or the
    attention block of an indexed-attention layer, which may still be a stub: the layer then calls the stub, and the stub is the only function the coder writes."""
    sub = "linear_attn" if "linear_attn" in layer.blocks else "self_attn"
    kids = set(layer.blocks)
    base = {"attn_hyper_connection", "mlp_hyper_connection", "mlp", sub}
    if kids not in (base, base | {"ple"}) or (flag is not None and flag != _COMPOSABLE_FLAG):
        return None
    kinds = {k: v[0] for k, v in layer.blocks.items()}
    want = {"attn_hyper_connection": "hyper_connection", "mlp_hyper_connection": "hyper_connection", "mlp": "mlp", "linear_attn": "gated_delta_net",
            "self_attn": ("stub", "qsa_attention"), "ple": "ple"}
    layer_type = attrs.get("layer_type")
    if any(kinds[k] not in (want[k] if isinstance(want[k], tuple) else (want[k],)) for k in kids) or layer_type != ("linear_attention" if sub == "linear_attn" else "indexed_attention"):
        return None
    a, m = layer.blocks["attn_hyper_connection"][1], layer.blocks["mlp_hyper_connection"][1]
    if not (a.get("inject") and m.get("inject")) or (a["hc"], a["hidden"]) != (m["hc"], m["hidden"]):
        return None          # both connections must inject, over the same streams
    if "ple" in kids and layer.blocks["ple"][1]["hc"] * layer.blocks["ple"][1]["hidden"] != a["hc"] * a["hidden"]:
        return None
    return {"hc": a["hc"], "hidden": a["hidden"], "ple": "ple" in kids, "sublayer": sub, "stub_sublayer": kinds[sub] == "stub"}


def _emit_composed_layer(layer: Layer, layers_path: str) -> list:
    c, key = layer.composed, layer.key
    lines = [f"def _layer_{key}(model, i, x, ids, cos, sin):",
             f'    """Layers {layer.indices}: {layer.cls}, written whole from generated blocks: '
             + ("the PLE block adds to the streams, then " if c["ple"] else "")
             + f'attention connection, {"the attention block (a stub the coder writes)" if c["stub_sublayer"] else ("gated delta net" if c["sublayer"] == "linear_attn" else "the indexed attention")}, MLP connection and MLP, each followed by its injection into the {c["hc"]} streams."""',
             f"    {GENERATED_BLOCK_MARK} layer",
             f'    p = f"{layers_path}.{{i}}"']
    if c["ple"]:
        lines.append(f"    x = ttnn.add(x, _{key}_ple(model, i, x, ids))")
    lines += [f"    mixed, hyper_input, weights = _{key}_attn_hyper_connection(model, i, x, ids)",
              f"    out = _{key}_{c['sublayer']}(model, i, mixed, ids)",
              f"    x = _hyper_connection_inject(out, hyper_input, weights, {c['hc']}, {c['hidden']})",
              f"    mixed, hyper_input, weights = _{key}_mlp_hyper_connection(model, i, x, ids)",
              '    out = _mlp(model, p + ".mlp", mixed)',
              f"    return _hyper_connection_inject(out, hyper_input, weights, {c['hc']}, {c['hidden']})", "", ""]
    return lines


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
        w = by_pattern.get(path + ".weight")
        cls, attrs = con.resolve(path, width=w.shape[0] if w is not None and len(w.shape) == 1 else None)
        place(path, cls, attrs)
    for p, t in by_pattern.items():
        placements.setdefault(p, "tile" if t.dtype in FLOAT_DTYPES else "host")
    if tied:
        placements["lm_head.weight"] = "linear_weight"      # uploaded from the embedding table in load
    # The final norm or the model-level stubs.
    final_norm = None
    site_rotary, _why = _rotary_plan(section, hidden)       # also computed at the end for the plan's note; a family that needs it is shown the same settings
    extra = {k: v for k, v in text_kids.items() if k not in MODEL_CHILDREN}
    norm_path = f"{text_path}.norm"
    if norm_path + ".weight" in by_pattern and placements.get(norm_path + ".weight", "").startswith("norm"):
        final_norm = (norm_path, norm_eps[norm_path])
    elif norm_path + ".weight" in by_pattern:
        cls, _ = con.resolve(norm_path)
        extra["norm"] = cls or "?"
    model_blocks: dict = {}
    for k, v in sorted(extra.items()):
        mp = f"{text_path}.{k}"
        mcls, mattrs = con.resolve(mp)
        hit = _recognize_family(Site(mp, mcls or v, mattrs, con.children(mattrs), by_pattern, con, frozenset(plus_one), "model", site_rotary))
        if hit:
            placements.update(hit[1].get("placements", {}))
            model_blocks[k] = (hit[0].name, {kk: vv for kk, vv in hit[1].items() if kk != "placements"})
            del extra[k]
    prelude_family = next(((name, st) for name, st in model_blocks.values() if BLOCK_FAMILIES[name].prelude), None)
    prelude_reason = None
    if extra:
        prelude_reason = (f"{text_cls or 'the text model'} has children the generated flow does not know: "
                          + ", ".join(f"{k} ({v})" for k, v in sorted(extra.items())))
    if text_path in flagged:          # an op labelled candidate or host in the text model's own forward, between the blocks; a flag under a child goes to that child's block
        flag = f"stage 0 labelled {flagged[text_path]} in {text_path} itself"
        if prelude_family and not prelude_reason:
            # The generated prelude only repeats the embedding into the streams. What the flag marks (masks and the position inputs of the PLE and
            # linear-attention blocks) is read by those blocks, which are stubs or take `ids`, so the flag stays on record as a note.
            notes.append(flag + "; the generated prelude only repeats the embedding into the residual streams, and the blocks that read masks or token ids are stubs of their own")
        else:
            prelude_reason = (prelude_reason + "; " if prelude_reason else "") + flag
    if prelude_reason:
        prelude_family = None        # a stub writes the step before the first layer, and it also repeats the embedding into the streams
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
            elif hit := _recognize_family(Site(kp, kcls, kattrs, kkids, by_pattern, con, frozenset(plus_one), "layer", site_rotary)):          # a flag under a family block does not stub it: the family writes its own ops and its reference test is the check
                fam, settings = hit
                placements.update(settings.get("placements", {}))
                layer.blocks[kid] = (fam.name, {k: v for k, v in settings.items() if k != "placements"})
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
        layer.composed = None if standard else _compose_layer(layer, attrs, flagged.get(lp))
        if not standard:
            parts = [f"{k} ({children[k]})" for k, v in sorted(layer.blocks.items()) if v[0] == "stub"]
            layer.reason = (own_flag or (f"{cls or 'the layer class'}'s children are not the standard pre-norm set (input_layernorm, self_attn, "
                            "post_attention_layernorm, mlp)" if set(kids) != STANDARD_LAYER else "a child block is a stub")) \
                + ("; stubs: " + ", ".join(parts) if parts else "")
            if layer.composed is None:
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
    rotary = None
    uses_rotary = any(v[0] in BLOCK_FAMILIES and BLOCK_FAMILIES[v[0]].needs_rotary for l in layers for v in l.blocks.values())
    if _has_attention(attention_settings, stubs) or uses_rotary:
        rotary, why = _rotary_plan(section, hidden)
        if rotary is None:
            notes.append(f"no rotary helpers (_rope_tables, _apply_rotary) are written: {why}; an attention block has to build its own rotary tables")
    return Plan(model_class=con.model_class, text_path=text_path, layers_path=layers_path, n_layers=n_layers, hidden=hidden, vocab=vocab,
                pad_id=pad if type(pad) is int and 0 <= pad < vocab else 0, config=small, embed_path=embed_path, final_norm=final_norm,
                lm_head_path=lm_head_path, tied=tied, prelude_reason=prelude_reason, epilogue=epilogue, layers=layers, placements=placements,
                norm_eps=norm_eps, plus_one_norms=plus_one, attention=attention_settings, stubs=stubs, notes=notes,
                model_blocks=model_blocks, prelude_family=prelude_family, rotary=rotary)


def _has_attention(attention_settings, stubs) -> bool:
    """True when the plan writes an attention block or leaves an attention stub (a stub whose class name holds `Attention` or whose function name
    holds `attn`): the rotary helpers are written only then."""
    return attention_settings is not None or any("Attention" in str(x.cls) or "attn" in x.function for x in stubs)


def _rope_dict(section: dict) -> dict:
    rope = section.get("rope_parameters") if isinstance(section.get("rope_parameters"), dict) else section.get("rope_scaling")
    return rope if isinstance(rope, dict) else {}


def _rotary_plan(section: dict, hidden: int):
    """({head_dim, rot, theta}, None) when the config gives the settings of the default rotary embedding, else (None, why). `rot` is the number of
    columns of each head that rotate: int(head_dim * partial_rotary_factor), the way the reference computes it. The factor is read from
    `rope_parameters` first and from the config section second (the transformers config moves it into `rope_parameters`)."""
    heads = _whole(section, "num_attention_heads")
    head_dim = _whole(section, "head_dim") or (hidden // heads if heads and hidden else None)
    if not head_dim:
        return None, "the config gives neither head_dim nor num_attention_heads"
    rope = _rope_dict(section)
    rope_type = rope.get("rope_type", rope.get("type", "default")) if rope else "default"
    if rope_type != "default":
        return None, f"rope_type is {rope_type!r}, not the default"
    factor = _number(rope.get("partial_rotary_factor", section.get("partial_rotary_factor", 1.0)))
    if factor is None or not 0 < factor <= 1:
        return None, "partial_rotary_factor is not a number above 0 and at most 1"
    rot = int(head_dim * factor)
    if rot < 2 or rot % 2:
        return None, f"the rotary width int({head_dim} * {factor}) is {rot}, which is not an even number of at least 2"
    theta = _number(rope.get("rope_theta")) or _number(section.get("rope_theta")) or 10000.0
    return {"head_dim": head_dim, "rot": rot, "theta": float(theta)}, None


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
    factor = _number(_rope_dict(section).get("partial_rotary_factor", section.get("partial_rotary_factor")))
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
          "    eps = NORM_EPS.get(_pattern(prefix))",
          "    if eps is None:",
          '        raise ValueError(f"{prefix} is not a plain RMS norm (NORM_EPS has no entry for it): the function that calls _rms_norm must compute this norm itself")',
          '    return ttnn.rms_norm(x, epsilon=eps, weight=model["dev"][prefix + ".weight"])', "", "",
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
    if p.rotary:
        L += [f"ROTARY = {_lit(p.rotary)}", "", ""]
        if not p.attention:
            L += ["def _rope_tables(model, n):",
                  '    """cos and sin for positions 0..n-1, [1, 1, n, head_dim] TILE bf16 on the device. The first ROTARY["rot"] columns hold cos and sin of the position times the inverse frequencies (repeated twice, the HF layout); the other columns hold 1 (cos) and 0 (sin), so that they pass through _apply_rotary."""',
                  '    hd, rot = ROTARY["head_dim"], ROTARY["rot"]',
                  '    inv = 1.0 / (ROTARY["theta"] ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))',
                  "    freqs = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]",
                  "    emb = torch.cat([freqs, freqs], dim=-1)",
                  "    cos = torch.cat([emb.cos(), torch.ones(n, hd - rot)], dim=-1).reshape(1, 1, n, hd)",
                  "    sin = torch.cat([emb.sin(), torch.zeros(n, hd - rot)], dim=-1).reshape(1, 1, n, hd)",
                  "    return _upload(model, cos.to(torch.bfloat16), ttnn.TILE_LAYOUT), _upload(model, sin.to(torch.bfloat16), ttnn.TILE_LAYOUT)", "", ""]
        L += ["def _apply_rotary(model, x, cos, sin):",
              '    """Rotary position embedding on x [1, heads, S, head_dim] with the tables of _rope_tables (cos, sin [1, 1, S, head_dim]): the first ROTARY["rot"] columns of each head are rotated (rotate-half), the others pass through. Computed as x * cos + (x @ R) * sin with R the constant rotate-half matrix, so it needs no slice and no concat. Returns [1, heads, S, head_dim]."""',
              '    hd, rot = ROTARY["head_dim"], ROTARY["rot"]',
              "    if len(x.shape) != 4 or x.shape[0] != 1 or x.shape[3] != hd or tuple(cos.shape) != (1, 1, x.shape[2], hd) or tuple(sin.shape) != tuple(cos.shape):",
              '        raise ValueError(f"_apply_rotary needs x [1, heads, S, {hd}] and tables [1, 1, S, {hd}] (from _rope_tables(model, S)); got x {list(x.shape)}, cos {list(cos.shape)}")',
              "    heads, seq = x.shape[1], x.shape[2]",
              "    half = rot // 2",
              "    rot_m = torch.zeros(hd, hd)",
              "    for i in range(half):",
              "        rot_m[i, i + half] = 1.0",
              "        rot_m[i + half, i] = -1.0",
              "    r = _upload(model, rot_m.to(torch.bfloat16), ttnn.TILE_LAYOUT)",
              "    flat = ttnn.reshape(x, (1, 1, heads * seq, hd))",
              '    rotated = ttnn.linear(flat, r, dtype=ttnn.bfloat16, compute_kernel_config=model["ck"], memory_config=ttnn.DRAM_MEMORY_CONFIG)',
              "    rotated = ttnn.reshape(rotated, (1, heads, seq, hd))",
              "    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))", "", ""]
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
    used = [name for name in BLOCK_FAMILIES
            if any(b[0] == name for l in p.layers for b in l.blocks.values()) or any(b[0] == name for b in p.model_blocks.values())
            or (p.prelude_family and p.prelude_family[0] == name)]
    for name in used:
        if BLOCK_FAMILIES[name].helpers:
            L += BLOCK_FAMILIES[name].helpers()
    if p.prelude_reason:
        L += _stub_function("_model_prelude", "model, x, ids", stubs["_model_prelude"],
                            [f"The reference's forward may change the embedding before the first layer; x is [1, 1, S, HIDDEN] TILE bf16 from _embedding."])
    elif p.prelude_family:
        L += BLOCK_FAMILIES[p.prelude_family[0]].prelude(p.prelude_family[1])
    for k, v in p.epilogue:
        L += _stub_function(f"_model_{k}", "model, x", stubs[f"_model_{k}"])
    for k, (name, settings) in sorted(p.model_blocks.items()):
        L += BLOCK_FAMILIES[name].emit(f"_model_{k}", f"{p.text_path}.{k}", settings, "model")
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
                                    [f"The Linear children of this block are uploaded by load() and can be called as _linear(model, f\"{lp}.{kid}.<name>\", x)."]
                                    + ([f"The generated _layer_{layer.key} calls this function on the connection's mixed input x [1, 1, S, {layer.composed['hidden']}] with the padded token ids; "
                                        f"return the block output [1, 1, S, {layer.composed['hidden']}]. The layer adds it to the residual streams through the injection, so add no residual here."]
                                       if layer.composed else []))
            elif kind in BLOCK_FAMILIES:
                L += BLOCK_FAMILIES[kind].emit(f"_{layer.key}_{kid}", f"{lp}.{kid}", details, "layer")
        if layer.composed:
            L += _emit_composed_layer(layer, p.layers_path)
            continue
        parts = []
        for kid, (kind, details) in sorted(layer.blocks.items()):
            call = {"mlp": f'_mlp(model, p + ".{kid}", x)', "rms_norm": f'_rms_norm(model, p + ".{kid}", x)',
                    "attention": f'_attention(model, p + ".{kid}", x, cos, sin)', "linear": f'_linear(model, p + ".{kid}", x)',
                    "stub": f"_{layer.key}_{kid}(model, i, x, ids)"}.get(kind) or BLOCK_FAMILIES[kind].describe(f"_{layer.key}_{kid}", details)
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
          '        elif role == "neg_exp":',
          "            dev[name] = _upload(model, (-t.float().exp()).to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)",
          '        elif role == "conv_taps":',
          "            for j in range(t.shape[-1]):",
          '                dev[f"{name}.tap{j}"] = _upload(model, t[:, 0, j].to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)',
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
    if p.prelude_reason or p.prelude_family:
        L.append("    x = _model_prelude(model, x, padded)")
    L += ["    for i in range(N_LAYERS):",
          "        x = LAYERS[LAYER_VARIANT[i]](model, i, x, padded, cos, sin)"]
    if p.final_norm:
        L.append(f'    x = _rms_norm(model, "{p.final_norm[0]}", x)')
    for k in sorted({k for k, _ in p.epilogue} | set(p.model_blocks)):
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


# The functions every generated port holds with the same text. A failure inside one is never sent to the coder as a repair target: the stub that
# called it passed it something it cannot take (arm E8 spent 11 repair rounds on `_rms_norm`; issue 698).
FIXED_HELPERS = ("read_tensors", "_pattern", "_n_devices", "_upload", "_upload_ids", "_to_host", "_linear", "_rms_norm", "_embedding", "_rope_tables", "_apply_rotary")


# What the generated port's `model` argument and module constants are. The stub prompt (orchard/stub_prompts.py `model_text`) prints these, and
# tests/test_thin_port_e2.py checks them against the generated text, so the prompt names exactly the keys `load()` sets and the constants the
# port defines (arm E's stubs wrote `model.config`, `model.rotary_emb` and `model.past_key_values`, which do not exist).
MODEL_KEYS = {
    "mesh_device": "the ttnn mesh device",
    "n_devices": "the number of devices (an int)",
    "dev": "dict: tensor name -> ttnn tensor, as load placed it (read a weight with model['dev'][name + '.weight'])",
    "host": "dict: tensor name -> torch tensor, for the tensors that stay on the host (integer tables)",
    "ids": "list of the token ids of the current request (prefill and decode_step set it)",
    "ck": "the compute kernel config to pass to ttnn.linear",
}
MODULE_CONSTANTS = {
    "CONFIG": "the model config as a dict: CONFIG['head_dim'], CONFIG['hidden_size']; there is no config attribute anywhere",
    "N_LAYERS": "the number of decoder layers (an int)",
    "HIDDEN": "the hidden size (an int)",
    "VOCAB": "the vocabulary size (an int)",
    "PAD_ID": "the token id used to pad the prompt to a multiple of PAD_TO",
    "PAD_TO": "the tile multiple the sequence is padded to (32)",
    "LAYER_VARIANT": "list: layer index -> the key of LAYERS for that layer",
    "PLACEMENT": "dict: tensor name pattern -> how load placed it",
    "NORM_EPS": "dict: norm name pattern -> epsilon",
    "ROTARY": "dict: head_dim, rot (the number of columns of each head that rotate) and theta, for _rope_tables and _apply_rotary",
    "LAYERS": "dict: layer variant key -> the layer function",
}


# ---- the CPU check -----------------------------------------------------------------------------------------------------------------------

STUB_RAISE_TAIL = " is not written yet"          # the end of the message every generated stub raises (`_stub_function`)


def _stub_raise_text(node) -> str | None:
    """The message of `raise NotImplementedError("... is not written yet")`, the generator's own stub body, else None."""
    exc = node.exc if isinstance(node, ast.Raise) else None
    if not (isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError" and exc.args):
        return None
    arg = exc.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.endswith(STUB_RAISE_TAIL):
        return arg.value
    return None


def stub_lines(source: str) -> list[tuple]:
    """(line number, text) of every stub: a function that holds a comment line starting with the stub marker, or that raises the generator's
    own `NotImplementedError("... is not written yet")`. A function is counted once however many of these signs it holds, so deleting the
    marker comment does not remove the stub (review of PR 685). A string that holds the marker is not a stub. A source that does not parse is
    judged by its marker comments alone (the parse error is reported by the check)."""
    marks = [(n, line.strip()[:160]) for n, line in enumerate(source.split("\n"), 1) if line.strip().startswith(STUB_MARK)]
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return marks
    spans = [(node.lineno, node.end_lineno, node.name) for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    def owner(n: int):
        for first, last, name in spans:
            if first <= n <= last:
                return name
        return ("line", n)

    found: dict = {}
    for n, text in marks:
        found.setdefault(owner(n), (n, text))
    for node in ast.walk(tree):
        text = _stub_raise_text(node)
        if text is not None:
            found.setdefault(owner(node.lineno), (node.lineno, ("raise NotImplementedError: " + text)[:160]))
    return sorted(found.values())


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
