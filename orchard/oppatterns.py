# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Recognise groups of aten ops that one ttnn op does in a single call.

torch.export splits RMSNorm into pow, mean, add, rsqrt, mul and a weight multiply, and each of those is mapped alone.
Nothing in the per-op table says the group is the native `ttnn.rms_norm`. This module reads the ordered op list the
trace template records (`sequence`: id, op, ids of the nodes each op reads, module, shape, constants) and finds the
groups in the table below. A recognised group is labelled `pattern`:

    rms_norm     pow(x, 2) -> mean(-1) -> add(eps) -> rsqrt -> mul(x, .) -> mul(weight)
    layer_norm   the same with x centred first (sub(x, mean(x)) or var), and an optional bias add
    silu_mul     silu(a) -> mul(., b)                                   ttnn.swiglu takes a and b joined on the last dim
    sdpa         matmul -> [mul or div] -> [add mask] -> softmax -> matmul   (an unfused attention)
    rotary       add(mul(x, cos), mul(cat(neg(x_hi), x_lo), sin))        the Hugging Face rotate_half form

A group needs its ttnn op to exist in the dump; when none of the listed names does, the group is reported under
`unavailable` and nothing is claimed. A recognised group says the shapes of ops match a known composite. It does not
say the ttnn op takes this model's shapes, dtype or layout, and `pattern` never makes the member ops `mapped`.
`tt-orchard inventory verify-candidates` runs the ttnn op on small tensors.

Standard library only.
"""
from __future__ import annotations

import re

# the ttnn names that implement each group, in order of preference
PATTERN_TTNN = {
    "rms_norm": ["rms_norm"],
    "layer_norm": ["layer_norm"],
    "silu_mul": ["swiglu"],
    "sdpa": ["transformer.scaled_dot_product_attention"],
    "rotary": ["experimental.rotary_embedding_hf", "experimental.rotary_embedding_llama",
               "experimental.rotary_embedding"],
}
CAST = frozenset({"to", "_to_copy", "type_as", "contiguous", "alias", "clone", "detach"})
SHAPE = CAST | {"view", "reshape", "unsqueeze", "expand", "transpose", "permute", "squeeze", "_unsafe_view"}
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.\d+(?:\.|$)")
MAX_EPS = 1e-2
MODULES_SHOWN = 3


def base(op: str) -> str:
    """`aten.mul.Tensor` -> `mul`."""
    s = op[5:] if op.startswith("aten.") else op
    return s.split(".", 1)[0]


class Graph:
    def __init__(self, sequence):
        self.nodes = [n for n in sequence if isinstance(n, dict) and isinstance(n.get("id"), str)
                      and isinstance(n.get("op"), str)]
        self.by_id = {n["id"]: n for n in self.nodes}
        self.cons: dict = {}
        for n in self.nodes:
            for i in n.get("in") or []:
                self.cons.setdefault(i, []).append(n)

    def node(self, ident, passes=CAST):
        """The node `ident` names after skipping pass-through ops (casts by default), or None for a weight or an input."""
        n = self.by_id.get(ident)
        seen = 0
        while n is not None and base(n["op"]) in passes and n.get("in") and seen < 16:
            n = self.by_id.get(n["in"][0])
            seen += 1
        return n

    def resolve(self, ident, passes=CAST):
        """The id after skipping pass-through ops (a weight or input keeps its own id)."""
        n = self.by_id.get(ident)
        seen = 0
        while n is not None and base(n["op"]) in passes and n.get("in") and seen < 16:
            ident = n["in"][0]
            n = self.by_id.get(ident)
            seen += 1
        return ident

    def readers(self, ident, passes=CAST):
        """Nodes that read `ident`, looking through pass-through readers."""
        out, todo, seen = [], list(self.cons.get(ident, [])), set()
        while todo:
            n = todo.pop(0)
            if n["id"] in seen:
                continue
            seen.add(n["id"])
            if base(n["op"]) in passes:
                todo += self.cons.get(n["id"], [])
            else:
                out.append(n)
        return out

    def others(self, n, ident, passes=CAST):
        """The inputs of `n` that do not resolve to `ident`."""
        return [i for i in n.get("in") or [] if self.resolve(i, passes) != ident]


def _consts(n):
    return [c for c in n.get("const") or []]


def _has_dim_last(n):
    for c in _consts(n):
        if c == -1 or (isinstance(c, list) and -1 in c):
            return True
    return False


def _is_eps(n):
    return any(isinstance(c, float) and 0 < c <= MAX_EPS for c in _consts(n))


def _is_square(g, n):
    b = base(n["op"])
    if b == "square":
        return True
    if b == "pow":
        return any(c in (2, 2.0) for c in _consts(n) if not isinstance(c, list))
    if b == "mul":
        ins = [g.resolve(i) for i in n.get("in") or []]
        return len(ins) == 2 and ins[0] == ins[1]
    return False


def _norms(g):
    """RMSNorm and LayerNorm groups: [(kind, anchor, node ids)]."""
    found = []
    for r in g.nodes:
        if base(r["op"]) != "rsqrt" or not r.get("in"):
            continue
        a = g.node(r["in"][0])
        if a is None or base(a["op"]) != "add" or not _is_eps(a) or not a.get("in"):
            continue
        m = g.node(a["in"][0])
        if m is None or not m.get("in"):
            continue
        ids = [r["id"], a["id"], m["id"]]
        target, kind = None, None
        if base(m["op"]) == "mean" and _has_dim_last(m):
            p = g.node(m["in"][0])
            if p is None or not _is_square(g, p) or not p.get("in"):
                continue
            ids.append(p["id"])
            x = g.resolve(p["in"][0])
            xn = g.by_id.get(x)
            if xn is not None and base(xn["op"]) == "sub" and len(xn.get("in") or []) == 2:
                y_id, mu_id = xn["in"]
                mu = g.node(mu_id)
                if (mu is not None and base(mu["op"]) == "mean" and _has_dim_last(mu) and mu.get("in")
                        and g.resolve(mu["in"][0]) == g.resolve(y_id)):
                    ids += [xn["id"], mu["id"]]           # x is centred first: that is a LayerNorm
                    target, kind = x, "layer_norm"
            if kind is None:
                target, kind = x, "rms_norm"
        elif base(m["op"]) == "var" and _has_dim_last(m):
            x = g.resolve(m["in"][0])
            ids.append(m["id"])
            kind = "layer_norm"
            target = None
            for c in g.readers(r["id"]):
                if base(c["op"]) != "mul":
                    continue
                for o in g.others(c, r["id"]):
                    on = g.node(o)
                    if on is not None and base(on["op"]) == "sub" and any(g.resolve(i) == x for i in on.get("in") or []):
                        target = g.resolve(o)
                        ids.append(on["id"])
                        break
        else:
            continue
        if target is None:
            continue
        norm = None
        for c in g.readers(r["id"]):
            if base(c["op"]) == "mul" and any(g.resolve(o) == target for o in g.others(c, r["id"])):
                norm = c
                break
        if norm is None:
            continue
        ids.append(norm["id"])
        scale = [c for c in g.readers(norm["id"]) if base(c["op"]) == "mul"]
        if scale:
            ids.append(scale[0]["id"])
            if kind == "layer_norm":
                bias = [c for c in g.readers(scale[0]["id"]) if base(c["op"]) == "add"]
                if bias:
                    ids.append(bias[0]["id"])
        found.append((kind, r, sorted(set(ids))))
    return found


def _silu_mul(g):
    found = []
    for s in g.nodes:
        if base(s["op"]) != "silu":
            continue
        for c in g.readers(s["id"]):
            if base(c["op"]) == "mul" and g.others(c, s["id"]):
                found.append(("silu_mul", s, [s["id"], c["id"]]))
                break
    return found


MASK_OPS = frozenset({"add", "masked_fill", "where"})
SCALE_OPS = frozenset({"mul", "div"})


def _sdpa(g):
    found = []
    for sm in g.nodes:
        if base(sm["op"]) not in ("softmax", "_softmax") or not sm.get("in"):
            continue
        ids, qk, todo, depth = [sm["id"]], None, [(sm["in"][0], 0)], 0
        while todo:
            ident, d = todo.pop(0)
            n = g.by_id.get(ident)
            if n is None or d > 6:
                continue
            b = base(n["op"])
            if b in SHAPE:
                todo += [(i, d + 1) for i in n.get("in") or []][:1]
            elif b in ("matmul", "bmm", "mm"):
                qk = n
                ids.append(n["id"])
                break
            elif b in MASK_OPS | SCALE_OPS:
                ids.append(n["id"])
                todo += [(i, d + 1) for i in n.get("in") or []]
        if qk is None:
            continue
        av = None
        for c in g.readers(sm["id"], SHAPE | {"dropout"}):
            if base(c["op"]) in ("matmul", "bmm", "mm"):
                av = c
                break
        if av is None:
            continue
        found.append(("sdpa", sm, sorted(set(ids + [av["id"]]))))
    return found


def _rotary(g):
    found = []
    for ad in g.nodes:
        if base(ad["op"]) != "add" or len(ad.get("in") or []) != 2:
            continue
        muls = [g.node(i) for i in ad["in"]]
        if any(m is None or base(m["op"]) != "mul" for m in muls):
            continue
        for rot, plain in (muls, muls[::-1]):
            cat = next((g.node(i) for i in rot.get("in") or [] if g.node(i) is not None and base(g.node(i)["op"]) == "cat"), None)
            if cat is None or not _has_dim_last(cat) or len(cat.get("in") or []) != 2:
                continue
            neg, lo = g.node(cat["in"][0]), g.node(cat["in"][1])
            if neg is None or lo is None or base(neg["op"]) != "neg" or base(lo["op"]) != "slice" or not neg.get("in"):
                continue
            hi = g.node(neg["in"][0])
            if hi is None or base(hi["op"]) != "slice" or not hi.get("in") or not lo.get("in"):
                continue
            x = g.resolve(hi["in"][0])
            if x != g.resolve(lo["in"][0]) or not any(g.resolve(i) == x for i in plain.get("in") or []):
                continue
            lc, hc = _consts(lo), _consts(hi)
            # slice(dim, start, end): the low half ends where the high half starts, on the same dimension
            if len(lc) < 3 or len(hc) < 3 or lc[0] != hc[0] or lc[2] != hc[1]:
                continue
            found.append(("rotary", ad, sorted({ad["id"], rot["id"], plain["id"], cat["id"], neg["id"], lo["id"], hi["id"]})))
            break
    return found


def recognise(sequence, names, layers_in_config=None) -> dict:
    """The groups found in `sequence`, those whose ttnn op is in `names`, and those whose ttnn op is not.

    {"found": [{pattern, ttnn, why: "pattern", occurrences, in_layer, outside, scaled_count, nodes, ops, modules}],
     "unavailable": [{pattern, wanted, occurrences}]}. `nodes` is the number of ops in one group, `ops` the aten ops of
    the first group found, `shape` the output shape of that group's anchor op, `eps` a norm's epsilon (else None),
    `in_layer` the groups inside the first layer (the only layer the sequence holds). A group
    inside the layers counts `layers_in_config` times in `scaled_count`; the number is an estimate."""
    g = Graph(sequence or [])
    hits = _norms(g) + _silu_mul(g) + _sdpa(g) + _rotary(g)
    names = set(names or ())
    by_kind: dict = {}
    for kind, anchor, ids in hits:
        by_kind.setdefault(kind, []).append((anchor, ids))
    found, unavailable = [], []
    for kind in PATTERN_TTNN:
        if kind not in by_kind:
            continue
        occ = by_kind[kind]
        ttnn = next((n for n in PATTERN_TTNN[kind] if n in names), None)
        if ttnn is None:
            unavailable.append({"pattern": kind, "wanted": list(PATTERN_TTNN[kind]), "occurrences": len(occ)})
            continue
        in_layer = sum(1 for a, _ in occ if LAYER_RE.search(a.get("module") or ""))
        outside = len(occ) - in_layer
        scale = layers_in_config if isinstance(layers_in_config, int) and layers_in_config > 0 else 1
        mods = []
        for a, _ in occ:
            m = a.get("module") or ""
            if m not in mods and len(mods) < MODULES_SHOWN:
                mods.append(m)
        first = occ[0][1]
        eps = next((c for i in first if base(g.by_id[i]["op"]) == "add" and _is_eps(g.by_id[i])
                    for c in _consts(g.by_id[i]) if isinstance(c, float) and 0 < c <= MAX_EPS), None)
        found.append({"pattern": kind, "ttnn": ttnn, "why": "pattern", "occurrences": len(occ),
                      "in_layer": in_layer, "outside": outside, "scaled_count": in_layer * scale + outside,
                      "nodes": len(first), "ops": sorted({g.by_id[i]["op"] for i in first}), "modules": mods,
                      "shape": occ[0][0].get("shape"), "eps": eps})
    return {"found": found, "unavailable": unavailable}
