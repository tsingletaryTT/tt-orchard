#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run candidate ttnn ops against torch references inside the ttsim Blackhole simulator image.

`tt-orchard inventory verify-candidates` starts this script in the operator's simulator image (torch and a ttnn wheel
are installed there) and reads one result line from stdout:

    python verify_in_sim.py cases.json

cases.json is {"cases": [{"id", "kind": "op" | "pattern", "aten", "ttnn", "pattern", "rows", "cols", "dtype", "const"}]}.
For each case the script builds small tensors (rows x cols, both multiples of 32, from a fixed seed), runs the torch
reference and the ttnn op, and compares them. The result line is `ORCHARD_VERIFY ` followed by JSON:

    {"ttnn_version", "torch_version", "bars", "results": [{"id", "outcome", "reason", "pcc", "rel_l2", "slope",
                                                          "max_abs", "exact", "seconds"}]}

`outcome` is `pass`, `fail` (it ran and the numbers are outside the bars), `error` (the call raised: the text is in
`reason`) or `no_reference` (this script has no torch reference for the pair, so nothing was tested). What a `pass`
means is stated in orchard/opverify.py: one small shape, the simulator, one seed.

Nothing here touches a chip. It runs under TT_METAL_SIMULATOR; the host orchestrator refuses to start it without one.
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback

MARK = "ORCHARD_VERIFY "
BARS = {"float": {"pcc": 0.99, "rel_l2": 5e-2, "slope": [0.95, 1.05]}, "exact": {"equal": True}}
SEED = 0


_DRAWS = [0]


def seeded(shape, kind="normal", lo=0, hi=2):
    """A reproducible tensor. Each call in one case uses the next seed, so two tensors of one shape are never equal
    (a pair of equal operands would make `a & b` and `a | b` the same, and a bar that cannot tell them apart)."""
    import torch
    g = torch.Generator().manual_seed(SEED + _DRAWS[0])
    _DRAWS[0] += 1
    if kind == "int":
        return torch.randint(lo, hi, tuple(shape), generator=g, dtype=torch.int32)
    if kind == "positive":
        return (torch.rand(tuple(shape), generator=g) + 0.25).bfloat16().float()
    return (torch.randn(tuple(shape), generator=g) * 0.5).bfloat16().float()


def stats(ref, out):
    """pcc, relative L2, least-squares slope of out against ref, largest absolute error, exact equality."""
    import torch
    ref, out = ref.double().flatten(), out.double().flatten()
    if ref.numel() != out.numel():
        raise ValueError(f"shape mismatch: reference {tuple(ref.shape)} against output {tuple(out.shape)}")
    if not bool(torch.isfinite(out).all()):
        return {"pcc": None, "rel_l2": None, "slope": None, "max_abs": None, "exact": False, "finite": False}
    rn = float(ref.norm())
    pcc = None
    if ref.std() > 0 and out.std() > 0:
        pcc = float(torch.corrcoef(torch.stack([ref, out]))[0, 1])
    elif torch.equal(ref, out):
        pcc = 1.0
    return {"pcc": pcc, "rel_l2": float((out - ref).norm() / rn) if rn else float(out.norm()),
            "slope": float((out * ref).sum() / (ref * ref).sum()) if rn else None,
            "max_abs": float((out - ref).abs().max()), "exact": bool(torch.equal(ref, out)), "finite": True}


def judge(s, exact):
    """pass or fail against the bars; the reason names the first bar missed."""
    if not s["finite"]:
        return "fail", "the output holds NaN or infinity"
    if exact:
        return ("pass", "") if s["exact"] else ("fail", f"values differ from the reference (max abs error {s['max_abs']:.6g})")
    b = BARS["float"]
    if s["pcc"] is None or s["pcc"] < b["pcc"]:
        return "fail", f"pcc {s['pcc']} is below {b['pcc']}"
    if s["rel_l2"] > b["rel_l2"]:
        return "fail", f"relative L2 error {s['rel_l2']:.4g} is above {b['rel_l2']}"
    if not (b["slope"][0] <= s["slope"] <= b["slope"][1]):
        return "fail", f"slope {s['slope']:.4g} is outside {b['slope']}: the output is about {s['slope']:.3g} times the reference"
    return "pass", ""


class Ctx:
    def __init__(self, dev):
        import torch
        import ttnn
        self.torch, self.ttnn, self.dev = torch, ttnn, dev

    def put(self, t, dtype=None, layout=None):
        ttnn = self.ttnn
        dtype = dtype or ttnn.bfloat16
        return ttnn.from_torch(t.to({ttnn.bfloat16: self.torch.bfloat16, ttnn.int32: self.torch.int32,
                                     ttnn.float32: self.torch.float32, ttnn.uint32: self.torch.int32}[dtype]),
                               dtype=dtype, layout=layout or ttnn.TILE_LAYOUT, device=self.dev)

    def get(self, t):
        return self.ttnn.to_torch(t).float()

    def fn(self, name):
        obj = self.ttnn
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj


UNARY_POSITIVE = {"rsqrt", "sqrt", "log", "log1p", "reciprocal"}
UNARY_FLOAT = {"exp2", "cos", "sin", "rsqrt", "sqrt", "erf", "gelu", "silu", "sigmoid", "expm1", "log1p", "softplus", "neg",
               "abs", "tanh", "relu", "exp", "square"}


def short(name):
    return name.rsplit(".", 1)[-1]


def base_of(aten):
    s = aten[5:] if aten.startswith("aten.") else aten
    b = s.split(".", 1)[0]
    return b[:-1] if b.endswith("_") and not b.startswith("__") else b


def a_unary(c, case):
    base, torch = base_of(case["aten"]), c.torch
    x = seeded((case["rows"], case["cols"]), "positive" if base in UNARY_POSITIVE else "normal")
    ref = getattr(torch, base)(x) if hasattr(torch, base) else getattr(torch.nn.functional, base)(x)
    return ref, c.get(c.fn(case["ttnn"])(c.put(x))), False


def a_bitwise_binary(c, case):
    """`__and__` and friends on masks or integers; the reference is torch's bitwise op."""
    torch = c.torch
    hi = 2 if "bool" in str(case.get("dtype")) else 1 << 12
    a, b = seeded((case["rows"], case["cols"]), "int", 0, hi), seeded((case["rows"], case["cols"]), "int", 0, hi)
    op = {"__and__": torch.bitwise_and, "__or__": torch.bitwise_or, "__xor__": torch.bitwise_xor}[base_of_dunder(case["aten"])]
    out = c.fn(case["ttnn"])(c.put(a, c.ttnn.int32), c.put(b, c.ttnn.int32))
    return op(a, b).float(), c.get(out), True


def base_of_dunder(aten):
    s = aten[5:] if aten.startswith("aten.") else aten
    return s.split(".", 1)[0]


def a_bitwise_not(c, case):
    a = seeded((case["rows"], case["cols"]), "int", 0, 1 << 12)
    out = c.fn(case["ttnn"])(c.put(a, c.ttnn.int32))
    return c.torch.bitwise_not(a).float(), c.get(out), True


def a_div(c, case):
    x, y = seeded((case["rows"], case["cols"])), seeded((case["rows"], case["cols"]), "positive")
    xt, yt = c.put(x), c.put(y)
    out = c.fn(case["ttnn"])(xt, yt)
    if out is None:                      # an in-place op writes into its first argument
        out = xt
    return x / y, c.get(out), False


def a_typecast(c, case):
    x = seeded((case["rows"], case["cols"]))
    out = c.fn(case["ttnn"])(c.put(x, c.ttnn.float32), c.ttnn.bfloat16)
    return x.bfloat16().float(), c.get(out), False


def a_ones(c, case):
    shape = (case["rows"], case["cols"])
    f = c.fn(case["ttnn"])
    if short(case["ttnn"]) == "ones":
        out = f(shape, dtype=c.ttnn.bfloat16, layout=c.ttnn.TILE_LAYOUT, device=c.dev)
    else:
        out = f(shape, 1.0, dtype=c.ttnn.bfloat16, layout=c.ttnn.TILE_LAYOUT, device=c.dev)
    return c.torch.ones(shape), c.get(out), True


def a_zeros(c, case):
    shape = (case["rows"], case["cols"])
    f = c.fn(case["ttnn"])
    if short(case["ttnn"]) == "zeros":
        out = f(shape, dtype=c.ttnn.bfloat16, layout=c.ttnn.TILE_LAYOUT, device=c.dev)
    else:
        out = f(shape, 0.0, dtype=c.ttnn.bfloat16, layout=c.ttnn.TILE_LAYOUT, device=c.dev)
    return c.torch.zeros(shape), c.get(out), True


def a_masked_fill(c, case):
    x = seeded((case["rows"], case["cols"]))
    mask = (seeded((case["rows"], case["cols"])) > 0)
    out = c.fn(case["ttnn"])(c.put(mask.float()), -5.0, c.put(x))
    return x.masked_fill(mask, -5.0), c.get(out), False


def a_split(c, case):
    x = seeded((case["rows"], case["cols"]))
    half = case["cols"] // 2
    f = c.fn(case["ttnn"])
    parts = f(c.put(x), half, -1) if short(case["ttnn"]) == "split" else f(c.put(x), 2, -1)
    got = c.torch.cat([c.get(p).reshape(case["rows"], -1) for p in parts], dim=-1)
    ref = c.torch.cat(list(c.torch.split(x, [half, case["cols"] - half], dim=-1)), dim=-1)
    return ref, got, False


def a_copy(c, case):
    x = seeded((case["rows"], case["cols"]))
    f = c.fn(case["ttnn"])
    xt = c.put(x)
    if short(case["ttnn"]) == "clone":
        out = f(xt)
    else:
        dst = c.put(c.torch.zeros_like(x))
        out = f(xt, dst)
        if out is None:
            out = dst
    return x, c.get(out), False


def a_index(c, case):
    torch, ttnn = c.torch, c.ttnn
    vocab, dim, n = 64, case["cols"], 32
    w = seeded((vocab, dim))
    idx = seeded((1, n), "int", 0, vocab)
    out = c.fn(case["ttnn"])(c.put(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), c.put(w, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT))
    return w[idx.long()].reshape(n, dim), c.get(out).reshape(n, dim), False


def a_to_device(c, case):
    x = seeded((case["rows"], case["cols"]))
    host = c.ttnn.from_torch(x.bfloat16(), dtype=c.ttnn.bfloat16, layout=c.ttnn.TILE_LAYOUT)
    out = c.fn(case["ttnn"])(host, c.dev)
    return x, c.get(out), True


def a_matmul(c, case):
    a, b = seeded((case["rows"], case["cols"])), seeded((case["cols"], case["rows"]))
    return a @ b, c.get(c.fn(case["ttnn"])(c.put(a), c.put(b))), False


def a_clamp(c, case):
    x = seeded((case["rows"], case["cols"]))
    f = c.fn(case["ttnn"])
    return x.clamp(-0.25, 0.25), c.get(f(c.put(x), -0.25, 0.25)), False


def a_softmax(c, case):
    x = seeded((case["rows"], case["cols"]))
    return c.torch.softmax(x, -1), c.get(c.fn(case["ttnn"])(c.put(x), -1)), False


def a_transpose(c, case):
    x = seeded((case["rows"], case["cols"]))
    return x.t(), c.get(c.fn(case["ttnn"])(c.put(x), -2, -1)), False


# (aten base, ttnn short name) -> adapter; (aten base, None) matches every ttnn name. Pairs with no adapter are
# reported `no_reference`: a candidate this script cannot test stays a candidate.
ADAPTERS = {}
for _b in UNARY_FLOAT:
    ADAPTERS[(_b, _b)] = a_unary
for _n in ("bitwise_and", "logical_and"):
    ADAPTERS[("__and__", _n)] = a_bitwise_binary
for _n in ("bitwise_or", "logical_or"):
    ADAPTERS[("__or__", _n)] = a_bitwise_binary
for _n in ("bitwise_xor", "logical_xor"):
    ADAPTERS[("__xor__", _n)] = a_bitwise_binary
ADAPTERS[("bitwise_not", "bitwise_not")] = a_bitwise_not
for _n in ("div", "divide", "div_", "divide_"):
    ADAPTERS[("div", _n)] = a_div
for _b in ("to", "type_as", "_to_copy"):
    ADAPTERS[(_b, "typecast")] = a_typecast
for _n in ("ones", "full"):
    ADAPTERS[("new_ones", _n)] = a_ones
for _n in ("zeros", "full"):
    ADAPTERS[("new_zeros", _n)] = a_zeros
ADAPTERS[("masked_fill", "where")] = a_masked_fill
for _b in ("split_with_sizes", "split"):
    for _n in ("split", "chunk"):
        ADAPTERS[(_b, _n)] = a_split
for _n in ("copy", "assign", "clone"):
    ADAPTERS[("copy", _n)] = a_copy
for _b in ("index", "index_select"):
    ADAPTERS[(_b, "embedding")] = a_index
ADAPTERS[("to", "to_device")] = a_to_device
for _b in ("mm", "bmm", "matmul"):
    ADAPTERS[(_b, "matmul")] = a_matmul
for _b in ("clamp", "clip"):
    for _n in ("clamp", "clip"):
        ADAPTERS[(_b, _n)] = a_clamp
for _b in ("softmax", "_softmax"):
    ADAPTERS[(_b, "softmax")] = a_softmax
ADAPTERS[("t", "transpose")] = a_transpose


# --- groups of ops -------------------------------------------------------------------------------------------------

def eps_of(case, default=1e-6):
    for v in case.get("const") or []:
        if isinstance(v, float) and 0 < v <= 1e-2:
            return v
    return default


def p_rms_norm(c, case):
    torch, ttnn = c.torch, c.ttnn
    rows, cols, eps = case["rows"], case["cols"], eps_of(case)
    x = seeded((rows, cols))
    x[3] *= 20                                                 # one row with a large scale
    w = seeded((cols,))
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    wt = c.put(w.reshape(1, 1, cols // 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.rms_norm(c.put(x), epsilon=eps, weight=wt)
    return ref, c.get(out).reshape(rows, cols), False


def p_layer_norm(c, case):
    torch, ttnn = c.torch, c.ttnn
    rows, cols, eps = case["rows"], case["cols"], eps_of(case, 1e-5)
    x, w, b = seeded((rows, cols)), seeded((cols,)), seeded((cols,))
    ref = torch.nn.functional.layer_norm(x, (cols,), w, b, eps)
    wt = c.put(w.reshape(1, 1, cols // 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    bt = c.put(b.reshape(1, 1, cols // 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.layer_norm(c.put(x), epsilon=eps, weight=wt, bias=bt)
    return ref, c.get(out).reshape(rows, cols), False


def p_silu_mul(c, case):
    torch, ttnn = c.torch, c.ttnn
    rows, cols = case["rows"], case["cols"]
    a, b = seeded((rows, cols)), seeded((rows, cols))
    ref = torch.nn.functional.silu(a) * b
    # ttnn.swiglu splits its input in two on the last dimension and returns first half * silu(second half), so the gate
    # (the silu side) goes second. It needs a 4-D tensor whose last dimension is divisible by 64.
    joined = torch.cat([b, a], dim=-1).reshape(1, 1, rows, 2 * cols)
    out = ttnn.swiglu(c.put(joined), dim=-1)
    return ref, c.get(out).reshape(rows, cols), False


def p_rotary(c, case):
    torch, ttnn = c.torch, c.ttnn
    rows, cols = case["rows"], case["cols"]
    x, cos, sin = seeded((1, 1, rows, cols)), torch.cos(seeded((1, 1, rows, cols))), torch.sin(seeded((1, 1, rows, cols)))
    half = cols // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    ref = x * cos + rot * sin
    out = ttnn.experimental.rotary_embedding_hf(c.put(x), c.put(cos), c.put(sin), is_decode_mode=False)
    return ref.reshape(rows, cols), c.get(out).reshape(rows, cols), False


# (group, ttnn name) -> function. A pair that is not here is `no_reference`: each function calls one fixed ttnn op, so
# it may not stand in for another name (a case naming rotary_embedding_llama must not be answered by the hf op's run).
PATTERNS = {("rms_norm", "rms_norm"): p_rms_norm, ("layer_norm", "layer_norm"): p_layer_norm,
            ("silu_mul", "swiglu"): p_silu_mul, ("rotary", "experimental.rotary_embedding_hf"): p_rotary}


def adapter_for(case):
    if case.get("kind") == "pattern":
        return PATTERNS.get((case.get("pattern"), case.get("ttnn")))
    a = case["aten"]
    base = base_of_dunder(a) if base_of_dunder(a).startswith("__") else base_of(a)
    t = short(case["ttnn"])
    if case.get("as"):                   # a control: look the adapter up as this pair, run the case's own ttnn op
        base, t = case["as"]
    return ADAPTERS.get((base, t)) or ADAPTERS.get((base, None))


def run_case(c, case):
    t0 = time.time()
    _DRAWS[0] = 0
    res = {"id": case["id"], "outcome": "no_reference", "reason": "", "pcc": None, "rel_l2": None, "slope": None,
           "max_abs": None, "exact": None, "seconds": 0.0}
    ad = adapter_for(case)
    if ad is None:
        res["reason"] = "this script has no torch reference for this pair, so it was not tested"
        return res
    try:
        ref, out, exact = ad(c, case)
        s = stats(ref, out)
        res.update({k: s[k] for k in ("pcc", "rel_l2", "slope", "max_abs", "exact")})
        res["outcome"], res["reason"] = judge(s, exact)
        res["compared"] = "exact" if exact else "float"
    except Exception as exc:                                    # the candidate raised: that is a result, not a crash
        res["outcome"] = "error"
        res["reason"] = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ''}"
        res["trace_tail"] = traceback.format_exc().strip().splitlines()[-3:]
    res["seconds"] = round(time.time() - t0, 2)
    return res


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cases = json.load(open(argv[0]))["cases"]
    import torch
    import ttnn
    dev = ttnn.open_device(device_id=0)
    results = []
    try:
        c = Ctx(dev)
        for case in cases:
            results.append(run_case(c, case))
    finally:
        ttnn.close_device(dev)
    try:
        import importlib.metadata as md
        tv = md.version("ttnn")
    except Exception:
        tv = None
    print("\n" + MARK + json.dumps({"ttnn_version": tv, "torch_version": torch.__version__, "bars": BARS,
                                      "results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
