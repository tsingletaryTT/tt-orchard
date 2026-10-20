#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Check an agent-written kernel.py against its op's reference on the tt-lang simulator.

    python3 kernel_check.py [--attempt-dir DIR] [--phase sim] [--spec FILE] [--sim PATH] [--timeout-s N]
                            [--tt-lang-json JSON] [--out FILE]

Defaults come from DIR/check_config.json: {"sim": "<tt-lang-sim>", "timeout_s": 300, "tt_lang": {...}}. It runs
kernel_harness.py from the directory THIS file is in, so the harness that judges a kernel is the copy next to the check
that was run. Exit 0 and `PASS` when the kernel passed; exit 1 and `FAIL: <category>: <first line>` when it did not;
exit 2 on a usage error. It writes check.json with sha256 values of the files that took part and the bars it applied.
An interrupt (SIGTERM, or SIGINT or SIGHUP unless they started ignored) stops the simulator and everything it started
and writes no check.json; the process then ends by that same signal, so its exit status names it (-15 for SIGTERM,
-1 for SIGHUP).

Before the simulator runs, a static gate (`check_kernel_source`) reads kernel.py with `ast` and refuses the shortcuts a
model is likely to try: importing anything but ttl, ttnn and math (and a short list of names from typing and
__future__), calling open/exec/eval and similar or reaching them or os/sys as an attribute, naming dunder
attributes, naming ttnn's to_torch/from_torch, and an operation that is not exactly one top-level def decorated
with ttl.operation (any name). A refused kernel ends `forbidden_code` and never runs. The gate is best effort. The kernel still runs in
the harness's own process, so a determined kernel can find a way around a static check; the gate stops the easy ways.

A kernel passes only when every input set is inside every bar: PCC, relative L2 error, least-squares slope, and
max_abs_err <= atol + rtol * max|expected|. PCC alone does not change when the output is scaled, so a kernel off by a
constant factor would pass on PCC. A missing or null field fails the set.

The category comes from the diagnostic text the simulator prints and the harness's result file, never from an exit code
alone. (The tt-lang-sim wheel without its diagnostics file ends every diagnosed error as FileNotFoundError with a
non-zero exit. That case is reported as harness_error, because it is a fault in the environment. The check looks for
the file next to the simulator, so a kernel that prints the same text cannot use it to shift the blame.)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE_RE = re.compile(r"Tensor shape .*does not match Block shape")
# The harness prints `PROGRESS import` just before it loads kernel.py and `PROGRESS run <set>` just before it calls
# OPERATION. When one of these is the last marker, the kernel had control when the process ended.
KERNEL_MARKS = ("PROGRESS import", "PROGRESS run ")

# The provisional bars the op specs ship with. classify falls back to these when its caller passes no `bars`.
DEFAULT_BARS = {
    "sim": {"pcc": 0.999, "rel_l2": 1e-2, "slope": [0.98, 1.02], "max_abs_atol": 1e-2, "max_abs_rtol": 1e-2},
    "hw": {"pcc": 0.99, "rel_l2": 5e-2, "slope": [0.95, 1.05], "max_abs_atol": 1e-1, "max_abs_rtol": 5e-2},
}

# The static gate's tables.
ALLOWED_IMPORTS = {"ttl", "ttnn", "math"}
HARMLESS_FROM_MODULES = {"__future__", "typing"}       # `from __future__ import ...` and `from typing import ...`
# The only names `from typing import` and `from __future__ import` may bring in. typing's namespace also holds modules
# (sys, and re or io in some Python versions), and `from typing import sys as s` would hand the kernel sys under a name
# that is not forbidden. A fixed list is used because the gate cannot know which Python the harness runs; a test
# checks that none of these is a module in the interpreter that runs the tests.
TYPING_NAMES = {"Any", "Optional", "Union", "List", "Tuple", "Dict", "Set", "FrozenSet", "Callable", "Sequence",
                "Mapping", "Iterable", "Iterator", "Literal", "Final", "ClassVar", "TypeVar", "Generic", "Protocol",
                "NamedTuple", "TypedDict", "TYPE_CHECKING", "Annotated", "Type", "NoReturn", "cast", "overload"}
FUTURE_NAMES = {"annotations", "division", "absolute_import", "print_function", "unicode_literals", "generator_stop",
                "nested_scopes", "generators", "with_statement"}
FROM_NAME_ALLOWED = {"typing": TYPING_NAMES, "__future__": FUTURE_NAMES}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "setattr", "getattr",
                   "delattr", "vars", "breakpoint", "sys", "os", "atexit", "builtins", "subprocess", "importlib",
                   "references"}
CALL_ONLY_NAMES = {"input"}                             # forbidden as a call; a variable or argument may be named input
HARMLESS_DUNDER_NAMES = {"__name__", "__all__"}         # `if __name__ == "__main__":` and `__all__ = [...]`
READBACK_NAMES = {"to_torch", "from_torch"}
REMEDIES = {
    "import": "import only ttl, ttnn and math, and compute with ttl.math and block operators",
    "name": "remove it, a kernel does not need it",
    "dunder": "remove it, a kernel does not need dunder attributes",
    "readback": "the harness reads the result back; remove the host code and keep only the @ttl.operation function",
    "operation": "define one top-level function decorated only with @ttl.operation(...), with any name",
}
MAX_LISTED = 8                                          # violations printed in the FAIL line; check.json keeps all


def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def bars_from_spec(spec: dict, phase: str) -> dict:
    """The bars for one phase, read from an op spec. A missing key raises KeyError."""
    return {"pcc": spec[f"pcc_bar_{phase}"], "rel_l2": spec[f"rel_l2_bar_{phase}"],
            "slope": list(spec[f"slope_band_{phase}"]), "max_abs_atol": spec[f"max_abs_atol_{phase}"],
            "max_abs_rtol": spec[f"max_abs_rtol_{phase}"]}


def diagnostics_present(sim) -> bool | None:
    """Whether tt-lang-sim's diagnostics.py exists in the venv the simulator lives in.

    The file is site-packages/ttl/ttl/diagnostics.py, which the wheel does not ship. The wheel's loader,
    ttl/sim/diagnostics.py, is always present and reads that file, so the loader says nothing about it.

    A bare command name is looked up on PATH, and the path is resolved, so a symlink to the simulator leads to its
    venv. None when the simulator cannot be found or no ttl package sits next to it (a layout this code does not
    know); classify then falls back to reading the text."""
    text = str(sim)
    if os.sep not in text:
        found = shutil.which(text)
        if not found:
            return None
        text = found
    venv = Path(text).resolve().parent.parent
    packages = list(venv.glob("lib/python*/site-packages/ttl"))
    if not packages:
        return None
    return any((p / "ttl" / "diagnostics.py").is_file() for p in packages)


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _module_scope(tree: ast.Module):
    """Every node evaluated in module scope. Function and class bodies, lambdas and comprehensions are not entered;
    a def's decorators and defaults and a class's bases are, because they run in module scope."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(node.decorator_list)
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d is not None)
        elif isinstance(node, ast.ClassDef):
            stack.extend(node.decorator_list)
            stack.extend(node.bases)
        elif not isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            stack.extend(ast.iter_child_nodes(node))


def _module_bindings(tree: ast.Module) -> dict:
    """name -> set of (line, kind) for every binding of a module-level name.

    kind is "import ttl" (`import ttl`, `import ttl as x`, `import ttl.sub`), "ttl operation" (`from ttl import
    operation [as x]`), "def" (a def directly in the module body) or "other" (anything else: assignment, for, with,
    walrus, del, except-as, class, other imports, a def nested in an if, a name declared global in a function)."""
    found: dict = {}

    def add(name, line, kind):
        found.setdefault(name, set()).add((line, kind))

    top = {id(n) for n in tree.body}
    for node in _module_scope(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            add(node.id, node.lineno, "other")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, node.lineno, "def" if id(node) in top else "other")
        elif isinstance(node, ast.ClassDef):
            add(node.name, node.lineno, "other")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name, node.lineno, "other")
        elif isinstance(node, ast.Import):
            for a in node.names:
                is_ttl = (a.asname is None and a.name.split(".")[0] == "ttl") or (a.asname is not None and a.name == "ttl")
                add(a.asname or a.name.split(".")[0], node.lineno, "import ttl" if is_ttl else "other")
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                is_op = node.level == 0 and node.module == "ttl" and a.name == "operation"
                add(a.asname or a.name, node.lineno, "ttl operation" if is_op else "other")
    for node in ast.walk(tree):                           # these bind module names from any depth
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            add(node.target.id, node.lineno, "other")
        elif isinstance(node, ast.Global):
            for name in node.names:
                add(name, node.lineno, "other")
    return found


def _resolves_to_ttl_operation(dec: ast.expr, ttl_names: set, op_names: set) -> bool:
    """`@<ttl>.operation`, `@<ttl>.operation(...)`, `@<op>` or `@<op>(...)`, where <ttl> was bound only by
    `import ttl [as x]` and <op> only by `from ttl import operation [as x]`."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Attribute):
        return target.attr == "operation" and isinstance(target.value, ast.Name) and target.value.id in ttl_names
    return isinstance(target, ast.Name) and target.id in op_names


def _violation(line: int, rule: str, detail: str, remedy: str | None = None) -> dict:
    return {"rule": rule, "line": line, "detail": detail, "remedy": remedy or REMEDIES[rule]}


def _analyse(path) -> tuple[list, str | None, bool]:
    """(violations sorted by line, the operation's name or None, whether the file parsed).

    The operation is the one top-level def decorated with ttl's operation, whatever its name (Task 14d: three live
    runs named it after the op although the skill said OPERATION). Its name must be bound only by that def, it must
    have that one decorator, and the name may not be one the gate forbids. Two or more such defs are a violation. A
    def named OPERATION must carry the decorator, as before. A file that does not parse gives ([], None, False), and
    the gate lets it through so the harness reports its SyntaxError."""
    source = Path(path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], None, False
    found = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    found.append(_violation(line, "import", f"import {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                found.append(_violation(line, "import", f"relative import from {'.' * node.level}{module}"))
            elif module not in HARMLESS_FROM_MODULES and module.split(".")[0] not in ALLOWED_IMPORTS:
                found.append(_violation(line, "import", f"import from {module}"))
            for a in node.names:
                if a.name == "*":
                    found.append(_violation(line, "import", f"from {module} import *",
                                            "import the names you need one by one"))
                    continue
                if not node.level and module in FROM_NAME_ALLOWED and a.name not in FROM_NAME_ALLOWED[module]:
                    found.append(_violation(line, "import", f"from {module} import {a.name}",
                                            f"from {module}, import only these names: "
                                            + ", ".join(sorted(FROM_NAME_ALLOWED[module]))))
                # A forbidden name is refused when it is imported and when it is the name an import binds, so
                # `from ttnn import os as o` and `from typing import Any as sys` are both caught.
                for bound in (a.name, a.asname):
                    if bound and (bound in FORBIDDEN_NAMES
                                  or (_is_dunder(bound) and bound not in HARMLESS_DUNDER_NAMES)):
                        found.append(_violation(line, "name", f"import of {bound}" if bound == a.name
                                                else f"import bound to {bound}"))
                if a.name in READBACK_NAMES:
                    found.append(_violation(line, "readback", f"import of {a.name}"))
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or (_is_dunder(node.id) and node.id not in HARMLESS_DUNDER_NAMES):
                found.append(_violation(line, "name", f"use of {node.id}"))
            elif node.id in READBACK_NAMES:
                found.append(_violation(line, "readback", f"use of {node.id}"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CALL_ONLY_NAMES:
            found.append(_violation(line, "name", f"call of {node.func.id}()"))
        elif isinstance(node, ast.Attribute):
            if _is_dunder(node.attr):
                found.append(_violation(line, "dunder", f"attribute {node.attr}"))
            elif node.attr in READBACK_NAMES:
                found.append(_violation(line, "readback", f"attribute {node.attr}"))
            elif node.attr in FORBIDDEN_NAMES:          # ttnn.os, x.sys, y.open: a module or builtin reached by name
                found.append(_violation(line, "name", f"attribute {node.attr}"))

    bindings = _module_bindings(tree)
    alias_kinds = {"import ttl", "ttl operation"}
    ttl_names, op_names = set(), set()
    for name, entries in bindings.items():
        kinds = {k for _, k in entries}
        if name == "OPERATION" or not kinds & alias_kinds:    # OPERATION's own binding rule reports it once
            continue
        if len(kinds) > 1:                                # bound to ttl (or its operation) and also to something else
            for line, kind in sorted(entries):
                if kind not in alias_kinds or len(kinds & alias_kinds) > 1:
                    found.append(_violation(line, "operation", f"{name} is rebound, so @{name} may not be ttl's operation",
                                            "keep the names ttl and operation for the imports and use another name"))
            continue
        (ttl_names if "import ttl" in kinds else op_names).add(name)
    top_defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    decorated = [n for n in top_defs if any(_resolves_to_ttl_operation(d, ttl_names, op_names) for d in n.decorator_list)]
    # A def named OPERATION keeps the old rule: it is meant as the operation, so it must carry the decorator.
    for n in top_defs:
        if n.name == "OPERATION" and n not in decorated:
            found.append(_violation(n.lineno, "operation", "OPERATION must be decorated with ttl.operation and nothing else"))
    if len(decorated) > 1:
        names = ", ".join(n.name for n in decorated)
        found.append(_violation(decorated[1].lineno, "operation",
                                f"{len(decorated)} functions are decorated with @ttl.operation ({names})",
                                "define exactly one @ttl.operation function"))
    op = decorated[0] if len(decorated) == 1 else None
    if op is not None:
        name = op.name
        if len(op.decorator_list) != 1:
            found.append(_violation(op.lineno, "operation", f"{name} must be decorated with ttl.operation and nothing else"))
        if any(kind != "def" for _, kind in bindings.get(name, ())) or \
                sum(1 for _, kind in bindings.get(name, ()) if kind == "def") > 1:
            for line, kind in sorted(bindings.get(name, ())):
                if line != op.lineno or kind != "def":
                    found.append(_violation(line, "operation",
                                            f"{name} is bound by something other than its decorated def"))
        if name in FORBIDDEN_NAMES or name in CALL_ONLY_NAMES or name in READBACK_NAMES or _is_dunder(name):
            found.append(_violation(op.lineno, "operation", f"the operation's name {name} is a name the gate refuses",
                                    "give the @ttl.operation function another name"))
    for line, kind in sorted(bindings.get("OPERATION", ())):
        if kind != "def" and (op is None or op.name != "OPERATION"):
            found.append(_violation(line, "operation", "OPERATION is bound by something other than its decorated def"))
    unique = {(v["line"], v["rule"], v["detail"]): v for v in found}
    return [unique[k] for k in sorted(unique)], (op.name if op is not None else None), True


def kernel_source_violations(path) -> list[dict]:
    """Every rule kernel.py breaks, as dicts {rule, line, detail, remedy}, sorted by line. Empty when it breaks none.

    Rules: `import` (only ttl, ttnn and math, plus `from __future__` and `from typing` with the names in
    FROM_NAME_ALLOWED; no star imports), `name` (FORBIDDEN_NAMES as a name, an attribute, an imported name or the name
    an import binds; a call of input(); any dunder name but __name__ and __all__), `dunder` (an attribute
    whose name starts and ends with two underscores), `readback` (to_torch and from_torch, as an attribute, a name
    or an imported name), `operation` (exactly one top-level def whose only decorator is ttl's operation, reached
    through `import ttl [as x]` or `from ttl import operation [as x]`, names nothing rebinds; its name, any name the
    gate does not refuse, is bound only by that def; a def named OPERATION must carry the decorator).

    The whole file is checked, including an `if __name__ == "__main__":` block. The harness loads kernel.py under
    another module name, so that block never runs, but the gate does not rely on that. Best effort: this reads the
    source, so it cannot see what code builds at run time."""
    return _analyse(path)[0]


NO_OPERATION = "kernel.py defines no @ttl.operation function"


def operation_candidate(path) -> str | None:
    """The name of the one top-level @ttl.operation function, whether or not the file breaks other rules; None with
    none or several, or when the file does not parse. The harness runs this: the gate has already refused a file that
    breaks a rule on every path that reaches the harness, and a test runs the harness directly on a refused kernel."""
    return _analyse(path)[1]                           # None for a file that does not parse


def operation_name(path) -> str | None:
    """The name of the operation the gate accepted: the one top-level @ttl.operation function, when kernel.py breaks
    no rule. None when the gate refuses the file, finds no operation, or the file does not parse. The harness calls
    this function, so every path that runs a kernel (the agent's check, the probe's re-check, the hardware phase)
    calls the function the gate judged."""
    found, name, parsed = _analyse(path)
    return name if parsed and not found else None


def violations_summary(found: list[dict]) -> str:
    """`N violations: <rule> at line N: <detail> (<remedy>); ...`, at most MAX_LISTED listed."""
    items = [f"{v['rule']} at line {v['line']}: {v['detail']} ({v['remedy']})" for v in found[:MAX_LISTED]]
    if len(found) > MAX_LISTED:
        items.append(f"and {len(found) - MAX_LISTED} more")
    return f"{len(found)} violation{'s' if len(found) != 1 else ''}: " + "; ".join(items)


def check_kernel_source(path) -> tuple[bool, str]:
    """The static gate. (True, "") when kernel.py breaks no rule; (False, violations_summary(...)) when it does;
    (False, "no-operation: ...") when it breaks none and defines no @ttl.operation function, which the check reports
    as no_kernel.
    A file that does not parse passes, so the harness reports its SyntaxError as import_error."""
    found, name, parsed = _analyse(path)
    if found:
        return False, violations_summary(found)
    if parsed and name is None:
        return False, "no-operation: " + NO_OPERATION
    return True, ""


def first_line(text: str) -> str:
    """The simulator's first diagnostic line: `error: ...`, a deadlock notice, or the `Error in kernel` header."""
    lines = [l.strip() for l in text.splitlines()]
    for l in lines:
        if l.startswith("error:") or l.startswith("Deadlock detected"):
            return l[:200]
    for l in lines:
        if "Error in kernel" in l:
            return l[:200]
    return ""


POINTER_RE = re.compile(r"^\s*-->\s+(?P<path>\S+):(?P<line>\d+):\d+\s*$")
WHERE_MAX = 160


def source_pointer(text: str) -> str:
    """`kernel.py line N: <that line>` for the first place the simulator pointed at, or "" when there is none.

    The simulator prints `--> <file>:<line>:<col>` and the source line under each diagnostic. The coder read only the
    one-line summary, so it fixed errors without knowing which line raised them (pilot 2, 2026-10-17). Only a pointer
    into kernel.py counts: a pointer into the simulator's own code says nothing the coder can act on. The path is
    cut to the file name, so no machine path reaches the coder, and the line is cut to WHERE_MAX characters."""
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        m = POINTER_RE.match(raw)
        if not m:
            continue
        if Path(m["path"]).name != "kernel.py":
            return ""
        for nxt in lines[i + 1:i + 6]:
            code = re.match(rf"^\s*{m['line']}\s*\|\s?(.*)$", nxt)
            if code:
                return f"kernel.py line {m['line']}: {code[1].strip()}"[:WHERE_MAX]
        return ""
    return ""


TILE_SLICE_RE = re.compile(r"(?P<axis>\w+) slice (?P<start>\d+):(?P<stop>\d+) is outside the tensor, "
                           r"which has (?P<extent>\d+) tile\(s\) along it")


def unit_hint(text: str) -> str:
    """One sentence for the commonest mistake: a slice counted in elements where the tensor counts tiles, or "".

    With the failing line in front of it, the coder (pilot 3, 2026-10-17) still passed `0:32` to a tensor of 2 tiles
    and did not fix it in 20 turns. The simulator's message says "tile(s)" and does not say that slices count tiles.
    The hint states the unit and the extent from the message. It names no op and gives no answer."""
    m = TILE_SLICE_RE.search(text)
    if not m:
        return ""
    return (f"tensor slices count tiles, not elements (one tile is 32x32). This tensor has {m['extent']} tile(s) along "
            f"the {m['axis']} axis, so a slice there must lie within 0:{m['extent']}; {m['start']}:{m['stop']} is too wide.")


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _fmt(v: float) -> float:
    """A value short enough for a FAIL line."""
    return round(v, 6)


def _set_problem(s: dict, bar: float, bars: dict) -> str | None:
    """Why one input set fails, or None when it is inside every bar."""
    pcc, finite = s.get("pcc"), s.get("finite")
    if not _number(pcc) or finite is not True or pcc < bar:
        return f"pcc {pcc}, finite {finite}, bar {bar}"
    for field in ("rel_l2", "slope", "max_abs_err", "exp_abs_max"):
        if not _number(s.get(field)):
            return f"{field} missing or not a number ({s.get(field)!r})"
    if s["rel_l2"] > bars["rel_l2"]:
        return f"rel_l2 {_fmt(s['rel_l2'])} above bar {bars['rel_l2']}"
    lo, hi = bars["slope"]
    if not lo <= s["slope"] <= hi:
        return f"slope {_fmt(s['slope'])} outside [{lo}, {hi}]"
    limit = bars["max_abs_atol"] + bars["max_abs_rtol"] * s["exp_abs_max"]
    if s["max_abs_err"] > limit:
        return (f"max_abs_err {_fmt(s['max_abs_err'])} above {bars['max_abs_atol']} + {bars['max_abs_rtol']} * "
                f"{_fmt(s['exp_abs_max'])} = {_fmt(limit)}")
    return None


def _pair(w: dict) -> str:
    def v(x):
        return "nan or inf" if x is None else f"{x:.5g}"
    exp = "nan or inf" if w.get("expected") is None else f"{w['expected']:.4g}"
    return f"{w.get('at')}: expected {exp}, got {v(w.get('got'))}"


def numeric_detail(sets: list, bar: float, bars: dict) -> str:
    """One line about the first input set that fails the numeric bars, or "" when none does.

    The line says what the first-line summary cannot: how far off the scale is (slope), the relative error, the largest
    error beside the size of the expected values, and the worst places with their expected and got values. In L4e
    (2026-10-17) two attempts at different seeds wrote the same wrong kernel and were told only `pcc 0.9796`; the result
    held a slope of 964 that they never saw. The line states measured facts and names no op and no fix."""
    failing = [s for s in sets if isinstance(s, dict) and _set_problem(s, bar, bars) is not None]
    if not failing:
        return ""
    s = failing[0]
    slope = s.get("slope")
    if s.get("finite") is not True:
        scale = "got is not finite (nan or inf somewhere)"
    elif not _number(slope):
        scale = "the scale could not be measured"
    elif slope < 0:
        scale = f"got has the opposite sign of expected (slope {slope:.3g})"
    elif slope > 1.25 or slope < 0.8:
        scale = f"got is about {slope:.3g}x expected"
    else:
        scale = f"scale is about right (slope {slope:.3g})"
    parts = [scale]
    if _number(s.get("pcc")):
        parts.append(f"pcc {s['pcc']:.4g}")
    if _number(s.get("rel_l2")):
        parts.append(f"rel_l2 {s['rel_l2']:.3g}")
    if _number(s.get("max_abs_err")):
        reach = f" where expected values reach {s['exp_abs_max']:.3g}" if _number(s.get("exp_abs_max")) else ""
        parts.append(f"max_abs_err {s['max_abs_err']:.5g}{reach}")
    line = f"{s.get('name')}: " + ", ".join(parts)
    worst = [w for w in s.get("worst") or [] if isinstance(w, dict)][:2]
    if worst:
        line += "; worst: " + "; ".join(_pair(w) for w in worst)
    if len(sets) > 1:
        line += f"; {len(failing)} of {len(sets)} input sets fail"
    return line


def classify(*, kernel_present, timed_out, returncode, text, result, bar, phase="sim", bars=None,
             diagnostics_present=None):
    """(category or None, first line). None means the kernel passed.

    `bar` is the PCC bar. `bars` holds the other bars (bars_from_spec); without it the phase's DEFAULT_BARS apply.
    `diagnostics_present` is what the check measured next to the simulator: True means a missing-diagnostics message
    in the text cannot be the environment's (the kernel printed or raised it), False means it is, None means unknown
    and the text decides."""
    numeric = "hw_numeric_mismatch" if phase == "hw" else "numeric_mismatch"
    runtime = "compile_error" if phase == "hw" else "tt_lang_error"
    bars = bars or DEFAULT_BARS[phase]
    if not kernel_present:
        return "no_kernel", "kernel.py does not exist"
    if timed_out:
        return "timeout", "no result within the time allowed"
    err = (result or {}).get("error")
    blob = text + "\n" + ((err or {}).get("text") or "")
    if diagnostics_present is not True and "FileNotFoundError" in blob and "diagnostics.py" in blob:
        return "harness_error", "tt-lang-sim's diagnostics.py is missing; run `tt-orchard kernel-env`"
    if err and err.get("type") == "NoOperation":
        return "no_kernel", NO_OPERATION
    if err and err.get("phase") == "setup":         # the spec, the references, the device or the inputs
        first = ((err.get("text") or "").splitlines() or [""])[0]
        return "harness_error", f"{err.get('type')}: {first}"[:200]
    if err and err.get("phase") == "import":
        first = ((err.get("text") or "").splitlines() or [""])[0]
        return "import_error", f"{err.get('type')}: {first}"[:200]
    m = SHAPE_RE.search(blob)
    if m:
        return "shape_mismatch", first_line(blob) or m.group(0)[:200]
    if err or "Deadlock detected" in blob:
        return runtime, first_line(blob) or ((err or {}).get("text") or "")[:200]
    if result is None:
        # The last marker says who had control. After `PROGRESS import` or `PROGRESS run <set>` it was the kernel.
        # After `open_device` or `set <name>` it was the harness, building inputs. (A kernel that prints a marker
        # itself can move this line; with the gate it has no way to end the process.)
        markers = [l.strip() for l in text.splitlines() if l.strip().startswith("PROGRESS ")]
        last = markers[-1] if markers else ""
        if last == KERNEL_MARKS[0] or last.startswith(KERNEL_MARKS[1]):
            return runtime, "the kernel ended the process without a result"
        return "harness_error", f"the harness ended with exit {returncode} and wrote no result"
    sets = result.get("sets", [])
    if not sets:
        return "harness_error", "the harness ran no input sets"
    for s in sets:
        problem = _set_problem(s, bar, bars)
        if problem:
            return numeric, f"set {s.get('name')}: {problem}"
    return None, ""


def _kill_own_group(proc) -> None:
    """SIGKILL the process group the simulator leads, so whatever it started goes with it. Only a group the child
    leads itself is signalled: never id 1 or below, never this process's own group, and never a pid that does not
    lead its own group (the guards orchard/stallwatch.py uses; this template stands alone, so it has its own copy).
    Anything else gets a plain kill of the child alone."""
    import signal
    if getattr(proc, "returncode", None) is not None:
        return                                       # already reaped: its pid may belong to another process now
    pgid = proc.pid
    try:
        leads = isinstance(pgid, int) and pgid > 1 and pgid != os.getpgrp() and os.getpgid(pgid) == pgid
    except OSError:
        leads = False
    try:
        if leads:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass


def _die_with_parent_fn(parent: int | None = None):
    """A preexec function for the simulator. Its parts are resolved here, in the parent, so the forked child only
    calls them: prctl(PR_SET_PDEATHSIG, SIGKILL) asks Linux to kill the simulator when this check process dies (it
    covers a SIGKILL of the check, after which no handler of ours runs). A parent that died between the fork and the
    prctl is caught by comparing os.getppid() with `parent` (this process's pid, recorded before the fork): the child
    exits at once. Where prctl is missing only the parent check runs."""
    parent = os.getpid() if parent is None else parent
    try:
        import ctypes
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except Exception:
        prctl = None

    def child() -> None:
        if prctl is not None:
            try:
                prctl(1, 9)                              # PR_SET_PDEATHSIG = 1, SIGKILL = 9
            except Exception:
                pass
        if os.getppid() != parent:
            os._exit(1)
    return child


_ARRIVED: list = []                                   # the signal _raise_interrupt turned into KeyboardInterrupt


def _raise_interrupt(signum, frame):
    _ARRIVED.append(signum)
    raise KeyboardInterrupt(f"signal {signum}")


def _exit_by_signal() -> None:
    """After an interrupt: end this process by the signal that arrived (SIGINT when none was recorded, as for a plain
    Ctrl-C), with its default action, so the exit status names that signal and not SIGINT."""
    import signal
    sig = _ARRIVED[-1] if _ARRIVED else signal.SIGINT
    signal.signal(sig, signal.SIG_DFL)
    os.kill(os.getpid(), sig)


def run_sim(argv, *, timeout_s, cwd) -> tuple[bool, int | None, str]:
    """(timed_out, return code, stdout and stderr) of one simulator run. The simulator starts in its own session; on a
    timeout its whole group is killed, so a process it started does not outlive the check. On any exception, including
    the KeyboardInterrupt that main's handlers raise for SIGTERM, SIGINT and SIGHUP (SIGINT and SIGHUP are left alone
    when they start ignored; SIGTERM is always handled), the group is killed before the exception goes on. If this process is killed outright, the simulator gets SIGKILL from its parent-death signal."""
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, preexec_fn=_die_with_parent_fn())
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return False, proc.returncode, (out or "") + "\n" + (err or "")
    except subprocess.TimeoutExpired:
        _kill_own_group(proc)
        try:
            out, err = proc.communicate(timeout=10)   # bounded: a process that left the group may hold the pipes
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = "", ""
        return True, None, (out or "") + "\n" + (err or "")
    except BaseException:
        _kill_own_group(proc)
        raise


def main(argv=None) -> int:
    """Run the check. SIGTERM, SIGINT and SIGHUP (the last two only when they did not start ignored) become a
    KeyboardInterrupt while it runs, so a watcher that stops this process (run_watched sends SIGTERM to this
    process's group) also stops the simulator, which runs in a session of its own. The previous handlers are restored
    at the end."""
    import signal
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        if sig != signal.SIGTERM and signal.getsignal(sig) == signal.SIG_IGN:
            continue                                  # a signal that starts ignored (nohup) stays ignored
        try:
            previous[sig] = signal.signal(sig, _raise_interrupt)
        except ValueError:                            # not the main thread: no handler can be installed
            pass
    try:
        return _main(argv)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt-dir", default=".")
    ap.add_argument("--phase", choices=("sim",), default="sim")
    ap.add_argument("--spec")
    ap.add_argument("--sim")
    ap.add_argument("--timeout-s", type=float)
    ap.add_argument("--tt-lang-json")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    adir = Path(a.attempt_dir).resolve()
    cfg_path = adir / "check_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    sim = a.sim or cfg.get("sim")
    timeout_s = a.timeout_s if a.timeout_s is not None else cfg.get("timeout_s")
    timeout_s = 300 if timeout_s is None else timeout_s
    spec_path = Path(a.spec) if a.spec else adir / "kernel_spec.json"
    if not sim or not spec_path.is_file():
        print("usage error: need --sim (or check_config.json) and a spec", file=sys.stderr)
        return 2
    tt_lang = json.loads(a.tt_lang_json) if a.tt_lang_json else cfg.get("tt_lang", {})
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    kernel, result_path, log_path = adir / "kernel.py", adir / "harness-result.json", adir / "harness-output.txt"
    result_path.unlink(missing_ok=True)
    timed_out, rc, text = False, None, ""
    start = time.monotonic()
    category = line = None
    bars = None
    try:
        bars = bars_from_spec(spec, "sim")
    except KeyError as exc:
        category, line = "harness_error", f"the spec has no {exc.args[0]}"
    violations = []
    if category is None and kernel.is_file():
        violations = kernel_source_violations(kernel)
        ok, reason = check_kernel_source(kernel)
        if not ok:
            category, line = ("no_kernel", reason.removeprefix("no-operation: ")) if reason.startswith("no-operation") \
                else ("forbidden_code", reason)
    if category is None and kernel.is_file():
        timed_out, rc, text = run_sim([sim, str(HERE / "kernel_harness.py"), "--", "--spec", str(spec_path),
                                       "--kernel", str(kernel), "--out", str(result_path), "--phase", "sim"],
                                      timeout_s=timeout_s, cwd=adir)
    log_path.write_text(text, encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    if category is None:
        category, line = classify(kernel_present=kernel.is_file(), timed_out=timed_out, returncode=rc, text=text,
                                  result=result, bar=spec["pcc_bar_sim"], phase="sim", bars=bars,
                                  diagnostics_present=diagnostics_present(sim))
    check = {"schema": 1, "op": spec["op"], "phase": "sim", "passed": category is None, "category": category,
             "first_line": line, "sets": (result or {}).get("sets", []), "pcc_bar": spec["pcc_bar_sim"], "bars": bars, "violations": violations,
             "provisional": True, "elapsed_s": round(time.monotonic() - start, 2), "tt_lang": tt_lang,
             "sha256": {"kernel.py": sha256_file(kernel), "kernel_harness.py": sha256_file(HERE / "kernel_harness.py"),
                        "references.py": sha256_file(HERE / "references.py"), "kernel_check.py": sha256_file(Path(__file__)),
                        "spec": sha256_file(spec_path)}}
    Path(a.out or adir / "check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print("PASS" if category is None else f"FAIL: {category}: {line}")
    where = source_pointer(text) if category is not None else ""
    if where:
        print(f"where: {where}")            # after the FAIL line, which the probe reads, so that line is unchanged
    hint = unit_hint(text) if category is not None else ""
    if hint:
        print(f"hint: {hint}")
    numbers = numeric_detail((result or {}).get("sets") or [], spec["pcc_bar_sim"], bars) \
        if category == "numeric_mismatch" and bars else ""
    if numbers:
        print(f"numbers: {numbers}")
    return 0 if category is None else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _exit_by_signal()
        raise
