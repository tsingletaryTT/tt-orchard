# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Count the ttnn calls in a port's source, and derive the floor of device matmuls a forward pass must make (issue 541).

In run 18 of the tiny Qwen4 rehearsal the coder wrote 22 versions of `port_model.py`. None of them called `ttnn.<op>(`. Each had a line
`import ttnn  # noqa: F401` and computed on the host in torch. The check opened the devices, the gate's source check ("a model source
mentions ttnn") passed on the import line, and the stage 2 record called it a chip run.

This module is the static half of the gate. It parses the source with `ast`, so a mention in a comment or a string is not a call, and an
`import ttnn` is not a call. A call counts when it is reachable by a plain reading: statements after a `return`, `raise`, `break` or `continue`
in the same block, the body of `if False:` or `if 0:`, and `if TYPE_CHECKING:` are skipped. Reaching a function that nothing calls is not
decided here. The check's own process counts the calls the port makes at run time (orchard/skills/full-port-templates/port_check.py),
and that count is what the verdict rests on. This count is the cheap early reject: a port with no matmul-family call at all cannot pass.

The block between the two `shared` markers below is copied unchanged into the port check template (the template runs from the stage
directory and cannot import `orchard`). tests/test_ttnn_calls.py compares the two copies byte for byte.
"""
# --- begin shared: ttnn_calls
import ast
import re

MATMUL_FAMILY = re.compile(r"matmul|linear|sdpa|scaled_dot_product|conv[123]d|conv_transpose|einsum")
# ttnn names that manage the device or build a shape. A call to one is not a computation on the device.
NOT_COMPUTE = frozenset({"open_device", "close_device", "open_mesh_device", "close_mesh_device", "set_fabric_config", "MeshShape",
                         "synchronize_device", "get_device_ids", "get_num_devices", "manage_device", "begin_trace_capture",
                         "end_trace_capture", "release_trace", "deallocate", "to_torch", "from_torch", "as_tensor", "to_device",
                         "from_device", "load_tensor", "dump_tensor", "to_dtype", "to_layout", "reshape", "typecast", "get_arch_name"})
LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
LAYER_COUNT_MAX = 10_000
_ZERO_NAMES = frozenset({"TYPE_CHECKING"})


def is_matmul(name: str) -> bool:
    """True when the dotted ttnn name (`ttnn.linear`, `ttnn.experimental.minimal_matmul`) is a matmul-family op."""
    return bool(MATMUL_FAMILY.search(name.lower()))


def is_compute(name: str) -> bool:
    """True when the call is an op that computes, and not device management, a shape or a copy between host and device."""
    return name.rsplit(".", 1)[-1] not in NOT_COMPUTE


def _chain(node):
    """The dotted name of a Name or Attribute chain (`a.b.c`), or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _constant_test(test):
    """True or False when an `if` test is a constant a plain reading can decide, else None."""
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.Name) and test.id in _ZERO_NAMES:
        return False
    if isinstance(test, ast.Attribute) and test.attr in _ZERO_NAMES:
        return False
    return None


def static_calls(source: str) -> dict:
    """{"calls": {dotted name: count}, "error": None} for the reachable `ttnn.<op>(` calls of `source`; with a syntax error, {"calls": {}, "error": reason}.

    A name counts as ttnn when it is bound by `import ttnn`, `import ttnn as t`, `from ttnn import linear` or `x = ttnn.linear`, and the call is
    `ttnn.linear(...)`, `t.linear(...)`, `linear(...)` or `x(...)`. Calls that only manage the device (NOT_COMPUTE) are listed in `calls` too; use
    `is_compute` to leave them out."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return {"calls": {}, "error": f"{type(exc).__name__}: the file does not parse"}
    roots: dict[str, str] = {}          # local name -> the dotted ttnn name it stands for

    def bind(stmt):
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                if a.name == "ttnn" or a.name.startswith("ttnn."):
                    roots[a.asname or "ttnn"] = a.name if a.asname else "ttnn"
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and (stmt.module == "ttnn" or (stmt.module or "").startswith("ttnn.")):
            for a in stmt.names:
                roots[a.asname or a.name] = f"{stmt.module}.{a.name}"
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            dotted = _chain(stmt.value)
            if dotted and dotted.split(".", 1)[0] in roots:
                head, _, rest = dotted.partition(".")
                roots[stmt.targets[0].id] = roots[head] + ("." + rest if rest else "")

    calls: dict[str, int] = {}

    def look(node):
        """The calls in an expression or in the parts of a statement that are not blocks of statements."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                dotted = _chain(sub.func)
                if dotted and dotted.split(".", 1)[0] in roots:
                    head, _, rest = dotted.partition(".")
                    name = roots[head] + ("." + rest if rest else "")
                    if name != "ttnn":
                        calls[name] = calls.get(name, 0) + 1

    def block(stmts):
        for stmt in stmts:
            bind(stmt)
            if isinstance(stmt, ast.If):
                look(stmt.test)
                decided = _constant_test(stmt.test)
                if decided is not False:
                    block(stmt.body)
                if decided is not True:
                    block(stmt.orelse)
            elif isinstance(stmt, ast.While) and _constant_test(stmt.test) is False:
                block(stmt.orelse)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for d in stmt.decorator_list:
                    look(d)
                if isinstance(stmt, ast.ClassDef):
                    for b in stmt.bases:
                        look(b)
                else:
                    for d in [*stmt.args.defaults, *[k for k in stmt.args.kw_defaults if k is not None]]:
                        look(d)
                block(stmt.body)
            elif hasattr(stmt, "body") or hasattr(stmt, "handlers"):          # for, while, with, try, match: the header, then each block
                for field, value in ast.iter_fields(stmt):
                    if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                        block(value)
                    elif field == "handlers":
                        for h in value:
                            block(h.body)
                    elif field == "cases":
                        for c in value:
                            block(c.body)
                    elif isinstance(value, ast.AST):
                        look(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, ast.AST) and not isinstance(v, (ast.stmt, ast.excepthandler, ast.match_case)):
                                look(v)
                            elif isinstance(v, ast.withitem):
                                look(v)
            else:
                look(stmt)
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                return                       # what follows in this block cannot run

    block(tree.body)
    return {"calls": dict(sorted(calls.items())), "error": None}


def layer_floor(config) -> tuple[int, int | None]:
    """(floor, layers): the number of matmul-family device calls one forward pass must make at least, and the layer count it came from (None
    when the config does not give one). Every decoder layer projects its input at least once, so a pass that touches the device for every layer
    makes at least one matmul or linear per layer. Without a layer count the floor is 1."""
    section = config.get("text_config") if isinstance(config, dict) and isinstance(config.get("text_config"), dict) else config
    if not isinstance(section, dict):
        return 1, None
    for key in LAYER_COUNT_KEYS:
        n = section.get(key)
        if type(n) is int and 0 < n <= LAYER_COUNT_MAX:
            return n, n
    kinds = section.get("layer_types")
    if isinstance(kinds, list) and 0 < len(kinds) <= LAYER_COUNT_MAX:
        return len(kinds), len(kinds)
    return 1, None
# --- end shared: ttnn_calls
