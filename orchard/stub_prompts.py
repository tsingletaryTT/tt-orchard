# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One request per stub: the thin prompt path for a generated port (docs/superpowers/specs/2026-11-10-thin-prompt-path.md).

The agent loop sends the coder the whole conversation every turn: in the four rehearsal arms 85 to 96 percent of the prompt tokens
were context sent again, and the first write came after 10 to 16 turns of reading. This module builds, for one `# AGENT:` stub of a
generated port (orchard/port_template.py), a single chat request with only what that block needs, and splices the reply back into the
port. The supervisor can then run the CPU check between requests and send a repair request built from `last-check` output. There is
no tool loop: the model reads what it is given and answers with one function.

What a stub request holds, in this order (the fixed-cost parts first, the block last, so a prefix cache across requests can reuse the
shared head): the task rules; the helpers of the generated port with their signatures; the cheat sheet entries the block is likely to
need (the sheet's Linear and matmul section first, then the entries stage 0 mapped for the block's aten ops, then the mistakes seen in
runs); the construction lines for the block's class; the reference class excerpt; the stub function as it stands. The reply format is
one Python code block that defines the function with the same name and signature.

`splice` accepts a reply only when it is one code block whose module holds import statements and function definitions only, with the
stub's name and argument names among them, no `# AGENT:` marker, and the result parses. Nothing here runs a model; the loop that sends
the requests is the supervisor's (not wired in this prototype)."""
from __future__ import annotations

import ast
import re

from orchard import op_cheats, port_template

EXCERPT_MAX_CHARS = 12_000           # one reference class's excerpt (the whole excerpts file is 46 KB for the tiny Qwen4; one class is a page)
CHECK_MAX_CHARS = 4_000              # the last-check report in a repair request
DIFF_MAX_CHARS = 3_000               # the layer-diff report
REPLY_MAX_CHARS = 200_000
MAX_SHEET_ENTRIES = 24
HELPER_NAMES = ("_linear", "_rms_norm", "_embedding", "_mlp", "_attention", "_upload", "_upload_ids", "_to_host", "_rope_tables")
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_HEADING = re.compile(r"^## (\S+)\s*$", re.M)

RULES = ("You write one function of a TTNN port of a transformer decoder for a Tenstorrent chip. The port's other functions are given. "
         "Activations are ttnn tensors of shape [1, 1, S, features] in TILE layout, bfloat16. Every projection goes through the given "
         "_linear helper (a ttnn.linear call on a weight uploaded at load); do not compute a projection in torch. Glue that has no ttnn "
         "entry in the sheet (a recurrence, an index computation) may run in torch on the host: read a tensor back with _to_host and "
         "upload the result with _upload. Use only the ttnn calls listed; an argument that is not listed does not exist. Take every "
         "setting (activation, eps, group size, head counts) from the construction lines; a class default may differ from the resolved value. Answer with one "
         "```python code block that defines the function with the same name and arguments, plus any import it needs, and nothing else. "
         "Do not leave the `# AGENT:` marker in it. The port is one module imported by its file name: use no relative import, and call the helpers by name. "
         "An activation or weight is a ttnn tensor with no torch device: do not read x.device or call a torch function on it.")


class SpliceError(Exception):
    """A reply that cannot go into the port. The message says what is wrong and never quotes the reply."""


def _functions(tree: ast.Module) -> dict:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _arg_names(fn: ast.FunctionDef) -> list:
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args] + (["*" + a.vararg.arg] if a.vararg else []) + [x.arg for x in a.kwonlyargs]


def function_source(port_text: str, name: str) -> str:
    """The source of the top-level function `name` in `port_text`. Raises SpliceError when the port does not parse or lacks it."""
    try:
        tree = ast.parse(port_text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise SpliceError(f"the port does not parse: {type(exc).__name__}") from None
    fn = _functions(tree).get(name)
    if fn is None:
        raise SpliceError(f"the port has no top-level function {name}")
    lines = port_text.split("\n")
    return "\n".join(lines[fn.lineno - 1: fn.end_lineno])


def stub_names(port_text: str) -> list:
    """The top-level functions of the port whose body holds the stub marker, in file order."""
    try:
        tree = ast.parse(port_text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    lines = port_text.split("\n")
    out = []
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and any(l.strip().startswith(port_template.STUB_MARK) for l in lines[fn.lineno - 1: fn.end_lineno]):
            out.append(fn.name)
    return out


def class_excerpt(excerpts: str, cls: str, *, max_chars: int = EXCERPT_MAX_CHARS) -> str | None:
    """The `## <cls>` section of the reference excerpts file, cut to `max_chars` (with a note when cut), or None when the file lacks it."""
    if not isinstance(excerpts, str):
        return None
    m = re.search(rf"^## {re.escape(cls)}\s*$", excerpts, re.M)
    if m is None:
        return None
    rest = excerpts[m.end():]
    nxt = _HEADING.search(rest)
    body = rest[: nxt.start()] if nxt else rest
    body = body.strip("\n")
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n[... cut at {max_chars} characters; the file holds the rest]"
    return body


_LAYER_LOOP = re.compile(r"^(?P<indent>\s*)for\s+[\w, ]+\s+in\s+(?:enumerate\()?self\.layers\b", re.M)


def forward_part(code: str, part: str) -> str | None:
    """The part of a text model's forward before ("before") or after ("after") the loop over its layers, or None when the code has no such
    loop. The prelude stub needs only what runs before the layers and the epilogue stubs only what runs after, and the whole forward (position
    ids, masks, caches, the loop) made the coder write torch code for all of it (arm E4)."""
    m = _LAYER_LOOP.search(code) if isinstance(code, str) else None
    if m is None:
        return None
    if part == "before":
        return code[: m.start()].rstrip() + "\n"
    indent = len(m.group("indent"))
    rest = code[m.end():].split("\n")[1:]               # the lines after the `for` line
    out = []
    in_loop = True
    for line in rest:
        if in_loop and line.strip() and len(line) - len(line.lstrip()) <= indent:
            in_loop = False
        if not in_loop:
            out.append(line)
    return "\n".join(out).strip("\n") + "\n"


def construction_lines(construction_text: str | None, cls: str) -> list:
    """The lines of effective_construction.txt that name the class (its module groups and its resolved-vs-default flags)."""
    if not isinstance(construction_text, str):
        return []
    return [ln.strip() for ln in construction_text.split("\n") if cls in ln and not ln.startswith("#")][:40]


def ops_for(inventory, path: str) -> tuple:
    """(ttnn names stage 0 mapped or verified for aten ops under the module path pattern, aten ops it labelled candidate, composable or host)."""
    mapped, other = [], []
    ops = inventory.get("ops") if isinstance(inventory, dict) else None
    for op in ops if isinstance(ops, list) else []:
        if not isinstance(op, dict):
            continue
        mods = [port_template._NUMBER.sub("{i}", m) for m in (op.get("modules") or []) if isinstance(m, str)]
        if not any(m == path or m.startswith(path + ".") for m in mods):
            continue
        name = op.get("op") if isinstance(op.get("op"), str) else "?"
        if op.get("status") in ("mapped", "verified") and isinstance(op.get("ttnn"), str):
            mapped.append(op["ttnn"])
        elif op.get("status") in ("candidate", "composable", "host"):
            other.append(f"{name[:60]} ({op['status']})")
    return sorted(set(mapped)), sorted(set(other))


def sheet_section(sheet: dict, *, wanted: list, max_entries: int = MAX_SHEET_ENTRIES) -> str:
    """The cheat sheet lines for a request: the Linear and matmul section first (every block projects), then the entries whose last name
    part is in `wanted`, then the mistakes. Entries are the sheet's own text."""
    lines = []
    picked: list = []
    for section in sheet["sections"]:
        if "linear" in section["title"].lower() or "matmul" in section["title"].lower():
            picked += [e for e in section["entries"] if e["name"] in ("ttnn.linear", "ttnn.matmul")]
    for e in op_cheats.entries(sheet):
        leaf = e["name"].rsplit(".", 1)[-1]
        if leaf in wanted and e not in picked:
            picked.append(e)
    for e in picked[:max_entries]:
        lines += [f"`{e['signature']}`", *["    " + ln for ln in e["usage"].split("\n")], f"    {e['notes']}"]
    missing = sorted(w for w in wanted if not any(e["name"].rsplit(".", 1)[-1] == w for e in picked))
    if missing:
        lines.append("Stage 0 mapped these ttnn names for the block's ops but the sheet has not verified them; use one only after checking its "
                     "signature with `stage_tools.py cheat`: " + ", ".join(missing))
    if sheet.get("mistakes"):
        lines += ["", "Mistakes seen in runs:"]
        lines += [f"- {m['symptom']} Fix: {m['fix']}" for m in sheet["mistakes"]]
    return "\n".join(lines)


def helpers_text(port_text: str) -> str:
    """The signature and first docstring line of each generated helper the port has."""
    try:
        tree = ast.parse(port_text)
    except (SyntaxError, ValueError, RecursionError):
        return ""
    out = []
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in HELPER_NAMES:
            doc = (ast.get_docstring(fn) or "").split("\n")[0]
            out.append(f"- {fn.name}({', '.join(_arg_names(fn))}): {doc}")
    return "\n".join(out)


def _hc_count(tree: ast.Module):
    """CONFIG['hc_count'] of the port's module-level CONFIG literal, or None."""
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets):
            try:
                value = ast.literal_eval(n.value)
            except (ValueError, SyntaxError, RecursionError, MemoryError):
                return None
            v = value.get("hc_count") if isinstance(value, dict) else None
            return v if isinstance(v, int) and not isinstance(v, bool) else None
    return None


def model_text(port_text: str) -> str:
    """The section of a request that says what the arguments of a port function are: `model` is a plain dict with the keys `load` sets, the
    module constants hold the config and the sizes, and a torch-style attribute does not exist. Derived from port_template.MODEL_KEYS and
    MODULE_CONSTANTS (a test holds them equal to the generated text); only the constants and helpers the given port has are listed."""
    try:
        tree = ast.parse(port_text)
    except (SyntaxError, ValueError, RecursionError):
        tree = ast.Module(body=[], type_ignores=[])
    defined = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    helpers = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPER_NAMES]
    lines = ["## The arguments of a port function",
             "`model` is a plain dict, not an object: read `model['dev']`, never `model.dev`. Its keys are exactly:"]
    lines += [f"- `{k}`: {v}" for k, v in port_template.MODEL_KEYS.items()]
    lines += ["The module constants of the port (read them directly):"]
    lines += [f"- `{k}`: {v}" for k, v in port_template.MODULE_CONSTANTS.items() if k in defined]
    hc = _hc_count(tree)
    if hc and hc > 1:
        lines += [f"The residual stream has hc_count = {hc} streams (CONFIG['hc_count']), laid out side by side: between the layers x is "
                  f"[1, 1, S, HC * HIDDEN], stream j in the columns j * HIDDEN to (j + 1) * HIDDEN. The prelude widens the embedding to that shape "
                  "(the embedding repeated HC times along the last dimension), every layer reads and returns that shape, and the epilogue narrows "
                  "it back to [1, 1, S, HIDDEN] before the head."]
    lines += ["There is no model.config, no rotary_emb and no past_key_values: build position tables from CONFIG with the helpers the port "
              "has (" + (", ".join(f"`{h}`" for h in helpers) or "none") + "), and keep any cache in a local or in model['ids']."]
    return "\n".join(lines)


def stub_request(port_text: str, stub, *, sheet: dict, excerpts: str | None, construction_text: str | None = None, inventory=None) -> dict:
    """The chat request for `stub` (a port_template.Stub, or a dict with function, cls and path): {"system", "user", "function", "args"}.
    The user text puts the shared parts first and the stub last."""
    function = stub["function"] if isinstance(stub, dict) else stub.function
    cls = stub["cls"] if isinstance(stub, dict) else stub.cls
    path = stub["path"] if isinstance(stub, dict) else stub.path
    source = function_source(port_text, function)
    args = _arg_names(_functions(ast.parse(port_text))[function])
    wanted, other = ops_for(inventory, path)
    parts = [model_text(port_text), "## Helpers of the port (call these; do not rewrite them)", helpers_text(port_text) or "(none)",
             "## Verified ttnn calls", sheet_section(sheet, wanted=wanted)]
    if other:
        parts += ["## Ops of this block that have no verified ttnn form (stage 0 labels)", ", ".join(other),
                  "These may run in torch on the host between _to_host and _upload."]
    lines = construction_lines(construction_text, cls)
    if lines:
        parts += [f"## Resolved settings of {cls} (effective_construction.txt)", *["- " + ln for ln in lines]]
    excerpt = class_excerpt(excerpts, cls)
    extra_excerpt = None
    if function == "_model_prelude":
        fence = re.search(r"```python\n(.*?)```", excerpt or "", re.S)
        piece = forward_part(fence.group(1), "before") if fence else None
        if piece:
            excerpt = (f"(the part of forward before the layer loop; the layers themselves, the position tables and the mask are not written "
                       f"by this function)\n\n```python\n{piece}```")
    elif function.startswith("_model_") and isinstance(excerpts, str):
        # An epilogue stub belongs to a child of the text model (its mixer): its own class is given whole, and the part of the text model's forward
        # after the layer loop shows where it is called.
        for sec in re.finditer(r"^## (\S+)\s*$", excerpts, re.M):
            code = class_excerpt(excerpts, sec.group(1)) or ""
            fence = re.search(r"```python\n(.*?)```", code, re.S)
            piece = forward_part(fence.group(1), "after") if fence else None
            if piece:
                extra_excerpt = (f"## Where the text model's forward uses it (the part of forward after the layer loop)\n\n```python\n{piece}```")
                break
    if "GatedDeltaNet" in cls or "linear_attn" in path:
        parts += ["## How to write this block",
                  "Write the recurrence on the host as a plain for loop over the tokens, with no chunking and no helper classes: read x back with "
                  "_to_host, loop over the tokens updating the state, upload the result with _upload, and return it. Keep the function under 60 lines; "
                  "a long reply is cut off."]
    parts += [f"## Reference class {cls} (stages/2/reference_excerpts.md)", excerpt or "(the excerpts file has no section for this class; read it from the installed transformers package)"]
    if extra_excerpt:
        parts.append(extra_excerpt)
    parts += [f"## The function to write: {function} for {cls} at {path}", "```python", source, "```",
              f"Write `{function}` with these arguments: {', '.join(args)}. Return a ttnn tensor of the shape the layer expects."]
    return {"system": RULES, "user": "\n\n".join(parts), "function": function, "args": args}


def repair_request(port_text: str, function: str, *, last_check: str, layer_diff: str | None = None, sheet: dict | None = None) -> dict:
    """A repair request for the function the last check named: its current source, the check's report (cut), the layer-diff report when
    there is one, and the sheet's mistakes. The reply format is the same as for a stub."""
    source = function_source(port_text, function)
    args = _arg_names(_functions(ast.parse(port_text))[function])
    parts = [model_text(port_text), "## What the last check said", (last_check or "")[:CHECK_MAX_CHARS]]
    if layer_diff:
        parts += ["## The first layer that differs from the CPU reference (layer-diff)", layer_diff[:DIFF_MAX_CHARS]]
    if sheet and sheet.get("mistakes"):
        parts += ["## Mistakes seen in runs", *[f"- {m['symptom']} Fix: {m['fix']}" for m in sheet["mistakes"]]]
    parts += [f"## The function to fix: {function}", "```python", source, "```",
              f"Rewrite `{function}` with the same arguments ({', '.join(args)}) so the check passes. Change only what the report names."]
    return {"system": RULES, "user": "\n\n".join(parts), "function": function, "args": args}


def extract_code(reply: str) -> str:
    """The code block of a reply (a fenced block, or the whole reply when it has no fence and parses as Python)."""
    if not isinstance(reply, str) or len(reply) > REPLY_MAX_CHARS:
        raise SpliceError("the reply is not a string of bounded size")
    blocks = _FENCE.findall(reply)
    if len(blocks) > 1:
        raise SpliceError(f"the reply holds {len(blocks)} code blocks; one is expected")
    return blocks[0] if blocks else reply


# What a port function has no use for. This is a guard on a coder reply, not a sandbox: the coder can already write any file in the port
# directory, and the port runs with the run's own settings. The guard keeps a reply from reaching the host by accident or by a trick the
# plain import list would miss (review of PR 685). Names that run code from a string, open files, load modules by name, reach the
# interpreter's internals or unpickle are refused anywhere in the reply, including default arguments and decorators, which run when the
# port is imported.
DENIED_MODULES = ("transformers", "subprocess", "os", "sys", "socket", "urllib", "requests", "importlib", "ctypes", "pickle", "marshal",
                  "shutil", "pathlib", "builtins", "runpy", "code", "codeop", "pty", "tempfile", "glob", "http", "ftplib", "smtplib",
                  "multiprocessing", "signal", "inspect", "gc", "atexit", "pdb", "site", "sysconfig", "webbrowser")
# The modules a port function may import (arm E3: replies imported modules the port does not have, such as `.helpers` and `port_helpers`, and
# the port failed at init on every board check). The denied list above stays as the reason given for the dangerous ones.
ALLOWED_MODULES = ("ttnn", "torch", "math", "numpy", "re", "functools", "itertools", "typing", "collections")
DENIED_CALLS = ("exec", "eval", "compile", "open", "input", "breakpoint", "setattr", "delattr", "globals", "locals", "vars")
DENIED_TORCH_CALLS = ("load", "save")


def _dunder(name) -> bool:
    return isinstance(name, str) and len(name) > 4 and name.startswith("__") and name.endswith("__")


def _refuse_unusable(tree: ast.Module) -> None:
    """Raises SpliceError for the first thing in the reply's syntax tree that a port function has no use for."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            raise SpliceError(f"{node.name} has a decorator; a decorator runs when the port is imported, and no port function needs one")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise SpliceError("the reply has a relative import; the port is one file, so there is no package: call its helpers by name")
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(n.split(".")[0] in DENIED_MODULES for n in names):
                raise SpliceError("the reply imports a module a port function may not use")
            outside = [n.split(".")[0] for n in names if n.split(".")[0] not in ALLOWED_MODULES]
            if outside:
                raise SpliceError(f"the reply imports {outside[0]}, which this port cannot import: the port is one file, so call its helpers by "
                                  f"name; the modules a port function may import are {', '.join(ALLOWED_MODULES)}")
        elif isinstance(node, ast.Name) and _dunder(node.id):
            raise SpliceError(f"the reply uses the dunder name {node.id}, which a port function has no use for")
        elif isinstance(node, ast.Attribute) and _dunder(node.attr):
            raise SpliceError(f"the reply uses the dunder attribute {node.attr}, which a port function has no use for")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in DENIED_CALLS:
                raise SpliceError(f"the reply may not call {fn.id}")
            if isinstance(fn, ast.Name) and fn.id == "getattr":
                second = node.args[1] if len(node.args) > 1 else None
                if not (isinstance(second, ast.Constant) and isinstance(second.value, str)):
                    raise SpliceError("getattr needs a constant attribute name in a port function")
                if second.value.startswith("_"):
                    raise SpliceError(f"getattr of {second.value!r}: a dunder or private attribute name is not allowed (dunder)")
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "torch" and fn.attr in DENIED_TORCH_CALLS:
                raise SpliceError(f"the reply may not call torch.{fn.attr}; the weights are loaded by load()")
        elif isinstance(node, ast.Raise) and port_template._stub_raise_text(node) is not None:
            raise SpliceError("the reply keeps the generated stub text (a NotImplementedError that says the block is not written yet)")


def splice(port_text: str, function: str, reply: str) -> str:
    """`port_text` with the stub function `function` replaced by the reply's definition of it. Raises SpliceError when the reply is not one
    code block of imports and function definitions that defines `function` with the stub's argument names, keeps the marker, or does not
    leave a port that parses. Extra functions in the reply are placed before the replaced one."""
    old = ast.parse(port_text)
    stub = _functions(old).get(function)
    if stub is None:
        raise SpliceError(f"the port has no top-level function {function}")
    code = extract_code(reply)
    try:
        new = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise SpliceError(f"the reply does not parse: {type(exc).__name__}") from None
    for node in new.body:
        if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            raise SpliceError(f"the reply has a top-level {type(node).__name__}; only imports and function definitions are accepted")
    fns = _functions(new)
    if function not in fns:
        raise SpliceError(f"the reply does not define {function}")
    if _arg_names(fns[function]) != _arg_names(stub):
        raise SpliceError(f"{function} must take exactly these arguments: {', '.join(_arg_names(stub))}")
    for name in fns:
        if name != function and name in _functions(old):
            raise SpliceError(f"the reply redefines {name}, which the port already has")
    if any(l.strip().startswith(port_template.STUB_MARK) for l in code.split("\n")):
        raise SpliceError("the reply keeps the stub marker")
    _refuse_unusable(new)
    lines = port_text.split("\n")
    new_lines = lines[: stub.lineno - 1] + code.strip("\n").split("\n") + lines[stub.end_lineno:]
    text = "\n".join(new_lines)
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise SpliceError(f"the spliced port does not parse: {type(exc).__name__}") from None
    return text
