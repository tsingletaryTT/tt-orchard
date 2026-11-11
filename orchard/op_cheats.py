# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Verified cheat sheets for the coder (issue 523): the TTNN calls a full-port decoder needs, and the TT-Lang calls a kernel needs.

Why. In the CAD benchmark a compact API sheet in the prompt took one task from no valid part to a perfect score; most failed runs called
functions that do not exist. Our stage 2 runs show the same pattern (the coder guesses TTNN calls and the port fails at prefill).

What. A sheet is a JSON data file (`ttnn_cheat_sheet.json`, `ttlang_cheat_sheet.json`) with one entry per call: name, signature, one working usage
line, a note, and where the call was seen in a port that ran. `render` turns the data into the Markdown the coder reads. `verify` checks every
entry against a source tree on the CPU. Nothing here imports ttnn, ttl or ttml: importing them opens the device. The check reads text.

How an entry is verified (`verify`):
- The name is found in each root the entry claims (`in`). A root is a source tree. For TTNN it is the tt-metal tree at the pinned commit; for
  TT-Lang it is one of `hardware` (the installed package that compiles for the chip), `tree` (the tt-lang checkout's compiler package) and `sim`
  (the checkout's simulator).
- A C++ binding is found by `bind_function<"name">(...)` and its argument names are the `nb::arg("...")` entries inside that call. A Python
  function is found with `ast`, and its argument names are its parameters. A name that is only a registered symbol (a layout, a unary
  or binary op bound by a template) is checked for existence, and its entry may not use keyword arguments.
- Every name in the signature is an argument of one definition or binding in the source, and every keyword used in the usage
  line is in the signature. This catches an invented argument.
- `status` (TT-Lang entries) must be supported: `compiles on hardware` needs the call in the `hardware` root and an `evidence` field;
  `simulator only` needs a name or argument the `hardware` root lacks, and the entry lists it in `hardware_absent`; and
  `hardware signature only` says the signature is in the hardware package and no compile was shown.
- A mistake may carry `checks`: each check says an argument a coder would guess is absent from a root (or present, for a correction).
- `reported` is a separate, optional list of pitfalls another project wrote down (symptom, fix, source, and a `reproduced` flag that is true only
  when it was seen on our pin). `verify` makes no claim about them, and `render` prints them under their own heading with a label per item, so a
  coder can tell them from the mistakes our own runs showed.

The reader of a tree and of the sheet uses orchard/safeio.py. The tree is outside input: it is walked by descriptor and never followed through
a link, and each file read has a byte cap.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from orchard import safeio

SCHEMA = "orchard-cheat-sheet/1"
SHEET_MAX_BYTES = 400_000
FILE_MAX_BYTES = 4_000_000                 # one source file of a tree (the largest nanobind file is well under this)
MAX_ENTRIES = 200
MAX_REPORTED = 20                          # reported pitfalls a sheet may carry (what another project saw and wrote down; our runs have not shown them)
MAX_FIELD = 600                            # characters in a signature, usage line or note
SKIP_DIRS = frozenset({"__pycache__", "_mlir_libs", "dialects", "extras", "third-party", "build", "tests", "test", ".git", "sim", "sim_stats"})   # the installed package ships its own simulator under sim/: it must not count as the compiled surface
STATUSES = ("compiles on hardware", "hardware signature only", "simulator only")
KINDS = ("op", "symbol", "method")
RENDER_LINE_BUDGET = 300                   # the rendered sheet stays under this many lines (a test keeps it); 250 at first, 270 when the PLE gate ops joined (issue 696, slice 3), 300 when the arm C to E14 failures joined as 13 mistakes and 3 entries (issue 677); mistakes print in every stub and repair request, so each one is kept to a line or two
HERE = Path(__file__).resolve().parent
TTNN_SHEET = HERE / "skills" / "full-port-templates" / "ttnn_cheat_sheet.json"
TTLANG_SHEET = HERE / "skills" / "kernel-loop-templates" / "ttlang_cheat_sheet.json"

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,4}")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NB_ARG = re.compile(r'nb::arg\("(\w+)"\)|"(\w+)"_a')


class SheetError(Exception):
    """A sheet file that cannot be read or has the wrong shape. The message never holds the file's content."""


# ---- the sheet ------------------------------------------------------------------------------------------------------------------------

def load_sheet(path) -> dict:
    """The sheet at `path` through the bounded reader, checked for shape (`shape_problems`). Raises SheetError."""
    try:
        data = safeio.read_json(path, max_bytes=SHEET_MAX_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        raise SheetError(f"{Path(path).name}: {exc}") from None
    problems = shape_problems(data)
    if problems:
        raise SheetError(f"{Path(path).name}: " + "; ".join(problems[:5]))
    return data


def _text(value, what: str, problems: list, *, required: bool = True) -> str:
    if not isinstance(value, str) or (required and not value.strip()):
        problems.append(f"{what} must be a non-empty string")
        return ""
    if len(value) > MAX_FIELD or "\0" in value:
        problems.append(f"{what} is over {MAX_FIELD} characters or holds a NUL")
        return ""
    return value


def shape_problems(sheet) -> list[str]:
    """What is wrong with the sheet's shape (types, required fields, bounds), without looking at any source tree."""
    problems: list[str] = []
    if not isinstance(sheet, dict):
        return ["the sheet must be a JSON object"]
    if sheet.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    if sheet.get("sheet") not in ("ttnn", "ttlang"):
        problems.append("sheet must be 'ttnn' or 'ttlang'")
    _text(sheet.get("title"), "title", problems)
    roots = sheet.get("roots")
    if not isinstance(roots, dict) or not roots or not all(isinstance(k, str) and isinstance(v, str) for k, v in roots.items()):
        problems.append("roots must be an object of root name to a description")
        roots = {}
    sections = sheet.get("sections")
    if not isinstance(sections, list) or not sections:
        return problems + ["sections must be a non-empty list"]
    names: set = set()
    count = 0
    for si, section in enumerate(sections):
        if not isinstance(section, dict) or not isinstance(section.get("entries"), list):
            problems.append(f"section {si} must be an object with a list of entries")
            continue
        _text(section.get("title"), f"section {si} title", problems)
        for entry in section["entries"]:
            count += 1
            where = f"entry {entry.get('name')!r}" if isinstance(entry, dict) else f"entry {count}"
            if not isinstance(entry, dict):
                problems.append(f"{where} must be an object")
                continue
            name = _text(entry.get("name"), f"{where} name", problems)
            if name and not _NAME.fullmatch(name):
                problems.append(f"{where}: the name is not a dotted identifier")
            if name in names:
                problems.append(f"{where}: the name appears twice")
            names.add(name)
            if entry.get("kind") not in KINDS:
                problems.append(f"{where}: kind must be one of {KINDS}")
            for key in ("signature", "usage", "notes", "seen_in"):
                _text(entry.get(key), f"{where} {key}", problems)
            claimed = entry.get("in")
            if not isinstance(claimed, list) or not claimed or not all(isinstance(r, str) and r in roots for r in claimed):
                problems.append(f"{where}: `in` must list roots named in `roots`")
            status = entry.get("status")
            if sheet.get("sheet") == "ttlang":
                if status not in STATUSES:
                    problems.append(f"{where}: status must be one of {STATUSES}")
                if status == "compiles on hardware":
                    _text(entry.get("evidence"), f"{where} evidence", problems)
                if status == "simulator only" and not (isinstance(entry.get("hardware_absent"), list) and entry["hardware_absent"]):
                    problems.append(f"{where}: a simulator-only entry lists what the hardware package lacks in `hardware_absent`")
    if count > MAX_ENTRIES:
        problems.append(f"more than {MAX_ENTRIES} entries")
    problems += _reported_problems(sheet.get("reported", []))
    mistakes = sheet.get("mistakes", [])
    if not isinstance(mistakes, list):
        return problems + ["mistakes must be a list"]
    for i, m in enumerate(mistakes):
        if not isinstance(m, dict):
            problems.append(f"mistake {i} must be an object")
            continue
        for key in ("symptom", "fix", "seen_in"):
            _text(m.get(key), f"mistake {i} {key}", problems)
        checks = m.get("checks", [])
        if not isinstance(checks, list):
            problems.append(f"mistake {i}: checks must be a list")
            continue
        for c in checks:
            if not isinstance(c, dict) or not _check_is_well_typed(c, roots):
                problems.append(f"mistake {i}: each check names a root, a name and absent_args, present_args or absent")
    return problems


def _reported_problems(reported) -> list[str]:
    """What is wrong with a sheet's `reported` list: pitfalls another project saw and wrote down, kept apart from `mistakes` (failures seen in our own
    runs). Each item has a `symptom`, a `fix` and a `source` (non-empty text) and a `reproduced` flag that is exactly true or false: true only when
    the pitfall was seen on our pin. The flag decides the label the coder reads, so a string or a number in its place is refused."""
    if not isinstance(reported, list):
        return ["reported must be a list"]
    problems: list[str] = []
    if len(reported) > MAX_REPORTED:
        problems.append(f"more than {MAX_REPORTED} reported entries")
    for i, item in enumerate(reported):
        if not isinstance(item, dict):
            problems.append(f"reported {i} must be an object")
            continue
        for key in ("symptom", "fix", "source"):
            _text(item.get(key), f"reported {i} {key}", problems)
        if not isinstance(item.get("reproduced"), bool):
            problems.append(f"reported {i}: reproduced must be true or false")
    return problems


def _check_is_well_typed(c: dict, roots) -> bool:
    """True when a mistake check has a string root named in `roots`, a string name, list-valued argument lists of strings and at least one claim.
    Every type is tested before it is used, so a null or list value is reported as a problem and never raises."""
    root, name = c.get("root"), c.get("name")
    if not isinstance(root, str) or root not in roots or not isinstance(name, str):
        return False
    lists = []
    for key in ("absent_args", "present_args"):
        value = c.get(key, [])
        if not isinstance(value, list) or not all(isinstance(a, str) for a in value):
            return False
        lists.append(value)
    return bool(lists[0] or lists[1] or c.get("absent"))


def entries(sheet: dict):
    """Every entry of the sheet in order."""
    for section in sheet["sections"]:
        yield from section["entries"]


def find_entry(sheet: dict, name: str) -> dict | None:
    """The entry called `name`, or the only entry whose last name part is `name`; None when there is none or the short name is ambiguous."""
    exact = [e for e in entries(sheet) if e["name"] == name]
    if exact:
        return exact[0]
    short = [e for e in entries(sheet) if e["name"].rsplit(".", 1)[-1] == name]
    return short[0] if len(short) == 1 else None


# ---- a source tree --------------------------------------------------------------------------------------------------------------------

class Index:
    """What a source tree says about names: for each name, the argument names of its definitions and whether it was found at all.

    `args[name]` is the set of argument names from every binding or definition of that last name part; `qualified["Class.name"]` the same for a method;
    `strings` is the text of every nanobind file, for a name that is only registered (a symbol); `where[name]` lists the files (relative paths)."""

    def __init__(self):
        self.args: dict[str, set] = {}
        self.order: dict[str, list] = {}
        self.where: dict[str, list] = {}
        self.qualified: dict[str, set] = {}
        self.alias: dict[str, str] = {}
        self.imported: set = set()
        self.text: list[str] = []
        self.text_where: list[str] = []      # the relative path of each entry of `text`, for the namespace filter of a template registration (issue 606)
        self.files = 0
        self.incomplete = False
        self.skipped: list[str] = []         # files that were over the size limit or could not be read: a definition in one of them is not in the index

    def _add(self, name: str, args, where: str) -> None:
        self.args.setdefault(name, set()).update(args)
        self.order.setdefault(name, [])
        self.order[name].append(list(args))
        self.where.setdefault(name, []).append(where)


def _cpp_blocks(text: str):
    """(name, argument names) for each `bind_function<"name"...>(...)` call in a C++ file. The call is taken to its matching parenthesis."""
    for m in re.finditer(r'bind_function<\s*"(\w+)"', text):
        start = text.find("(", m.end())
        if start < 0:
            continue
        depth, k = 0, start
        while k < len(text):
            c = text[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        yield m.group(1), [a or b for a, b in _NB_ARG.findall(text[start:k])]


_CLASS = re.compile(r'(?:tt_serializable_class|nb::class_)<[^>]*>\(\s*\w+\s*,\s*"(\w+)"')
_FIELD = re.compile(r'\.def_(?:rw|ro)\("(\w+)"')


_DEF_STATEMENT = re.compile(r'\.def\w*\(')


def _matching_paren(text: str, open_at: int) -> int:
    """The index of the `)` that closes the `(` at `open_at`, or len(text) when it is never closed. Parentheses inside a double- or
    single-quoted literal (with backslash escapes), a `//` comment or a `/* */` comment do not count (issue 605: a constructor binding ends at
    its own closing parenthesis, not at the next class or the end of the file)."""
    depth, i, n = 0, open_at, len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            i += 1
            while i < n and text[i] != ch:
                i += 2 if text[i] == "\\" else 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n


def _cpp_classes(text: str):
    """(class name, one argument list per constructor binding) for each bound class in a C++ file. The text from one class binding to the next
    is cut into `.def...(` statements; each `.def(nb::init<...>(), ...)` or `.def("__init__", ...)` is a constructor and its `nb::arg` names are one overload. Method
    arguments (`.def("method", ..., nb::arg(...))`) and properties (`.def_rw`, `.def_ro`) are not constructor arguments. A class with no
    constructor binding is one overload with no arguments (the name exists; nothing is read)."""
    marks = list(_CLASS.finditer(text))
    for i, m in enumerate(marks):
        region = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        found = list(_DEF_STATEMENT.finditer(region))
        ctors = []
        for d in found:
            st = d.start()
            statement = region[st: _matching_paren(region, d.end() - 1) + 1]          # this call only (issue 605)
            if re.match(r'\.def\(\s*(?:nb::init\b|"__init__")', statement):       # `.def(nb::init<...>(), ...)` or a `.def("__init__", lambda, ...)`
                ctors.append([a or b for a, b in _NB_ARG.findall(statement)])
        yield m.group(1), ctors or [[]]


def _params(fn) -> list:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    return [n for n in names if n not in ("self", "ast_self", "cls")]


def _index_python(index: Index, source: str, rel: str) -> None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            index._add(node.name, _params(node), rel)
        elif isinstance(node, ast.ClassDef):
            ctor = [i for i in node.body if isinstance(i, ast.FunctionDef) and i.name == "__init__"]
            index._add(node.name, _params(ctor[0]) if ctor else [], rel)      # a class is called with its __init__ arguments
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    index.qualified.setdefault(f"{node.name}.{item.name}", set()).update(_params(item))
                    index.where.setdefault(f"{node.name}.{item.name}", []).append(rel)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    index._add(t.id, [], rel)               # a name bound at module level (TILE_LAYOUT = ...): it exists, no arguments to read
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.asname and a.asname != a.name:
                    index.alias[a.asname] = a.name
                index.imported.add(a.asname or a.name)      # re-exported from a compiled module (ttnn/__init__.py): the name exists


def build_index(root, *, max_depth: int = 24) -> Index:
    """Index the source tree at `root`: the `*nanobind.cpp` files (bindings) and the `.py` files (definitions). Walks by descriptor and
    reads each file through `safeio.read_bytes`; a file that cannot be read is skipped and sets `incomplete`."""
    index = Index()
    try:
        pin_cm = safeio.pinned_root(root)
        pin = pin_cm.__enter__()
    except (safeio.SafeReadError, OSError, ValueError):
        index.incomplete = True
        return index
    try:
        found = safeio.scan_beneath(pin, max_depth=max_depth, max_entries=400_000)
        index.incomplete = found.incomplete
        if found.incomplete:
            index.skipped.append("(the directory walk stopped early: too deep or too many entries)")
        for entry in found.entries:
            if entry.kind != "file" or any(p in SKIP_DIRS for p in entry.parts[:-1]):
                continue
            name = entry.parts[-1]
            in_binding_dir = any("nanobind" in p for p in entry.parts[:-1]) and name.endswith(".cpp")
            if not (name.endswith(".py") or name.endswith("nanobind.cpp") or in_binding_dir):
                continue
            try:
                data = safeio.read_beneath(pin.fd, entry.parts, max_bytes=FILE_MAX_BYTES)
            except (safeio.SafeReadError, OSError, ValueError):
                index.incomplete = True
                index.skipped.append("/".join(entry.parts))
                continue
            rel = "/".join(entry.parts)
            source = data.decode("utf-8", "replace")
            index.files += 1
            if name.endswith(".py"):
                _index_python(index, source, rel)
            else:
                index.text.append(source)
                index.text_where.append(rel)
                for leaf, args in _cpp_blocks(source):
                    index._add(leaf, args, rel)
                for leaf, overloads in _cpp_classes(source):
                    for args in overloads:
                        index._add(leaf, args, rel)           # each constructor binding is its own overload (issue 531)
    finally:
        pin_cm.__exit__(None, None, None)
    return index


_SPACES = ("experimental", "transformer")


def _space_of(parts: list):
    """The namespace part of a dotted name (`experimental` in `ttnn.experimental.foo`), or None."""
    return parts[-2] if len(parts) >= 3 and parts[-2] in _SPACES else None


def _bindings(index: Index, name: str) -> list:
    """(argument list, file) for each separate definition or binding of the last part of `name`, an alias followed once. When the name has a
    namespace part (`ttnn.experimental.foo`), only the bindings whose file path contains it count, for the signature check and for the
    lookup alike (issue 532)."""
    parts = name.split(".")
    leaf = parts[-1]
    space = _space_of(parts)
    out = []
    for key in (leaf, index.alias.get(leaf)):
        if key is None:
            continue
        for args, where in zip(index.order.get(key, []), index.where.get(key, [])):
            if space is None or space in where:
                out.append((list(args), where))
    return out


def alternatives(index: Index, name: str) -> list:
    """The argument lists of each separate definition or binding of `name` (an overload is one list). A signature must fit inside one of them."""
    parts = name.split(".")
    if len(parts) >= 2 and f"{parts[-2]}.{parts[-1]}" in index.qualified:
        return [list(index.qualified[f"{parts[-2]}.{parts[-1]}"])]
    return [args for args, _ in _bindings(index, name)]


def _registered(leaf: str, text: str) -> bool:
    """Whether a C++ file registers `leaf` as a name: a template registration (`bind_unary<"relu">`) or a binding call that names it first
    (`.def("leaf"`, `.def_static("leaf"`, `.attr("leaf")`, `.value("leaf"`). A quoted word elsewhere (`nb::arg("leaf")`, a docstring) is not one."""
    n = re.escape(leaf)
    return bool(re.search(r'<\s*"%s"\s*[>,]|\.(?:def|def_static|attr|value)\(\s*"%s"' % (n, n), text))


def resolve(index: Index, name: str):
    """(found, argument names or None, where) for a dotted `name`. A method is `Class.method`; otherwise only the last part is looked up
    (the namespace prefix, `experimental` or `transformer`, is checked against the path of the C++ file, and the arguments come only from
    bindings under it). An alias (`operation` for `pykernel_gen`) is followed once."""
    parts = name.split(".")
    if len(parts) >= 2 and f"{parts[-2]}.{parts[-1]}" in index.qualified:
        key = f"{parts[-2]}.{parts[-1]}"
        return True, set(index.qualified[key]), index.where.get(key, [])
    leaf = parts[-1]
    bound = _bindings(index, name)
    if bound:
        args = set()
        for a, _ in bound:
            args.update(a)
        return True, args, [w for _, w in bound]
    if leaf in index.args or leaf in index.alias and index.alias[leaf] in index.args:
        return False, None, []                # bound, but not under the requested namespace
    if leaf in index.imported:
        return True, None, []
    space = _space_of(parts)
    for i, t in enumerate(index.text):
        where = index.text_where[i] if i < len(index.text_where) else ""
        if (space is None or space in where) and _registered(leaf, t):
            return True, None, []             # registered by a template or class: the name exists, its arguments are not read (under the namespace, issue 606)
    return False, None, []


# ---- verification ---------------------------------------------------------------------------------------------------------------------

def signature_names(signature: str) -> list[str]:
    """The argument names in a signature string such as `ttnn.linear(a, b, *, bias=None)`: identifiers before `=`, `,` or the closing parenthesis."""
    start, end = signature.find("("), signature.rfind(")")
    if start < 0 or end < start:
        return []
    names = []
    for piece in _split_top(signature[start + 1:end]):
        piece = piece.strip().lstrip("*").strip()
        m = _IDENT.match(piece)
        if m and piece not in ("", "/"):
            names.append(m.group(0))
    return names


def _split_top(text: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur] if cur.strip() else out


def usage_keywords(usage: str, name: str) -> tuple[set, bool]:
    """(keyword names used in calls to `name` within the usage line, whether the usage parsed and names the call). `usage` may be several
    statements; a call is any Call whose callee text ends with the entry's last name part."""
    leaf = name.rsplit(".", 1)[-1]
    try:
        tree = ast.parse(usage.strip())
    except (SyntaxError, ValueError, RecursionError):
        return set(), False
    kw, seen = set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            callee = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if callee == leaf:
                seen = True
                kw.update(k.arg for k in node.keywords if k.arg)
    return kw, seen


def verify(sheet: dict, indexes: dict) -> list[str]:
    """Problems found by checking `sheet` against `indexes` (root name to Index). An empty list means every entry's name and arguments were found.
    A root that was claimed and has no index is a problem: nothing was checked for it."""
    problems = shape_problems(sheet)
    if problems:
        return problems
    for entry in entries(sheet):
        name, where = entry["name"], f"{entry['name']}"
        sig = signature_names(entry["signature"])
        for root in entry["in"]:
            index = indexes.get(root)
            if index is None or index.files == 0:
                unread = f"root {root!r} was not read, so nothing was checked for it"
                if unread not in problems:                      # said once for the sheet, not once per entry
                    problems.append(unread)
                continue
            found, args, _ = resolve(index, name)
            if not found:
                problems.append(f"{where}: the name was not found in root {root!r}")
                continue
            if args is None:
                # The name is registered by a template or re-exported: its argument names cannot be read, so the signature's positional names
                # are not checked and the usage may not pass any keyword.
                if usage_keywords(entry["usage"], name)[0]:
                    problems.append(f"{where}: root {root!r} registers the name without arguments to read, so the usage may not pass a keyword")
                continue
            alts = alternatives(index, name)
            if sig and not any(all(a in alt for a in sig) for alt in alts):
                missing = [a for a in sig if a not in args]
                problems.append(f"{where}: " + (f"argument {missing[0]!r} is not an argument of the call in root {root!r}" if missing else
                                                f"no single definition in root {root!r} has all of the signature's arguments"))
        kw, parsed = usage_keywords(entry["usage"], name)
        if not parsed and entry["kind"] == "symbol" and not sig:
            parsed = name.rsplit(".", 1)[-1] in entry["usage"]      # a constant is used, not called
        if not parsed:
            problems.append(f"{where}: the usage line does not parse as Python and call the name")
        for k in kw:
            if k not in sig:
                problems.append(f"{where}: the usage passes {k!r}, which the signature does not list")
        if sheet["sheet"] == "ttlang":
            problems += _status_problems(entry, indexes)
    for i, m in enumerate(sheet.get("mistakes", [])):
        for c in m.get("checks", []):
            index = indexes.get(c["root"])
            if index is None or index.files == 0:
                problems.append(f"mistake {i}: root {c['root']!r} was not read")
                continue
            found, args, _ = resolve(index, c["name"])
            if (c.get("absent") or c.get("absent_args")) and index.incomplete:
                problems.append(f"mistake {i}: {c['name']} is claimed absent from {c['root']!r}, but the index of that root is incomplete "
                                f"(skipped: {', '.join(index.skipped[:3]) or 'unknown'}), so absence cannot be certified")
                continue
            if c.get("absent"):
                if found:
                    problems.append(f"mistake {i}: {c['name']} is claimed absent from {c['root']!r} but was found")
                continue
            if not found or args is None:
                problems.append(f"mistake {i}: {c['name']} was not found in {c['root']!r}")
                continue
            for a in c.get("absent_args", []):
                if a in args:
                    problems.append(f"mistake {i}: {a!r} is claimed absent from {c['name']} in {c['root']!r} but is there")
            for a in c.get("present_args", []):
                if a not in args:
                    problems.append(f"mistake {i}: {a!r} is claimed present in {c['name']} in {c['root']!r} but is not")
    return problems


def _status_problems(entry: dict, indexes: dict) -> list[str]:
    """The checks behind an entry's `status` (TT-Lang sheet)."""
    status, name, out = entry["status"], entry["name"], []
    hw = indexes.get("hardware")
    if hw is None or hw.files == 0:
        return [f"{name}: the hardware root was not read, so its status was not checked"]
    found, args, _ = resolve(hw, name)
    sig = signature_names(entry["signature"])
    if status in ("compiles on hardware", "hardware signature only"):
        if "hardware" not in entry["in"]:
            out.append(f"{name}: status {status!r} needs the hardware root in `in`")
        if not found:
            out.append(f"{name}: status {status!r} but the hardware root lacks the name")
        elif args is not None and any(a not in args for a in sig):
            out.append(f"{name}: status {status!r} but the signature has an argument the hardware package lacks")
    else:
        if hw.incomplete and entry.get("hardware_absent"):
            return [f"{name}: the hardware index is incomplete (skipped: {', '.join(hw.skipped[:3]) or 'unknown'}), so a claim that the hardware "
                    f"package lacks something cannot be certified"]
        if "hardware" in entry["in"]:
            out.append(f"{name}: a simulator-only entry does not claim the hardware root")
        leaf = name.rsplit(".", 1)[-1]
        for item in entry["hardware_absent"]:
            if not isinstance(item, str) or not item:
                out.append(f"{name}: hardware_absent holds a non-string")
            elif item == leaf or "." in item:                   # a whole name the hardware package lacks
                if resolve(hw, item)[0]:
                    out.append(f"{name}: {item!r} is listed as absent from the hardware package but is there")
            elif not found:
                out.append(f"{name}: argument {item!r} is listed as absent, but the hardware package lacks the whole name (list the name instead)")
            elif args is not None and item in args:
                out.append(f"{name}: argument {item!r} is listed as absent from the hardware package but is there")
    return out


# ---- rendering ------------------------------------------------------------------------------------------------------------------------

def render(sheet: dict) -> str:
    """The Markdown the coder reads: one block per entry, then the mistakes. The sheet's own text is used as given; nothing is added."""
    lines = [f"# {sheet['title']}", ""]
    for key, value in sheet["roots"].items():
        lines.append(f"- source `{key}`: {value}")
    if sheet.get("intro"):
        lines += ["", sheet["intro"]]
    for section in sheet["sections"]:
        lines += ["", f"## {section['title']}"]
        for e in section["entries"]:
            tag = f" [{e['status']}]" if e.get("status") else ""
            lines += ["", f"`{e['signature']}`{tag}", *["    " + ln for ln in e["usage"].split("\n")], f"    {e['notes']} (seen: {e['seen_in']})"]
    if sheet.get("mistakes"):
        lines += ["", "## Mistakes seen in runs", ""]
        for m in sheet["mistakes"]:
            lines.append(f"- {m['symptom']} Fix: {m['fix']} ({m['seen_in']})")
    reported = sheet.get("reported") or []
    if reported:
        # The heading says "not reproduced" only when no item was reproduced; each line carries its own label either way.
        unreproduced = not any(r["reproduced"] for r in reported)
        lines += ["", "## Reported by another project" + (" (not reproduced on our pin)" if unreproduced else ""), ""]
        for r in reported:
            label = "reproduced on our pin" if r["reproduced"] else "not reproduced on our pin"
            lines.append(f"- {r['symptom']} Fix: {r['fix']} (reported by {r['source']}; {label})")
    return "\n".join(lines) + "\n"


def show_entry(sheet: dict, name: str) -> list[str]:
    """The lines `stage_tools.py cheat NAME` prints for one entry (the same text as in the rendered sheet, with the roots it was verified in)."""
    e = find_entry(sheet, name)
    if e is None:
        return [f"no single entry is called {name!r} in the {sheet['sheet']} sheet."]
    out = [f"`{e['signature']}`" + (f" [{e['status']}]" if e.get("status") else ""), *["    " + ln for ln in e["usage"].split("\n")], f"    {e['notes']}",
           f"    seen: {e['seen_in']}; checked against: {', '.join(e['in'])}"]
    if e.get("evidence"):
        out.append(f"    evidence: {e['evidence']}")
    return out
