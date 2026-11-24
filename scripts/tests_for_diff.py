#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Print the pytest arguments for the tests a diff needs (docs/test-tiers.md).

    python3 -m pytest $(python3 scripts/tests_for_diff.py)             # the tests a change to this branch can affect
    python3 -m pytest $(python3 scripts/tests_for_diff.py --fast)      # the same, without the tests marked slow
    python3 -m pytest $(python3 scripts/tests_for_diff.py --eager)     # follow only the imports that run when a module loads
    python3 -m pytest $(python3 scripts/tests_for_diff.py --direct)    # only the tests that import a changed module themselves

Standard output is one line of arguments, ready for the shell. Standard error says what changed, how each test was chosen and why a
change selected the full suite. The selection is a fast loop while developing. The full suite on the integration state is still the
gate before a merge (docs/test-tiers.md).

How a test is chosen. Python files under `orchard/`, `scripts/`, `bin/` and `tests/` are read as syntax trees (never imported). A file
depends on another when it imports it (in any form, at any depth of a function or `try`), when a string in it names the other file
(`"scripts/tool.py"`, `Path(...) / "tool.py"`, `python3 scripts/tool.py`), or when a string holds code that imports a module
(`-m orchard.a`, `"from orchard import a"`). The tests that depend on a changed file, directly or through any chain of files, are
selected. A test file that computes an import (`importlib.import_module(name)`) or does not parse is selected for every change to Python
source, because its dependencies cannot be read.

The tests named in `WHOLE_REPO` in `tests/tier_lists.py` are always selected: they scan the whole repository, so any new module, flag or file
can fail them.

What it will not guess. A change it cannot map selects the full suite and names the file: `tests/conftest.py` and any other test
helper, `tests/tier_lists.py`, `pyproject.toml`, requirements files, a deleted or renamed module, anything under `orchard/skills/` or
`orchard/package_templates/` (templates loaded by path), a data file that no test names, a base ref that does not exist and a directory
that is not a git repository. `changes/` fragments are ignored. Documents (`*.md`, `docs/`) select the tests that name them and the guards.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import shlex
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = "tests"
REGISTRY = "tests/tier_lists.py"
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
# Paths whose change needs the full suite: shared test machinery, build and CI configuration, and templates loaded by path.
FULL_NAMES = {"pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py", "poetry.lock", "uv.lock", "Pipfile"}
FULL_PREFIXES = (".github/", "orchard/skills/", "orchard/package_templates/")
INERT_NAMES = {"LICENSE", "NOTICE", ".gitignore", ".gitattributes"}
GENERIC_DIRS = {"tests", "fixtures", "docs", "config", "orchard", "scripts", "bin", "changes", "site"}

_TOKEN = re.compile(r"[\w][\w.-]*\.(?:py|md|toml|json|jsonl|txt|sh|yaml|yml|cfg|ini|csv)\b")
_WORD_SPLIT = re.compile(r"""[/\s"'=:,;()\[\]{}]+""")
_DOTTED = re.compile(r"\b(?:orchard|scripts)(?:\.\w+)+")
_FROM_IMPORT = re.compile(r"\bfrom\s+((?:orchard|scripts)(?:\.\w+)*)\s+import\s+([\w, ]+)")
_SRC_IMPORT = re.compile(r"^\s*import\s+([\w., ]+)", re.M)
_SRC_FROM = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+([\w, *]+)", re.M)
_NODE_GLOBS = (("orchard", True), ("scripts", True), ("bin", False), ("tests", False))


@dataclass
class Selection:
    full: bool = False
    reasons: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    why: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    total_tests: int = 0

    def args(self, fast: bool = False) -> list[str]:
        out = ["tests"] if self.full else sorted(self.files) + sorted(self.nodes)
        return out + (["-m", "not slow"] if fast else [])


@dataclass
class Node:
    rel: str
    refs: set = field(default_factory=set)        # dotted names this file imports when it loads, or names in a string of code
    lazy_refs: set = field(default_factory=set)   # dotted names imported inside a function, which run only when it is called
    tokens: set = field(default_factory=set)      # file names found in its strings
    words: set = field(default_factory=set)       # path segments and words found in its strings
    dynamic: bool = False                         # imports a module chosen at run time
    parsed: bool = True


# ---- reading the files ------------------------------------------------------------------------------

def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield Path(dirpath, name)


def _module_names(rel: str) -> set[str]:
    """The dotted names a file can be imported under: its path, and the bare name when its folder is on `sys.path`."""
    parts = rel.split("/")
    names = set()
    if parts[-1].endswith(".py"):
        stem = parts[-1][:-3]
        base = parts[:-1] if stem == "__init__" else parts[:-1] + [stem]
        if base:
            names.add(".".join(base))
        if parts[0] in ("scripts", TESTS_DIR) and stem != "__init__":
            names.add(stem)                                         # `import tool` after sys.path gains scripts/ or tests/ is on it
    return names


def _constants(tree) -> list[str]:
    """The string constants of a tree, leaving out docstrings: a docstring describes a file, and a string that is used names one."""
    docstrings = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def _scan_strings(node: Node, strings) -> None:
    for s in strings:
        if len(s) > 400_000:
            continue
        node.tokens.update(_TOKEN.findall(s))
        node.words.update(w for w in _WORD_SPLIT.split(s) if 0 < len(w) <= 80)
        node.refs.update(_DOTTED.findall(s))
        for base, names in _FROM_IMPORT.findall(s):
            node.refs.add(base)
            node.refs.update(f"{base}.{n.strip()}" for n in names.split(",") if n.strip())


def _package_of(rel: str) -> list[str]:
    parts = rel.split("/")
    return parts[:-1]


def _from_base(rel: str, module: str | None, level: int) -> str:
    pkg = _package_of(rel)
    if level:
        pkg = pkg[: len(pkg) - (level - 1)] if level - 1 <= len(pkg) else []
        return ".".join(pkg + ([module] if module else []))
    return module or ""


def read_node(root: Path, rel: str) -> Node:
    node = Node(rel)
    try:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        node.parsed = False
        return node
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        node.parsed = False
        _scan_strings(node, [text])
        for m in _SRC_IMPORT.findall(text):
            node.refs.update(n.strip().split(" as ")[0] for n in m.split(",") if n.strip())
        for base, names in _SRC_FROM.findall(text):
            node.refs.add(base)
            node.refs.update(f"{base}.{n.strip().split(' as ')[0]}" for n in names.split(",") if n.strip() and n.strip() != "*")
        return node
    def visit(n, lazy: bool) -> None:
        target = node.lazy_refs if lazy else node.refs
        if isinstance(n, ast.Import):
            target.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            base = _from_base(rel, n.module, n.level)
            target.add(base)
            target.update(f"{base}.{a.name}" if base else a.name for a in n.names if a.name != "*")
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in ("import_module", "__import__"):
                first = n.args[0] if n.args else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    target.add(first.value)
                else:
                    node.dynamic = True
        inner = lazy or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for child in ast.iter_child_nodes(n):
            visit(child, inner)

    visit(tree, False)
    _scan_strings(node, _constants(tree))
    return node


class Graph:
    """Every Python file the tests can reach, with the files each one depends on."""

    def __init__(self, root: Path):
        self.root = root
        self.nodes: dict[str, Node] = {}
        for top, deep in _NODE_GLOBS:
            base = root / top
            if not base.is_dir():
                continue
            for p in (_walk(base) if deep else base.iterdir()):
                rel = p.relative_to(root).as_posix()
                if not p.is_file() or "/fixtures/" in f"/{rel}" or "__pycache__" in rel:
                    continue
                if top == "bin" or rel.endswith(".py"):
                    self.nodes[rel] = read_node(root, rel)
        self.names: dict[str, set[str]] = {}
        for rel in self.nodes:
            for name in _module_names(rel):
                self.names.setdefault(name, set()).add(rel)
        self.by_basename: dict[str, set[str]] = {}
        for rel in self.nodes:
            base = rel.rsplit("/", 1)[-1]
            if base != "__init__.py":
                self.by_basename.setdefault(base, set()).add(rel)
        self.deps: dict[str, set[str]] = {rel: self._deps_of(n, lazy=True) - {rel} for rel, n in self.nodes.items()}
        self.eager_deps: dict[str, set[str]] = {rel: self._deps_of(n, lazy=False) - {rel} for rel, n in self.nodes.items()}
        self.rdeps = self._reverse(self.deps)
        self.eager_rdeps = self._reverse(self.eager_deps)

    @staticmethod
    def _reverse(deps):
        out: dict[str, set[str]] = {}
        for rel, ds in deps.items():
            for d in ds:
                out.setdefault(d, set()).add(rel)
        return out

    def _resolve(self, dotted: str) -> set[str]:
        out: set[str] = set()
        parts = [p for p in dotted.split(".") if p]
        for i in range(1, len(parts) + 1):
            out |= self.names.get(".".join(parts[:i]), set())
        return out

    def _deps_of(self, node: Node, lazy: bool = True) -> set[str]:
        out: set[str] = set()
        for ref in (node.refs | node.lazy_refs) if lazy else node.refs:
            out |= self._resolve(ref)
        for tok in node.tokens:
            out |= self.by_basename.get(tok, set())
        if node.rel.startswith(f"{TESTS_DIR}/"):                    # a test that names a program with no extension (bin/tt-orchard) runs it
            for word in node.words:
                if word in self.by_basename and "." not in word:
                    out |= self.by_basename[word]
        return out

    def referrers(self, rel: str) -> set[str]:
        """Nodes whose strings name a file that is not a node itself: a document or a data file."""
        base = rel.rsplit("/", 1)[-1]
        comps = [c for c in rel.split("/")[:-1] if c not in GENERIC_DIRS]
        return {r for r, n in self.nodes.items() if (base in n.tokens and base != "__init__.py") or any(c in n.words for c in comps)}

    def closure(self, seeds, direct: bool = False, eager: bool = False):
        """The files that depend on any seed, with the chain that led to each. `direct` stops at the first test file and does not pass
        through orchard, scripts or bin; test helpers are passed through because a test imports them as part of itself. `eager` follows
        only the imports that run when a module loads and leaves out the ones inside functions."""
        rdeps = self.eager_rdeps if eager else self.rdeps
        parent: dict[str, str | None] = {s: None for s in seeds}
        queue = deque(seeds)
        while queue:
            cur = queue.popleft()
            for nxt in sorted(rdeps.get(cur, ())):
                if nxt in parent:
                    continue
                parent[nxt] = cur
                is_test = nxt.startswith("tests/test_")
                in_tests = nxt.startswith(f"{TESTS_DIR}/")
                if direct and not in_tests:
                    del parent[nxt]
                    continue
                if not (direct and is_test):
                    queue.append(nxt)
        return parent


# ---- the registry -----------------------------------------------------------------------------------

def load_registry(root: Path):
    """(WHOLE_REPO as {path: tuple of test names}, problems). Executes tests/tier_lists.py, which holds data only."""
    path = root / REGISTRY
    if not path.is_file():
        return {}, [f"{REGISTRY} is missing, so the tests that must always run are not known"]
    try:
        spec = importlib.util.spec_from_file_location("tier_lists_for_diff", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        whole = dict(mod.WHOLE_REPO)
    except Exception as exc:                                       # a registry that does not load is a registry problem, whatever it raised
        return {}, [f"{REGISTRY} could not be read: {type(exc).__name__}: {exc}"]
    problems = []
    for rel, names in whole.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"{REGISTRY} names {rel}, which does not exist")
            continue
        if names:
            try:
                defined = {n.name for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            except (SyntaxError, OSError, ValueError):
                defined = set()
            problems += [f"{REGISTRY} names {rel}::{n}, which does not exist" for n in names if n not in defined]
    return whole, problems


# ---- classification ---------------------------------------------------------------------------------

def _is_python_source(path: str) -> bool:
    return path.startswith(("orchard/", "scripts/", "bin/")) and (path.endswith(".py") or path.startswith("bin/"))


def _full_reason(path: str) -> str | None:
    name = path.rsplit("/", 1)[-1]
    if path.startswith(f"{TESTS_DIR}/") and "/" not in path[len(TESTS_DIR) + 1:] and path.endswith(".py") and not name.startswith("test_"):
        return f"{path} is shared by many tests (a test helper or conftest.py)"
    if name in FULL_NAMES or re.fullmatch(r"requirements.*\.txt", name):
        return f"{path} configures how every test runs"
    if path.startswith(FULL_PREFIXES):
        return f"{path} is a template or skill file loaded by path, which this script does not map to tests"
    return None


def select(root, changes, direct: bool = False, eager: bool = False) -> Selection:
    """The tests that `changes` need. `changes` is a list of (status, path) or (status, old_path, new_path) with status A, M, D or R."""
    root = Path(root)
    sel = Selection()
    graph = Graph(root)
    test_files = sorted(r for r in graph.nodes if r.startswith(f"{TESTS_DIR}/test_") and r.count("/") == 1)
    sel.total_tests = len(test_files)
    whole, problems = load_registry(root)
    sel.reasons += problems
    seeds: set[str] = set()
    own: set[str] = set()                                          # changed test files
    python_changed = False
    for change in changes:
        status, paths = change[0], change[1:]
        old, new = paths[0], paths[-1]
        sel.changed.append(new if status != "D" else old)
        if status in ("D", "R"):
            if _is_python_source(old) and not old.startswith("orchard/skills/"):
                sel.reasons.append(f"{old} was deleted or renamed, so the files that used it cannot be listed")
                continue
            if status == "D":
                if old.startswith(f"{TESTS_DIR}/test_") or old.startswith("changes/"):
                    continue
                path = old
            else:
                path = new
                if not (old.startswith(f"{TESTS_DIR}/test_") or old.startswith("changes/")):
                    seeds |= _map_other(graph, old, sel)
        else:
            path = new
        reason = _full_reason(path)
        if reason:
            sel.reasons.append(reason)
            continue
        if path.startswith("changes/"):
            continue
        if path.startswith(f"{TESTS_DIR}/test_") and path.endswith(".py") and path.count("/") == 1:
            if status != "D":
                own.add(path)
            continue
        if _is_python_source(path) and path in graph.nodes:
            python_changed = True
            seeds.add(path)
            continue
        if _is_python_source(path):
            python_changed = True                                   # a new file that was not scanned: nothing depends on it yet
            continue
        seeds |= _map_other(graph, path, sel)
    if sel.reasons:
        sel.full = True
        return sel
    reached = graph.closure(seeds, direct=direct, eager=eager) if seeds else {}
    chosen = {r for r in reached if r in test_files}
    chain_of = {}
    for r in chosen:
        chain, cur = [], r
        while cur is not None:
            chain.append(cur)
            cur = reached[cur]
        chain_of[r] = [" -> ".join(chain)]
    if python_changed:
        for rel in test_files:
            node = graph.nodes[rel]
            if node.dynamic:
                chosen.add(rel)
                chain_of.setdefault(rel, []).append(f"{rel} computes an import at run time")
            if not node.parsed:
                chosen.add(rel)
                sel.notes.append(f"{rel} does not parse, so it is selected for every change to Python source")
                chain_of.setdefault(rel, []).append(f"{rel} does not parse")
    for rel in own:
        chosen.add(rel)
        chain_of.setdefault(rel, []).append(f"{rel} changed")
    files = set(chosen)
    nodes = set()
    for rel, names in whole.items():
        if not names:
            files.add(rel)
            chain_of.setdefault(rel, []).append("guard: always selected")
        elif rel not in files:
            nodes |= {f"{rel}::{n}" for n in names}
    sel.files = sorted(files)
    sel.nodes = sorted(nodes)
    sel.why = chain_of
    return sel


def _map_other(graph: Graph, path: str, sel: Selection) -> set[str]:
    """Seeds for a changed file that is not Python source: the files whose strings name it. A document needs no such file. Any other file
    that no string names selects the full suite."""
    name = path.rsplit("/", 1)[-1]
    refs = graph.referrers(path)
    inert = path.endswith(".md") or path.startswith("docs/") or name in INERT_NAMES
    if not refs and not inert:
        sel.reasons.append(f"no test names {path}, so the tests it affects are not known")
    return refs


# ---- git --------------------------------------------------------------------------------------------

def _git(root: Path, *args: str):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)


def git_changes(root: Path, base: str):
    """([(status, path...)], problem). `problem` is set when the changes cannot be listed (not a repository, an unknown base ref)."""
    if _git(root, "rev-parse", "--is-inside-work-tree").returncode != 0:
        return [], f"{root} is not a git repository, so the changes cannot be listed"
    if _git(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").returncode != 0:
        return [], f"the base ref {base} does not exist, so the changes cannot be listed"
    mb = _git(root, "merge-base", base, "HEAD")
    start = mb.stdout.strip() if mb.returncode == 0 and mb.stdout.strip() else base
    diff = subprocess.run(["git", "-C", str(root), "diff", "--name-status", "-z", "-M", start], capture_output=True, check=False)
    if diff.returncode != 0:
        return [], f"git diff against {start} failed: {diff.stderr.decode(errors='replace').strip()}"
    parts = diff.stdout.decode("utf-8", "replace").split("\0")
    changes, i = [], 0
    while i < len(parts) and parts[i]:
        status = parts[i][0]
        if status in "RC":
            old, new = parts[i + 1], parts[i + 2]
            changes.append(("R", old, new) if status == "R" else ("A", new))
            i += 3
        else:
            changes.append((status if status in "ADM" else "M", parts[i + 1]))
            i += 2
    other = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    changes += [("A", p) for p in other.stdout.split("\0") if p]
    return changes, None


# ---- command line -----------------------------------------------------------------------------------

def explain(sel: Selection, fast: bool, direct: bool, eager: bool = False) -> str:
    lines = [f"changed: {', '.join(sel.changed) if sel.changed else '(nothing)'}"]
    if not sel.changed:
        lines.append("nothing changed: only the guards are selected")
    if sel.full:
        lines += [f"full suite: {r}" for r in sel.reasons]
        return "\n".join(lines)
    lines.append(f"selected {len(sel.files)} of {sel.total_tests} test files and {len(sel.nodes)} guard tests"
                 + (" (direct importers only)" if direct else "") + (" (imports that run at load only)" if eager else "")
                 + (" without the slow tests" if fast else ""))
    lines += [f"  {rel}  ({'; '.join(sel.why.get(rel, ['selected']))})" for rel in sel.files]
    lines += [f"  {n}  (guard: always selected)" for n in sel.nodes]
    lines += [f"note: {n}" for n in sel.notes]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print the pytest arguments for the tests a diff needs.")
    ap.add_argument("--base", default="origin/main", help="the ref the diff starts from (default origin/main)")
    ap.add_argument("--root", default=str(ROOT), help="the repository (default: this one)")
    ap.add_argument("--direct", action="store_true", help="only the tests that import a changed module themselves (fastest, least safe)")
    ap.add_argument("--eager", action="store_true", help="follow only the imports that run when a module loads, not the ones inside functions (faster, less safe)")
    ap.add_argument("--fast", action="store_true", help="add -m 'not slow'")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    changes, problem = git_changes(root, ns.base)
    if problem:
        sel = Selection(full=True, reasons=[problem])
    else:
        sel = select(root, changes, direct=ns.direct, eager=ns.eager)
    print(explain(sel, ns.fast, ns.direct, ns.eager), file=sys.stderr)
    print(shlex.join(sel.args(fast=ns.fast)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
