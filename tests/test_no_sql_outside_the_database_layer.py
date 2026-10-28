"""No SQL outside orchard/db.py and orchard/telegraph_store.py (design note, section 4: one access layer).

Every other module reaches the database through those two. The scan reads each module's string constants (including the literal parts
of f-strings) and reports a SQL statement or an import of `sqlite3`. `orchard/_proofs/` is the earlier proof module and is not part of
the product. The checker is tested on crafted sources, and it was seen to fail by adding a statement to a scratch module (the
mutation log is in docs/project-log.md).
"""
import ast
import re
from pathlib import Path

import pytest

ORCHARD = Path(__file__).resolve().parents[1] / "orchard"
ALLOWED = {"db.py", "telegraph_store.py"}
SKIP_DIRS = {"_proofs", "skills", "package_templates", "lm_eval_tasks", "web", "__pycache__"}
STATEMENT = re.compile(
    r"\b(SELECT\b[^;]*\bFROM\b|INSERT\s+(?:OR\s+\w+\s+)?INTO\b|UPDATE\s+\w+\s+SET\b|DELETE\s+FROM\b|CREATE\s+(?:UNIQUE\s+)?"
    r"(?:TABLE|INDEX|VIEW|TRIGGER)\b|DROP\s+(?:TABLE|INDEX|VIEW|TRIGGER)\b|ALTER\s+TABLE\b|PRAGMA\s+\w+|BEGIN\s+(?:IMMEDIATE|EXCLUSIVE)\b|"
    r"ATTACH\s+DATABASE\b|VACUUM\b\s+INTO\b)")


def sql_in(source: str) -> list[tuple[int, str]]:
    """(line, text) for each SQL statement in a string constant of `source`, and for each import of sqlite3."""
    hits = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            match = STATEMENT.search(node.value)
            if match:
                hits.append((node.lineno, match.group(0)))
        elif isinstance(node, ast.Import):
            hits += [(node.lineno, "import sqlite3") for a in node.names if a.name.split(".")[0] == "sqlite3"]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "sqlite3":
            hits.append((node.lineno, "from sqlite3 import"))
    return hits


def product_modules():
    for path in sorted(ORCHARD.rglob("*.py")):
        rel = path.relative_to(ORCHARD)
        if rel.parts[0] in SKIP_DIRS or (len(rel.parts) == 1 and rel.name in ALLOWED):
            continue
        yield rel, path


def test_no_module_but_the_database_layer_holds_sql_or_imports_sqlite3():
    found = {}
    for rel, path in product_modules():
        hits = sql_in(path.read_text(encoding="utf-8"))
        if hits:
            found[str(rel)] = hits
    assert found == {}


@pytest.mark.parametrize("source", [
    'conn.execute("SELECT raw_json FROM telegraph_events")',
    'q = "INSERT INTO t (a) VALUES (?)"',
    'q = "INSERT OR REPLACE INTO t (a) VALUES (?)"',
    'q = "UPDATE t SET a = 1"',
    'q = "DELETE FROM t"',
    'q = "CREATE TABLE t (a INTEGER)"',
    'q = "PRAGMA journal_mode = WAL"',
    'q = f"SELECT a FROM {table} WHERE x = 1"',
    'c.execute("BEGIN IMMEDIATE")',
    'import sqlite3',
    'from sqlite3 import connect',
])
def test_the_checker_finds_a_statement_in_a_crafted_module(source):
    assert sql_in(source)


@pytest.mark.parametrize("source", [
    'print("Select the files from the list")',
    'text = "update the table of results"',
    'import json',
    'name = "telegraph_events"',
    '"""This module reads rows from the database through orchard.telegraph_store."""',
])
def test_the_checker_leaves_ordinary_text_alone(source):
    assert sql_in(source) == []


def test_the_two_allowed_modules_are_the_ones_that_hold_sql():
    for name in ALLOWED:
        assert sql_in((ORCHARD / name).read_text(encoding="utf-8")) or name == "telegraph_store.py"
    assert sql_in((ORCHARD / "db.py").read_text(encoding="utf-8"))
