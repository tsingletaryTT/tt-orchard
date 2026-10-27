# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The computed version of tt-orchard.

How the number is made
----------------------
* ``RELEASE`` is the last released version. It is one literal line in ``orchard/__init__.py``
  (``RELEASE = "X.Y.Z"``), so setuptools can read it without importing the package.
* Each feature adds one file, ``changes/<slug>.toml``, holding the version it claims
  (``target = "X.Y.Z"``) and a one-sentence ``summary``. Two features may claim the same number.
* ``current()`` is the highest of ``RELEASE`` and every valid fragment target. ``orchard.__version__``
  is that value. When no ``changes/`` directory sits beside the package (an installed copy),
  ``current()`` is ``RELEASE``.

Version syntax: ``MAJOR.MINOR.PATCH``, three whole numbers without leading zeros (except ``0``), each below
1,000,000. Versions are compared as tuples of numbers, so ``0.9.10`` is higher than ``0.9.9``.

Runtime must never fail on a bad fragment. ``current()`` skips it and records ``(file name, reason)`` in the
module-level list ``SKIPPED`` (cleared at the start of every call). ``problems()`` is the strict view that
``scripts/version.py check`` and the tests use: it lists everything wrong.

Reading is defensive. A fragment must be a regular file (no link, FIFO or directory; the file is opened with
``O_NOFOLLOW | O_NONBLOCK`` and checked with ``fstat`` on the open descriptor), at most 4 KiB, valid UTF-8 and
valid TOML. At most 500 entries are read. This module reads fragments with its own small private reader because
the shared ``orchard/safeio.py`` is being written in a separate pull request and this module must not wait for
it or depend on it. It uses the standard library only.
"""
from __future__ import annotations

import os
import re
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path

MAX_FRAGMENT_BYTES = 4096
MAX_FRAGMENTS = 500
MAX_SUMMARY_CHARS = 300
MAX_SLUG_CHARS = 60
MAX_PART = 1_000_000                       # each of MAJOR, MINOR and PATCH is below this
KINDS = ("feature", "fix", "docs", "chore")
PLACEHOLDER_PREFIX = "TODO"                # `new` writes this summary when none is given; check refuses it
FIELDS = {"target", "summary", "pr", "kind"}

# ASCII digits only ([0-9], never \d), and \Z so a trailing newline does not match.
_VERSION = re.compile(r"\A(0|[1-9][0-9]{0,6})\.(0|[1-9][0-9]{0,6})\.(0|[1-9][0-9]{0,6})\Z")
_SLUG = re.compile(r"\A[a-z0-9][a-z0-9-]{0,%d}\Z" % (MAX_SLUG_CHARS - 1))
_RELEASE_LINE = re.compile(r'^RELEASE = "([^"\n]*)"[ \t]*$', re.MULTILINE)


def parse(text) -> tuple[int, int, int] | None:
    """The version as a tuple of three numbers, or None when `text` is not a valid version string."""
    if type(text) is not str:
        return None
    m = _VERSION.match(text)
    if not m:
        return None
    parts = tuple(int(g) for g in m.groups())
    return parts if all(p < MAX_PART for p in parts) else None


def fmt(parts: tuple[int, int, int]) -> str:
    return ".".join(str(p) for p in parts)


def slug_ok(slug) -> bool:
    return type(slug) is str and _SLUG.match(slug) is not None


def read_release(init_path: Path) -> str:
    """The RELEASE literal of an ``__init__.py``, read as text (the package is not imported)."""
    text = Path(init_path).read_text(encoding="utf-8")
    m = _RELEASE_LINE.search(text)
    if not m or parse(m.group(1)) is None:
        raise ValueError(f'{init_path}: no line `RELEASE = "X.Y.Z"` with a valid version')
    return m.group(1)


_PACKAGE_DIR = Path(__file__).resolve().parent
INIT_PATH = _PACKAGE_DIR / "__init__.py"
CHANGES_DIR = _PACKAGE_DIR.parent / "changes"
RELEASE = read_release(INIT_PATH)

SKIPPED: list[tuple[str, str]] = []        # (file name, reason) for each fragment the last current() call skipped


@dataclass(frozen=True)
class Fragment:
    """One file of ``changes/``. `problem` is None for a valid fragment, otherwise the reason it is skipped."""
    slug: str
    target: str | None = None
    summary: str | None = None
    pr: int | None = None
    kind: str | None = None
    problem: str | None = None
    name: str = ""


def _read_bytes(path: str) -> bytes:
    """Read a regular file of at most MAX_FRAGMENT_BYTES; raise ValueError (with a reason) for anything else."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise ValueError(f"cannot be opened ({e.strerror or e.__class__.__name__}; a link is refused)") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("is not a regular file")
        if st.st_size > MAX_FRAGMENT_BYTES:
            raise ValueError(f"is {st.st_size} bytes; the limit is {MAX_FRAGMENT_BYTES}")
        data = b""
        while len(data) <= MAX_FRAGMENT_BYTES:               # one byte past the limit shows a file that grew
            chunk = os.read(fd, MAX_FRAGMENT_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
        if len(data) > MAX_FRAGMENT_BYTES:
            raise ValueError(f"is more than {MAX_FRAGMENT_BYTES} bytes")
        return data
    finally:
        os.close(fd)


def _parse_fragment(path: Path) -> Fragment:
    """Read one fragment file. Never raises: a bad file comes back with `problem` set."""
    name = path.name
    slug = name[:-5] if name.endswith(".toml") else name

    def bad(why: str) -> Fragment:
        return Fragment(slug=slug, problem=why, name=name)

    if not name.endswith(".toml"):
        return bad("is not a .toml file (only README.md and <slug>.toml belong in changes/)")
    if not slug_ok(slug):
        return bad(f"has a name that is not a slug (lower-case letters, digits and hyphens, at most {MAX_SLUG_CHARS} characters)")
    try:
        data = _read_bytes(str(path))
        doc = tomllib.loads(data.decode("utf-8"))
    except ValueError as e:                                   # includes UnicodeDecodeError and TOMLDecodeError
        why = str(e)
        return bad(why if why.startswith(("is ", "cannot")) else f"is not valid UTF-8 TOML ({why})")
    except (RecursionError, MemoryError):
        return bad("is too deeply nested to read")
    unknown = sorted(set(doc) - FIELDS)
    if unknown:
        return bad(f"has unknown key(s): {', '.join(unknown)}")
    target, summary, pr, kind = doc.get("target"), doc.get("summary"), doc.get("pr"), doc.get("kind")
    if "target" not in doc:
        return bad("has no target")
    if parse(target) is None:
        return bad(f"has a target that is not MAJOR.MINOR.PATCH with whole numbers below {MAX_PART} and no leading zeros: {target!r}")
    if "summary" not in doc:
        return bad("has no summary")
    if type(summary) is not str or not summary.strip():
        return bad("has a summary that is not a non-empty string")
    if len(summary) > MAX_SUMMARY_CHARS:
        return bad(f"has a summary of {len(summary)} characters; the limit is {MAX_SUMMARY_CHARS}")
    if "pr" in doc and (type(pr) is not int or pr < 1):
        return bad(f"has a pr that is not a whole number of 1 or more: {pr!r}")
    if "kind" in doc and kind not in KINDS:
        return bad(f"has a kind that is not one of {', '.join(KINDS)}: {kind!r}")
    return Fragment(slug=slug, target=target, summary=summary, pr=pr, kind=kind, name=name)


def load_fragments(changes_dir: Path | None = None) -> list[Fragment]:
    """Every entry of `changes_dir` except README.md, sorted by name, as Fragments (bad ones carry `problem`).
    A missing directory gives []. Never raises."""
    d = Path(CHANGES_DIR if changes_dir is None else changes_dir)
    try:
        names = sorted(e.name for e in os.scandir(d) if e.name != "README.md")
    except FileNotFoundError:
        return []
    except NotADirectoryError:
        return [Fragment(slug=d.name, problem="is not a directory", name=d.name)]
    except OSError as e:
        return [Fragment(slug=d.name, problem=f"cannot be listed ({e.strerror})", name=d.name)]
    out = [_parse_fragment(d / n) for n in names[:MAX_FRAGMENTS]]
    if len(names) > MAX_FRAGMENTS:
        out.append(Fragment(slug="(directory)", name="(directory)", problem=(
            f"has {len(names)} entries; only the first {MAX_FRAGMENTS} were read")))
    return out


def current(changes_dir: Path | None = None, release: str | None = None) -> str:
    """The highest of `release` (default RELEASE) and every valid fragment target. Never raises on a bad
    fragment: it is skipped and recorded in SKIPPED."""
    rel = RELEASE if release is None else release
    best = parse(rel) or (0, 0, 0)
    SKIPPED.clear()
    for f in load_fragments(changes_dir):
        if f.problem is not None:
            SKIPPED.append((f.name, f.problem))
            continue
        best = max(best, parse(f.target))
    return fmt(best)


def problems(changes_dir: Path | None = None, release: str | None = None) -> list[str]:
    """Everything `check` refuses, as sentences. An empty list means the fragments are in order."""
    rel = RELEASE if release is None else release
    rel_t = parse(rel) or (0, 0, 0)
    out: list[str] = []
    by_pr: dict[int, list[str]] = {}
    for f in load_fragments(changes_dir):
        if f.problem is not None:
            out.append(f"changes/{f.name} {f.problem}")
            continue
        if parse(f.target) <= rel_t:
            out.append(f"changes/{f.name} targets {f.target}, which is not above RELEASE {rel}: the feature was merged after "
                       f"a release was cut, so retarget it (python3 scripts/version.py new, or edit target)")
        if f.summary.startswith(PLACEHOLDER_PREFIX):
            out.append(f"changes/{f.name} still has the {PLACEHOLDER_PREFIX} placeholder summary; write one plain sentence")
        if f.pr is not None:
            by_pr.setdefault(f.pr, []).append(f.slug)
    for pr, slugs in sorted(by_pr.items()):
        if len(slugs) > 1:
            out.append(f"pr {pr} has more than one fragment: {', '.join(sorted(slugs))}")
    return out


if __name__ == "__main__":
    import sys
    from orchard import version_cli
    sys.exit(version_cli.main(sys.argv[1:]))
