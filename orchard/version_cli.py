# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The command behind ``scripts/version.py`` and ``python3 -m orchard.version``.

Commands: ``show``, ``check``, ``new``, ``release [--dry-run]`` and ``migrate-pr --from-ref REF``.
See docs/releases.md for the process. This command never tags, commits or pushes: ``release`` prints the
``git tag`` and ``gh release create`` commands for a person to run. Every file it writes goes through
``atomic_write`` (a temporary file in the same directory, fsync, ``os.replace``).
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from orchard import version as V

KIND_TITLES = {"feature": "Features", "fix": "Fixes", "docs": "Documentation", "chore": "Maintenance", None: "Other changes"}


def repo_root() -> Path:
    return V.CHANGES_DIR.parent


def atomic_write(path: Path, text: str, *, replace: bool = True) -> None:
    """Write `text` to `path` through a temporary file in the same directory, fsync, then os.replace.
    With replace=False an existing file is never touched: the link step fails with FileExistsError."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)                     # atomic, and fails when `path` exists
            os.unlink(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    dfd = os.open(path.parent, os.O_RDONLY)        # make the rename itself durable
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def toml_string(s: str) -> str:
    """A TOML basic string for `s` (quotes, backslashes and control characters escaped)."""
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def fragment_text(target: str, summary: str, pr: int | None = None, kind: str | None = None) -> str:
    lines = [f"target = {toml_string(target)}", f"summary = {toml_string(summary)}"]
    if pr is not None:
        lines.append(f"pr = {pr}")
    if kind is not None:
        lines.append(f"kind = {toml_string(kind)}")
    return "\n".join(lines) + "\n"


def next_patch(version: str) -> str:
    a, b, c = V.parse(version)
    return V.fmt((a, b, c + 1))


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


# ---- commands ------------------------------------------------------------------------------------

def cmd_show(root: Path, args, out, err) -> int:
    ch = root / "changes"
    rel = V.read_release(root / "orchard" / "__init__.py")
    frags = V.load_fragments(ch)
    cur = V.current(ch, release=rel)
    print(f"RELEASE {rel}", file=out)
    if not frags:
        print("no fragments in changes/", file=out)
    for f in frags:
        if f.problem is not None:
            print(f"  SKIPPED  {f.name}: {f.problem}", file=out)
        else:
            extra = "".join([f"  #{f.pr}" if f.pr else "", f"  [{f.kind}]" if f.kind else ""])
            print(f"  {f.slug:<32} {f.target:<10} {f.summary}{extra}", file=out)
    print(f"current {cur}", file=out)
    return 0


def cmd_check(root: Path, args, out, err) -> int:
    rel = V.read_release(root / "orchard" / "__init__.py")
    found = V.problems(root / "changes", release=rel)
    for p in found:
        print(f"problem: {p}", file=err)
    if found:
        print(f"check failed: {len(found)} problem(s)", file=out)
        return 1
    print(f"check ok (RELEASE {rel}, current {V.current(root / 'changes', release=rel)})", file=out)
    return 0


def cmd_new(root: Path, args, out, err) -> int:
    rel = V.read_release(root / "orchard" / "__init__.py")
    ch = root / "changes"
    if not V.slug_ok(args.slug):
        print(f"error: slug {args.slug!r} must be lower-case letters, digits and hyphens, 1 to {V.MAX_SLUG_CHARS} characters", file=err)
        return 2
    target = args.target if args.target is not None else next_patch(V.current(ch, release=rel))
    if V.parse(target) is None:
        print(f"error: target {target!r} is not MAJOR.MINOR.PATCH (whole numbers, no leading zeros)", file=err)
        return 2
    summary = args.summary if args.summary is not None else f"{V.PLACEHOLDER_PREFIX}: describe this change in one sentence."
    if not summary.strip() or len(summary) > V.MAX_SUMMARY_CHARS:
        print(f"error: the summary must be 1 to {V.MAX_SUMMARY_CHARS} characters", file=err)
        return 2
    if args.pr is not None and args.pr < 1:
        print("error: --pr must be 1 or more", file=err)
        return 2
    dest = ch / f"{args.slug}.toml"
    try:
        atomic_write(dest, fragment_text(target, summary, args.pr, args.kind), replace=False)
    except FileExistsError:
        print(f"error: {dest.relative_to(root)} exists; edit it instead", file=err)
        return 1
    print(f"wrote changes/{args.slug}.toml (target {target})", file=out)
    if args.summary is None:
        print(f"edit the {V.PLACEHOLDER_PREFIX} summary before you push; check refuses it", file=out)
    return 0


def release_notes(version: str, date: str, frags) -> str:
    lines = [f"# {version}", "", f"Released {date}.", ""]
    for kind in (*V.KINDS, None):
        group = [f for f in frags if f.kind == kind]
        if not group:
            continue
        lines += [f"## {KIND_TITLES[kind]}", ""]
        for f in sorted(group, key=lambda x: (V.parse(x.target), x.slug)):
            pr = f" (#{f.pr})" if f.pr else ""
            lines.append(f"- {f.summary}{pr} [{f.slug}, target {f.target}]")
        lines.append("")
    return "\n".join(lines)


def cmd_release(root: Path, args, out, err) -> int:
    init = root / "orchard" / "__init__.py"
    ch = root / "changes"
    rel = V.read_release(init)
    found = V.problems(ch, release=rel)
    if found:
        for p in found:
            print(f"problem: {p}", file=err)
        print("error: fix the problems above before a release", file=err)
        return 1
    frags = V.load_fragments(ch)
    if not frags:
        print(f"error: nothing to release: changes/ has no fragments above RELEASE {rel}", file=err)
        return 1
    new = V.current(ch, release=rel)
    notes_path = root / "docs" / "releases" / f"{new}.md"
    if notes_path.exists():
        print(f"error: {notes_path.relative_to(root)} exists; refusing to overwrite it", file=err)
        return 1
    if not args.dry_run:
        st = git(root, "status", "--porcelain")
        if st.returncode != 0:
            print(f"error: {root} is not a usable git work tree ({st.stderr.strip()}); a release needs one", file=err)
            return 1
        if st.stdout.strip():
            print("error: the git tree is dirty; commit or stash first (--dry-run works on a dirty tree)", file=err)
            return 1
    date = datetime.date.today().isoformat()
    notes = release_notes(new, date, frags)
    init_text = init.read_text(encoding="utf-8")
    new_init = V._RELEASE_LINE.sub(f'RELEASE = "{new}"', init_text, count=1)
    verb = "would" if args.dry_run else "will"
    print(f"release {rel} -> {new}", file=out)
    print(f"{verb} set RELEASE in orchard/__init__.py to {new}", file=out)
    print(f"{verb} write docs/releases/{new}.md:", file=out)
    print("".join(f"    | {line}\n" for line in notes.splitlines()), end="", file=out)
    for f in frags:
        print(f"{verb} delete changes/{f.name}", file=out)
    if not args.dry_run:
        atomic_write(notes_path, notes, replace=False)
        atomic_write(init, new_init)
        for f in frags:
            os.unlink(ch / f.name)
        print("done: the files above are changed and nothing is committed", file=out)
    print("", file=out)
    print("Then, from a branch that is merged to main, run these yourself (this command never does):", file=out)
    print(f"  git add -A && git commit -m 'Release {new}'", file=out)
    print(f"  git tag -a v{new} -m 'Release {new}'", file=out)
    print(f"  git push origin main v{new}", file=out)
    print(f"  gh release create v{new} --title 'v{new}' --notes-file docs/releases/{new}.md", file=out)
    return 0


def slug_from_ref(ref: str) -> str:
    name = ref.split("/", 1)[1] if ref.startswith("origin/") else ref
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[: V.MAX_SLUG_CHARS].strip("-")
    return slug or "change"


def cmd_migrate_pr(root: Path, args, out, err) -> int:
    ref = args.from_ref
    shown = git(root, "show", f"{ref}:orchard/__init__.py")
    if shown.returncode != 0:
        print(f"error: cannot read orchard/__init__.py at {ref!r}: {shown.stderr.strip()}", file=err)
        return 1
    if V._RELEASE_LINE.search(shown.stdout):
        print(f"{ref} already uses fragments (its orchard/__init__.py has a RELEASE literal); nothing to migrate", file=out)
        return 0
    m = re.search(r'^__version__ = "([^"\n]*)"', shown.stdout, re.MULTILINE)
    if not m or V.parse(m.group(1)) is None:
        print(f"error: {ref!r} has no `__version__ = \"X.Y.Z\"` line with a valid version", file=err)
        return 1
    subject = git(root, "log", "-1", "--format=%s", ref).stdout.strip()
    subject = re.sub(r"\s*\((?:version|v)\s*[0-9.]+\)\s*$", "", subject)[: V.MAX_SUMMARY_CHARS] or "TODO: describe this change."
    slug = slug_from_ref(ref)
    print(f"# {ref} bumps the old way. Run `python3 scripts/version.py new {slug} --target {m.group(1)}` "
          f"or save this as changes/{slug}.toml.", file=out)
    print("# Take pyproject.toml and orchard/__init__.py from main unchanged; edit the summary (it is the last commit subject).", file=out)
    print(f"changes/{slug}.toml:", file=out)
    print(fragment_text(m.group(1), subject, args.pr, args.kind), end="", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="version.py", description="Computed version, change fragments and releases (docs/releases.md).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="print RELEASE, every fragment and the computed current version")
    sub.add_parser("check", help="exit 1 on any problem in changes/")
    n = sub.add_parser("new", help="write changes/<slug>.toml")
    n.add_argument("slug")
    n.add_argument("--target", help="version this feature claims (default: next patch above the current version)")
    n.add_argument("--summary", help="one plain sentence")
    n.add_argument("--pr", type=int, help="pull request number")
    n.add_argument("--kind", choices=V.KINDS)
    r = sub.add_parser("release", help="cut a release: set RELEASE, write docs/releases/<version>.md, consume the fragments")
    r.add_argument("--dry-run", action="store_true", help="print what would be written and change nothing")
    m = sub.add_parser("migrate-pr", help="print the fragment for a branch that still bumps the version the old way")
    m.add_argument("--from-ref", required=True, help="branch or ref to read orchard/__init__.py from")
    m.add_argument("--pr", type=int, help="pull request number to put in the fragment")
    m.add_argument("--kind", choices=V.KINDS)
    return p


def main(argv=None, root: Path | None = None, out=None, err=None) -> int:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    args = build_parser().parse_args(argv)
    root = repo_root() if root is None else Path(root)
    handler = {"show": cmd_show, "check": cmd_check, "new": cmd_new, "release": cmd_release, "migrate-pr": cmd_migrate_pr}[args.cmd]
    try:
        return handler(root, args, out, err)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=err)
        return 1
