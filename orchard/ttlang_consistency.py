# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard ttlang-consistency`: does the machine match the tt-lang pin (orchard/ttlang_pin.py)?

One row per place that must agree with the pin, each ok, mismatch, todo or unknown, with a one-line fix on a mismatch:

    checkout        the tt-lang checkout's HEAD against the pinned commit (and the specification it documents)
    checkout files  every file under examples/ and docs/ is as the pinned commit has it (none modified or untracked)
    simulator       the tt-lang-sim wheel's version, which carries the commit it was built from
    skill           the specification version the installed tt-lang skill documents against the checkout's
    references      the specification version of every file the coder is given against the checkout's
    hardware tree   the tt-metal tree's version against the pinned tt-metal ref (todo until issue 95 pins one)

A row `blocks` when the kernel probe must refuse to run on it (the probe does, unless --allow-stale-skills). An old
skill does not block: the probe then builds the guide from the checkout (orchard/ttlang_reference.py), so the coder
still reads the version the simulator runs.

Everything is read only. The tt-metal tree is read with `git describe` and `git rev-parse` and nothing else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from orchard import ttlang_env, ttlang_pin, ttlang_reference

OK, MISMATCH, TODO, UNKNOWN = "ok", "mismatch", "todo", "unknown"
checkout_spec_version = ttlang_reference.checkout_spec_version


@dataclass
class Row:
    name: str
    status: str
    detail: str
    fix: str = ""
    blocks: bool = False


def _checkout_row(env) -> Row:
    pin = ttlang_pin.SHA
    have = ttlang_env.head_sha(env)
    ver = checkout_spec_version(env.src_dir)
    doc = f"documents specification {ver}" if ver else "documents no readable specification version"
    if have == pin:
        return Row("checkout", OK, f"{env.src_dir} is at the pinned commit {pin[:9]} and {doc}")
    return Row("checkout", MISMATCH,
               f"{env.src_dir} is at {have[:9] if have else 'no readable commit'} and the pin is {pin[:9]}",
               "run `tt-orchard kernel-env` (it never resets a checkout at another commit: move it aside or pass "
               "--src-dir), or move the pin in orchard/ttlang_pin.py", blocks=True)


def _simulator_row(env) -> Row:
    want = ttlang_pin.SIM_VERSION
    have = ttlang_env.sim_version(env)
    fix = "run `tt-orchard kernel-env` to build the wheel from the pinned checkout"
    if have is None:
        return Row("simulator", MISMATCH, f"tt-lang-sim is not installed in {env.venv_dir}; the pin wants {want}", fix,
                   blocks=True)
    built = have.partition("+g")[2] or "an unknown commit"
    if have == want:
        return Row("simulator", OK, f"tt-lang-sim {have}, built from {built}, matches the pin")
    return Row("simulator", MISMATCH, f"tt-lang-sim {have} was built from {built}; the pin wants {want} (built from "
               f"{ttlang_pin.SIM_SOURCE_SHA[:7]})", fix, blocks=True)


def _files_row(env) -> Row:
    tracked, dirty = ttlang_reference.git_state(env)
    if dirty is None or tracked is None:
        return Row("checkout files", UNKNOWN, f"git could not say which files under examples/ and docs/ of {env.src_dir} "
                   "the pinned commit has, so the extra examples are not given to the coder")
    if dirty:
        shown = ", ".join(f"{p} ({w})" for p, w in sorted(dirty.items())[:5])
        return Row("checkout files", MISMATCH, f"{len(dirty)} file(s) under examples/ or docs/ of {env.src_dir} are "
                   f"not as the pinned commit has them: {shown}",
                   f"run `git -C {env.src_dir} status` and restore or move the files aside", blocks=True)
    return Row("checkout files", OK, "every file under examples/ and docs/ is as the pinned commit has it")


def _skill_row(env, skill_dir) -> Row:
    mine, theirs = ttlang_reference.skill_spec_version(skill_dir), checkout_spec_version(env.src_dir)
    if not mine or not theirs:
        return Row("skill", UNKNOWN, f"the skill documents specification {mine or 'an unreadable version'} and the "
                   f"checkout documents {theirs or 'an unreadable version'}; they cannot be compared")
    if mine == theirs:
        return Row("skill", OK, f"the skill and the checkout both document specification {mine}")
    return Row("skill", MISMATCH, f"the skill documents specification {mine} and the checkout documents {theirs}",
               "nothing to run: the probe builds guide.md and examples.md from the checkout; update the skill with "
               "`tt-warp fastmodels sync` when upstream catches up")


def _references_row(env, skill_dir, references) -> Row:
    want = checkout_spec_version(env.src_dir)
    if references is None:
        references, problems = ttlang_reference.assemble(env, skill_dir)
        if problems:
            return Row("references", MISMATCH, "the reference files cannot be assembled: " + "; ".join(problems),
                       "restore the missing files in the checkout or the skill", blocks=True)
    if not want:
        return Row("references", UNKNOWN, "the checkout's specification version cannot be read, so the reference files "
                   "cannot be compared with it")
    # The worked-kernel library (library_*.py) is orchard's own and has no specification version.
    own = [r for r in references if not str(r.get("name", "")).startswith("library_")]
    unread = [r["name"] for r in own if not r.get("spec_version")]
    off = [f"{r['name']} documents {r['spec_version']}" for r in own
           if r.get("spec_version") and r["spec_version"] != want]
    if off:
        return Row("references", MISMATCH, f"the checkout documents specification {want} but " + ", ".join(off),
                   "build the reference files from the checkout (the probe does this when the skill is older)",
                   blocks=True)
    if unread:
        return Row("references", UNKNOWN, f"the checkout documents specification {want}, but the version of "
                   f"{', '.join(unread)} cannot be read, so they cannot be compared with it")
    return Row("references", OK, f"every reference file documents specification {want}")


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:                          # a directory that cannot be searched has no readable tree
        return False


def _tree_row(env, tree: Path | None) -> Row:
    ref = ttlang_pin.TT_METAL_REF
    have = None
    if tree is not None and _exists(Path(tree) / ".git"):
        rc, out, _ = env.run(["git", "-C", str(tree), "describe", "--tags", "--always"], 30)
        have = out.strip() if rc == 0 and out.strip() else None
        rc, out, _ = env.run(["git", "-C", str(tree), "rev-parse", "HEAD"], 30)
        sha = out.strip() if rc == 0 and out.strip() else None
    else:
        sha = None
    seen = have or "no readable tt-metal tree"
    if ref is None:
        return Row("hardware tree", TODO, f"the tt-metal ref is {ttlang_pin.TT_METAL_NOTE}; the host tree is {seen}",
                   "when issue 95 is settled, set TT_METAL_REF in orchard/ttlang_pin.py")
    if have is None:
        return Row("hardware tree", MISMATCH, f"no readable tt-metal tree; the pin wants {ref}",
                   "name the tree with --tt-metal-dir or ORCHARD_TT_METAL_DIR", blocks=True)
    if have == ref or sha == ref or (sha and len(ref) >= 7 and sha.startswith(ref)):
        return Row("hardware tree", OK, f"the tt-metal tree is at {have}, the pinned ref {ref}")
    return Row("hardware tree", MISMATCH, f"the tt-metal tree is at {have} and the pin wants {ref}",
               "check out the pinned ref in the host tree, or move TT_METAL_REF with the tt-lang pin (issue 95)",
               blocks=True)


def check(env, skill_dir, *, tt_metal_dir=None, references: list[dict] | None = None) -> list[Row]:
    """Every row. `references` are the files the probe already assembled (so the check judges what the coder gets);
    without them the check assembles them itself, in memory."""
    return [_checkout_row(env), _simulator_row(env), _files_row(env), _skill_row(env, skill_dir),
            _references_row(env, skill_dir, references), _tree_row(env, tt_metal_dir)]


def blocking(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.blocks]


def render(rows: list[Row]) -> list[str]:
    """Terminal lines: one per row, and the fix on its own indented line under a mismatch. No right-hand border."""
    lines = []
    for r in rows:
        lines.append(f"{r.status:9s}{r.name}: {r.detail}")
        if r.status == MISMATCH and r.fix:
            lines.append(f"{'':9s}fix: {r.fix}")
    return lines


def pin_record(env) -> dict:
    """The pin as the probe's report states it, with the specification version read from the checkout."""
    return {"sha": ttlang_pin.SHA, "sim_version": ttlang_pin.SIM_VERSION, "sim_source_sha": ttlang_pin.SIM_SOURCE_SHA,
            "spec_version": checkout_spec_version(env.src_dir),
            "tt_metal_ref": ttlang_pin.TT_METAL_REF or ttlang_pin.TT_METAL_NOTE}


def tt_metal_dir_from(args, environ=None, home: Path | None = None) -> Path:
    environ = os.environ if environ is None else environ
    flag = getattr(args, "tt_metal_dir", None)
    return Path(flag or environ.get("ORCHARD_TT_METAL_DIR") or Path(home or Path.home()) / "tt-metal")


def main(args, *, env=None, out=None) -> int:
    """`tt-orchard ttlang-consistency [--src-dir D] [--venv-dir D] [--tt-lang-skill-dir D] [--tt-metal-dir D]`.
    Exit 0 when no row blocks, 2 otherwise. Changes nothing."""
    say = (lambda line: print(line, file=out)) if out is not None else print
    home = Path.home()
    env = env or ttlang_env.KEnv(
        src_dir=Path(args.src_dir or os.environ.get("ORCHARD_TTLANG_SRC") or home / "orchard-src" / "tt-lang"),
        venv_dir=Path(args.venv_dir or os.environ.get("ORCHARD_TTLANG_VENV") or home / "orchard-venvs" / "ttlang"),
        scratch_dir=Path(os.environ.get("ORCHARD_SCRATCH") or home / "orchard-src" / "scratch"),
        run=ttlang_env.real_run, environ=dict(os.environ))
    rows = check(env, ttlang_env.skill_dir_from(args), tt_metal_dir=tt_metal_dir_from(args))
    say(f"tt-lang pin {ttlang_pin.SHA[:9]} (simulator {ttlang_pin.SIM_VERSION})")
    for line in render(rows):
        say(line)
    return 2 if blocking(rows) else 0
