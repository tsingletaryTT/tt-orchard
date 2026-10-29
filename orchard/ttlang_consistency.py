# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard ttlang-consistency`: does the machine match the tt-lang pin (orchard/ttlang_pin.py)?

One row per place that must agree with the pin, each ok, mismatch, todo or unknown, with a one-line fix on a mismatch:

    checkout        the tt-lang checkout's HEAD against the pinned commit (and the specification it documents)
    checkout files  every file under examples/ and docs/ is as the pinned commit has it (none modified or untracked)
    simulator       the tt-lang-sim wheel's version, which carries the commit it was built from
    skill           the specification version the installed tt-lang skill documents against the checkout's
    references      the specification version of every file the coder is given against the checkout's
    hardware tree   the hardware venv's `ttl.build_info()` (tt-lang and tt-metal commits) against the pin and the pinned
                    tt-metal ref. todo until a caller that holds a lease hands the build info in

A row `blocks` when the kernel probe must refuse to run on it (the probe does, unless --allow-stale-skills). An old
skill does not block: the probe then builds the guide from the checkout (orchard/ttlang_reference.py), so the coder
still reads the version the simulator runs.

Everything is read only. The tt-metal tree is read with `git describe` and `git rev-parse` and nothing else. This module
never imports the device library: `ttl.build_info()` opens the runtime, so a caller that already holds a gozer lease
runs it and passes the result in as `build_info` (a dict with `tt_lang_commit`, `tt_metal_commit` and optionally `version`
and `tt_metal_tag`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from orchard import safeio, ttlang_env, ttlang_pin, ttlang_reference

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
    differs = {p: w for p, w in dirty.items() if w != "unchecked"}
    if differs:
        shown = ", ".join(f"{p} ({w})" for p, w in sorted(differs.items())[:5])
        return Row("checkout files", MISMATCH, f"{len(differs)} file(s) under examples/ or docs/ of {env.src_dir} are "
                   f"not as the pinned commit has them: {shown}",
                   "restore the listed files from the pinned commit or move them aside", blocks=True)
    note = f" ({len(dirty)} larger file(s) were not compared)" if dirty else ""
    return Row("checkout files", OK, f"every file under examples/ and docs/ that was compared is as the pinned commit "
                                     f"has it{note}")


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


def _references_row(env, skill_dir, references, reference_source="checkout") -> Row:
    want = checkout_spec_version(env.src_dir)
    if references is None:
        references, problems = ttlang_reference.assemble(env, skill_dir, reference_source=reference_source)
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
    if off and reference_source == "skill":         # the operator asked for the older set, to measure it
        return Row("references", MISMATCH, f"the checkout documents specification {want} but " + ", ".join(off) +
                   " (the operator chose --reference-source skill)", "none: the choice is recorded in the report")
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


def _same_commit(have: str | None, want: str) -> bool:
    """Equal, or one is a prefix of the other (at least 7 characters, so a stub cannot match everything)."""
    have = (have or "").strip().lower()
    short = min(len(have), len(want))
    return bool(have) and short >= 7 and have[:short] == want.lower()[:short]


def _tree_row(env, tree: Path | None, build_info: dict | None = None) -> Row:
    """The hardware row. The host tt-metal tree is only named in the detail: the compiler was built against another
    tt-metal commit (ttlang_pin.TT_METAL_REF), so the tree's own version is not what the pin describes."""
    seen = "no readable tt-metal tree"
    if tree is not None and _exists(Path(tree) / ".git"):
        rc, out, _ = env.run(["git", "-C", str(tree), "describe", "--tags", "--always"], 30, ttlang_reference.git_env(env))
        seen = out.strip() if rc == 0 and out.strip() else seen
        env.run(["git", "-C", str(tree), "rev-parse", "HEAD"], 30, ttlang_reference.git_env(env))
    pin_lang, pin_metal = ttlang_pin.HARDWARE_TTLANG_SHA, ttlang_pin.TT_METAL_REF
    if build_info is None:
        return Row("hardware tree", TODO, f"todo: needs a leased run. `ttl.build_info()` in the hardware venv is compared with "
                   f"tt-lang {pin_lang[:9]} and tt-metal {pin_metal[:7]}; the host tree is {seen}",
                   "run `ttl.build_info()` under a gozer lease and pass the result in")
    lang, metal = build_info.get("tt_lang_commit"), build_info.get("tt_metal_commit")
    tag = build_info.get("tt_metal_tag")
    if not lang or not metal:
        return Row("hardware tree", UNKNOWN, "ttl.build_info() gave no " + " or ".join(
            n for n, v in (("tt-lang commit", lang), ("tt-metal commit", metal)) if not v) + ", so it cannot be compared with the pin")
    if _same_commit(lang, pin_lang) and _same_commit(metal, pin_metal):
        return Row("hardware tree", OK, f"the hardware venv has tt-lang {lang[:9]} and tt-metal {metal[:7]}"
                   f"{f' ({tag})' if tag else ''}, the pinned pair; the host tree is {seen}")
    parts = []
    if not _same_commit(lang, pin_lang):
        parts.append(f"tt-lang {lang[:12]} where the pin has {pin_lang[:12]}")
    if not _same_commit(metal, pin_metal):
        parts.append(f"tt-metal {metal[:12]}{f' ({tag})' if tag else ''} where the pin has {pin_metal[:12]}")
    return Row("hardware tree", MISMATCH, "the hardware venv has " + " and ".join(parts) +
               f" (version {build_info.get('version') or 'unknown'})",
               f"make the hardware venv from the pair: tt-lang-light=={ttlang_pin.WHEEL_VERSION} from {ttlang_pin.WHEEL_INDEX} "
               "(README section 5.9a), or move the pin", blocks=True)


def check(env, skill_dir, *, tt_metal_dir=None, references: list[dict] | None = None,
          reference_source: str = "checkout", build_info: dict | None = None) -> list[Row]:
    """Every row. `references` are the files the probe already assembled (so the check judges what the coder gets);
    without them the check assembles them itself, in memory, from `reference_source`. `build_info` is the hardware venv's
    `ttl.build_info()`, taken by the caller under a lease; without it the hardware row is a todo."""
    return [_checkout_row(env), _simulator_row(env), _files_row(env), _skill_row(env, skill_dir),
            _references_row(env, skill_dir, references, reference_source), _tree_row(env, tt_metal_dir, build_info)]


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


def load_build_info(path) -> dict:
    """The record `scripts/ttlang_build_info.py` wrote (tt_lang_commit, tt_metal_commit, optionally version and
    tt_metal_tag). ValueError for a file that is missing, not JSON or not an object."""
    try:
        data = safeio.read_json(path, max_bytes=64 * 1024)
    except (safeio.SafeReadError, ValueError) as exc:
        raise ValueError(f"{path} cannot be read as JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} holds {type(data).__name__}, not a JSON object")
    return data


def main(args, *, env=None, out=None) -> int:
    """`tt-orchard ttlang-consistency [--src-dir D] [--venv-dir D] [--tt-lang-skill-dir D] [--tt-metal-dir D]
    [--build-info-file F]`. `--build-info-file` is the JSON of scripts/ttlang_build_info.py; it turns the hardware row from todo into a comparison.
    Exit 0 when no row blocks, 2 otherwise. Changes nothing."""
    say = (lambda line: print(line, file=out)) if out is not None else print
    home = Path.home()
    env = env or ttlang_env.KEnv(
        src_dir=Path(args.src_dir or os.environ.get("ORCHARD_TTLANG_SRC") or home / "orchard-src" / "tt-lang"),
        venv_dir=Path(args.venv_dir or os.environ.get("ORCHARD_TTLANG_VENV") or home / "orchard-venvs" / "ttlang"),
        scratch_dir=Path(os.environ.get("ORCHARD_SCRATCH") or home / "orchard-src" / "scratch"),
        run=ttlang_env.real_run, environ=dict(os.environ))
    build_info = None
    if getattr(args, "build_info_file", None):
        try:
            build_info = load_build_info(args.build_info_file)
        except ValueError as exc:
            say(f"refused: --build-info-file: {exc}")
            return 2
    rows = check(env, ttlang_env.skill_dir_from(args), tt_metal_dir=tt_metal_dir_from(args), build_info=build_info)
    say(f"tt-lang pin {ttlang_pin.SHA[:9]} (simulator {ttlang_pin.SIM_VERSION})")
    for line in render(rows):
        say(line)
    return 2 if blocking(rows) else 0
