# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The managed tt-lang pair venv: the hardware environment the kernel probe's hardware phase runs in.

The pair is the compiler wheel and tt-metal commit recorded in orchard/ttlang_pin.py (`WHEEL_VERSION`, `WHEEL_SHA256`,
`TT_METAL_REF`). `tt-orchard kernel-env --pair` handles it in three ways:

    (no flag)   print the commands that build the venv, then say what the venv at the chosen path holds. Runs nothing
                that changes anything.
    --check     only the second part; exit 2 when the venv is missing or is not the pair.
    --apply     run the commands: download the two wheels, compare their sha256 with the pin BEFORE anything is
                installed, make the venv with `uv venv`, install the checked wheel files, and record the exports that
                `tt-lang-setup-external-tt-metal` prints for the host tt-metal tree in `<venv>/orchard-pair-exports.sh`.

Nothing here opens a device and nothing imports ttl or ttnn. The venv is read through a subprocess of its own python that
imports only importlib.metadata and json (`importlib.util.find_spec` and `pip show` give the same facts).

The venv is the whole interface: `resolve_hardware_python` picks it for the probe when it exists, and `hw_command` puts the
recorded exports (`TT_METAL_HOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`) in front of its python, because the light wheel has no
ttnn of its own and finds the host tree's through them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from orchard import ttlang_pin

EXPORTS_FILE = "orchard-pair-exports.sh"
# The only variables the exports file may set. Anything else the setup tool prints is dropped.
ALLOWED_EXPORTS = ("TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT", "PYTHONPATH", "LD_LIBRARY_PATH")
PYTHON_VERSION = "3.12"                      # the light wheel is a cp312 build

NOTES = (
    "The light wheel does not ship safetensors, which a model port needs to read its weights. Do not install it into the pair "
    "venv: put a directory that links to another venv's safetensors at the END of PYTHONPATH for that port. The tt-lang "
    "version and the build_info commits stay as the pin has them. ttl.build_info() opens the runtime, so run it only under a "
    "gozer lease (scripts/ttlang_build_info.py does, and `tt-orchard ttlang-consistency --build-info-file` reads its output).")


@dataclass
class Step:
    name: str
    shell: str                  # the command as a person would type it
    argv: list[str] | None      # the same command as an argument list, None when orchard does the step itself
    timeout: int = 1800


@dataclass
class Row:
    name: str
    status: str                 # ok, mismatch or todo
    detail: str


@dataclass
class Choice:
    python: str
    source: str                 # flag, pair or fallback
    warning: str | None = None


def default_venv(environ=None, home: Path | None = None) -> Path:
    """$ORCHARD_PAIR_VENV, else <cache root>/ttlang-pair/venv with the cache root $ORCHARD_CACHE_ROOT or ~/.cache/tt-orchard."""
    environ = os.environ if environ is None else environ
    if environ.get("ORCHARD_PAIR_VENV"):
        return Path(environ["ORCHARD_PAIR_VENV"])
    root = environ.get("ORCHARD_CACHE_ROOT") or Path(home or Path.home()) / ".cache" / "tt-orchard"
    return Path(root) / "ttlang-pair" / "venv"


def _wheel_pins() -> list[str]:
    """pip requirements for the two wheels: the light wheel, and the metapackage it points at (version plus +light)."""
    return [f"tt-lang-light=={ttlang_pin.WHEEL_VERSION}", f"tt-lang=={ttlang_pin.WHEEL_VERSION}+light"]


def build_plan(venv: Path, tt_metal_dir: Path, download_dir: Path) -> list[Step]:
    """The five steps, in order, from the pin's constants only."""
    venv, tree, dl = Path(venv), Path(tt_metal_dir), Path(download_dir)
    py = venv / "bin" / "python"
    pins = _wheel_pins()
    sums = [x for name, sha in ttlang_pin.WHEEL_SHA256.items() for x in (sha, name)]
    wheels = [str(dl / name) for name in ttlang_pin.WHEEL_SHA256]
    setup = str(venv / "bin" / "tt-lang-setup-external-tt-metal")
    download = ["python3", "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--find-links",
                ttlang_pin.WHEEL_INDEX, "-d", str(dl), *pins]
    return [
        Step("download", shlex.join(download), download),
        Step("hashes", f"(cd {shlex.quote(str(dl))} && printf '%s  %s\\n' {shlex.join(sums)} | sha256sum -c -)", None),
        Step("venv", shlex.join(["uv", "venv", "--python", PYTHON_VERSION, str(venv)]),
             ["uv", "venv", "--python", PYTHON_VERSION, str(venv)]),
        Step("install", shlex.join(["uv", "pip", "install", "--python", str(py), *wheels]),
             ["uv", "pip", "install", "--python", str(py), *wheels]),
        Step("exports", f"{shlex.quote(setup)} --tt-metal-dir {shlex.quote(str(tree))} > "
                        f"{shlex.quote(str(venv / EXPORTS_FILE))}", [setup, "--tt-metal-dir", str(tree)], 120),
    ]


def verify_wheel_hashes(download_dir: Path) -> list[str]:
    """Problems with the downloaded wheels against ttlang_pin.WHEEL_SHA256; empty when every file is there and matches."""
    problems = []
    for name, want in ttlang_pin.WHEEL_SHA256.items():
        path = Path(download_dir) / name
        try:
            have = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            problems.append(f"{name} was not downloaded")
            continue
        if have != want:
            problems.append(f"{name} has sha256 {have[:16]}... and the pin has {want[:16]}...")
    return problems


_SUFFIX = re.compile(r"\$\{(\w+):\+:\$\1\}")        # ${PYTHONPATH:+:$PYTHONPATH}: keep what the caller had, dropped here


def parse_exports(text: str) -> dict:
    """The allowed variables of `export NAME=value` lines. A value may use $TT_METAL_HOME and the ${X:+:$X} suffix; any other
    `$`, backtick or substitution is refused (ValueError), because the file is read back and put in a child's environment."""
    found: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("export "):
            continue
        name, _, value = line[len("export "):].partition("=")
        if name not in ALLOWED_EXPORTS:
            continue
        value = _SUFFIX.sub("", value).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        value = value.replace("${TT_METAL_HOME}", found.get("TT_METAL_HOME", "$")).replace(
            "$TT_METAL_HOME", found.get("TT_METAL_HOME", "$"))
        if "$" in value or "`" in value or "\n" in value:
            raise ValueError(f"export {name} uses a substitution this reader does not expand: {value!r}")
        found[name] = value
    return found


def exports_path(python) -> Path:
    return Path(python).parent.parent / EXPORTS_FILE


def read_exports(python) -> dict | None:
    """The recorded exports of the venv that `python` belongs to, or None when there is no readable, valid record."""
    try:
        got = parse_exports(exports_path(python).read_text())
    except (OSError, ValueError):
        return None
    return got or None


def hw_command(python) -> list[str]:
    """The command prefix for running `python` under a lease: `env NAME=value ... python` for a pair venv, else `python`."""
    got = read_exports(python)
    if not got:
        return [str(python)]
    return ["env", *[f"{k}={v}" for k, v in got.items()], str(python)]


def resolve_hardware_python(flag, *, environ=None, home: Path | None = None, fallback: str) -> Choice:
    """--shared-python wins; else the managed pair venv when it exists with its exports; else `fallback` (the old shared
    venv) with a warning that names the fix."""
    if flag:
        return Choice(str(flag), "flag")
    venv = default_venv(environ, home)
    py = venv / "bin" / "python"
    if not py.is_file():
        return Choice(fallback, "fallback", f"warning: no pair venv at {venv}; the hardware phase runs on the shared venv "
                      f"({fallback}), whose tt-lang is 1.1.6 and does not match the pin. Run `tt-orchard kernel-env --pair "
                      "--apply` to build the pair, or pass --shared-python.")
    if read_exports(py) is None:
        return Choice(fallback, "fallback", f"warning: the pair venv at {venv} has no valid {EXPORTS_FILE} (the host tt-metal "
                      f"exports), so it cannot find ttnn; the hardware phase runs on the shared venv ({fallback}). Run "
                      "`tt-orchard kernel-env --pair --apply` again.")
    return Choice(str(py), "pair")


_META = ("import importlib.metadata as m, json\n"
         "def v(n):\n"
         "    try:\n        return m.version(n)\n"
         "    except m.PackageNotFoundError:\n        return None\n"
         "print(json.dumps({n: v(n) for n in ('tt-lang-light', 'tt-lang')}))\n")


def verify(venv: Path, *, run: Callable) -> list[Row]:
    """What the venv holds, read through its own python without importing ttl or ttnn."""
    venv = Path(venv)
    py = venv / "bin" / "python"
    if not py.is_file():
        return [Row("venv", "todo", f"no venv at {venv}; run `tt-orchard kernel-env --pair --apply`")]
    rows = [Row("venv", "ok", f"{venv}")]
    rc, out, err = run([str(py), "-I", "-c", _META], 60)
    try:
        seen = json.loads(out) if rc == 0 else {}
    except ValueError:
        seen = {}
    want = {"tt-lang-light": ttlang_pin.WHEEL_VERSION, "tt-lang": ttlang_pin.WHEEL_VERSION + "+light"}
    for name, version in want.items():
        have = seen.get(name)
        if rc != 0 or not seen:
            rows.append(Row(name, "mismatch", f"the venv's python could not read its package metadata: {(err or out).strip()[:200]}"))
        elif have is None:
            rows.append(Row(name, "mismatch", f"{name} is not installed; the pin wants {version}"))
        elif have == version:
            rows.append(Row(name, "ok", f"{name} {have}"))
        else:
            rows.append(Row(name, "mismatch", f"{name} {have} is installed; the pin wants {version}"))
    got = read_exports(py)
    rows.append(Row("exports", "ok", f"{EXPORTS_FILE} names tt-metal {got['TT_METAL_HOME']}") if got and "TT_METAL_HOME" in got
                else Row("exports", "mismatch", f"{EXPORTS_FILE} is missing or has no TT_METAL_HOME; run "
                         "`tt-orchard kernel-env --pair --apply`"))
    return rows


def apply(steps: list[Step], *, run: Callable, say: Callable, venv: Path) -> bool:
    """Run the plan. Stops at the first failure. The wheel hashes are compared before `uv` makes the venv."""
    for step in steps:
        say(f"step {step.name}: {step.shell}")
        if step.name == "hashes":
            dl = Path(next(a for a in steps if a.name == "download").argv[steps[0].argv.index("-d") + 1])
            problems = verify_wheel_hashes(dl)
            for p in problems:
                say(f"fail sha256: {p}")
            if problems:
                return False
            continue
        rc, out, err = run(step.argv, step.timeout)
        if rc != 0:
            say(f"fail {step.name}: exit {rc}: {(err or out).strip()[-400:]}")
            return False
        if step.name == "exports":
            try:
                if not parse_exports(out):
                    raise ValueError("it printed no export of an allowed variable")
            except ValueError as exc:
                say(f"fail exports: {exc}")
                return False
            Path(venv).mkdir(parents=True, exist_ok=True)
            (Path(venv) / EXPORTS_FILE).write_text(out)
    return True


def main(args, *, run=None, out=None) -> int:
    """`tt-orchard kernel-env --pair [--check] [--apply] [--pair-venv-dir D] [--tt-metal-dir D] [--pair-download-dir D]`.
    Exit 0 when the venv matches the pin, 2 otherwise (printing the plan exits by the same rule)."""
    from orchard import ttlang_env
    say = (lambda line: print(line, file=out)) if out is not None else print
    run = run or ttlang_env.real_run
    venv = Path(getattr(args, "pair_venv_dir", None) or default_venv())
    tree = Path(getattr(args, "tt_metal_dir", None) or os.environ.get("ORCHARD_TT_METAL_DIR") or Path.home() / "tt-metal")
    dl = Path(getattr(args, "pair_download_dir", None) or venv.parent / "wheels")
    steps = build_plan(venv, tree, dl)
    if getattr(args, "apply", False):
        venv.parent.mkdir(parents=True, exist_ok=True)
        if not apply(steps, run=run, say=say, venv=venv):
            return 2
    elif not getattr(args, "check", False):
        say(f"tt-lang pair {ttlang_pin.WHEEL_VERSION} (tt-lang {ttlang_pin.SHA[:9]}, tt-metal {str(ttlang_pin.TT_METAL_REF)[:7]}).")
        say("These commands build the pair venv. Nothing was run; add --apply to run them.")
        for s in steps:
            say(f"  {s.name}: {s.shell}")
        say(f"note: {NOTES}")
    rows = verify(venv, run=run)
    for r in rows:
        say(f"{r.status:9s}{r.name}: {r.detail}")
    return 0 if all(r.status == "ok" for r in rows) else 2
