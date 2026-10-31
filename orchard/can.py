# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The canning step: turn a brought-up model's package pieces into a v6 thin bundle that `tt-model` can install and serve.

    tt-orchard can stage SPEC --out DIR        build the bundle (runs `tt-model package-thin --out`, never with a repo id)
    tt-orchard can gate  DIR  SPEC             check a staged bundle; prints each problem, exits 1 if there is one
    tt-orchard can boot  DIR  SPEC --work DIR  install a COPY of the bundle, serve it with its own run.sh, send requests, stop it
    tt-orchard can commands SPEC --namespace N print the publish commands as text (nothing here runs them)

A model's can is described by one small TOML file (config/can.granite.toml is the first). The spec names the server entry, the code that goes
with it, the pins, the weights pointer, the card, the mesh and the device count. Everything that is the same for every model lives here: the
package-thin call, the edits package-thin leaves wrong, the gate and the boot check.

Two kinds:
  * `vllm`: the model runs behind vLLM's OpenAI server through the tt-vllm plugin. The spec gives `metadata` (vllm_metadata.json) and `model_py`
    (the class the plugin loads). Used for models that vLLM can drive (Qwen3.5 on the vLLM path).
  * `tt-dit-server`: `run.sh` serves an ASGI app with uvicorn. The spec gives `app` ("module:attribute") and `model_py` (the file that holds it).
    A model with its own server (a standard-library HTTP server, a Gradio demo) needs a thin ASGI adapter that calls the same code; see
    pkgsrc/server/granite_asgi.py.

What the can does that package-thin does not:
  * `producer.hostname` in the manifest is replaced (package-thin writes the build host's name), run.sh and install.sh are made executable;
  * run.sh binds uvicorn to 127.0.0.1 (package-thin writes 0.0.0.0) and defaults to the spec's port (package-thin writes 8000);
    `HOST` and `PORT` still override both;
  * the pins must be exact (`name==version`), so the install builds the same environment each time;
  * SHA256SUMS lists every file, and the gate recomputes it.

The can never publishes. It runs `tt-model package-thin --out` and, in the boot check, the staged copy's own install.sh and run.sh. It never
calls `tt-model push` or `publish`, `hf upload` or `git push`. `publish_commands` returns text for the operator. The gate reuses stage 7's
package scrub (orchard/scrub.py, `scrub_package`) and stage 7's refusal of any upload-capable tt-model call (orchard/package.py,
`assert_no_publish`).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from orchard import safeio
from orchard.defaults import PACKAGE_INSTALL_TIMEOUT_S, PACKAGE_THIN_TIMEOUT_S
from orchard.package import PackageError, assert_no_publish, run_logged
from orchard.scrub import scrub_package

MAX_SPEC_BYTES = 256 * 1024         # choice: a spec is a few kilobytes
MAX_JSON_BYTES = 4 * 1024 * 1024    # choice: the manifest and the evidence file are a few kilobytes
KINDS = ("vllm", "tt-dit-server")
SUMS_FILE = "SHA256SUMS"
REDACTED = "redacted"
BOOT_HEALTH_TIMEOUT_S = 1800.0      # choice: how long the boot check waits for /health (a first start loads the weights and compiles kernels)
BOOT_REQUEST_TIMEOUT_S = 600.0      # choice: one chat request in the boot check
HOST_ARG = "--host 0.0.0.0"         # what package-thin writes into the uvicorn line
PORT_DEFAULT = '"${PORT:-8000}"'

TOP_KEYS = {"can", "weights", "env", "check", "extra_code", "extra_tree", "bundle_file"}
CAN_KEYS = {"name", "kind", "app", "model_py", "metadata", "readme", "arch", "mesh", "device_count", "python", "port", "requirements",
            "verify", "max_num_seqs", "block_size", "max_model_len", "mesh_fabric", "plugin_wheel", "ops_wheels", "models_wheels",
            "vllm_version"}
WEIGHTS_KEYS = {"repo", "revision"}
CHECK_KEYS = {"prompt", "expect", "max_tokens", "ready_contains", "min_decode_tok_s", "stream", "rate_prompt", "rate_max_tokens",
              "mode", "health_path", "lyrics", "caption", "seconds", "seed", "sample_rate", "min_seconds", "min_peak", "min_loud_fraction"}
CHECK_MODES = ("chat", "music")
FILE_KEYS = {"src", "dst"}
TREE_SKIP_DIRS = ("__pycache__",)          # a directory in an [[extra_tree]] that is never copied
PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?===?[A-Za-z0-9][A-Za-z0-9.!+_-]*(\s*;.*)?$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_PORTS = (8000, 8011, 8012, 8780)


class CanError(Exception):
    """The spec or the bundle is wrong, or a step failed. The message says what and where."""


@dataclass(frozen=True)
class Check:
    prompt: str = "What is the capital of France? Answer in one word."
    expect: str = "Paris"                  # a regular expression the chat answer must match
    max_tokens: int = 32
    ready_contains: tuple[str, ...] = ()   # strings the server's start-up log must contain (for example '"chunked_prefill": true')
    min_decode_tok_s: float | None = None  # a long answer's reported decode rate must be at least this (the server reports `timings`)
    rate_prompt: str = "Write a long story about a dragon who learns to read."   # asks for an answer long enough to time (the short answer is one token)
    rate_max_tokens: int = 64
    stream: bool = True
    # mode "chat" (the default) asks /v1/chat/completions. Mode "music" asks /v1/audio/speech for a short WAV clip and tests the audio:
    mode: str = "chat"
    health_path: str = "/health"           # the path the boot check waits on (a server of its own may use /v1/health)
    lyrics: str = "[verse]\nla la la la la\nla la la la la\n[chorus]\nla la la la la\nla la la la la"
    caption: str = "A warm acoustic pop song with a female vocal and fingerpicked guitar."
    seconds: float = 5.0                   # the clip length asked for (audio_duration)
    seed: int = 7
    sample_rate: int = 32000               # the rate the clip must have
    min_seconds: float | None = None       # the shortest clip that passes (default: 0.8 of `seconds`; a song ends when its lyrics are finished)
    min_peak: float = 0.05                 # the loudest sample, as a fraction of full scale
    min_loud_fraction: float = 0.3         # the share of 100 ms windows that must be louder than LOUD_DBFS


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    model_py: Path
    readme: Path
    repo: str
    revision: str
    arch: str = "blackhole"
    mesh: str = "P150"
    device_count: int = 1
    python: str = "3.12"
    port: int = 20110
    app: str | None = None
    metadata: Path | None = None
    requirements: tuple[str, ...] = ()
    verify: tuple[str, ...] = ()
    env: dict = field(default_factory=dict)
    extra_code: tuple[tuple[Path, str], ...] = ()      # (source file, path inside extra_code/)
    bundle_files: tuple[tuple[Path, str], ...] = ()    # (source file, path inside the bundle)
    check: Check = field(default_factory=Check)
    max_num_seqs: int | None = None
    block_size: int | None = None
    max_model_len: int | None = None
    mesh_fabric: str | None = None
    # kind vllm: wheels that ship in wheels/ and the vLLM version the plugin builds against
    plugin_wheel: Path | None = None
    ops_wheels: tuple[Path, ...] = ()
    models_wheels: tuple[Path, ...] = ()
    vllm_version: str | None = None


# ---- the spec ------------------------------------------------------------------------------------------------------------

def _table(raw: dict, key: str, allowed: set, what: str) -> dict:
    t = raw.get(key, {})
    if not isinstance(t, dict):
        raise CanError(f"{what} must be a table")
    extra = set(t) - allowed
    if extra:
        raise CanError(f"{what} has unknown keys {sorted(extra)}; allowed: {sorted(allowed)}")
    return t


def _inside(base: Path, rel, what: str, *, must_exist: bool = True) -> Path:
    """`rel` names a file inside `base`: relative, no `..`, no link, and it exists."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in Path(rel).parts:
        raise CanError(f"{what} must be a relative path inside the base directory, got {rel!r}")
    p = base / rel
    if must_exist and (p.is_symlink() or not p.is_file()):
        raise CanError(f"{what}: {rel} is not a regular file under {base}")
    if not p.resolve().is_relative_to(base.resolve()):      # a link in a directory component can lead outside base
        raise CanError(f"{what}: {rel} resolves to a path outside {base}")
    return p


def _dst(rel, what: str) -> str:
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in Path(rel).parts:
        raise CanError(f"{what} must be a relative path with no '..', got {rel!r}")
    return rel


def _tree(raw: dict, key: str, base: Path) -> tuple[tuple[Path, str], ...]:
    """`[[extra_tree]]` entries: every regular file under the directory `src` goes to `dst/<its path under src>`. A link anywhere in it is refused,
    and `__pycache__` directories and `.pyc` files are skipped."""
    rows = raw.get(key, [])
    if not isinstance(rows, list):
        raise CanError(f"[[{key}]] must be an array of tables")
    out = []
    for i, row in enumerate(rows):
        what = f"[[{key}]] {i}"
        if not isinstance(row, dict) or set(row) != FILE_KEYS:
            raise CanError(f"{what} needs exactly the keys {sorted(FILE_KEYS)}")
        src, dst = row["src"], _dst(row["dst"], f"{what} dst")
        root = _inside(base, src, f"{what} src", must_exist=False)
        if root.is_symlink() or not root.is_dir():
            raise CanError(f"{what} src: {src} is not a directory under {base} (a link to one is refused)")
        found = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in TREE_SKIP_DIRS)
            for name in sorted(dirnames) + sorted(filenames):
                p = Path(dirpath) / name
                if p.is_symlink():
                    raise CanError(f"{what}: {p.relative_to(base)} is a link; a tree holds regular files only")
            for name in sorted(filenames):
                if not name.endswith(".pyc"):
                    p = Path(dirpath) / name
                    found.append((p, (Path(dst) / p.relative_to(root)).as_posix()))
        if not found:
            raise CanError(f"{what}: {src} has no files to copy")
        out += found
    return tuple(out)


def _files(raw: dict, key: str, base: Path) -> tuple[tuple[Path, str], ...]:
    rows = raw.get(key, [])
    if not isinstance(rows, list):
        raise CanError(f"[[{key}]] must be an array of tables")
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != FILE_KEYS:
            raise CanError(f"[[{key}]] entry {i} needs exactly the keys {sorted(FILE_KEYS)}")
        out.append((_inside(base, row["src"], f"[[{key}]] {i} src"), _dst(row["dst"], f"[[{key}]] {i} dst")))
    return tuple(out)


def _extra_code(raw: dict, base: Path) -> tuple[tuple[Path, str], ...]:
    """[[extra_code]] files and [[extra_tree]] directories, as (source, destination) pairs. A destination named twice is refused."""
    pairs = _files(raw, "extra_code", base) + _tree(raw, "extra_tree", base)
    seen = set()
    for _, dst in pairs:
        if dst in seen:
            raise CanError(f"extra_code/{dst} is named twice")
        seen.add(dst)
    return pairs


def load_spec(path, *, base=None) -> Spec:
    """Read and check a spec. Paths in it are relative to `base` (default: the current directory). Raises CanError on any fault."""
    base = Path(base or Path.cwd()).resolve()
    try:
        raw = tomllib.loads(safeio.read_bytes(path, max_bytes=MAX_SPEC_BYTES).decode("utf-8"))
    except (OSError, safeio.SafeReadError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CanError(f"cannot read the spec {path}: {exc}") from exc
    extra = set(raw) - TOP_KEYS
    if extra:
        raise CanError(f"the spec has unknown tables {sorted(extra)}; allowed: {sorted(TOP_KEYS)}")
    can = _table(raw, "can", CAN_KEYS, "[can]")
    weights = _table(raw, "weights", WEIGHTS_KEYS, "[weights]")
    check = _table(raw, "check", CHECK_KEYS, "[check]")
    for need in ("name", "kind", "model_py", "readme"):
        if need not in can:
            raise CanError(f"[can] needs `{need}`")
    for need in ("repo", "revision"):
        if need not in weights:
            raise CanError(f"[weights] needs `{need}`")
    if not NAME.match(str(can["name"])):
        raise CanError(f"[can] name {can['name']!r} is not a plain repository name")
    if can["kind"] not in KINDS:
        raise CanError(f"[can] kind must be one of {KINDS}, got {can['kind']!r}")
    if not REVISION.match(str(weights["revision"])):
        raise CanError("[weights] revision must be a 40-character commit hash, so the bundle names one exact set of weights")
    if "/" not in str(weights["repo"]):
        raise CanError("[weights] repo must be ORG/NAME")
    port = can.get("port", 20110)
    if not isinstance(port, int) or not 1024 <= port <= 65535 or port in FORBIDDEN_PORTS or 8100 <= port <= 8103:
        raise CanError(f"[can] port {port!r} is not usable (it must be 1024 to 65535 and not one of the ports other work on this machine owns)")
    app = can.get("app")
    if can["kind"] == "tt-dit-server":
        if not isinstance(app, str) or not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", app):
            raise CanError("kind tt-dit-server needs [can] app = \"module:attribute\"")
        if can.get("metadata"):
            raise CanError("kind tt-dit-server serves no vLLM; remove [can] metadata")
    else:
        if app:
            raise CanError("kind vllm takes no [can] app")
        if not can.get("metadata"):
            raise CanError("kind vllm needs [can] metadata (the vllm_metadata.json)")
    pins = can.get("requirements", [])
    if not isinstance(pins, list) or not pins or not all(isinstance(p, str) for p in pins):
        raise CanError("[can] requirements must be a non-empty list of pin strings")
    pin_problems = requirement_problems("\n".join(pins))
    if pin_problems:
        raise CanError("[can] requirements: " + "; ".join(pin_problems))
    verify = can.get("verify", [])
    if not isinstance(verify, list) or not all(isinstance(v, str) for v in verify):
        raise CanError("[can] verify must be a list of python statements")
    env = raw.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise CanError("[env] must map names to strings")
    device_count = can.get("device_count", 1)
    if not isinstance(device_count, int) or isinstance(device_count, bool) or device_count < 1:
        raise CanError("[can] device_count must be a whole number of 1 or more")
    ck = Check(**{k: (tuple(v) if k == "ready_contains" else v) for k, v in check.items()})
    if ck.mode not in CHECK_MODES:
        raise CanError(f"[check] mode must be one of {CHECK_MODES}, got {ck.mode!r}")
    if not isinstance(ck.health_path, str) or not re.fullmatch(r"/[A-Za-z0-9._/-]*", ck.health_path):
        raise CanError(f"[check] health_path must be a URL path such as /health, got {ck.health_path!r}")
    try:
        re.compile(ck.expect)
    except re.error as exc:
        raise CanError(f"[check] expect is not a regular expression: {exc}") from exc
    optional = {k: can[k] for k in ("max_num_seqs", "block_size", "max_model_len", "mesh_fabric", "vllm_version") if k in can}
    wheel_keys = ("plugin_wheel", "ops_wheels", "models_wheels")
    if can["kind"] != "vllm" and any(k in can for k in wheel_keys + ("vllm_version",)):
        raise CanError("plugin_wheel, ops_wheels, models_wheels and vllm_version are for kind vllm")
    if "plugin_wheel" in can:
        optional["plugin_wheel"] = _inside(base, can["plugin_wheel"], "[can] plugin_wheel")
    for key in ("ops_wheels", "models_wheels"):
        if key in can:
            if not isinstance(can[key], list):
                raise CanError(f"[can] {key} must be a list of paths")
            optional[key] = tuple(_inside(base, w, f"[can] {key}") for w in can[key])
    return Spec(name=can["name"], kind=can["kind"], model_py=_inside(base, can["model_py"], "[can] model_py"),
                readme=_inside(base, can["readme"], "[can] readme"), repo=weights["repo"], revision=weights["revision"],
                arch=can.get("arch", "blackhole"), mesh=can.get("mesh", "P150"), device_count=device_count,
                python=can.get("python", "3.12"), port=port, app=app,
                metadata=_inside(base, can["metadata"], "[can] metadata") if can.get("metadata") else None,
                requirements=tuple(pins), verify=tuple(verify), env=dict(env),
                extra_code=_extra_code(raw, base), bundle_files=_files(raw, "bundle_file", base), check=ck, **optional)


# ---- staging -------------------------------------------------------------------------------------------------------------

def thin_argv(spec: Spec, *, out: Path, requirements: Path, extra_code: Path | None) -> list[str]:
    """The package-thin call for `spec`. It always has --out and never a repo id (`assert_no_publish` checks)."""
    argv = ["tt-model", "package-thin", "--model-py", spec.model_py, "--kind", spec.kind, "--requirements", requirements,
            "--weights", spec.repo, "--weights-revision", spec.revision, "--arch", spec.arch, "--mesh", spec.mesh,
            "--device-count", spec.device_count, "--python", spec.python, "--name", spec.name, "--readme", spec.readme]
    if spec.kind == "tt-dit-server":
        argv += ["--app", spec.app, "--no-vllm"]
    else:
        argv += ["--metadata", spec.metadata]
        if spec.plugin_wheel:
            argv += ["--plugin-wheel", spec.plugin_wheel]
        for flag, wheels in (("--ops-wheel", spec.ops_wheels), ("--models-wheel", spec.models_wheels)):
            for w in wheels:
                argv += [flag, w]
        if spec.vllm_version:
            argv += ["--vllm-version", spec.vllm_version]
    for flag, value in (("--max-num-seqs", spec.max_num_seqs), ("--block-size", spec.block_size),
                        ("--max-model-len", spec.max_model_len), ("--mesh-fabric", spec.mesh_fabric)):
        if value is not None:
            argv += [flag, value]
    if extra_code is not None:
        argv += ["--extra-code", extra_code]
    for statement in spec.verify:
        argv += ["--verify", statement]
    for k, v in spec.env.items():
        argv += ["--env", f"{k}={v}"]
    argv += ["--out", out]
    return [str(a) for a in argv]


def fix_run_sh(text: str, port: int) -> str:
    """package-thin's run.sh for a tt-dit-server binds uvicorn to every interface and defaults to port 8000. The server has no sign-in, so the
    bundle binds 127.0.0.1 unless HOST says otherwise, and defaults to the spec's port. Anything else in the line is left alone."""
    if HOST_ARG not in text or PORT_DEFAULT not in text:
        raise CanError("run.sh has no uvicorn line in the form this step edits (--host 0.0.0.0 and ${PORT:-8000}); "
                       "tt-model's launcher changed, and this step needs to follow it")
    return text.replace(HOST_ARG, '--host "${HOST:-127.0.0.1}"', 1).replace(PORT_DEFAULT, f'"${{PORT:-{port}}}"', 1)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files_of(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != SUMS_FILE)


def write_sums(root: Path) -> None:
    (root / SUMS_FILE).write_text("".join(f"{sha256(root / rel)}  {rel}\n" for rel in _files_of(root)), encoding="utf-8")


def sums_problems(root: Path) -> list[str]:
    path = root / SUMS_FILE
    if not path.is_file():
        return [f"{SUMS_FILE} is missing"]
    listed = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        digest, _, rel = line.partition("  ")
        listed[rel] = digest
    problems = [f"{SUMS_FILE} does not list {rel}" for rel in _files_of(root) if rel not in listed]
    for rel, digest in sorted(listed.items()):
        f = root / rel
        if not f.is_file():
            problems.append(f"{SUMS_FILE} lists {rel}, which is not in the bundle")
        elif sha256(f) != digest:
            problems.append(f"{rel} changed after {SUMS_FILE} was written")
    return problems


def _copy_in(pairs, dest: Path) -> None:
    for src, rel in pairs:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)


def stage(spec: Spec, out, *, env: dict | None = None, hostname: str | None = None, namespace: str | None = None) -> dict:
    """Build the bundle in the new directory `out`, edit what package-thin leaves wrong, write SHA256SUMS and run the gate.
    Returns the record (what was built, from what). Raises CanError or PackageError; a failed gate leaves `out` for inspection."""
    out = Path(out)
    if out.exists():
        raise CanError(f"{out} already exists; the can is staged into a fresh directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{spec.name}.can-", dir=out.parent))
    try:
        req = work / "requirements.txt"
        req.write_text("# Exact pins. The install builds a venv from these. ttnn comes from PyPI (it carries the TT-Metalium runtime).\n"
                       + "\n".join(spec.requirements) + "\n", encoding="utf-8")
        code = None
        if spec.extra_code:
            code = work / "extra_code"
            _copy_in(spec.extra_code, code)
        argv = thin_argv(spec, out=out, requirements=req, extra_code=code)
        assert_no_publish(argv)
        log = work / "package-thin.log"
        rc = run_logged(argv, log=log, timeout=PACKAGE_THIN_TIMEOUT_S, env=env)
        if rc != 0:
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]) if log.is_file() else ""
            raise CanError(f"tt-model package-thin for {spec.name} exited {rc}; the end of its log:\n{tail}")
        manifest_path = out / "tt_kernel_manifest.json"
        manifest = _json(manifest_path)
        manifest.setdefault("producer", {})["hostname"] = REDACTED
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        run_sh = out / "run.sh"
        if spec.kind == "tt-dit-server":
            run_sh.write_text(fix_run_sh(run_sh.read_text(encoding="utf-8"), spec.port), encoding="utf-8")
        _copy_in(spec.bundle_files, out)
        for f in ("run.sh", "install.sh"):
            (out / f).chmod(0o755)
        write_sums(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    problems = gate(out, spec, hostname=hostname, namespace=namespace)
    if problems:
        raise CanError(f"the gate found problems in {out}:\n  " + "\n  ".join(problems))
    return {"name": spec.name, "kind": spec.kind, "weights": {"repo": spec.repo, "revision": spec.revision}, "mesh": spec.mesh,
            "device_count": spec.device_count, "port": spec.port, "files": _files_of(out),
            "manifest_sha256": sha256(out / "tt_kernel_manifest.json"), "sums_sha256": sha256(out / SUMS_FILE)}


def _json(path: Path):
    try:
        value = safeio.read_json(path, max_bytes=MAX_JSON_BYTES)
    except (OSError, safeio.SafeReadError) as exc:
        raise CanError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise CanError(f"{path.name} must hold a JSON object")
    return value


# ---- the gate ------------------------------------------------------------------------------------------------------------

def requirement_problems(text: str) -> list[str]:
    """Every pin line must be `name==version`. Index options and comments are allowed; a TODO or an unpinned name is not."""
    problems = []
    for n, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split(" #", 1)[0].strip() if not raw_line.lstrip().startswith("#") else ""
        if not line or line.startswith("--"):
            continue
        if "TODO" in raw_line:
            problems.append(f"requirements.txt line {n} is a TODO: {raw_line.strip()}")
        elif not PIN.match(line):
            problems.append(f"requirements.txt line {n} is not an exact pin (name==version): {line}")
    return problems


def gate(pkg, spec: Spec, *, hostname: str | None = None, namespace: str | None = None) -> list[str]:
    """Every problem in a staged bundle (an empty list is a pass). It reads the directory as it is on disk and runs nothing."""
    pkg = Path(pkg)
    problems: list[str] = []
    if not pkg.is_dir():
        return [f"{pkg} is not a directory"]
    problems += [f"scrub: {hit}" for hit in scrub_package(pkg, hostname=hostname, namespace=namespace)]
    for need in ("tt_kernel_manifest.json", "run.sh", "install.sh", "requirements.txt", "README.md", SUMS_FILE, spec.model_py.name):
        if not (pkg / need).is_file():
            problems.append(f"{need} is missing")
    if problems and any("is missing" in p for p in problems):
        return problems
    problems += sums_problems(pkg)
    # the manifest loads with tt-model's own reader, and says what the spec says
    try:
        manifest = _json(pkg / "tt_kernel_manifest.json")
    except CanError as exc:
        return problems + [str(exc)]
    try:
        from tt_kernel.manifest import Manifest
        loaded = Manifest.from_json((pkg / "tt_kernel_manifest.json").read_text(encoding="utf-8"))
        if str(getattr(loaded, "schema_version", "")) != "6":
            problems.append(f"manifest schema_version is {getattr(loaded, 'schema_version', None)!r}, not '6'")
    except ImportError:
        problems.append("tt_kernel is not importable, so the manifest could not be loaded with tt-model's reader")
    except Exception as exc:  # noqa: BLE001  the reader raises its own error types
        problems.append(f"tt-model's reader refuses the manifest: {type(exc).__name__}: {exc}")
    deps, weights = manifest.get("deps") or {}, manifest.get("weights") or {}
    expect = {"kind": (deps.get("kind"), spec.kind), "app": (deps.get("app"), spec.app), "weights repo": (weights.get("repo_id"), spec.repo),
              "weights revision": (weights.get("revision"), spec.revision), "name": (manifest.get("name"), spec.name),
              "device_count": (manifest.get("device_count"), spec.device_count), "arch": (manifest.get("arch"), spec.arch),
              "mesh": ((manifest.get("mesh") or {}).get("topology"), spec.mesh)}
    for what, (got, want) in expect.items():
        if got != want:
            problems.append(f"manifest {what} is {got!r}; the spec says {want!r}")
    if (manifest.get("producer") or {}).get("hostname") != REDACTED:
        problems.append("manifest producer.hostname is not redacted")
    for f in ("run.sh", "install.sh"):
        if not os.access(pkg / f, os.X_OK):
            problems.append(f"{f} is not executable")
    run_sh = (pkg / "run.sh").read_text(encoding="utf-8", errors="replace")
    if spec.kind == "tt-dit-server":
        if HOST_ARG in run_sh:
            problems.append("run.sh binds uvicorn to 0.0.0.0; the server has no sign-in, so it must default to 127.0.0.1")
        if f"${{PORT:-{spec.port}}}" not in run_sh:
            problems.append(f"run.sh does not default to port {spec.port}")
    problems += requirement_problems((pkg / "requirements.txt").read_text(encoding="utf-8", errors="replace"))
    # the server entry is in the bundle and the code compiles
    if spec.kind == "tt-dit-server":
        module = spec.app.split(":")[0]
        as_path = Path(*module.split(".")).with_suffix(".py")       # server.api is server/api.py
        if not any((pkg / base / as_path).is_file() for base in (".", "extra_code")):
            problems.append(f"the ASGI module {module!r} is not at the bundle root or in extra_code/")
    # install.sh runs each verify statement with only extra_code/ on PYTHONPATH (found on the first real install: a statement that
    # imported a module at the bundle root passed in the bundle's own directory and failed under tt-model's installer)
    for statement in spec.verify:
        for name in re.findall(r"(?:^|;)\s*(?:import|from)\s+([A-Za-z_]\w*)", statement):
            if (pkg / f"{name}.py").is_file() and not (pkg / "extra_code" / f"{name}.py").is_file():
                problems.append(f"verify statement {statement!r} imports {name}, which is at the bundle root; install.sh puts only extra_code/ on the path, so move it there")
    for py in sorted(pkg.rglob("*.py")):
        try:
            compile(py.read_text(encoding="utf-8"), str(py), "exec")
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            problems.append(f"{py.relative_to(pkg).as_posix()} does not compile: {type(exc).__name__}: {exc}")
    sources = {w.name: w for w in (*([spec.plugin_wheel] if spec.plugin_wheel else []), *spec.ops_wheels, *spec.models_wheels)}
    shipped = {w.name: w for w in (pkg / "wheels").glob("*.whl")} if (pkg / "wheels").is_dir() else {}
    for name, w in sorted(shipped.items()):         # wheels are binary and the scrub does not read them
        if name not in sources:
            problems.append(f"wheels/{name} is not a wheel the spec names")
        elif sha256(w) != sha256(sources[name]):
            problems.append(f"wheels/{name} is not byte-identical to the spec's source")
    problems += [f"wheels/{name} is named in the spec and not in the bundle" for name in sorted(set(sources) - set(shipped))]
    for dst in (d for _, d in spec.extra_code):
        if not (pkg / "extra_code" / dst).is_file():
            problems.append(f"extra_code/{dst} is not in the bundle")
    return problems


# ---- the boot check ------------------------------------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, body: dict | None = None, *, timeout: float = 30.0) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _stop(proc: subprocess.Popen, grace: float = 60.0) -> None:
    """Stop the server's session. It started in its own session, so its process group id is its pid; nothing is matched by name."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 30.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


LOUD_DBFS = -40.0        # choice: a 100 ms window louder than this counts as sound (the clips of the first music runs had at least 59% of windows above it)
WINDOW_S = 0.1


def wav_stats(data: bytes) -> dict:
    """Sample rate, channels, length, peak (a fraction of full scale) and the share of 100 ms windows above LOUD_DBFS, for a 16-bit PCM WAV file.
    Raises CanError for anything else. Standard library only."""
    import array
    import wave
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            rate, channels, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError, struct.error) as exc:
        raise CanError(f"the response is not a readable WAV file: {exc}") from exc
    if width != 2:
        raise CanError(f"the WAV file is {8 * width}-bit; the check reads 16-bit PCM")
    if rate <= 0 or channels <= 0:
        raise CanError("the WAV file has no sample rate or no channels")
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - len(raw) % 2])
    if sys.byteorder == "big":
        samples.byteswap()
    peak = (max(abs(v) for v in samples) / 32768.0) if samples else 0.0
    step = max(1, int(rate * WINDOW_S)) * channels
    windows = loud = 0
    for start in range(0, len(samples), step):
        chunk = samples[start:start + step]
        windows += 1
        rms = math.sqrt(sum(v * v for v in chunk) / len(chunk)) / 32768.0
        if rms > 0 and 20 * math.log10(rms) > LOUD_DBFS:
            loud += 1
    return {"sample_rate": rate, "channels": channels, "seconds": (n / rate), "peak": round(peak, 4),
            "loud_fraction": round(loud / windows, 4) if windows else 0.0}


def _music_checks(base: str, ck: Check, check, evidence: dict) -> None:
    """The boot check of a music model: one short clip through /v1/audio/speech, then its sample rate, length and level."""
    body = {"input": ck.lyrics, "instructions": ck.caption, "audio_duration": ck.seconds, "response_format": "wav",
            "sample_rate": ck.sample_rate, "seed": ck.seed}
    t0 = time.time()
    status, raw = _http("POST", f"{base}/v1/audio/speech", body, timeout=BOOT_REQUEST_TIMEOUT_S)
    evidence["clip_wall_s"] = round(time.time() - t0, 1)
    if not check("music_request", status == 200, f"status {status}" + ("" if status == 200 else f": {raw[:200].decode('utf-8', errors='replace')}")):
        return
    try:
        st = wav_stats(raw)
    except CanError as exc:
        check("music_decodes", False, str(exc))
        return
    check("music_decodes", True, f"{len(raw)} bytes")
    evidence["audio"] = {**st, "requested_seconds": ck.seconds, "seed": ck.seed}
    check("music_sample_rate", st["sample_rate"] == ck.sample_rate, f"{st['sample_rate']} Hz; {ck.sample_rate} Hz wanted")
    low = ck.min_seconds if ck.min_seconds is not None else 0.8 * ck.seconds
    high = ck.seconds + 0.5            # the server counts 25 frames a second and may round up
    check("music_length", low <= st["seconds"] <= high, f"{st['seconds']:.2f} s; between {low:g} and {high:g} s wanted")
    check("music_not_silent", st["peak"] >= ck.min_peak and st["loud_fraction"] >= ck.min_loud_fraction,
          f"peak {st['peak']}, {st['loud_fraction']:.0%} of 100 ms windows above {LOUD_DBFS:g} dBFS; at least {ck.min_peak} and {ck.min_loud_fraction:.0%} wanted")


def boot_check(bundle, work, spec: Spec, *, env: dict | None = None, health_timeout: float = BOOT_HEALTH_TIMEOUT_S,
               port: int | None = None, install: bool = True) -> dict:
    """Install a COPY of the staged bundle under `work`, serve it with the copy's own run.sh and test it. The staged directory is never
    installed into or served. `env` is the environment of both children (for a hardware run it carries the lease's TT_VISIBLE_DEVICES).
    Returns the evidence; `passed` is true only if every check passed. A server that is still running is stopped on every path."""
    work = Path(work).resolve()          # the server starts in the copy's directory, so a relative path would double up
    work.mkdir(parents=True, exist_ok=True)
    copy = work / "bundle"
    if not copy.exists():
        shutil.copytree(bundle, copy, symlinks=True)
    env = dict(os.environ if env is None else env)
    log_dir = work / "logs"
    log_dir.mkdir(exist_ok=True)
    evidence: dict = {"name": spec.name, "checks": {}, "passed": False}

    def check(name: str, ok: bool, detail=None) -> bool:
        evidence["checks"][name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        return bool(ok)

    stale = [rel for rel in _files_of(Path(bundle)) if not (copy / rel).is_file() or sha256(copy / rel) != sha256(Path(bundle) / rel)]
    if not check("bundle_copy", not stale, "the copy matches the bundle" if not stale else
                 f"the copy in {work} differs from the requested bundle ({stale[0]} and {len(stale) - 1} more); use a new --work directory"):
        return _finish(work, evidence)
    if install and not (copy / "venv" / "bin" / "python").exists():
        rc = run_logged(["bash", copy / "install.sh"], log=log_dir / "install.log", timeout=PACKAGE_INSTALL_TIMEOUT_S, env=env)
        if not check("install", rc == 0, f"install.sh exited {rc}"):
            return _finish(work, evidence)
    port = port or free_port()
    base = f"http://127.0.0.1:{port}"
    server_log = log_dir / "server.log"
    serve_env = {**env, "PORT": str(port), "HOST": "127.0.0.1"}
    with open(server_log, "wb") as out:          # a log from an earlier run must not satisfy ready_contains
        proc = subprocess.Popen(["bash", str(copy / "run.sh")], stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=serve_env,
                                start_new_session=True, cwd=str(copy))
    evidence["server_pid"] = proc.pid
    try:
        t0, healthy = time.time(), False
        while time.time() - t0 < health_timeout and proc.poll() is None:
            try:
                healthy = _http("GET", f"{base}{spec.check.health_path}", timeout=5)[0] == 200
            except (OSError, urllib.error.URLError):
                healthy = False
            if healthy:
                break
            time.sleep(3)
        evidence["ready_s"] = round(time.time() - t0, 1)
        if not check("health", healthy, f"200 from {spec.check.health_path}" if healthy else
                     f"no 200 from {spec.check.health_path} ({'the server exited with ' + str(proc.returncode) if proc.poll() is not None else 'timed out'})"):
            return _finish(work, evidence)
        status, body = _http("GET", f"{base}/v1/models")
        check("models", status == 200)
        ck = spec.check
        if ck.mode == "music":
            _music_checks(base, ck, check, evidence)
            status, _ = _http("GET", f"{base}{ck.health_path}")
            check("health_after", status == 200)
            log_text = server_log.read_text(encoding="utf-8", errors="replace")
            for needle in ck.ready_contains:
                check(f"log contains {needle}", needle in log_text)
            return _finish(work, evidence)
        status, body = _http("POST", f"{base}/v1/chat/completions", {"messages": [{"role": "user", "content": ck.prompt}], "max_tokens": ck.max_tokens},
                             timeout=BOOT_REQUEST_TIMEOUT_S)
        answer, timings = "", {}
        try:
            answer_obj = safeio.parse_json_bytes(body, max_bytes=MAX_JSON_BYTES, expect=dict)
            answer = answer_obj["choices"][0]["message"]["content"]
            timings = answer_obj.get("timings") or {}
        except (safeio.SafeReadError, KeyError, IndexError, TypeError):
            pass
        evidence["answer"], evidence["timings"] = answer[:400], timings
        check("chat", status == 200 and re.search(ck.expect, answer) is not None, f"status {status}; expected /{ck.expect}/")
        if ck.min_decode_tok_s is not None:
            # a separate request for a long answer, after the first one has warmed the kernels (a short answer has no decode rate)
            code, raw = _http("POST", f"{base}/v1/chat/completions", {"messages": [{"role": "user", "content": ck.rate_prompt}],
                                                                       "max_tokens": ck.rate_max_tokens}, timeout=BOOT_REQUEST_TIMEOUT_S)
            rate_timings = {}
            try:
                rate_timings = safeio.parse_json_bytes(raw, max_bytes=MAX_JSON_BYTES, expect=dict).get("timings") or {}
            except safeio.SafeReadError:
                pass
            evidence["rate_timings"] = rate_timings
            rate = rate_timings.get("decode_tok_s")
            check("decode_rate", code == 200 and isinstance(rate, (int, float)) and rate >= ck.min_decode_tok_s,
                  f"{rate} tok/s reported over {rate_timings.get('completion_tokens')} tokens; at least {ck.min_decode_tok_s} wanted")
        if ck.stream:
            status, body = _http("POST", f"{base}/v1/chat/completions", {"messages": [{"role": "user", "content": ck.prompt}],
                                                                           "max_tokens": ck.max_tokens, "stream": True}, timeout=BOOT_REQUEST_TIMEOUT_S)
            text = body.decode("utf-8", errors="replace")
            check("stream", status == 200 and text.rstrip().endswith("data: [DONE]"), f"status {status}; the event stream {'ends' if text.rstrip().endswith('data: [DONE]') else 'does not end'} with [DONE]")
        status, _ = _http("GET", f"{base}/health")
        check("health_after", status == 200)
        log_text = server_log.read_text(encoding="utf-8", errors="replace")
        for needle in ck.ready_contains:
            check(f"log contains {needle}", needle in log_text)
        evidence["passed"] = all(c["ok"] for c in evidence["checks"].values())
        return _finish(work, evidence)
    except OSError as exc:       # a timeout or a refused connection after the server was ready: record it, and overwrite any earlier evidence
        check("transport", False, f"{type(exc).__name__}: {exc}")
        return _finish(work, evidence)
    finally:
        _stop(proc)


def _finish(work: Path, evidence: dict) -> dict:
    evidence["passed"] = bool(evidence["checks"]) and all(c["ok"] for c in evidence["checks"].values())
    (work / "boot-check.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


# ---- text for the operator ----------------------------------------------------------------------------------------------

def publish_commands(spec: Spec, namespace: str = "NAMESPACE") -> str:
    """The commands that would publish the can, as text. Nothing in this module runs them."""
    repo = f"{namespace}/{spec.name}"
    return "\n".join([
        f"# Commands that would publish the can for {spec.repo}. The canning step never ran them. Each needs the operator's explicit go-ahead.",
        "# 1. Create a private repository and upload the bundle directory (it holds no weights):",
        f"#    hf upload --repo-type model --private {repo} <staged bundle directory> .",
        "# 2. Pull and serve it from the private repository before anything else:",
        f"#    tt-model serve {repo}",
        "# 3. After that works, make it public (a separate decision):",
        f"#    hf repos settings {repo} --no-private",
        "# 4. List it in the tt-model catalog (a separate decision):",
        f"#    tt-model publish {repo}", ""])


def publish_problems(text: str) -> list[str]:
    """Any line of `text` that is a command and not a comment. The publish text must hold only comments."""
    return [f"line {n} would run: {line}" for n, line in enumerate(text.splitlines(), 1) if line.strip() and not line.lstrip().startswith("#")]


def record(run_dir, record_: dict, evidence: dict | None = None) -> Path:
    """Write stages/7/can.json in a run directory and append an evidence entry to its ledger."""
    from orchard.ledger import Ledger
    run = Path(run_dir)
    stage_dir = run / "stages" / "7"
    stage_dir.mkdir(parents=True, exist_ok=True)
    body = {**record_, **({"boot_check": evidence} if evidence is not None else {})}
    path = stage_dir / "can.json"
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    ledger = Ledger(run / "ledger.jsonl")
    try:
        ledger.append("evidence", 7, what="can staged", file="stages/7/can.json", sha256=sha256(path),
                      name=record_["name"], kind=record_["kind"], booted=bool(evidence and evidence.get("passed")))
    finally:
        ledger.close()
    return path


# ---- command line --------------------------------------------------------------------------------------------------------

def main(args, out=None) -> int:
    out = out or sys.stdout
    say = lambda text: print(text, file=out)  # noqa: E731
    try:
        spec = load_spec(args.spec, base=getattr(args, "base", None))
        if args.action == "stage":
            if not args.out:
                raise CanError("can stage needs --out")
            rec = stage(spec, args.out, namespace=getattr(args, "namespace", None))
            say(f"staged {rec['name']} ({len(rec['files'])} files) in {args.out}")
            if getattr(args, "run_dir", None):
                say(f"recorded in {record(args.run_dir, rec)}")
            return 0
        if args.action == "gate":
            problems = gate(args.bundle, spec, namespace=getattr(args, "namespace", None))
            for p in problems:
                say(f"problem: {p}")
            say("gate: " + ("passed" if not problems else f"{len(problems)} problem(s)"))
            return 1 if problems else 0
        if args.action == "boot":
            if not args.work:
                raise CanError("can boot needs --work")
            env = dict(os.environ)
            if getattr(args, "hf_home", None):
                env["HF_HOME"] = str(args.hf_home)
            ev = boot_check(args.bundle, args.work, spec, env=env, health_timeout=args.health_timeout)
            for name, c in ev["checks"].items():
                say(f"{'ok  ' if c['ok'] else 'FAIL'} {name} {c.get('detail', '')}".rstrip())
            say(f"boot check: {'passed' if ev['passed'] else 'FAILED'} (evidence: {Path(args.work) / 'boot-check.json'})")
            if getattr(args, "run_dir", None):
                rec = {"name": spec.name, "kind": spec.kind, "weights": {"repo": spec.repo, "revision": spec.revision}}
                say(f"recorded in {record(args.run_dir, rec, ev)}")
            return 0 if ev["passed"] else 1
        if args.action == "commands":
            say(publish_commands(spec, args.namespace or "NAMESPACE"))
            return 0
    except (CanError, PackageError) as exc:
        say(f"can: {exc}")
        return 2
    say(f"unknown action {args.action!r}")
    return 2
