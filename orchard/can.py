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
    (the class the plugin loads), the wheels the model needs (`plugin_wheel`, `ops_wheels`, `models_wheels`), the fixed serving arguments
    (`extra_args`, `tt_config`, `trace_region_bytes`, the limits and `[env]`) and, when the checkpoint needs a config of its own, a `[model_dir]`
    that wires the weights through stage 7's prepare_model_dir.py. Used for models that vLLM can drive (Qwen3.5-35B, config/can.qwen35.toml).
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
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from orchard import can_audio, safeio, served_verify
from orchard.defaults import PACKAGE_INSTALL_TIMEOUT_S, PACKAGE_THIN_TIMEOUT_S
from orchard.package import (PackageError, TEMPLATES, assert_no_publish, run_logged, splice_extra_args, weights_wiring_problems,
                             wire_weights)
from orchard.scrub import scrub_package

MAX_SPEC_BYTES = 256 * 1024         # choice: a spec is a few kilobytes
MAX_JSON_BYTES = 4 * 1024 * 1024    # choice: the manifest and the evidence file are a few kilobytes
MAX_LOG_BYTES = 64 * 1024 * 1024    # choice: the most of a server log the boot check reads to look for the mesh line (a larger log is not read)
KINDS = ("vllm", "tt-dit-server")
SUMS_FILE = "SHA256SUMS"
REDACTED = "redacted"
BOOT_HEALTH_TIMEOUT_S = 1800.0      # choice: how long the boot check waits for /health (a first start loads the weights and compiles kernels)
BOOT_REQUEST_TIMEOUT_S = 600.0      # choice: one chat request in the boot check
HOST_ARG = "--host 0.0.0.0"         # what package-thin writes into the uvicorn line
PORT_DEFAULT = '"${PORT:-8000}"'

TOP_KEYS = {"can", "weights", "env", "check", "extra_code", "extra_tree", "bundle_file", "model_dir", "plugin_edits"}
CAN_KEYS = {"name", "kind", "app", "model_py", "metadata", "readme", "arch", "mesh", "device_count", "python", "port", "requirements",
            "verify", "max_num_seqs", "block_size", "max_model_len", "mesh_fabric", "plugin_wheel", "ops_wheels", "models_wheels",
            "vllm_version", "extra_args", "tt_config", "trace_region_bytes", "card_profile"}
# card profiles: "catalog" asks the card for the sections tt-model's card_publish_gaps() names as required, with content in them
CARD_PROFILES = ("catalog", "catalog-measured")
# "catalog-measured" (issue 365) adds the conditions a performance claim needs, from tenstorrent/skills common/tt-perf-claim-review: whether the
# kernel cache was warm or cold, how many runs stand behind a number, and the load on the host during the measurement. The gate checks that the
# section says each one. It cannot check that what it says is true; a reviewer reads the evidence (docs/can-cards.md).
_COND_CACHE = re.compile(r"\b(?:warm|cold)\b|\bempty (?:kernel )?cache\b|\bkernels? (?:already )?compiled\b", re.I)
_COND_RUNS = re.compile(r"\b[1-9]\d*\s+(?:runs?|samples?|repetitions?|trials?)\b|\b(?:runs?|samples?|repetitions?|trials?|N)\s*[:=]?\s*[1-9]\d*\b"
                        r"|\b(?:one|single|a single)\s+(?:run|sample|request)\b", re.I)
_COND_LOAD = re.compile(r"\bload(?: average)?\b\W{0,4}(?:of |was |about )?\d|\b(?:host|machine|box)(?: was)? (?:idle|quiet)\b|\bidle (?:host|machine|box)\b", re.I)
MODEL_DIR_KEYS = {"base_config"}
# set only by a kind vllm spec: the wheels, the serving arguments (an error names these for a tt-dit-server spec that sets one)
VLLM_ONLY_KEYS = ("plugin_wheel", "ops_wheels", "models_wheels", "vllm_version", "extra_args", "tt_config", "trace_region_bytes")
# package-thin owns these `tt` keys of --additional-config (it writes them from --mesh-fabric and --trace-region-bytes)
RESERVED_TT_KEYS = ("fabric_config", "trace_region_size")
# vLLM flags the can sets itself, so a spec's extra_args may not repeat them
OWNED_FLAGS = ("--model", "--host", "--port", "--revision", "--tokenizer_revision", "--additional_config",
               "--max_num_seqs", "--block_size", "--max_model_len")     # names as flag_name() writes them: vLLM takes --max-model-len and --max_model_len alike


def flag_name(word: str) -> str:
    """A command-line word's option name with dashes and underscores unified (`--max-model-len=4096` and `--max_model_len` are the same option)."""
    return "--" + word.split("=", 1)[0].lstrip("-").replace("-", "_")
# stage 7's weights_wiring_problems asks that run.sh name no "nearest model" (a fine-tune is built from a base model). A can has none, and its
# run.sh names its own weights repo, so the check gets an id that cannot occur.
NO_NEAREST_MODEL = "orchard-can/no-nearest-model"
WEIGHTS_KEYS = {"repo", "revision"}
CHECK_KEYS = {"prompt", "expect", "max_tokens", "ready_contains", "min_decode_tok_s", "stream", "rate_prompt", "rate_max_tokens", "request_extra",
              "mode", "health_path", "lyrics", "caption", "seconds", "seed", "sample_rate", "min_seconds", "min_peak", "min_loud_fraction",
              "path", "body", "expect_seconds", "channels", "max_warm_s", "repeat_identical",
              "audio_seconds", "audio_steps", "audio_seed", "max_rtf"}
CHECK_MODES = ("chat", "music", "audio", "audio-job")
CHAT_ONLY_KEYS = {"expect", "max_tokens", "min_decode_tok_s", "stream", "rate_prompt", "rate_max_tokens"}   # refused in mode "audio-job"
AUDIO_JOB_KEYS = {"audio_seconds", "audio_steps", "audio_seed", "max_rtf"}                                  # only for mode "audio-job" (orchard/can_audio.py)
FILE_KEYS = {"src", "dst"}
TREE_SKIP_DIRS = ("__pycache__",)          # a directory in an [[extra_tree]] that is never copied
PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?===?[A-Za-z0-9][A-Za-z0-9.!+_-]*(\s*;.*)?$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_PORTS = (8000, 8011, 8012, 8780)
# Issue 362 (tenstorrent/skills PR 40): a bundle file under one of these directory names (or named like one) changes the vLLM plugin or vLLM
# itself. The spec's [plugin_edits] table must give a reason and say where the plugin interface was checked for an option that already does it.
PLUGIN_DIRS = frozenset({"vllm", "vllm_tt_plugin", "tt_vllm_plugin", "vllm-tt-plugin", "vllm_tt"})
PLUGIN_EDIT_KEYS = {"reason", "checked"}
PLUGIN_EDIT_MIN = 20                # choice: the shortest reason or check text; a sentence, not a word
PLUGIN_EDIT_MAX = 600


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
    # fields merged into every chat request body (for a thinking model: { chat_template_kwargs = { enable_thinking = false } })
    request_extra: dict = field(default_factory=dict)
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
    # mode "audio": the model answers `POST path` with a WAV clip instead of a chat answer (ACE-Step). The boot check sends `body` twice (the
    # first request compiles kernels, the second is timed) and checks the clip. `ready_contains` and the health checks apply to both modes.
    path: str = ""
    body: dict | None = None
    expect_seconds: float | None = None    # the clip's length, within 5 percent (and at least 0.1 s)
    channels: int = 2
    max_warm_s: float | None = None        # the second request must finish within this many seconds (none: the time is recorded, not checked)
    repeat_identical: bool = False         # the second request (same body, same seed) must return the same samples (the WAV header may differ)
    audio_seconds: float = 10.0            # mode audio-job: the clip length the boot check asks for
    audio_steps: int = 8
    audio_seed: int = 42
    max_rtf: float | None = None           # mode audio-job: the warm job's generation time divided by its audio length must be at most this


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
    # kind vllm: fixed vLLM arguments after the generated ones, the `tt` keys of --additional-config, the trace region, and the files that
    # make the model directory (a config.json that names the architecture the model class registers; prepare_model_dir.py links the
    # weights from the Hugging Face snapshot beside it)
    extra_args: tuple[str, ...] = ()
    tt_config: dict = field(default_factory=dict)
    trace_region_bytes: int | None = None
    base_config: tuple[tuple[Path, str], ...] = ()
    # None: the gate does not read the card. "catalog": the gate requires "Expected performance" with a measured row and "Limitations" with text
    card_profile: str | None = None
    # path inside the bundle -> {"reason", "checked"} for each file that changes the vLLM plugin or vLLM (issue 362)
    plugin_edits: dict = field(default_factory=dict)


# ---- the spec ------------------------------------------------------------------------------------------------------------

def check_port(port, what: str):
    """`port` if it is a whole number from 1024 to 65535 that is not one of the ports other work on this machine owns; else raises CanError.
    The spec's port and the `can boot --port` override both go through this."""
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535 or port in FORBIDDEN_PORTS or 8100 <= port <= 8103:
        raise CanError(f"{what} {port!r} is not usable (it must be 1024 to 65535 and not one of the ports other work on this machine owns)")
    return port


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


def _tree(raw: dict, key: str, base: Path, expand: bool = True) -> tuple[tuple[Path, str], ...]:
    """`[[extra_tree]]` entries: every regular file under the directory `src` goes to `dst/<its path under src>`. A link anywhere in it is refused,
    and `__pycache__` directories and `.pyc` files are skipped. With `expand` false the entries are checked (keys, `dst`, `src` is a directory
    and not a link) and the directory is not walked: the result is empty. The Cans view loads specs that way, so a spec that names a large
    subtree costs it nothing."""
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
        if not expand:
            continue
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


def _serving(can: dict, raw: dict, base: Path) -> dict:
    """The kind-vllm serving fields: extra vLLM arguments, `tt` additional-config keys, the trace region and the model directory's files."""
    out: dict = {}
    if "extra_args" in can:
        args = can["extra_args"]
        if not isinstance(args, list) or not all(isinstance(a, str) and a for a in args):
            raise CanError("[can] extra_args must be a list of non-empty strings (one vLLM command-line word each)")
        for a in args:
            if a.startswith("--") and flag_name(a) in OWNED_FLAGS:
                raise CanError(f"[can] extra_args must not set {flag_name(a)}: the can sets it (see docs/canning.md)")
        out["extra_args"] = tuple(args)
    if "tt_config" in can:
        tt = can["tt_config"]
        if not isinstance(tt, dict):
            raise CanError("[can] tt_config must be a table of `tt` additional-config keys")
        for key in RESERVED_TT_KEYS:
            if key in tt:
                raise CanError(f"[can] tt_config key {key} is reserved: package-thin writes it from mesh_fabric and trace_region_bytes")
        out["tt_config"] = dict(tt)
    if "trace_region_bytes" in can:
        n = can["trace_region_bytes"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise CanError("[can] trace_region_bytes must be a whole number of bytes")
        out["trace_region_bytes"] = n
    if "model_dir" in raw:
        table = _table(raw, "model_dir", MODEL_DIR_KEYS, "[model_dir]")
        files = table.get("base_config")
        if not isinstance(files, list) or not files:
            raise CanError("[model_dir] base_config must be a non-empty list of files (config.json at least)")
        pairs = tuple((_inside(base, f, "[model_dir] base_config"), Path(f).name) for f in files)
        if len({d for _, d in pairs}) != len(pairs):
            raise CanError("[model_dir] base_config names two files with the same file name")
        if "config.json" not in {d for _, d in pairs}:
            raise CanError("[model_dir] base_config must include a file named config.json (the model class's architecture name goes there)")
        out["base_config"] = pairs
    return out


def _plugin_edits(raw: dict) -> dict:
    """The [plugin_edits] table: each key is a path inside the bundle, each value a table with a `reason` and a `checked` text."""
    table = raw.get("plugin_edits", {})
    if not isinstance(table, dict):
        raise CanError("[plugin_edits] must be a table")
    out = {}
    for path, entry in table.items():
        what = f"[plugin_edits.{path!r}]"
        _dst(path, what)
        if not isinstance(entry, dict):
            raise CanError(f"{what} must be a table with the keys {sorted(PLUGIN_EDIT_KEYS)}")
        extra = set(entry) - PLUGIN_EDIT_KEYS
        if extra:
            raise CanError(f"{what} has unknown keys {sorted(extra)}; allowed: {sorted(PLUGIN_EDIT_KEYS)}")
        for key in sorted(PLUGIN_EDIT_KEYS):
            v = entry.get(key)
            if not isinstance(v, str) or not PLUGIN_EDIT_MIN <= len(v.strip()) <= PLUGIN_EDIT_MAX:
                raise CanError(f"{what} {key} must be a sentence of {PLUGIN_EDIT_MIN} to {PLUGIN_EDIT_MAX} characters "
                               "(reason: why the plugin or vLLM must change; checked: where you looked for an option the plugin already offers)")
        out[path] = {k: entry[k].strip() for k in PLUGIN_EDIT_KEYS}
    return out


def plugin_edit_files(pkg) -> list[str]:
    """The bundle files that change the vLLM plugin or vLLM: a path with a directory (or a `.py` file) named like PLUGIN_DIRS. A wheel's file name never matches, so the shipped wheels are not listed. Sorted, relative to the bundle."""
    root = Path(pkg)
    found = []
    for rel in _files_of(root):
        parts = Path(rel).parts
        if any(part in PLUGIN_DIRS or (part.endswith(".py") and part[:-3] in PLUGIN_DIRS) for part in parts):
            found.append(rel)
    return found


def plugin_edit_problems(pkg, reasons: dict) -> list[str]:
    """One problem for each plugin or vLLM file with no entry in `reasons`, and one for each entry that names a file the bundle does not have."""
    files = plugin_edit_files(pkg)
    problems = [f"{rel} changes the vLLM plugin or vLLM and has no stated reason (add a [plugin_edits.\"{rel}\"] table with `reason` and `checked`; "
                "see the canning skill)" for rel in files if rel not in reasons]
    problems += [f"[plugin_edits] names {rel}, which is not in the bundle" for rel in sorted(set(reasons) - set(files))]
    return problems


def _extra_code(raw: dict, base: Path, expand_trees: bool = True) -> tuple[tuple[Path, str], ...]:
    """[[extra_code]] files and [[extra_tree]] directories, as (source, destination) pairs. A destination named twice is refused."""
    pairs = _files(raw, "extra_code", base) + _tree(raw, "extra_tree", base, expand_trees)
    seen = set()
    for _, dst in pairs:
        if dst in seen:
            raise CanError(f"extra_code/{dst} is named twice")
        seen.add(dst)
    return pairs


def load_spec(path, *, base=None, expand_trees: bool = True) -> Spec:
    """Read and check a spec. Paths in it are relative to `base` (default: the current directory). Raises CanError on any fault. With
    `expand_trees` false the `[[extra_tree]]` directories are checked but not walked, so `extra_code` holds only the `[[extra_code]]` files
    (for a reader that only needs the spec's facts, such as the Cans view)."""
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
    for what, value in (("[can] name", can["name"]), ("[can] kind", can["kind"]), ("[weights] repo", weights["repo"]),
                        ("[weights] revision", weights["revision"]), ("[can] mesh", can.get("mesh", "P150")), ("[can] arch", can.get("arch", "blackhole"))):
        if not isinstance(value, str):
            raise CanError(f"{what} must be a string")
    if not NAME.fullmatch(str(can["name"])):
        raise CanError(f"[can] name {can['name']!r} is not a plain repository name")
    if can["kind"] not in KINDS:
        raise CanError(f"[can] kind must be one of {KINDS}, got {can['kind']!r}")
    if not REVISION.fullmatch(str(weights["revision"])):
        raise CanError("[weights] revision must be a 40-character commit hash, so the bundle names one exact set of weights")
    if "/" not in str(weights["repo"]):
        raise CanError("[weights] repo must be ORG/NAME")
    port = check_port(can.get("port", 20110), "[can] port")
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
    mode = check.get("mode", "chat")
    if mode not in CHECK_MODES:
        raise CanError(f"[check] mode must be one of {CHECK_MODES}, got {mode!r}")
    wrong = set(check) & (CHAT_ONLY_KEYS if mode == "audio-job" else AUDIO_JOB_KEYS)
    if wrong:
        raise CanError(f"[check] {sorted(wrong)} belong to the {'chat' if mode == 'audio-job' else 'audio-job'} mode, and mode is {mode!r}")
    if mode == "audio-job":
        if "prompt" not in check:
            raise CanError("[check] mode audio-job needs a prompt")
        for key in ("audio_seconds", "max_rtf"):
            v = check.get(key)
            if key in check and (isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0):
                raise CanError(f"[check] {key} must be a number above 0")
        for key in ("audio_steps", "audio_seed"):
            v = check.get(key)
            if key in check and (isinstance(v, bool) or not isinstance(v, int) or v < (1 if key == "audio_steps" else 0)):
                raise CanError(f"[check] {key} must be a whole number ({'1' if key == 'audio_steps' else '0'} or more)")
    ck = Check(**{k: (tuple(v) if k == "ready_contains" else v) for k, v in check.items()})
    if ck.mode not in CHECK_MODES:
        raise CanError(f"[check] mode must be one of {CHECK_MODES}, got {ck.mode!r}")
    if not isinstance(ck.health_path, str) or not re.fullmatch(r"/[A-Za-z0-9._/-]*", ck.health_path):
        raise CanError(f"[check] health_path must be a URL path such as /health, got {ck.health_path!r}")
    audio_check_problems(ck)
    try:
        re.compile(ck.expect)
    except (re.error, TypeError) as exc:          # a number or a list in `expect` raises TypeError
        raise CanError(f"[check] expect is not a regular expression: {exc}") from exc
    if not isinstance(ck.request_extra, dict):
        raise CanError("[check] request_extra must be a table")
    if "card_profile" in can and can["card_profile"] not in CARD_PROFILES:
        raise CanError(f"[can] card_profile must be one of {CARD_PROFILES}, got {can['card_profile']!r}")
    optional = {k: can[k] for k in ("max_num_seqs", "block_size", "max_model_len", "mesh_fabric", "vllm_version", "card_profile") if k in can}
    if can["kind"] != "vllm" and (any(k in can for k in VLLM_ONLY_KEYS) or "model_dir" in raw):
        raise CanError(f"{', '.join(VLLM_ONLY_KEYS)} and the [model_dir] table are for kind vllm")
    optional.update(_serving(can, raw, base))
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
                extra_code=_extra_code(raw, base, expand_trees), bundle_files=_files(raw, "bundle_file", base), check=ck,
                plugin_edits=_plugin_edits(raw), **optional)


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
        for word in spec.extra_args:           # package-thin puts these after the arguments it generates and before the operator's "$@"
            argv += ["--extra-arg", word]
        if spec.tt_config:
            argv += ["--tt-config-json", json.dumps(spec.tt_config)]
    for flag, value in (("--max-num-seqs", spec.max_num_seqs), ("--block-size", spec.block_size),
                        ("--max-model-len", spec.max_model_len), ("--mesh-fabric", spec.mesh_fabric),
                        ("--trace-region-bytes", spec.trace_region_bytes)):
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


def host_port_args(port: int) -> str:
    """The words a kind-vllm run.sh passes so the server listens on 127.0.0.1 and the spec's port (HOST and PORT override)."""
    return f'--host "${{HOST:-127.0.0.1}}" --port "${{PORT:-{port}}}"'


def fix_run_sh_vllm(text: str, port: int) -> str:
    """package-thin's run.sh for kind vllm passes no --host, so vLLM listens on every interface, and no --port, so it uses 8000. The server has
    no sign-in. The words go just before the operator's "$@", so an argument given at the command line still wins."""
    cmd = [ln for ln in text.splitlines() if ln.startswith("CMD=(")]
    if len(cmd) != 1 or "--host" in cmd[0] or "--port" in cmd[0]:
        raise CanError("run.sh has no single CMD=(...) command in the form this step edits (one CMD line with no --host or --port); "
                       "tt-model's launcher changed, and this step needs to follow it")
    try:
        return splice_extra_args(text, host_port_args(port))
    except PackageError as exc:
        raise CanError(f"the run.sh command is not in the form this step edits: {exc}") from exc


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


SCRIPTS = ("run.sh", "install.sh", "prepare_model_dir.py")      # the files a bundle runs; package-thin leaves them without +x


def make_executable(pkg) -> None:
    """Mode 0755 on the scripts a bundle has (`SCRIPTS`)."""
    for f in SCRIPTS:
        if (Path(pkg) / f).is_file():
            (Path(pkg) / f).chmod(0o755)


def harden(pkg) -> None:
    """The canning step's last edits to a vllm-kind bundle, as one call: the scripts are executable and SHA256SUMS lists every file.
    `stage` calls it for a can and stage 7 (orchard/package.py) calls it for a run's package, so both carry the same hardening.
    Call it again after any later write to the bundle, because the sums list the files as they are now."""
    make_executable(pkg)
    write_sums(Path(pkg))


def hardening_problems(pkg) -> list[str]:
    """What `gate` checks of a vllm-kind bundle that needs no spec: SHA256SUMS lists the files as they are, the scripts are executable,
    every requirement is an exact pin, and a pin for a shipped wheel names that wheel's version. Stage 7's gate (orchard/stages.py)
    runs this on each profile."""
    pkg = Path(pkg)
    problems = sums_problems(pkg)
    for f in SCRIPTS:
        if f in ("run.sh", "install.sh") and not (pkg / f).is_file():
            problems.append(f"{f} is missing")       # a package with no installer or launcher must not pass on fresh sums
        elif (pkg / f).is_file() and not os.access(pkg / f, os.X_OK):
            problems.append(f"{f} is not executable")
    req = pkg / "requirements.txt"
    if not req.is_file():
        return problems + ["requirements.txt is missing"]
    return problems + requirement_problems(req.read_text(encoding="utf-8", errors="replace")) + wheel_pin_problems(pkg)


def wire_model_dir(run_sh: str, spec: Spec, out: Path) -> str:
    """Make vLLM load the weights through `model-dir/`, which prepare_model_dir.py builds at start-up from the pinned Hugging Face snapshot
    plus the spec's base_config files (they replace the snapshot's own, for example a config.json that names the architecture the model class
    registers). This is stage 7's wiring (`wire_weights`) and its prepare script; the weights stay a pointer. Returns the edited run.sh."""
    text = wire_weights(run_sh, spec.repo)
    problems = weights_wiring_problems(text, nearest_model=NO_NEAREST_MODEL)
    if problems:
        raise CanError("run.sh weights wiring: " + "; ".join(problems))
    shutil.copyfile(TEMPLATES / "prepare_model_dir.py", out / "prepare_model_dir.py")
    (out / "base_config").mkdir()
    _copy_in(((src, f"base_config/{dst}") for src, dst in spec.base_config), out)
    return text


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
        origin = ("ttnn and the models library are the wheels in wheels/, installed by path" if spec.models_wheels
                  else "ttnn comes from PyPI (it carries the TT-Metalium runtime)")
        req.write_text(f"# Exact pins. The install builds a venv from these. {origin}.\n" + "\n".join(spec.requirements) + "\n", encoding="utf-8")
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
        else:
            text = fix_run_sh_vllm(run_sh.read_text(encoding="utf-8"), spec.port)
            if spec.base_config:
                text = wire_model_dir(text, spec, out)
            run_sh.write_text(text, encoding="utf-8")
        _copy_in(spec.bundle_files, out)
        harden(out)
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


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_pin_problems(pkg: Path) -> list[str]:
    """A requirements pin for a package whose wheel the bundle ships must name the shipped wheel's version. A different pin would send the
    install to the package index for another build (for ttnn that is the PyPI wheel, which lacks the model's ops)."""
    wheels = {}
    for w in sorted((pkg / "wheels").glob("*.whl")) if (pkg / "wheels").is_dir() else []:
        parts = w.name[:-4].split("-")
        if len(parts) >= 5:
            wheels[_norm(parts[0])] = (parts[1], w.name)
    problems = []
    for line in (pkg / "requirements.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*===?\s*([^\s;#]+)", line)
        if m and _norm(m.group(1)) in wheels:
            version, name = wheels[_norm(m.group(1))]
            if m.group(2).lower() != version.lower():
                problems.append(f"requirements.txt pins {m.group(1)}=={m.group(2)} but wheels/{name} is version {version}")
    return problems


def vllm_problems(pkg: Path, spec: Spec, run_sh: str) -> list[str]:
    """What is wrong with a kind-vllm bundle beyond the checks every bundle gets: the vLLM command in run.sh, the weights wiring, the model
    directory's files, the metadata, the shipped wheels' pins."""
    problems: list[str] = []
    cmd = [ln for ln in run_sh.splitlines() if ln.startswith("CMD=(")]
    if len(cmd) != 1:
        return [f"run.sh must have exactly one CMD=(...) line; it has {len(cmd)}"]
    cmd = cmd[0]
    if "--host \"${HOST:-127.0.0.1}\"" not in cmd or "0.0.0.0" in run_sh:
        problems.append("run.sh must pass --host \"${HOST:-127.0.0.1}\" (the server has no sign-in) and never name 0.0.0.0")
    if f"${{PORT:-{spec.port}}}" not in cmd:
        problems.append(f"run.sh does not default to port {spec.port}")
    passed = [flag_name(w) for w in re.findall(r"(?<=\s)--[\w-]+", cmd)]
    for flag, value in (("--max_num_seqs", spec.max_num_seqs), ("--block_size", spec.block_size), ("--max_model_len", spec.max_model_len)):
        if value is not None and f" {flag} {value}" not in cmd:
            problems.append(f"run.sh does not pass {flag} {value}, the spec's serving limit")
        if passed.count(flag) > 1:         # a later copy (an extra argument) would win over the spec's value
            problems.append(f"run.sh: {flag} is passed {passed.count(flag)} times; the spec's value must be the only one")
    if spec.extra_args and " ".join(spec.extra_args) not in cmd:
        problems.append(f"run.sh does not pass the spec's extra_args as one run: {' '.join(spec.extra_args)}")
    wanted_tt = {**spec.tt_config, **({"fabric_config": spec.mesh_fabric} if spec.mesh_fabric else {}),
                 **({"trace_region_size": spec.trace_region_bytes} if spec.trace_region_bytes else {})}
    if wanted_tt:
        m = re.search(r"--additional-config '([^']*)'", cmd)
        try:
            got = (safeio.parse_json_bytes(m.group(1).encode(), max_bytes=MAX_JSON_BYTES, expect=dict) if m else {}).get("tt", {})
        except safeio.SafeReadError:
            got = {}
        for key, value in wanted_tt.items():
            if got.get(key) != value:
                problems.append(f"run.sh --additional-config has tt.{key} = {got.get(key)!r}; the spec says {value!r}")
    for key, value in spec.env.items():
        if f'export {key}="{value}"' not in run_sh:
            problems.append(f'run.sh does not export {key}="{value}", the spec\'s [env] setting')
    if spec.base_config:
        problems += weights_wiring_problems(run_sh, nearest_model=NO_NEAREST_MODEL)
        prepare = pkg / "prepare_model_dir.py"
        if not prepare.is_file():
            problems.append("prepare_model_dir.py is missing")
        elif not os.access(prepare, os.X_OK):
            problems.append("prepare_model_dir.py is not executable")
        elif prepare.read_bytes() != (TEMPLATES / "prepare_model_dir.py").read_bytes():
            problems.append("prepare_model_dir.py differs from stage 7's template")
        for src, dst in spec.base_config:
            shipped = pkg / "base_config" / dst
            if not shipped.is_file():
                problems.append(f"base_config/{dst} is missing")
            elif shipped.read_bytes() != src.read_bytes():
                problems.append(f"base_config/{dst} differs from the spec's file")
    meta = pkg / "vllm_models" / spec.name / "vllm_metadata.json"
    try:
        same = _json(meta) == _json(spec.metadata)
    except CanError as exc:
        problems.append(str(exc))
    else:
        if not same:
            problems.append("vllm_models/%s/vllm_metadata.json differs from the spec's metadata file" % spec.name)
    return problems + wheel_pin_problems(pkg)


_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _without_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """`line` with its HTML comments removed, and whether a comment is still open at its end (a comment may span lines)."""
    out = []
    while line:
        if in_comment:
            end = line.find("-->")
            if end < 0:
                break
            line, in_comment = line[end + 3:], False
        else:
            start = line.find("<!--")
            if start < 0:
                out.append(line)
                break
            out.append(line[:start])
            line, in_comment = line[start + 4:], True
    return "".join(out), in_comment


def _sections(text: str) -> dict[str, list[str]]:
    """The body lines of each level-2 heading of a card, keyed by the heading text. A fenced code block (the fence's own character,
    closed by a fence of that character at least as long and with nothing after it) holds no headings and adds no lines to a section."""
    sections: dict[str, list[str]] = {}
    current, fence, in_comment = None, None, False
    for line in text.splitlines():
        m = _FENCE.match(line)
        if fence is None:
            line, in_comment = _without_comments(line, in_comment)
            m = _FENCE.match(line)
        if fence is not None:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
                fence = None
            continue
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            fence = m.group(1)
            if current is not None:
                sections[current].append("")      # a fence ends any table around it
            continue
        heading = re.match(r" {0,3}(#{1,2}) +(.*?)\s*$", line)
        if heading and heading.group(1) == "##":
            current = heading.group(2).rstrip("#").strip()
            sections.setdefault(current, [])
            continue
        if heading:
            current = None
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _cells(line: str) -> list[str]:
    """The cells of a table row. The outer pipes are optional in Markdown, so one leading and one trailing pipe are dropped if present."""
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return [c.strip() for c in re.split(r"(?<!\\)\|", body)]


def _is_row(line: str) -> bool:
    """A table row has a pipe and is not indented four or more spaces (that is an indented code block)."""
    return bool(re.search(r"(?<!\\)\|", line)) and not re.match(r" {4}|\t", line)


def _has_measured_row(lines: list[str]) -> bool:
    """True when the lines hold a table (a header row, then a separator row with one dash cell per header cell) and a later body row has
    a cell starting with a number. Markdown drops body cells past the header's width, the first cell is the label, and a digit in the
    header or the label does not count. Rows are the lines that follow the separator without a blank line."""
    i = 1
    while i < len(lines):
        if not (_is_row(lines[i]) and _is_row(lines[i - 1])):
            i += 1
            continue
        sep, head = _cells(lines[i]), _cells(lines[i - 1])
        if len(sep) != len(head) or not all(re.fullmatch(r":?-+:?", c) for c in sep):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and _is_row(lines[j]):
            if any(re.match(r"[~<>]?\d", c) for c in _cells(lines[j])[1:len(head)]):
                return True
            j += 1
        i = j          # a separator row inside this table's rows is a body row, so the scan goes on after them (one pass, never quadratic)
    return False


def card_problems(text: str, profile: str | None) -> list[str]:
    """What the card lacks for `profile` (an empty list is a pass; `None` asks for nothing). The "catalog" profile mirrors the sections
    tt-model's card_publish_gaps() names as required: "Expected performance" must hold a table row that has a number, and "Limitations"
    must hold text. A graduated tt-inference-server entry carries the same two things (measured targets and stated known issues)."""
    if profile is None:
        return []
    sections = _sections(text)
    problems: list[str] = []
    perf = sections.get("Expected performance")
    if perf is None:
        problems.append("the card has no `## Expected performance` section")
    elif not _has_measured_row(perf):
        problems.append("the card's `## Expected performance` section has no row with a number in a table")
    if perf is not None and profile == "catalog-measured":
        body = "\n".join(perf)
        for rx, what in ((_COND_CACHE, "whether the kernel cache was warm or cold"), (_COND_RUNS, "how many runs stand behind the numbers"),
                         (_COND_LOAD, "the load on the host during the measurement (a load average, or that the host was idle)")):
            if not rx.search(body):
                problems.append(f"the card's `## Expected performance` section does not say {what}")
    limits = sections.get("Limitations")
    if limits is None:
        problems.append("the card has no `## Limitations` section")
    elif not any(line.strip() for line in limits):
        problems.append("the card's `## Limitations` section is empty")
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
    problems += plugin_edit_problems(pkg, spec.plugin_edits)
    problems += requirement_problems((pkg / "requirements.txt").read_text(encoding="utf-8", errors="replace"))
    problems += [f"card: {p}" for p in card_problems((pkg / "README.md").read_text(encoding="utf-8", errors="replace"), spec.card_profile)]
    if spec.kind == "vllm":
        problems += vllm_problems(pkg, spec, run_sh)
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


def stream_rate(events, reported: int | None = None) -> float | None:
    """Tokens per second between the first and the last streamed chunk. `events` is a list of (time in seconds, chunks so far), one per chunk with
    text. A chunk may hold more than one token (speculative decoding), so when the server reported its completion_tokens the rate is those
    tokens after the first one, over the span; without a report each chunk counts as one token. The time to the first token is left out on
    purpose: it is prefill, and the card reports it apart. None when there is no span."""
    if len(events) < 2:
        return None
    span = events[-1][0] - events[0][0]
    if span <= 0:
        return None
    tokens = (reported - 1) if isinstance(reported, int) and reported > 1 else events[-1][1] - events[0][1]
    return tokens / span


def _stream_events(url: str, body: dict, *, timeout: float) -> tuple[int, list[tuple[float, int]], int | None]:
    """POST a streaming chat request and read the event stream as it arrives. Returns (status, [(time, token count)] per chunk with text,
    the completion_tokens the server reported at the end, if it did). Each chunk with text counts as one token."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    events: list[tuple[float, int]] = []
    reported = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    chunk = safeio.parse_json_bytes(line[5:].encode(), max_bytes=MAX_JSON_BYTES, expect=dict)
                except safeio.SafeReadError:
                    continue
                if isinstance(chunk.get("usage"), dict):
                    reported = chunk["usage"].get("completion_tokens", reported)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning")):
                        events.append((time.monotonic(), len(events) + 1))
            return r.status, events, reported
    except urllib.error.HTTPError as exc:
        return exc.code, events, reported


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
    if len(raw) != n * channels * width:      # readframes returns a short payload without raising when the file is cut off
        raise CanError(f"the WAV header declares {n} frames ({n * channels * width} bytes) and the file holds {len(raw)} bytes of audio")
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
               port: int | None = None, install: bool = True, job_timeout: float = BOOT_REQUEST_TIMEOUT_S) -> dict:
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
        if spec.check.mode == "audio":
            _audio_checks(base, spec.check, check, evidence, server_log)
            return _finish(work, evidence)
        ck = spec.check
        if ck.mode == "audio-job":
            can_audio.audio_checks(_http, base, ck, evidence, lambda name, ok, detail=None: check(name, ok, detail), job_timeout=job_timeout)
            status, _ = _http("GET", f"{base}{ck.health_path}")
            check("health_after", status == 200)
            log_text = server_log.read_text(encoding="utf-8", errors="replace")
            for needle in ck.ready_contains:
                check(f"log contains {needle}", needle in log_text)
            return _finish(work, evidence)
        status, body = _http("GET", f"{base}/v1/models")
        check("models", status == 200)
        ck = spec.check
        # A vLLM server must list the model it serves, at the context length the can declares (kind tt-dit-server serves no model list).
        probing = spec.kind != "tt-dit-server" and ck.mode == "chat"
        declared = ck.request_extra.get("model") if isinstance(ck.request_extra.get("model"), str) else None
        # A name the can sets only through `--served-model-name` in its serving arguments is the one the request carries, unless request_extra names a model.
        aliases = [] if declared is not None else served_verify.served_names(spec.extra_args)
        listing = None
        if probing:
            listing = served_verify.check_listing(status, body, declared_name=declared, declared_names=aliases, declared_context=spec.max_model_len)
            served_verify.record(listing, check, evidence)
        if ck.mode == "music":
            _music_checks(base, ck, check, evidence)
            status, _ = _http("GET", f"{base}{ck.health_path}")
            check("health_after", status == 200)
            log_text = server_log.read_text(encoding="utf-8", errors="replace")
            for needle in ck.ready_contains:
                check(f"log contains {needle}", needle in log_text)
            return _finish(work, evidence)
        # A vLLM server serves the model under a name of its own and may refuse a request that names none, so the check sends the first
        # name /v1/models lists. request_extra (for a thinking model, enable_thinking false) goes into every chat request.
        extra: dict = {}
        try:
            models = safeio.parse_json_bytes(body, max_bytes=MAX_JSON_BYTES, expect=dict).get("data") or []
            if models and isinstance(models[0], dict) and models[0].get("id"):
                extra["model"] = models[0]["id"]
        except (safeio.SafeReadError, TypeError):
            pass
        if aliases and listing is not None and listing["state"] == "pass":
            extra["model"] = listing["served"]["id"]          # the first declared name the server lists (the default is the first model it lists)
        extra.update(ck.request_extra)
        chat = f"{base}/v1/chat/completions"
        counter_before = served_verify.scrape(base) if probing else None      # the engine's own token counter, read around the first chat request
        status, body = _http("POST", chat, {"messages": [{"role": "user", "content": ck.prompt}], "max_tokens": ck.max_tokens, **extra},
                             timeout=BOOT_REQUEST_TIMEOUT_S)
        answer, timings, usage = "", {}, None
        try:
            answer_obj = safeio.parse_json_bytes(body, max_bytes=MAX_JSON_BYTES, expect=dict)
            answer = answer_obj["choices"][0]["message"]["content"]
            if not isinstance(answer, str):          # a reasoning reply may carry null content; check_reply reads the body and names the cause
                answer = ""
            timings = answer_obj.get("timings") or {}
            usage = answer_obj.get("usage")
        except (safeio.SafeReadError, KeyError, IndexError, TypeError):
            pass
        evidence["answer"], evidence["timings"] = answer[:400], timings
        check("chat", status == 200 and re.search(ck.expect, answer) is not None, f"status {status}; expected /{ck.expect}/")
        if probing:
            counter_after = served_verify.scrape(base)
            tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            served_verify.record(served_verify.check_reply(status, body, asked_model=extra.get("model")), check, evidence)
            served_verify.record(served_verify.check_counter(counter_before, counter_after, completion_tokens=tokens), check, evidence)
            if extra.get("model"):
                for probe in served_verify.probe_completions(base, extra["model"], timeout=BOOT_REQUEST_TIMEOUT_S):
                    served_verify.record(probe, check, evidence)
            try:
                log_for_mesh = safeio.read_prefix(server_log, max_bytes=MAX_LOG_BYTES).decode("utf-8", errors="replace")
            except safeio.SafeReadError:
                log_for_mesh = None
            served_verify.record(served_verify.check_mesh(log_for_mesh, expected_devices=spec.device_count), check, evidence)
        if ck.min_decode_tok_s is not None:
            # a separate request for a long answer, after the first one has warmed the kernels (a short answer has no decode rate)
            rate_body = {"messages": [{"role": "user", "content": ck.rate_prompt}], "max_tokens": ck.rate_max_tokens, **extra}
            code, raw = _http("POST", chat, rate_body, timeout=BOOT_REQUEST_TIMEOUT_S)
            rate_timings = {}
            try:
                rate_timings = safeio.parse_json_bytes(raw, max_bytes=MAX_JSON_BYTES, expect=dict).get("timings") or {}
            except safeio.SafeReadError:
                pass
            rate = rate_timings.get("decode_tok_s")
            if rate is None:
                # The server reports no timings (vLLM does not). Stream a request of the same size and time the tokens between the first and
                # the last one; the time to the first token (prefill) is not in the rate.
                code, events, reported = _stream_events(chat, {**rate_body, "stream": True, "stream_options": {"include_usage": True}},
                                                        timeout=BOOT_REQUEST_TIMEOUT_S)
                rate = stream_rate(events, reported)
                rate_timings = {"source": "stream", "decode_tok_s": rate and round(rate, 2), "completion_tokens": reported or len(events),
                                "streamed_chunks": len(events), "stream_seconds": round(events[-1][0] - events[0][0], 3) if len(events) > 1 else None}
            evidence["rate_timings"] = rate_timings
            check("decode_rate", code == 200 and isinstance(rate, (int, float)) and rate >= ck.min_decode_tok_s,
                  f"{rate} tok/s reported over {rate_timings.get('completion_tokens')} tokens; at least {ck.min_decode_tok_s} wanted")
        if ck.stream:
            status, body = _http("POST", chat, {"messages": [{"role": "user", "content": ck.prompt}], "max_tokens": ck.max_tokens, "stream": True,
                                                **extra}, timeout=BOOT_REQUEST_TIMEOUT_S)
            text = body.decode("utf-8", errors="replace")
            check("stream", status == 200 and text.rstrip().endswith("data: [DONE]"), f"status {status}; the event stream {'ends' if text.rstrip().endswith('data: [DONE]') else 'does not end'} with [DONE]")
        status, _ = _http("GET", f"{base}{ck.health_path}")
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


def audio_check_problems(ck: Check) -> None:
    """Refuse a malformed `[check]` for mode audio (and a mode that is not known). Raises CanError."""
    if ck.mode != "audio":
        return
    if not ck.path or ck.body is None or ck.expect_seconds is None:
        raise CanError("an audio check needs [check] path, body and expect_seconds")
    if not ck.path.startswith("/"):
        raise CanError(f"[check] path must start with '/', got {ck.path!r}")
    if not isinstance(ck.body, dict):
        raise CanError("[check] body must be a table (the JSON the request sends)")
    if isinstance(ck.expect_seconds, bool) or not isinstance(ck.expect_seconds, (int, float)) or ck.expect_seconds <= 0:
        raise CanError("[check] expect_seconds must be a number above 0")
    if isinstance(ck.min_loud_fraction, bool) or not isinstance(ck.min_loud_fraction, (int, float)) or not 0 <= ck.min_loud_fraction <= 1:
        raise CanError("[check] min_loud_fraction must be a number from 0 to 1")
    if ck.max_warm_s is not None and (isinstance(ck.max_warm_s, bool) or not isinstance(ck.max_warm_s, (int, float)) or ck.max_warm_s <= 0):
        raise CanError("[check] max_warm_s must be a number above 0")


LOUD_MEAN_SQUARE = 1e-4        # -40 dBFS is an RMS of 0.01, a mean square of 1e-4 (samples scaled to +-1)


def wav_facts(data: bytes) -> dict:
    """What the boot check needs to know about a WAV clip, with the standard library only: sample rate, channels, seconds, the share of 100 ms
    windows above -40 dBFS, the share of samples at full scale and whether every sample is finite. Reads 32-bit float and 16-bit integer samples.
    Raises CanError for anything else (a file that is not a RIFF WAVE, a missing chunk, another sample format, no samples)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise CanError("the response is not a RIFF WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        tag, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if pos + 8 + size > len(data):
            raise CanError(f"the WAV is cut short: the {tag.decode('latin-1')!r} chunk declares {size} bytes and {len(data) - pos - 8} remain")
        chunk = data[pos + 8:pos + 8 + size]
        if tag == b"fmt " and len(chunk) >= 16:
            fmt = struct.unpack("<HHIIHH", chunk[:16])
        elif tag == b"data":
            pcm = chunk
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise CanError("the WAV has no fmt chunk or no data chunk")
    tag, channels, rate, _, _, bits = fmt
    if channels <= 0 or rate <= 0:
        raise CanError("the WAV has no channels or a zero sample rate")
    if tag == 3 and bits == 32:
        samples, scale = array("f"), 1.0
        samples.frombytes(pcm[:len(pcm) // 4 * 4])
    elif tag == 1 and bits == 16:
        samples, scale = array("h"), 1 / 32768.0
        samples.frombytes(pcm[:len(pcm) // 2 * 2])
    else:
        raise CanError(f"the WAV has format tag {tag} with {bits} bits; only 32-bit float and 16-bit integer samples are read")
    if sys.byteorder == "big":
        samples.byteswap()
    frames = len(samples) // channels
    if frames == 0:
        raise CanError("the WAV has no samples")
    finite = all(math.isfinite(x) for x in samples) if tag == 3 else True
    window = max(1, int(rate * 0.1)) * channels
    loud = windows = 0
    for start in range(0, frames * channels - window + 1, window):
        windows += 1
        if finite and sum(x * x for x in samples[start:start + window]) * scale * scale / window > LOUD_MEAN_SQUARE:
            loud += 1
    full = sum(1 for x in samples if abs(x * scale) >= 0.9999) if finite else 0
    return {"sample_rate": rate, "channels": channels, "seconds": frames / rate, "loud_window_fraction": (loud / windows) if windows else 0.0,
            "full_scale_fraction": full / len(samples), "finite": finite,
            "pcm_sha256": hashlib.sha256(pcm).hexdigest()}


def _audio_checks(base: str, ck: Check, check, evidence: dict, server_log: Path) -> None:
    """The checks of mode audio, after /health answered. The first request compiles kernels and is only timed; the second is the warm one."""
    clips = []
    for n in ("first", "warm"):
        t0 = time.perf_counter()
        status, body = _http("POST", f"{base}{ck.path}", ck.body, timeout=BOOT_REQUEST_TIMEOUT_S)
        evidence[f"{n}_request_s"] = round(time.perf_counter() - t0, 3)
        try:
            facts = wav_facts(body) if status == 200 else None
            fault = None if status == 200 else f"status {status}"
        except CanError as exc:
            facts, fault = None, str(exc)
        if not check("audio_request" if n == "first" else "audio_request_warm", facts is not None, fault or f"200 with a clip in {evidence[f'{n}_request_s']} s"):
            return
        clips.append(facts["pcm_sha256"])
        evidence["audio"] = facts
    facts = evidence["audio"]
    check("audio_format", facts["sample_rate"] == ck.sample_rate and facts["channels"] == ck.channels,
          f"{facts['sample_rate']} Hz and {facts['channels']} channels; {ck.sample_rate} Hz and {ck.channels} wanted")
    tolerance = max(0.05 * ck.expect_seconds, 0.1)
    check("audio_length", abs(facts["seconds"] - ck.expect_seconds) <= tolerance, f"{facts['seconds']:.2f} s; {ck.expect_seconds} s wanted (within {tolerance:.2f})")
    check("audio_finite", facts["finite"])
    check("audio_not_silent", facts["loud_window_fraction"] >= ck.min_loud_fraction,
          f"{facts['loud_window_fraction']:.1%} of 100 ms windows are above -40 dBFS; at least {ck.min_loud_fraction:.0%} wanted")
    check("audio_not_clipped", facts["full_scale_fraction"] <= 0.01, f"{facts['full_scale_fraction']:.2%} of samples at full scale")
    if ck.repeat_identical:
        check("repeat_identical", clips[0] == clips[1], "the same request returned the same samples" if clips[0] == clips[1] else "the same request returned different samples")
    if ck.max_warm_s is not None:
        check("warm_time", evidence["warm_request_s"] <= ck.max_warm_s, f"{evidence['warm_request_s']} s for the warm request; at most {ck.max_warm_s} wanted")
    status, _ = _http("GET", f"{base}{ck.health_path}")
    check("health_after", status == 200)
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    for needle in ck.ready_contains:
        check(f"log contains {needle}", needle in log_text)


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


# ---- the optional recorded state ------------------------------------------------------------------------------------------
# `--state-dir DIR` on gate, boot and stage writes DIR/<name>.can-state.json, which the Cans view of the web page reads (orchard/cans_view.py,
# docs/superpowers/specs/2026-11-02-canning-tab-design.md). The file sits beside the staged bundle, never inside it: a file inside would be
# one more file that SHA256SUMS does not list.

STATE_SUFFIX = ".can-state.json"
STATE_MAX_BYTES = 64 * 1024
STATE_LIST_MAX = 10                 # problems and failed checks kept in a record
STATE_TEXT_MAX = 300
CACHE_STATES = ("empty", "warm")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _file_digest(path, max_bytes: int) -> str | None:
    """The sha256 of a regular file read with a size limit, or None when it cannot be read."""
    try:
        return hashlib.sha256(safeio.read_bytes(path, max_bytes=max_bytes)).hexdigest()
    except (OSError, safeio.SafeReadError):
        return None


def _sums_digest(bundle) -> str | None:
    """The sha256 of the bundle's SHA256SUMS file, or None when it cannot be read (a record then names no bundle)."""
    return _file_digest(Path(bundle) / SUMS_FILE, 1024 * 1024)


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _short(text) -> str:
    return str(text)[:STATE_TEXT_MAX]


def gate_record(problems: list[str]) -> dict:
    """What `can gate` found, as a record: passed, how many problems, the first STATE_LIST_MAX of them, and when. `write_state` adds the digest
    of the bundle's SHA256SUMS, which lets a reader tell that the record is for another version of the bundle."""
    return {"passed": not problems, "problem_count": len(problems), "problems": [_short(p) for p in problems[:STATE_LIST_MAX]],
            "bundle_sums_sha256": None, "spec_sha256": None, "at": _stamp()}


def _finite_rate(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def boot_record(evidence: dict, cache_state: str | None) -> dict:
    """What `can boot` found, as a record. The decode rate is the one the check timed (rate_timings, then timings); a value that is not a
    finite number is dropped. `cache_state` is what the operator says about the kernel cache the boot used: "empty", "warm" or None."""
    if cache_state is not None and cache_state not in CACHE_STATES:
        raise CanError(f"--cache-state must be one of {CACHE_STATES}, got {cache_state!r}")
    rate = _finite_rate((evidence.get("rate_timings") or {}).get("decode_tok_s"))
    if rate is None:
        rate = _finite_rate((evidence.get("timings") or {}).get("decode_tok_s"))
    checks = evidence.get("checks") or {}
    return {"passed": bool(evidence.get("passed")), "decode_tok_s": rate, "ready_s": _finite_rate(evidence.get("ready_s")),
            "cache_state": cache_state, "failed": [_short(n) for n, c in checks.items() if not (c or {}).get("ok")][:STATE_LIST_MAX],
            "bundle_sums_sha256": None, "spec_sha256": None, "at": _stamp()}


def _strict_bool(x, what):
    if not isinstance(x, bool):
        raise CanError(f"state: {what} must be true or false")


def _strings(x, what):
    if not isinstance(x, list) or len(x) > STATE_LIST_MAX or not all(isinstance(i, str) and len(i) <= STATE_TEXT_MAX for i in x):
        raise CanError(f"state: {what} must be a list of at most {STATE_LIST_MAX} strings of at most {STATE_TEXT_MAX} characters")


def _count(x, what):
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise CanError(f"state: {what} must be a whole number, zero or more")


def _rate_or_none(x, what):
    if x is not None and _finite_rate(x) is None:
        raise CanError(f"state: {what} must be a finite number, zero or more, or null")


def _cache_or_none(x, what):
    if x is not None and x not in CACHE_STATES:
        raise CanError(f"state: {what} must be one of {CACHE_STATES} or null")


def _state_part(part, what: str, fields: dict) -> None:
    keys = {"bundle_sums_sha256", "spec_sha256", "at", *fields}
    if not isinstance(part, dict) or set(part) != keys:
        raise CanError(f"state: {what} must be an object with exactly {sorted(keys)}")
    for key, check in fields.items():
        check(part[key], f"{what}.{key}")
    for key in ("bundle_sums_sha256", "spec_sha256"):
        digest = part[key]
        if digest is not None and not (isinstance(digest, str) and _SHA.fullmatch(digest)):
            raise CanError(f"state: {what}.{key} must be 64 hex digits or null")
    if not (isinstance(part["at"], str) and _STAMP.fullmatch(part["at"])):
        raise CanError(f"state: {what}.at must be a UTC time like 2026-11-02T10:00:00Z")


def check_state(raw) -> dict:
    """`raw` (a parsed state file) if every field is as the format says, else CanError naming the field. Used by the writer, to merge with
    an earlier file, and by the Cans view, which shows nothing from a file with a bad field."""
    if not isinstance(raw, dict) or not {"schema", "name"} <= set(raw) or set(raw) - {"schema", "name", "gate", "boot"}:
        raise CanError("state: the file must be an object with schema, name and optionally gate and boot")
    if raw["schema"] != 1 or isinstance(raw["schema"], bool):
        raise CanError("state: schema must be 1")
    if not (isinstance(raw["name"], str) and NAME.fullmatch(raw["name"])):
        raise CanError("state: name must be a plain name")
    if "gate" in raw:
        _state_part(raw["gate"], "gate", {"passed": _strict_bool, "problem_count": _count, "problems": _strings})
    if "boot" in raw:
        _state_part(raw["boot"], "boot", {"passed": _strict_bool, "decode_tok_s": _rate_or_none, "ready_s": _rate_or_none,
                                          "cache_state": _cache_or_none, "failed": _strings})
    return raw


def write_state(state_dir, name: str, bundle, *, gate: dict | None = None, boot: dict | None = None, digest_of=None, spec_path=None, spec_digest=None) -> Path:
    """Write `state_dir/<name>.can-state.json`: `gate` and `boot` replace the same part of an earlier file and the other part is kept. Each
    part written is stamped with the digest of the SHA256SUMS in `digest_of` (default: `bundle`) as it is now. A boot passes the copy it
    booted, so the record is bound to the files that were tested even if the source bundle is rebuilt during the boot. Each part also
    carries the sha256 of the spec file (`spec_path`), because the gate and the boot depend on what the spec says (the weights revision, the
    pins) as well as on the bundle. An earlier file that
    fails `check_state`, or that is the record of another can, is replaced."""
    if not NAME.fullmatch(str(name)):
        raise CanError(f"the state name {name!r} is not a plain name")
    folder = Path(state_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}{STATE_SUFFIX}"
    state = {"schema": 1, "name": name}
    try:
        earlier = check_state(safeio.read_json(path, max_bytes=STATE_MAX_BYTES))
        if earlier["name"] == name:
            state = earlier
    except (OSError, safeio.SafeReadError, CanError):
        pass
    digest = _sums_digest(bundle if digest_of is None else digest_of)
    if spec_digest is None and spec_path is not None:
        spec_digest = _file_digest(spec_path, MAX_SPEC_BYTES)
    for key, part in (("gate", gate), ("boot", boot)):
        if part is not None:
            state[key] = {**part, "bundle_sums_sha256": digest, "spec_sha256": spec_digest}
    check_state(state)
    safeio.atomic_write(path, json.dumps(state, indent=2) + "\n", mode=0o644)
    return path


# ---- command line --------------------------------------------------------------------------------------------------------

def _boot_port(args):
    """The `--port` override of `can boot`, checked by the rule the spec's port follows."""
    port = getattr(args, "port", None)
    return None if port is None else check_port(port, "--port")


def main(args, out=None) -> int:
    out = out or sys.stdout
    say = lambda text: print(text, file=out)  # noqa: E731
    try:
        before = _file_digest(args.spec, MAX_SPEC_BYTES)
        spec = load_spec(args.spec, base=getattr(args, "base", None))
        # The fingerprint is of the spec the command ran with: taken before the long work, and dropped if the file changed while it was loaded.
        spec_digest = before if before == _file_digest(args.spec, MAX_SPEC_BYTES) else None
        if args.action == "stage":
            if not args.out:
                raise CanError("can stage needs --out")
            rec = stage(spec, args.out, namespace=getattr(args, "namespace", None))
            say(f"staged {rec['name']} ({len(rec['files'])} files) in {args.out}")
            if getattr(args, "state_dir", None):     # stage raises when the gate finds a problem, so reaching here means it passed
                say(f"recorded in {write_state(args.state_dir, spec.name, args.out, gate=gate_record([]), spec_digest=spec_digest)}")
            if getattr(args, "run_dir", None):
                say(f"recorded in {record(args.run_dir, rec)}")
            return 0
        if args.action == "gate":
            problems = gate(args.bundle, spec, namespace=getattr(args, "namespace", None))
            for p in problems:
                say(f"problem: {p}")
            say("gate: " + ("passed" if not problems else f"{len(problems)} problem(s)"))
            if getattr(args, "state_dir", None):
                say(f"recorded in {write_state(args.state_dir, spec.name, args.bundle, gate=gate_record(problems), spec_digest=spec_digest)}")
            return 1 if problems else 0
        if args.action == "boot":
            if not args.work:
                raise CanError("can boot needs --work")
            cache_state = getattr(args, "cache_state", None)
            boot_record({"checks": {}}, cache_state)             # refuse a bad --cache-state before the long boot starts
            env = dict(os.environ)
            if getattr(args, "hf_home", None):
                env["HF_HOME"] = str(args.hf_home)
            ev = boot_check(args.bundle, args.work, spec, env=env, health_timeout=args.health_timeout, port=_boot_port(args))
            for name, c in ev["checks"].items():
                say(f"{'ok  ' if c['ok'] else 'FAIL'} {name} {c.get('detail', '')}".rstrip())
            say(f"boot check: {'passed' if ev['passed'] else 'FAILED'} (evidence: {Path(args.work) / 'boot-check.json'})")
            if getattr(args, "state_dir", None):
                booted_copy = Path(args.work) / "bundle"       # the record is bound to the copy that was served, not to the source
                say(f"recorded in {write_state(args.state_dir, spec.name, args.bundle, boot=boot_record(ev, cache_state), digest_of=booted_copy, spec_digest=spec_digest)}")
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
