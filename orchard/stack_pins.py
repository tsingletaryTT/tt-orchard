# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 stack freshness record: `stages/0/stack-pins.json`.

Taken from tenstorrent/skills PR 44 (`tt-model-bringup: pin and verify stack freshness at startup`). That PR found three stale parts that
went unnoticed until an operator looked: the skills plugin, a TT vLLM plugin checkout 65 commits behind, and a `transformers` older than the
checkpoint's `transformers_version`. This module records the same four comparisons for a new run, reusing orchard/skillsfresh.py for every
git and GitHub call (its read-only runner, github.com only, network budget, and the states CURRENT, BEHIND, AHEAD, DIFFERS, UNKNOWN):

- tt_metal: the tt-metal checkout's HEAD against upstream main.
- tt_vllm_plugin: the TT vLLM plugin checkout's HEAD against its upstream branch (main unless the operator names another).
- skills: each tenstorrent/skills clone against upstream main.
- transformers: the version installed in the reference interpreter against the checkpoint's `transformers_version` (and the table
  KNOWN_MINIMUMS), plus whether the installed `transformers` has each class the checkpoint's `architectures` names. The interpreter is asked
  with a fixed script that uses importlib and imports `transformers` only; it never imports ttnn or ttml and never opens a device.

Severity. For a new run these are critical and each writes a hazard with a named reason: a checkout more than MAX_BEHIND_DAYS days behind, a
`transformers` below the checkpoint's minimum, a `transformers` that cannot be imported, an architecture class that is missing. With
`new_run=False` (a resume, or a second attempt at stage 0) the same findings are advisories. An upstream that cannot be reached, a part
nobody named, a DIFFERS result and a BEHIND result whose age could not be told are advisories in both modes. A comparison that was not made is
never a pass: its part is UNKNOWN and it appears in `advisories`. Severity is `critical`, `advisory` or `ok`; `ok` means every part was
compared and none needs a note.

The record is written by the supervisor (never by an agent) with safeio.atomic_write. The supervisor records its sha256 in the ledger, and the
stage 0 gate (`gate_reasons`) reads the file through safeio and accepts it only when the hash is the ledger's. Home directories in any text are
shown as `~`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from orchard import model_config, safeio, skillsfresh

FILE = "stack-pins.json"
SCHEMA = 1
MAX_BEHIND_DAYS = 7.0           # choice: the threshold in tenstorrent/skills PR 44 (a checkout older than a week is stale)
MAX_RECORD_BYTES = 1 << 20
PROBE_TIMEOUT_S = 120.0         # choice: importing transformers with torch takes a few seconds; a hung import ends here
PROBE_MAX_BYTES = 64 * 1024
MAX_CLASSES = 64                # choice: no real checkpoint names this many architectures; more is an unchecked finding, never ok
CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,100}")        # used with fullmatch: `$` would also accept a final newline
MODEL_TYPE = re.compile(r"[A-Za-z0-9_]{1,100}")
SEVERITIES = ("critical", "advisory", "ok")
OFFLINE_ENV = "ORCHARD_STACK_PINS_OFFLINE"
BLOCK_ENV = "ORCHARD_STACK_PINS_BLOCK"      # "1": a critical finding of a new run blocks stage 0. Without it the findings are recorded as would_block advisories.
METAL_REPO = "tenstorrent/tt-metal"
VLLM_REPO = "tenstorrent/vllm-tt-plugin"
SKILLS_REPO = "tenstorrent/skills"

# What the probe cannot match in the reference step, listed in every record (`probe_limits`). The probe is deliberately not the reference step:
# it must not load files an agent can write.
PROBE_LIMITS = (
    "The probe sets PYTHONNOUSERSITE=1 and starts in an empty temporary directory, so a reference step that imports transformers from a user-site install can differ.",
    "PYTHONPATH entries that are relative, the current directory, or under the run directory are dropped, and LD_LIBRARY_PATH and other import settings are not passed on.",
    "The interpreter is the run's reference_python input or ORCHARD_REFERENCE_PYTHON; stage 1 and the text-runtime references can select another one.",
    "On a resumed stage 0 the sources come from the inputs of the current command; the agent receives the inputs the run started with.",
    "The class lookup uses the model_type and architectures of the checkpoint's config.json only.",
)
# Checkpoints whose config.json does not state the minimum transformers they need. model_type -> (minimum, source). The config's own
# `transformers_version`, when it has one, is compared as well and the larger minimum wins.
KNOWN_MINIMUMS = {
    "qwen4_exp": ("5.19.0", "operator, 2026-11-02: Qwen4ExpForCausalLM first appears in transformers 5.19.0"),
    # The flat text checkpoints this repository's tiny-checkpoint script writes name the same class and carry no transformers_version.
    "qwen4_exp_text": ("5.19.0", "operator, 2026-11-02: the flat text form of qwen4_exp names Qwen4ExpForCausalLM"),
}

# Hazard names (stable: the ledger and the bundle quote them).
H_TT_METAL = "tt_metal_checkout_stale"
H_VLLM = "tt_vllm_plugin_stale"
H_SKILLS = "skills_clone_stale"
H_BELOW = "transformers_below_checkpoint_minimum"
H_NOT_IMPORTABLE = "transformers_not_importable"
H_CLASS = "transformers_missing_architecture_class"

# Run by the reference interpreter. Imports transformers and nothing else of ours. The first argument is the checkpoint's model_type (or `-`),
# the rest are class names, each already checked against a plain pattern. The current directory is taken off sys.path so a file there cannot
# shadow a module. Prints one JSON object on its last line.
PROBE_SCRIPT = r'''
import importlib, importlib.util, json, os, sys
_here = os.path.realpath(os.getcwd())
sys.path[:] = [p for p in sys.path if p not in ("", ".") and os.path.realpath(p) != _here]
out = {"importable": False, "version": None, "classes": {}}
try:
    if importlib.util.find_spec("transformers") is not None:
        import transformers
        out["version"] = str(transformers.__version__)
        out["importable"] = True
        model_type = sys.argv[1]
        for name in sys.argv[2:]:
            found = None
            try:
                found = getattr(transformers, name, None)
            except Exception:
                found = None
            if not isinstance(found, type):            # a string such as __version__ is an attribute, not a model class
                found = None
            if found is None and model_type != "-":
                # The stage 1 reference loader's second lookup (reference_gate.resolve_class): many classes are not exported at the top level.
                try:
                    found = getattr(importlib.import_module("transformers.models.%s.modeling_%s" % (model_type, model_type)), name, None)
                except Exception:
                    found = None
                if not isinstance(found, type):
                    found = None
            out["classes"][name] = found is not None
except Exception as exc:
    out["importable"] = False
    out["error"] = type(exc).__name__
print(json.dumps(out))
'''


class StackPinsError(Exception):
    """An input this module cannot use. Always one sentence with no file content."""


@dataclass(frozen=True)
class Sources:
    """What to compare. A None or empty source gives an UNKNOWN part that says it was not named."""
    model_dir: str | None = None
    ref_python: str | None = None
    tt_metal_dir: str | None = None
    vllm_dir: str | None = None
    vllm_branch: str = "main"
    skills_dirs: tuple = ()
    metal_repo: str = METAL_REPO    # the repositories each origin must be (case-insensitive); another origin gives an unknown part
    vllm_repo: str = VLLM_REPO
    exclude_paths: tuple = ()       # directories whose PYTHONPATH entries the probe drops (the run directory: an agent can write there)
    homes: tuple = ()               # home directories to show as <home> (the run's operator home); the invoking user's is always `~`


# ---- small helpers --------------------------------------------------------------------------------------------------

_OTHER_HOME = re.compile(r"(?:/home|/Users)/[^/\s'\"`]+|/root(?=/|\b)")


def _tilde(value, homes: tuple = ()):
    """`value` with the invoking user's home shown as `~`, every other home directory (and each of `homes`, such as the run's operator home)
    as `<home>`, URL user information and token-shaped words removed, through lists and dicts."""
    try:
        home = str(Path.home())
    except (RuntimeError, KeyError):
        home = ""
    if isinstance(value, str):
        value = skillsfresh.scrub_secrets(value)
        value = value.replace(home, "~") if len(home) > 1 else value
        for other in sorted((h for h in homes if isinstance(h, str) and len(h.rstrip("/")) > 1), key=len, reverse=True):
            value = value.replace(other.rstrip("/"), "<home>")
        return _OTHER_HOME.sub("<home>", value)             # any other user's home, without its name
    if isinstance(value, list):
        return [_tilde(v, homes) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = _tilde(k, homes)                  # a key can be a checkpoint-supplied name (parts.transformers.classes)
            n = 2
            while key in out and isinstance(key, str):         # two names that redact alike keep separate results
                key, n = f"{_tilde(k, homes)}#{n}", n + 1
            out[key] = _tilde(v, homes)
        return out
    return value


def redact(text, homes: tuple = ()):
    """The redaction every record string goes through, for text that reaches a ledger entry or a block reason without being in a record."""
    return _tilde(text, homes)


_VERSION = re.compile(r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-_.]?(dev|a|alpha|b|beta|c|rc|pre|preview|post)[-_.]?(\d*))?\s*$", re.IGNORECASE)
# PEP 440 order of the suffix kinds: X.devN < X.aN < X.bN < X.rcN < X < X.postN.
_SUFFIX_RANK = {"dev": 0, "a": 1, "alpha": 1, "b": 2, "beta": 2, "c": 3, "rc": 3, "pre": 3, "preview": 3, "post": 5}
_FINAL_RANK = 4


def parse_version(text) -> tuple | None:
    """A sort key (four numbers, suffix rank, suffix number) for a version such as `5.19.0`, `4.51.0.dev0`, `5.19.0rc1` or `5.19.0.post1`,
    else None. The order is PEP 440's: dev, a, b, rc, the release, post. A form this does not know (`5.19.0-custom`, `5.19.0.dev1.post2`)
    gives None, and the caller reports the comparison as not made. Local parts (`+cu121`) are dropped."""
    if not isinstance(text, str) or len(text) > 64:
        return None
    m = _VERSION.match(text.split("+", 1)[0])
    if not m:
        return None
    nums = (tuple(int(g) for g in m.groups()[:4] if g is not None) + (0, 0, 0, 0))[:4]
    kind = m.group(5)
    if kind is None:
        return nums, _FINAL_RANK, 0
    return nums, _SUFFIX_RANK[kind.lower()], int(m.group(6) or 0)


def _below(installed: tuple, required: tuple) -> bool:
    return installed < required


def _entry(name: str, reason: str) -> dict:
    return {"name": name, "reason": reason[:800]}


# ---- the checkpoint ---------------------------------------------------------------------------------------------------

def config_facts(model_dir) -> dict:
    """{"model_type", "architectures", "transformers_version"} from `model_dir/config.json`; each is None or [] when the config does not
    hold a usable one. Raises StackPinsError when the config cannot be read (missing, link out of reach, oversize, or text that is not strict JSON)."""
    if not isinstance(model_dir, str) or not model_dir or "\0" in model_dir:
        raise StackPinsError("no checkpoint directory is named")
    try:
        path = os.path.realpath(os.path.join(model_dir, "config.json"))     # a cache snapshot's config.json is a link into blobs/
        cfg = model_config.read_model_config(path)
    except (ValueError, OSError, model_config.ModelConfigError) as exc:
        raise StackPinsError(f"the checkpoint's config.json cannot be read ({type(exc).__name__})") from None

    def text(key, limit):
        try:
            return safeio.bounded_str(cfg.get(key), max_len=limit, min_len=1, what=key)
        except safeio.BadValue:
            return None
    archs = cfg.get("architectures")
    valid = [a for a in archs if isinstance(a, str) and CLASS_NAME.fullmatch(a)] if isinstance(archs, list) else []
    valid = list(dict.fromkeys(valid))              # a name listed twice is checked once (the probe answers one value per name)
    problem = None
    if archs is not None and not isinstance(archs, list):
        problem = "architectures is not a list"
    elif isinstance(archs, list) and any(not (isinstance(a, str) and CLASS_NAME.fullmatch(a)) for a in archs):
        bad = sum(1 for a in archs if not (isinstance(a, str) and CLASS_NAME.fullmatch(a)))
        problem = f"{bad} of the {len(archs)} architecture entries are not plain class names"
    if len(valid) > MAX_CLASSES:
        problem = (problem + "; " if problem else "") + f"{len(valid)} architectures are named and only the first {MAX_CLASSES} can be checked"
    return {"model_type": text("model_type", 100), "architectures": valid[:MAX_CLASSES], "architectures_problem": problem,
            "transformers_version": text("transformers_version", 64)}


def required_minimum(facts: dict) -> tuple[str | None, str | None]:
    """(minimum version text, where it came from): the larger of the config's `transformers_version` and the KNOWN_MINIMUMS entry."""
    found = []
    tv = facts.get("transformers_version")
    if parse_version(tv) is not None:
        found.append((tv, "the checkpoint's config.json transformers_version"))
    known = KNOWN_MINIMUMS.get(facts.get("model_type") or "")
    if known:
        found.append((known[0], f"KNOWN_MINIMUMS for model_type {facts['model_type']} ({known[1]})"))
    if not found:
        return None, None
    return max(found, key=lambda f: parse_version(f[0]))


# ---- the reference interpreter ----------------------------------------------------------------------------------------

def _probe_env(base: dict | None = None, exclude: tuple = ()) -> dict:
    """The environment of the probe. `base` is the environment the reference step runs in (the supervisor's agent_env: HOME inside the run
    directory, the allow-listed variables and the run's --env values); without one the supervisor's own environment is used. The reference
    step runs without isolated mode, so PYTHONPATH and the user site count for it, and the probe sees the same. A relative PYTHONPATH entry
    (`.`, an empty one) and the current directory are dropped: they would put a file of the invoking directory ahead of transformers."""
    src = os.environ if base is None else base
    env = {"PATH": src.get("PATH") or "/usr/bin:/bin", "LC_ALL": "C", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false",
           "PYTHONNOUSERSITE": "1"}     # the run's HOME is writable by an agent: no .pth file or module of its user site may start with the probe
    for name in ("HOME", "PYTHONPATH", "VIRTUAL_ENV"):
        if src.get(name):
            env[name] = src[name]
    if env.get("PYTHONPATH"):
        try:
            here = os.path.realpath(os.getcwd())
        except OSError:
            here = None
        def fine(p):
            if not p or not os.path.isabs(p) or os.path.realpath(p) == here:
                return False
            real = os.path.realpath(p)
            return not any(real == r or real.startswith(r.rstrip("/") + "/") for r in (os.path.realpath(e) for e in exclude if e))
        keep = [p for p in env["PYTHONPATH"].split(os.pathsep) if fine(p)]
        if keep:
            env["PYTHONPATH"] = os.pathsep.join(keep)
        else:
            del env["PYTHONPATH"]
    return env


def real_probe_run(argv, timeout, env=None):
    """The process runner for the interpreter probe. skillsfresh's runner (its own session, the group killed on a timeout) is looked up at
    each call, so the test suite's guard covers this too."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="orchard-probe-") as empty:         # the child starts where no file can shadow a module
        return skillsfresh._raw_run(argv, timeout, env, cwd=empty, max_bytes=PROBE_MAX_BYTES)


def probe_interpreter(python, classes, run: Callable | None = None, model_type=None, env: dict | None = None, exclude: tuple = ()) -> dict:
    """Ask `python` for its transformers version and for each class in `classes`. {"state": "answered"|"unknown", "importable", "version",
    "classes": {name: bool}, "why"}. Never raises. `run` is (argv, timeout, env) -> (returncode, stdout, stderr)."""
    unknown = lambda why: {"state": "unknown", "importable": None, "version": None, "classes": {}, "why": why}     # noqa: E731
    if not isinstance(python, str) or not python or "\0" in python:
        return unknown("no reference interpreter is named (set ORCHARD_REFERENCE_PYTHON)")
    if not os.path.isabs(python):
        return unknown("the reference interpreter is a relative path, and the probe starts in another directory; name it with an absolute path")
    try:
        if not (os.path.isfile(python) and os.access(python, os.X_OK)):
            return unknown("the reference interpreter is not an executable file")
    except (ValueError, OSError):
        return unknown("the reference interpreter path cannot be checked")
    classes = list(dict.fromkeys(c for c in classes if isinstance(c, str) and CLASS_NAME.fullmatch(c)))[:MAX_CLASSES]
    mt = model_type if isinstance(model_type, str) and MODEL_TYPE.fullmatch(model_type) else "-"
    try:
        rc, out, err = (run or real_probe_run)([python, "-c", PROBE_SCRIPT, mt, *classes], PROBE_TIMEOUT_S, _probe_env(env, exclude))
    except Exception as exc:                        # a broken runner is a failed probe
        return unknown(f"the interpreter could not be run ({type(exc).__name__})")
    if rc != 0:
        return unknown(f"the interpreter exited with status {rc}")
    lines = [ln for ln in str(out).splitlines() if ln.strip()]
    try:
        data = safeio.parse_json_bytes(lines[-1] if lines else "", max_bytes=PROBE_MAX_BYTES, expect=dict)
        importable = data.get("importable")
        version = data.get("version")
        got = data.get("classes")
        if not isinstance(importable, bool) or not isinstance(got, dict):
            raise safeio.BadValue("wrong shape")
        if importable:
            version = safeio.bounded_str(version, max_len=64, min_len=1, what="version")
        have = {c: got[c] for c in classes if isinstance(got.get(c), bool)}
        if importable and len(have) != len(classes):
            raise safeio.BadValue("a class answer is missing")
    except safeio.SafeReadError:
        return unknown("the interpreter's answer was not usable")
    return {"state": "answered", "importable": importable, "version": version if importable else None, "classes": have, "why": None}


# ---- one part each ----------------------------------------------------------------------------------------------------

def _fresh_part(f: skillsfresh.Freshness, hazard: str, new_run: bool) -> tuple[dict, list, list]:
    """(part, critical entries, advisory entries) for one checkout result."""
    part = {"state": f.state, "path": f.path, "local": f.local, "remote": f.remote, "behind": f.behind, "ahead": f.ahead,
            "behind_days": round(f.behind_days, 2) if f.behind_days is not None else None, "branch": f.branch, "dirty": f.dirty,
            "detail": f.detail, "warnings": list(f.warnings), "update": f.update if f.state in (skillsfresh.BEHIND, skillsfresh.DIFFERS) else None}
    crit, adv = [], []
    if f.state == skillsfresh.BEHIND:
        if f.behind_days is None:
            adv.append(_entry(f"{hazard}_age_unknown", f"{f.source} is {f.behind} commits behind upstream and the age of the gap could not be told. {f.detail}"))
        elif f.behind_days > MAX_BEHIND_DAYS:
            crit.append(_entry(hazard, f"{f.source} is {f.behind} commits and {round(f.behind_days, 2):g} days behind upstream, more than "
                                       f"{MAX_BEHIND_DAYS:g} days. {f.update_text()}"))
    elif f.state == skillsfresh.DIFFERS:
        adv.append(_entry(f"{hazard}_differs", f"{f.source} is not upstream and how far apart they are could not be counted. {f.detail}"))
    elif f.state == skillsfresh.UNKNOWN:
        adv.append(_entry(f"{hazard}_unchecked", f"{f.source} was not compared with upstream: {f.detail}"))
    return part, crit, adv


_HOME_PREFIX = re.compile(r"^(?:/home/[^/]+|/Users/[^/]+|/root)(?=/|$)")


def _path_word(path, homes: tuple = ()) -> tuple[str, str]:
    """(a shell word for `path`, a note). The home directory is replaced before quoting, so the word stays valid shell: the invoking user's home
    is `~` (unquoted, so the shell expands it) and any other home is `<home>`, which the note tells the operator to replace."""
    path = str(path)
    try:
        mine = str(Path.home())
    except (RuntimeError, KeyError):
        mine = ""
    for prefix, token in sorted([(mine, "~")] + [(h, "<home>") for h in homes if isinstance(h, str)], key=lambda t: -len(t[0])):
        prefix = prefix.rstrip("/")
        if len(prefix) > 1 and (path == prefix or path.startswith(prefix + "/")):
            break
    else:
        m = _HOME_PREFIX.match(path)
        if not m:
            return shlex.quote(path), ""
        prefix, token = m.group(0), "<home>"
    rest = path[len(prefix):].lstrip("/")
    word = token + (("/" + shlex.quote(rest)) if rest else "")
    return word, ("" if token == "~" else " (replace <home> with that user's home directory)")


def _branch_update(upstream: str, homes: tuple = ()) -> Callable:
    """The update hint for a checkout compared with `upstream`: a fast-forward when that branch is checked out, else a request to switch first."""
    def hint(root, branch, remote=None):
        word, note = _path_word(root, homes)
        if branch == upstream:
            return f"git -C {word} pull --ff-only origin {upstream}{note}"
        return (f"the checkout is on {branch or 'no branch'}, and it is compared with upstream {upstream}. Commit your work, then `git -C {word} switch "
                f"{upstream}` and update it with `git -C {word} pull --ff-only origin {upstream}`{note}")
    return hint


def _repo_part(f: skillsfresh.Freshness, hazard: str, expected: str, new_run: bool) -> tuple[dict, list, list]:
    """A checkout is compared only when its origin is the repository it should be (`expected`, case-insensitive: GitHub names are). A fork or
    another repository would be reported current or stale against something that is not the project the run uses, so its part is UNKNOWN."""
    if f.state != skillsfresh.UNKNOWN and (f.repo or "").lower() != expected.lower():
        part = {"state": skillsfresh.UNKNOWN, "path": f.path, "repo": f.repo,
                "detail": f"the checkout's origin is {f.repo or 'not a GitHub repository'}, not {expected}, so it was not compared with that upstream"}
        return part, [], [_entry(hazard + "_unchecked", f"{f.source} was not compared with upstream: {part['detail']}")]
    return _fresh_part(f, hazard, new_run)


def _unnamed(source: str, hazard: str, why: str) -> tuple[dict, list, list]:
    part = {"state": skillsfresh.UNKNOWN, "path": None, "detail": why}
    return part, [], [_entry(f"{hazard}_unchecked", f"{source} was not compared with upstream: {why}")]


def _transformers_part(sources: Sources, run, probe_env=None) -> tuple[dict, list, list]:
    crit, adv = [], []
    part = {"state": skillsfresh.UNKNOWN, "installed": None, "checkpoint_minimum": None, "minimum_source": None, "model_type": None,
            "architectures": [], "classes": {}, "detail": ""}
    try:
        facts = config_facts(sources.model_dir)
    except StackPinsError as exc:
        part["detail"] = str(exc)
        return part, crit, [_entry("transformers_unchecked", f"transformers was not compared with the checkpoint: {exc}")]
    minimum, source = required_minimum(facts)
    if facts["transformers_version"] is not None and parse_version(facts["transformers_version"]) is None:
        adv.append(_entry("transformers_minimum_unreadable", f"the checkpoint's transformers_version {facts['transformers_version']!r} is not a version form "
                                                             "this check reads, so it was not used as a minimum"))
    part.update(model_type=facts["model_type"], architectures=facts["architectures"], checkpoint_minimum=minimum, minimum_source=source)
    probe = probe_interpreter(sources.ref_python, facts["architectures"], run, facts["model_type"], probe_env, sources.exclude_paths)
    if probe["state"] != "answered":
        part["detail"] = probe["why"]
        return part, crit, adv + [_entry("transformers_unchecked", f"transformers was not compared with the checkpoint: {probe['why']}")]
    if not probe["importable"]:
        part.update(state=skillsfresh.BEHIND, detail="transformers cannot be imported in the reference interpreter")
        return part, [_entry(H_NOT_IMPORTABLE, "transformers cannot be imported in the reference interpreter, so no class of the checkpoint can load")], adv
    installed = probe["version"]
    part.update(installed=installed, classes=probe["classes"])
    have, need = parse_version(installed), parse_version(minimum) if minimum else None
    state = skillsfresh.CURRENT
    if have is None:
        # The version cannot be compared, and the class check below does not depend on it, so it still runs.
        state = skillsfresh.UNKNOWN
        adv.append(_entry("transformers_unchecked", f"the installed version {installed!r} is not a version number this check reads, so it was not compared with the minimum"))
    elif need is not None and _below(have, need):
        state = skillsfresh.BEHIND
        crit.append(_entry(H_BELOW, f"transformers {installed} is installed in the reference interpreter and the checkpoint needs {minimum} or later "
                                    f"({source})"))
    missing = [c for c, ok in probe["classes"].items() if not ok]
    if missing:
        state = skillsfresh.BEHIND
        crit.append(_entry(H_CLASS, f"transformers {installed} has no class {', '.join(missing)} that the checkpoint's architectures name"))
    if facts["architectures_problem"]:
        state = skillsfresh.UNKNOWN if state == skillsfresh.CURRENT else state
        adv.append(_entry("transformers_architectures_unchecked", f"the checkpoint's architectures could not all be checked: {facts['architectures_problem']}"))
    if not facts["architectures"]:
        adv.append(_entry("transformers_architecture_unchecked", "the checkpoint's config.json names no architecture class, so the class check was not made"))
    if minimum is None and not facts["architectures"]:
        state = skillsfresh.UNKNOWN
        adv.append(_entry("transformers_unchecked", "the checkpoint names no minimum transformers version and no class to check"))
    part["state"] = state
    part["detail"] = (f"installed {installed}; the checkpoint needs {minimum or 'no stated minimum'}"
                      + (f"; missing classes: {', '.join(missing)}" if missing else ""))
    return part, crit, adv


# ---- the record -------------------------------------------------------------------------------------------------------

def build_record(sources: Sources, *, new_run: bool, run=None, probe_run=None, cache: skillsfresh.Cache | None = None,
                 offline: bool = False, now: float | None = None, probe_env: dict | None = None, block: bool = True) -> dict:
    """Make every comparison and return the record. Never raises for an input it cannot use: that part is UNKNOWN with a reason.
    `run` is the git and gh runner (a ReadOnlyRunner, or a function for one), `probe_run` runs the reference interpreter. No test passes
    the real ones."""
    runner = run if isinstance(run, skillsfresh.ReadOnlyRunner) else skillsfresh.ReadOnlyRunner(run or skillsfresh._raw_run, offline=offline)
    cache = cache if cache is not None else skillsfresh.Cache(write=False)
    parts: dict = {}
    crit: list = []
    adv: list = []

    def take(key, triple):
        parts[key] = triple[0]
        crit.extend(triple[1])
        adv.extend(triple[2])

    def clone(key, directory, hazard, source, update, branch="main", expected=""):
        if not directory:
            return take(key, _unnamed(source, hazard, "no checkout is named for it"))
        try:
            if not Path(directory).is_dir():
                return take(key, _unnamed(source, hazard, "the named directory does not exist"))
        except (ValueError, OSError):
            return take(key, _unnamed(source, hazard, "the named directory cannot be checked"))
        f = skillsfresh.check_clone(directory, source=source, runner=runner, cache=cache, update=update, upstream_branch=branch, want_age=True,
                                    expected_repo=expected or None)
        take(key, _repo_part(f, hazard, expected, new_run))

    clone("tt_metal", sources.tt_metal_dir, H_TT_METAL, "tt-metal", _branch_update("main", sources.homes), "main", sources.metal_repo)
    clone("tt_vllm_plugin", sources.vllm_dir, H_VLLM, "TT vLLM plugin", _branch_update(sources.vllm_branch, sources.homes), sources.vllm_branch,
          sources.vllm_repo)
    dirs = [d for d in sources.skills_dirs if d]
    if not dirs:
        take("skills", _unnamed("tenstorrent/skills", H_SKILLS, "no clone is named for it"))
    else:
        results = skillsfresh.check_skills_dirs(dirs, run=runner, cache=cache, want_age=True, update=_branch_update("main", sources.homes),
                                               expected_repo=SKILLS_REPO)
        subs = [_repo_part(f, H_SKILLS, SKILLS_REPO, new_run) for f in results]
        parts["skills"] = [s[0] for s in subs]
        for s in subs:
            crit.extend(s[1])
            adv.extend(s[2])
    take("transformers", _transformers_part(sources, probe_run, probe_env))

    if not new_run:                                 # a resume holds the same findings as notes
        adv = [dict(e, would_be_critical=True) for e in crit] + adv
        crit = []
    elif not block:                                 # blocking is opt-in (ORCHARD_STACK_PINS_BLOCK=1): without it the findings are recorded, not enforced
        adv = [dict(e, would_block=True) for e in crit] + adv
        crit = []
    severity = "critical" if crit else ("advisory" if adv else "ok")
    stamp = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _tilde({"schema": SCHEMA, "label": "comparison with upstream at the time of the check", "checked_at": stamp,
                   "mode": "new" if new_run else "resume", "severity": severity, "max_behind_days": MAX_BEHIND_DAYS,
                   "blocking_requested": bool(block), "probe_limits": list(PROBE_LIMITS),
                   "hazards": crit, "advisories": adv, "parts": parts}, sources.homes)


def unchecked_record(why: str, *, new_run: bool, now: float | None = None) -> dict:
    """The record when no check was supplied: every part UNKNOWN, never a pass."""
    parts = {"tt_metal": {"state": skillsfresh.UNKNOWN, "detail": why}, "tt_vllm_plugin": {"state": skillsfresh.UNKNOWN, "detail": why},
             "skills": [{"state": skillsfresh.UNKNOWN, "detail": why}], "transformers": {"state": skillsfresh.UNKNOWN, "detail": why}}
    stamp = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"schema": SCHEMA, "label": "no comparison was made", "checked_at": stamp, "mode": "new" if new_run else "resume",
            "severity": "advisory", "max_behind_days": MAX_BEHIND_DAYS, "blocking_requested": False,
            "probe_limits": list(PROBE_LIMITS), "hazards": [],
            "advisories": [_entry("stack_unchecked", f"the stack was not compared with upstream: {why}")], "parts": parts}


def write_record(stage_dir, rec: dict) -> str:
    """Write the record (all or nothing) and return the sha256 of the bytes written. The hash comes from the bytes in hand, so a file
    swapped afterwards is not what the ledger recorded."""
    data = (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode("utf-8")
    safeio.atomic_write(Path(stage_dir) / FILE, data)
    return hashlib.sha256(data).hexdigest()


def make_check(sources: Sources, *, offline: bool | None = None, run=None, probe_run=None, probe_env: dict | None = None,
               block: bool | None = None) -> Callable:
    """The callable the supervisor holds: (stage_dir, new_run) -> (record, sha256). `offline` None reads $ORCHARD_STACK_PINS_OFFLINE."""
    def check(stage_dir, new_run):
        off = bool(os.environ.get(OFFLINE_ENV)) if offline is None else offline
        want_block = os.environ.get(BLOCK_ENV) == "1" if block is None else block
        rec = build_record(sources, new_run=new_run, run=run, probe_run=probe_run, offline=off, probe_env=probe_env, block=want_block)
        return rec, write_record(stage_dir, rec)
    return check


# ---- the gate ---------------------------------------------------------------------------------------------------------

def read_record(stage_dir) -> tuple[dict, str]:
    """(record, sha256 of the bytes read) from the stage directory, through safeio. Raises SafeReadError for anything else than a
    well-formed record."""
    data = safeio.read_bytes(Path(stage_dir) / FILE, max_bytes=MAX_RECORD_BYTES)
    rec = safeio.parse_json_bytes(data, max_bytes=MAX_RECORD_BYTES, expect=dict)
    if rec.get("schema") != SCHEMA or rec.get("severity") not in SEVERITIES or rec.get("mode") not in ("new", "resume"):
        raise safeio.WrongType("the record has no usable schema, severity or mode")
    for key in ("hazards", "advisories"):
        items = rec.get(key)
        if not isinstance(items, list) or not all(isinstance(i, dict) and isinstance(i.get("name"), str) and isinstance(i.get("reason"), str)
                                                  for i in items):
            raise safeio.WrongType(f"the record's {key} are not a list of named reasons")
    if not isinstance(rec.get("parts"), dict):
        raise safeio.WrongType("the record has no parts")
    return rec, hashlib.sha256(data).hexdigest()


def gate_reasons(stage_dir, recorded_sha) -> list[str]:
    """Why stage 0's stack record does not pass: empty when it is the file the ledger recorded, well formed, and holds no critical hazard.
    A missing file, a file that is not the recorded one and a record with no ledger entry all fail; none skips the check."""
    if not isinstance(recorded_sha, str) or not recorded_sha:
        return [f"{FILE}: the ledger holds no entry for this record"]
    try:
        rec, sha = read_record(stage_dir)
    except safeio.SafeReadError as exc:
        return [f"{FILE}: cannot be used ({exc.reason if hasattr(exc, 'reason') else type(exc).__name__})"]
    if sha != recorded_sha:
        return [f"{FILE}: is not the file the supervisor recorded (its hash changed)"]
    if rec["severity"] == "critical" or rec["hazards"]:
        return [f"{FILE}: critical, {h['name']}: {h['reason']}"[:500] for h in rec["hazards"]] or [f"{FILE}: critical"]
    return []
