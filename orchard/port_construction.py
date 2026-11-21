# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The effective construction of the reference model, for the stage 2 coder (issue 541).

The reference excerpts (orchard/port_excerpts.py) give each class's constructor signature and `forward`, with the constructor body
elided. A setting that the constructor resolves from the config is therefore not visible. In run 18 of the tiny Qwen4 rehearsal the gated
norm's class default was `activation="silu"`, the model's constructor passed `config.output_gate_type`, which was `sigmoid`, and the coder's
port used silu. It scored 0.5625 for five repair rounds.

This module runs `orchard/skills/full-port-templates/effective_construction.py` under the reference interpreter (CPU, `meta` device, no
weights read, no ttnn, no chip) and turns its output into two things:

  * `stages/2/effective_construction.txt`: for every module class the reference instantiates, the settings as constructed, one line per
    group of modules that share them, then the list "resolved vs class default";
  * a short summary the facts file (orchard/port_facts.py) carries: the file's hash and size, the class count, and the first lines of the
    "resolved vs class default" list.

"Resolved vs class default" compares each constructor parameter that has a default with the attribute of the same name on the constructed
module (or a known alias such as `eps` and `variance_epsilon`). A difference is listed. The comparison cannot see a setting that the
constructor stores under another name or does not store; those still appear in the per-class lines as attributes.

A model that ships its own code (issue 781). The script takes stage 1's remote-code route when transformers refuses a load, and its result then carries
a `remote_code` report: the sha256 of each `.py` file that may have run, and the names (never the values) of the environment variables removed. `clean`
checks the report, the file says that the model's code ran, and the summary and the ledger carry it. When the script fails it prints an
ERROR_MARK line with the exception class name only; `run` copies the name into the reason when it is a plain identifier and drops the line
otherwise, so no text of an exception, a path or a model file reaches the ledger.

The text is deterministic: classes and keys are sorted, no address or time is written, and the numbers are printed with `repr`. The file's
sha256 is in the facts file and in the ledger decision that records it, so a later reader can tell whether the coder's file was the one the
supervisor wrote. Everything that comes from the interpreter's output is bounded and type-checked here: this step is advisory, so a failure
is a reason string and the coder works without the file as before."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from orchard import safeio
from orchard.commands import run_command

FILE_NAME = "effective_construction.txt"
# The audit of the model's own code (stage 1's `allow_remote_code`), which the script writes into a directory the supervisor names, before the model's
# code imports. The supervisor reads it after the script ends, whatever became of the script (a timeout, a kill, an error).
AUDIT_FILE = "remote-code-audit.json"
AUDIT_MAX_BYTES = 1 << 20          # choice: 10,000 names and hashes are about 0.85 MB; more than 1 MiB is not an audit this module reads
MAX_AUDIT_FILES = 10000            # choice: a repository with more .py files than this gives "unreadable" (the size limit above is not what stops it)
MAX_AUDIT_FILES_SHOWN = 50         # the ledger entry carries this many names and hashes; the file on disk has them all
MAX_AUDIT_NAME = 200
MAX_AUDIT_SCRUBBED = 100
_AUDIT_PATH = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./+-]{0,199}")
_AUDIT_HASH = re.compile(r"[0-9a-f]{64}")
_AUDIT_WORDS = ("unreadable", "not a regular file", "over the size limit")      # what stage 1's `bounded_sha256` gives instead of a hash
_AUDIT_VAR = re.compile(r"[A-Za-z0-9_]{1,100}")
SCRIPT = Path(__file__).resolve().parent / "skills" / "full-port-templates" / "effective_construction.py"
MARK = "ORCHARD_EFFECTIVE_CONSTRUCTION"
ERROR_MARK = "ORCHARD_EFFECTIVE_CONSTRUCTION_ERROR"
TIMEOUT_S = 600                    # choice: a meta-device build of a large model is seconds; this ends a hang
OUTPUT_MAX = 4 << 20               # the script's output is bounded by its own caps; this stops a hostile interpreter
MAX_TEXT_BYTES = 48 * 1024         # the written file
LINE_CHARS = 400                   # one attribute line is cut here
SUMMARY_FLAGS = 12                 # lines of "resolved vs class default" the facts file carries
SUMMARY_FLAG_CHARS = 200
MAX_FLAGS = 200
NOTE = ("effective_construction.txt lists, for every module class the reference instantiates, the settings its constructor resolved (the "
        "activation, eps, group_size, dilation, padding, scales and so on), and lists every place the resolved value differs from the "
        "class's default signature. The reference excerpts elide constructor bodies, and a default shown there may not be the value used. "
        "Read the lines for the classes you port before you write them.")
# Constructor parameter -> attribute names that hold it after construction (the same name is always tried first).
ALIASES = {"eps": ("variance_epsilon", "epsilon", "layer_norm_epsilon", "norm_eps", "layer_norm_eps"),
           "epsilon": ("eps", "variance_epsilon"), "activation": ("act", "act_fn", "activation_fn", "hidden_act"),
           "act": ("activation", "act_fn"), "act_fn": ("activation", "act")}
# Parameters that decide what a layer computes come first in the list, so the facts file's short copy of it keeps them.
_FIRST = re.compile(r"act|eps|group|dilation|scale|gate|theta|base|window|budget|top_k|temperature|norm|bias|softcap|clip", re.I)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,119}")
_PATH = re.compile(r"[A-Za-z0-9_.{}<>-]{1,160}")
_REMOTE_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,159}")      # a path inside the snapshot: no leading dot or slash
_SHA256 = re.compile(r"[0-9a-f]{64}")
_UNHASHED = ("unreadable", "not a regular file", "over the size limit")      # the words stage 1's hasher gives for a file it could not hash
_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,59}")
MAX_REMOTE_FILES = 64
MAX_SCRUBBED = 40
TEXT_SCRUBBED = 10                 # names of removed variables the file itself lists


class ConstructionError(Exception):
    """The output could not be read or used. The message names the cause and never quotes the interpreter's text."""


def default_env() -> dict:
    """A nearly empty environment for the reference interpreter: offline, no device variables, no tokens."""
    return {"PATH": os.environ.get("PATH", os.defpath), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
            "TOKENIZERS_PARALLELISM": "false"}


def _failure_cause(out) -> str:
    """"" or `: <exception class name>[, the model's own code was in use]` from the script's ERROR_MARK line. The name must be a plain identifier
    and the flag a boolean, else the whole line is dropped: the reason holds nothing the model or an exception wrote."""
    if not isinstance(out, str) or len(out) > OUTPUT_MAX:
        return ""
    lines = [l for l in out.splitlines() if l.startswith(ERROR_MARK + " ")]
    if len(lines) != 1:
        return ""
    try:
        data = json.loads(lines[0][len(ERROR_MARK) + 1:])
    except (ValueError, RecursionError):
        return ""
    if not isinstance(data, dict) or not isinstance(data.get("error"), str) or not _CLASS_NAME.fullmatch(data["error"]) \
            or not isinstance(data.get("remote_code"), bool):
        return ""
    return f": {data['error']}" + (" (the model's own code was in use)" if data["remote_code"] else "")


def run(python, snapshot, *, runner=run_command, audit_dir=None) -> tuple[dict | None, str]:
    """(the script's parsed output, "") or (None, the reason). `python` is the reference interpreter. `runner(argv, timeout, env=...)` is
    commands.run_command in production. With `audit_dir` (an absolute directory) the script writes its audit of the model's own code there, before
    that code runs; without it the script keeps the audit in a temporary directory, as before."""
    if not isinstance(python, str) or not python:
        return None, "no reference interpreter"
    argv = [python, "-I", str(SCRIPT), str(Path(snapshot).absolute())]
    if audit_dir is not None:
        argv.append(str(Path(audit_dir).absolute()))
    try:
        result = runner(argv, TIMEOUT_S, env=default_env())
    except Exception as exc:
        return None, f"the reference interpreter could not be started ({exc.__class__.__name__})"
    if getattr(result, "returncode", 1) != 0:
        return None, f"the script exited with code {getattr(result, 'returncode', None)!r}" + _failure_cause(getattr(result, "stdout", ""))
    out = getattr(result, "stdout", "")
    if not isinstance(out, str) or len(out) > OUTPUT_MAX:
        return None, "the script's output is missing or too large"
    lines = [l for l in out.splitlines() if l.startswith(MARK + " ")]
    if len(lines) != 1:
        return None, f"the script printed {len(lines)} result lines, expected 1"
    try:
        data = json.loads(lines[0][len(MARK) + 1:])
    except (ValueError, RecursionError):
        return None, "the script's result line is not JSON"
    return (data, "") if isinstance(data, dict) else (None, "the script's result is not an object")


# ---- reading the output ---------------------------------------------------------------------------------------------------------

def _value(v, depth: int = 0):
    """A setting as the script prints it: None, bool, int, finite float, a short string, or a short list of those. Anything else is refused."""
    if v is None or isinstance(v, (bool, int)):
        if isinstance(v, int) and not isinstance(v, bool) and abs(v) > 10 ** 18:
            raise ConstructionError("a number is out of range")
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ConstructionError("a number is not finite")
        return v
    if isinstance(v, str):
        return v[:80]
    if isinstance(v, list) and depth == 0 and len(v) <= 8:
        return [_value(x, 1) for x in v]
    raise ConstructionError("a setting has an unexpected type")


def _count(x, what: str) -> int:
    if type(x) is not int or not 0 <= x <= 10 ** 9:
        raise ConstructionError(f"{what} is not a count")
    return x


def _remote_code(report) -> dict | None:
    """The `remote_code` report of the script's output, type-checked and bounded, or None when the key is absent. ConstructionError otherwise."""
    if report is None:
        return None
    if not isinstance(report, dict) or report.get("used") is not True:
        raise ConstructionError("remote_code is not an object that says used")
    files, scrubbed = report.get("files"), report.get("scrubbed")
    if not isinstance(files, dict) or len(files) > MAX_REMOTE_FILES:
        raise ConstructionError(f"remote_code files is not an object of at most {MAX_REMOTE_FILES}")
    out = {}
    for name, digest in files.items():
        if not isinstance(name, str) or not _REMOTE_FILE.fullmatch(name) or ".." in name.split("/") or "//" in name:
            raise ConstructionError("a remote_code file name is not a path inside the snapshot")
        if not isinstance(digest, str) or not (_SHA256.fullmatch(digest) or digest in _UNHASHED):
            raise ConstructionError("a remote_code hash is not a sha256")
        out[name] = digest
    if not isinstance(scrubbed, list) or len(scrubbed) > MAX_SCRUBBED or not all(isinstance(n, str) and _NAME.fullmatch(n) for n in scrubbed):
        raise ConstructionError("remote_code scrubbed is not a list of names")
    return {"files": dict(sorted(out.items())), "files_total": _count(report.get("files_total"), "files_total"), "scrubbed": sorted(set(scrubbed))}


def clean(data: dict) -> dict:
    """The script's output with every field type-checked and bounded, or ConstructionError."""
    classes = data.get("classes")
    if not isinstance(classes, list) or len(classes) > 200:
        raise ConstructionError("classes is not a list of at most 200")
    out = []
    for c in classes:
        if not isinstance(c, dict) or not isinstance(c.get("class"), str) or not _NAME.fullmatch(c["class"]):
            raise ConstructionError("a class name is not an identifier")
        sig = []
        for p in c.get("signature") or []:
            if not isinstance(p, dict) or not isinstance(p.get("name"), str) or not _NAME.fullmatch(p["name"]) or len(sig) >= 40:
                raise ConstructionError("a signature entry is malformed")
            sig.append({"name": p["name"], "default": _value(p.get("default")), "shown": p.get("shown") is True})
        variants = []
        for v in c.get("variants") or []:
            if not isinstance(v, dict) or not isinstance(v.get("attrs"), dict) or len(variants) >= 12:
                raise ConstructionError("a variant is malformed")
            paths = [x for x in (v.get("paths") or []) if isinstance(x, str) and _PATH.fullmatch(x)][:3]
            attrs, children = {}, {}
            for k, val in v["attrs"].items():
                if not isinstance(k, str) or not _NAME.fullmatch(k) or len(attrs) >= 60:
                    raise ConstructionError("an attribute name is not an identifier")
                if k == "children":
                    if not isinstance(val, dict):
                        raise ConstructionError("children is not an object")
                    children = {str(a)[:60]: str(b)[:80] for a, b in list(val.items())[:16]}
                else:
                    attrs[k] = _value(val)
            variants.append({"count": _count(v.get("count"), "a module count"), "paths": paths, "attrs": attrs, "children": children})
        variants.sort(key=lambda v: (-v["count"], json.dumps(v["attrs"], sort_keys=True), v["paths"]))       # the order of the text does not depend on the input's
        out.append({"class": c["class"], "signature": sig, "variants": variants,
                    "variants_left_out": _count(c.get("variants_left_out", 0), "variants_left_out")})
    return {"model_class": str(data.get("model_class", ""))[:80], "modules_total": _count(data.get("modules_total", 0), "modules_total"),
            "classes": sorted(out, key=lambda c: c["class"]), "classes_left_out": _count(data.get("classes_left_out", 0), "classes_left_out"),
            "modules_cut": data.get("modules_cut") is True, "remote_code": _remote_code(data.get("remote_code"))}


# ---- resolved vs class default --------------------------------------------------------------------------------------------------

def _same(a, b) -> bool:
    """Equal after a one-element list is read as its element (torch stores `dilation=1` as `(1,)`), and an int equals the same float."""
    if isinstance(a, list) and len(a) == 1:
        a = a[0]
    if isinstance(b, list) and len(b) == 1:
        b = b[0]
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return a == b


def _show(v) -> str:
    return repr(v)


def differences(data: dict) -> list[dict]:
    """Each place a constructed module's setting differs from the default in its class's signature: parameters that decide the computation (`_FIRST`) first, then by class, parameter and value.
    {"class", "param", "attr", "resolved", "default", "modules", "example"}. A class whose variants disagree gives one row per distinct value."""
    rows = {}
    for c in data["classes"]:
        for p in c["signature"]:
            if not p["shown"]:
                continue
            for v in c["variants"]:
                attr = next((a for a in (p["name"], *ALIASES.get(p["name"], ())) if a in v["attrs"]), None)
                if attr is None or _same(v["attrs"][attr], p["default"]):
                    continue
                key = (c["class"], p["name"], attr, json.dumps(v["attrs"][attr]))
                row = rows.setdefault(key, {"class": c["class"], "param": p["name"], "attr": attr, "resolved": v["attrs"][attr],
                                            "default": p["default"], "modules": 0, "example": v["paths"][0] if v["paths"] else ""})
                row["modules"] += v["count"]
    return sorted(rows.values(), key=lambda r: (not _FIRST.search(r["param"]), r["class"], r["param"], json.dumps(r["resolved"])))[:MAX_FLAGS]


def flag_line(row: dict) -> str:
    name = row["param"] if row["attr"] == row["param"] else f"{row['param']} (stored as {row['attr']})"
    where = f", e.g. {row['example']}" if row["example"] else ""
    return (f"{row['class']}.{name}: resolved {_show(row['resolved'])}, class default {_show(row['default'])} "
            f"({row['modules']} module{'s' if row['modules'] != 1 else ''}{where})")[:LINE_CHARS]


# ---- the file -------------------------------------------------------------------------------------------------------------------

def render(data: dict) -> tuple[str, list[str]]:
    """(the file's text, the "resolved vs class default" lines). `data` is `clean`'s output."""
    flags = [flag_line(r) for r in differences(data)]
    lines = ["# Effective construction of the reference model", "",
             f"Built on the CPU from the checkpoint's config by the reference interpreter (meta device: no weights). Model class "
             f"{data['model_class'] or '(unknown)'}, {data['modules_total']} modules, {len(data['classes'])} module classes"
             + (f" ({data['classes_left_out']} more not listed)" if data["classes_left_out"] else "") + ".",
             "Each line below is one class and one group of modules that share the same settings. `{i}` stands for a layer number. "
             "These are the values the constructors resolved from the config, which can differ from the default in the class's "
             "signature in the reference excerpts. Settings the module keeps under another name appear as attributes.", "",
             "## Resolved vs class default", ""]
    remote = data.get("remote_code")
    if remote:
        n = remote["files_total"]
        names = ", ".join(remote["scrubbed"][:TEXT_SCRUBBED])
        lines[2:2] = [f"The model ships its own code ({n} .py file{'s' if n != 1 else ''}). The script ran it on the meta device to build this file, after "
                      f"environment variables with credential-like names were removed ({names or 'none were set'}), with the hub offline and HOME "
                      "moved to an empty directory. The ledger lists the sha256 of each file. This is not a sandbox.", ""]
    lines += [f"- {f}" for f in flags] or ["- (none: every constructor parameter that has a default kept it)"]
    lines += ["", "## Modules", ""]
    for c in data["classes"]:
        if not c["variants"] or all(not v["attrs"] and not v["children"] for v in c["variants"]):
            continue
        for v in c["variants"]:
            attrs = ", ".join(f"{k}={_show(x)}" for k, x in v["attrs"].items())
            kids = ", ".join(f"{k}: {t}" for k, t in v["children"].items())
            where = " ".join(v["paths"])
            row = f"{c['class']} x{v['count']} at {where}: {attrs}" + (f" | children {kids}" if kids else "")
            lines.append(row[:LINE_CHARS * 2])
        if c["variants_left_out"]:
            lines.append(f"{c['class']}: {c['variants_left_out']} more groups of modules with other settings are not listed")
    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        text = text.encode("utf-8")[:MAX_TEXT_BYTES].decode("utf-8", "ignore").rsplit("\n", 1)[0] + "\n[cut: the rest of the modules are not listed]\n"
    return text, flags


def write(stage_dir, text: str) -> Path:
    path = Path(stage_dir) / FILE_NAME
    safeio.atomic_write(path, text.encode("utf-8"), mode=0o644)
    return path


def summary(text: str, flags: list[str], data: dict) -> dict:
    """What the facts file carries about the file: its name, sha256 and size, the class count and the first lines of the list. The hash is of
    the bytes written, so the facts file pins the sibling file."""
    raw = text.encode("utf-8")
    out = {"file": FILE_NAME, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "classes": len(data["classes"]),
           "resolved_vs_default": [f[:SUMMARY_FLAG_CHARS] for f in flags[:SUMMARY_FLAGS]],
           "resolved_vs_default_total": len(flags), "note": NOTE}
    if data.get("remote_code"):
        out["remote_code"] = data["remote_code"]           # the ledger records it; the facts file keeps its own bounded copy of the summary
    return out


def for_snapshot(python, snapshot, *, runner=run_command, audit_dir=None) -> tuple[str, list[str], dict, str]:
    """(text, flags, summary, "") on success; ("", [], {}, the reason) when the interpreter, the script or its output failed. `audit_dir` is passed on
    to `run`."""
    data, why = run(python, snapshot, runner=runner, audit_dir=audit_dir)
    if data is None:
        return "", [], {}, why
    try:
        cleaned = clean(data)
    except ConstructionError as exc:
        return "", [], {}, f"the script's output was refused: {exc}"
    text, flags = render(cleaned)
    return text, flags, summary(text, flags, cleaned), ""


# ---- the audit of the model's own code -----------------------------------------------------------------------------------------------

def _clean_audit(data, sha256: str) -> dict:
    """The audit as the ledger may carry it, or {"unreadable": a fixed reason}. Every field is type-checked and bounded; a reason never quotes the
    file, which the model's own code could have written after the script did."""
    bad = {"unreadable": "the audit file is not in the expected shape"}
    if not isinstance(data, dict) or data.get("used") is not True:
        return bad
    files, scrubbed = data.get("files"), data.get("scrubbed")
    if not isinstance(files, dict) or not isinstance(scrubbed, list) or len(files) > MAX_AUDIT_FILES or len(scrubbed) > MAX_AUDIT_SCRUBBED:
        return bad
    for name, digest in files.items():
        if (not isinstance(name, str) or len(name) > MAX_AUDIT_NAME or not _AUDIT_PATH.fullmatch(name) or ".." in name.split("/")
                or not isinstance(digest, str) or not (_AUDIT_HASH.fullmatch(digest) or digest in _AUDIT_WORDS)):
            return bad
    if not all(isinstance(v, str) and _AUDIT_VAR.fullmatch(v) for v in scrubbed):
        return bad
    shown = dict(sorted(files.items())[:MAX_AUDIT_FILES_SHOWN])
    return {"used": True, "files": shown, "files_total": len(files), "scrubbed": sorted(set(scrubbed)), "sha256": sha256}


def read_audit(directory) -> dict | None:
    """What the script wrote to `directory/remote-code-audit.json`: None when no file is there (the route was not used, or the script never got that
    far), {"unreadable": reason} when there is one that cannot be used, else the cleaned audit (`_clean_audit`) with the sha256 of the bytes that were
    parsed (one read, so the hash and the content belong together). Never raises. The directory is held open without following a link in its last
    name and the file is opened beneath it, size-limited, without following a link."""
    try:
        with safeio.pinned_root(directory) as pin:
            raw = safeio.read_bytes(AUDIT_FILE, max_bytes=AUDIT_MAX_BYTES, anchor=pin.fd)
            data = safeio.parse_json_bytes(raw, max_bytes=AUDIT_MAX_BYTES)
    except safeio.SafeReadError as exc:
        if "(ENOENT)" in str(getattr(exc, "reason", "")) or "(ENOENT)" in str(exc):
            return None
        return {"unreadable": f"the audit file could not be read: {str(getattr(exc, 'reason', exc.__class__.__name__))[:120]}"}
    except OSError as exc:
        return {"unreadable": f"the audit file could not be read ({exc.__class__.__name__})"}
    return _clean_audit(data, hashlib.sha256(raw).hexdigest())
