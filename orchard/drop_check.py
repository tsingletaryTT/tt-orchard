# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The release-day check: which route a new Qwen release takes, from its config alone.

    tt-orchard drop-check Qwen/Qwen4-27B [--revision SHA] [--capabilities FILE] [--from-dir DIR] [--json] [--out FILE]

It reads only small files of the release (config.json, generation_config.json, tokenizer files, the chat template and the
weight index), never a weight file, and never sends a Hugging Face token. It refuses any organisation outside ALLOWED_ORGS
before it makes a request: on 2026-11-09 several look-alike "Qwen4-27B" repositories appeared under other names, one holding a
zip file that looked like malware bait.

Verdicts and routes:
  covered / covered-with-notes  route "qwen4-package": the Qwen4 package's capabilities file (orchard-capabilities.json) covers
                                every value the release's config sets; or route "qwen3.8-bundle": the release has Qwen3.8-27B's
                                architecture and widths, so the Qwen3.8 bundle serves it as a weights swap.
  not-covered                   route "full-port": something the package cannot serve (experts, an n-gram embedding, a missing or
                                malformed key, a width that does not split over 2 and 4 chips, a value outside a capability rule,
                                or no capabilities file yet). Each reason is named.
  needs-operator                the release needs a reference interpreter whose transformers has its architecture class
                                (scripts/reference_venv.sh builds one); the operator sets reference_python and runs it again.
  refused                       the repository id or revision is not acceptable; nothing was read.

The capabilities file (schema 1), shipped with the Qwen4 package:
  {"schema": 1, "bundle": "<org>/<name>", "model_types": [...], "architectures": [...],
   "keys": {"<config key>": {"equals": v} | {"one_of": [...]} | {"multiple_of": n} | {"min": n, "max": n}, ...},
   "features": {"indexer": "off" | "exact-up-to-2051" | "full", "mtp": "ignored", "vision": "off"}}
EXAMPLE_CAPABILITIES is the file for the package as planned; stage 0's triage reads the same file (plan track C1).

The Qwen4 rules follow transformers 5.19.0 `qwen4_exp` (hyper-connections, QSA indexer, gated delta net) and the dense layer rule
of orchard/skills/reference-gate-templates/qwen4_dense.py. No Qwen4-27B config was public when this was written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ALLOWED_ORGS = ("Qwen",)
QWEN4_BUNDLE = "mando2222/qwen4-27b-dense-v6-thin"
QWEN38_BUNDLE = "mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin"
HUB = "https://huggingface.co"
USER_AGENT = "tt-orchard-drop-check"
TIMEOUT_S = 30

EXIT_COVERED, EXIT_REFUSED, EXIT_NOT_COVERED, EXIT_NEEDS_OPERATOR, EXIT_NOT_FOUND = 0, 2, 3, 4, 5
EXIT_FOR = {"covered": EXIT_COVERED, "covered-with-notes": EXIT_COVERED, "refused": EXIT_REFUSED,
            "not-covered": EXIT_NOT_COVERED, "needs-operator": EXIT_NEEDS_OPERATOR, "not-found": EXIT_NOT_FOUND}

# The files read, with the most bytes each may have. Only config.json is required.
METADATA_FILES = {"config.json": 1 << 20, "generation_config.json": 1 << 20, "tokenizer_config.json": 8 << 20,
                  "chat_template.jinja": 1 << 20, "tokenizer.json": 64 << 20, "model.safetensors.index.json": 16 << 20}

REPO_RE = re.compile(r"([A-Za-z0-9][A-Za-z0-9_-]{0,38})/([A-Za-z0-9][A-Za-z0-9._-]{0,95})")
REV_RE = re.compile(r"[0-9a-f]{40}")
FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

QWEN35_TYPES = ("qwen3_5", "qwen3_5_text")
QWEN4_EXP_TYPES = ("qwen4_exp", "qwen4_exp_text")
QWEN4_NEW_TYPES = ("qwen4", "qwen4_text")          # a type the reference interpreter (transformers 5.19.0) does not have
QWEN4_TYPES = QWEN4_EXP_TYPES + QWEN4_NEW_TYPES

# Qwen3.8-27B's widths (tests/fixtures/model-configs/qwen3.8-27b.json). A qwen3_5 release with all of them is a weights swap.
QWEN38_SHAPE = {"hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
                "head_dim": 256, "vocab_size": 248320, "intermediate_size": 17408, "linear_num_key_heads": 16,
                "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4}
QWEN38_TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"

COUNT_KEYS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
              "vocab_size", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
              "linear_conv_kernel_dim", "hc_count", "hc_lowrank")
REQUIRED_QWEN4_KEYS = COUNT_KEYS + ("rms_norm_eps", "hidden_act", "layer_types", "output_gate_type", "rope_parameters",
                                    "tie_word_embeddings")
INDEXER_KEYS = ("indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio")
LAYER_KINDS = ("linear_attention", "full_attention", "indexed_attention")
MAX_COUNT = 10 ** 7
CHIP_COUNTS = (2, 4)
TILE = 32

EXAMPLE_CAPABILITIES = {
    "schema": 1,
    "bundle": QWEN4_BUNDLE,
    "model_types": list(QWEN4_EXP_TYPES),
    "architectures": ["Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"],
    "keys": {"hidden_size": {"equals": 5120}, "num_attention_heads": {"equals": 24}, "head_dim": {"equals": 256},
             "num_key_value_heads": {"one_of": [2, 4]}, "intermediate_size": {"multiple_of": 128},
             "linear_num_key_heads": {"equals": 16}, "linear_num_value_heads": {"equals": 48},
             "linear_key_head_dim": {"equals": 128}, "linear_value_head_dim": {"equals": 128},
             "linear_conv_kernel_dim": {"equals": 4}, "vocab_size": {"equals": 248320}, "hc_count": {"equals": 4},
             "hc_lowrank": {"one_of": [320, 640]}, "output_gate_type": {"one_of": ["silu", "sigmoid"]},
             "hidden_act": {"equals": "silu"}},
    "features": {"indexer": "exact-up-to-2051", "mtp": "ignored", "vision": "off"},
}
RULE_NAMES = ("equals", "one_of", "multiple_of", "min", "max")
INDEXER_FEATURES = ("off", "exact-up-to-2051", "full")


class Unreadable(RuntimeError):
    """A file of the release exists but cannot be read safely (a link, a special file, a read error)."""


class TooLarge(Unreadable):
    """A file is larger than its budget."""


_BAD = object()     # _parse_json's result when the bytes are missing or not JSON (a JSON null is a value)


# ---- identifiers ----------------------------------------------------------------------------------------------------

def repo_problem(repo) -> str | None:
    """None when `repo` is a plain org/name in an allowed organisation; else why it is refused."""
    if not isinstance(repo, str):
        return f"the repository id must be a string, got {type(repo).__name__}"
    m = REPO_RE.fullmatch(repo)
    if not m or ".." in repo:
        return "the repository id must be org/name with letters, digits, '.', '_' and '-' only"
    if m.group(1) not in ALLOWED_ORGS:
        return f"organisation {m.group(1)!r} is not allowed (allowed: {', '.join(ALLOWED_ORGS)})"
    return None


def revision_problem(rev) -> str | None:
    if not isinstance(rev, str) or not REV_RE.fullmatch(rev):
        return "the revision must be a 40-character commit hash"
    return None


def _file_problem(name) -> str | None:
    if not isinstance(name, str) or not FILE_RE.fullmatch(name) or ".." in name:
        return "a file name must be a plain name in the repository root"
    return None


# ---- value checks ---------------------------------------------------------------------------------------------------

def _count(tc: dict, key: str, reasons: list) -> int | None:
    v = tc.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_COUNT:
        reasons.append(f"{key} is {_show(v)}; it must be a whole number from 1 to {MAX_COUNT}")
        return None
    return v


def _show(v) -> str:
    text = json.dumps(v, default=repr) if not isinstance(v, float) or math.isfinite(v) else repr(v)
    return text if len(text) <= 60 else text[:57] + "..."


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _parse_json(raw, name: str, reasons: list):
    if not isinstance(raw, (bytes, bytearray)):
        reasons.append(f"{name} is missing")
        return _BAD
    try:
        value = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        reasons.append(f"{name} is not valid JSON ({type(exc).__name__})")
        return _BAD
    return value


def _capabilities_problems(caps) -> list[str]:
    if caps is None:
        return ["the Qwen4 package declares no capabilities (no orchard-capabilities.json), so the drop-in route is not ready"]
    if not isinstance(caps, dict):
        return [f"the capabilities file is malformed: a JSON object is needed, got {type(caps).__name__}"]
    out = []
    if caps.get("schema") != 1:
        out.append(f"the capabilities file has schema {_show(caps.get('schema'))}; this check reads schema 1")
    for field in ("model_types", "architectures"):
        v = caps.get(field)
        if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
            out.append(f"the capabilities file is malformed: {field} must be a non-empty list of strings")
    keys = caps.get("keys")
    if not isinstance(keys, dict):
        out.append("the capabilities file is malformed: keys must be an object")
    else:
        for k, rule in keys.items():
            if not isinstance(rule, dict) or not rule or set(rule) - set(RULE_NAMES):
                out.append(f"the capabilities file is malformed: the rule for {k!r} must be an object with {', '.join(RULE_NAMES)}")
                continue
            if "one_of" in rule and not isinstance(rule["one_of"], list):
                out.append(f"the capabilities file is malformed: one_of for {k!r} must be a list")
            for n in ("multiple_of", "min", "max"):
                if n in rule and not _number(rule[n]):
                    out.append(f"the capabilities file is malformed: {n} for {k!r} must be a number")
            if "multiple_of" in rule and _number(rule["multiple_of"]) and rule["multiple_of"] <= 0:
                out.append(f"the capabilities file is malformed: multiple_of for {k!r} must be positive")
    feats = caps.get("features", {})
    if not isinstance(feats, dict) or feats.get("indexer", "off") not in INDEXER_FEATURES:
        out.append(f"the capabilities file is malformed: features.indexer must be one of {', '.join(INDEXER_FEATURES)}")
    return out


def _json_eq(a, b) -> bool:
    """Equality of JSON values: true is not 1 and false is not 0 (stage 0's capability reader compares the same way)."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_eq(a[k], b[k]) for k in a)
    return type(a) in (int, float) and type(b) in (int, float) and a == b or type(a) is type(b) and a == b


def _lookup(tc: dict, key: str):
    """(found, value) for `key` in the text config; a dotted key ("rope_parameters.rope_theta") walks nested objects."""
    node = tc
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _rule_problem(key: str, value, rule: dict) -> str | None:
    if "equals" in rule and not _json_eq(value, rule["equals"]):
        return f"{key} is {_show(value)}; the package serves {_show(rule['equals'])}"
    if "one_of" in rule and not any(_json_eq(value, v) for v in rule["one_of"]):
        return f"{key} is {_show(value)}; the package serves one of {_show(rule['one_of'])}"
    if "multiple_of" in rule and (not _number(value) or value % rule["multiple_of"]):
        return f"{key} is {_show(value)}; the package needs a multiple of {rule['multiple_of']}"
    if "min" in rule and (not _number(value) or value < rule["min"]):
        return f"{key} is {_show(value)}; the package needs at least {rule['min']}"
    if "max" in rule and (not _number(value) or value > rule["max"]):
        return f"{key} is {_show(value)}; the package needs at most {rule['max']}"
    return None


def _moe_layers(tc: dict, n_layers: int | None, reasons: list) -> list[int]:
    """Layers with a routed-expert block, by the dense rule of qwen4_dense.py (Intel's vLLM patch)."""
    experts = tc.get("num_experts", 0)
    if experts in (None, 0):
        return []
    if isinstance(experts, bool) or not isinstance(experts, int) or experts < 0:
        reasons.append(f"num_experts is {_show(experts)}; it must be a whole number")
        return []
    step = tc.get("decoder_sparse_step", 1)
    dense = tc.get("mlp_only_layers", []) or []
    if isinstance(step, bool) or not isinstance(step, int) or step < 1 or not isinstance(dense, list):
        reasons.append("decoder_sparse_step or mlp_only_layers is malformed")
        return []
    return [i for i in range(n_layers or 0) if i not in dense and (i + 1) % step == 0]


def _split_problems(tc: dict, counts: dict) -> list[str]:
    out = []
    for n in CHIP_COUNTS:
        for key in ("num_attention_heads", "linear_num_key_heads", "linear_num_value_heads"):
            v = counts.get(key)
            if v and v % n:
                out.append(f"{key} {v} does not divide over {n} chips")
        kv = counts.get("num_key_value_heads")
        if kv and kv % n and n % kv:
            out.append(f"num_key_value_heads {kv} neither divides over {n} chips nor replicates across them")
        for key in ("hidden_size", "intermediate_size"):
            v = counts.get(key)
            if v and v % (n * TILE):
                out.append(f"{key} {v} does not give whole {TILE}-wide tiles on {n} chips")
    return sorted(set(out))


# ---- the classifier -------------------------------------------------------------------------------------------------

def _command(route: str | None, repo: str, qwen4_bundle: str, qwen38_bundle: str) -> list[str]:
    if route == "qwen4-package":
        return ["tt-orchard", "bringup", repo, "--base", qwen4_bundle]
    if route == "qwen3.8-bundle":
        return ["tt-orchard", "bringup", repo, "--base", qwen38_bundle]
    if route == "full-port":
        return ["tt-orchard", "bringup", repo, "--base", qwen38_bundle, "--full-port"]
    return []


def _tokenizer_notes(files: dict) -> list[str]:
    notes = []
    tok = files.get("tokenizer.json")
    if isinstance(tok, (bytes, bytearray)):
        sha = hashlib.sha256(tok).hexdigest()
        notes.append("tokenizer matches Qwen3.8" if sha == QWEN38_TOKENIZER_SHA256 else
                     f"tokenizer differs from Qwen3.8 (sha256 {sha[:16]})")
    template = files.get("chat_template.jinja")
    if not isinstance(template, (bytes, bytearray)):
        tcfg = _parse_json(files.get("tokenizer_config.json"), "tokenizer_config.json", [])
        template = tcfg.get("chat_template") if isinstance(tcfg, dict) else None
        template = template.encode() if isinstance(template, str) else None
    if template is not None:
        notes.append("chat template has enable_thinking" if b"enable_thinking" in template else
                     "chat template has no enable_thinking toggle; thinking on/off is prompted differently from Qwen3.8")
    return notes


def classify(repo, revision, files: dict, *, capabilities=None, class_probe=None, qwen4_bundle: str = QWEN4_BUNDLE,
             qwen38_bundle: str = QWEN38_BUNDLE) -> dict:
    """The verdict for `repo` at `revision` from its metadata `files` ({name: bytes}). Pure: no I/O apart from `class_probe`,
    a callable (list of class names) -> {name: bool} that asks the reference interpreter, or None when no interpreter is named.
    """
    out = {"repo": repo if isinstance(repo, str) else None, "revision": revision if isinstance(revision, str) else None,
           "verdict": None, "route": None, "reasons": [], "notes": [], "facts": {}, "command": []}
    problem = repo_problem(repo) or revision_problem(revision)
    if problem:
        out.update(verdict="refused", reasons=[problem])
        return out
    reasons, operator, notes = [], [], []
    cfg = _parse_json(files.get("config.json"), "config.json", reasons)
    if cfg is _BAD:
        cfg = None
    elif not isinstance(cfg, dict):
        reasons.append(f"config.json is a JSON {type(cfg).__name__}; an object is needed")
        cfg = None
    route = "full-port"
    if cfg is not None:
        tc = cfg.get("text_config", cfg)
        if not isinstance(tc, dict):
            reasons.append("config.json text_config is not an object")
            tc = {}
        mtype = cfg.get("model_type") if isinstance(cfg.get("model_type"), str) else tc.get("model_type")
        arch = cfg.get("architectures")
        arch = [a for a in arch if isinstance(a, str)] if isinstance(arch, list) else []
        out["facts"] = {"model_type": mtype, "architectures": arch,
                        **{k: tc.get(k) for k in COUNT_KEYS if _number(tc.get(k))}}
        if mtype in QWEN35_TYPES:
            diff = [f"{k} is {_show(tc.get(k))}; Qwen3.8-27B has {v}" for k, v in QWEN38_SHAPE.items() if tc.get(k) != v]
            if diff:
                reasons += diff
            else:
                route = "qwen3.8-bundle"
        elif mtype in QWEN4_TYPES:
            route = "qwen4-package"
            counts = {k: _count(tc, k, reasons) for k in COUNT_KEYS if k in tc}
            reasons += [f"{k} is missing from the config" for k in REQUIRED_QWEN4_KEYS if k not in tc]
            rope = tc.get("rope_parameters")
            theta = rope.get("rope_theta") if isinstance(rope, dict) else tc.get("rope_theta")
            if not _number(theta) or theta <= 0:
                reasons.append("rope_theta is missing or not a positive number (transformers would silently use 10000)")
            if not isinstance(tc.get("rms_norm_eps"), (int, float)) or isinstance(tc.get("rms_norm_eps"), bool):
                reasons.append(f"rms_norm_eps is {_show(tc.get('rms_norm_eps'))}; a number is needed") if "rms_norm_eps" in tc else None
            layers = tc.get("layer_types")
            n_layers = counts.get("num_hidden_layers")
            if "layer_types" in tc and (not isinstance(layers, list) or any(x not in LAYER_KINDS for x in layers)
                                         or (n_layers and len(layers) != n_layers)):
                reasons.append(f"layer_types must list one of {', '.join(LAYER_KINDS)} for each of the num_hidden_layers layers")
            moe = _moe_layers(tc, n_layers, reasons)
            if moe:
                reasons.append(f"mixture-of-experts layers ({len(moe)} of {n_layers}); the Qwen4 package serves dense layers only")
            ple = tc.get("ple_layer_ids")
            if ple:
                reasons.append(f"an n-gram per-layer embedding (ple_layer_ids {_show(ple)}); the package does not serve it")
            present = [k for k in INDEXER_KEYS if k in tc]
            indexer = bool(present)
            if present and len(present) != len(INDEXER_KEYS):
                reasons.append("the indexer keys are incomplete: " + ", ".join(k for k in INDEXER_KEYS if k not in tc) + " missing")
            elif present:
                for k in INDEXER_KEYS:
                    _count(tc, k, reasons)
                if tc.get("indexer_kv_heads") not in (1, None):
                    reasons.append(f"indexer_kv_heads is {_show(tc.get('indexer_kv_heads'))}; the QSA indexer has 1")
            reasons += _split_problems(tc, {k: v for k, v in counts.items() if v})
            cap_problems = _capabilities_problems(capabilities)
            reasons += cap_problems
            if not cap_problems:
                if mtype not in capabilities["model_types"]:
                    reasons.append(f"model_type {mtype!r} is not one the package serves ({', '.join(capabilities['model_types'])})")
                for a in arch:
                    if a not in capabilities["architectures"]:
                        reasons.append(f"architecture {a!r} is not one the package registers")
                for k, rule in capabilities["keys"].items():
                    found, value = _lookup(tc, k)
                    if found:
                        p = _rule_problem(k, value, rule)
                        if p:
                            reasons.append(p)
                feature = capabilities.get("features", {}).get("indexer", "off")
                if indexer and feature == "off":
                    reasons.append("the release has a QSA indexer and the package does not serve one")
                elif indexer and feature == "exact-up-to-2051":
                    notes.append("the release has a QSA indexer; the package serves it with max_model_len 2048, where the "
                                 "indexer selects every block and attention is exact")
            mtp = tc.get("mtp_num_hidden_layers") or (cfg.get("mtp") if isinstance(cfg.get("mtp"), dict) else None)
            if mtp:
                notes.append("the release has an MTP head; it is served without speculative decoding")
            if mtype in QWEN4_NEW_TYPES:
                probe = class_probe(arch) if callable(class_probe) and arch else None
                missing = [a for a in arch if not (isinstance(probe, dict) and probe.get(a) is True)]
                if missing:
                    operator.append(f"the reference interpreter's transformers has no {', '.join(missing)} (or was not "
                                    "checked); build one with scripts/reference_venv.sh '<transformers spec>' and set "
                                    "reference_python in config/bringup.toml")
        else:
            reasons.append(f"model_type {mtype!r} is neither Qwen3.8's ({', '.join(QWEN35_TYPES)}) nor Qwen4's "
                           f"({', '.join(QWEN4_TYPES)})")
    notes += _tokenizer_notes(files)
    if reasons:
        route = "full-port"
        out.update(verdict="not-covered", reasons=reasons)
    elif operator:
        out.update(verdict="needs-operator", reasons=operator)
    else:
        out.update(verdict="covered-with-notes" if notes and route == "qwen4-package" else "covered")
    out["route"] = route
    out["notes"] = notes
    out["command"] = _command(route, repo, qwen4_bundle, qwen38_bundle)
    return out


# ---- reading the release ----------------------------------------------------------------------------------------------

class HubClient:
    """Reads public files of a Hugging Face repository with no token. `opener` stands in for urllib.request.urlopen."""

    def __init__(self, opener=None, base: str = HUB):
        self.opener = opener or urllib.request.urlopen
        self.base = base

    def _open(self, url: str, max_bytes: int) -> bytes | None:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with self.opener(req, timeout=TIMEOUT_S) as resp:
                data = resp.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                return None
            raise
        if len(data) > max_bytes:
            raise TooLarge(f"{url.rsplit('/', 1)[-1]} is larger than {max_bytes} bytes")
        return data

    def info(self, repo: str, revision: str | None = None) -> dict | None:
        """The model's API record (sha, lastModified, siblings with sizes), or None when it does not exist or is gated."""
        p = repo_problem(repo) or (revision_problem(revision) if revision is not None else None)
        if p:
            raise ValueError(p)
        path = f"/api/models/{repo}" + (f"/revision/{revision}" if revision else "") + "?blobs=true"
        raw = self._open(self.base + path, 8 << 20)
        if raw is None:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return None
        return value if isinstance(value, dict) else None

    def get(self, repo: str, revision: str, name: str, *, max_bytes: int) -> bytes | None:
        p = repo_problem(repo) or revision_problem(revision) or _file_problem(name)
        if p:
            raise ValueError(p)
        return self._open(f"{self.base}/{repo}/resolve/{revision}/{name}", max_bytes)


class DirClient:
    """A release laid out in a directory: info.json (the API record) and the files. For tests and simulated releases."""

    def __init__(self, path):
        self.path = Path(path)

    def info(self, repo: str, revision: str | None = None) -> dict | None:
        from orchard import safeio
        try:
            value = safeio.read_json(self.path / "info.json", max_bytes=8 << 20)
        except (OSError, ValueError, safeio.SafeReadError):
            return None
        return value if isinstance(value, dict) else None

    def get(self, repo: str, revision: str, name: str, *, max_bytes: int) -> bytes | None:
        import os

        from orchard import safeio
        if _file_problem(name):
            raise ValueError(_file_problem(name))
        path = self.path / name
        if not os.path.lexists(path):
            return None
        try:
            return safeio.read_bytes(path, max_bytes=max_bytes)
        except safeio.TooLarge as exc:
            raise TooLarge(f"{name} is larger than {max_bytes} bytes") from exc
        except (OSError, ValueError, safeio.SafeReadError) as exc:
            raise Unreadable(f"{name} cannot be read safely ({type(exc).__name__})") from exc


def fetch_metadata(client, repo: str, revision: str) -> tuple[dict, list[str]]:
    """({name: bytes}, problems) for METADATA_FILES. A missing file is left out; an oversize file is a problem."""
    files, problems = {}, []
    for name, cap in METADATA_FILES.items():
        try:
            data = client.get(repo, revision, name, max_bytes=cap)
        except Unreadable as exc:
            problems.append(str(exc))
            continue
        if data is not None:
            files[name] = data
    return files, problems


def index_shards(raw) -> set[str] | None:
    """The shard file names a safetensors weight index lists, or None when it cannot be read."""
    idx = _parse_json(raw, "model.safetensors.index.json", [])
    idx = None if idx is _BAD else idx
    wm = idx.get("weight_map") if isinstance(idx, dict) else None
    if not isinstance(wm, dict) or not wm or not all(isinstance(v, str) and not _file_problem(v) for v in wm.values()):
        return None
    return set(wm.values())


def class_probe_for(python: str | None):
    """A class_probe that asks `python` (the reference interpreter) through orchard.stack_pins; None without an interpreter."""
    if not python:
        return None
    from orchard import stack_pins

    def probe(classes):
        res = stack_pins.probe_interpreter(python, list(classes))
        return res.get("classes") if res.get("state") == "answered" else None
    return probe


# ---- command line ---------------------------------------------------------------------------------------------------

def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard drop-check", add_help=add_help,
                                description="say which bring-up route a new Qwen release takes, from its config alone")
    p.add_argument("repo", help="the release, org/name (organisation must be one of: " + ", ".join(ALLOWED_ORGS) + ")")
    p.add_argument("--revision", help="a 40-character commit (default: the repository's current one)")
    p.add_argument("--capabilities", help="the Qwen4 package's orchard-capabilities.json")
    p.add_argument("--bundle", default=QWEN4_BUNDLE, help=f"the Qwen4 package (default {QWEN4_BUNDLE})")
    p.add_argument("--qwen38-bundle", default=QWEN38_BUNDLE, help="the Qwen3.8 bundle for a same-architecture release")
    p.add_argument("--reference-python", help="the reference interpreter to ask for a new architecture class")
    p.add_argument("--from-dir", help="read the release from a directory (info.json plus files), for tests and rehearsals")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--out", help="also write the result as JSON to this file")
    return p


def run(args) -> tuple[dict, int]:
    problem = repo_problem(args.repo) or (revision_problem(args.revision) if args.revision else None)
    if problem:
        res = classify(args.repo, args.revision or "", {})
        res["reasons"] = [problem]
        return res, EXIT_REFUSED
    client = DirClient(args.from_dir) if args.from_dir else HubClient()
    info = client.info(args.repo, args.revision)
    if info is None:
        return ({"repo": args.repo, "revision": args.revision, "verdict": "not-found", "route": None,
                 "reasons": ["the repository does not exist, is private or gated, or the Hub did not answer"], "notes": [],
                 "facts": {}, "command": []}, EXIT_NOT_FOUND)
    revision = args.revision or info.get("sha")
    caps = None
    if args.capabilities:
        from orchard import safeio
        try:
            caps = safeio.read_json(args.capabilities, max_bytes=1 << 20)
        except (OSError, ValueError, safeio.SafeReadError) as exc:
            caps = f"unreadable ({type(exc).__name__})"
    files, problems = fetch_metadata(client, args.repo, revision) if not revision_problem(revision) else ({}, [])
    res = classify(args.repo, revision, files, capabilities=caps, class_probe=class_probe_for(args.reference_python),
                   qwen4_bundle=args.bundle, qwen38_bundle=args.qwen38_bundle)
    if problems:
        res["reasons"] = problems + res["reasons"]
        if res["verdict"] in ("covered", "covered-with-notes"):
            res["verdict"], res["route"] = "not-covered", "full-port"
            res["command"] = _command("full-port", args.repo, args.bundle, args.qwen38_bundle)
    return res, EXIT_FOR[res["verdict"]]


def render(res: dict) -> str:
    lines = [f"{res['repo']} @ {(res.get('revision') or '?')[:12]}: {res['verdict']}" +
             (f" (route {res['route']})" if res.get("route") else "")]
    lines += [f"  reason: {r}" for r in res.get("reasons", [])]
    lines += [f"  note:   {n}" for n in res.get("notes", [])]
    if res.get("command"):
        lines.append("  next:   " + " ".join(res["command"]))
    return "\n".join(lines)


def report(args, out=None) -> int:
    """Run the check for parsed `args`, print the result to `out` and write --out. The exit code follows the verdict."""
    res, code = run(args)
    print(json.dumps(res, indent=1) if args.json else render(res), file=out or sys.stdout)
    if args.out:
        from orchard import safeio
        safeio.atomic_write(args.out, json.dumps(res, indent=1) + "\n")
    return code


def main(argv=None) -> int:
    return report(parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
