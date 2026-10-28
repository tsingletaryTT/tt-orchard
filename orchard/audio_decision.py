# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The gate's check of the audio class decision that stage 0's template writes under `delta.json` `audio.decision`.

The template decides (orchard/skills/delta-triage-templates/delta_triage.py, `decide_audio`); this module re-reads the record the way `gate_package` re-reads
a package, so an agent that edited delta.json cannot pass a class its own evidence does not support. A delta.json with no decision that the current template
did not write (a text model, or a run from before the decision existed) has nothing to check.

What it requires:
- a decision at all, when the delta carries the current template's stamp and is an audio result (`current_template`): deleting `audio.decision` and changing
  `class` and `path` together is refused;
- the fields: class, path, alternative_path, confidence, strict, differences, overrides, and the class and path equal to delta.json's;
- `strict` equal to the stage's `triage_config.json` `audio_strict` (when the caller gives it);
- each difference: a key listed once, both values, a category (serving, structural or unknown), `decided_by` (`allow-list` or `override`), and a non-empty
  `source` when an override decided it. An `allow-list` category has to be what the allow-list table gives for that key and those values (the table is
  audio_config_categories.json next to the template; `category_rules`); an `override` category has to be recorded in `overrides`;
- each override: a kind (category or path), a non-empty source and the fields its kind needs. A category override names a difference that says it was decided
  by an override, with the same category and source, and its `from` is what the allow-list gave. At most one path override; its path is the decision's path;
- the differences against the comparison evidence in `audio.config`: every key the evidence lists is in the decision with the same values, and the decision
  names no key the evidence does not list (the evidence is cut at AUDIO_MAX_LISTED keys per section, and a cut section is not held to the second half);
- consistency: a weights-only decision has only serving differences (none at all when strict), unless a weights-only path override with a source explains
  them; and it has no tensor-level difference, no text-encoder difference and no read problem in the evidence, which no override can explain away. A class of
  unknown has path full-port and confidence low.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

from orchard import safeio

CLASSES = ("weights-only", "full-port", "unknown")
PATHS = ("weights-only", "full-port")
CONFIDENCES = ("high", "medium", "low")
CATEGORIES = ("serving", "structural", "unknown")
DECIDERS = ("allow-list", "override")
FIELDS = ("class", "path", "alternative_path", "confidence", "strict", "differences", "overrides")
ABSENT = "<absent>"                          # the template's text for a key a config does not have
AUDIO_MAX_LISTED = 50                        # the template lists at most this many keys per section in `audio.config` (a test pins it to the template's)
TABLE_FILE = Path(__file__).with_name("skills") / "delta-triage-templates" / "audio_config_categories.json"
TABLE_MAX_BYTES = 1 << 20


def _text(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


# ---- the allow-list table (the template's own rule, re-run here) ----------------------------------------------------------------

def load_category_rules(path=TABLE_FILE) -> list[dict]:
    """The rules of the allow-list table at `path`. Raises safeio.SafeReadError for a file that cannot be read or is not an object with a list of
    rules, each an object with a category and a key or a prefix."""
    data = safeio.read_json(path, max_bytes=TABLE_MAX_BYTES, expect=dict)
    rules = data.get("rules")
    if not isinstance(rules, list):
        raise safeio.WrongType("the table has no list of rules")
    for r in rules:
        if (not isinstance(r, dict) or r.get("category") not in CATEGORIES or not (isinstance(r.get("key"), str) or isinstance(r.get("prefix"), str))):
            raise safeio.WrongType("a rule in the table has no category and key or prefix")
    return rules


@functools.lru_cache(maxsize=1)
def category_rules() -> tuple | None:
    """The shipped table as a tuple of rules, or None when it cannot be read (the check then says so and refuses an audio decision)."""
    try:
        return tuple(load_category_rules())
    except (safeio.SafeReadError, OSError):
        return None


def _value_ok(constraint, base, model) -> bool:
    """The template's `_audio_value_ok`: both values satisfy a serving rule's `values` constraint (a list of allowed texts, `positive-int`, `number` or `text`)."""
    def one(v) -> bool:
        if isinstance(constraint, list):
            return isinstance(v, str) and v in constraint
        if (isinstance(v, str) and v == ABSENT) or isinstance(v, bool):
            return False
        if constraint == "positive-int":
            return isinstance(v, int) and v > 0
        if constraint == "number":
            return isinstance(v, (int, float)) and math.isfinite(v)
        if constraint == "text":
            return isinstance(v, str) and bool(v.strip())
        return False
    return one(base) and one(model)


def allow_list_category(key: str, base, model, rules) -> str:
    """The category the allow-list gives a differing key: an exact `key` rule first, then the first matching `prefix` rule, else `unknown`. A serving rule applies
    only when both values satisfy its constraint. This is the template's `audio_category`."""
    rule = next((r for r in rules if r.get("key") == key), None) or next((r for r in rules if r.get("prefix") and key.startswith(r["prefix"])), None)
    if rule is None:
        return "unknown"
    if rule["category"] == "serving" and not _value_ok(rule.get("values"), base, model):
        return "unknown"
    return rule["category"]


def _same(a, b) -> bool:
    """Equal in JSON terms: the same type and value (1, 1.0 and True differ)."""
    return type(a) is type(b) and a == b


# ---- the check ----------------------------------------------------------------------------------------------------------------------

def _is_audio(delta: dict) -> bool:
    facts = delta.get("facts")
    return isinstance(delta.get("audio"), dict) or (isinstance(facts, dict) and facts.get("modality") == "audio")


def decision_problems(delta, *, current_template: bool = False, expected_strict: bool | None = None) -> list[str]:
    """The reasons the decision in `delta` (a parsed delta.json) is malformed or does not support its class. Empty when there is no decision to check.
    `current_template` says the delta carries the stamp of the shipped template, so a missing decision, and missing evidence records, are reasons.
    `expected_strict` is the stage's `audio_strict` (None: not known, so the flag is not compared)."""
    if not isinstance(delta, dict):
        return []
    audio = delta.get("audio")
    if not isinstance(audio, dict) or "decision" not in audio:
        if current_template and _is_audio(delta):
            return ["delta.json is an audio result from the current stage 0 template and has no audio.decision, so its class and path are not backed by a "
                    "recorded decision"]
        return []
    d = audio["decision"]
    if not isinstance(d, dict):
        return ["delta.json audio.decision must be an object"]
    out = []
    for f in FIELDS:
        if f not in d:
            out.append(f"audio.decision has no {f}")
    if out:
        return out
    if d["class"] not in CLASSES:
        out.append(f"audio.decision class must be one of {CLASSES}")
    if d["path"] not in PATHS:
        out.append(f"audio.decision path must be one of {PATHS}")
    if d["alternative_path"] is not None and (d["alternative_path"] not in PATHS or d["alternative_path"] == d["path"]):
        out.append("audio.decision alternative_path must be null or the path that is not the chosen one")
    if d["confidence"] not in CONFIDENCES:
        out.append(f"audio.decision confidence must be one of {CONFIDENCES}")
    if not isinstance(d["strict"], bool):
        out.append("audio.decision strict must be true or false")
    elif expected_strict is not None and d["strict"] != expected_strict:
        out.append(f"audio.decision strict is {d['strict']} but the triage configuration's audio_strict is {expected_strict}, so the decision was made under "
                   "a different rule from the configured one")
    if not isinstance(d["differences"], list):
        out.append("audio.decision differences must be a list")
    if not isinstance(d["overrides"], list):
        out.append("audio.decision overrides must be a list")
    if out:
        return out
    if delta.get("class") != d["class"]:
        out.append(f"audio.decision class {d['class']!r} is not delta.json's class {delta.get('class')!r}")
    if delta.get("path") != d["path"]:
        out.append(f"audio.decision path {d['path']!r} is not delta.json's path {delta.get('path')!r}")
    rules = category_rules()
    if rules is None:
        out.append("the allow-list table (audio_config_categories.json) cannot be read, so the decision's categories cannot be checked")
        rules = ()
    categories, by_key = [], {}
    for i, x in enumerate(d["differences"]):
        if not isinstance(x, dict) or not _text(x.get("key")):
            out.append(f"audio.decision differences[{i}] needs a key")
            continue
        if x.get("category") not in CATEGORIES:
            out.append(f"audio.decision differences[{i}] category must be one of {CATEGORIES}")
            continue
        if x.get("decided_by") not in DECIDERS:
            out.append(f"audio.decision differences[{i}] decided_by must be one of {DECIDERS}")
            continue
        if x["decided_by"] == "override" and not _text(x.get("source")):
            out.append(f"audio.decision differences[{i}] was decided by an override and needs a source")
        if x["key"] in by_key:
            out.append(f"audio.decision lists the difference {x['key']!r} twice")
            continue
        by_key[x["key"]] = x
        categories.append(x["category"])
        if "base" not in x or "model" not in x:
            out.append(f"audio.decision differences[{i}] ({x['key']}) needs both values, base and model")
            continue
        if rules and x["decided_by"] == "allow-list":
            given = allow_list_category(x["key"], x["base"], x["model"], rules)
            if given != x["category"]:
                out.append(f"audio.decision records {x['key']} as {x['category']} (decided by the allow-list), but the allow-list gives {given} for "
                           f"{x['key']} with the recorded values; only an override can move a category")
    out += _override_problems(d, by_key, rules)
    out += _evidence_problems(audio, by_key, current_template)
    path_override = next((o for o in d["overrides"] if isinstance(o, dict) and o.get("kind") == "path"), None)
    if d["class"] == "unknown" and (d["path"] != "full-port" or d["confidence"] != "low"):
        out.append("audio.decision class unknown needs path full-port and confidence low")
    if d["class"] == "weights-only":
        explained = path_override is not None and path_override.get("path") == "weights-only" and _text(path_override.get("source"))
        if not explained:
            wrong = [c for c in categories if c != "serving"]
            if wrong:
                out.append(f"audio.decision class weights-only has {len(wrong)} difference(s) that are not serving settings and no path override to "
                           "weights-only explains them")
            if d["strict"] and categories:
                out.append("audio.decision class weights-only is strict but lists differences and no path override to weights-only explains them")
        out += _hard_evidence_problems(d, audio, current_template)
    return out


def _override_problems(d: dict, by_key: dict, rules) -> list[str]:
    """The overrides listed in the decision: their kinds and fields, one path override at most and to the decision's path, and each category override matching
    the difference it says it moved (and each difference that says an override moved it matching a record)."""
    out, path_overrides, category_keys = [], 0, {}
    for i, o in enumerate(d["overrides"]):
        if not isinstance(o, dict) or o.get("kind") not in ("category", "path"):
            out.append(f"audio.decision overrides[{i}] needs a kind of category or path")
            continue
        if not _text(o.get("source")):
            out.append(f"audio.decision overrides[{i}] needs a source")
        if o["kind"] == "path":
            path_overrides += 1
            if o.get("path") not in PATHS:
                out.append(f"audio.decision overrides[{i}] needs a path of {PATHS}")
            elif o["path"] != d["path"]:
                out.append(f"audio.decision overrides[{i}] moves the path to {o['path']}, but the decision's path is {d['path']}, so the override does not "
                           "explain the decision")
            if o.get("from") not in PATHS:
                out.append(f"audio.decision overrides[{i}] needs a from of {PATHS}")
            continue
        key = o.get("key")
        if not _text(key):
            out.append(f"audio.decision overrides[{i}] needs a key")
            continue
        if o.get("category") not in CATEGORIES:
            out.append(f"audio.decision overrides[{i}] needs a category of {CATEGORIES}")
            continue
        if o.get("from") not in CATEGORIES:
            out.append(f"audio.decision overrides[{i}] needs a from of {CATEGORIES}")
        if key in category_keys:
            out.append(f"audio.decision has two category overrides for {key!r}")
            continue
        category_keys[key] = o
        x = by_key.get(key)
        if x is None:
            out.append(f"audio.decision overrides[{i}] names {key!r}, which is not among the differences")
            continue
        if x["decided_by"] != "override" or x["category"] != o["category"]:
            out.append(f"audio.decision overrides[{i}] moves {key!r} to {o['category']}, but the difference is recorded as {x['category']} decided by "
                       f"{x['decided_by']}")
        if _text(x.get("source")) and x.get("source") != o.get("source"):
            out.append(f"audio.decision overrides[{i}] source is not the source recorded on the difference {key!r}")
        if rules and "base" in x and "model" in x and o.get("from") in CATEGORIES and o["from"] != allow_list_category(key, x["base"], x["model"], rules):
            out.append(f"audio.decision overrides[{i}] has from {o['from']!r} for {key!r}, which is not what the allow-list gave before the override: it gives "
                       f"{allow_list_category(key, x['base'], x['model'], rules)}")
    if path_overrides > 1:
        out.append("audio.decision has more than one path override; at most one path override is allowed")
    for key, x in by_key.items():
        if x["decided_by"] == "override" and key not in category_keys:
            out.append(f"audio.decision records {key!r} as decided by an override, but no category override for it is recorded")
    return out


def _evidence_problems(audio: dict, by_key: dict, current_template: bool) -> list[str]:
    """The decision's differences against `audio.config`, the comparison evidence. Every listed key is in the decision with the same values; the decision names
    no other key unless the evidence was cut at AUDIO_MAX_LISTED keys in a section."""
    config = audio.get("config")
    if not isinstance(config, dict):
        return ["audio.config (the comparison evidence for the decision's differences) is missing"] if current_template else []
    listed, cut = {}, False
    for section in ("top_level_differences", "model_differences"):
        part = config.get(section)
        if not isinstance(part, dict):
            return [f"audio.config {section} (the comparison evidence for the decision's differences) is missing"] if current_template else []
        listed.update(part)
        cut = cut or len(part) >= AUDIO_MAX_LISTED
    out = []
    for key, rec in listed.items():
        x = by_key.get(key)
        if x is None:
            out.append(f"the comparison evidence lists the difference {key!r}, and audio.decision does not")
        elif isinstance(rec, dict) and "base" in x and "model" in x and not (_same(rec.get("base"), x["base"]) and _same(rec.get("model"), x["model"])):
            out.append(f"audio.decision records other values for {key!r} than the comparison evidence does")
    if not cut:
        out += [f"audio.decision names the difference {key!r}, which the comparison evidence does not list" for key in by_key if key not in listed]
    return out


def _hard_evidence_problems(d: dict, audio: dict, current_template: bool) -> list[str]:
    """A weights-only decision needs a comparison with no tensor-level difference, no text-encoder difference and no read problem. No override explains
    these away: they are facts about the files. Each evidence record has to be present when the current template wrote the delta."""
    out = []
    problems = d.get("problems", [])
    if not isinstance(problems, list) or problems:
        out.append("audio.decision class weights-only, but the decision lists problems (evidence that could not be read or compared)")
    tensors = audio.get("tensors")
    if not isinstance(tensors, dict):
        if current_template:
            out.append("audio.tensors (the tensor comparison) is missing, so a weights-only class has no tensor evidence")
    else:
        found = []
        for name in ("only_in_model", "only_in_base", "shape_differences"):
            if not isinstance(tensors.get(name), list) or tensors[name]:
                found.append(name)
        dtypes = tensors.get("dtype_differences")
        if not isinstance(dtypes, list) or any(not isinstance(x, dict) or x.get("tolerated") is not True for x in dtypes):
            found.append("dtype_differences")
        counts = [tensors.get(k) for k in ("model_tensors", "base_tensors", "common")]
        if not all(isinstance(c, int) and not isinstance(c, bool) for c in counts) or not counts[0] == counts[1] == counts[2]:
            found.append("tensor counts")
        if found:
            out.append("audio.decision class weights-only, but the tensor evidence shows a difference (" + ", ".join(found) + "); no override explains a tensor difference")
    encoders = audio.get("text_encoders")
    if not isinstance(encoders, list):
        if current_template:
            out.append("audio.text_encoders (the text-encoder comparison) is missing, so a weights-only class has no text-encoder evidence")
    elif any(not isinstance(r, dict) or r.get("problems") or r.get("differences") for r in encoders):
        out.append("audio.decision class weights-only, but a text encoder differs or could not be compared; no override explains a text-encoder difference")
    return out


# ---- the gate's entry point ------------------------------------------------------------------------------------------------------------

def stage_problems(delta, stage_dir, shipped_sha256) -> list[str]:
    """`decision_problems` for a delta.json in `stage_dir`: the delta is from the current template when its `template_sha256` is `shipped_sha256`, and the strict
    flag is compared with `triage_config.json` in the stage directory. A configuration that cannot be read is a reason only for a current-template audio result;
    a flag that is not a boolean means relaxed, as the template records it."""
    if not isinstance(delta, dict):
        return []
    current = isinstance(shipped_sha256, str) and delta.get("template_sha256") == shipped_sha256
    expected, extra = None, []
    if current and _is_audio(delta):
        try:
            cfg = safeio.read_json(Path(stage_dir) / "triage_config.json", max_bytes=TABLE_MAX_BYTES, expect=dict)
            flag = cfg.get("audio_strict", False)
            expected = flag if isinstance(flag, bool) else False
        except (safeio.SafeReadError, OSError):
            extra = ["triage_config.json cannot be read, so the decision's strict flag cannot be checked against the configuration"]
    return extra + decision_problems(delta, current_template=current, expected_strict=expected)
