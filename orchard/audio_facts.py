# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Three facts about a run that the outcome class does not hold (docs/superpowers/specs/2026-10-15-audio-bringup-design.md, section 3).

    form       adapter | full           an adapter needs a merge before it can be served; full weights do not
    serving    thin-bundle | container | none
                                        how the nearest model is served on a Tenstorrent chip. `none` means no installed
                                        bundle or container package serves it (the spec lists two values; Stable Audio 3 is
                                        the first model where the third case matters)
    modality   text | audio             what a serving check has to measure

Stage 0 decides `form` and `modality` from the files and the supervisor adds `serving` from the installed bundles. The record sits
in stage 0's passing `stage_end`, so a resume uses what the first attempt decided. A ledger without it means a text run on a thin
bundle, which is what every run before this module was.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import PurePosixPath

FORMS = ("adapter", "full")
SERVINGS = ("thin-bundle", "container", "none")
MODALITIES = ("text", "audio")
FIELDS = {"form": FORMS, "serving": SERVINGS, "modality": MODALITIES}
DEFAULT_FACTS = {"form": "full", "serving": "thin-bundle", "modality": "text"}


def facts_problems(facts, required=tuple(FIELDS)) -> list[str]:
    """The reasons a facts record is not usable: not a dict, a required key missing, an unknown key, or a value outside its set.
    Stage 0's template writes `form` and `modality` only (`required`); the supervisor adds `serving` to the ledger's record."""
    if not isinstance(facts, dict):
        return ["facts must be an object with form, serving and modality"]
    out = []
    for key, allowed in FIELDS.items():
        if key not in facts:
            if key in required:
                out.append(f"facts.{key} is missing")
        elif facts[key] not in allowed:   # a list or dict is never equal to a string, so no type check is needed
            out.append(f"facts.{key} must be one of {allowed}, got {facts[key]!r}")
    out += [f"facts has an unknown key {k!r}" for k in facts if k not in FIELDS]
    return out


def serving_of(base_repo, installed) -> str:
    """How the base model is served on this machine: `thin-bundle` when an installed schema 6 bundle serves it, else `container`
    when an installed package does, else `none`. `installed` holds nearest.Installed rows. A revision (`org/name@sha`) on the
    id is ignored."""
    repo = str(base_repo or "").split("@", 1)[0]
    serving = [i for i in installed if i.weights_repo == repo]
    if any(i.schema == "6" for i in serving):
        return "thin-bundle"
    return "container" if serving else "none"


# ---- is this model audio? (the preflight's rule) ---------------------------------------------------
#
# Stage 0 decides modality from the snapshot's files (`detect_audio` in skills/delta-triage-templates/delta_triage.py). The preflight has to
# decide earlier, before anything is downloaded, from what the Hub says. The two must not disagree, so the layout half of the rule is one
# function here and the template's copy is pinned to it by tests/test_audio_preflight.py. (The template runs as a script in a run
# directory and cannot import orchard, so it keeps a copy.)
#
# The rule. A model counts as audio when BOTH hold:
#   1. Tag evidence: the Hub's `pipeline_tag` is `text-to-audio` or `audio-to-audio`. When the Hub gives no usable pipeline tag (absent, or
#      not a string), a tag with one of those names decides instead. A pipeline tag that is present and says anything else (for example
#      `text-to-speech`, which is out of scope) decides alone: the model is not audio for this rule.
#   2. Layout evidence: the Hub's file list must be present and non-empty, and it and the local snapshot's top level (when a snapshot exists) each show `model_config.json`, no `config.json`
#      and at least one `.safetensors` file at the top level. If two lists disagree, the model is not audio.
# Limits. Anything unclear is "not audio", which keeps the old behaviour (the base row asks for an installed bundle): unknown tags, a file list
# that cannot be read, both config files present, a hub lookup that failed. The rule says nothing about whether the files are valid:
# stage 0 reads `model_config.json` and refuses what it cannot parse. Speech models (`text-to-speech`, `automatic-speech-recognition`) are
# not in scope and are served by other paths.

AUDIO_PIPELINE_TAGS = ("text-to-audio", "audio-to-audio")
AUDIO_CONFIG, TEXT_CONFIG = "model_config.json", "config.json"


def _strings(items) -> list[str]:
    """The strings of a list. Anything else (not a list, or a list with numbers, None or objects in it) contributes nothing."""
    return [i for i in items if isinstance(i, str)] if isinstance(items, list) else []


def audio_layout(names) -> bool:
    """True when the top level of a file list is the stable-audio-tools layout: `model_config.json`, no `config.json`, and a top-level
    `.safetensors` file. `names` is a list of repository-relative paths; non-strings are ignored."""
    root = {n for n in _strings(names) if "/" not in n}
    return AUDIO_CONFIG in root and TEXT_CONFIG not in root and any(n.endswith(".safetensors") for n in root)


def audio_tag(pipeline_tag, tags) -> bool:
    """Tag evidence (rule 1 above). Only the exact strings in AUDIO_PIPELINE_TAGS count."""
    if isinstance(pipeline_tag, str) and pipeline_tag:
        return pipeline_tag in AUDIO_PIPELINE_TAGS
    return any(t in AUDIO_PIPELINE_TAGS for t in _strings(tags))


def audio_evidence(pipeline_tag, tags, hub_files=None, *other_lists) -> str | None:
    """A sentence saying why the model counts as audio, or None. `hub_files` is the Hub's file list and is required: an empty, missing or
    malformed list gives no layout evidence, and a local snapshot alone is not enough. Each of `other_lists` (a local snapshot's top
    level) is either None (that source is absent and is skipped) or a list that must show the layout too; a present but empty list fails."""
    if not audio_tag(pipeline_tag, tags):
        return None
    if not isinstance(hub_files, list) or not hub_files or not audio_layout(hub_files):
        return None
    if not all(isinstance(n, list) and audio_layout(n) for n in other_lists if n is not None):
        return None
    if isinstance(pipeline_tag, str) and pipeline_tag:
        how = f"pipeline tag {pipeline_tag}"
    else:
        how = f"tag {next(t for t in _strings(tags) if t in AUDIO_PIPELINE_TAGS)}"
    return f"{how}; {AUDIO_CONFIG} and no {TEXT_CONFIG} at the repository root"


# ---- the cached config (what stage 0's detector requires) ---------------------------------------------------
#
# `detect_audio` in the stage 0 template accepts a snapshot only when `model_config.json` is a JSON object of at most 1 MiB, with no repeated
# key, no NaN or infinity, no integer of more than 30 digits and no nesting deeper than 32, a string `model_type` and an object `model`. The
# preflight reads the cached config with the same limits (a bounded read), so a snapshot the preflight calls audio is one stage 0 calls audio
# (tests/test_audio_preflight_review3.py runs both on the same files). The template keeps its own copy of these limits.

AUDIO_JSON_MAX = 1 << 20
AUDIO_MAX_DEPTH = 32
AUDIO_MAX_INT_DIGITS = 30


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name):
    raise ValueError(f"{name} is not allowed")


def _int(text):
    if len(text.lstrip("-")) > AUDIO_MAX_INT_DIGITS:
        raise ValueError("an integer has too many digits")
    return int(text)


def _float(text):
    """A float that fits. `1e999` is valid JSON text that Python reads as infinity, and `parse_constant` never sees it (it handles the words
    NaN and Infinity only), so the value is checked here."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"the number {text} is not finite")
    return value


def _depth(data) -> int:
    deepest, stack = 0, [(data, 1)]
    while stack:
        item, d = stack.pop()
        kids = list(item.values()) if isinstance(item, dict) else item if isinstance(item, list) else None
        if kids is None:
            continue
        deepest = max(deepest, d)
        if deepest > AUDIO_MAX_DEPTH:
            return deepest
        stack.extend((k, d + 1) for k in kids)
    return deepest


def audio_config_problem(raw) -> tuple[dict | None, str | None]:
    """(the parsed config, None) or (None, why not) for the bytes of a `model_config.json`. `raw` longer than AUDIO_JSON_MAX is refused
    without parsing. The config must be an object with a string `model_type` and an object `model`."""
    if not isinstance(raw, (bytes, bytearray)):
        return None, "model_config.json could not be read"
    if len(raw) > AUDIO_JSON_MAX:
        return None, f"model_config.json is larger than {AUDIO_JSON_MAX} bytes"
    try:
        cfg = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_no_constant, parse_int=_int,
                         parse_float=_float)
    except (ValueError, RecursionError) as exc:             # UnicodeDecodeError is a ValueError
        return None, f"model_config.json is not readable JSON: {exc}"
    if not isinstance(cfg, dict):
        return None, "model_config.json is not a JSON object"
    if _depth(cfg) > AUDIO_MAX_DEPTH:
        return None, f"model_config.json is nested more than {AUDIO_MAX_DEPTH} levels deep"
    if not isinstance(cfg.get("model_type"), str) or not isinstance(cfg.get("model"), dict):
        return None, "model_config.json has no string model_type and object model"
    return cfg, None


def same_repository(a, b) -> bool:
    """True when two repository ids are the same text: `owner/name` with a non-empty owner and name, equal in case, owner and name. There is no
    prefix, substring, case-folded or revision match (`org/name@rev` is a different text and is not the repository id). The stage 0 template
    compares ids the same way (`repo in (model_repo, base_repo)`)."""
    if not (isinstance(a, str) and isinstance(b, str)) or a != b:
        return False
    owner, _, name = a.partition("/")
    return bool(owner) and bool(name) and "/" not in name


def component_subfolders(cfg, own_repo: str, partners=()) -> list[str]:
    """Directories (relative, with no `..`) that the config's conditioners load from in one of the two repositories. Stage 0 compares a text
    encoder's `config.json` and tensors when its `repo_id` names one of the two repositories (the template tests `repo_id in (model_repo,
    base_repo)`); a snapshot is complete only when those directories' configs are present. `repo_id` has to equal `own_repo` or one of
    `partners` exactly (`same_repository`: case, owner and name, no prefix or substring). A `repo_id` that is anything else is loaded from
    elsewhere and is not required here."""
    repos = [r for r in (own_repo, *partners) if same_repository(r, r)]
    out = []
    try:
        confs = cfg["model"]["conditioning"]["configs"]
    except (KeyError, TypeError):
        return out
    for c in confs if isinstance(confs, list) else []:
        inner = c.get("config") if isinstance(c, dict) else None
        if not isinstance(inner, dict):
            continue
        sub, repo = inner.get("subfolder"), inner.get("repo_id")
        if not (isinstance(sub, str) and sub and any(same_repository(repo, r) for r in repos)):
            continue
        parts = PurePosixPath(sub).parts
        if os.path.isabs(sub) or ".." in parts or "\\" in sub:
            continue
        out.append(sub)
    return out
