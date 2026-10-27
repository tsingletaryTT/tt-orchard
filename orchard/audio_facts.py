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
