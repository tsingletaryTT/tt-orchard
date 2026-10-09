# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The machine paths a stage skill may name, and the rendering of their placeholders.

The skills under orchard/skills are written once and run on any machine. Where a skill needs a
path on the machine (the tt-orchard checkout to copy templates from, the operator's Hugging Face
cache, the tensor-cache disk), it writes a placeholder such as `{{CACHE_ROOT}}`. The context
builder (orchard/context.py) replaces each placeholder with this run's value before the agent
sees the skill.

The values:

    ORCHARD_DIR     the tt-orchard checkout, found from this file's location
    HF_HOME         --hf-home, else $HF_HOME, else <operator home>/.cache/huggingface
    OPERATOR_HOME   --operator-home, else the operator's home from the passwd entry
    TT_MODEL_ROOT   <operator home>/.cache/tt-model/models, where tt-model installs bundles
    CACHE_ROOT      --cache-root, else <parent of the run directory>/cache. Each configuration's
                    tensor cache goes under it, as <CACHE_ROOT>/<slug>/<N>chip-<package>/tt_cache

The supervisor records the values in the ledger's run_start entry. A resumed run uses the
recorded values, so a skill names the same paths on every attempt, and a resume that passes a
different --cache-root, --hf-home or --operator-home is refused. A run whose run_start predates
this record gets one "run paths recorded" decision at its next start, which then holds.

Rendering fails closed. Any `{{...}}` that is not one of the five names raises
UnknownPlaceholder, which names the placeholder and the skill file. A typo in a skill therefore
stops the stage with a clear message, and the agent never sees a half-rendered path.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# Any double-brace span is checked. Only the exact names below are accepted.
_SPAN = re.compile(r"\{\{(.*?)\}\}", re.S)

# Placeholder name -> RunPaths field.
PLACEHOLDERS = {
    "ORCHARD_DIR": "orchard_dir",
    "HF_HOME": "hf_home",
    "OPERATOR_HOME": "operator_home",
    "TT_MODEL_ROOT": "tt_model_root",
    "CACHE_ROOT": "cache_root",
}

# The tt-orchard checkout: the directory that holds the `orchard` package.
ORCHARD_DIR = Path(__file__).resolve().parent.parent


class UnknownPlaceholder(ValueError):
    """A skill holds a `{{...}}` this module does not know, or no run paths were given."""


def absolute_path(path) -> str:
    """`~` expanded and made absolute against the current directory. Symlinks are kept."""
    return os.path.abspath(os.path.expanduser(str(path)))


@dataclass(frozen=True)
class RunPaths:
    """One run's machine paths. Every value is an absolute path, stored as a string so that the
    ledger can record it as JSON."""
    orchard_dir: str
    hf_home: str
    operator_home: str
    tt_model_root: str
    cache_root: str

    @classmethod
    def resolve(cls, run_dir, *, home, environ=None, cache_root=None, hf_home=None,
                operator_home=None, tt_model_root=None) -> "RunPaths":
        """The values for a new run. `home` is the operator's home from the passwd entry; an
        explicit `operator_home` replaces it. `environ` supplies $HF_HOME (default os.environ)."""
        env = os.environ if environ is None else environ
        op_home = absolute_path(operator_home if operator_home is not None else home)
        if hf_home is None:
            hf_home = env.get("HF_HOME") or os.path.join(op_home, ".cache", "huggingface")
        if cache_root is None:
            cache_root = Path(absolute_path(run_dir)).parent / "cache"
        return cls(orchard_dir=str(ORCHARD_DIR), hf_home=absolute_path(hf_home), operator_home=op_home,
                   tt_model_root=absolute_path(tt_model_root) if tt_model_root is not None
                   else os.path.join(op_home, ".cache", "tt-model", "models"),
                   cache_root=absolute_path(cache_root))

    def record(self) -> dict:
        """The form the ledger stores."""
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> "RunPaths":
        return cls(**{field: record[field] for field in PLACEHOLDERS.values()})

    def placeholders(self) -> dict:
        """Placeholder name -> value."""
        return {name: getattr(self, field) for name, field in PLACEHOLDERS.items()}


# The decision a resumed run writes when its run_start entry predates the "paths" field.
PATHS_RECORDED = "run paths recorded"


def recorded_paths(entries: list[dict]) -> dict | None:
    """The paths this run recorded: run_start's "paths", else the latest "run paths recorded"
    decision (written once for a run whose run_start predates the field), else None."""
    found = None
    for e in entries:
        d = e.get("data") or {}
        if e["event"] == "run_start" and d.get("paths"):
            found = d["paths"]
        elif e["event"] == "decision" and d.get("decision") == PATHS_RECORDED:
            found = d["paths"]
    return found


def render(text: str, paths: RunPaths | None, *, source: str = "") -> str:
    """`text` with every placeholder replaced by its value in `paths`.

    Raises UnknownPlaceholder for any `{{...}}` that is not a known name, and for any placeholder
    at all when `paths` is None. `source` (the skill file) goes into the message."""
    where = f" in {source}" if source else ""
    spans = _SPAN.findall(text)
    if not spans:
        return text
    if paths is None:
        raise UnknownPlaceholder(f"{{{{{spans[0]}}}}}{where} cannot be filled: no run paths were given")
    values = paths.placeholders()
    unknown = sorted({s for s in spans if s not in values})
    if unknown:
        names = ", ".join(f"{{{{{s}}}}}" for s in unknown)
        raise UnknownPlaceholder(f"unknown placeholder {names}{where}. The known placeholders are "
                                 + ", ".join(f"{{{{{n}}}}}" for n in values))
    return _SPAN.sub(lambda m: values[m.group(1)], text)
