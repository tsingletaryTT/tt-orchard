# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What a model produces: its kind (issues 378 and 380, docs/superpowers/specs/2026-11-02-scoring-model-check-design.md section 2).

    generative   text from tokens: every model before the scoring check
    reranker     a score for a (query, document) prompt: a Qwen3-style yes/no head read at the last position
    embedding    a vector for an input: the last position's final-normed hidden state

The kind is a fact about how the model is used. It is separate from the outcome class (orchard/classes.py): a reranker with the nearest
model's tensors is `weights-only` and one with a different encoder is `full-port`. The kind decides which stage 1 reference is recorded
and which stage 2 check runs (orchard/scoring_reference.py, orchard/scoring_check.py).

`classify(model_dir, override)` reads, in this order:

1. the operator's override (`tt-orchard run --model-kind`);
2. the model card's `pipeline_tag` (README.md front matter): `text-ranking` is a reranker, `feature-extraction` and
   `sentence-similarity` are embedding models;
3. the architecture names in config.json: a name that contains `Embedding` is an embedding model;
4. otherwise generative.

Qwen3-Reranker's config names `Qwen3ForCausalLM`, because the model is a causal language model read at two rows of its head, so only the
card or the operator can say it is a reranker. The record names the source of the answer, so a person can see which of the four decided.

It also records whether the model has an output head to read. A config that lists architectures, none of which reads a vocabulary head,
describes a bare backbone (`has_output_head` false). The hazard scan uses that to leave out the hazards of a full vocabulary projection
(orchard/hazards.py), and stage 0's tensor comparison does not count the nearest model's lm_head as missing (delta_triage.py,
`has_no_output_head`; the two use the same architecture rule).
"""
from __future__ import annotations

import re
from pathlib import Path

from orchard import model_config as mc, safeio

KINDS = ("generative", "reranker", "embedding")
SCORING_KINDS = ("reranker", "embedding")
PIPELINE_TAGS = {"text-ranking": "reranker", "feature-extraction": "embedding", "sentence-similarity": "embedding"}
# The same rule as delta_triage.py's HEAD_ARCH (a test compares the two): architecture names whose models read a vocabulary head.
HEAD_ARCH = re.compile(r"(ForCausalLM|ForConditionalGeneration|LMHeadModel|ForMaskedLM|ForSeq2SeqLM|ForPreTraining)$")
README_MAX_BYTES = 1 << 20          # a model card is a few tens of kilobytes; a megabyte is far beyond any real one
FRONT_MATTER_LINES = 200            # the front matter of a model card is a few lines
_TAG = re.compile(r"^pipeline_tag:\s*['\"]?([A-Za-z0-9_.-]+)['\"]?\s*$")


class ModelKindError(ValueError):
    """An override that is not a kind."""


def pipeline_tag(model_dir) -> str | None:
    """The `pipeline_tag` in the front matter of README.md, or None (no card, an unreadable card, no front matter, no tag). Reads through
    `safeio` with a size cap; a Hugging Face cache snapshot's README.md is a link into blobs/, so the path is resolved first."""
    try:
        raw = safeio.read_bytes(safeio.resolved(Path(model_dir) / "README.md"), max_bytes=README_MAX_BYTES)
    except (safeio.SafeReadError, OSError, ValueError):
        return None
    lines = raw.decode("utf-8", "replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:FRONT_MATTER_LINES]:
        if line.rstrip() == "---":
            return None
        m = _TAG.match(line.rstrip())          # not stripped at the start: an indented `pipeline_tag` belongs to a nested mapping
        if m:
            return m.group(1)
    return None


def _architectures(config) -> list[str] | None:
    archs = config.get("architectures") if isinstance(config, dict) else None
    if isinstance(archs, list) and archs and all(isinstance(a, str) for a in archs):
        return archs
    return None


def has_output_head(config) -> bool | None:
    """True when an architecture of the config reads a vocabulary head, False when the config lists architectures and none does (a bare
    backbone), None when it lists none that can be read. A model with a head architecture and tied embeddings has a head."""
    archs = _architectures(config)
    if archs is None:
        return None
    return any(HEAD_ARCH.search(a) for a in archs)


def classify(model_dir, override: str | None = None) -> dict:
    """The kind of the model in `model_dir` and how it was decided: {"kind", "source", "pipeline_tag", "architectures", "has_output_head"}.
    `source` is "operator", "model card", "architecture" or "default". An override that is not a kind raises ModelKindError. An unreadable
    config or card is not an error: it gives the default with the reason in `source`."""
    if override is not None and override not in KINDS:
        raise ModelKindError(f"model kind must be one of {', '.join(KINDS)}, got {override!r}")
    config = tag = None
    if model_dir:
        try:
            config = mc.read_model_config(safeio.resolved(Path(model_dir) / "config.json"))
        except (mc.ModelConfigError, OSError, ValueError):
            config = None
        tag = pipeline_tag(model_dir)
    archs = _architectures(config)
    out = {"pipeline_tag": tag, "architectures": archs, "has_output_head": has_output_head(config)}
    if override is not None:
        return {"kind": override, "source": "operator", **out}
    if tag in PIPELINE_TAGS:
        return {"kind": PIPELINE_TAGS[tag], "source": "model card", **out}
    if archs and any("Embedding" in a for a in archs):
        return {"kind": "embedding", "source": "architecture", **out}
    return {"kind": "generative", "source": "default", **out}
