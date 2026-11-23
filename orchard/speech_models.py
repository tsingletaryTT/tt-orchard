# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speech models: what the preflight recognises and what it says about them.

A speech model takes speech in and gives text (`speech-to-text`), or takes text and gives speech (`text-to-speech`). The harness's stages cover
text models, scoring models (orchard/model_kind.py) and the stable-audio-tools layout (orchard/audio_facts.py). They have no stage for either
speech task. Before this module a Voxtral repository went down the text path: the Mini model's base row warned that its card names no base
model, and the TTS model's base row blocked on `mistralai/Ministral-3-3B-Base-2512`, which no bundle serves and which is not a model to
compare it with. Neither message said that the model is a speech model, which bundle serves it, or how to check a server that runs it.

The rule for recognising a model, in this order:

  1. A Hub `pipeline_tag` that is a non-empty string decides alone: `automatic-speech-recognition` is speech-to-text, `text-to-speech` is
     text-to-speech, anything else is not a speech model for this rule.
  2. With no usable pipeline tag, a tag in the tag list with one of those two names decides.
  3. With neither, a listed architecture in ARCHITECTURE_TASKS decides. The upstream Voxtral-Mini card has no pipeline tag, so its architecture
     is the only evidence.

`text-to-audio` and `audio-to-audio` are the stable-audio-tools tags and are not speech here. Anything unclear is "not speech", which keeps the
earlier behaviour.

PUBLISHED lists the bundles known on 2026-11-23, from their cards and the manifests the bundles ship (docs/run-logs/2026-11-23-voxtral-speech-
models.md). It is a short table that a person updates when a bundle appears. It does not ask the Hub, so it can be out of date; the row says
"known" and not "all".
"""
from __future__ import annotations

from dataclasses import dataclass

TASKS = ("speech-to-text", "text-to-speech")
PIPELINE_TASKS = {"automatic-speech-recognition": "speech-to-text", "text-to-speech": "text-to-speech"}
# Architecture names that mean a task when the card gives no usable tag. Only names read from a real upstream config are listed.
ARCHITECTURE_TASKS = {"VoxtralForConditionalGeneration": "speech-to-text"}


@dataclass(frozen=True)
class Bundle:
    """A published tt-model bundle for a speech model. Every field is a statement of its card or manifest, as of KNOWN_ON."""
    task: str
    upstream: str                     # the model whose weights it serves
    bundle: str                       # the Hub repository of the bundle
    endpoints: tuple[str, ...]        # what the server answers (the card's "Interface" row)
    chips: int                        # chips the card says it needs
    hardware: str                     # the card's name for that configuration
    license: str                      # the card's licence id
    non_commercial: bool              # the licence forbids commercial use
    quality: str                      # how the card measured quality, in a sentence


KNOWN_ON = "2026-11-23"

PUBLISHED = (
    Bundle(task="speech-to-text", upstream="mistralai/Voxtral-Mini-3B-2507", bundle="apande-TT/mistralai-voxtral-mini-3b-2507-tt",
           endpoints=("POST /v1/audio/transcriptions", "POST /v1/chat/completions"), chips=1, hardware="Blackhole p150",
           license="apache-2.0", non_commercial=False,
           quality="the card compares the whole greedy output token by token with transformers 4.56.0 on CPU"),
    Bundle(task="text-to-speech", upstream="mistralai/Voxtral-4B-TTS-2603", bundle="apande-TT/mistralai-voxtral-4b-tts-2603-tt",
           endpoints=("POST /v1/audio/speech",), chips=1, hardware="Blackhole p150",
           license="cc-by-nc-4.0", non_commercial=True,
           quality="the card transcribes the speech with Whisper large-v3-turbo for word error rate and scores it with UTMOS22"),
)


def _strings(items) -> list[str]:
    return [i for i in items if isinstance(i, str)] if isinstance(items, list) else []


def task_of(pipeline_tag, tags, architectures) -> str | None:
    """`speech-to-text`, `text-to-speech` or None, by the rule in the module docstring. Never raises: a wrong shape is "not speech"."""
    if isinstance(pipeline_tag, str) and pipeline_tag:
        return PIPELINE_TASKS.get(pipeline_tag)
    for tag in _strings(tags):
        if tag in PIPELINE_TASKS:
            return PIPELINE_TASKS[tag]
    for arch in _strings(architectures):
        if arch in ARCHITECTURE_TASKS:
            return ARCHITECTURE_TASKS[arch]
    return None


def bundle_for(repo) -> Bundle | None:
    """The known bundle that serves `repo`, where `repo` is the upstream model or the bundle's own repository. A revision (`org/name@sha`) is
    ignored. The match is on the exact text: a different case is a different repository (the Hub is case-insensitive about names, but the
    harness compares ids the same way everywhere else, orchard/audio_facts.py `same_repository`)."""
    if not isinstance(repo, str):
        return None
    repo = repo.split("@", 1)[0]
    for b in PUBLISHED:
        if repo in (b.upstream, b.bundle):
            return b
    return None


def row_text(model_id: str, task: str, bundle: Bundle | None) -> str:
    """The sentence block of the preflight's base row for a speech model. Plain statements: what the model is, why no run starts, what serves
    it, and which command checks a server."""
    head = (f"{model_id} is a {task} model. The harness has no speech stage (its stages cover text, scoring and stable-audio models), "
            "so it does not start a run for it.")
    check = f"`tt-orchard speech-check URL --task {task}` checks a server that runs it (see `tt-orchard speech-check --help`)."
    if bundle is None:
        return (f"{head} No published bundle is known for it as of {KNOWN_ON}: `tt-model search` lists the bundles on the Hub. "
                f"Any server that answers the OpenAI speech endpoints can be checked. {check}")
    chips = "one chip" if bundle.chips == 1 else f"{bundle.chips} chips"
    lic = f"Its licence is {bundle.license}" + (" (non-commercial use only)." if bundle.non_commercial else ".")
    return (f"{head} A published bundle serves it on {chips} ({bundle.hardware}): {bundle.bundle}. It answers "
            f"{' and '.join(bundle.endpoints)}. Pull it with `tt-model pull {bundle.bundle}` and start it with `tt-model serve {bundle.bundle}` "
            f"under a gozer lease. {lic} {check}")
