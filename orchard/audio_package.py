# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package builder for a finished Stable Audio 3 small-music run (issue 169): stage a local package, render its card from the run's
measured numbers, and check the card and the package against the license the model ships under.

What it stages (`stage_package`), under `<run>/stages/7/package/<name>/`:

    README.md              the card, rendered from the run (`render_card`)
    LICENSE.md             a byte-for-byte copy of the model's license agreement, read from the pinned snapshot
    NOTICE                 the attribution sentence the agreement requires, and nothing the agreement did not ask for
    tt-model.yaml          a v5.1 container manifest of kind `http-server` (draft: it names a tt-metal checkout it cannot supply)
    run.sh                 a direct launcher for the job server and the page (the way that is runnable without tt-model)
    requirements.txt       what the host Python environment needs besides ttnn and the model library
    runtime/               the chip runtime files, copied unmodified, laid out as in this repository so their relative paths resolve
    demo/                  the job server, the client and the Gradio page, copied from the separate demo repository
    docs/running.md        how to start it, with the attribution line
    evidence/              the run's agreement result and the benchmark file the card's numbers come from
    package-record.json    the sha256 of every file, what each source was, and the only edit made to a copied file

No weights are staged. The weights are gated: each user downloads them with their own Hugging Face account after accepting the terms.

`package_problems` is the gate. It reads the directory as it is on disk, so a card edited after it was rendered is checked as edited.
`card_problems` checks the card text; the license elements are each one check, so a card that lacks one fails with the missing element named.

What the gate does not do: judge whether prose is true, catch a claim worded in a way its patterns do not list, decide whether using the
model's name to identify the model is within the license's trademark clause (that is a question for the operator and counsel), or decide
whether Tenstorrent may publish under a Community License at all (the run log says why that needs the operator's decision).

The module reads JSON only through `orchard/safeio`. It writes nothing outside the package directory and the record beside it, and it
never runs a publish command: `publish_commands` returns text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from orchard import safeio
from orchard.card_use import PERF_IN_PROSE
from orchard.scrub import scrub_package
from orchard.stages import inside

MODEL_ID = "stabilityai/stable-audio-3-small-music"
MODEL_KEY = "small-music"                  # the name the chip runtime and the audio result use
PACKAGE_NAME = "stable-audio-3-small-music-blackhole"
LICENSE_NAME = "Stability AI Community License"
LICENSE_FRONT_NAME = "stability-ai-community-license"
GEMMA_LINK = "https://ai.google.dev/gemma/terms"
POWERED_BY = "Powered by Stability AI"
REGISTER_LINK = "https://stability.ai/community-license"
AUP_LINK = "https://stability.ai/use-policy"
LIBRARY_LINK = "https://github.com/Stability-AI/stable-audio-3"

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 120                          # the model's own maximum clip length, and the longest a sweep may record
MIN_REPEATS = 3                            # docs/benchmark-rules.md: measured runs for the label "benchmark"
MIN_WARMUP = 1                             # and discarded warm-up runs in the same process
REVISION = re.compile(r"[0-9a-f]{40}")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".tensorbin", ".npz")
WEIGHT_MAX_BYTES = 20 * 1024 * 1024        # no file in the package is larger than this; the clips and arrays stay in the run

LABEL_BENCHMARK = "benchmark"
LABEL_SINGLE = "single sample"
LABEL_UNWARMED = "unwarmed repeats"
LABEL_AGREEMENT = "measured (provisional bars)"
LABELS = (LABEL_BENCHMARK, LABEL_SINGLE, LABEL_UNWARMED, LABEL_AGREEMENT, "TODO")

# The license's own sentences, cut from the agreement text by these markers. The card quotes them and the gate looks them up again in the
# package's own LICENSE.md, so neither a changed agreement nor a changed quotation passes.
QUOTES = (("research", "Subject to the terms of this Agreement, Stability AI grants You", "for any Research or Non-Commercial Purpose."),
          ("register", "If You are using or distributing the Stability AI Materials for a Commercial Purpose", "stability.ai/community-license)."),
          ("threshold", "If at any time You or Your Affiliate(s)", "in its sole discretion."))
ATTRIBUTION_START = "This Stability AI Model is licensed under"
ATTRIBUTION_END = "All Rights Reserved"
DEMO_ANCHOR = 'PAGE_NOTE = ("Describe the music'
DEMO_PATCHED = f'PAGE_NOTE = ("{POWERED_BY}. Describe the music'
DEMO_FILES = (("demo/audio_client.py", "audio_client.py"), ("demo/audio_app.py", "audio_app.py"), ("serve/serve_audio.py", "serve_audio.py"))
RUNTIME_FILES = (("skills/audio-chip-templates/chip_generate.py", "orchard/skills/audio-chip-templates/chip_generate.py"),
                 ("skills/audio-chip-templates/check_revision.py", "orchard/skills/audio-chip-templates/check_revision.py"),
                 ("skills/audio-chip-templates/chip_dit.py", "orchard/skills/audio-chip-templates/chip_dit.py"),
                 ("skills/audio-reference-templates/make_reference.py", "orchard/skills/audio-reference-templates/make_reference.py"),
                 ("audio_reference.py", "orchard/audio_reference.py"),
                 ("pinned_files.py", "orchard/pinned_files.py"))
CHECK_FILE = "runtime/orchard/skills/audio-chip-templates/check_revision.py"
REQUIRED_FILES = (("README.md", "LICENSE.md", "NOTICE", "tt-model.yaml", "run.sh", "requirements.txt", "docs/running.md", "demo/requirements.txt",
                   "runtime/orchard/__init__.py", CHECK_FILE, "evidence/audio_result.json", "evidence/bench-chip.json", "package-record.json")
                  + tuple(f"demo/{name}" for _, name in DEMO_FILES) + tuple(f"runtime/{dest}" for _, dest in RUNTIME_FILES))
RUNTIME_INIT = '"""A stand-in package so the copied chip runtime files import `orchard.audio_reference` from this directory."""\n'
RECORD_FILE = "package-record.json"
BENCH_EVIDENCE = "evidence/bench-chip.json"
RESULT_EVIDENCE = "evidence/audio_result.json"
NUMBERS_HEAD = "## Numbers"
NOT_MEASURED_HEAD = "## Not measured"

# What the card must not claim. Each pattern is one kind of overclaim; the gate names the match.
COMMERCIAL_CLAIMS = (r"(?<!non-)(?<!non )commercial use is (?:allowed|permitted|free|unrestricted|fine)", r"free for commercial", r"free to use commercially",
                     r"open[- ]source (?:model|weights|license|licen[cs]e)", r"openly licen[cs]ed", r"permissive", r"production[- ]ready",
                     r"\bfor (?:your )?customers\b", r"\bmonetiz", r"\bresell", r"no (?:restrictions|strings)", r"use it for anything")
QUALITY_CLAIMS = (r"high[- ]quality", r"studio[- ]quality", r"\bsounds? (?:great|good|natural|realistic|professional)", r"state[- ]of[- ]the[- ]art",
                  r"\b(?:excellent|impressive|stunning|realistic|professional|best)\b", r"human[- ]like", r"indistinguishable")
MARK_CLAIMS = (r"\b(?:official|officially|endorsed|sponsored|certified|partnered|partnership)\b", r"!\[[^\]]*\]\([^)]*(?:stability|stable[-_]audio)[^)]*\)",
               r"<img\b")
OPEN_LICENSES = ("apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0", "openrail", "openrail++", "gpl-3.0", "mpl-2.0")


class AudioPackageError(Exception):
    """The run, the benchmark file or the package cannot be used; the message says why."""


# ---- reading -----------------------------------------------------------------------------------------------------

def read_json_file(path, what: str, *, expect=dict):
    """JSON from a file that something else wrote, through safeio. Any failure is an AudioPackageError that names `what`."""
    try:
        return safeio.read_json(path, max_bytes=MAX_JSON_BYTES, expect=expect)
    except safeio.SafeReadError as exc:
        raise AudioPackageError(f"{what} is missing or unreadable: {exc.reason if hasattr(exc, 'reason') else exc}") from None


def _file_text(path, what: str, max_bytes: int = 2 * 1024 * 1024) -> str:
    try:
        return safeio.read_bytes(path, max_bytes=max_bytes).decode("utf-8")
    except safeio.SafeReadError as exc:
        raise AudioPackageError(f"{what} is missing or unreadable: {getattr(exc, 'reason', exc)}") from None
    except UnicodeDecodeError:
        raise AudioPackageError(f"{what} is not UTF-8 text") from None


def _file_bytes(path, what: str, max_bytes: int = 2 * 1024 * 1024) -> bytes:
    try:
        return safeio.read_bytes(path, max_bytes=max_bytes)
    except safeio.SafeReadError as exc:
        raise AudioPackageError(f"{what} is missing or unreadable: {getattr(exc, 'reason', exc)}") from None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _num(x, what: str):
    try:
        return safeio.finite_number(x, lo=0, what=what)
    except safeio.SafeReadError:
        raise AudioPackageError(f"{what} is not a finite number of 0 or more") from None


def _get(d, *keys, what: str):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise AudioPackageError(f"{what}: {'.'.join(keys)} is missing")
        cur = cur[k]
    return cur


# ---- the benchmark file -------------------------------------------------------------------------------------------

def label_for(runs: int, warmup: int) -> str:
    """The label docs/benchmark-rules.md gives a set of timings: 3 or more measured runs after 1 or more warm-up runs is a benchmark.
    Recomputed here from the counts. A label written in a file is never believed."""
    if runs >= MIN_REPEATS and warmup >= MIN_WARMUP:
        return LABEL_BENCHMARK
    if runs >= MIN_REPEATS:
        return LABEL_UNWARMED
    return LABEL_SINGLE


@dataclass(frozen=True)
class SizeRow:
    seconds: float
    label: str                              # recomputed from the counts
    warmup: int
    runs: int
    parts: dict                             # part name -> {"min", "median", "max"}, recomputed from the runs


@dataclass(frozen=True)
class Facts:
    model_id: str
    revision: str
    library_commit: str
    settings: dict
    device: str
    agreement: dict                         # chip against the CPU reference: the numbers the card shows
    floor: dict                             # the environment floor: two CPU environments against each other
    bars: str
    sizes: tuple[SizeRow, ...]
    load: dict
    bench_file: str                         # package-relative path of the benchmark copy
    result_file: str


MAIN_PART = "generate_and_decode_s"
CARD_PARTS = (("generate_and_decode_s", "generate and decode"), ("chip_transformer_call_s", "diffusion transformer calls"),
              ("conditioning_s", "T5Gemma encoder on the host"))


def bench_sizes(bench: dict, settings: dict) -> tuple[tuple[SizeRow, ...], dict]:
    """The sizes of an `audio_bench.py` sweep file, with every label and spread recomputed from the runs, and the once-per-process load.

    Refuses a sweep of another model or device, one made with other settings than the run's, a size over the model's maximum, a label the
    counts do not support, and a spread that its runs do not give."""
    what = "the benchmark file"
    if bench.get("model") != MODEL_KEY:
        raise AudioPackageError(f"{what} is for model {bench.get('model')!r}; this builder packages {MODEL_KEY!r}")
    if bench.get("device") != "chip":
        raise AudioPackageError(f"{what} is for device {bench.get('device')!r}; the package's numbers are the chip's")
    got = bench.get("settings")
    if not isinstance(got, dict):
        raise AudioPackageError(f"{what} has no settings")
    for key in ("prompt", "steps", "cfg_scale", "sampler", "seed", "sample_size"):
        if got.get(key) != settings.get(key):
            raise AudioPackageError(f"{what} was made with {key} {got.get(key)!r} and the run used {settings.get(key)!r}; the numbers are not comparable")
    sizes = bench.get("sizes")
    if not isinstance(sizes, list) or not sizes:
        raise AudioPackageError(f"{what} has no sizes")
    rows = []
    for entry in sizes:
        if not isinstance(entry, dict):
            raise AudioPackageError(f"{what} has a size that is not an object")
        seconds = _num(entry.get("seconds"), "a size's seconds")
        if not 0 < seconds <= MAX_SECONDS:
            raise AudioPackageError(f"{what} has a size of {seconds:g} s; a size is above 0 and at most {MAX_SECONDS} s")
        runs = entry.get("runs")
        if not isinstance(runs, list) or not runs:
            raise AudioPackageError(f"{what}: the {seconds:g} s size has no measured runs")
        try:
            warmup = safeio.whole_number(entry.get("warmup_runs"), lo=0, hi=1000, what="warmup_runs")
        except safeio.SafeReadError:
            raise AudioPackageError(f"{what}: warmup_runs at {seconds:g} s is not a whole number from 0 to 1000") from None
        label = label_for(len(runs), warmup)
        if entry.get("label") != label:
            raise AudioPackageError(f"{what}: the {seconds:g} s size is labelled {entry.get('label')!r} and its {len(runs)} runs after "
                                    f"{warmup} warm-up run(s) make {label!r}")
        parts: dict = {}
        for name, _ in CARD_PARTS:
            vals = []
            for r in runs:
                t = r.get("timings") if isinstance(r, dict) else None
                if not isinstance(t, dict) or name not in t:
                    raise AudioPackageError(f"{what}: a run at {seconds:g} s has no {name}")
                vals.append(_num(t[name], name))
            stats = {"min": min(vals), "median": statistics.median(vals), "max": max(vals)}
            recorded = _get(entry, "parts", name, what=f"{what}, {seconds:g} s")
            for k, v in stats.items():
                if not isinstance(recorded, dict) or not isinstance(recorded.get(k), (int, float)) or abs(recorded[k] - v) > 0.0015:
                    raise AudioPackageError(f"{what}: the {k} of {name} at {seconds:g} s is not what its runs give ({v:g})")
            parts[name] = stats
        rows.append(SizeRow(seconds, label, warmup, len(runs), parts))
    load = bench.get("load")
    if not isinstance(load, dict):
        raise AudioPackageError(f"{what} has no load record")
    load = {k: _num(load.get(k), k) for k in ("model_load_s", "chip_weights_upload_s")}
    return tuple(sorted(rows, key=lambda r: r.seconds)), load


# ---- the run -------------------------------------------------------------------------------------------------------

def _agreement(block: dict, what: str) -> dict:
    lat = _obj(_get(block, "latent_final", what=what), f"{what}.latent_final")
    wav = _obj(_get(block, "waveform", what=what), f"{what}.waveform")
    return {"latent_pcc": _num(lat.get("pcc"), "latent pcc"), "latent_rel_l2": _num(lat.get("rel_l2"), "latent rel_l2"),
            "wave_pcc": _num(wav.get("pcc"), "waveform pcc"), "wave_rel_l2": _num(wav.get("rel_l2"), "waveform rel_l2"),
            "log_spectrum": _num(wav.get("log_spectrum_distance"), "log-spectrum distance")}


def _obj(x, what: str) -> dict:
    """`x` when it is a JSON object, else an AudioPackageError that names `what`."""
    if not isinstance(x, dict):
        raise AudioPackageError(f"{what} is not a JSON object")
    return x


def facts_from_evidence(result, bench, *, snapshot_name: str | None = None) -> Facts:
    """The facts a card is rendered from, read from the two evidence files (the audio result and the benchmark sweep), both already parsed.
    The builder calls it on the run's files and the gate calls it again on the copies in the package, so evidence that cannot back a package
    is refused at both. Refuses: no identity pin, the chip run not matching its CPU reference, a control that did not tell the fine-tune
    from the base model, a benchmark file with no size that earns the label 'benchmark'."""
    result = _obj(result, "the audio result")
    chip = _obj(_get(result, "runs", "chip", what="the audio result"), "runs.chip of the audio result")
    if chip.get("model") != MODEL_KEY:
        raise AudioPackageError(f"the audio result is for model {chip.get('model')!r}; this builder packages {MODEL_KEY!r}")
    revision = chip.get("revision")
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise AudioPackageError("the run has no identity pin: the chip run records no 40-character model revision")
    library = chip.get("library_commit")
    if not isinstance(library, str) or not REVISION.fullmatch(library):
        raise AudioPackageError("the run has no identity pin: the chip run records no 40-character library commit")
    if snapshot_name is not None and snapshot_name != revision:
        raise AudioPackageError(f"the license snapshot is revision {snapshot_name} and the run served revision {revision}")
    settings = _obj(chip.get("settings"), "the chip run's settings")
    agreement = _obj(_get(result, "agreement", "chip_vs_reference", what="the audio result"), "agreement.chip_vs_reference")
    if agreement.get("ok") is not True or agreement.get("comparable") is not True:
        raise AudioPackageError("the chip run did not agree with the CPU reference within the bars (or the two were not comparable); nothing to package")
    disc, diff = result.get("control_discrimination"), result.get("control_difference")
    if not (isinstance(disc, dict) and disc.get("ok") is True and isinstance(diff, dict) and diff.get("ok") is True):
        raise AudioPackageError("the run's control did not show that the chip served the fine-tune and not the base model; nothing to package")
    floor = _obj(_get(result, "environment_floor", what="the audio result"), "environment_floor")
    bars = agreement.get("bars")
    if not isinstance(bars, str) or "provisional" not in bars:
        raise AudioPackageError("the agreement record does not say its bars are provisional; this builder labels agreement by that statement")
    sizes, load = bench_sizes(_obj(bench, "the benchmark file"), settings)
    if not any(s.label == LABEL_BENCHMARK for s in sizes):
        raise AudioPackageError(f"no size of the benchmark file is a benchmark (it needs {MIN_REPEATS} or more measured runs after {MIN_WARMUP} or more "
                                "warm-up run(s) in one process); the card would have no number it may call one")
    return Facts(MODEL_ID, revision, library, {k: settings[k] for k in sorted(settings)}, str(chip.get("device", "")), _agreement(agreement, "chip_vs_reference"),
                 _agreement(floor, "environment_floor"), bars, sizes, load, BENCH_EVIDENCE, RESULT_EVIDENCE)


def read_facts(run_dir, bench_path, *, snapshot_name: str | None = None) -> tuple[Facts, bytes, bytes]:
    """(facts, the bytes of the audio result, the bytes of the benchmark file) for the run at `run_dir`. Refuses a run that cannot back a package
    (see `facts_from_evidence`), a run whose ledger names another model, and any file of an unexpected shape."""
    run = Path(run_dir)
    result_path = run / "stages" / "2" / "audio_result.json"
    result = read_json_file(result_path, "stages/2/audio_result.json")
    bench_bytes = _file_bytes(bench_path, "the benchmark file", MAX_JSON_BYTES)
    try:
        bench = safeio.parse_json_bytes(bench_bytes, max_bytes=MAX_JSON_BYTES)
    except safeio.SafeReadError as exc:
        raise AudioPackageError(f"the benchmark file is not valid JSON: {getattr(exc, 'reason', exc)}") from None
    try:
        facts = facts_from_evidence(result, bench, snapshot_name=snapshot_name)
    except (KeyError, TypeError, ValueError, AttributeError, safeio.SafeReadError) as exc:      # a shape no check above named
        raise AudioPackageError(f"the audio result or the benchmark file has an unexpected shape ({type(exc).__name__})") from None
    model_id = _ledger_model(run)
    if model_id != MODEL_ID:
        raise AudioPackageError(f"the run's ledger names the model {model_id!r}; this builder packages {MODEL_ID}")
    return facts, _file_bytes(result_path, "audio_result.json", MAX_JSON_BYTES), bench_bytes


def _ledger_model(run: Path) -> str | None:
    """The model id of the run's first ledger entry (`run_start`)."""
    try:
        first = safeio.read_bytes(run / "ledger.jsonl", max_bytes=MAX_JSON_BYTES).split(b"\n", 1)[0]
        entry = safeio.parse_json_bytes(first, max_bytes=MAX_JSON_BYTES)
    except safeio.SafeReadError as exc:
        raise AudioPackageError(f"the run has no readable ledger.jsonl: {getattr(exc, 'reason', exc)}") from None
    data = entry.get("data") if isinstance(entry, dict) else None
    model = data.get("model") if isinstance(data, dict) else None
    return model if isinstance(model, str) else None


def ledger_snapshot(run_dir) -> str | None:
    """The model snapshot path the run recorded at its start (`inputs.model`), or None."""
    try:
        first = safeio.read_bytes(Path(run_dir) / "ledger.jsonl", max_bytes=MAX_JSON_BYTES).split(b"\n", 1)[0]
        entry = safeio.parse_json_bytes(first, max_bytes=MAX_JSON_BYTES)
    except safeio.SafeReadError:
        return None
    data = entry.get("data") if isinstance(entry, dict) else None
    inputs = data.get("inputs") if isinstance(data, dict) else None
    value = inputs.get("model") if isinstance(inputs, dict) else None
    return value if isinstance(value, str) else None


# ---- the license --------------------------------------------------------------------------------------------------

def _flat(text: str) -> str:
    return " ".join(text.split())


def quote(license_text: str, start: str, end: str) -> str | None:
    """The sentence(s) of the agreement from `start` to `end`, whitespace collapsed, or None when the agreement does not have them."""
    flat = _flat(license_text)
    i = flat.find(start)
    if i < 0:
        return None
    j = flat.find(end, i)
    return flat[i:j + len(end)] if j >= 0 else None


def attribution_sentence(license_bytes: bytes) -> str | None:
    """The attribution sentence exactly as the agreement writes it (its own spaces and its copyright sign), or None."""
    text = license_bytes.decode("utf-8", errors="replace")
    i = text.find(ATTRIBUTION_START)
    if i < 0:
        return None
    j = text.find(ATTRIBUTION_END, i)
    return text[i:j + len(ATTRIBUTION_END)] if j >= 0 else None


def read_license_files(snapshot) -> tuple[bytes, bytes]:
    """(LICENSE.md, the snapshot's own NOTICE) from the pinned snapshot. The license is required; the snapshot's NOTICE only has to be readable."""
    snap = Path(snapshot)
    # A Hugging Face cache snapshot holds links to blobs, so the two files are resolved first and then read as regular files.
    lic = _file_bytes((snap / "LICENSE.md").resolve(), "the snapshot's LICENSE.md")
    if attribution_sentence(lic) is None:
        raise AudioPackageError("the snapshot's LICENSE.md has no attribution sentence; the license is not the one this builder was written for")
    for name, start, end in QUOTES:
        if quote(lic.decode("utf-8", errors="replace"), start, end) is None:
            raise AudioPackageError(f"the snapshot's LICENSE.md no longer has the {name} sentence this builder quotes; read the agreement again")
    if "USD $1,000,000" not in (quote(lic.decode("utf-8", errors="replace"), *QUOTES[2][1:]) or ""):
        raise AudioPackageError("the snapshot's LICENSE.md no longer gives the revenue threshold as USD $1,000,000, which the card states; read the agreement again")
    return lic, _file_bytes((snap / "NOTICE").resolve(), "the snapshot's NOTICE")


def notice_text(license_bytes: bytes) -> str:
    """The NOTICE file: the attribution sentence the agreement asks for, on its own line, then one line about what the package does not change."""
    sentence = attribution_sentence(license_bytes)
    assert sentence is not None
    return (sentence + "\n\n"
            "The runtime code in this package is Tenstorrent's. This package does not contain the Stability AI model weights and does not\n"
            "modify them. The text encoder inside the model download is provided under the Gemma Terms of Use (" + GEMMA_LINK + ").\n")


# ---- the card ------------------------------------------------------------------------------------------------------

def _fmt(x) -> str:
    """A timing as the card shows it: at most 4 decimals (the files hold 3, and the median of an even number of runs can have a fourth)."""
    return f"{round(x, 4):g}" if isinstance(x, (int, float)) else str(x)


def number_rows(f: Facts) -> list[tuple[str, str, str, tuple[str, ...]]]:
    """(name, value, label, evidence) for every row of the Numbers table. Values are the ones in the evidence files."""
    rows = []
    bench = (f.bench_file,)
    for s in f.sizes:
        for part, text in CARD_PARTS:
            stats = s.parts[part]
            if part != MAIN_PART:
                rows.append((f"{s.seconds:g} s clip, {text}, median of {s.runs}", f"{_fmt(stats['median'])} s", s.label, bench))
                continue
            for k in ("min", "median", "max"):
                rows.append((f"{s.seconds:g} s clip, {text}, {k} of {s.runs}", f"{_fmt(stats[k])} s", s.label, bench))
    rows.append(("model load, once per process (first run)", f"{_fmt(f.load['model_load_s'])} s", LABEL_SINGLE, bench))
    rows.append(("weight upload to the chip, once per process (first run)", f"{_fmt(f.load['chip_weights_upload_s'])} s", LABEL_SINGLE, bench))
    res = (f.result_file,)
    a, e = f.agreement, f.floor
    for name, key, nd in (("final latent PCC", "latent_pcc", 4), ("final latent relative L2", "latent_rel_l2", 4), ("waveform PCC", "wave_pcc", 4),
                          ("waveform relative L2", "wave_rel_l2", 4), ("log-spectrum distance", "log_spectrum", 4)):
        rows.append((f"chip against the CPU reference, {name}", f"{a[key]:.{nd}f}", LABEL_AGREEMENT, res))
    for name, key, nd in (("final latent PCC", "latent_pcc", 4), ("waveform PCC", "wave_pcc", 4)):
        rows.append((f"environment floor (two CPU environments), {name}", f"{e[key]:.{nd}f}", LABEL_AGREEMENT, res))
    for name in ("boot check of this package on a chip", "download of this package from the Hub and a boot from it"):
        rows.append((name, "TODO", "TODO", ()))
    return rows


def _row(name, value, label, evidence) -> str:
    ev = ", ".join(f"`{p}`" for p in evidence) if evidence else "-"
    return f"| {name} | {value} | {label} | {ev} |"


def render_card(f: Facts, *, namespace: str, license_bytes: bytes) -> str:
    q = {name: quote(license_bytes.decode("utf-8"), start, end) for name, start, end in QUOTES}
    a = f.settings
    lines = ["---", "license: other", f"license_name: {LICENSE_FRONT_NAME}", "license_link: LICENSE.md", "pipeline_tag: text-to-audio",
             f"base_model: {f.model_id}", "tags:", "- blackhole", "- p150", "- text-to-audio", "- music-generation", "- diffusion", "---", "",
             f"# {PACKAGE_NAME}", "", f"**{POWERED_BY}**", "",
             f"A package that runs {f.model_id} (revision `{f.revision}`) on one Tenstorrent Blackhole chip: the diffusion transformer runs on the chip, "
             "and a local job server and a Gradio page start it and play the clips. The tt-orchard harness staged it from a run of that model. "
             f"The package is staged locally and was not published when this card was written (the repository would be `{namespace}/{PACKAGE_NAME}`).", "",
             "## License", "",
             f"The model is licensed under the [{LICENSE_NAME}](LICENSE.md). This package carries a copy of the agreement (`LICENSE.md`) and a `NOTICE` file "
             "with the attribution the agreement requires. Read the license before you use or share the model or this package. "
             "The runtime code in this package is Tenstorrent's. The license applies to the model, and the person who runs this package and every user of it is "
             "responsible for following it. Before the package is published, the account that publishes it has to decide which of the agreement's terms apply "
             "to that account, because the commercial terms depend on the annual revenue of the licensee and its affiliates.", "",
             "Research and non-commercial use is permitted. The agreement says:", "", f"> {q['research']}", "",
             "Commercial use requires registering with Stability AI, and a license from Stability AI is required above USD $1,000,000 in annual revenue. "
             "The agreement says:", "", f"> {q['register']}", ">", f"> {q['threshold']}", "",
             f"Registration is at {REGISTER_LINK}. Use of the model and of its output must also follow Stability AI's acceptable use policy ({AUP_LINK}).", "",
             "## Weights are not included", "",
             "This package does not include the model weights. The weights are gated: each user downloads "
             f"[{f.model_id}](https://huggingface.co/{f.model_id}) with their own Hugging Face account after accepting the terms on that page. "
             "The download also holds the T5Gemma text encoder, which is provided under the "
             f"[Gemma Terms of Use]({GEMMA_LINK}); accepting the model's terms includes accepting those.", "",
             "## What runs where", "",
             "- On the chip: the 20-layer diffusion transformer.",
             "- On the host CPU, from the model library's own code: the T5Gemma text encoder, the sampler and the autoencoder decoder.",
             f"- The model library is [stable-audio-3]({LIBRARY_LINK}) at commit `{f.library_commit}`, the commit the run used.",
             f"- Settings of the run: {a.get('steps')} steps, guidance scale {a.get('cfg_scale')}, the {a.get('sampler')} sampler, seed {a.get('seed')}.", "",
             "## Try it", "",
             "```bash", "pip install -r requirements.txt", f"hf download {f.model_id} --revision {f.revision}      # your own Hugging Face account, after accepting the terms",
             "export ORCHARD_TTNN_PYTHON=<python with ttnn>   ORCHARD_AUDIO_SRC=<stable-audio-3 checkout at the commit above>",
             "./run.sh          # checks the cached revision first; on a shared machine, take a chip lease first; see docs/running.md", "```", "",
             "The job server listens on http://127.0.0.1:20100 and the page on http://127.0.0.1:7872. Both bind the local machine only. "
             "The job API is `POST /v1/audio/jobs`, `GET /v1/audio/jobs/<id>`, `GET /v1/audio/jobs/<id>/audio`, `DELETE /v1/audio/jobs/<id>` and `GET /v1/health`.", "",
             "`tt-model.yaml` is a draft v5.1 manifest of kind `http-server`. It has not been built into an image or booted.", "",
             NUMBERS_HEAD, "",
             "Every number is in this table and comes from a file in `evidence/`. A row labelled `benchmark` has 3 or more measured runs after a discarded warm-up "
             "run in one process. A row labelled `single sample` has fewer runs. The agreement rows compare the chip run with the CPU reference; their pass bars "
             "are provisional choices that were not set from a measured noise floor. `TODO` marks a measurement that was not taken.", "",
             "| Number | Value | Label | Evidence |", "| --- | --- | --- | --- |"]
    lines += [_row(*r) for r in number_rows(f)]
    lines += ["", "The timings are from one process that loaded the model once. The job server in this package starts a new process for each clip, so it loads the "
              "model again for each clip and a clip takes longer than the table shows by about the load rows.", "",
              NOT_MEASURED_HEAD, "",
              "- Sound quality. No quality evaluation was run, and this card makes no claim about it.",
              "- A boot of this package on a chip, and a download of it from the Hub: neither has been done.",
              "- Requests that arrive together, and a server that keeps the model loaded.",
              "- Any prompt, seed or length other than the run's. Agreement was measured on one prompt, one length and one seed.", "",
              "## Intended use", "",
              "Making music clips from a text prompt on a Tenstorrent Blackhole chip, for research, evaluation, testing or personal use under the license above.", "",
              "## Limitations", "",
              "- The text encoder, the sampler and the decoder run on the host CPU. Only the diffusion transformer runs on the chip.",
              "- The server makes one clip at a time and keeps a short queue. A request is refused when the queue is full.",
              "- The pass bars for agreement are provisional.",
              "- The server has no sign-in. Run it on a network you trust.", "",
              "## Risks and safety considerations", "",
              f"Generated audio can resemble existing recordings. Check the output before you use or share it, and follow the acceptable use policy ({AUP_LINK}).", "",
              "## Provenance", "",
              f"- Model: {f.model_id}, revision `{f.revision}`. Weights are not part of this package.",
              "- License files: `LICENSE.md` and `NOTICE` are copied from the model's own repository at that revision.",
              "- Chip runtime: `runtime/`, copied unmodified from tt-orchard.",
              "- Job server and page: `demo/`, copied from the Tenstorrent demo repository for this model. One edit was made to the page's text, to add the "
              f"line \"{POWERED_BY}\"; `{RECORD_FILE}` has the hashes before and after.", ""]
    return "\n".join(lines)


def running_doc(f: Facts) -> str:
    return (f"# Running {PACKAGE_NAME}\n\n**{POWERED_BY}**\n\n"
            f"1. Accept the model's terms on its Hugging Face page and download revision `{f.revision}`: `hf download {MODEL_ID} --revision {f.revision}`.\n"
            "2. Check out the stable-audio-3 library at the commit the card names and set `ORCHARD_AUDIO_SRC` to it. "
            "Set `ORCHARD_TTNN_PYTHON` to a Python that has ttnn.\n"
            "3. Start the server and the page: `./run.sh`. It first checks that the cache resolves the model to the revision above and stops if it does not. "
            "On a shared machine, hold a chip lease for the whole run.\n\n"
            "The model is licensed under the " + LICENSE_NAME + " (`LICENSE.md`). Read it before you use or share the model or this package. "
            "This package does not include the model weights.\n")


def run_script(f: Facts) -> str:
    return f"""#!/usr/bin/env bash
# Starts the job server and the Gradio page for {MODEL_ID} on one Blackhole chip.
# Powered by Stability AI. The model is licensed under the {LICENSE_NAME} (LICENSE.md). This package does not include the weights:
# download them first with your own Hugging Face account (hf download {MODEL_ID} --revision {f.revision}).
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REVISION="{f.revision}"
: "${{ORCHARD_TTNN_PYTHON:?set ORCHARD_TTNN_PYTHON to a Python that has ttnn}}"
: "${{ORCHARD_AUDIO_SRC:?set ORCHARD_AUDIO_SRC to the stable-audio-3 checkout named in README.md}}"
echo "{POWERED_BY}"
# Refuse to start on another revision than the one the card, the evidence and the license copy were made for. This opens no device.
PYTHONPATH="$ORCHARD_AUDIO_SRC${{PYTHONPATH:+:$PYTHONPATH}}" "$ORCHARD_TTNN_PYTHON" "$HERE/runtime/orchard/skills/audio-chip-templates/check_revision.py" "$REVISION" {MODEL_KEY}
exec "$ORCHARD_TTNN_PYTHON" "$HERE/demo/serve_audio.py" --generator chip --src-dir "$ORCHARD_AUDIO_SRC" \\
  --script "$HERE/runtime/orchard/skills/audio-chip-templates/chip_generate.py" --ui "$@"
"""


def manifest_text(f: Facts, namespace: str) -> str:
    return f"""# DRAFT. NOT BUILT. A v5.1 container manifest of kind `http-server` for the job server in demo/. It cannot be built from this directory alone:
# `source.tt_metal` must be a tt-metal checkout, and `source.extra_code` must name this directory. Why http-server: the server is a standard-library
# HTTP server with its own launch line, which is what that kind is for; the v6 thin kind `tt-dit-server` serves an ASGI app with uvicorn, and this
# server is not one. Use `run.sh` to run the package without tt-model.
schema: "5.1"
repo: {namespace}/{PACKAGE_NAME}
name: {PACKAGE_NAME}
weights:
  repo: {f.model_id}
  revision: "{f.revision}"
kind: http-server
arch: blackhole
image:
  registry: hf
source:
  tt_metal: /path/to/tt-metal
  code:
    - models/common
  extra_code:
    - root: /path/to/this/package
      paths: [demo, runtime]
  ubuntu: "22.04"
  python: "3.12"
runtime:
  command: [python, demo/serve_audio.py, --generator, chip, --host, "{{host}}", --port, "{{port}}",
            --script, runtime/orchard/skills/audio-chip-templates/chip_generate.py]
  ready_line: "model ready"
  packages: ["numpy", "soundfile", "torch"]
serve:
  port: 20100
serve_profiles:
  - name: p150
    description: One Blackhole chip (a P150, or the first chip of a P300 or QuietBox 2).
    hardware: p150
    mesh_device: P150
default_profile: p150
"""


def publish_commands(name: str = PACKAGE_NAME, namespace: str = "NAMESPACE") -> str:
    """The commands that would publish the staged package, as text. Nothing here runs them, and no function in this module does."""
    return "\n".join([
        "# Commands that would publish the staged audio package. The builder never ran them. Each needs the operator's explicit go-ahead.",
        "# Read the run log's license section first: publishing under a company account needs a decision about the Community License.",
        "# 1. Create a private repository and upload the package directory (the weights are not in it):",
        f"#    hf upload --repo-type model --private {namespace}/{name} stages/7/package/{name} .",
        "# 2. After a boot check and a pull from the private repository, make it public (a separate decision):",
        f"#    hf repos settings {namespace}/{name} --no-private",
        "# 3. List it in the tt-model catalog (a separate decision; the catalog reads a built package, and tt-model.yaml here is a draft that has not been built):",
        f"#    tt-model publish {namespace}/{name}",
        ""])


# ---- the gate ------------------------------------------------------------------------------------------------------

def _front(card: str) -> dict:
    if not card.startswith("---\n"):
        return {}
    out = {}
    for ln in card[4:].split("\n---", 1)[0].splitlines():
        m = re.match(r"^([A-Za-z_]+):\s*(.*)$", ln)
        if m:
            out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


def _table(card: str) -> tuple[list[list[str]], str]:
    """(the Numbers table's rows as cells, the card without the table's rows)."""
    rows, rest = [], []
    in_numbers = False
    for ln in card.splitlines():
        if ln.startswith("## "):
            in_numbers = ln.strip() == NUMBERS_HEAD
        if in_numbers and ln.startswith("| ") and not ln.startswith("| Number ") and not ln.startswith("| ---"):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
        else:
            rest.append(ln)
    return rows, "\n".join(rest)


def _bench_label_at(evidence: Path, seconds: float) -> str | None:
    """The label the benchmark file's runs earn for the size, recomputed; None when the file has no such size."""
    try:
        bench = safeio.read_json(evidence, max_bytes=MAX_JSON_BYTES, expect=dict)
    except safeio.SafeReadError:
        return None
    for s in bench.get("sizes") or []:
        if isinstance(s, dict) and s.get("seconds") == seconds and isinstance(s.get("runs"), list) and isinstance(s.get("warmup_runs"), int):
            return label_for(len(s["runs"]), s["warmup_runs"])
    return None


EVIDENCE_PATH = re.compile(r"evidence/[A-Za-z0-9_.-]+\.json")


def package_facts(pkg: Path) -> tuple[Facts | None, str | None]:
    """(the facts the package's own evidence files give, None) or (None, why the evidence cannot back a package). Read again from the files in the
    package, so a card is checked against the evidence that ships with it and not against anything the builder remembered."""
    try:
        result = read_json_file(pkg / RESULT_EVIDENCE, RESULT_EVIDENCE)
        bench = read_json_file(pkg / BENCH_EVIDENCE, BENCH_EVIDENCE)
        return facts_from_evidence(result, bench), None
    except (AudioPackageError, KeyError, TypeError, ValueError, AttributeError, safeio.SafeReadError) as exc:
        return None, str(exc) if isinstance(exc, AudioPackageError) else f"unexpected shape ({type(exc).__name__})"


def _row_key(row) -> tuple:
    name, value, label, evidence = row
    return (name, value, label, ", ".join(f"`{p}`" for p in evidence) if evidence else "-")


def license_problems(card: str, license_bytes: bytes | None, notice: str | None) -> list[str]:
    """The license elements, one check each. `license_bytes` is the package's LICENSE.md, `notice` the package's NOTICE (None when absent)."""
    problems = []
    flat = _flat(card)
    low = flat.lower()
    front = _front(card)
    if front.get("license") != "other":
        problems.append(f"the card's front matter must say `license: other`; it says {front.get('license')!r}")
    if front.get("license", "").lower() in OPEN_LICENSES:
        problems.append(f"the card claims the open license {front['license']}; the model is under the {LICENSE_NAME}")
    if front.get("license_name") != LICENSE_FRONT_NAME:
        problems.append(f"the card's front matter must say `license_name: {LICENSE_FRONT_NAME}`")
    if front.get("license_link") != "LICENSE.md":
        problems.append("the card's front matter must say `license_link: LICENSE.md`, the copy of the agreement in the package")
    if LICENSE_NAME not in card:
        problems.append(f"the card must name the {LICENSE_NAME}")
    if "](LICENSE.md)" not in card:
        problems.append("the card must link the license (the copy of the agreement, `LICENSE.md`)")
    if "research and non-commercial use is permitted" not in low:
        problems.append("the card must say that research and non-commercial use is permitted")
    if POWERED_BY not in card:
        problems.append(f"the card must display {POWERED_BY!r}")
    for ph, what in (("does not include the model weights", "that the weights are not included"), ("gated", "that the weights are gated")):
        if ph not in low:
            problems.append(f"the card must say {what}")
    if "Gemma Terms of Use" not in card or GEMMA_LINK not in card:
        problems.append(f"the card must name the Gemma Terms of Use for the T5Gemma text encoder and link {GEMMA_LINK}")
    if "runtime code in this package is tenstorrent's" not in low:
        problems.append("the card must say the runtime code is Tenstorrent's")
    if "read the license" not in low:
        problems.append("the card must tell the reader to read the license")
    if REGISTER_LINK not in card or "must register with stability ai" not in low:
        problems.append(f"the card must say commercial use requires registering with Stability AI ({REGISTER_LINK})")
    if "usd $1,000,000" not in low or "stability.ai/enterprise" not in low:
        problems.append("the card must give the revenue threshold (USD $1,000,000) and say a license from Stability AI is required above it")
    for pattern in COMMERCIAL_CLAIMS:
        hit = re.search(pattern, card, re.I)
        if hit:
            problems.append(f"the card claims or implies an open or commercial license: {hit.group(0)!r}")
    for pattern in MARK_CLAIMS:
        hit = re.search(pattern, card, re.I)
        if hit:
            problems.append(f"the card uses a name or mark in a way the license does not allow (it permits them only as it requires): {hit.group(0)!r}")
    if license_bytes is None:
        problems.append("the package has no LICENSE.md")
    else:
        text = license_bytes.decode("utf-8", errors="replace")
        if "STABILITY AI COMMUNITY LICENSE AGREEMENT" not in text:
            problems.append("LICENSE.md is not the Stability AI Community License Agreement")
        for name, start, end in QUOTES:
            wanted = quote(text, start, end)
            if wanted is None:
                problems.append(f"LICENSE.md has no {name} sentence to quote, so it is not the agreement this card quotes")
            elif wanted not in flat:
                problems.append(f"the card must quote the license's {name} sentence as the agreement writes it: {wanted[:70]!r}...")
        sentence = attribution_sentence(license_bytes)
        if sentence is None:
            problems.append("LICENSE.md has no attribution sentence")
        elif notice is None:
            problems.append("the package has no NOTICE file")
        elif sentence not in notice:
            problems.append("NOTICE must contain the attribution sentence exactly as the agreement writes it, "
                            f"including its spaces and its copyright sign: {sentence!r}")
    if notice is None and license_bytes is not None and attribution_sentence(license_bytes) is not None:
        pass                                    # reported above
    return problems


def card_problems(card: str, pkg: Path, *, license_bytes: bytes | None = None, notice: str | None = None) -> list[str]:
    """Everything wrong with the card: the license elements, then the numbers and the claims. The Numbers table is compared, row by row, with the rows
    the package's own evidence files give (`number_rows` on `package_facts`): a value taken from another field, a label the runs do not earn, an extra
    row and a missing row are each refused."""
    problems = license_problems(card, license_bytes, notice)
    rows, prose = _table(card)
    if not rows:
        problems.append("the card has no Numbers table")
    facts, why = package_facts(pkg)
    if facts is None:
        problems.append(f"the package's evidence cannot back a card: {why}")
    has_benchmark = False
    shown = set()
    for cells in rows:
        if len(cells) != 4:
            problems.append(f"Numbers row {cells!r} needs four cells")
            continue
        shown.add(tuple(cells))
        name, value, label, evidence = cells
        if label not in LABELS:
            problems.append(f"number {name!r} has label {label!r}; a label is one of {', '.join(LABELS)}")
            continue
        if label == "TODO":
            if value != "TODO":
                problems.append(f"number {name!r} is TODO and shows a value")
            continue
        paths = [p.strip().strip("`") for p in evidence.split(",") if p.strip() not in ("", "-")]
        if not paths:
            problems.append(f"number {name!r} is labelled {label} and names no evidence")
            continue
        for p in paths:
            if not EVIDENCE_PATH.fullmatch(p) or inside(pkg, p) is None:
                problems.append(f"number {name!r}: evidence {p} must be a JSON file in evidence/ inside the package")
        if label == LABEL_BENCHMARK:
            has_benchmark = True
            m = re.match(r"([0-9.]+) s clip", name)
            found = inside(pkg, paths[0]) if EVIDENCE_PATH.fullmatch(paths[0]) else None
            earned = _bench_label_at(found, float(m.group(1))) if m and found is not None else None
            if earned != LABEL_BENCHMARK:
                problems.append(f"number {name!r} is labelled benchmark and its runs earn {earned!r}: a benchmark has {MIN_REPEATS} or more measured runs "
                                f"after {MIN_WARMUP} or more warm-up run(s)")
        elif label == LABEL_AGREEMENT and "provisional" not in prose.lower():
            problems.append(f"number {name!r} is an agreement number and the card does not say the bars are provisional")
    if facts is not None:
        wanted = {_row_key(r): r for r in number_rows(facts)}
        have_names = {c[0]: c for c in shown}
        for key, r in wanted.items():
            if key not in shown:
                got = have_names.get(r[0])
                detail = f"the card has {got[1]} ({got[2]})" if got else "the card has no such row"
                problems.append(f"the Numbers row {r[0]!r} must be {key[1]} ({key[2]}) as the evidence gives; {detail}")
        names = {r[0] for r in wanted.values()}
        for c in shown:
            if c[0] not in names:
                problems.append(f"the Numbers table has a row the evidence does not give: {c[0]!r}")
    if not has_benchmark and re.search(r"benchmark", prose, re.I):
        problems.append("the card uses the word benchmark and has no row that earns the label")
    for hit in PERF_IN_PROSE.finditer(prose):
        problems.append(f"a number outside the Numbers table: {hit.group(0)!r}")
    for hit in re.finditer(r"\b\d[\d.,]*\s*(?:seconds|sec|s)\b(?!\w)", prose):
        problems.append(f"a timing outside the Numbers table: {hit.group(0)!r}")
    for pattern in QUALITY_CLAIMS:
        hit = re.search(pattern, prose, re.I)
        if hit:
            problems.append(f"the card makes a claim about sound quality, which the run did not measure: {hit.group(0)!r}")
    if "No quality evaluation was run" not in card:
        problems.append("the card must say under Not measured that no quality evaluation was run")
    for head in ("## Intended use", "## Limitations", "## Risks and safety considerations", NOT_MEASURED_HEAD):
        if head not in card:
            problems.append(f"the card has no {head!r} section")
    return problems


def _tree(pkg: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(pkg):
        dirnames.sort()
        out += [Path(dirpath) / n for n in sorted(filenames)]
    return out


def weights_problems(pkg: Path) -> list[str]:
    """Any file that looks like model weights, and any file large enough to be one. The package ships none."""
    problems = []
    for f in _tree(pkg):
        rel = f.relative_to(pkg).as_posix()
        if f.suffix.lower() in WEIGHT_SUFFIXES:
            problems.append(f"{rel}: the package must not contain weights ({f.suffix.lower()} file)")
        elif not f.is_symlink() and f.stat().st_size > WEIGHT_MAX_BYTES:
            problems.append(f"{rel}: {f.stat().st_size} bytes; no file in the package is over {WEIGHT_MAX_BYTES} bytes, and the weights are not shipped")
    return problems


def package_problems(pkg, *, hostname: str | None = None, home: str | None = None, validate_manifest=None) -> list[str]:
    """The gate: everything wrong with the staged package at `pkg`, read from disk. An empty list means the package passes.

    `validate_manifest(path)` is optional: a callable that returns a list of problems with tt-model.yaml (tests pass one; the command passes the
    installed tt_kernel's loader when it is importable)."""
    pkg = Path(pkg)
    if not pkg.is_dir():
        return [f"{pkg} is not a directory"]
    problems = weights_problems(pkg)
    problems += [f"scrub: {hit}" for hit in scrub_package(pkg, hostname=hostname, home=home)]
    try:
        card = (pkg / "README.md").read_text(encoding="utf-8")
    except OSError:
        return problems + ["the package has no README.md card"]
    lic = (pkg / "LICENSE.md")
    notice = (pkg / "NOTICE")
    lic_bytes = lic.read_bytes() if lic.is_file() else None
    notice_s = notice.read_text(encoding="utf-8") if notice.is_file() else None
    problems += card_problems(card, pkg, license_bytes=lic_bytes, notice=notice_s)
    page = pkg / "demo" / "audio_app.py"
    if not page.is_file():
        problems.append("the package has no demo/audio_app.py, the page")
    elif POWERED_BY not in page.read_text(encoding="utf-8"):
        problems.append(f"the page (demo/audio_app.py) must display {POWERED_BY!r}")
    docs = [p for p in (pkg / "docs").glob("*.md")] if (pkg / "docs").is_dir() else []
    if not docs:
        problems.append("the package has no documentation file in docs/")
    for d in docs:
        if POWERED_BY not in d.read_text(encoding="utf-8"):
            problems.append(f"docs/{d.name} must display {POWERED_BY!r}")
    for name in REQUIRED_FILES:
        if not (pkg / name).is_file():
            problems.append(f"the package has no {name}")
    facts, _ = package_facts(pkg)
    if facts is not None:                      # the pin is the same everywhere it is written, and the launcher checks it
        run_sh = (pkg / "run.sh").read_text(encoding="utf-8") if (pkg / "run.sh").is_file() else ""
        manifest = (pkg / "tt-model.yaml").read_text(encoding="utf-8") if (pkg / "tt-model.yaml").is_file() else ""
        if f'REVISION="{facts.revision}"' not in run_sh or "check_revision.py" not in run_sh:
            problems.append("run.sh must carry the evidence's model revision and run its check_revision.py before it starts the server")
        if f'revision: "{facts.revision}"' not in manifest:
            problems.append("tt-model.yaml must pin the evidence's model revision")
        if f"`{facts.revision}`" not in card:
            problems.append("the card must name the evidence's model revision")
    if validate_manifest is not None and (pkg / "tt-model.yaml").is_file():
        problems += [f"tt-model.yaml: {p}" for p in validate_manifest(pkg / "tt-model.yaml")]
    return problems


def kernel_manifest_problems(path) -> list[str]:
    """tt-model.yaml checked by the installed tt-model's own loader, when it can be imported (it opens no device). [] when it cannot be imported."""
    try:
        from tt_kernel.container_manifest import ContainerManifestError, load_container_manifest
    except ImportError:
        return []
    try:
        load_container_manifest(path)
    except ContainerManifestError as exc:
        return [str(exc).splitlines()[0]]
    return []


# ---- staging --------------------------------------------------------------------------------------------------------

def _write(pkg: Path, rel: str, data: bytes | str, mode: int = 0o644) -> dict:
    target = pkg / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = data.encode("utf-8") if isinstance(data, str) else data
    safeio.atomic_write(target, raw, mode=mode)
    return {"path": rel, "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _copy_source(root: Path, rel: str, what: str) -> bytes:
    return _file_bytes(root / rel, f"{what} {rel}")           # safeio refuses a link, so a link is never copied


def gradio_requirement(demo_repo: Path) -> str:
    """The Gradio pin the demo repository states in package/audio_package.py (read as text, never run)."""
    text = _file_text(demo_repo / "package" / "audio_package.py", "the demo repository's package/audio_package.py")
    m = re.search(r'^GRADIO_REQUIREMENT\s*=\s*"([^"\n]+)"', text, re.M)
    if m is None:
        raise AudioPackageError("package/audio_package.py in the demo repository states no GRADIO_REQUIREMENT")
    return m.group(1)


def stage_package(run_dir, *, bench_path, demo_repo, snapshot, namespace: str, repo_root=None, out=None) -> dict:
    """Stage the package under the run's `stages/7/package/<name>/` and return the record that was written beside the files.

    Order of refusals: the package directory exists and holds anything (a weights file is named first); the run cannot back a package
    (`read_facts`); the license files are not the ones this builder quotes; a source file is missing or is a link; the page has changed so the
    one edit does not apply. Nothing is written before every source has been read."""
    run = Path(run_dir)
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent
    pkg = Path(out) if out else run / "stages" / "7" / "package" / PACKAGE_NAME
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", namespace or ""):
        raise AudioPackageError("--namespace is the Hugging Face user or organisation the package would be published under, for example my-user")
    if pkg.exists() or pkg.is_symlink():
        if pkg.is_dir() and not pkg.is_symlink():
            weights = weights_problems(pkg)
            if weights:
                raise AudioPackageError(f"{pkg} already holds weights and is not staged over: {weights[0]}")
            if any(pkg.iterdir()):
                raise AudioPackageError(f"{pkg} is not empty; remove it to stage again (a package is never overwritten in place)")
        else:
            raise AudioPackageError(f"{pkg} exists and is not an empty directory")
    snap = Path(snapshot)
    facts, result_bytes, bench_bytes = read_facts(run, bench_path, snapshot_name=snap.resolve().name)
    license_bytes, _ = read_license_files(snap)
    demo = Path(demo_repo)
    sources = {rel: _copy_source(demo, rel, "the demo repository's") for rel, _ in DEMO_FILES}
    runtime = {dest: _copy_source(root, rel, "the tt-orchard") for rel, dest in RUNTIME_FILES}
    requirement = gradio_requirement(demo)
    page_src = sources["demo/audio_app.py"].decode("utf-8")
    if page_src.count(DEMO_ANCHOR) != 1:
        raise AudioPackageError("the demo page's text no longer has the line this builder edits to add the attribution; update the builder")
    page = page_src.replace(DEMO_ANCHOR, DEMO_PATCHED).encode("utf-8")

    files: list[dict] = []
    pkg.mkdir(parents=True, exist_ok=True)
    files.append(_write(pkg, "LICENSE.md", license_bytes))
    files.append(_write(pkg, "NOTICE", notice_text(license_bytes)))
    files.append(_write(pkg, "tt-model.yaml", manifest_text(facts, namespace)))
    files.append(_write(pkg, "run.sh", run_script(facts), mode=0o755))
    files.append(_write(pkg, "requirements.txt", f"{requirement}\nnumpy\nsoundfile\ntorch\n"))
    files.append(_write(pkg, "docs/running.md", running_doc(facts)))

    for dest, data in runtime.items():
        files.append(_write(pkg, "runtime/" + dest, data))
    files.append(_write(pkg, "runtime/orchard/__init__.py", RUNTIME_INIT))
    edited = {}
    for rel, name in DEMO_FILES:
        data = page if name == "audio_app.py" else sources[rel]
        files.append(_write(pkg, f"demo/{name}", data))
        if name == "audio_app.py":
            edited = {"path": "demo/audio_app.py", "source_sha256": sha256_bytes(sources[rel]), "staged_sha256": sha256_bytes(page),
                      "edit": f"the line `{DEMO_ANCHOR}` became `{DEMO_PATCHED}`"}
    files.append(_write(pkg, "demo/requirements.txt", requirement + "\n"))
    files.append(_write(pkg, "evidence/audio_result.json", result_bytes))
    files.append(_write(pkg, "evidence/bench-chip.json", bench_bytes))
    files.append(_write(pkg, "README.md", render_card(facts, namespace=namespace, license_bytes=license_bytes)))
    record = {"schema": 1, "model": facts.model_id, "revision": facts.revision, "library_commit": facts.library_commit, "namespace": namespace,
              "license": {"name": LICENSE_NAME, "snapshot_revision": snap.resolve().name, "license_sha256": sha256_bytes(license_bytes)},
              "copied_with_one_edit": edited, "files": sorted(files, key=lambda r: r["path"])}
    _write(pkg, RECORD_FILE, json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


# ---- the command ------------------------------------------------------------------------------------------------------

def main(args, *, out=None, hostname=None) -> int:
    """`tt-orchard audio-package RUN_DIR --namespace ORG --bench FILE --demo-repo DIR [--snapshot DIR] [--check]`. Stages locally and gates the result.
    Exit 0 when the package passes the gate, 1 when it does not, 2 when the run cannot be packaged."""
    import sys
    out = out or sys.stdout

    def say(text=""):
        print(text, file=out, flush=True)
    run = Path(args.run_dir)
    pkg = run / "stages" / "7" / "package" / PACKAGE_NAME
    try:
        if not args.check:
            snapshot = args.snapshot or ledger_snapshot(run)
            if not snapshot:
                raise AudioPackageError("the run records no snapshot path; give --snapshot (the pinned model snapshot that holds LICENSE.md)")
            if not args.bench or not args.demo_repo:
                raise AudioPackageError("staging needs --bench (the audio_bench.py sweep file) and --demo-repo (the separate demo repository)")
            record = stage_package(run, bench_path=args.bench, demo_repo=args.demo_repo, snapshot=snapshot, namespace=args.namespace)
            say(f"staged {len(record['files'])} files in {pkg}")
        problems = package_problems(pkg, hostname=hostname, validate_manifest=kernel_manifest_problems)
    except AudioPackageError as exc:
        say(f"refused: {exc}")
        return 2
    text = publish_commands(PACKAGE_NAME, args.namespace)
    try:
        safeio.atomic_write(run / "stages" / "7" / "AUDIO_PUBLISH_COMMANDS.txt", text, mode=0o644)
    except OSError:
        pass
    if problems:
        say("the package does not pass the gate:")
        for p in problems:
            say(f"  - {p}")
        return 1
    say("the package passes the gate. Nothing was uploaded or published; the commands that would publish it are in stages/7/AUDIO_PUBLISH_COMMANDS.txt (text only).")
    return 0
