#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Compare a new model with the nearest supported model and draft stage 0's delta.json.

The delta-triage skill copies this file into the stage directory and runs it with `python3`. It
reads `triage_config.json` from its own directory:

    {"run_dir": "<the run directory>",
     "model_id": "<the run's model, <repo>@<revision>>",
     "model_snapshot": "<the new model's Hugging Face snapshot directory>",
     "nearest_model_id": "<the nearest supported model, <repo>@<revision>>",
     "nearest_snapshot": "<the nearest model's snapshot directory>"}

An id without `@<revision>` gets one from its snapshot directory's name, when that directory sits
in a `snapshots/` folder (the Hugging Face cache layout).

It measures, and writes each measurement under `evidence/` next to itself:

    config-compare.json      the text configs (a nested `text_config` or a flat config), the
                             values that decide the path, every other differing key, and the
                             architecture names
    tensor-compare.json      tensor names, shapes and dtypes from the safetensors HEADERS. Only the
                             headers are read (8-byte little-endian length, then that much JSON), so
                             no weights are loaded. Names are compared as written and again after
                             `model.language_model.` is normalized to `model.`
    tokenizer-compare.json   vocab, merges (a merge may be a 2-element list or one space-joined
                             string; both forms compare equal), added tokens, normalizer,
                             pre-tokenizer, decoder and post-processor, the chat template's sha256 (the
                             template inside tokenizer_config.json and the template in chat_template.jinja
                             separately, because the file wins), and the eos, bos, pad and unk tokens
    tokenizer-encode.json    ids from both tokenizers (the `tokenizers` package) for the fixed strings
                             below and 100 fixed-seed random-unicode strings
    sidecar-compare.json     weights files outside the backbone, their tensor names, and any problem
    files-compare.json       every file and its size, and the weight shards
    genconfig-license.json   generation_config.json and the license (model card front matter, then
                             the LICENSE file)
    disk-free.json           free space on the run directory's filesystem (os.statvfs)

Then it writes a DRAFT `delta.json` in the schema the stage 0 gate reads. The agent reads it and
the evidence, and edits a finding only where a measured fact needs explaining.

The path rule. `weights-only` needs all of these, otherwise the path is `full-port` and
`path_reasons` says why:
- the text config values in DECISIVE are equal (head_dim is taken as hidden_size divided by
  num_attention_heads when a config leaves it out);
- the two models share at least one text tensor, and every shared tensor has an equal shape and
  the same dtype, except a tensor of equal shape and at most 1,048,576 elements whose two dtypes are both BF16, F16 or
  F32 (that difference gets a hazard and does not change the path; DTYPE_TOLERANCE_ELEMENTS);
- neither model has a tensor the other lacks, except vision-tower, projector and mtp tensors
  (EXPLAINED_EXTRA). A new text tensor means code the nearest model's implementation does not run.

A tokenizer difference does not change the path: the runtime loads the new model's tokenizer.
It is reported in the tokenizer finding.

Alignment. Names the `model.language_model.` rule leaves over are paired by what they share (a model that
calls its text tensors `language_model.layers.5...` and its vision tensors `visual.blocks.0...` against a nearest
model that says `model.language_model.layers.5...`): see align_leftovers. delta.json's `alignment` lists the
pairs and the prefix rewrites; evidence/value-compare.json samples the paired tensors' values (identical, near,
changed or unrelated); evidence/head-compare.json says whether a sidecar is a head cut down from the nearest
model's lm_head (numpy needed; the check says when it was skipped).

The class. delta.json also carries a `class` (orchard/classes.py): `weights-only`, `weights+sidecar`,
`restricted-head` (the backbone lines up and the only missing tensor is an lm_head that a sidecar's rows stand
in for; the harness stops after stage 0 and a person decides), `full-port` or `unknown`. A sidecar is a weights file in the new model's snapshot that is not one of
the backbone shards (the shards its `model.safetensors.index.json` lists, or `model*.safetensors`
when there is no index) and that the nearest model does not have, such as a task head. The backbone
is compared as above, without the sidecar. When the backbone is weights-only and every sidecar is a
readable safetensors file whose tensor names do not overlap the backbone's, the class is
`weights+sidecar` and the path stays `weights-only`. An unreadable sidecar, or one that overlaps,
makes the class `unknown` and the path `full-port`, with the reason in `path_reasons`. The class is
also `unknown` when the backbone could not be compared at all. `sidecars` lists each file with its
size, sha256, tensor count, dtypes and tensor names, and `code_files` lists each `.py`
file (down four directory levels) with its sha256. Nothing here runs a code file.

Exit codes: 0 when delta.json was written (whatever the path), 2 when triage_config.json is
missing, incomplete or names a directory that does not exist, or the stage directory is not inside
run_dir. The script writes only inside its own directory and opens no network connection.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import random
import re
import stat
import struct
import sys
from html.parser import HTMLParser
from pathlib import Path

try:                                     # numpy is optional: only the restricted-head check needs it
    import numpy as np
except ImportError:                      # that check then says it was skipped
    np = None

HERE_DIR = Path(__file__).resolve().parent
# The sha256 of this file, written into delta.json so the stage gate can compare it with the shipped template.
THIS_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EVIDENCE = HERE_DIR / "evidence"
CONFIG_KEYS = ("run_dir", "model_id", "model_snapshot", "nearest_model_id", "nearest_snapshot")
REPORT_MAX_BYTES = 2 * 1024 * 1024        # the research report is read only up to this size
OPTIONAL_KEYS = ("base_candidates",)      # read when present: the research report (research/base-candidates.json)

# The text config values that decide whether the nearest model's code can run the new weights.
DECISIVE = ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "layer_types", "vocab_size")
# Further text config values shown side by side when either model has them.
SHOWN = ("max_position_embeddings", "intermediate_size", "rope_theta", "rms_norm_eps",
         "tie_word_embeddings", "full_attention_interval", "linear_num_key_heads",
         "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
         "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
         "sliding_window")
# Keys that record how a file was written and say nothing about the model.
BOOKKEEPING = ("architectures", "transformers_version", "_name_or_path")
ABSENT = "<absent>"

# The vision-language layout puts the text model under this prefix.
LM_PREFIX = "model.language_model."
# Tensors one model may have and the other lack without new text-model code.
EXPLAINED_EXTRA = re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision|multi_modal_projector|"
                             r"mm_projector|mtp)(\.|$)")
MTP_NAME = re.compile(r"(^|\.)mtp\.")        # the MTP head's tensors, with or without a prefix
MAX_HEADER = 100 * 1024 * 1024          # a header larger than this is refused as corrupt

GIB = float(1 << 30)

# ---- aligning names, sampling values, finding a restricted head -----------------------------------------------------------
HEAD_NAME = re.compile(r"(^|\.)lm_head(\.|$)")
MIN_RULE_VOTES = 3            # choice: a prefix rewrite needs this many tensors that agree before it reaches short names
SAMPLE_ELEMENTS = 4096        # choice: values read from the start of each sampled tensor
SAMPLE_TENSORS = 8            # choice: tensors sampled per comparison, spread evenly over the sorted names
NEAR_COSINE = 0.999           # choice: at or above this the sampled values are "near" (a fine-tune moves them a little)
CHANGED_COSINE = 0.5          # choice: below this they are "unrelated"
HEAD_MATCH_COSINE = 0.999     # choice: a sidecar row this close to a base head row is a copy of it
HEAD_CHUNK_ROWS = 8192        # base head rows compared at a time (8192 rows of 5120 floats is 168 MB)
MAX_CODE_FILES = 60           # choice: code files listed
MAX_CODE_DEPTH = 4            # choice: directories searched for code below the model directory
SKIP_CODE_DIRS = {".git", ".cache", "__pycache__", "node_modules", "venv", ".venv", "site-packages"}
ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4}
# choice: a tensor that is BF16, F16 or F32 in one model and another of those in the other, with the same shape and at most this many
# elements, does not force a full port. The evidence is one model: sabicap/sabi-noema keeps 96 small tensors (A_log and linear-attention
# norm weights, at most 128 elements each) in F32 where Qwen3.8-27B has BF16; the base's 2-chip bundle served it with the weights as they
# are (2026-10-24, 31 of 32 tokens, coherent). Nothing larger or quantized was tested, so those stay full-port.
DTYPE_TOLERANCE_ELEMENTS = 1_048_576
READOUT_JSON_MAX = 256 * 1024     # choice: a small JSON file that declares an attention mode is read up to this size
CODE_SCAN_MAX = 400 * 1024        # choice: a code file larger than this is not scanned for hints
MAX_HINT_LINES = 8               # line numbers listed per hint
CODE_HINTS = (("noncausal", re.compile(r"non[-_ ]?causal", re.IGNORECASE)),
              ("causal flag", re.compile(r"is_causal\s*[=:]\s*(True|False)")),
              ("forward hook", re.compile(r"register_forward(_pre)?_hook")),
              ("attention mask", re.compile(r"create_\w*attention_mask|attention_mask")),
              ("readout", re.compile(r"readout", re.IGNORECASE)))

# ---- the encode test strings ----------------------------------------------------------------------
# Each base string is used as written and with a leading space (a byte-level tokenizer encodes a
# word after a space differently). The categories let a finding say which scripts differ.
_BASE_STRINGS = [
    ("latin", "Hello, world."),
    ("latin", "The quick brown fox jumps over the lazy dog."),
    ("latin", "Write the text I send my landlord about the broken boiler."),
    ("latin", "It's 3:45 pm; we'll meet at 221B Baker Street."),
    ("latin", "don't won't can't I'm you're they've"),
    ("latin", "CamelCaseIdentifier and snake_case_name"),
    ("latin", "def main(argv=None) -> int:\n    return 0"),
    ("latin", "Numbers 0 1 12 123 1234 12345 3.14159 -42"),
    ("latin", "e-mail: name at example dot org"),
    ("latin", "Tabs\tand\nnewlines\r\nmixed"),
    ("latin", "   leading and trailing spaces   "),
    ("latin", "UPPER lower MiXeD"),
    ("latin", "Hemingway wrote short sentences."),
    ("accented", "café naïve résumé façade"),
    ("accented", "Ångström São Paulo Kraków Zürich"),
    ("accented", "Dvořák Łódź Škoda Ærø"),
    ("accented", "¿Dónde está la biblioteca? ¡Olé!"),
    ("decomposed", "café naïve résumé"),
    ("decomposed", "Ångström"),
    ("decomposed", "ñ ô ü è"),
    ("greek-cyrillic", "Καλημέρα κόσμε"),
    ("greek-cyrillic", "Привет, мир! Съешь же ещё этих мягких булок."),
    ("cjk", "你好，世界。"),
    ("cjk", "我把租房合同发给房东了。"),
    ("cjk", "日本語のテキストです。"),
    ("cjk", "東京都渋谷区"),
    ("cjk", "中文、日本語、한국어 mixed"),
    ("korean", "안녕하세요 세계"),
    ("korean", "집주인에게 보일러 고장에 대해 문자를 보냅니다."),
    ("korean", "한글 자모 ㄱㄴㄷ ㅏㅑㅓ"),
    ("devanagari", "नमस्ते दुनिया"),
    ("devanagari", "मकान मालिक को बॉयलर के बारे में संदेश लिखें।"),
    ("devanagari", "क्षत्रिय ज्ञान श्री"),
    ("devanagari", "हिन्दी में लिखा गया वाक्य"),
    ("devanagari", "ऋषि"),
    ("thai", "สวัสดีครับ"),
    ("thai", "สวัสดีชาวโลก"),
    ("thai", "เขียนข้อความถึงเจ้าของบ้านเรื่องหม้อต้มน้ำเสีย"),
    ("thai", "ภาษาไทยมีวรรณยุกต์ ่ ้ ๊ ๋"),
    ("thai", "กรุงเทพมหานคร"),
    ("thai", "น้ำ ที่ นี่ ไม่ ได้"),
    ("thai", "ผู้ใหญ่บ้าน"),
    ("thai", "เด็กๆ เล่นกันอยู่ที่สนาม"),
    ("arabic-hebrew", "مرحبا بالعالم"),
    ("arabic-hebrew", "שלום עולם"),
    ("emoji", "I love it 😀👍🏽"),
    ("emoji", "Family: 👨‍👩‍👧‍👦 flags 🇹🇭🇯🇵"),
    ("emoji", "❤️ ✨ 🔥"),
    ("symbols", "→ ← ≤ ≥ ∑ ∫ √ ∞"),
    ("symbols", "“Curly quotes” and ‘single’ — dashes – too…"),
    ("symbols", "<|im_start|>user\nhi<|im_end|>"),
    ("whitespace", "\n\n\n"),
    ("whitespace", "a  b   c    d"),
]
FIXED_STRINGS = _BASE_STRINGS + [(c, " " + t) for c, t in _BASE_STRINGS]

RANDOM_SEED = 20261004
RANDOM_COUNT = 100
# Code point ranges the random strings draw from: ASCII, Latin-1 and Latin Extended, combining
# diacritical marks, Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai, Hangul Jamo, kana, CJK,
# Hangul syllables and emoji.
RANDOM_RANGES = [(0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF),
                 (0x590, 0x5FF), (0x600, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F), (0x1100, 0x11FF),
                 (0x3040, 0x30FF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0x1F300, 0x1F64F)]


def random_strings(seed: int = RANDOM_SEED, count: int = RANDOM_COUNT) -> list[str]:
    """`count` strings of 1 to 16 code points from RANDOM_RANGES, the same on every run."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        chars = []
        for _ in range(rng.randint(1, 16)):
            lo, hi = rng.choice(RANDOM_RANGES)
            chars.append(chr(rng.randint(lo, hi)))
        out.append("".join(chars))
    return out


# ---- small helpers --------------------------------------------------------------------------------

def fail(message: str) -> None:
    print(f"delta_triage: {message}", file=sys.stderr)
    sys.exit(2)


def say(message: str) -> None:
    print(f"delta_triage: {message}", flush=True)


def load_config() -> dict:
    path = HERE_DIR / "triage_config.json"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")
    if not isinstance(cfg, dict):
        fail(f"{path} is not a JSON object")
    missing = [k for k in CONFIG_KEYS if not isinstance(cfg.get(k), str) or not cfg[k].strip()]
    if missing:
        fail(f"triage_config.json is missing {missing}")
    for key in ("run_dir", "model_snapshot", "nearest_snapshot"):
        if not Path(cfg[key]).is_dir():
            fail(f"triage_config.json {key} is not a directory: {cfg[key]}")
    run_dir = os.path.realpath(cfg["run_dir"])
    if os.path.commonpath([run_dir, str(HERE_DIR)]) != run_dir:
        fail(f"this script's directory {HERE_DIR} is not inside run_dir {run_dir}; copy it into "
             "the stage directory and run it from there")
    return cfg


def with_revision(model_id: str, snapshot: str) -> str:
    """`model_id`, with `@<snapshot directory name>` added when it names no revision and the
    snapshot sits in a Hugging Face `snapshots/` folder."""
    snap = Path(os.path.abspath(snapshot.rstrip("/")))
    if "@" in model_id or snap.parent.name != "snapshots":
        return model_id
    return f"{model_id}@{snap.name}"


# The estimate comes from headers and configs alone, so it cannot name a sidecar. A measured
# weights+sidecar model has a matching backbone, which is what the estimate describes, so it is compared
# as weights-only. `measured_class` still records the real measured class.
SEEN_BY_HEADERS = {"weights+sidecar": "weights-only"}


def report_text(value):
    """A string from the research report, which is fetched text: one line, control characters removed,
    at most 300 characters. Anything that is not a string gives None. It is only recorded, never used as a
    path, a command or an instruction."""
    if not isinstance(value, str):
        return None
    return " ".join("".join(ch for ch in value if ch.isprintable() or ch.isspace()).split())[:300]


def base_research(cfg: dict, nearest_id: str, measured_class: str):
    """How the research report's estimate for the nearest model compared with this script's measurement.
    Only recorded; the class and the path come from the measurement. None when `triage_config.json` names no
    report. A report that cannot be read, or has no row for the nearest model, gives a null candidate."""
    path = cfg.get("base_candidates")
    if not isinstance(path, str) or not path.strip():
        return None
    try:
        too_big = Path(path).stat().st_size > REPORT_MAX_BYTES
    except OSError:
        too_big = False
    data, _ = (None, None) if too_big else read_json(Path(path))
    rows = data.get("candidates") if isinstance(data, dict) else None
    repo = nearest_id.split("@", 1)[0]
    row = (next((r for r in rows if isinstance(r, dict) and r.get("repo") == repo), None)
           if isinstance(rows, list) else None)
    estimate = row.get("class_estimate") if row else None
    # The file the run hands stage 0 is a view (it has a "selected" key) whose class_estimate is the class string.
    # The full report's class_estimate is an object with a "class" key. A string in the full shape is ignored.
    if isinstance(data, dict) and "selected" in data and isinstance(estimate, str):
        estimate = {"class": estimate}
    est = report_text(estimate.get("class") if isinstance(estimate, dict) else None) or None
    return {"candidate": ({"repo": repo, "confidence": report_text(row.get("confidence"))} if row else None),
            "estimated_class": est, "measured_class": measured_class,
            "agreed": (est == (SEEN_BY_HEADERS.get(measured_class, measured_class))) if est else None}


def read_json(path: Path):
    """(data, error). A missing file gives (None, None)."""
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"{path.name} is not readable JSON: {exc}"


def write_json(name: str, data) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short(values, limit: int = 8) -> str:
    """A list for a sentence: the first `limit` items, and how many more there are."""
    values = list(values)
    head = ", ".join(str(v) for v in values[:limit])
    return head + (f" and {len(values) - limit} more" if len(values) > limit else "")


# ---- config ---------------------------------------------------------------------------------------

def text_part(cfg: dict) -> dict:
    """The text model's config: `text_config` when the config nests one, else the whole config."""
    inner = cfg.get("text_config")
    return inner if isinstance(inner, dict) else cfg


def text_value(cfg: dict, key: str):
    """A text config value. A nested config may keep some text values at the top level."""
    text = text_part(cfg)
    if key in text:
        return text[key]
    if text is not cfg and key in cfg:
        return cfg[key]
    return ABSENT


def effective(cfg: dict, key: str):
    """The value the runtime uses: head_dim falls back to hidden_size / num_attention_heads."""
    value = text_value(cfg, key)
    if key == "head_dim" and value == ABSENT:
        hidden, heads = text_value(cfg, "hidden_size"), text_value(cfg, "num_attention_heads")
        if isinstance(hidden, int) and isinstance(heads, int) and heads:
            return hidden // heads
    return value


def compare_configs(model_dir: Path, base_dir: Path) -> dict:
    mcfg, merr = read_json(model_dir / "config.json")
    bcfg, berr = read_json(base_dir / "config.json")
    return compare_config_dicts(mcfg, bcfg, [e for e in (merr, berr) if e],
                                str(model_dir / "config.json"), str(base_dir / "config.json"))


def compare_config_dicts(mcfg, bcfg, errors: list, model_path: str, base_path: str) -> dict:
    """The config comparison on two already-read configs. `compare_configs` reads them from snapshot
    directories; the research package reads them from the hub."""
    out = {"model_config_path": model_path, "base_config_path": base_path, "errors": list(errors)}
    if not isinstance(mcfg, dict) or not isinstance(bcfg, dict):
        out["errors"].append("config.json is missing or not an object in "
                             + ("the new model" if not isinstance(mcfg, dict) else "the nearest model"))
        out["decisive_keys_equal"] = False
        return out
    mt, bt = text_part(mcfg), text_part(bcfg)
    out.update({
        "model_config_nested": mt is not mcfg, "base_config_nested": bt is not bcfg,
        "model_top_level_model_type": mcfg.get("model_type"), "base_top_level_model_type": bcfg.get("model_type"),
        "model_text_model_type": mt.get("model_type"), "base_text_model_type": bt.get("model_type"),
        "model_architectures": mcfg.get("architectures"), "base_architectures": bcfg.get("architectures"),
        "model_transformers_version": mcfg.get("transformers_version"),
        "base_transformers_version": bcfg.get("transformers_version"),
        "model_has_vision_config": "vision_config" in mcfg,
        "base_has_vision_config": "vision_config" in bcfg,
    })
    diffs = {}
    for key in sorted(set(mt) | set(bt)):
        if key in ("text_config", "vision_config"):
            continue
        a, b = mt.get(key, ABSENT), bt.get(key, ABSENT)
        if a != b:
            diffs[key] = {"model": a, "base": b}
    out["text_config_diffs"] = diffs
    out["text_config_diffs_bookkeeping_only"] = all(k in BOOKKEEPING for k in diffs)
    out["base_top_level_keys_not_in_model"] = sorted(set(bcfg) - set(mcfg))
    out["model_top_level_keys_not_in_base"] = sorted(set(mcfg) - set(bcfg))
    values = {}
    for key in DECISIVE + SHOWN:
        a, b = effective(mcfg, key), effective(bcfg, key)
        if key in SHOWN and a == ABSENT and b == ABSENT:
            continue
        values[key] = {"model": a, "base": b, "equal": a == b}
    out["text_key_values"] = values
    out["decisive_keys"] = list(DECISIVE)
    out["decisive_keys_equal"] = all(values[k]["equal"] for k in DECISIVE)
    out["decisive_keys_differing"] = [k for k in DECISIVE if not values[k]["equal"]]
    out["text_keys_all_equal"] = all(v["equal"] for v in values.values())
    return out


# ---- tensors --------------------------------------------------------------------------------------

def raw_header(path: Path) -> tuple[dict, int]:
    """(the whole header with data_offsets, the file offset where the tensor data starts). Reads only the header."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("shorter than 8 bytes")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > MAX_HEADER:
            raise ValueError(f"header length {n} is not plausible")
        blob = f.read(n)
    if len(blob) != n:
        raise ValueError(f"header is cut short ({len(blob)} of {n} bytes)")
    header = json.loads(blob.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return header, 8 + n


def safetensors_header(path: Path) -> dict:
    """{name: {"dtype", "shape"}} from a safetensors file's header. Reads only the header."""
    header, _ = raw_header(path)
    return {k: {"dtype": v.get("dtype"), "shape": v.get("shape")}
            for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}


INDEX_NAME = "model.safetensors.index.json"
BACKBONE_SHARD = re.compile(r"model(-\d+-of-\d+)?\.safetensors")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".npz")
BACKBONE_BIN = re.compile(r"pytorch_model.*\.bin")
MAX_LISTED_NAMES = 1000


def backbone_shards(snapshot: Path) -> list[str]:
    """The safetensors files that hold the backbone: the shards the model's index lists, or
    `model*.safetensors` when there is no index. A snapshot with odd names and no index treats every
    safetensors file as the backbone, as this script always did."""
    top = sorted(p.name for p in snapshot.iterdir() if p.is_file() and p.name.endswith(".safetensors"))
    index, _ = read_json(snapshot / INDEX_NAME)
    if isinstance(index, dict) and isinstance(index.get("weight_map"), dict):
        listed = set(index["weight_map"].values())
        return [n for n in top if n in listed]
    return [n for n in top if BACKBONE_SHARD.fullmatch(n)] or top


def read_tensors(snapshot: Path) -> tuple[list[str], dict, list[str]]:
    """(shard names, {tensor: {"dtype", "shape", "shard"}}, errors). Backbone shards only."""
    shards = backbone_shards(snapshot)
    tensors, errors = {}, []
    for shard in shards:
        try:
            for name, info in safetensors_header(snapshot / shard).items():
                tensors[name] = {**info, "shard": shard}
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            errors.append(f"{shard}: {exc}")
    return shards, tensors, errors


def normalize(name: str) -> str:
    """The vision-language layout's `model.language_model.` prefix becomes `model.`."""
    return "model." + name[len(LM_PREFIX):] if name.startswith(LM_PREFIX) else name


def group(names) -> dict:
    """Counts by the name up to its first number-free three parts, for a short summary."""
    out: dict[str, int] = {}
    for n in names:
        parts = []
        for p in n.split("."):
            if p.isdigit() or len(parts) == 2:
                break
            parts.append(p)
        key = ".".join(parts) or n
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def dtype_histogram(tensors: dict) -> dict:
    out: dict[str, int] = {}
    for info in tensors.values():
        out[str(info["dtype"])] = out.get(str(info["dtype"]), 0) + 1
    return dict(sorted(out.items()))


def compare_tensors(model_dir: Path, base_dir: Path) -> dict:
    mshards, mt, merr = read_tensors(model_dir)
    bshards, bt, berr = read_tensors(base_dir)
    return compare_tensor_maps(mt, bt, mshards=mshards, bshards=bshards, merr=merr, berr=berr,
                               model_path=str(model_dir), base_path=str(base_dir))


def _suffix_len(a: list[str], b: list[str]) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def align_leftovers(only_m: dict, only_b: dict) -> dict:
    """Pair the names the `model.language_model.` rule left over, the way a person does: by what the names share.

    {"pairs": {model name: base name}, "rules": [{"model_prefix", "base_prefix", "tensors", "same_shape"}]}. Both arguments map
    a name to its {"shape", "dtype"}. Two phases:

    1. A model name pairs with the base name that shares the longest run of trailing dotted parts, if the run is at least three
       parts, or at least two when one name is wholly the end of the other (`visual.pos_embed.weight` and
       `model.visual.pos_embed.weight`). A tie leaves the name unmatched, and so does a base name wanted by two model names. The
       layer number is part of the run, so a layer cannot pair with another layer.
    2. Pairs from phase 1 vote for prefix rewrites (`language_model` -> `model`). A rewrite with MIN_RULE_VOTES votes also pairs
       the names too short to match alone (`language_model.norm.weight`, `language_model.embed_tokens.weight`).
    """
    by_tail: dict[tuple, list[str]] = {}
    for b in only_b:
        by_tail.setdefault(tuple(b.split(".")[-2:]), []).append(b)
    wants: dict[str, str] = {}
    for m in only_m:
        ms = m.split(".")
        best, best_k, tie = None, 0, False
        for b in by_tail.get(tuple(ms[-2:]), []):
            bs = b.split(".")
            k = _suffix_len(ms, bs)
            if not (k >= 3 or (k >= 2 and k in (len(ms), len(bs)))):
                continue
            if k > best_k:
                best, best_k, tie = b, k, False
            elif k == best_k:
                tie = True
        if best is not None and not tie:
            wants[m] = best
    claimed = collections.Counter(wants.values())
    pairs = {m: b for m, b in wants.items() if claimed[b] == 1}

    def rule_of(m: str, b: str) -> tuple[str, str]:
        ms, bs = m.split("."), b.split(".")
        k = _suffix_len(ms, bs)
        return ".".join(ms[:len(ms) - k]), ".".join(bs[:len(bs) - k])

    votes = collections.Counter(rule_of(m, b) for m, b in pairs.items())
    used = set(pairs.values())
    for (mp, bp), n in votes.most_common():
        if n < MIN_RULE_VOTES:
            break
        for m in only_m:
            if m in pairs or (mp and not m.startswith(mp + ".")):
                continue
            cand = (bp + "." if bp else "") + (m[len(mp) + 1:] if mp else m)
            if cand in only_b and cand not in used:
                pairs[m] = cand
                used.add(cand)
    tally: dict[tuple, list[int]] = {}
    for m, b in pairs.items():
        row = tally.setdefault(rule_of(m, b), [0, 0])
        row[0] += 1
        row[1] += int(only_m[m]["shape"] == only_b[b]["shape"] and only_m[m]["dtype"] == only_b[b]["dtype"])
    rules = [{"model_prefix": mp, "base_prefix": bp, "tensors": n, "same_shape": same}
             for (mp, bp), (n, same) in sorted(tally.items(), key=lambda kv: (-kv[1][0], kv[0]))]
    return {"pairs": pairs, "rules": rules}


def _pair_names(mt: dict, bt: dict) -> dict:
    """Everything the name matching knows, so the comparison and `raw_pair_map` cannot disagree."""
    mraw = {normalize(k): k for k in mt}
    braw = {normalize(k): k for k in bt}
    mn = {k: mt[v] for k, v in mraw.items()}
    bn = {k: bt[v] for k, v in braw.items()}
    common = sorted(set(mn) & set(bn))
    only_m, only_b = sorted(set(mn) - set(bn)), sorted(set(bn) - set(mn))
    al = align_leftovers({k: mn[k] for k in only_m}, {k: bn[k] for k in only_b})
    pairs = [(k, k) for k in common] + sorted(al["pairs"].items())
    gone_m, gone_b = set(al["pairs"]), set(al["pairs"].values())
    return {"mn": mn, "bn": bn, "mraw": mraw, "braw": braw, "pairs": pairs, "aligned": al,
            "only_m": [k for k in only_m if k not in gone_m], "only_b": [k for k in only_b if k not in gone_b]}


def raw_pair_map(mt: dict, bt: dict) -> dict:
    """{raw model tensor name: raw base tensor name} for every pair the comparison found, aligned pairs included."""
    p = _pair_names(mt, bt)
    return {p["mraw"][m]: p["braw"][b] for m, b in p["pairs"]}


def raw_rules(raw_pairs: dict) -> list[dict]:
    """Prefix rewrites that turn the model's tensor names into the nearest model's, from {raw model name: raw base name}: {"from", "to",
    "tensors"} where names starting with `from.` are called `to.` there (`to` is empty when the nearest model has no prefix). A name that
    already matches makes no rule. The reference loader turns these into transformers' `key_mapping` when a load misses keys."""
    tally: collections.Counter = collections.Counter()
    for m, b in raw_pairs.items():
        ms, bs = m.split("."), b.split(".")
        k = _suffix_len(ms, bs)
        mp, bp = ms[:len(ms) - k], bs[:len(bs) - k]
        if mp:
            tally[(".".join(mp), ".".join(bp))] += 1
        elif bp:
            tally[(ms[0], ".".join(bp + [ms[0]]))] += 1
    return [{"from": f, "to": t, "tensors": n} for (f, t), n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))]


def _dtype_difference_tolerated(m: dict, b: dict) -> bool:
    """True for a float-to-float dtype difference on a small tensor of equal shape (see DTYPE_TOLERANCE_ELEMENTS)."""
    if m["dtype"] not in ITEMSIZE or b["dtype"] not in ITEMSIZE or m["shape"] != b["shape"]:
        return False
    n = 1
    for d in b["shape"] or []:
        n *= int(d)
    return n <= DTYPE_TOLERANCE_ELEMENTS


def tolerated_dtype_hazards(t: dict, ev) -> list:
    """One hazard when the new model keeps small tensors in another float dtype than the nearest model (see DTYPE_TOLERANCE_ELEMENTS)."""
    ok = [x for x in t["dtype_diffs_normalized"] if x["tolerated"]]
    if not ok:
        return []
    kinds = sorted({f"{x['model']} against {x['base']}" for x in ok})
    return [{"area": "other", "evidence": [ev("tensor-compare.json")],
             "finding": (f"{len(ok)} tensors differ from the nearest model only in float dtype ({'; '.join(kinds)}), each small with the same "
                         f"shape, for example {short([x['name'] for x in ok], 3)}. They did not make the path a full port. The weights are "
                         "served as they are, and whether the runtime casts them or uses them in the stored dtype was not checked. One model "
                         "was served this way on hardware (sabicap/sabi-noema, 31 of 32 tokens equal to the CPU reference); a larger or "
                         "quantized dtype difference was not tested and still makes a full port.")}]


def compare_tensor_maps(mt: dict, bt: dict, *, mshards: list, bshards: list, merr: list, berr: list,
                        model_path: str, base_path: str) -> dict:
    """The tensor comparison on two already-read {name: {"dtype", "shape", "shard"}} maps. The research
    package builds the maps from safetensors headers fetched by range request."""
    m_raw, b_raw = set(mt), set(bt)
    p = _pair_names(mt, bt)
    mn, bn, al = p["mn"], p["bn"], p["aligned"]
    only_m, only_b = p["only_m"], p["only_b"]
    shape_diffs = sorted(({"name": b, "model": mn[m]["shape"], "base": bn[b]["shape"]}
                          for m, b in p["pairs"] if mn[m]["shape"] != bn[b]["shape"]), key=lambda x: x["name"])
    dtype_diffs = sorted(({"name": b, "model": mn[m]["dtype"], "base": bn[b]["dtype"],
                           "shape": bn[b]["shape"], "tolerated": _dtype_difference_tolerated(mn[m], bn[b])}
                          for m, b in p["pairs"] if mn[m]["dtype"] != bn[b]["dtype"]), key=lambda x: x["name"])
    return {
        "model": {"path": model_path, "shards": mshards, "num_tensors": len(mt),
                  "dtype_histogram": dtype_histogram(mt),
                  "uses_language_model_prefix": any(k.startswith(LM_PREFIX) for k in mt)},
        "base": {"path": base_path, "shards": bshards, "num_tensors": len(bt),
                 "dtype_histogram": dtype_histogram(bt),
                 "uses_language_model_prefix": any(k.startswith(LM_PREFIX) for k in bt)},
        "errors": merr + berr,
        "raw_name_sets": {"common": len(m_raw & b_raw), "only_in_model": sorted(m_raw - b_raw),
                          "only_in_base": sorted(b_raw - m_raw)},
        "normalized_name_sets": {
            "note": (f"names normalized by changing the prefix '{LM_PREFIX}' to 'model.' in both models; names still "
                     "unmatched were then paired when they share the same trailing parts and differ only by a prefix "
                     "(see aligned)"),
            "common": len(p["pairs"]), "only_in_model": only_m, "only_in_base": only_b,
            "aligned": {"pairs": len(al["pairs"]), "rules": al["rules"],
                        "examples": [list(x) for x in sorted(al["pairs"].items())[:5]],
                        "all_pairs": dict(sorted(al["pairs"].items())),
                        "raw_rules": raw_rules({p["mraw"][m]: p["braw"][b] for m, b in p["pairs"]})},
            "only_in_model_groups": group(only_m), "only_in_base_groups": group(only_b),
            "unexplained_only_in_model": [k for k in only_m if not EXPLAINED_EXTRA.search(k)],
            "unexplained_only_in_base": [k for k in only_b if not EXPLAINED_EXTRA.search(k)]},
        "shape_diffs_normalized": shape_diffs,
        "dtype_diffs_normalized": dtype_diffs,
        "mtp_tensors": {"model": sum(1 for k in mt if MTP_NAME.search(k)),
                        "base": sum(1 for k in bt if MTP_NAME.search(k))},
        "vision_tensors": {"model": sum(1 for k in mn if re.search(r"(^|\.)(visual|vision)", k)),
                           "base": sum(1 for k in bn if re.search(r"(^|\.)(visual|vision)", k))},
    }


# ---- values: are the analogues the same weights, a fine-tune of them, or unrelated? ---------------------------------------------

def tensor_index(snapshot: Path) -> dict:
    """{name: {"file", "dtype", "shape", "start", "end"}} for the backbone shards, with absolute byte offsets."""
    out = {}
    for shard in backbone_shards(snapshot):
        try:
            header, base = raw_header(snapshot / shard)
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        for name, v in header.items():
            if name == "__metadata__" or not isinstance(v, dict) or not isinstance(v.get("data_offsets"), list):
                continue
            lo, hi = v["data_offsets"]
            out[name] = {"file": snapshot / shard, "dtype": v.get("dtype"), "shape": v.get("shape") or [],
                         "start": base + lo, "end": base + hi}
    return out


def decode_values(buf: bytes, dtype: str):
    """floats from raw bytes, or None for a dtype this does not read. Plain Python: a few thousand values is cheap."""
    size = ITEMSIZE.get(dtype)
    if size is None:
        return None
    n = len(buf) // size
    if dtype == "BF16":
        return [struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in struct.unpack(f"<{n}H", buf[:n * 2])]
    return list(struct.unpack(f"<{n}{'e' if dtype == 'F16' else 'f'}", buf[:n * size]))


def read_prefix(entry: dict, elements: int):
    size = ITEMSIZE.get(entry["dtype"])
    if size is None:
        return None
    numel = 1
    for d in entry["shape"]:
        numel *= int(d)
    with open(entry["file"], "rb") as f:
        f.seek(entry["start"])
        return decode_values(f.read(min(elements, numel) * size), entry["dtype"])


def relation(a: list[float], b: list[float]) -> tuple[str, float | None]:
    if a == b:
        return "identical", 1.0
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return "unrelated", None
    cos = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return ("near" if cos >= NEAR_COSINE else "changed" if cos >= CHANGED_COSINE else "unrelated"), cos


def sample_value_diffs(model_dir: Path, base_dir: Path, pairs: dict) -> dict:
    """Compare the first SAMPLE_ELEMENTS values of SAMPLE_TENSORS paired tensors. `pairs` maps a raw model name to a raw base name.
    identical: equal; near: cosine at least NEAR_COSINE (what a fine-tune looks like); changed; unrelated."""
    mi, bi = tensor_index(model_dir), tensor_index(base_dir)
    names = sorted(k for k in pairs if k in mi and pairs[k] in bi)
    chosen = names if len(names) <= SAMPLE_TENSORS else [names[i * len(names) // SAMPLE_TENSORS] for i in range(SAMPLE_TENSORS)]
    samples, skipped = [], []
    for m in chosen:
        b = pairs[m]
        if mi[m]["shape"] != bi[b]["shape"]:
            skipped.append(f"{m}: shapes differ ({mi[m]['shape']} and {bi[b]['shape']})")
            continue
        a, c = read_prefix(mi[m], SAMPLE_ELEMENTS), read_prefix(bi[b], SAMPLE_ELEMENTS)
        if a is None or c is None:
            skipped.append(f"{m}: dtype {mi[m]['dtype']} / {bi[b]['dtype']} is not read")
            continue
        rel, cos = relation(a, c)
        diffs = [abs(x - y) for x, y in zip(a, c)]
        samples.append({"model_tensor": m, "base_tensor": b, "elements": len(a), "relation": rel,
                        "cosine": None if cos is None else round(cos, 6),
                        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0, "max_abs_diff": max(diffs, default=0.0),
                        "mean_abs_base": sum(abs(y) for y in c) / len(c) if c else 0.0})
    summary = {k: sum(1 for x in samples if x["relation"] == k) for k in ("identical", "near", "changed", "unrelated")}
    return {"samples": samples, "summary": summary, "skipped": skipped,
            "note": (f"the first {SAMPLE_ELEMENTS} values of up to {SAMPLE_TENSORS} paired tensors; a fine-tune is 'near' "
                     f"(cosine at least {NEAR_COSINE}), identical weights are 'identical'")}


# ---- a restricted head: no lm_head, and a sidecar whose rows are copies of some of the base head's rows --------------------------

def _np_decode(buf: bytes, dtype: str):
    if dtype == "BF16":
        return (np.frombuffer(buf, dtype="<u2").astype("<u4") << 16).view("<f4")
    if dtype == "F16":
        return np.frombuffer(buf, dtype="<f2").astype("<f4")
    if dtype == "F32":
        return np.frombuffer(buf, dtype="<f4")
    return None


def _unit_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def detect_restricted_head(model_dir: Path, base_dir: Path, sc: dict) -> dict:
    """Is a sidecar a head restricted to some of the nearest model's lm_head rows (a decision model's answer codes, for example)?
    restricted is True when every row of a sidecar matrix is a copy (cosine at least HEAD_MATCH_COSINE) of a base head row;
    None when the check could not run (no numpy)."""
    out = {"restricted": False, "rows": None, "base_rows": None, "matched_rows": 0, "token_ids": [], "min_cosine": None,
           "sidecar": None, "tensor": None, "note": ""}
    sidecars = [e for e in sc.get("sidecars", []) if str(e.get("file", "")).endswith(".safetensors")]
    if not sidecars:
        out["note"] = "the new model has no safetensors sidecar"
        return out
    if np is None:
        out.update(restricted=None, note="numpy is not installed, so the restricted-head check was skipped")
        return out
    if any(HEAD_NAME.search(n) for n in tensor_index(model_dir)):
        out["note"] = "the new model has its own lm_head, so no sidecar stands in for it"
        return out
    head = tensor_index(base_dir).get("lm_head.weight")
    if head is None or len(head["shape"]) != 2 or head["dtype"] not in ITEMSIZE:
        out["note"] = "the nearest model has no readable lm_head.weight to compare with"
        return out
    vocab, dim = head["shape"]
    size = ITEMSIZE[head["dtype"]]
    best_try = None
    unreadable = []
    for entry in sidecars:
        try:
            header, base_off = raw_header(model_dir / entry["file"])
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        for name, v in sorted(header.items()):
            shape = v.get("shape") if isinstance(v, dict) else None
            if name == "__metadata__" or not shape or len(shape) != 2 or shape[1] != dim or shape[0] >= vocab \
                    or v.get("dtype") not in ITEMSIZE:
                continue
            rows = shape[0]
            try:
                # A truncated file, a header with no offsets or offsets that do not fit the shape each raise here. The check then
                # says it could not run (restricted None), where it used to end stage 0 with a traceback.
                lo, hi = v["data_offsets"]
                with open(model_dir / entry["file"], "rb") as f:
                    f.seek(base_off + lo)
                    rowsmat = _unit_rows(_np_decode(f.read(hi - lo), v["dtype"]).reshape(rows, dim))
                best = np.full(rows, -2.0, dtype="float32")
                best_idx = np.zeros(rows, dtype="int64")
                with open(head["file"], "rb") as f:
                    for r0 in range(0, vocab, HEAD_CHUNK_ROWS):
                        n = min(HEAD_CHUNK_ROWS, vocab - r0)
                        f.seek(head["start"] + r0 * dim * size)
                        chunk = _unit_rows(_np_decode(f.read(n * dim * size), head["dtype"]).reshape(n, dim))
                        sim = rowsmat @ chunk.T
                        j, val = sim.argmax(axis=1), sim.max(axis=1)
                        better = val > best
                        best[better], best_idx[better] = val[better], j[better] + r0
            except (OSError, ValueError, KeyError, TypeError) as exc:
                unreadable.append(f"{entry['file']} {name}: {type(exc).__name__}: {exc}"[:300])
                continue
            matched = int((best >= HEAD_MATCH_COSINE).sum())
            found = {"restricted": bool(matched == rows and rows >= 2), "rows": rows, "base_rows": vocab,
                     "matched_rows": matched, "token_ids": [int(i) for i in best_idx][:MAX_LISTED_NAMES],
                     "min_cosine": round(min(1.0, float(best.min())), 6), "sidecar": entry["file"], "tensor": name,
                     "note": ("every row of the sidecar matrix is a copy of a row of the nearest model's lm_head.weight"
                              if matched == rows else f"{matched} of {rows} rows match a row of lm_head.weight")}
            if found["restricted"]:
                return found
            if best_try is None or matched > best_try["matched_rows"]:
                best_try = found
    if unreadable:
        # Something could not be read, so "not restricted" would be a guess. A match found before the failure was returned above.
        return {**(best_try or out), "restricted": None,
                "note": "the restricted-head check could not run to the end: " + "; ".join(unreadable[:3]) + " (a sidecar or the nearest "
                        "model's lm_head.weight could not be read in full; the file may be truncated)"}
    if best_try is not None:
        return best_try
    out["note"] = "no sidecar matrix has the head's width and fewer rows"
    return out


def readout_facts(model_dir: Path, code: list[dict]) -> dict:
    """How a head is read, from the model's own files, without running anything: small top-level JSON files that declare an
    `attention_mode` (with their pooling, temperature and code count), and the lines of the listed code files that mention masks, hooks
    and the readout. A line number is a place to read, not a finding: the code decides what the mask is."""
    declared, hints, skipped = [], [], []
    for p in sorted(model_dir.glob("*.json")):
        try:
            if not p.is_file() or p.stat().st_size > READOUT_JSON_MAX:
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("attention_mode"), str):
            codes = d.get("codes")
            declared.append({"file": p.name, "attention_mode": d["attention_mode"], "pooling": d.get("pooling"),
                             "temperature": d.get("temperature"), "num_codes": len(codes) if isinstance(codes, list) else None})
    for entry in code:
        p = model_dir / entry["file"]
        try:
            if p.stat().st_size > CODE_SCAN_MAX:
                skipped.append(f"{entry['file']}: over {CODE_SCAN_MAX} bytes, not scanned")
                continue
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            skipped.append(f"{entry['file']}: unreadable")
            continue
        for name, pattern in CODE_HINTS:
            hit = [i + 1 for i, line in enumerate(lines) if pattern.search(line)]
            if hit:
                hints.append({"file": entry["file"], "hint": name, "lines": hit[:MAX_HINT_LINES], "count": len(hit)})
    return {"declared": declared, "code_hints": hints, "skipped": skipped,
            "note": "declared: small JSON files that state an attention_mode; code_hints: lines in the listed code files that mention it. "
                    "Nothing was run."}


def token_names(base_dir: Path, ids: list[int], limit: int = 400) -> list[str]:
    """The nearest model's token strings for some ids, from tokenizer.json; [] when it cannot be read."""
    try:
        tok = json.loads((base_dir / "tokenizer.json").read_text(encoding="utf-8"))
        wanted = set(ids[:limit])
        names = {i: t for t, i in (tok.get("model", {}).get("vocab") or {}).items() if i in wanted}
        for a in tok.get("added_tokens") or []:
            if a.get("id") in wanted:
                names[a["id"]] = a.get("content")
        return [str(names.get(i, "?")) for i in ids[:limit]]
    except (OSError, ValueError, AttributeError, TypeError):
        return []


# ---- sidecars and code files -----------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_sidecars(model_dir: Path, base_dir: Path, backbone_names: set[str]) -> dict:
    """Weights files in the new model that are not backbone shards and that the nearest model lacks.
    `problems` says why a sidecar cannot be trusted (unreadable, or its names overlap the backbone)."""
    shards = set(backbone_shards(model_dir))
    base_files = {p.name for p in base_dir.iterdir() if p.is_file()}
    backbone = backbone_names | {normalize(n) for n in backbone_names}
    sidecars, problems = [], []
    for p in sorted(model_dir.iterdir()):
        if (not p.is_file() or not p.name.endswith(WEIGHT_SUFFIXES) or p.name in shards
                or p.name in base_files or BACKBONE_BIN.fullmatch(p.name)):
            continue
        entry = {"file": p.name, "size": p.stat().st_size, "sha256": file_sha256(p), "num_tensors": None,
                 "dtypes": [], "tensor_names": []}
        if p.name.endswith(".safetensors"):
            try:
                header = safetensors_header(p)
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                problems.append(f"sidecar {p.name} cannot be read: {exc}")
            else:
                names = sorted(header)
                entry.update(num_tensors=len(names), dtypes=sorted({v["dtype"] for v in header.values()}),
                             tensor_names=names[:MAX_LISTED_NAMES], names_truncated=len(names) > MAX_LISTED_NAMES)
                clash = sorted(n for n in names if n in backbone or normalize(n) in backbone)
                if clash:
                    problems.append(f"sidecar {p.name} has {len(clash)} tensor names that overlap the "
                                    f"backbone: {short(clash, 5)}")
        else:
            problems.append(f"sidecar {p.name} is not a safetensors file, so its tensors cannot be read "
                            "without loading it")
        sidecars.append(entry)
    return {"sidecars": sidecars, "problems": problems,
            "note": "a sidecar is a weights file outside the backbone shards that the nearest model lacks"}


def code_files(model_dir: Path) -> list[dict]:
    """Every `.py` file in the model directory, down MAX_CODE_DEPTH levels, listed with its sha256 (at most MAX_CODE_FILES, sorted
    by path). A model that ships its own inference code usually keeps it under source/ or src/. Nothing here imports or runs it."""
    found = []
    for root, dirs, files in os.walk(model_dir):
        depth = len(Path(root).relative_to(model_dir).parts)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_CODE_DIRS and not d.startswith(".") and depth < MAX_CODE_DEPTH)
        for name in files:
            p = Path(root) / name
            if name.endswith(".py") and p.is_file():           # is_file follows a link: a cache snapshot is all links
                found.append(p)
    found.sort(key=lambda p: p.relative_to(model_dir).as_posix())
    return [{"file": p.relative_to(model_dir).as_posix(), "sha256": file_sha256(p)} for p in found[:MAX_CODE_FILES]]


def sidecar_finding(sc: dict) -> str:
    parts = [f"{e['file']} ({e['size'] / 1e6:.1f} MB, {e['num_tensors']} tensors, sha256 {e['sha256'][:12]})"
             for e in sc["sidecars"]]
    s = ("The new model ships weights outside the backbone that the nearest model's runtime does not load: "
         + "; ".join(parts) + ".")
    if sc["problems"]:
        s += " Problems: " + "; ".join(sc["problems"]) + "."
    return s


# ---- tokenizer ------------------------------------------------------------------------------------

def merge_pair(m):
    """A merge as a (left, right) tuple, whether written as a 2-element list or as "left right"."""
    if isinstance(m, (list, tuple)) and len(m) == 2:
        return (str(m[0]), str(m[1]))
    if isinstance(m, str) and " " in m:
        a, b = m.split(" ", 1)
        return (a, b)
    return (str(m), "")


def vocab_items(model: dict):
    """The vocab as a comparable value: a dict for BPE and WordLevel, a list for Unigram."""
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        return dict(vocab)
    if isinstance(vocab, list):
        return [tuple(v) if isinstance(v, list) else v for v in vocab]
    return None


def regexes(obj) -> list[str]:
    """Every `{"Regex": ...}` pattern inside a tokenizer component."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "Regex" and isinstance(v, str):
                out.append(v)
            else:
                out += regexes(v)
    elif isinstance(obj, list):
        for v in obj:
            out += regexes(v)
    return out


def flatten(obj, prefix: str = "") -> dict:
    """{dotted path: leaf value} for a nested component, to name the fields that differ."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def field_diffs(a, b) -> dict:
    fa, fb = flatten(a), flatten(b)
    return {k: {"model": fa.get(k, ABSENT), "base": fb.get(k, ABSENT)}
            for k in sorted(set(fa) | set(fb)) if fa.get(k, ABSENT) != fb.get(k, ABSENT)}


SPECIAL_TOKEN_NAMES = ("eos_token", "bos_token", "pad_token", "unk_token")


def template_variants(snapshot: Path) -> tuple[str | None, str | None]:
    """(the template inside tokenizer_config.json, the text of chat_template.jinja); None for one that is absent."""
    jinja = snapshot / "chat_template.jinja"
    file_text = jinja.read_text(encoding="utf-8", errors="replace") if jinja.is_file() else None
    cfg, _ = read_json(snapshot / "tokenizer_config.json")
    t = cfg.get("chat_template") if isinstance(cfg, dict) else None
    if isinstance(t, list):            # named templates: [{"name": ..., "template": ...}]
        t = json.dumps(t, sort_keys=True, ensure_ascii=False)
    return (t if isinstance(t, str) else None), file_text


def chat_template(snapshot: Path) -> tuple[str | None, str | None]:
    """(source file, template text): chat_template.jinja first, then tokenizer_config.json. transformers 5.19.0 loads the
    file when both exist (checked on Qwen/Qwen3-Coder-Next, whose two templates differ)."""
    inline, file_text = template_variants(snapshot)
    if file_text is not None:
        return "chat_template.jinja", file_text
    return ("tokenizer_config.json", inline) if inline is not None else (None, None)


def special_tokens(snapshot: Path) -> dict:
    """eos, bos, pad and unk tokens as tokenizer_config.json writes them (a string, or an object with `content`);
    None for one that is absent or unreadable."""
    cfg, _ = read_json(snapshot / "tokenizer_config.json")
    cfg = cfg if isinstance(cfg, dict) else {}
    out = {}
    for name in SPECIAL_TOKEN_NAMES:
        v = cfg.get(name)
        if isinstance(v, dict):
            v = v.get("content")
        out[name] = v if isinstance(v, str) else None
    return out


def tokenizer_side(snapshot: Path) -> tuple[dict | None, dict, str | None]:
    """(tokenizer.json data, summary, error)."""
    data, err = read_json(snapshot / "tokenizer.json")
    src, tmpl = chat_template(snapshot)
    inline, file_text = template_variants(snapshot)
    summary = {"chat_template_source": src, "chat_template_sha256": sha256_text(tmpl) if tmpl else None,
               "chat_template_inline_sha256": sha256_text(inline) if inline else None,
               "chat_template_file_sha256": sha256_text(file_text) if file_text else None,
               "special_tokens": special_tokens(snapshot)}
    if not isinstance(data, dict):
        return None, summary, err or "tokenizer.json is missing"
    model = data.get("model") or {}
    vocab = vocab_items(model)
    summary.update({"model_type": model.get("type"),
                    "vocab_size": len(vocab) if vocab is not None else None,
                    "num_merges": len(model.get("merges") or []),
                    "num_added_tokens": len(data.get("added_tokens") or [])})
    return data, summary, None


def compare_tokenizers(model_dir: Path, base_dir: Path) -> dict:
    md, ms, merr = tokenizer_side(model_dir)
    bd, bs, berr = tokenizer_side(base_dir)
    st_diffs = {k: {"model": ms["special_tokens"][k], "base": bs["special_tokens"][k]}
                for k in SPECIAL_TOKEN_NAMES if ms["special_tokens"][k] != bs["special_tokens"][k]}
    out = {"model": ms, "base": bs, "errors": [e for e in (merr, berr) if e],
           "chat_template_equal": ms["chat_template_sha256"] == bs["chat_template_sha256"],
           "special_tokens_equal": not st_diffs, "special_tokens_diffs": st_diffs}
    if md is None or bd is None:
        return out
    mm, bm = md.get("model") or {}, bd.get("model") or {}
    m_merges, b_merges = mm.get("merges") or [], bm.get("merges") or []
    m_added = {(t.get("content"), t.get("id"), bool(t.get("special"))) for t in md.get("added_tokens") or []}
    b_added = {(t.get("content"), t.get("id"), bool(t.get("special"))) for t in bd.get("added_tokens") or []}
    m_ids = {t[0]: t[1] for t in m_added}
    b_ids = {t[0]: t[1] for t in b_added}
    out.update({
        "vocab_size_equal": ms["vocab_size"] == bs["vocab_size"],
        "vocab_equal": vocab_items(mm) == vocab_items(bm),
        "merges_equal_raw": m_merges == b_merges,
        "merges_equal": [merge_pair(m) for m in m_merges] == [merge_pair(m) for m in b_merges],
        "merges_format": {"model": type(m_merges[0]).__name__ if m_merges else None,
                          "base": type(b_merges[0]).__name__ if b_merges else None},
        "added_tokens_equal": m_added == b_added,
        "added_tokens_only_in_model": sorted(c for c in m_ids if c not in b_ids),
        "added_tokens_only_in_base": sorted(c for c in b_ids if c not in m_ids),
        "added_token_id_diffs": {c: {"model": m_ids[c], "base": b_ids[c]}
                                 for c in sorted(set(m_ids) & set(b_ids)) if m_ids[c] != b_ids[c]},
    })
    for part in ("normalizer", "pre_tokenizer", "decoder", "post_processor"):
        a, b = md.get(part), bd.get(part)
        out[f"{part}_equal"] = a == b
        if a != b:
            out[f"{part}_field_diffs"] = field_diffs(a, b)
    out["pre_tokenizer_regex_model"] = regexes(md.get("pre_tokenizer"))
    out["pre_tokenizer_regex_base"] = regexes(bd.get("pre_tokenizer"))
    out["pre_tokenizer_model"] = md.get("pre_tokenizer")
    out["pre_tokenizer_base"] = bd.get("pre_tokenizer")
    out["decoder_model"] = md.get("decoder")
    out["decoder_base"] = bd.get("decoder")
    return out


def encode_test(model_dir: Path, base_dir: Path) -> dict:
    """Encode every test string with both tokenizers and count the strings whose ids differ."""
    fixed = list(FIXED_STRINGS)
    rand = [("random", s) for s in random_strings()]
    out = {"fixed": len(fixed), "random": len(rand), "total": len(fixed) + len(rand),
           "random_seed": RANDOM_SEED, "add_special_tokens": False}
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        return {**out, "ran": False, "reason": f"the `tokenizers` package is not importable: {exc}"}
    try:
        tm = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tb = Tokenizer.from_file(str(base_dir / "tokenizer.json"))
    except Exception as exc:  # noqa: BLE001 - any load failure goes into the evidence file
        return {**out, "ran": False, "reason": f"a tokenizer.json did not load: {exc}"}
    differing, by_cat, examples = 0, {}, []
    for cat, text in fixed + rand:
        a = tm.encode(text, add_special_tokens=False).ids
        b = tb.encode(text, add_special_tokens=False).ids
        if a != b:
            differing += 1
            by_cat[cat] = by_cat.get(cat, 0) + 1
            if len(examples) < 25:
                examples.append({"category": cat, "text": text, "model_ids": a, "base_ids": b})
    out.update({"ran": True, "differing": differing, "fraction_differing": round(differing / out["total"], 4),
                "by_category": dict(sorted(by_cat.items())), "examples": examples,
                "fixed_strings": [t for _, t in fixed], "random_strings": [t for _, t in rand]})
    return out


# ---- files, generation config, license, disk -----------------------------------------------------

def file_side(snapshot: Path) -> dict:
    files = {}
    for p in sorted(snapshot.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(snapshot))] = p.stat().st_size
    shards = [n for n in files if n.endswith(".safetensors")]
    total = sum(files[n] for n in shards)
    return {"path": str(snapshot), "num_files": len(files), "weight_shards": shards,
            "num_weight_shards": len(shards), "weight_shard_total_bytes": total,
            "weight_shard_total_gib": round(total / GIB, 3), "all_files": files}


def compare_files(model_dir: Path, base_dir: Path) -> dict:
    m, b = file_side(model_dir), file_side(base_dir)
    mf, bf = set(m["all_files"]), set(b["all_files"])
    return {"model": m, "base": b,
            "files_only_in_model": sorted(n for n in mf - bf if not n.endswith(".safetensors")),
            "files_only_in_base": sorted(n for n in bf - mf if not n.endswith(".safetensors")),
            "shards_only_in_model": sorted(n for n in mf - bf if n.endswith(".safetensors")),
            "shards_only_in_base": sorted(n for n in bf - mf if n.endswith(".safetensors")),
            "common_files": sorted(mf & bf),
            "common_files_with_different_size": {
                n: {"model": m["all_files"][n], "base": b["all_files"][n]}
                for n in sorted(mf & bf) if m["all_files"][n] != b["all_files"][n]}}


def front_matter(readme: str) -> dict:
    """The simple `key: value` lines of a model card's YAML front matter."""
    if not readme.startswith("---"):
        return {}
    end = readme.find("\n---", 3)
    if end < 0:
        return {}
    out = {}
    for line in readme[3:end].splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


# LICENSE file openings and the identifier each one means, for a card without a license field.
LICENSE_HINTS = (("Apache License", "apache-2.0"), ("Attribution-NonCommercial 4.0", "cc-by-nc-4.0"),
                 ("CC BY-NC 4.0", "cc-by-nc-4.0"), ("MIT License", "mit"),
                 ("Attribution 4.0 International", "cc-by-4.0"))


def license_side(snapshot: Path) -> dict:
    readme = snapshot / "README.md"
    lic = next((snapshot / n for n in ("LICENSE", "LICENSE.txt", "LICENSE.md") if (snapshot / n).is_file()), None)
    fm = front_matter(readme.read_text(encoding="utf-8", errors="replace")) if readme.is_file() else {}
    head = lic.read_text(encoding="utf-8", errors="replace")[:400] if lic else None
    field = fm.get("license")
    guess = next((ident for hint, ident in LICENSE_HINTS if head and hint in head), None)
    ident = field.lower() if field else guess
    return {"readme_license_field": field, "readme_license_name": fm.get("license_name"),
            "license_file": lic.name if lic else None, "license_file_head": head,
            "license_from_file_text": guess, "license": ident,
            "source": "model card front matter" if field else ("LICENSE file" if guess else None)}


# ---- the scores the authors report (issue 19) ----------------------------------------------------------------------
#
# extract_reported reads the model card's own structured forms and nothing else: the front matter's `model-index`
# block, markdown tables and HTML tables. It reads text it is given. It opens no connection and runs no model, and it
# never reads prose ("reaches 84.2 on GSM8K" is not a table). A row counts only when it is about THIS model: in a
# comparison table the row of another model is somebody else's score. Every row quotes the card's text and line.
# Anything it will not read goes in `skipped` with the reason. The result is a draft: nobody has confirmed it, and no
# card shows it until `tt-orchard reported accept` copies chosen rows into reported-results.json.

MAX_README_BYTES = 1 << 20          # characters of the card that are read
MAX_TABLES = 40                     # tables read from one card
MAX_REPORTED_ROWS = 60              # the cap orchard/package.py puts on reported-results.json
MAX_REPORTED_NAME, MAX_REPORTED_NOTE, MAX_REPORTED_VALUE = 100, 300, 40
MAX_SKIPPED = 200                   # skipped entries listed (the rest are counted)
MAX_QUOTE = 300                     # characters of a card line quoted in a row
MAX_YAML_LINES, MAX_YAML_DEPTH = 4000, 12

PLAIN_NUMBER = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)%?$")
NO_RESULT = {"", "-", "--", "–", "—", "n/a", "na", "none"}
# A table about the model's speed, or one measured on Tenstorrent hardware or by this harness, is not the authors' report.
MEASURED_HERE = re.compile(r"measured|reproduc|tenstorrent|blackhole|wormhole|quietbox|orchard|\bp[0-9]{3}x?[0-9]*\b|"
                           r"tok(?:en)?s?\s*/\s*s|tokens per second", re.I)
GENERIC_FIRST = {"", "benchmark", "benchmarks", "metric", "metrics", "task", "tasks", "dataset", "datasets", "eval",
                 "evals", "evaluation", "test", "name"}
MODEL_FIRST = {"model", "models", "name", "system", "method", "variant"}
GENERIC_SCORE = {"score", "scores", "result", "results", "reported"}
# Headers of columns that say what a row measures. They belong to the row's name (a CER row is not a WER row).
LABEL_HEADERS = {"benchmark", "benchmarks", "metric", "metrics", "task", "tasks", "dataset", "datasets", "category", "split",
                 "shots", "shot", "nshots", "unit", "testset", "setting", "subset", "language", "measure", "suite"}


class UnsupportedYaml(Exception):
    pass


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _clean_cell(text: str) -> str:
    """A table cell without its markdown wrapping: links keep their text, emphasis and code marks are removed."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text.strip())
    for _ in range(2):
        text = text.strip()
        for mark in ("**", "__", "`", "*", "_"):
            if len(text) > 2 * len(mark) and text.startswith(mark) and text.endswith(mark):
                text = text[len(mark):-len(mark)]
    return re.sub(r"\s+", " ", text).strip()


def _plain_number(text: str) -> bool:
    if not PLAIN_NUMBER.match(text):
        return False
    try:
        return math.isfinite(float(text.rstrip("%")))
    except (ValueError, OverflowError):
        return False


def _quote(text: str) -> str:
    return text.strip()[:MAX_QUOTE]


class _Rows:
    """The rows found so far, in order, with the checks every row passes however it was read."""

    def __init__(self, model_id: str):
        base = model_id.split("@")[0]
        self.names = {_norm(base), _norm(base.split("/")[-1])} - {""}
        self.rows: list[dict] = []
        self.skipped: list[dict] = []
        self.dropped = 0
        self.other_model_rows = 0
        self.seen: set[str] = set()

    def is_model(self, label: str) -> bool:
        return _norm(_clean_cell(label)) in self.names

    def skip(self, reason: str, line: int, text: str) -> None:
        if len(self.skipped) >= MAX_SKIPPED:
            self.dropped += 1
        else:
            self.skipped.append({"reason": reason, "line": line, "text": _quote(text)})

    def add(self, name: str, value: str, kind: str, line: int, text: str, *, heading: str = "", extra_note: str = "",
            alt_name: str = "") -> None:
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            return self.skip("the row has no name", line, text)
        if MEASURED_HERE.search(name) or MEASURED_HERE.search(heading):
            return self.skip("looks like a number measured on Tenstorrent hardware or by this harness (speed, a board, "
                             "or a heading that says so), not the authors' report", line, text)
        if len(name) > MAX_REPORTED_NAME:
            return self.skip(f"the name is too long (over {MAX_REPORTED_NAME} characters)", line, text)
        if len(value) > MAX_REPORTED_VALUE:
            return self.skip(f"the value is too long (over {MAX_REPORTED_VALUE} characters)", line, text)
        if name in self.seen and alt_name and alt_name not in self.seen:
            name = alt_name
        if name in self.seen:
            return self.skip(f"{name!r} appears twice; the first one is kept", line, text)
        if len(self.rows) >= MAX_REPORTED_ROWS:
            return self.skip(f"over the {MAX_REPORTED_ROWS}-row cap of reported-results.json", line, text)
        self.seen.add(name)
        note = f"model card {'model-index' if kind == 'model-index' else 'table'}, line {line}" + extra_note
        self.rows.append({"name": name, "value": value, "note": note[:MAX_REPORTED_NOTE],
                          "extracted_from": {"kind": kind, "line": line, "text": _quote(text)}})


# ---- the model-index block of the front matter ------------------------------------------------------------------------

_KEY = re.compile(r"""^(?P<key>"[^"]*"|'[^']*'|[^\s'"\[{&*!|>#-][^:#]*?|-[^\s:][^:]*?)\s*:(?:\s+(?P<rest>.*))?$""")


def _scalar(text: str) -> str:
    text = text.strip()
    if text[:1] in "[{&*!|>%@`":
        raise UnsupportedYaml(f"a value starting with {text[:1]!r}")
    if text[:1] in "'\"":
        quote = text[0]
        end = text.find(quote, 1)
        after = text[end + 1:].strip() if end > 0 else "x"
        if end < 0 or (after and not after.startswith("#")):
            raise UnsupportedYaml("a quoted value that does not end on its line")
        return text[1:end]
    return re.sub(r"\s+#.*$", "", text).strip()


def _tokens(lines: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    out = []
    for number, raw in lines:
        if "\t" in raw[:len(raw) - len(raw.lstrip())]:
            raise UnsupportedYaml("a tab in the indentation")
        body = raw.strip()
        if not body or body.startswith("#"):
            continue
        out.append((number, len(raw) - len(raw.lstrip(" ")), body))
        if len(out) > MAX_YAML_LINES:
            raise UnsupportedYaml("more lines than the limit")
    return out


def _is_dash(text: str) -> bool:
    return text == "-" or text.startswith("- ")


def _parse_yaml(toks, i, indent, depth):
    if depth > MAX_YAML_DEPTH:
        raise UnsupportedYaml("nested deeper than the limit")
    return (_parse_seq if _is_dash(toks[i][2]) else _parse_map)(toks, i, indent, depth)


def _child(toks, i, indent, depth):
    """The value of a `key:` or `-` that has nothing after it: a block on the next lines (a sequence may sit at the key's own
    indent), or an empty value."""
    if i < len(toks):
        _, n_indent, n_text = toks[i]
        if n_indent > indent or (n_indent == indent and _is_dash(n_text)):
            return _parse_yaml(toks, i, n_indent, depth + 1)
    return None, i


def _parse_seq(toks, i, indent, depth):
    items = []
    while i < len(toks) and toks[i][1] == indent and _is_dash(toks[i][2]):
        line, _, text = toks[i]
        rest = text[1:].lstrip()
        if not rest:
            item, i = _child(toks, i + 1, indent, depth)
        elif _KEY.match(rest):
            inner = indent + (len(text) - len(rest))
            toks = toks[:i] + [(line, inner, rest)] + toks[i + 1:]
            item, i = _parse_map(toks, i, inner, depth + 1)
        else:
            item, i = ("s", _scalar(rest), line), i + 1
        items.append(item)
    if i < len(toks) and toks[i][1] > indent:
        raise UnsupportedYaml(f"line {toks[i][0]} is indented in a way this reader does not follow")
    return items, i


def _parse_map(toks, i, indent, depth):
    out = {}
    while i < len(toks) and toks[i][1] == indent and not _is_dash(toks[i][2]):
        line, _, text = toks[i]
        m = _KEY.match(text)
        if not m:
            raise UnsupportedYaml(f"line {line} is not a key")
        key = m.group("key").strip("'\"")
        rest = (m.group("rest") or "").strip()
        if not rest or rest.startswith("#"):
            node, i = _child(toks, i + 1, indent, depth)
        else:
            node, i = ("s", _scalar(rest), line), i + 1
        out[key] = node
    if i < len(toks) and toks[i][1] > indent:
        raise UnsupportedYaml(f"line {toks[i][0]} is indented in a way this reader does not follow")
    return out, i


def _text(node) -> str:
    return node[1] if isinstance(node, tuple) and node[0] == "s" else ""


def _front_matter_lines(lines: list[str]) -> list[tuple[int, str]]:
    if not lines or lines[0].strip() != "---":
        return []
    for k in range(1, len(lines)):
        if lines[k].strip() == "---":
            return [(n + 1, lines[n]) for n in range(1, k)]
    return []


def _read_model_index(lines: list[str], rows: _Rows) -> None:
    fm = _front_matter_lines(lines)
    start = next((k for k, (_, raw) in enumerate(fm) if re.match(r"^model-index:\s*(#.*)?$", raw)), None)
    if start is None:
        return
    block = []
    for number, raw in fm[start + 1:]:
        if raw.strip() and not raw.startswith((" ", "\t", "-", "#")):
            break                                           # the next top-level key
        block.append((number, raw))
    try:
        toks = _tokens(block)
        if not toks:
            return
        tree, _ = _parse_yaml(toks, 0, toks[0][1], 0)
    except (UnsupportedYaml, RecursionError) as exc:
        rows.skip(f"the model-index block was not read ({exc}); this reader follows plain block YAML only",
                  fm[start][0], "model-index:")
        return
    for entry in tree if isinstance(tree, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("results"), list):
            continue
        for result in entry["results"]:
            if not isinstance(result, dict) or not isinstance(result.get("metrics"), list):
                continue
            dataset = result.get("dataset") if isinstance(result.get("dataset"), dict) else {}
            data_name = _text(dataset.get("name")) or _text(dataset.get("type")) or "unnamed dataset"
            where = ", ".join(x for x in (_text(dataset.get("config")), _text(dataset.get("split"))) if x)
            src = result.get("source") if isinstance(result.get("source"), dict) else entry.get("source")
            url = _text(src.get("url")) if isinstance(src, dict) else ""
            extra = f"; the card's own source: {url}" if url.startswith("https://") and len(url) <= 120 else ""
            for metric in result["metrics"]:
                if not isinstance(metric, dict):
                    continue
                kind = _text(metric.get("name")) or _text(metric.get("type")) or "metric"
                name = f"{data_name} / {kind}"
                node = metric.get("value")
                value, line = (_text(node), node[2]) if isinstance(node, tuple) else ("", fm[start][0])
                text = f"value: {value}" if value else f"{name}: no value"
                if not _plain_number(value):
                    rows.skip(f"the value is not a plain number ({value or 'missing'})", line, text)
                    continue
                rows.add(name, value, "model-index", line, text, extra_note=extra,
                         alt_name=f"{name} ({where})" if where else "")


# ---- tables ---------------------------------------------------------------------------------------------------------

class _Table:
    def __init__(self, line: int, heading: str):
        self.line, self.heading = line, heading
        self.rows: list[tuple[int, str, list[str]]] = []      # (line, quoted text, cells)


def _split_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return [c.replace("\\|", "|") for c in re.split(r"(?<!\\)\|", body)]


def _markdown_tables(lines: list[str]) -> list[_Table]:
    out, heading, i = [], "", 0
    while i < len(lines):
        raw = lines[i]
        if re.match(r"^\s{0,3}#{1,6}\s", raw):
            heading = raw.strip()
        if raw.lstrip().startswith("|") and i + 1 < len(lines) and "|" in lines[i + 1] \
                and re.match(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", lines[i + 1]):
            table = _Table(i + 1, heading)
            table.rows.append((i + 1, raw, _split_row(raw)))
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table.rows.append((i + 1, lines[i], _split_row(lines[i])))
                i += 1
            out.append(table)
            continue
        i += 1
    return out


class _HtmlTables(HTMLParser):
    """Collects the cells of every outermost <table> in the text it is fed. Script and style content is dropped."""

    def __init__(self, line_offset: int, heading: str):
        super().__init__(convert_charrefs=True)
        self.line_offset, self.heading, self.tables = line_offset, heading, []
        self.depth = self.skip = 0
        self.cur_row = self.cur_cell = None
        self.row_line = 0
        self.table = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.table = _Table(self.getpos()[0] + self.line_offset, self.heading)
        elif self.depth == 1 and tag == "tr":
            self.cur_row, self.row_line = [], self.getpos()[0] + self.line_offset
        elif self.depth == 1 and tag in ("td", "th") and self.cur_row is not None:
            self.cur_cell = []
        elif tag in ("script", "style"):
            self.skip += 1
        elif self.cur_cell is not None:
            self.cur_cell.append(" ")                        # elements inside a cell are separate words

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif self.depth == 1 and tag in ("td", "th") and self.cur_cell is not None and self.cur_row is not None:
            self.cur_row.append(re.sub(r"\s+", " ", "".join(self.cur_cell)).strip())
            self.cur_cell = None
        elif self.depth == 1 and tag == "tr" and self.cur_row is not None:
            self.table.rows.append((self.row_line, "", self.cur_row))
            self.cur_row = None
        elif tag == "table" and self.depth:
            self.depth -= 1
            if self.depth == 0 and self.table is not None:
                self.tables.append(self.table)
                self.table = None

    def handle_data(self, data):
        if self.cur_cell is not None and not self.skip:
            self.cur_cell.append(data)


def _html_tables(text: str, lines: list[str], headings: list[tuple[int, str]]) -> list[_Table]:
    out = []
    for m in re.finditer(r"<table\b.*?</table\s*>", text, re.S | re.I):
        offset = text.count("\n", 0, m.start())
        heading = next((h for n, h in reversed(headings) if n <= offset + 1), "")
        parser = _HtmlTables(offset, heading)
        try:
            parser.feed(m.group(0))
            parser.close()
        except Exception:                                  # a malformed table is skipped, never fatal
            continue
        for table in parser.tables:
            # a row's quote is the card's own text on its first line (a row may span lines)
            table.rows = [(n, lines[n - 1].strip()[:MAX_QUOTE] if 0 < n <= len(lines) else "", c) for n, _, c in table.rows]
            out.append(table)
    return out


def _read_table(table: _Table, rows: _Rows) -> None:
    if not table.rows:
        return
    head = [_clean_cell(c) for c in table.rows[0][2]]
    data = [(n, q, cells) for n, q, cells in table.rows[1:]
            if len(cells) >= 2 and any(c.strip() for c in cells[1:])]      # a one-cell (colspan) row is a section label
    first = _norm(head[0]) if head else ""
    width = len(head)
    candidates = []                                                         # (name, cell, line, text, column)
    model_rows = [d for d in data if rows.is_model(d[2][0])]
    prefix = "" if first in {_norm(x) for x in GENERIC_FIRST} else f"{head[0]}: "
    label_cols = [j for j in range(1, width) if _norm(head[j]) in LABEL_HEADERS]

    def misaligned(n, q, cells) -> bool:
        """Merged cells (rowspan, colspan) or missing ones shift the columns, so a value could come from the wrong one."""
        if len(cells) == width:
            return False
        rows.skip(f"the row has {len(cells)} cells and the header has {width}: merged or missing cells shift the columns, "
                  "so no value is taken from it", n, q or " | ".join(cells))
        return True

    if model_rows:
        n, q, cells = model_rows[0]
        if not misaligned(n, q, cells):
            candidates += [(head[j], cells[j], n, q, j) for j in range(1, width) if head[j] and j not in label_cols]
        rows.other_model_rows += len(data) - len(model_rows)
    else:
        col = next((j for j in range(1, width) if rows.is_model(head[j])), None)
        # A two-column `Benchmark | Score` table is the model's own. A `Model | Score` table lists models, so its rows are
        # somebody's scores and stay unread (they are counted below).
        if col is None and width == 2 and _norm(head[1]) in {_norm(x) for x in GENERIC_SCORE} \
                and first not in {_norm(x) for x in MODEL_FIRST}:
            col = 1
        if col is not None:
            before = [j for j in label_cols if j < col]
            for n, q, cells in data:
                if misaligned(n, q, cells):
                    continue
                parts = [_clean_cell(cells[0])] + [_clean_cell(cells[j]) for j in before]
                name = " / ".join(p for p in parts if p)
                if name:
                    candidates.append(((name if before else prefix + name), cells[col], n, q, col))
        elif first in {_norm(x) for x in MODEL_FIRST}:
            rows.other_model_rows += len(data)                              # a comparison table with no row for this model
    cooked = []
    for name, cell, n, q, col in candidates:
        value = _clean_cell(cell)
        if value.lower() in NO_RESULT:
            continue
        if not _plain_number(value):
            rows.skip(f"the cell is not a plain number ({value[:40]}); numbers with notes, ranges, units or "
                      "footnote marks are not read", n, q or value)
            continue
        cooked.append((name, value, n, q, col))
    for col in sorted({c[4] for c in cooked}):                              # percent and plain values in one column
        group = [c for c in cooked if c[4] == col]
        if len({v.endswith("%") for _, v, _, _, _ in group}) > 1:
            for name, value, n, q, _ in group:
                rows.skip("percent and plain numbers are mixed in one column, so the scale is not stated", n, q or value)
            cooked = [c for c in cooked if c[4] != col]
    for name, value, n, q, _ in sorted(cooked, key=lambda c: (c[2], c[4])):
        rows.add(name, value, "table", n, q or f"{name}: {value}", heading=" ".join([table.heading] + head))


def extract_reported(readme: str, model_id: str) -> dict:
    """The scores a model card reports for the model, as a draft: {rows, skipped, other_model_rows, truncated, ...}. See above."""
    truncated = len(readme) > MAX_README_BYTES
    lines = readme[:MAX_README_BYTES].split("\n")
    rows = _Rows(model_id)
    _read_model_index(lines, rows)
    plain, fenced = [], False
    for raw in lines:                                       # a fenced block is an example, not the card's table
        if raw.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            plain.append("")
        else:
            plain.append("" if fenced else raw)
    headings = [(n + 1, raw.strip()) for n, raw in enumerate(plain) if re.match(r"^\s{0,3}#{1,6}\s", raw)]
    tables = _markdown_tables(plain) + _html_tables("\n".join(plain), plain, headings)
    tables.sort(key=lambda t: t.line)
    for k, table in enumerate(tables):
        if k >= MAX_TABLES:
            rows.skip(f"only the first {MAX_TABLES} tables are read (the table limit); {len(tables) - k} more were not",
                      table.line, "")
            break
        _read_table(table, rows)
    return {"rows": rows.rows, "skipped": rows.skipped, "skipped_dropped": rows.dropped,
            "other_model_rows": rows.other_model_rows, "truncated": truncated, "tables_read": min(len(tables), MAX_TABLES)}


def eos_ids(gen: dict) -> list[int]:
    """The integer stop token ids of a generation config's top-level eos_token_id (an int or a list of ints).
    Empty when the key is absent or holds anything else: the bundle's serving code reads it only there."""
    v = gen.get("eos_token_id")
    if isinstance(v, int) and not isinstance(v, bool):
        return [v]
    if isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        return list(v)
    return []


def config_eos_ids(cfg) -> list[int]:
    """The stop token ids a model's config.json names: a top-level eos_token_id, else the one under text_config
    (an int or a list of ints; empty for anything else)."""
    if not isinstance(cfg, dict):
        return []
    top = eos_ids(cfg)
    if top:
        return top
    text = cfg.get("text_config")
    return eos_ids(text) if isinstance(text, dict) else []


def tokenizer_token_id(data, token) -> int | None:
    """The id tokenizer.json gives a token string: its added tokens first (special tokens live there), then the
    vocabulary. None when the token is unknown, `data` is not a tokenizer, or the id is not an integer."""
    if not isinstance(data, dict) or not isinstance(token, str):
        return None
    added = data.get("added_tokens")
    for item in added if isinstance(added, list) else []:
        if isinstance(item, dict) and item.get("content") == token:
            tid = item.get("id")
            return tid if isinstance(tid, int) and not isinstance(tid, bool) else None
    model = data.get("model")
    vocab = vocab_items(model) if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        tid = vocab.get(token)
        return tid if isinstance(tid, int) and not isinstance(tid, bool) else None
    if isinstance(vocab, list):
        for i, entry in enumerate(vocab):
            if (entry[0] if isinstance(entry, tuple) and entry else entry) == token:
                return i
    return None


HINT_NAMES = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty")
MAX_HINT_VALUES = 4                 # distinct values kept per setting
CARD_READ_CHARS = 200_000           # the part of a model card that is read for hints
_HINT = re.compile(r"(?<![A-Za-z0-9_])(temperature|top[_-]?p|top[_-]?k|min[_-]?p|repetition[_-]?penalty|"
                   r"presence[_-]?penalty|frequency[_-]?penalty)[`'\"*]*[ \t]*(?:=|:|of|to|at|is)?[ \t]*[`'\"*]*"
                   r"(-?\d+(?:\.\d+)?)(?![0-9A-Za-z_])(?!\.\d)", re.IGNORECASE)
# A card can write a negative number. Only some settings take one: presence_penalty and frequency_penalty run from -2 to 2
# (vLLM and the OpenAI API), and a top_k of -1 turns top-k filtering off. A negative temperature, top_p, min_p or
# repetition_penalty, or a value outside those ranges, is not something the server could apply, so it is left out.
NEGATIVE_OK = {"presence_penalty": lambda v: v >= -2.0,      # called only for a value written with a minus sign
               "frequency_penalty": lambda v: v >= -2.0,
               "top_k": lambda v: v == -1.0}


def card_sampling_hints(text: str) -> dict:
    """{setting: [values]} for sampling settings a model card writes as a number next to the setting's name
    (`temperature=0.6`, `top_k: 20`, "top-p of 0.8"). Order of first appearance, no repeats, at most MAX_HINT_VALUES
    per setting. The card is prose, so this is a hint and nothing more."""
    out: dict[str, list[float]] = {}
    for m in _HINT.finditer(text[:CARD_READ_CHARS]):
        name = re.sub(r"[-]", "_", m.group(1).lower())
        name = {"topp": "top_p", "topk": "top_k", "minp": "min_p"}.get(name.replace("_", ""), name)
        try:
            value = float(m.group(2))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if m.group(2).startswith("-") and not NEGATIVE_OK.get(name, lambda v: False)(value):
            continue          # written with a minus sign, and this setting cannot take one ("-0" counts as written negative)
        values = out.setdefault(name, [])
        if value not in values and len(values) < MAX_HINT_VALUES:
            values.append(value)
    return out


def card_hints_of(snapshot: Path) -> dict:
    readme = snapshot / "README.md"
    return card_sampling_hints(readme.read_text(encoding="utf-8", errors="replace")) if readme.is_file() else {}


def compare_genconfig_license(model_dir: Path, base_dir: Path) -> dict:
    mg, merr = read_json(model_dir / "generation_config.json")
    bg, berr = read_json(base_dir / "generation_config.json")
    bc, _ = read_json(base_dir / "config.json")             # staging copies this one, so its stop token can count
    mg, bg = mg if isinstance(mg, dict) else {}, bg if isinstance(bg, dict) else {}
    bc = bc if isinstance(bc, dict) else {}
    diffs = {k: {"model": mg.get(k, ABSENT), "base": bg.get(k, ABSENT)}
             for k in sorted(set(mg) | set(bg)) if mg.get(k, ABSENT) != bg.get(k, ABSENT)}
    mc, _ = read_json(model_dir / "config.json")
    tok_name = special_tokens(model_dir)["eos_token"]
    tok_data, _ = read_json(model_dir / "tokenizer.json")
    ml, bl = license_side(model_dir), license_side(base_dir)
    changed = None if ml["license"] is None or bl["license"] is None else ml["license"] != bl["license"]
    return {"model_generation_config": mg, "base_generation_config": bg,
            "model_generation_config_present": (model_dir / "generation_config.json").is_file(),
            "base_generation_config_present": (base_dir / "generation_config.json").is_file(),
            "errors": [e for e in (merr, berr) if e],
            "model_eos_token_ids": eos_ids(mg), "base_eos_token_ids": eos_ids(bg),
            "base_config_eos_token_ids": eos_ids(bc),
            "model_config_eos_token_ids": config_eos_ids(mc),
            "model_tokenizer_eos_token": tok_name,
            "model_tokenizer_eos_token_id": tokenizer_token_id(tok_data, tok_name),
            "card_sampling_hints": card_hints_of(model_dir),
            "generation_config_diffs": diffs, "generation_config_equal": not diffs,
            "generation_config_equal_except_bookkeeping": all(k in BOOKKEEPING for k in diffs),
            "model_license": ml, "base_license": bl, "license_changed": changed}


def disk_free(run_dir: str) -> dict:
    st = os.statvfs(run_dir)
    return {"path": run_dir, "measured_with": "os.statvfs", "free_bytes": st.f_bavail * st.f_frsize,
            "total_bytes": st.f_blocks * st.f_frsize,
            "free_gib": round(st.f_bavail * st.f_frsize / GIB, 1)}


# ---- the draft delta -----------------------------------------------------------------------------

def fmt(value) -> str:
    if isinstance(value, list) and len(value) > 6:
        kinds = sorted(set(map(str, value)))
        return f"a list of {len(value)} ({short(kinds, 4)})"
    return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


def config_finding(c: dict) -> str:
    if c.get("errors") and "text_key_values" not in c:
        return "The configs could not be compared: " + "; ".join(c["errors"]) + "."
    v = c["text_key_values"]
    if c["decisive_keys_equal"]:
        s = ("The text configs agree on every value that decides the path: "
             + ", ".join(f"{k} {fmt(v[k]['model'])}" for k in DECISIVE) + ".")
    else:
        s = ("These text config values differ: "
             + "; ".join(f"{k} is {fmt(v[k]['model'])} in the new model and {fmt(v[k]['base'])} in "
                         "the nearest model" for k in c["decisive_keys_differing"]) + ".")
    other = [k for k in c["text_config_diffs"] if k not in DECISIVE]
    book = [k for k in other if k in BOOKKEEPING]
    rest = [k for k in other if k not in BOOKKEEPING]
    if book:
        s += f" Bookkeeping keys that differ: {short(book)}."
    if rest:
        s += (f" {len(rest)} other text config keys differ: "
              + "; ".join(f"{k} {fmt(c['text_config_diffs'][k]['model'])} vs "
                          f"{fmt(c['text_config_diffs'][k]['base'])}" for k in rest[:6])
              + (f" and {len(rest) - 6} more" if len(rest) > 6 else "") + ".")
    if not other and c["decisive_keys_equal"]:
        s += " No other text config key differs."
    nest = {True: "nests its text config under text_config", False: "has a flat config"}
    s += f" The new model {nest[c['model_config_nested']]}; the nearest model {nest[c['base_config_nested']]}."
    return s


def architecture_finding(c: dict, t: dict) -> str:
    if "model_architectures" not in c:
        return "The architectures could not be compared: the configs did not load."
    ma, ba = c["model_architectures"], c["base_architectures"]
    s = (f"The new model's architectures entry is {fmt(ma)}; the nearest model's is {fmt(ba)}"
         + (" (the same)." if ma == ba else "."))
    vm, vb = t["vision_tensors"]["model"], t["vision_tensors"]["base"]

    def tower(vision_cfg: bool, n: int) -> str:
        if n:
            return f"has a vision tower ({n} vision tensors" + (", and a vision_config)" if vision_cfg else ")")
        return "has no vision tower (no vision tensors" + (", though its config has a vision_config)"
                                                           if vision_cfg else ")")
    s += (f" The new model {tower(c['model_has_vision_config'], vm)}. The nearest model "
          f"{tower(c['base_has_vision_config'], vb)}.")
    if vm and not vb:
        s += " The new model keeps a vision tower that the nearest model lacks."
    elif vb and not vm:
        s += " The new model dropped the nearest model's vision tower."
    return s


def tensors_finding(t: dict) -> str:
    ns = t["normalized_name_sets"]
    s = (f"The new model has {t['model']['num_tensors']} tensors in {len(t['model']['shards'])} "
         f"safetensors files; the nearest model has {t['base']['num_tensors']} in "
         f"{len(t['base']['shards'])}. After the name prefix is normalized, {ns['common']} tensors are "
         f"shared; {len(t['shape_diffs_normalized'])} of them differ in shape and "
         f"{len(t['dtype_diffs_normalized'])} in dtype.")
    if t["shape_diffs_normalized"]:
        s += " Shape differences: " + "; ".join(f"{x['name']} {x['model']} vs {x['base']}"
                                                for x in t["shape_diffs_normalized"][:5]) + "."
    if t["dtype_diffs_normalized"]:
        s += " Dtype differences: " + "; ".join(f"{x['name']} {x['model']} vs {x['base']}"
                                                for x in t["dtype_diffs_normalized"][:5]) + "."
    for side, label in (("model", "the new model has"), ("base", "the nearest model has")):
        extra = ns[f"only_in_{side}"]
        if extra:
            groups = ns[f"only_in_{side}_groups"]
            s += (f" {len(extra)} tensors only {label.split(' has')[0]} has: "
                  + ", ".join(f"{k}.* ({n})" for k, n in list(groups.items())[:6]) + ".")
    s += (f" Dtypes: new model {fmt(t['model']['dtype_histogram'])}, nearest model "
          f"{fmt(t['base']['dtype_histogram'])}.")
    if t["errors"]:
        s += " Unreadable headers: " + "; ".join(t["errors"]) + "."
    return s


def tensor_names_finding(t: dict) -> str:
    mp, bp = t["model"]["uses_language_model_prefix"], t["base"]["uses_language_model_prefix"]
    raw, ns = t["raw_name_sets"]["common"], t["normalized_name_sets"]["common"]
    al = t["normalized_name_sets"]["aligned"]
    if al["pairs"]:
        rules = "; ".join(f"'{r['model_prefix'] or '(none)'}' in the new model is '{r['base_prefix'] or '(none)'}' in the nearest model "
                          f"({r['tensors']} tensors, {r['same_shape']} with the same shape and dtype)" for r in al["rules"][:4])
        return (f"The two models name their tensors differently: {raw} names match as written. After the "
                f"'{LM_PREFIX}' rewrite and pairing names that differ only by a prefix, {ns} tensors have an analogue "
                f"({al['pairs']} of them paired by prefix): {rules}. A loader that accepts only one prefix needs the names remapped "
                f"(delta.json's alignment.all_pairs lists every pair).")
    if mp == bp:
        where = LM_PREFIX if mp else "model."
        return (f"Both models name their text tensors under the same prefix ({where}). "
                f"{raw} names match as written.")
    new, old = (LM_PREFIX, "model.") if mp else ("model.", LM_PREFIX)
    return (f"The new model's text tensors use the prefix '{new}' and the nearest model's use "
            f"'{old}'. {raw} names match as written and {ns} match after the prefix is normalized. "
            "A loader that accepts only one prefix needs the names remapped.")


def values_finding(v: dict) -> str:
    sm, n = v["summary"], len(v["samples"])
    parts = [f"{sm[k]} {k}" for k in ("identical", "near", "changed", "unrelated") if sm[k]]
    s = (f"Sampled values of {n} paired tensors (the first {SAMPLE_ELEMENTS} values of each): {', '.join(parts) or 'none'}. "
         f"'near' means a cosine of at least {NEAR_COSINE} with the nearest model's tensor, which is what a fine-tune looks like; "
         "'identical' would mean the weights were not trained at all.")
    if sm["unrelated"] or sm["changed"]:
        s += " A pair that is changed or unrelated may not be an analogue: check the name alignment."
    if v["skipped"]:
        s += " Skipped: " + "; ".join(v["skipped"][:3]) + "."
    return s


def head_finding(h: dict) -> str:
    toks = h.get("tokens") or []
    shown = (" for the tokens " + ", ".join(repr(t) for t in toks[:12]) + (" ..." if len(toks) > 12 else "")) if toks else ""
    return (f"The new model has no lm_head.weight. Its sidecar {h['sidecar']} (tensor '{h['tensor']}', {h['rows']} rows) holds rows that "
            f"are copies of {h['matched_rows']} of the nearest model's {h['base_rows']} lm_head rows (lowest cosine "
            f"{h['min_cosine']}){shown}. The model is the nearest model's backbone with its head cut down to those tokens.")


def tokenizer_finding(tc: dict, enc: dict) -> str:
    m, b = tc["model"], tc["base"]
    if tc.get("errors") and "vocab_equal" not in tc:
        return "The tokenizers could not be compared: " + "; ".join(tc["errors"]) + "."
    yes = {True: "identical", False: "different"}
    s = (f"Vocabulary size {m['vocab_size']} in the new model and {b['vocab_size']} in the nearest "
         f"model; the vocab is {yes[tc['vocab_equal']]}. Merges: {m['num_merges']} and "
         f"{b['num_merges']}, {yes[tc['merges_equal']]} in content")
    if tc["merges_format"]["model"] != tc["merges_format"]["base"]:
        s += (f" (one file writes each merge as a {tc['merges_format']['model']}, the other as a "
              f"{tc['merges_format']['base']})")
    s += (f". Added tokens: {m['num_added_tokens']} and {b['num_added_tokens']}, "
          f"{yes[tc['added_tokens_equal']]}.")
    for part in ("normalizer", "pre_tokenizer", "decoder", "post_processor"):
        if not tc.get(f"{part}_equal", True):
            fields = tc.get(f"{part}_field_diffs", {})
            s += f" The {part.replace('_', '-')} differs in {short(fields, 4)}."
    rm, rb = tc.get("pre_tokenizer_regex_model", []), tc.get("pre_tokenizer_regex_base", [])
    if rm != rb and any(r"\p{M}" in p for p in rb) and not any(r"\p{M}" in p for p in rm):
        s += r" The new model's pre-tokenizer pattern drops \p{M} (combining marks) from its letter classes."
    elif rm != rb and any(r"\p{M}" in p for p in rm) and not any(r"\p{M}" in p for p in rb):
        s += r" The new model's pre-tokenizer pattern adds \p{M} (combining marks) to its letter classes."
    if not enc.get("ran"):
        s += f" The encode test did not run: {enc.get('reason')}."
    else:
        s += (f" Encoding {enc['total']} test strings ({enc['fixed']} fixed, {enc['random']} fixed-seed "
              f"random) with both tokenizers gave different ids for {enc['differing']} of {enc['total']} "
              f"({enc['fraction_differing']:.1%}).")
        if enc["differing"]:
            s += " By category: " + ", ".join(f"{k} {n}" for k, n in enc["by_category"].items()) + "."
    return s


def token_text(value) -> str:
    return value if isinstance(value, str) else "absent"


def chat_template_finding(tc: dict) -> str:
    m, b = tc["model"], tc["base"]
    if m["chat_template_sha256"] is None and b["chat_template_sha256"] is None:
        s = "Neither model ships a chat template."
    elif tc["chat_template_equal"]:
        s = (f"Identical. Both chat templates have sha256 {m['chat_template_sha256'][:16]} "
             f"(new model from {m['chat_template_source']}, nearest model from {b['chat_template_source']}).")
    elif m["chat_template_sha256"] is None or b["chat_template_sha256"] is None:
        who = "the new model" if m["chat_template_sha256"] else "the nearest model"
        s = f"Only {who} ships a chat template."
    else:
        s = (f"Different. The new model's chat template ({m['chat_template_source']}) has sha256 "
             f"{m['chat_template_sha256'][:16]}; the nearest model's ({b['chat_template_source']}) has "
             f"{b['chat_template_sha256'][:16]}.")
    for who, side in (("new", m), ("nearest", b)):
        i, f = side["chat_template_inline_sha256"], side["chat_template_file_sha256"]
        if i and f and i != f:
            s += (f" The {who} model's tokenizer_config.json carries a chat template (sha256 {i[:16]}) that differs from its "
                  f"chat_template.jinja (sha256 {f[:16]}); transformers 5.19.0 loads the file.")
    if tc["special_tokens_equal"]:
        if any(v for v in m["special_tokens"].values()):
            s += " The eos, bos, pad and unk tokens are the same in both."
    else:
        s += (" Special tokens that differ: "
              + "; ".join(f"{k} is {token_text(d['model'])} in the new model and {token_text(d['base'])} in the nearest model"
                          for k, d in tc["special_tokens_diffs"].items()) + ".")
    return s


def files_finding(f: dict) -> str:
    m, b = f["model"], f["base"]
    s = (f"The new model has {m['num_files']} files, with {m['num_weight_shards']} weight shards "
         f"totalling {m['weight_shard_total_gib']} GiB. The nearest model has {b['num_files']} files, "
         f"with {b['num_weight_shards']} shards totalling {b['weight_shard_total_gib']} GiB.")
    if f["files_only_in_model"]:
        s += f" Files only in the new model (not counting shards): {short(f['files_only_in_model'])}."
    if f["files_only_in_base"]:
        s += f" Files only in the nearest model (not counting shards): {short(f['files_only_in_base'])}."
    if not f["files_only_in_model"] and not f["files_only_in_base"]:
        s += " Apart from the shards, both models have the same file names."
    return s


def genconfig_finding(g: dict) -> str:
    diffs = g["generation_config_diffs"]
    if not g["model_generation_config_present"] and not g["base_generation_config_present"]:
        return "Neither model has a generation_config.json."
    keys = ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty")
    shown = ", ".join(f"{k} {fmt(g['model_generation_config'][k])}" for k in keys
                      if k in g["model_generation_config"])
    if not diffs:
        return f"Identical. Sampling defaults: {shown or 'none set'}."
    if g["generation_config_equal_except_bookkeeping"]:
        return (f"The sampling defaults are identical ({shown or 'none set'}). Only bookkeeping keys "
                f"differ: {short(diffs)}.")
    return ("These generation_config keys differ: "
            + "; ".join(f"{k} is {fmt(v['model'])} in the new model and {fmt(v['base'])} in the nearest model"
                        for k, v in list(diffs.items())[:8]) + ".")


def license_finding(g: dict) -> str:
    ml, bl = g["model_license"], g["base_license"]

    def desc(side: dict) -> str:
        if side["license"] is None:
            return "unknown (no license field in the model card and no recognised LICENSE file)"
        return f"{side['license']} (from the {side['source']})"
    s = f"The new model's license is {desc(ml)}. The nearest model's is {desc(bl)}."
    if g["license_changed"] is True:
        s += " The license changed."
    elif g["license_changed"] is False:
        s += " The license is the same."
    else:
        s += " Whether it changed is unknown; read both licenses."
    return s


VOCAB_TENSOR = re.compile(r"(^|\.)(embed_tokens|lm_head)\.weight$")
VOCAB_CHUNK_BYTES = 64 * 1024 * 1024        # rows are read and hashed this many bytes at a time


def _row_digests(entry: dict, start_row: int, rows: int, row_bytes: int):
    with open(entry["file"], "rb") as f:
        f.seek(entry["start"] + start_row * row_bytes)
        buf = f.read(rows * row_bytes)
    if len(buf) != rows * row_bytes:
        raise ValueError(f"{entry['file']}: read {len(buf)} bytes where {rows * row_bytes} were expected (the file is shorter than its header says)")
    view = memoryview(buf)
    return [hashlib.blake2b(view[i * row_bytes:(i + 1) * row_bytes], digest_size=16).digest() for i in range(rows)]


def rows_found_in_base(m: dict, b: dict) -> int:
    """How many rows of the model's 2-D matrix are, byte for byte, rows of the nearest model's matrix. Both are read in chunks; the
    nearest model's rows are kept as 16-byte digests."""
    size = ITEMSIZE[m["dtype"]]
    row_bytes = int(m["shape"][1]) * size
    for e in (m, b):
        if e["end"] - e["start"] != int(e["shape"][0]) * row_bytes:
            raise ValueError(f"{e['file']}: the header's byte range does not hold {e['shape'][0]} rows of {row_bytes} bytes")
    per_chunk = max(1, VOCAB_CHUNK_BYTES // row_bytes)
    base_rows = set()
    for r in range(0, int(b["shape"][0]), per_chunk):
        base_rows.update(_row_digests(b, r, min(per_chunk, int(b["shape"][0]) - r), row_bytes))
    found = 0
    for r in range(0, int(m["shape"][0]), per_chunk):
        found += sum(d in base_rows for d in _row_digests(m, r, min(per_chunk, int(m["shape"][0]) - r), row_bytes))
    return found


def detect_vocab_trim(c: dict, t: dict, sc: dict, model_dir: Path, base_dir: Path, pairs: dict) -> dict:
    """Is the new model the nearest model with some vocabulary rows removed (ProprietaryLegal/Qwen3.8-27B-CJK-Trim, issue 56)?
    trimmed: the only config difference is vocab_size (smaller), the only shape differences are the first dimension of embed_tokens
    and lm_head (smaller, same row width), and nothing else differs, as decide_path counts it. kept_subset: every row of those matrices
    is, byte for byte, a row of the nearest model's (True), some row is not (False), or it could not be read (None, with the reason in
    note). explained_reasons are the path reasons this accounts for."""
    out = {"trimmed": False, "kept_subset": None, "model_vocab": None, "base_vocab": None, "tensors": [], "explained_reasons": [],
           "note": ""}
    vals = (c.get("text_key_values") or {}).get("vocab_size")
    if list(c.get("decisive_keys_differing") or []) != ["vocab_size"] or not isinstance(vals, dict):
        out["note"] = "the config difference is not vocab_size alone"
        return out
    mv, bv = vals.get("model"), vals.get("base")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (mv, bv)) or not mv < bv:
        out["note"] = "vocab_size is not a smaller whole number"
        return out
    out["model_vocab"], out["base_vocab"] = mv, bv
    diffs = t["shape_diffs_normalized"]
    if not diffs or sc["sidecars"] or sc["problems"]:
        out["note"] = "no tensor differs in shape, or the model has a sidecar"
        return out
    for x in diffs:
        m, b = x["model"], x["base"]
        if not (VOCAB_TENSOR.search(x["name"]) and len(m) == 2 and len(b) == 2 and m[1] == b[1] and m[0] < b[0]):
            out["note"] = f"{x['name']} differs in shape in a way a vocabulary trim does not"
            return out
    if decide_path({**c, "decisive_keys_differing": []}, {**t, "shape_diffs_normalized": []}):
        out["note"] = "something besides the vocabulary differs"
        return out
    mi, bi = tensor_index(model_dir), tensor_index(base_dir)
    # Every vocabulary matrix the two models share, changed or not, must have the configured number of rows: staging writes the model's
    # vocab_size into the served config, so one matrix left at the old size would recreate the load-time mismatch.
    for m, b in sorted(pairs.items()):
        if m in mi and b in bi and VOCAB_TENSOR.search(m):
            for who, entry, want in (("new", mi[m], mv), ("nearest", bi[b], bv)):
                rows = entry["shape"][0] if entry["shape"] else None
                if rows != want:
                    out["note"] = f"{m} has {rows} rows in the {who} model, but its config says the vocabulary is {want}"
                    return out
    out["trimmed"] = True
    out["explained_reasons"] = decide_path(c, t)
    chosen = sorted((m, b) for m, b in pairs.items() if m in mi and b in bi and VOCAB_TENSOR.search(m)
                    and mi[m]["shape"][:1] != bi[b]["shape"][:1])
    if len(chosen) != len(diffs):
        out["note"] = "the tensors that differ in shape could not all be found in the shard headers"
        return out
    unreadable = [m for m, b in chosen if mi[m]["dtype"] != bi[b]["dtype"] or mi[m]["dtype"] not in ITEMSIZE]
    if unreadable:
        out["note"] = f"{unreadable[0]}: dtype {mi[unreadable[0]]['dtype']} is not one this check reads or differs from the nearest model's"
        return out
    try:
        for m, b in chosen:
            out["tensors"].append({"name": m, "rows": int(mi[m]["shape"][0]), "base_rows": int(bi[b]["shape"][0]),
                                   "rows_found_in_base": rows_found_in_base(mi[m], bi[b])})
    except (OSError, ValueError) as exc:
        out["tensors"] = []
        out["note"] = f"the tensor bodies could not be compared: {exc}"
        return out
    out["kept_subset"] = all(x["rows_found_in_base"] == x["rows"] for x in out["tensors"])
    out["note"] = ("every row of " + ", ".join(x["name"] for x in out["tensors"]) + " is a row of the nearest model's" if out["kept_subset"]
                   else "a row of the new model's matrices is not a row of the nearest model's")
    return out


def decide_path(c: dict, t: dict) -> list[str]:
    """The reasons the path is full-port. An empty list means weights-only."""
    reasons = []
    if c.get("errors") and "text_key_values" not in c:
        reasons.append("the configs could not be compared: " + "; ".join(c["errors"]))
    else:
        for k in c["decisive_keys_differing"]:
            v = c["text_key_values"][k]
            reasons.append(f"text config {k} differs: {fmt(v['model'])} vs {fmt(v['base'])}")
    if t["errors"]:
        reasons.append("some safetensors headers could not be read: " + "; ".join(t["errors"]))
    ns = t["normalized_name_sets"]
    if ns["common"] == 0:
        reasons.append("the two models share no tensor name, even after the prefix is normalized")
    for x in t["shape_diffs_normalized"]:
        reasons.append(f"tensor {x['name']} has shape {x['model']} in the new model and {x['base']} in the nearest model")
    for x in t["dtype_diffs_normalized"]:
        if x.get("tolerated"):
            continue
        reasons.append(f"tensor {x['name']} has dtype {x['model']} in the new model and {x['base']} in the nearest model")
    for side, who in (("model", "the new model"), ("base", "the nearest model")):
        extra = ns[f"unexplained_only_in_{side}"]
        if side == "base" and (t.get("head") or {}).get("restricted"):
            extra = [k for k in extra if not HEAD_NAME.search(k)]      # the readout rows stand in for lm_head
        if extra:
            reasons.append(f"{len(extra)} text tensors exist only in {who}: {short(extra, 5)}")
    return reasons


def decide_class(c: dict, t: dict, reasons: list[str], sc: dict, vt: dict | None = None) -> str:
    """weights-only, weights+sidecar, restricted-head, full-port or unknown. `reasons` already holds the backbone's
    reasons for full-port and the sidecar problems (sc["problems"], told apart by value). restricted-head: the backbone matches and the only thing missing is an
    lm_head that a sidecar's rows stand in for (t["head"]["restricted"], set by detect_restricted_head)."""
    ns = t["normalized_name_sets"]
    could_not_compare = (c.get("errors") and "text_key_values" not in c) or bool(t["errors"]) or ns["common"] == 0
    if could_not_compare:
        return "unknown"
    # A definite difference in the backbone is full-port whatever a sidecar's problems are (summerMC/Diffusion-Qwen3.8-27B: 48 extra text tensors
    # and a dllm_extras.pt that cannot be read). The sidecar problems alone, on a backbone that lines up, are what make the class unknown.
    if vt and vt.get("trimmed") and vt.get("kept_subset") is True and not [r for r in reasons if r not in vt["explained_reasons"]]:
        return "vocab-trim"
    if [r for r in reasons if r not in sc["problems"]]:
        return "full-port"
    if sc["problems"]:
        return "unknown"
    if (t.get("head") or {}).get("restricted"):
        return "restricted-head"
    return "weights+sidecar" if sc["sidecars"] else "weights-only"


SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty", "do_sample")


def graft_candidate(mt: dict, bt: dict, hidden_size: int | None, base_revision: str | None) -> dict | None:
    """Could the nearest model's `mtp.*` tensors be grafted onto a model that has none? None when the model has its own or the
    nearest model has none. The record says what the graft would copy (count, shard, dtypes) and whether the head fits the model:
    `shapes_match` compares `mtp.norm.weight`'s width with the model's hidden size (None when either is unknown). `graftable` is true
    when the tensors are BF16, sit in one shard and do not contradict the hidden size. Nothing is grafted by stage 0: stage 2's graft
    trial (orchard/graft_trial.py) serves the graft once, tries it against the safe configuration, and keeps it only when it measures better;
    `graft_mtp` in swap_config.json and `tt-orchard package --graft-mtp` still force it by hand."""
    if any(MTP_NAME.search(k) for k in mt):
        return None
    head = {k: v for k, v in bt.items() if MTP_NAME.search(k)}
    if not head:
        return None
    shards, dtypes = sorted({v["shard"] for v in head.values()}), sorted({v["dtype"] for v in head.values()})
    norm = next((v for k, v in head.items() if k.endswith("mtp.norm.weight")), None)
    match = None if hidden_size is None or norm is None else norm["shape"] == [hidden_size]
    return {"base_tensors": len(head), "base_revision": base_revision, "base_shards": shards, "dtypes": dtypes,
            "hidden_size": hidden_size, "shapes_match": match,
            "graftable": len(shards) == 1 and dtypes == ["BF16"] and match is not False}


def model_hidden_size(model_dir: Path) -> int | None:
    cfg, _ = read_json(model_dir / "config.json")
    for holder in (cfg.get("text_config") if isinstance(cfg, dict) else None, cfg):
        v = holder.get("hidden_size") if isinstance(holder, dict) else None
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def generation_config_hazards(nearest_id: str, g: dict, ev) -> list[dict]:
    """Stage 2 serves a directory whose generation_config.json is the new model's when it ships one and the
    nearest model's when it does not. The nearest bundle's engine reads eos_token_id from that file (or from
    config.json, which here is the nearest model's and nests it) and stops after loading the weights when it
    finds none (BAAI/AREX-2, 2026-10-21). Flag each case that is not "the model's own file with stop tokens"."""
    finding = None
    base_ids = g["base_eos_token_ids"]
    # Staging copies the nearest model's config.json, and the engine reads a top-level eos_token_id from it when
    # generation_config.json has none. The new model's own config.json is replaced, so its eos_token_id never counts.
    cfg_ids = g["base_config_eos_token_ids"]
    from_config = (f"the eos_token_id of the nearest model's config.json, {fmt(cfg_ids)}, which staging copies"
                   if cfg_ids else None)
    if not g["model_generation_config_present"]:
        if g["base_generation_config_present"]:
            base = g["base_generation_config"]
            defaults = ", ".join(f"{k} {fmt(base[k])}" for k in SAMPLING_KEYS if k in base)
            if base_ids:
                stop = f"the server will stop on that model's eos_token_id {fmt(base_ids)}"
            elif from_config:
                stop = f"the server will stop on {from_config} (the nearest model's generation_config.json has none)"
            else:
                stop = "the server will stop on that model's eos_token_id []"
            finding = (f"The new model ships no generation_config.json. Staging copies the nearest model's "
                       f"({nearest_id}), so {stop}"
                       + (f" and apply its sampling defaults ({defaults}) to a request that sets none" if defaults else "")
                       + (". The nearest model's file has no eos_token_id and neither has its config.json, so the "
                          "engine stops after loading the weights" if not (base_ids or cfg_ids) else "")
                       + ". Read the new model's card for the stop tokens and settings it recommends.")
        elif from_config:
            finding = ("Neither model ships a generation_config.json, so the server has no sampling defaults. It will "
                       f"stop on {from_config}. Read the new model's card for the stop tokens and settings it recommends.")
        else:
            finding = ("Neither model ships a generation_config.json, so the server has no eos_token_id and the "
                       "nearest bundle's engine stops after loading the weights.")
    elif not g["model_eos_token_ids"]:
        if from_config:
            finding = ("The new model's generation_config.json has no eos_token_id. The server will stop on "
                       f"{from_config}. Read the new model's card for the stop tokens it recommends.")
        else:
            finding = ("The new model's generation_config.json has no eos_token_id. The nearest bundle's engine reads "
                       "it from that file or from config.json (the nearest model's here, which nests it) and stops "
                       "after loading the weights when it finds none.")
    return [{"area": "generation_config", "finding": finding, "evidence": [ev("genconfig-license.json")]}] if finding else []


def stop_token_hazards(g: dict, ev) -> list[dict]:
    """The server stops on the eos_token_id of the served generation_config.json (the new model's own, or the nearest
    model's when the model ships none; the nearest model's config.json when that file has none). The new model's
    tokenizer and its own config.json name stop tokens too. One the server does not stop on lets a reply run to
    the length limit. Quiet when the server will have no stop token at all: generation_config_hazards says that."""
    cfg_ids = g["base_config_eos_token_ids"]            # staging copies the nearest model's config.json as well
    from_config = "the nearest model's config.json, which staging copies"
    if g["model_generation_config_present"] and g["model_eos_token_ids"]:
        served, origin = list(g["model_eos_token_ids"]), "the new model's own generation_config.json"
    elif g["model_generation_config_present"] or not g["base_generation_config_present"]:
        served, origin = [], from_config
    elif g["base_eos_token_ids"]:
        served, origin = list(g["base_eos_token_ids"]), "the nearest model's generation_config.json, which staging copies"
    else:
        served, origin = [], from_config
    # prepare_swap.stop_token_ids reads the eos_token_id of generation_config.json and then of config.json and keeps
    # both, so the ids the server can stop on are the two files' together (an id in both counts once).
    extra = [i for i in cfg_ids if i not in served]
    if extra and served:
        origin += " and " + from_config
    served += extra
    if not served:
        return []
    items = []
    tid = g["model_tokenizer_eos_token_id"]
    if tid is not None and tid not in served:
        items.append(f"tokenizer names {g['model_tokenizer_eos_token']} (id {tid}) as its eos token")
    cfg_ids = g["model_config_eos_token_ids"]
    if cfg_ids and any(i not in served for i in cfg_ids):
        items.append(f"config.json names the stop tokens {fmt(cfg_ids)}")
    if not items:
        return []
    return [{"area": "stop_tokens",
             "finding": ("The new model's " + " and its ".join(items) + f", but the server will stop on {fmt(served)} "
                         f"({origin}). A reply that ends on a token the server does not stop on runs to the length "
                         "limit. Check the model card for the stop tokens it recommends."),
             "evidence": [ev("genconfig-license.json")]}]


def sampling_default_hazards(g: dict, ev) -> list[dict]:
    """The server applies the served generation_config.json's sampling defaults to a request that sets none. When the
    model's card writes a setting with a value the served file does not hold (or does not set), the package serves
    with settings its authors did not choose. The card is prose: the hazard quotes what it found."""
    hints = g["card_sampling_hints"]
    if g["model_generation_config_present"]:
        served, origin = g["model_generation_config"], "the new model's own generation_config.json"
    elif g["base_generation_config_present"]:
        served, origin = g["base_generation_config"], "the nearest model's generation_config.json, which staging copies"
    else:
        served, origin = {}, "no generation_config.json, so it has no sampling defaults"
    items = []
    for name in HINT_NAMES:
        values = hints.get(name)
        if not values:
            continue
        have = served.get(name)
        numeric = isinstance(have, (int, float)) and not isinstance(have, bool)
        if numeric and any(math.isclose(have, v) or (name == "top_k" and have <= 0 and v <= 0) for v in values):
            continue          # top_k of 0 and of -1 both turn top-k filtering off, so they do not disagree
        shown = " or ".join(f"{v:g}" for v in values)
        items.append(f"{name} {shown} (" + (f"served {fmt(have)}" if numeric else "not set") + ")")
    if not items:
        return []
    return [{"area": "sampling_defaults",
             "finding": ("The new model's card mentions " + ", ".join(items) + f". A request that sets none gets the "
                         f"defaults of {origin}. Send the card's settings with each request, or ship them in a "
                         "generation_config.json."),
             "evidence": [ev("genconfig-license.json")]}]


def chat_template_hazards(tc: dict, ev) -> list[dict]:
    """A hazard when the new model's chat template differs from the nearest model's, when the nearest model has one and the
    new model has none, or when the special tokens differ. A template only the new model ships, or none on either side, is
    no hazard. Every swap check sends plain text to /v1/completions (serve_and_compare.py), so none applies a template."""
    m, b = tc["model"], tc["base"]
    reasons = []
    if m["chat_template_sha256"] and b["chat_template_sha256"] and not tc["chat_template_equal"]:
        reasons.append(f"The new model's chat template (sha256 {m['chat_template_sha256'][:16]}) differs from the nearest "
                       f"model's (sha256 {b['chat_template_sha256'][:16]}).")
    if m["chat_template_sha256"] is None and b["chat_template_sha256"]:
        reasons.append("The new model ships no chat template and the nearest model does, so the staged model directory has "
                       "none, and transformers 5.19.0 raises an error when asked to apply one.")
    if not tc["special_tokens_equal"]:
        reasons.append("Its special tokens differ from the nearest model's: "
                       + "; ".join(f"{k} {token_text(d['model'])} against {token_text(d['base'])}"
                                  for k, d in tc["special_tokens_diffs"].items())
                       + ".")
    if not reasons:
        return []
    return [{"area": "chat_template", "evidence": [ev("tokenizer-compare.json")],
             "finding": (" ".join(reasons) + " The swap checks in stages 2 and 4 send plain text to /v1/completions, so none "
                         "of them applies a chat template. Chat output was not checked.")}]


def graft_hazard(nearest_id: str, graft: dict, ev) -> dict:
    """The one hazard line for a graft candidate: what it would do, or why it cannot be done."""
    return {"area": "drafter_graft",
            "finding": ((f"The {graft['base_tensors']} mtp.* tensors of {nearest_id} could be grafted onto the new model, which "
                         "would let the drafter and on-device sampling stay on. Stage 2 does not graft by itself: after the safe "
                         "configuration passes, its graft trial serves the grafted variant once and keeps it only when the measurement "
                         "says it is better (stages/2/graft-decision.json; --no-graft-trial skips the trial). The head was "
                         "trained on the nearest model, so the drafter's acceptance on the new model is not known until it is measured.")
                        if graft["graftable"] else
                        (f"The {graft['base_tensors']} mtp.* tensors of {nearest_id} cannot be grafted onto the new model: "
                         + "; ".join(r for r in (
                             "they are not all BF16 (" + ", ".join(graft["dtypes"]) + ")" if graft["dtypes"] != ["BF16"] else "",
                             f"they sit in {len(graft['base_shards'])} shards" if len(graft["base_shards"]) != 1 else "",
                             f"mtp.norm.weight does not match the new model's hidden size {graft['hidden_size']}"
                             if graft["shapes_match"] is False else "") if r) + ".")),
            "evidence": [ev("tensor-compare.json")]}


def hazards(model_id: str, nearest_id: str, t: dict, f: dict, g: dict, disk: dict, ev, tc: dict,
            graft: dict | None = None) -> list[dict]:
    out = [
        {"area": "tensor_cache",
         "finding": (f"A converted tensor cache of the nearest model ({nearest_id}) is keyed by layer "
                     f"name only. The new model shares {t['normalized_name_sets']['common']} tensor names "
                     "with it after the prefix is normalized, so that cache would serve the nearest "
                     "model's weights under the new model's names without any error. Every configuration "
                     "of this run needs a fresh, empty tensor cache."),
         "evidence": [ev("tensor-compare.json")]},
        {"area": "drafter",
         "finding": (f"A speculative-decoding drafter trained on the nearest model ({nearest_id}) learned "
                     f"that model's outputs. Its acceptance rate on {model_id} may differ. Measure the "
                     "acceptance rate on the new model before relying on the nearest model's figure."),
         "evidence": [ev("config-compare.json")]},
        {"area": "disk",
         "finding": (f"The run directory's filesystem has {disk['free_gib']} GiB free (measured with "
                     f"os.statvfs). The new model's weight shards total {f['model']['weight_shard_total_gib']} "
                     "GiB, and each converted tensor cache adds space of its own. Check the free space "
                     "against each heavy stage's need before it starts."),
         "evidence": [ev("disk-free.json"), ev("files-compare.json")]},
    ]
    mtp = t["mtp_tensors"]
    if mtp["model"] == 0 and mtp["base"] > 0:
        out.append({"area": "drafter",
                    "finding": (f"The new model has no mtp.* tensors; the nearest model ({nearest_id}) has "
                                f"{mtp['base']}. A speculative drafter that needs the MTP head cannot run on "
                                "the new model: the nearest bundle's engine stops at start with 'model has no "
                                "MTP head'. Swap checks and packages must serve it without the drafter, and "
                                "sampling then moves to the host. Speed figures from the nearest model's "
                                "packages do not carry over."),
                    "evidence": [ev("tensor-compare.json")]})
    if graft:
        out.append(graft_hazard(nearest_id, graft, ev))
    out += generation_config_hazards(nearest_id, g, ev)
    out += stop_token_hazards(g, ev)
    out += sampling_default_hazards(g, ev)
    out += chat_template_hazards(tc, ev)
    if g["license_changed"] is True:
        out.append({"area": "license",
                    "finding": (f"The license changed from {g['base_license']['license']} (nearest model) to "
                                f"{g['model_license']['license']} (new model). The new license governs how "
                                "this run's output may be served and shared. Read it before any publish."),
                    "evidence": [ev("genconfig-license.json")]})
    return out


REPORTED_DRAFT = "reported-results.draft.json"


def reported_draft(model_dir: Path, model_id: str, ev) -> list[dict]:
    """Write evidence/reported-results.draft.json from the model card's own tables and model-index, and return the one
    hazard that says nobody has confirmed those scores. Nothing is written when the card has no README or nothing in it
    reads as a table or model-index entry. The draft is not reported-results.json: a person accepts rows with
    `tt-orchard reported accept`, and only accepted rows reach a package card."""
    readme = model_dir / "README.md"
    try:
        with open(readme, "rb") as fh:
            raw = fh.read(MAX_README_BYTES + 1)
    except OSError:
        return []
    res = extract_reported(raw.decode("utf-8", errors="replace"), model_id)
    if not res["rows"] and not res["skipped"]:
        return []
    repo, _, rev = model_id.partition("@")
    res["truncated"] = bool(res["truncated"] or len(raw) > MAX_README_BYTES)
    write_json(REPORTED_DRAFT, {
        "model": model_id,
        "source": f"the {repo} model card (README.md{', revision ' + rev[:12] if rev else ''})",
        "extractor": 1, "readme_sha256": sha256_text(raw.decode("utf-8", errors="replace")),
        **res,
        "draft": ("Read from the model card by delta_triage.py. These are the authors' claims, quoted with their line. "
                  "Nobody has confirmed them. A row reaches a package card only after `tt-orchard reported accept`."),
    })
    n, left = len(res["rows"]), len(res["skipped"]) + res["skipped_dropped"]
    if not n:
        return []
    kinds = collections.Counter(r["extracted_from"]["kind"] for r in res["rows"])
    plural = lambda k, one, many: one if k == 1 else many      # noqa: E731
    return [{"area": "other", "evidence": [ev(REPORTED_DRAFT)],
             "finding": (f"The model card reports {n} score{plural(n, '', 's')} for this model ({kinds['model-index']} from its "
                         f"model-index, {kinds['table']} from tables), quoted with their lines in evidence/{REPORTED_DRAFT}. "
                         "These are the authors' claims and nobody has confirmed them. No package card shows one until a "
                         "person runs `tt-orchard reported accept RUN` for chosen rows."
                         + (f" {left} more {plural(left, 'entry was', 'entries were')} left out, each with its reason in the draft."
                            if left else ""))}]


# ---- audio models in the stable-audio-tools layout (plan docs/superpowers/plans/2026-10-28-audio-bringup.md, task A2) ----------------
#
# A model of this layout has no top-level config.json. Its architecture is in `model_config.json`, its weights are one top-level
# safetensors file that holds several components under name prefixes (a diffusion transformer, an autoencoder, a conditioner), and a
# text encoder may sit in a subdirectory of its own. The text rules above (config.json keys such as hidden_size) do not apply, so
# `main` hands such a pair to `audio_main`. The comparison reads configs and tensor headers only, and a few values for the copy check.
# Every read is bounded. A config or header that does not parse leaves the class `unknown`.

AUDIO_CONFIG = "model_config.json"
AUDIO_JSON_MAX = 1 << 20                  # a config larger than this is not read
AUDIO_HEADER_MAX = 8 << 20                # a safetensors header larger than this is refused (the research code uses the same bound)
AUDIO_ROLES = {"model.model": "diffusion transformer", "pretransform.model": "autoencoder", "conditioner.conditioners": "conditioner"}
AUDIO_POINTER_KEYS = ("repo_id",)         # a config leaf that says where to load from; naming another repository is not an architecture change
# Bytes per element of the safetensors dtypes this script knows. A tensor of one of these must occupy exactly shape-product times that many
# bytes; a dtype not listed (a packed or newer format) is checked for a valid range only.
AUDIO_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1,
                     "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
AUDIO_MAX_LISTED = 50                    # differences listed per kind in the record (the counts are exact)


AUDIO_MAX_DEPTH = 32                      # containers nested deeper than this are refused
AUDIO_MAX_INT_DIGITS = 30                 # an integer longer than this is refused


def _strict_pairs(pairs):
    """A JSON object with a repeated key is refused: json.loads would keep the last value and hide the first."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name):
    raise ValueError(f"the non-finite number {name} is not allowed")


def _strict_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"the number {text} is not finite")
    return value


def _strict_int(text):
    if len(text.lstrip("-")) > AUDIO_MAX_INT_DIGITS:
        raise ValueError(f"an integer of more than {AUDIO_MAX_INT_DIGITS} digits is not allowed")
    return int(text)


def _depth(obj) -> int:
    """The nesting depth of parsed JSON, counted without recursion."""
    deepest, stack = 0, [(obj, 1)]
    while stack:
        item, d = stack.pop()
        if isinstance(item, dict):
            children = item.values()
        elif isinstance(item, list):
            children = item
        else:
            continue
        deepest = max(deepest, d)
        stack.extend((c, d + 1) for c in children)
    return deepest


def audio_loads(text: str):
    """json.loads with the limits above: no repeated key, no NaN or infinity, no huge integer, no deep nesting. Raises ValueError."""
    try:
        data = json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=_no_constant, parse_float=_strict_float, parse_int=_strict_int)
    except RecursionError:
        raise ValueError("nested too deeply") from None
    if _depth(data) > AUDIO_MAX_DEPTH:
        raise ValueError(f"nested more than {AUDIO_MAX_DEPTH} levels deep")
    return data


AUDIO_IDENTITY: dict = {}                 # path -> (device, inode, size, mtime_ns) of the FIRST open of every file the audio branch read
AUDIO_JSON_SHA: dict = {}                 # path -> sha256 of the bytes of the first read of each JSON file
AUDIO_REOPENED_DIFFERENT: set = set()     # paths whose later open did not match the first one


def audio_identity_reset() -> None:
    AUDIO_IDENTITY.clear()
    AUDIO_JSON_SHA.clear()
    AUDIO_REOPENED_DIFFERENT.clear()


def _audio_open_regular(path: Path):
    """(file object, stat) of a regular file, opened once. Links are followed (a Hugging Face snapshot is made of links into blobs/), the path is
    resolved a single time, and the type is checked on the open descriptor, never by name; a FIFO is not opened for reading (O_NONBLOCK), so it
    cannot hang the script. The identity of what was opened is recorded, so audio_identity_problems can say if a file was replaced afterwards."""
    real = os.path.realpath(path)
    fd = os.open(real, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0))
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise OSError(f"{path.name} is not a regular file")
    ident = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    # The first identity is kept. A later open of the same path (the layout check opens a config and the comparison opens it again) is compared with
    # it and never replaces it, so a file swapped between two opens cannot become the new baseline.
    if AUDIO_IDENTITY.setdefault(str(path), ident) != ident:
        AUDIO_REOPENED_DIFFERENT.add(str(path))
    return os.fdopen(fd, "rb"), st


def audio_identity_problems() -> list[str]:
    """One message for every file read earlier that has since been replaced, resized, touched or removed: the evidence would then describe a
    file that is no longer there."""
    out = []
    for name, ident in sorted(AUDIO_IDENTITY.items()):
        if name in AUDIO_REOPENED_DIFFERENT:
            out.append(f"{Path(name).name} changed while stage 0 was reading it (it was a different file at a later open)")
            continue
        try:
            st = os.stat(os.path.realpath(name))
        except OSError:
            out.append(f"{Path(name).name} changed while stage 0 was reading it (it is gone)")
            continue
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != ident:
            out.append(f"{Path(name).name} changed while stage 0 was reading it (replaced, resized or touched)")
    return out


def _audio_json(path: Path):
    """(object, error): a JSON file of at most AUDIO_JSON_MAX bytes, read through one open descriptor. A missing, oversized, unreadable, non-regular or
    non-object file is an error, and so is one with a repeated key, a non-finite or huge number, or deep nesting (audio_loads)."""
    try:
        f, st = _audio_open_regular(path)
        with f:
            if st.st_size > AUDIO_JSON_MAX:
                return None, f"{path.name} is larger than {AUDIO_JSON_MAX} bytes"
            raw = f.read(AUDIO_JSON_MAX + 1)
        if len(raw) > AUDIO_JSON_MAX:
            return None, f"{path.name} is larger than {AUDIO_JSON_MAX} bytes"
        digest = hashlib.sha256(raw).hexdigest()
        if AUDIO_JSON_SHA.setdefault(str(path), digest) != digest:
            return None, f"{path.name} changed between two reads (its content is not the same)"
        data = audio_loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{path.name} is not readable JSON: {exc}"
    return (data, None) if isinstance(data, dict) else (None, f"{path.name} is not a JSON object")


def looks_like_audio(snapshot: Path) -> bool:
    """A snapshot that claims the stable-audio-tools layout: no top-level config.json and a `model_config.json` entry (a file, a directory or a link,
    readable or not). It is routed to the audio branch even when the config cannot be read, so such a run is recorded as audio with class unknown and
    stops there, and is never handed to the text path (whose result carries no audio facts and so would not stop before stage 1)."""
    if os.path.lexists(snapshot / "config.json"):
        return False
    return os.path.lexists(snapshot / AUDIO_CONFIG)


def audio_layout_problem(snapshot: Path) -> str | None:
    """Why `snapshot` is not a usable stable-audio-tools layout, or None when it is: a top-level config.json (a text model), a model_config.json that
    cannot be read or lacks a string `model_type` or a `model` object, or no top-level safetensors file."""
    if os.path.lexists(snapshot / "config.json"):
        return "it has a top-level config.json, so it is a text model"
    cfg, err = _audio_json(snapshot / AUDIO_CONFIG)
    if err:
        return err
    if not isinstance(cfg.get("model_type"), str) or not isinstance(cfg.get("model"), dict):
        return f"{AUDIO_CONFIG} has no string model_type and model object"
    try:
        has_weights = any(p.name.endswith(".safetensors") for p in snapshot.iterdir() if p.is_file())
    except OSError as exc:
        return f"the directory cannot be listed: {exc}"
    return None if has_weights else "it has no top-level safetensors file"


def detect_audio(snapshot: Path) -> bool:
    """True for a snapshot in the stable-audio-tools layout: a readable `model_config.json` with a string `model_type` and a `model`
    object, no top-level config.json, and at least one top-level safetensors file."""
    return audio_layout_problem(snapshot) is None


ENTRY_KEYS = {"dtype", "shape", "data_offsets"}


def audio_header(path: Path) -> dict:
    """{name: {"dtype", "shape", "shard"}} for one safetensors file, header only. The header is refused when it is over AUDIO_HEADER_MAX, repeats a
    name or any key, holds a non-finite number, has `__metadata__` that is not an object of text values, or has an entry with other keys than
    dtype, shape and data_offsets, a dtype that is not text, a shape that is not a list of non-negative whole numbers, or a byte range that is
    not two whole numbers with 0 <= start <= end <= the data section. For a dtype this script knows, the range must be exactly the shape product
    times the element size. The non-empty ranges must tile the data section: no overlap (two tensors cannot share bytes), no gap (bytes no
    tensor claims could hide data) and no bytes left over after the last one."""
    f, st = _audio_open_regular(path)
    with f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("shorter than 8 bytes")
        length = struct.unpack("<Q", raw)[0]
        if length > AUDIO_HEADER_MAX:
            raise ValueError(f"header length {length} is larger than {AUDIO_HEADER_MAX}")
        blob = f.read(length)
    if len(blob) != length:
        raise ValueError(f"header is cut short ({len(blob)} of {length} bytes)")
    header = audio_loads(blob.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    data_size = st.st_size - 8 - length          # the size of the file that was opened, not a second look at the name
    meta = header.get("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        raise ValueError("__metadata__ is not an object of text values")
    out, spans = {}, []
    for name, v in header.items():
        if name == "__metadata__":
            continue
        shape = v.get("shape") if isinstance(v, dict) else None
        offsets = v.get("data_offsets") if isinstance(v, dict) else None
        if (not isinstance(v, dict) or set(v) != ENTRY_KEYS
                or not isinstance(shape, list) or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
                or not isinstance(v.get("dtype"), str)
                or not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(o, int) and not isinstance(o, bool) for o in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_size):
            raise ValueError(f"tensor {name} has a malformed entry")
        size = AUDIO_DTYPE_BYTES.get(v["dtype"])
        if size is not None:
            expected = _numel(shape) * size
            if offsets[1] - offsets[0] != expected:
                raise ValueError(f"tensor {name} has a malformed entry: its byte span is {offsets[1] - offsets[0]}, but shape {shape} "
                                 f"of {v['dtype']} needs {expected}")
        if offsets[1] > offsets[0]:
            spans.append((offsets[0], offsets[1], name))
        out[name] = {"dtype": v["dtype"], "shape": shape, "shard": path.name}
    spans.sort()
    cursor = 0
    for start, end, name in spans:
        if start < cursor:
            raise ValueError(f"tensor {name} has a byte range [{start}, {end}) that overlaps the one before it (a tensor cannot share bytes)")
        if start > cursor:
            raise ValueError(f"the data section has {start - cursor} bytes before tensor {name} that no tensor claims (they could hide data)")
        cursor = end
    if cursor != data_size:
        raise ValueError(f"the data section has {data_size - cursor} bytes after the last tensor that no tensor claims (they could hide data)")
    return out


def audio_tensors(directory: Path) -> tuple[dict, list[str]]:
    """({name: {"dtype", "shape", "shard"}}, errors) over the top-level safetensors files of `directory`. A tensor name that appears in two
    files is an error (the second would otherwise replace the first and could hide an incompatible shape); the first one is kept."""
    tensors, errors = {}, []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.name.endswith(".safetensors"):
            try:
                header = audio_header(p)
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                errors.append(f"{p.name}: {exc}")
                continue
            for name, info in header.items():
                if name in tensors:
                    errors.append(f"tensor {name} appears in both {tensors[name]['shard']} and {p.name}")
                else:
                    tensors[name] = info
    return tensors, errors


def _numel(shape: list) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def audio_components(tensors: dict) -> list[dict]:
    """The tensors grouped by their first two name parts: count, parameters, dtypes and a role when the prefix is a known one."""
    out: dict[str, dict] = {}
    for name, t in tensors.items():
        prefix = ".".join(name.split(".")[:2])
        c = out.setdefault(prefix, {"prefix": prefix, "role": AUDIO_ROLES.get(prefix), "tensors": 0, "parameters": 0, "dtypes": {}})
        c["tensors"] += 1
        c["parameters"] += _numel(t["shape"])
        c["dtypes"][t["dtype"]] = c["dtypes"].get(t["dtype"], 0) + 1
    return [out[k] for k in sorted(out)]


_MISSING = object()


def _audio_flatten(obj, prefix: str = "") -> dict:
    """{dotted path: value}, with an empty dict or list kept as a leaf of its own, so a key whose value is `{}` or `[]` is still a key."""
    out = {}
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            out.update(_audio_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list) and obj:
        for i, v in enumerate(obj):
            out.update(_audio_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _same_json(a, b) -> bool:
    """Equal in JSON terms: the same type and value (Python's 1 == True == 1.0 is not enough), and empty containers of one kind."""
    return type(a) is type(b) and a == b          # the _MISSING sentinel has a type no JSON value has


def audio_field_diffs(a, b) -> dict:
    """{dotted path: {"model": value, "base": value}} for every leaf that differs, with `<absent>` for a missing key. Unlike field_diffs it keeps
    empty containers (a key whose value is {} is not the same as a missing key) and tells True from 1 from 1.0."""
    fa, fb = _audio_flatten(a), _audio_flatten(b)
    out = {}
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k, _MISSING), fb.get(k, _MISSING)
        if not _same_json(x, y):
            out[k] = {"model": ABSENT if x is _MISSING else x, "base": ABSENT if y is _MISSING else y}
    return out


def compare_audio_configs(mcfg: dict, bcfg: dict) -> dict:
    """The two model_config.json files. Differences in the `model` section and in the top-level keys (sample rate, channels, sample size,
    model type) are architecture differences. A `repo_id` leaf is a pointer. The `training` section is listed and does not count."""
    def rest(c):
        return {k: v for k, v in c.items() if k not in ("model", "training")}
    top = audio_field_diffs(rest(mcfg), rest(bcfg))
    model = {f"model.{k}": v for k, v in audio_field_diffs(mcfg.get("model"), bcfg.get("model")).items()}
    training = {f"training.{k}": v for k, v in audio_field_diffs(mcfg.get("training", {}), bcfg.get("training", {})).items()}
    pointers = {k: v for k, v in model.items() if k.rsplit(".", 1)[-1] in AUDIO_POINTER_KEYS}
    return {"top_level_differences": top,
            "model_differences": {k: v for k, v in model.items() if k not in pointers},
            "pointer_differences": pointers, "training_differences": training}


def compare_audio_tensors(mt: dict, bt: dict) -> dict:
    only_m, only_b = sorted(set(mt) - set(bt)), sorted(set(bt) - set(mt))
    shape, dtype = [], []
    for name in sorted(set(mt) & set(bt)):
        m, b = mt[name], bt[name]
        if m["shape"] != b["shape"]:
            shape.append({"name": name, "model": m["shape"], "base": b["shape"]})
        elif m["dtype"] != b["dtype"]:
            dtype.append({"name": name, "model": m["dtype"], "base": b["dtype"], "tolerated": _dtype_difference_tolerated(m, b)})
    return {"model_tensors": len(mt), "base_tensors": len(bt), "common": len(set(mt) & set(bt)), "only_in_model": only_m,
            "only_in_base": only_b, "shape_differences": shape, "dtype_differences": dtype}


def audio_text_encoders(mcfg: dict, bcfg: dict, model_dir: Path, base_dir: Path, model_repo: str, base_repo: str) -> tuple[list[dict], list[str]]:
    """One row per conditioner that loads from a subdirectory. When both configs point at their own repository the two subdirectories
    are compared (config.json, tensor names, shapes and dtypes; the weights may differ). A pointer to another repository is not
    compared: its files are not in either snapshot. Returns (rows, problems): a conditioning section that is not an object, a `configs` that is
    not a list, an entry that is not an object, a `config` that is not an object, or a `subfolder` that is not text is a problem (class unknown)."""
    problems: list[str] = []

    def subs(cfg, who):
        out = {}
        cond = (cfg.get("model") or {}).get("conditioning")
        if cond is None:
            return out
        if not isinstance(cond, dict):
            problems.append(f"{who} model_config.json model.conditioning is not an object")
            return out
        confs = cond.get("configs")
        if confs is None:
            return out
        if not isinstance(confs, list):
            problems.append(f"{who} model_config.json model.conditioning.configs is not a list")
            return out
        for i, c in enumerate(confs):
            if not isinstance(c, dict):
                problems.append(f"{who} model_config.json model.conditioning.configs[{i}] is not an object")
                continue
            inner = c.get("config")
            if inner is not None and not isinstance(inner, dict):
                problems.append(f"{who} model_config.json model.conditioning.configs[{i}].config is not an object")
                continue
            sub = inner.get("subfolder") if inner else None
            if sub is not None and not isinstance(sub, str):
                problems.append(f"{who} model_config.json model.conditioning.configs[{i}].config.subfolder is not text")
            elif isinstance(sub, str) and sub and not os.path.isabs(sub) and ".." not in Path(sub).parts:
                out[sub] = inner.get("repo_id")
        return out
    ms, bs = subs(mcfg, "new model"), subs(bcfg, "nearest model")
    rows = []
    for sub in sorted(set(ms) | set(bs)):
        row = {"subfolder": sub, "compared": False, "equal": None, "problems": [], "differences": []}
        rows.append(row)
        if sub not in ms or sub not in bs:
            row["differences"].append(f"{sub} is named by the {'nearest' if sub in bs else 'new'} model's config only")
            continue
        if ms[sub] not in (model_repo, base_repo) or bs[sub] not in (model_repo, base_repo):    # a repo_id naming either of the two is local
            row["note"] = f"loads from another repository ({ms[sub]!r} and {bs[sub]!r}); not in either snapshot, so not compared"
            continue
        row["compared"] = True
        mdir, bdir = model_dir / sub, base_dir / sub
        mc, merr = _audio_json(mdir / "config.json")
        bc, berr = _audio_json(bdir / "config.json")
        row["problems"] += [f"{sub}: {e}" for e in (merr, berr) if e]
        mt, e1 = audio_tensors(mdir) if mdir.is_dir() else ({}, [f"{sub} is not a directory in the new model"])
        bt, e2 = audio_tensors(bdir) if bdir.is_dir() else ({}, [f"{sub} is not a directory in the nearest model"])
        row["problems"] += [f"{sub}: {e}" for e in e1 + e2]
        if not mt or not bt:
            row["problems"].append(f"{sub} holds no readable safetensors file in " + ("the new model" if not mt else "the nearest model"))
        if row["problems"]:
            continue
        config_diff = audio_field_diffs(mc, bc)
        if config_diff:
            row["differences"].append(f"{sub}/config.json differs: " + short(list(config_diff), 5))
        diff = compare_audio_tensors(mt, bt)
        for kind, key in (("only in the new model", "only_in_model"), ("only in the nearest model", "only_in_base")):
            if diff[key]:
                row["differences"].append(f"{len(diff[key])} {sub} tensors exist {kind}: {short(diff[key], 3)}")
        for x in diff["shape_differences"]:
            row["differences"].append(f"{sub} tensor {x['name']} has shape {x['model']} and {x['base']}")
        for x in diff["dtype_differences"]:
            if not x["tolerated"]:
                row["differences"].append(f"{sub} tensor {x['name']} has dtype {x['model']} and {x['base']}")
        row["equal"] = not row["differences"]
    return rows, problems


def _int_or_none(cfg, key):
    value = cfg.get(key) if isinstance(cfg, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def audio_main(cfg: dict) -> int:
    """The stage 0 comparison for a model in the stable-audio-tools layout (see the block comment above)."""
    run_dir = os.path.realpath(cfg["run_dir"])
    model_dir, base_dir = Path(cfg["model_snapshot"]), Path(cfg["nearest_snapshot"])
    model_id = with_revision(cfg["model_id"].strip(), cfg["model_snapshot"])
    nearest_id = with_revision(cfg["nearest_model_id"].strip(), cfg["nearest_snapshot"])
    model_repo, base_repo = model_id.split("@", 1)[0], nearest_id.split("@", 1)[0]

    def ev(name: str) -> str:
        return os.path.relpath(EVIDENCE / name, run_dir)

    say(f"comparing {model_id} with {nearest_id} (audio layout: model_config.json and combined safetensors)")
    reasons: list[str] = []          # differences that make this a full port
    problems: list[str] = []         # things that could not be read or compared: the class is unknown
    audio_identity_reset()
    model_problem = audio_layout_problem(model_dir)
    if model_problem:
        problems.append(f"the new model is not a usable stable-audio-tools layout: {model_problem}")
    base_problem = audio_layout_problem(base_dir)
    base_is_audio = base_problem is None
    if base_problem:
        problems.append(f"the nearest model is not in the stable-audio-tools layout, so the two cannot be compared: {base_problem}")
    mcfg = None if model_problem else _audio_json(model_dir / AUDIO_CONFIG)[0]
    bcfg = _audio_json(base_dir / AUDIO_CONFIG)[0] if base_is_audio else None
    mt, merrs = ({}, []) if model_problem else audio_tensors(model_dir)
    bt, berrs = audio_tensors(base_dir) if base_is_audio else ({}, [])
    problems += [f"new model {e}" for e in merrs] + [f"nearest model {e}" for e in berrs]
    # A header with no tensor entry reads without an error. Two empty maps would otherwise look equal, and one empty map would look like a
    # full port; neither says anything about the model, so the class is unknown.
    if not model_problem and not merrs and not mt:
        problems.append("the new model holds no tensors: its safetensors header is empty")
    if base_is_audio and not berrs and not bt:
        problems.append("the nearest model holds no tensors: its safetensors header is empty")
    config = {"top_level_differences": {}, "model_differences": {}, "pointer_differences": {}, "training_differences": {}}
    tensors = compare_audio_tensors(mt, bt)
    encoders: list[dict] = []
    if not problems:
        config = compare_audio_configs(mcfg, bcfg)
        for k, v in {**config["top_level_differences"], **config["model_differences"]}.items():
            reasons.append(f"model_config.json {k} differs: {fmt(v['model'])} vs {fmt(v['base'])}")
        for k in ("only_in_model", "only_in_base"):
            if tensors[k]:
                who = "the new model" if k == "only_in_model" else "the nearest model"
                reasons.append(f"{len(tensors[k])} tensors exist only in {who}: {short(tensors[k], 5)}")
        for x in tensors["shape_differences"]:
            reasons.append(f"tensor {x['name']} has shape {x['model']} in the new model and {x['base']} in the nearest model")
        for x in tensors["dtype_differences"]:
            if not x["tolerated"]:
                reasons.append(f"tensor {x['name']} has dtype {x['model']} in the new model and {x['base']} in the nearest model")
        encoders, encoder_problems = audio_text_encoders(mcfg, bcfg, model_dir, base_dir, model_repo, base_repo)
        problems += encoder_problems
        for row in encoders:
            problems += row["problems"]
            reasons += row["differences"]
    f = compare_files(model_dir, base_dir)
    write_json("files-compare.json", f)
    values = ({"samples": [], "summary": {}, "skipped": [], "note": "not run: the headers could not be read"} if problems
              else sample_value_diffs(model_dir, base_dir, {k: k for k in sorted(set(mt) & set(bt))}))
    write_json("value-compare.json", values)
    # The value sample opens the weight files by name again. Checking every file's identity after it, a file replaced at any point since it was first
    # read makes the evidence describe something that is no longer there, and the class unknown.
    problems += audio_identity_problems()
    cls = "unknown" if problems else "full-port" if reasons else "weights-only"
    path = "weights-only" if cls == "weights-only" else "full-port"
    all_reasons = problems + reasons
    components = audio_components(mt)
    audio = {"model_type": mcfg.get("model_type") if mcfg else None, "sample_rate": _int_or_none(mcfg, "sample_rate"),
             "audio_channels": _int_or_none(mcfg, "audio_channels"), "sample_size": _int_or_none(mcfg, "sample_size"),
             "components": components, "base_components": audio_components(bt),
             "config": {k: dict(list(v.items())[:AUDIO_MAX_LISTED]) for k, v in config.items()},
             "tensors": {"model_tensors": tensors["model_tensors"], "base_tensors": tensors["base_tensors"], "common": tensors["common"],
                         "only_in_model": tensors["only_in_model"][:AUDIO_MAX_LISTED], "only_in_base": tensors["only_in_base"][:AUDIO_MAX_LISTED],
                         "shape_differences": tensors["shape_differences"][:AUDIO_MAX_LISTED],
                         "dtype_differences": tensors["dtype_differences"][:AUDIO_MAX_LISTED]},
             "text_encoders": encoders, "files_only_in_base": f["files_only_in_base"] + f["shards_only_in_base"],
             "files_only_in_model": f["files_only_in_model"] + f["shards_only_in_model"]}
    write_json("audio-compare.json", {**audio, "problems": problems, "reasons": reasons})
    ml, bl = license_side(model_dir), license_side(base_dir)
    say(f"wrote evidence/audio-compare.json, files-compare.json and value-compare.json ({len(mt)} and {len(bt)} tensors; "
        f"{len(components)} components)")
    code = code_files(model_dir)
    comp_text = ", ".join(f"{c['prefix']} ({c['role'] or 'no known role'}, {c['tensors']} tensors, {c['parameters']:,} parameters)"
                          for c in components) or "none"
    cd = (config["model_differences"], config["top_level_differences"], config["pointer_differences"], config["training_differences"])
    delta = {
        "model": model_id, "nearest_model": nearest_id, "path": path, "class": cls, "template_sha256": THIS_SHA256,
        "path_reasons": all_reasons, "sidecars": [], "code_files": code,
        "facts": {"form": "full", "modality": "audio"},
        "values": {"summary": values["summary"], "samples": [{k: x[k] for k in ("model_tensor", "relation", "cosine")}
                                                              for x in values["samples"]]},
        "audio": audio,
        "draft": "written by delta_triage.py from the evidence files; the agent reviews each finding and adds hazards it can justify",
        "differences": [
            {"area": "config", "evidence": [ev("audio-compare.json")],
             "finding": (f"model_config.json: {len(cd[0])} differences in the model section, {len(cd[1])} in the top-level keys "
                         f"(sample rate, channels, sample size, model type), {len(cd[2])} pointers to another repository (ignored) and "
                         f"{len(cd[3])} in the training section (listed, ignored).")},
            {"area": "architecture", "evidence": [ev("audio-compare.json")],
             "finding": f"The new model's components: {comp_text}."},
            {"area": "tensors", "evidence": [ev("audio-compare.json")],
             "finding": (f"{tensors['common']} tensors are in both models; {len(tensors['only_in_model'])} only in the new model, "
                         f"{len(tensors['only_in_base'])} only in the nearest model, {len(tensors['shape_differences'])} with another "
                         f"shape and {len(tensors['dtype_differences'])} with another dtype.")},
            {"area": "tensor_names", "evidence": [ev("audio-compare.json")],
             "finding": ("The names are compared as written (no prefix rewrite is applied to this layout): "
                         + ("all shared names match." if not tensors["only_in_model"] and not tensors["only_in_base"]
                            else f"{len(tensors['only_in_model'])} and {len(tensors['only_in_base'])} names do not match."))},
            {"area": "files", "evidence": [ev("files-compare.json")], "finding": files_finding(f)},
            {"area": "license", "evidence": [ev("files-compare.json")],
             "finding": (f"The new model's license is {ml['license'] or 'unknown'} (name {ml['readme_license_name'] or 'none'}); the nearest "
                         f"model's is {bl['license'] or 'unknown'} (name {bl['readme_license_name'] or 'none'}).")},
        ],
        "hazards": [
            {"area": "other", "evidence": [ev("audio-compare.json")],
             "finding": ("No stage listens to the output. Whether the generated audio sounds right is a person's judgment; a check that "
                         "the sound exists (duration, level, no clipping) says nothing about quality or about the prompt being followed.")},
            {"area": "license", "evidence": [ev("files-compare.json")],
             "finding": (f"The model card's license is {ml['license'] or 'unknown'}"
                         + (f" ({ml['readme_license_name']})" if ml["readme_license_name"] else "")
                         + ". Read the LICENSE files in the snapshot before anything is packaged or published; a text encoder may be "
                           "redistributed under its own terms.")},
        ],
    }
    if values["summary"].get("identical") and values["summary"]["identical"] == sum(values["summary"].values()):
        delta["hazards"].append({"area": "other", "evidence": [ev("value-compare.json")],
                                 "finding": ("Every sampled tensor is identical to the nearest model's. The new model may be a copy of it, "
                                             "so a bring-up of it would measure the nearest model. Compare the files' hashes before going on.")})
    if f["files_only_in_base"] or f["shards_only_in_base"]:
        delta["hazards"].append({"area": "other", "evidence": [ev("files-compare.json")],
                                 "finding": ("The nearest model has files the new model lacks: "
                                             + short(f["files_only_in_base"] + f["shards_only_in_base"]) + ". They were not read; whether "
                                             "serving the base needs them is not known.")})
    for row in encoders:
        if row.get("note"):
            delta["hazards"].append({"area": "other", "evidence": [ev("audio-compare.json")],
                                     "finding": f"Text encoder {row['subfolder']}: {row['note']}."})
    if code:
        delta["hazards"].append({"area": "other", "evidence": [ev("files-compare.json")],
                                 "finding": ("The model repository ships code: " + ", ".join(f"{x['file']} (sha256 {x['sha256'][:12]})" for x in code)
                                             + ". It was not run. Read it before any run imports it.")})
    delta["hazards"].extend(reported_draft(model_dir, model_id, ev))
    research = base_research(cfg, nearest_id, cls)
    if research is not None:
        delta["base_research"] = research
    (HERE_DIR / "delta.json").write_text(json.dumps(delta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    say(f"wrote delta.json: audio layout, path {path}, class {cls}")
    for r in all_reasons:
        say(f"  {'unknown' if r in problems else 'full-port'} because {r}")
    return 0


def main() -> int:
    cfg = load_config()
    run_dir = os.path.realpath(cfg["run_dir"])
    model_dir, base_dir = Path(cfg["model_snapshot"]), Path(cfg["nearest_snapshot"])
    if looks_like_audio(model_dir):
        return audio_main(cfg)
    model_id = with_revision(cfg["model_id"].strip(), cfg["model_snapshot"])
    nearest_id = with_revision(cfg["nearest_model_id"].strip(), cfg["nearest_snapshot"])

    def ev(name: str) -> str:
        return os.path.relpath(EVIDENCE / name, run_dir)

    say(f"comparing {model_id} with {nearest_id}")
    c = compare_configs(model_dir, base_dir)
    write_json("config-compare.json", c)
    say("wrote evidence/config-compare.json")
    t = compare_tensors(model_dir, base_dir)
    write_json("tensor-compare.json", t)
    say(f"wrote evidence/tensor-compare.json ({t['model']['num_tensors']} and {t['base']['num_tensors']} tensors)")
    tc = compare_tokenizers(model_dir, base_dir)
    write_json("tokenizer-compare.json", tc)
    say("wrote evidence/tokenizer-compare.json")
    enc = encode_test(model_dir, base_dir)
    write_json("tokenizer-encode.json", enc)
    say(f"wrote evidence/tokenizer-encode.json ({enc.get('differing', 'not run')} strings differ)")
    f = compare_files(model_dir, base_dir)
    write_json("files-compare.json", f)
    g = compare_genconfig_license(model_dir, base_dir)
    write_json("genconfig-license.json", g)
    disk = disk_free(run_dir)
    write_json("disk-free.json", disk)
    say("wrote evidence/files-compare.json, genconfig-license.json and disk-free.json")

    model_tensors, base_tensors = read_tensors(model_dir)[1], read_tensors(base_dir)[1]
    backbone_names = set(model_tensors)
    graft = graft_candidate(model_tensors, base_tensors, model_hidden_size(model_dir),
                            base_dir.name if re.fullmatch(r"[0-9a-f]{40}", base_dir.name) else None)
    sc = compare_sidecars(model_dir, base_dir, backbone_names)
    write_json("sidecar-compare.json", sc)
    code = code_files(model_dir)
    say(f"wrote evidence/sidecar-compare.json ({len(sc['sidecars'])} sidecars, {len(code)} code files)")
    values = sample_value_diffs(model_dir, base_dir, raw_pair_map(model_tensors, base_tensors))
    write_json("value-compare.json", values)
    say(f"wrote evidence/value-compare.json ({len(values['samples'])} tensors sampled: {fmt(values['summary'])})")
    head = detect_restricted_head(model_dir, base_dir, sc)
    if head.get("restricted"):
        head["tokens"] = token_names(base_dir, head["token_ids"])
        head["readout"] = readout_facts(model_dir, code)
        write_json("readout-facts.json", head["readout"])
    write_json("head-compare.json", head)
    say(f"wrote evidence/head-compare.json (restricted head: {head['restricted']}; {head['note']})")
    reasons = decide_path({**c}, {**t, "head": head}) + sc["problems"]
    vt = detect_vocab_trim(c, {**t, "head": head}, sc, model_dir, base_dir, raw_pair_map(model_tensors, base_tensors))
    if vt["trimmed"]:
        write_json("vocab-compare.json", vt)
        say(f"wrote evidence/vocab-compare.json (vocabulary trim: kept rows are rows of the nearest model's: {vt['kept_subset']}; {vt['note']})")
    cls = decide_class(c, {**t, "head": head}, reasons, sc, vt)
    if cls == "vocab-trim":
        reasons = []
    if cls == "restricted-head":
        reasons = [(f"the nearest model's lm_head.weight is replaced by {head['rows']} sidecar rows ({head['sidecar']}) that are "
                    f"copies of rows of it: the harness does not serve a restricted head, so a person decides")]
    path = "full-port" if reasons else "weights-only"
    ns = t["normalized_name_sets"]
    delta = {
        "model": model_id, "nearest_model": nearest_id, "path": path, "class": cls,
        "template_sha256": THIS_SHA256, "path_reasons": reasons, "sidecars": sc["sidecars"], "code_files": code,
        "alignment": {"pairs": ns["common"], "aligned_by_prefix": ns["aligned"]["pairs"], "rules": ns["aligned"]["rules"],
                      "raw_rules": ns["aligned"]["raw_rules"], "all_pairs": ns["aligned"]["all_pairs"], "only_in_model": ns["only_in_model"][:50],
                      "only_in_base": ns["only_in_base"][:50]},
        "values": {"summary": values["summary"], "samples": [{k: x[k] for k in ("model_tensor", "relation", "cosine")}
                                                              for x in values["samples"]]},
        "head": head,
        **({"vocab_trim": vt} if cls == "vocab-trim" else {}),
        "mtp": {"nearest_tensors": t["mtp_tensors"]["base"], "new_tensors": t["mtp_tensors"]["model"]},
        **({"graft_candidate": graft} if graft else {}),
        "draft": ("written by delta_triage.py from the evidence files; the agent reviews each finding "
                  "and adds hazards it can justify"),
        "differences": [
            {"area": "config", "finding": config_finding(c), "evidence": [ev("config-compare.json")]},
            {"area": "architecture", "finding": architecture_finding(c, t),
             "evidence": [ev("config-compare.json"), ev("tensor-compare.json")]},
            {"area": "tensors", "finding": tensors_finding(t), "evidence": [ev("tensor-compare.json")]},
            {"area": "tensor_names", "finding": tensor_names_finding(t), "evidence": [ev("tensor-compare.json")]},
            {"area": "tokenizer", "finding": tokenizer_finding(tc, enc),
             "evidence": [ev("tokenizer-compare.json"), ev("tokenizer-encode.json")]},
            {"area": "chat_template", "finding": chat_template_finding(tc),
             "evidence": [ev("tokenizer-compare.json")]},
            {"area": "files", "finding": files_finding(f), "evidence": [ev("files-compare.json")]},
            {"area": "generation_config", "finding": genconfig_finding(g),
             "evidence": [ev("genconfig-license.json")]},
            {"area": "license", "finding": license_finding(g), "evidence": [ev("genconfig-license.json")]},
        ],
        "hazards": hazards(model_id, nearest_id, t, f, g, disk, ev, tc, graft),
    }
    if sc["sidecars"]:
        delta["differences"].append({"area": "other", "finding": sidecar_finding(sc),
                                     "evidence": [ev("sidecar-compare.json")]})
        delta["hazards"].append({"area": "other", "evidence": [ev("sidecar-compare.json")],
                                 "finding": ("The new model has a sidecar the nearest model's runtime does not "
                                             "load. A check that only generates text from the backbone does "
                                             "not exercise it. Its output must be checked on the host against "
                                             "the CPU reference, with the exact file hash recorded.")})
    delta["hazards"].extend(tolerated_dtype_hazards(t, ev))
    if ns["aligned"]["pairs"]:
        # Names paired by prefix count as the nearest model's tensors, so the path stays weights-only. Only the stage 1 CPU reference
        # loader and the restricted-head parity script apply the name map. The swap and serve templates of stages 2 and 4 link the
        # new model's files as they are, so the TT runtime sees the new names.
        delta["hazards"].append({"area": "other", "evidence": [ev("tensor-compare.json")],
                                 "finding": (f"{ns['aligned']['pairs']} tensors of the new model were paired by prefix with the nearest "
                                             "model's tensors, because their names differ (alignment.rules lists the rewrites). Only "
                                             "the stage 1 CPU reference applies that map. Nothing in stages 2 and 4 renames the weights "
                                             "before the TT runtime reads them, and stage 0 does not check that the runtime accepts the "
                                             "new names. If a boot finds no weights or the wrong ones, check the tensor names first.")})
    if values["samples"]:
        delta["differences"].append({"area": "other", "finding": values_finding(values),
                                     "evidence": [ev("value-compare.json")]})
    if head.get("restricted"):
        delta["differences"].append({"area": "other", "finding": head_finding(head), "evidence": [ev("head-compare.json")]})
        declared = head["readout"]["declared"]
        stated = ("; ".join(f"{d['file']} says attention_mode {d['attention_mode']!r}"
                            + (f" and pooling {d['pooling']!r}" if d.get("pooling") else "") for d in declared)
                  if declared else "")
        delta["hazards"].append({"area": "other", "evidence": [ev("head-compare.json"), ev("readout-facts.json")],
                                 "finding": ("A restricted head is read in a way the nearest model's runtime may not support: the "
                                             "answers come from a few token rows, and the model's own code decides at which positions "
                                             "they are read and under which attention mask. "
                                             + (f"The model's own files state it: {stated}. " if stated else "")
                                             + "A runtime that keeps only the last position's hidden state, or applies a causal mask, "
                                             "would give plausible but wrong probabilities. Neither is in the weights or the config. "
                                             "The harness stops after stage 0 on this class; read code_files (code_hints in "
                                             "evidence/readout-facts.json says where) before deciding.")})
    elif head.get("restricted") is None and sc["sidecars"]:
        delta["hazards"].append({"area": "other", "evidence": [ev("head-compare.json")],
                                 "finding": f"The restricted-head check did not run: {head['note']}. A sidecar may be a head cut down "
                                            "from the nearest model's lm_head; compare its rows with lm_head.weight by hand."})
    if cls == "vocab-trim":
        delta["hazards"].append({"area": "other", "evidence": [ev("vocab-compare.json")],
                                 "finding": (f"The new model's vocabulary has {vt['model_vocab']} entries against the nearest model's {vt['base_vocab']}, and "
                                             "every row of its embedding and lm_head is a row of the nearest model's. The token ids are renumbered "
                                             "(the tokenizer encode test shows how many strings change), so a client that sends or reads token ids "
                                             "gets different ids. Stages 2 and 4 write the model's own vocab_size into the served config.json and "
                                             "switch the speculative drafter off, because the drafter works on the nearest model's token ids. One "
                                             "model was served this way on hardware (ProprietaryLegal/Qwen3.8-27B-CJK-Trim, 30 of 32 tokens equal "
                                             "to the CPU reference); the answers of a trimmed model beyond that, 4 chips and the packaged form "
                                             "were not tested.")})
    if code:
        delta["hazards"].append({"area": "other", "evidence": [ev("files-compare.json")],
                                 "finding": ("The model repository ships code: "
                                             + ", ".join(f"{x['file']} (sha256 {x['sha256'][:12]})" for x in code)
                                             + ". It was not run. Read it before any run imports it, and "
                                             "record the hash of the file a check used.")})
    delta["hazards"].extend(reported_draft(model_dir, model_id, ev))
    research = base_research(cfg, nearest_id, cls)
    if research is not None:
        delta["base_research"] = research
    (HERE_DIR / "delta.json").write_text(json.dumps(delta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    say(f"wrote delta.json: path {path}, class {cls}")
    for r in reasons:
        say(f"  full-port because {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
