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
                             pre-tokenizer, decoder and post-processor, and the chat template's sha256
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
- the two models share at least one text tensor, and every shared tensor has the same shape and
  dtype;
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
import struct
import sys
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
    dtype_diffs = sorted(({"name": b, "model": mn[m]["dtype"], "base": bn[b]["dtype"]}
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


def chat_template(snapshot: Path) -> tuple[str | None, str | None]:
    """(source file, template text): chat_template.jinja first, then tokenizer_config.json."""
    jinja = snapshot / "chat_template.jinja"
    if jinja.is_file():
        return "chat_template.jinja", jinja.read_text(encoding="utf-8", errors="replace")
    cfg, _ = read_json(snapshot / "tokenizer_config.json")
    t = cfg.get("chat_template") if isinstance(cfg, dict) else None
    if isinstance(t, list):            # named templates: [{"name": ..., "template": ...}]
        t = json.dumps(t, sort_keys=True, ensure_ascii=False)
    return ("tokenizer_config.json", t) if isinstance(t, str) else (None, None)


def tokenizer_side(snapshot: Path) -> tuple[dict | None, dict, str | None]:
    """(tokenizer.json data, summary, error)."""
    data, err = read_json(snapshot / "tokenizer.json")
    src, tmpl = chat_template(snapshot)
    summary = {"chat_template_source": src, "chat_template_sha256": sha256_text(tmpl) if tmpl else None}
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
    out = {"model": ms, "base": bs, "errors": [e for e in (merr, berr) if e],
           "chat_template_equal": ms["chat_template_sha256"] == bs["chat_template_sha256"]}
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


def compare_genconfig_license(model_dir: Path, base_dir: Path) -> dict:
    mg, merr = read_json(model_dir / "generation_config.json")
    bg, berr = read_json(base_dir / "generation_config.json")
    mg, bg = mg if isinstance(mg, dict) else {}, bg if isinstance(bg, dict) else {}
    diffs = {k: {"model": mg.get(k, ABSENT), "base": bg.get(k, ABSENT)}
             for k in sorted(set(mg) | set(bg)) if mg.get(k, ABSENT) != bg.get(k, ABSENT)}
    ml, bl = license_side(model_dir), license_side(base_dir)
    changed = None if ml["license"] is None or bl["license"] is None else ml["license"] != bl["license"]
    return {"model_generation_config": mg, "base_generation_config": bg,
            "model_generation_config_present": (model_dir / "generation_config.json").is_file(),
            "base_generation_config_present": (base_dir / "generation_config.json").is_file(),
            "errors": [e for e in (merr, berr) if e],
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


def chat_template_finding(tc: dict) -> str:
    m, b = tc["model"], tc["base"]
    if m["chat_template_sha256"] is None and b["chat_template_sha256"] is None:
        return "Neither model ships a chat template."
    if tc["chat_template_equal"]:
        return (f"Identical. Both chat templates have sha256 {m['chat_template_sha256'][:16]} "
                f"(new model from {m['chat_template_source']}, nearest model from {b['chat_template_source']}).")
    if m["chat_template_sha256"] is None or b["chat_template_sha256"] is None:
        who = "the new model" if m["chat_template_sha256"] else "the nearest model"
        return f"Only {who} ships a chat template."
    return (f"Different. The new model's chat template ({m['chat_template_source']}) has sha256 "
            f"{m['chat_template_sha256'][:16]}; the nearest model's ({b['chat_template_source']}) has "
            f"{b['chat_template_sha256'][:16]}.")


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
        reasons.append(f"tensor {x['name']} has dtype {x['model']} in the new model and {x['base']} in the nearest model")
    for side, who in (("model", "the new model"), ("base", "the nearest model")):
        extra = ns[f"unexplained_only_in_{side}"]
        if side == "base" and (t.get("head") or {}).get("restricted"):
            extra = [k for k in extra if not HEAD_NAME.search(k)]      # the readout rows stand in for lm_head
        if extra:
            reasons.append(f"{len(extra)} text tensors exist only in {who}: {short(extra, 5)}")
    return reasons


def decide_class(c: dict, t: dict, reasons: list[str], sc: dict) -> str:
    """weights-only, weights+sidecar, restricted-head, full-port or unknown. `reasons` already holds the backbone's
    reasons for full-port and the sidecar problems (sc["problems"], told apart by value). restricted-head: the backbone matches and the only thing missing is an
    lm_head that a sidecar's rows stand in for (t["head"]["restricted"], set by detect_restricted_head)."""
    ns = t["normalized_name_sets"]
    could_not_compare = (c.get("errors") and "text_key_values" not in c) or bool(t["errors"]) or ns["common"] == 0
    if could_not_compare:
        return "unknown"
    # A definite difference in the backbone is full-port whatever a sidecar's problems are (summerMC/Diffusion-Qwen3.8-27B: 48 extra text tensors
    # and a dllm_extras.pt that cannot be read). The sidecar problems alone, on a backbone that lines up, are what make the class unknown.
    if [r for r in reasons if r not in sc["problems"]]:
        return "full-port"
    if sc["problems"]:
        return "unknown"
    if (t.get("head") or {}).get("restricted"):
        return "restricted-head"
    return "weights+sidecar" if sc["sidecars"] else "weights-only"


def hazards(model_id: str, nearest_id: str, t: dict, f: dict, g: dict, disk: dict, ev) -> list[dict]:
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
    if g["license_changed"] is True:
        out.append({"area": "license",
                    "finding": (f"The license changed from {g['base_license']['license']} (nearest model) to "
                                f"{g['model_license']['license']} (new model). The new license governs how "
                                "this run's output may be served and shared. Read it before any publish."),
                    "evidence": [ev("genconfig-license.json")]})
    return out


def main() -> int:
    cfg = load_config()
    run_dir = os.path.realpath(cfg["run_dir"])
    model_dir, base_dir = Path(cfg["model_snapshot"]), Path(cfg["nearest_snapshot"])
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
    cls = decide_class(c, {**t, "head": head}, reasons, sc)
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
        "mtp": {"nearest_tensors": t["mtp_tensors"]["base"], "new_tensors": t["mtp_tensors"]["model"]},
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
        "hazards": hazards(model_id, nearest_id, t, f, g, disk, ev),
    }
    if sc["sidecars"]:
        delta["differences"].append({"area": "other", "finding": sidecar_finding(sc),
                                     "evidence": [ev("sidecar-compare.json")]})
        delta["hazards"].append({"area": "other", "evidence": [ev("sidecar-compare.json")],
                                 "finding": ("The new model has a sidecar the nearest model's runtime does not "
                                             "load. A check that only generates text from the backbone does "
                                             "not exercise it. Its output must be checked on the host against "
                                             "the CPU reference, with the exact file hash recorded.")})
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
    if code:
        delta["hazards"].append({"area": "other", "evidence": [ev("files-compare.json")],
                                 "finding": ("The model repository ships code: "
                                             + ", ".join(f"{x['file']} (sha256 {x['sha256'][:12]})" for x in code)
                                             + ". It was not run. Read it before any run imports it, and "
                                             "record the hash of the file a check used.")})
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
