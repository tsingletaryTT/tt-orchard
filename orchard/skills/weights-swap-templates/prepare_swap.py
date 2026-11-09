#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the model directory and the run script for a weights-only swap (stage 2).

The weights-swap-check skill copies this file into the stage directory and runs it. It reads
`swap_config.json` from its own directory and writes two things next to itself:

1. `model-dir/`, a directory that looks like the nearest model's but holds the new weights.
   - COPIED from base_snapshot: config.json, preprocessor_config.json and
     video_preprocessor_config.json. The bundle's TT model class is registered for the nearest
     model's architecture name, and that model is a vision-language model, so its config must be
     the one vLLM sees. A file absent from base_snapshot is skipped without a message.
   - SYMLINKED from new_snapshot (absolute, fully resolved targets): tokenizer.json,
     tokenizer_config.json, chat_template.jinja, generation_config.json,
     model.safetensors.index.json and every *.safetensors file. tokenizer.json and at least one
     *.safetensors file are required (exit 2 without them); the other names are linked when present
     and reported when absent.
   - generation_config.json, when new_snapshot has none (or only a link to a blob that is gone), is COPIED from
     base_snapshot instead, and the script says so. The bundle's serving code reads `eos_token_id` from the
     model dir's generation_config.json or config.json, and config.json here is the BASE model's, which
     nests it under text_config. BAAI/AREX-2 ships no generation_config.json: the server loaded all 64 layers
     and died with "no eos_token_id found for the served checkpoint" (2026-10-21). A generation_config.json the
     model ships is always linked and never replaced. The copy also brings the base's sampling defaults
     (vLLM reads them from the model dir); the checks that use this directory send their own sampling
     settings. When neither snapshot has the file the script warns and goes on.
2. `run.sh`, a copy of `<bundle_dir>/run.sh` with these edits. Two forms of bundle run.sh are known, and the
   `--model` argument tells them apart:
   - OLDER form: `--model "<nearest_model_id>"` (quoted or unquoted), exactly once. The configuration files in
     model-dir are copied from base_snapshot, as listed above.
   - CURRENT form: `--model "$HERE/model-dir"` (also `${HERE}` or unquoted), exactly once. run.sh builds that
     directory at each start by running the bundle's prepare_model_dir.py, which reads the weights repo and revision
     from tt_kernel_manifest.json (HF_MODEL and TT_MODEL_WEIGHTS_REVISION feed only a download check, so no
     variable can select other weights) and copies every file of the bundle's base_config/ into it. This script
     then (a) copies the configuration files from `<bundle_dir>/base_config/` and none from base_snapshot, because
     base_config/ holds the configuration the bundle's model code was built and tested with (the Qwen3.5 thin
     bundle's has an edited architecture and no mrope keys), (b) replaces the single line that runs
     prepare_model_dir.py with a comment, so the copy never rebuilds the bundle's own model-dir with the nearest
     model's weights, and (c) rewrites `export MODEL_WEIGHTS_DIR=` as well as `export HF_MODEL=`. The bundle
     needs `base_config/` with a config.json, exactly one line that runs prepare_model_dir.py, and no other
     mention of `$HERE/model-dir`. A graft.json in the bundle is not applied: the drafter handling below decides,
     and `graft_mtp` is the way to graft. A command that matches neither form, or both, exits 2 with a message
     that names both forms.
   Both forms get these edits:
   - the `HERE=...` line becomes `HERE="<bundle_dir>"` (exactly once), so the copy still finds
     the bundle's venv and model code;
   - ` --revision <40 hex>` and ` --tokenizer-revision <40 hex>` are deleted, because a local
     directory has no revision. Each may appear at most once; an absent one is reported.
   - every `export HF_MODEL=...` line becomes `export HF_MODEL="<model-dir>"`, and every `export MODEL_WEIGHTS_DIR=...`
     line becomes `export MODEL_WEIGHTS_DIR="<model-dir>"`. The TT runtime
     takes its weights directory from MODEL_WEIGHTS_DIR, then HF_MODEL, then the config path.
     The bundle sets HF_MODEL to the nearest model's id, which resolves to that model's HF cache,
     so `--model <model-dir>` alone serves the BASE weights. An absent line is reported; it is not
     an error, because serve_and_compare.py also sets both variables in the server's environment.

   - when the new model has no `mtp.*` tensor, `export QWEN36_DRAFTER=...` becomes
     `export QWEN36_DRAFTER=""` (at most one such line). The nearest bundle may serve with a speculative
     drafter that needs the model's MTP head, and a model without those tensors dies at engine start with
     "model has no MTP head" (Cloudflare/clef, 2026-10-06). With the drafter off the bundle serves plain
     decoding, which gives the same greedy tokens, so the check measures the same weights. The check reads
     the new model's `model.safetensors.index.json`, or the shard headers when there is no index. When it
     cannot tell, the line is left alone and the script says so. With the drafter off the script also
     removes `"sample_on_device_mode"` from the command's single `--additional-config`, because plain
     decoding on a 1x2 mesh refuses on-device sampling ("Unset sample_on_device_mode for host sampling");
     the host samples instead.
   - when the bundle's drafter is a DFlash drafter (its last `export QWEN36_DRAFTER=` line says "dflash2"), the drafter is switched
     off in the same way whatever mtp.* tensors the new model has, if the new model's text config differs from the nearest model's
     in model_type (a "_text" suffix aside), hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim,
     intermediate_size, vocab_size or layer_types, or if either config cannot be read. A
     DFlash drafter is a separate checkpoint (DFLASH_WEIGHTS) that reads the hidden states of the model it was trained on, the
     nearest model, at fixed layers; a key only one config states is not compared. A drafter of kind "mtp" uses the model's own MTP
     head, and only the mtp.* rule above applies to it.
   - with the drafter off for any reason, the drafter's repo (the one `export DFLASH_WEIGHTS=` names, at any revision) is removed
     from the single `export TT_AUXILIARY_WEIGHTS=` list, which the bundle's launcher downloads before vLLM starts; other entries
     stay and a line left empty is removed. Without a DFLASH_WEIGHTS line the list is left alone and the script says so.
   - `"drafter": "off"` in swap_config.json does the same edits whatever the model's tensors say. A
     suite control (orchard/suite_control.py) uses it so that the nearest model is served the way a run
     without `mtp.*` tensors is served. Any other value is refused, and so is the key with `graft_mtp`.

3. `model-dir-sources.json`, next to `model-dir/` (not inside it, because vLLM reads that directory). It records
   where each file came from: the nearest model (copied) or the new model (linked), why a copy was made, the
   snapshot revisions, the files that were absent, and whose generation_config.json (`sampling_defaults_from`:
   "model", "base" or "none") the server reads its sampling defaults and stop tokens from. The weights are one
   entry with a count. The stage 8 bundle and the package card read it, so a reader does not need this script
   to know that a run served the base model's sampling defaults. It holds no machine paths. `eos_token_ids` lists
   the stop tokens the Qwen3.8 bundle's serving code will find in `model-dir/` (a top-level `eos_token_id` of
   generation_config.json, then config.json). When it is empty the script warns at once, because that bundle
   stops after loading the weights without one, 15 to 25 minutes into a start.
   `tokenizer_eos_token_id` is the id tokenizer.json gives the new model's eos token (None when it names none or the
   id cannot be found) and `tokenizer_eos_served` says whether that id is among `eos_token_ids` (None when either is
   unknown). A False answer prints a warning: a reply that ends on that token runs to the length limit.

4. Optional, `"graft_mtp": true` in swap_config.json (default off): when the new model has no `mtp.*` tensor and
   the nearest model has them, `model-dir/` also gets `model-mtp-from-base.safetensors`, a shard of the nearest
   model's `mtp.*` tensors copied byte for byte, and its copy of the weight index lists them. The drafter and
   on-device sampling stay on, because the engine then finds an MTP head. The head was trained against the
   nearest model, so how well the drafter's guesses are accepted on the new model is not known until it is
   measured. The script exits 2, and builds nothing, when the new model already has `mtp.*` tensors, when it
   cannot tell, when the nearest model has none or keeps them in more than one shard, when a tensor is missing
   from the shard or is not BF16, or when `mtp.norm.weight` does not match the new model's hidden size. The new
   snapshot's cached index is never written through (the model-dir copy replaces the link). The record in
   `model-dir-sources.json` is `mtp_graft`.

An expected edit that does not happen exactly once exits 2 with a message that names it, and no
run.sh is written. Everything else in the script, including its environment lines, stays as it is.

For a container package (stage 4's container configuration) swap_config.json names `package`
and no `bundle_dir`. Then only model-dir/ is built: serve_and_compare_container.py edits the
container's docker command and needs no run.sh.

Config keys read here: nearest_model_id, base_snapshot, new_snapshot, and bundle_dir or package.
The other keys in swap_config.json are for serve_and_compare.py and serve_and_compare_container.py.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import sys
from pathlib import Path

HERE_DIR = Path(__file__).resolve().parent
COPIED = ("config.json", "preprocessor_config.json", "video_preprocessor_config.json")
LINKED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "model.safetensors.index.json")
REQUIRED_LINKS = ("tokenizer.json",)
FALLBACK_COPY = "generation_config.json"      # copied from base_snapshot when new_snapshot has none
GRAFT_SHARD = "model-mtp-from-base.safetensors"


class SwapFail(SystemExit):
    """What `fail` raises in quiet mode: a SystemExit with exit status 2 that carries the message, so a caller that runs an edit in
    memory (orchard/weights_source.py) reads it from the exception and the process's stderr is not touched."""

    def __init__(self, message: str):
        super().__init__(2)
        self.message = message


# True inside one caller's context (a thread, or an asyncio task) that asked for quiet failures. A ContextVar keeps the choice per
# caller: two preflights on two threads cannot see each other's, which a redirect of the process-wide stderr could not promise.
QUIET = contextvars.ContextVar("prepare_swap_quiet", default=False)


def fail(message: str) -> None:
    if QUIET.get():
        raise SwapFail(message)
    print(f"prepare_swap: {message}", file=sys.stderr)
    sys.exit(2)


def load_config() -> dict:
    path = HERE_DIR / "swap_config.json"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")
    missing = [k for k in ("nearest_model_id", "base_snapshot", "new_snapshot") if not cfg.get(k)]
    if not cfg.get("bundle_dir") and not cfg.get("package"):
        missing.append("bundle_dir (or package, for a container package)")
    if missing:
        fail(f"swap_config.json is missing {missing}")
    if "graft_mtp" in cfg and not isinstance(cfg["graft_mtp"], bool):
        fail(f"swap_config.json's graft_mtp must be true or false, not {cfg['graft_mtp']!r}")
    if "drafter" in cfg:
        # "off" is the only value. It makes a control serve the way a model without mtp.* tensors is served (suite control, issue 15).
        if not isinstance(cfg["drafter"], str) or cfg["drafter"] != "off":
            fail(f"swap_config.json's drafter must be \"off\" when present, not {cfg['drafter']!r}")
        if cfg.get("graft_mtp") is True:
            fail("swap_config.json has drafter \"off\" and graft_mtp true; a graft exists to keep the drafter on")
    return cfg


def build_model_dir(base: Path, new: Path, config_dir: Path | None = None) -> tuple[Path, list[str], list[str], list[str]]:
    """Make model-dir from scratch. Returns (path, copied, linked, absent link names). The configuration files are
    COPIED from `config_dir` (every regular file in it, as the current bundles' prepare_model_dir.py does) when it is
    given, else the COPIED names from `base`. A name that new_snapshot also has is then replaced by a link to it and
    leaves `copied`."""
    md = HERE_DIR / "model-dir"
    if md.is_symlink():
        fail(f"{md} is a symlink; remove it so this script can build a real directory")
    if md.exists():
        shutil.rmtree(md)      # only ever a directory this script built: copies and symlinks
    md.mkdir()
    copied = []
    if config_dir is not None:
        for src in sorted(config_dir.iterdir()):
            if src.is_file():
                shutil.copyfile(src, md / src.name)
                copied.append(src.name)
    else:
        for name in COPIED:
            src = base / name
            if src.is_file():
                shutil.copyfile(src, md / name)        # follows the snapshot link to the blob
                copied.append(name)
    names = list(LINKED) + sorted(p.name for p in new.glob("*.safetensors"))
    linked, absent = [], []
    for name in names:
        src = new / name
        if not src.exists():
            absent.append(name)
            continue
        if os.path.lexists(md / name):      # the model's own file replaces a configuration copy of it
            os.unlink(md / name)
            copied.remove(name)
        os.symlink(os.path.realpath(src), md / name)
        linked.append(name)
    if FALLBACK_COPY in absent and FALLBACK_COPY not in copied and (base / FALLBACK_COPY).is_file():
        shutil.copyfile(base / FALLBACK_COPY, md / FALLBACK_COPY)      # follows the snapshot link to the blob
        absent.remove(FALLBACK_COPY)
        copied.append(FALLBACK_COPY)
    elif FALLBACK_COPY in absent and FALLBACK_COPY in copied:
        absent.remove(FALLBACK_COPY)           # the bundle's base_config/ already supplied one
    for name in REQUIRED_LINKS:
        if name not in linked:
            fail(f"{name} is not in new_snapshot {new}")
    if not any(n.endswith(".safetensors") for n in linked):
        fail(f"new_snapshot {new} has no *.safetensors file")
    return md, copied, linked, absent


REVISION = re.compile(r"[0-9a-f]{40}")


def stop_token_ids(model_dir: Path) -> list[int]:
    """The stop tokens the Qwen3.8 bundle's serving code finds in `model_dir`: the top-level integer or list of
    integers `eos_token_id` of generation_config.json, then of config.json, in that order, without repeats. The
    bundle reads only the top level (the nearest model's config.json nests it under text_config) and stops after
    loading the weights when it finds none. Another bundle may read these files differently."""
    ids: list[int] = []
    for name in ("generation_config.json", "config.json"):
        try:
            v = json.loads((model_dir / name).read_text(encoding="utf-8")).get("eos_token_id")
        except (OSError, ValueError, AttributeError):
            continue
        for x in ([v] if isinstance(v, int) and not isinstance(v, bool) else v if isinstance(v, list) else []):
            if isinstance(x, int) and not isinstance(x, bool) and x not in ids:
                ids.append(x)
    return ids
SOURCES_NAME = "model-dir-sources.json"
SPECIAL_TOKEN_NAMES = ("eos_token", "bos_token", "pad_token", "unk_token")


def tokenizer_config(d: Path) -> dict:
    try:
        cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def chat_template_of(d: Path) -> tuple[str | None, str | None]:
    """(file the template comes from, its sha256) in directory `d`: chat_template.jinja first, then the template inside
    tokenizer_config.json, the order transformers 5.19.0 uses. (None, None) when there is no template."""
    try:
        text = (d / "chat_template.jinja").read_text(encoding="utf-8", errors="replace")
        return "chat_template.jinja", hashlib.sha256(text.encode("utf-8")).hexdigest()
    except OSError:
        pass
    inline = tokenizer_config(d).get("chat_template")
    if isinstance(inline, list):            # named templates: [{"name": ..., "template": ...}]
        inline = json.dumps(inline, sort_keys=True, ensure_ascii=False)
    if isinstance(inline, str):
        return "tokenizer_config.json", hashlib.sha256(inline.encode("utf-8")).hexdigest()
    return None, None


def special_tokens_of(d: Path) -> dict:
    """eos, bos, pad and unk tokens as tokenizer_config.json writes them (a string, or an object with `content`)."""
    cfg, out = tokenizer_config(d), {}
    for name in SPECIAL_TOKEN_NAMES:
        v = cfg.get(name)
        if isinstance(v, dict):
            v = v.get("content")
        out[name] = v if isinstance(v, str) else None
    return out


def tokenizer_token_id(model_dir: Path, token) -> int | None:
    """The id tokenizer.json gives `token`: its added tokens first (special tokens live there), then the vocabulary.
    None when the token is unknown or tokenizer.json is missing, unreadable or of another shape."""
    if not isinstance(token, str):
        return None
    try:
        data = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    is_id = lambda v: isinstance(v, int) and not isinstance(v, bool)     # noqa: E731
    added = data.get("added_tokens")
    for item in added if isinstance(added, list) else []:
        if isinstance(item, dict) and item.get("content") == token:
            return item["id"] if is_id(item.get("id")) else None
    model = data.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        return vocab[token] if is_id(vocab.get(token)) else None
    if isinstance(vocab, list):
        for i, entry in enumerate(vocab):
            if (entry[0] if isinstance(entry, list) and entry else entry) == token:
                return i
    return None


def chat_facts(model_dir: Path, base: Path) -> dict:
    """What the served model directory holds for chat requests, set against the nearest model's snapshot: the template
    (read the way the loader reads it) and the special tokens. Nothing here is sent to a server."""
    file, digest = chat_template_of(model_dir)
    _, base_digest = chat_template_of(base)
    served, theirs = special_tokens_of(model_dir), special_tokens_of(base)
    return {"chat_template": {"file": file, "sha256": digest, "nearest_sha256": base_digest,
                              "same_as_nearest": None if digest is None or base_digest is None else digest == base_digest},
            "special_tokens": served,
            "special_tokens_differ_from_nearest": sorted(k for k in SPECIAL_TOKEN_NAMES if served[k] != theirs[k])}


def snapshot_revision(snapshot: Path) -> str | None:
    """The snapshot directory's name when it is a 40-character commit, else None."""
    return snapshot.name if REVISION.fullmatch(snapshot.name) else None


CONFIG_FROM_BUNDLE = "bundle base_config/"
CONFIG_FROM_SNAPSHOT = "nearest model's snapshot"


def sources_record(base: Path, new: Path, nearest: str, copied: list[str], linked: list[str],
                   absent: list[str], eos: list[int], config_from: str = CONFIG_FROM_SNAPSHOT) -> dict:
    """Where each file of model-dir came from. Weights are one entry with a count."""
    files = []
    for name in copied:
        entry = {"name": name, "source": "base", "how": "copy"}
        if name == FALLBACK_COPY:
            entry["reason"] = "new_snapshot has none"
        files.append(entry)
    files += [{"name": n, "source": "model", "how": "link"} for n in linked if not n.endswith(".safetensors")]
    files.sort(key=lambda f: f["name"])
    if FALLBACK_COPY in copied:
        sampling = "base"
    elif FALLBACK_COPY in linked:
        sampling = "model"
    else:
        sampling = "none"
    return {"schema": 1, "base_model": nearest, "base_revision": snapshot_revision(base),
            "model_revision": snapshot_revision(new), "config_from": config_from, "files": files,
            "weights": {"source": "model", "how": "link", "count": sum(n.endswith(".safetensors") for n in linked)},
            "absent": sorted(absent), "sampling_defaults_from": sampling, "eos_token_ids": eos}


def _vocab_in(cfg) -> int | None:
    """The text vocabulary size a config.json states: text_config's, else the top level's. A whole number above zero only."""
    if not isinstance(cfg, dict):
        return None
    text = cfg.get("text_config")
    for holder in ((text,) if isinstance(text, dict) and "vocab_size" in text else ()) + (cfg,):
        v = holder.get("vocab_size")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
        if "vocab_size" in holder:
            return None
    return None


def vocab_facts(base: Path, new: Path) -> dict:
    """{"model", "base", "trimmed"}: both vocabulary sizes and whether they differ (None when either is unknown).
    A model whose vocabulary differs from the nearest model's (ProprietaryLegal/Qwen3.8-27B-CJK-Trim: 181760 against 248320)
    needs its own size in the served config.json and the speculative drafter off (issue 56)."""
    sizes = {}
    for key, snap in (("model", new), ("base", base)):
        try:
            sizes[key] = _vocab_in(json.loads((snap / "config.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            sizes[key] = None
    trimmed = None if sizes["model"] is None or sizes["base"] is None else sizes["model"] != sizes["base"]
    return {"model": sizes["model"], "base": sizes["base"], "trimmed": trimmed}


def write_vocab(model_dir: Path, size: int) -> None:
    """Set vocab_size to the model's own wherever the copied (base) config.json has the key."""
    path = model_dir / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    holders = [cfg] + ([cfg["text_config"]] if isinstance(cfg.get("text_config"), dict) else [])
    for h in holders:
        if "vocab_size" in h:
            h["vocab_size"] = size
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


MTP_NAME = re.compile(r"(^|\.)mtp\.")


def shard_tensor_names(path: Path) -> list[str] | None:
    """Tensor names from a safetensors header, or None when the file is not readable as one."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            (n,) = struct.unpack("<Q", raw)
            if not 0 < n <= 100 * 1024 * 1024:
                return None
            header = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return [k for k in header if k != "__metadata__"] if isinstance(header, dict) else None


def has_mtp_tensors(new: Path) -> bool | None:
    """Does the new model hold `mtp.*` tensors? Read from its weight index, else from the shard headers.
    None when neither can be read, so the caller can leave things alone."""
    try:
        weight_map = json.loads((new / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        if isinstance(weight_map, dict):
            return any(MTP_NAME.search(k) for k in weight_map)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    shards = sorted(new.glob("*.safetensors"))
    if not shards:
        return None
    names = []
    for shard in shards:
        got = shard_tensor_names(shard)
        if got is None:
            return None
        names += got
    return any(MTP_NAME.search(k) for k in names)


# A DFlash drafter (QWEN36_DRAFTER="dflash2") is a separate checkpoint (DFLASH_WEIGHTS) that reads the target model's hidden states at fixed
# layers; it was trained on the nearest model. These text config values decide whether the new model's hidden states are of that kind.
DRAFTER_TARGET_KEYS = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                       "intermediate_size", "vocab_size", "layer_types")
DRAFTER_LINE = re.compile(r"^[ \t]*export[ \t]+QWEN36_DRAFTER=(.*)$", re.MULTILINE)


def bundle_drafter(text: str) -> str | None:
    """The drafter the bundle's run.sh turns on: the value of its last `export QWEN36_DRAFTER=` line, unquoted ("" is off), or None when
    it has no such line."""
    found = [m.group(1).strip().strip("\"'") for m in DRAFTER_LINE.finditer(text)]
    return found[-1] if found else None


def _text_config(path: Path):
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))  # hardening: ok a snapshot config, read as vocab_facts reads it (a stand-alone template)
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    inner = cfg.get("text_config")
    return {**cfg, **inner} if isinstance(inner, dict) else cfg


def _target_value(cfg: dict, key: str):
    v = cfg.get(key)
    if key == "model_type" and isinstance(v, str) and v.endswith("_text"):
        return v[:-len("_text")]                 # qwen3_5 and qwen3_5_text are the same text model
    return v


def drafter_target_differences(base: Path, new: Path) -> list[str] | None:
    """The DRAFTER_TARGET_KEYS whose values differ between the new model's text config and the nearest model's, or None when either
    config cannot be read (then the drafter cannot be shown to fit). A key only one config states is not compared."""
    a, b = _text_config(new / "config.json"), _text_config(base / "config.json")
    if a is None or b is None:
        return None
    return [k for k in DRAFTER_TARGET_KEYS
            if _target_value(a, k) is not None and _target_value(b, k) is not None and _target_value(a, k) != _target_value(b, k)]


def read_shard_header(path: Path) -> tuple[dict, int] | None:
    """(the safetensors header, the file offset where tensor data starts), or None when it cannot be read."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            (n,) = struct.unpack("<Q", raw)
            if not 0 < n <= 100 * 1024 * 1024:
                return None
            header = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return (header, 8 + n) if isinstance(header, dict) else None


def base_mtp_files(base: Path) -> dict[str, str]:
    """{tensor name: shard file name} for the nearest model's `mtp.*` tensors, from its weight index, else from
    its shard headers. Empty when it has none or cannot be read."""
    try:
        weight_map = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        if isinstance(weight_map, dict):
            return {k: v for k, v in weight_map.items() if MTP_NAME.search(k)}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    out = {}
    for shard in sorted(base.glob("*.safetensors")):
        got = shard_tensor_names(shard)
        out.update({k: shard.name for k in got or [] if MTP_NAME.search(k)})
    return out


def model_hidden_size(new: Path) -> int | None:
    try:
        cfg = json.loads((new / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for holder in (cfg.get("text_config"), cfg):
        v = holder.get("hidden_size") if isinstance(holder, dict) else None
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def plan_mtp_graft(base: Path, new: Path) -> dict:
    """Check that the nearest model's `mtp.*` tensors can be grafted onto the new model, and say where they are.
    Exits 2 with the reason when they cannot. Nothing is written here."""
    if os.path.lexists(new / GRAFT_SHARD):
        # The name is reserved. model-dir links every *.safetensors of the new model, so a file with this name would be
        # replaced by the graft and its own tensors would be missing; the package's launcher would write through the link.
        fail(f"the new model ships a file named {GRAFT_SHARD}, the name the graft uses for its shard; the graft would "
             "replace it, so this model cannot be grafted")
    vocab = vocab_facts(base, new)
    if vocab["trimmed"]:
        fail(f"graft_mtp cannot be used: the new model's vocabulary ({vocab['model']}) differs from the nearest model's "
             f"({vocab['base']}), and the speculative drafter works on the nearest model's token ids")
    has = has_mtp_tensors(new)
    if has is True:
        fail("graft_mtp is set, but the new model already has mtp.* tensors; there is nothing to graft")
    if has is None:
        fail("graft_mtp is set, but I cannot tell whether the new model has mtp.* tensors (no readable weight "
             "index or shard headers), so I will not add a second head")
    where = base_mtp_files(base)
    if not where:
        fail("graft_mtp is set, but the nearest model has no mtp.* tensors to graft")
    files = sorted(set(where.values()))
    if len(files) > 1:
        fail(f"the nearest model keeps its mtp.* tensors in more than one shard ({', '.join(files)}); the graft "
             "reads from one")
    src = base / files[0]
    parsed = read_shard_header(src)
    if parsed is None:
        fail(f"cannot read the safetensors header of {files[0]} in the nearest model")
    header, start = parsed
    size = src.stat().st_size
    entries = {}
    for name in sorted(where):
        e = header.get(name)
        if not isinstance(e, dict):
            fail(f"{name} is in the nearest model's index but not in {files[0]}")
        if e.get("dtype") != "BF16":
            fail(f"{name} is {e.get('dtype')}; the graft copies BF16 tensors only")
        try:
            a, b = e["data_offsets"]
            elements = 1
            for d in e["shape"]:
                elements *= d
        except (KeyError, TypeError, ValueError):
            fail(f"{name} has a header entry the graft cannot read")
        if not (0 <= a <= b and start + b <= size and b - a == 2 * elements):
            fail(f"{name}'s header entry does not match the bytes in {files[0]}")
        entries[name] = e
    hidden, norm = model_hidden_size(new), next((e for k, e in entries.items() if k.endswith("mtp.norm.weight")), None)
    if hidden is not None and norm is not None and norm["shape"] != [hidden]:
        fail(f"mtp.norm.weight has shape {norm['shape']}, but the new model's hidden size is {hidden}; "
             "this head is not for this model")
    return {"file": files[0], "path": src, "start": start, "entries": entries}


def write_graft_shard(plan: dict, dest: Path) -> tuple[str, int]:
    """Write the grafted shard (the tensors in name order, bytes copied as they are). Returns (sha256, byte count)."""
    if os.path.lexists(dest):
        fail(f"{dest} already exists; the graft shard is only ever written to a new file, never through a link")
    header, off = {"__metadata__": {"format": "pt"}}, 0
    for name, e in sorted(plan["entries"].items()):
        n = e["data_offsets"][1] - e["data_offsets"][0]
        header[name] = {"dtype": "BF16", "shape": e["shape"], "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    digest, tmp = hashlib.sha256(), dest.with_name(dest.name + ".tmp")
    with open(plan["path"], "rb") as src, open(tmp, "wb") as out:
        for chunk in (struct.pack("<Q", len(raw)), raw):
            out.write(chunk)
            digest.update(chunk)
        for name, e in sorted(plan["entries"].items()):
            a, b = e["data_offsets"]
            src.seek(plan["start"] + a)
            left = b - a
            while left:
                chunk = src.read(min(left, 1 << 20))
                if not chunk:
                    fail(f"{plan['file']} ended while copying {name}")
                out.write(chunk)
                digest.update(chunk)
                left -= len(chunk)
    os.replace(tmp, dest)
    return digest.hexdigest(), 8 + len(raw) + off


def graft_index(model_dir: Path, names: list[str], total: int) -> None:
    """Replace the model dir's link to the new model's weight index with a real file that also lists `names`."""
    path = model_dir / "model.safetensors.index.json"
    if not os.path.lexists(path):
        return                  # no index: the loader reads every *.safetensors file in the directory
    try:
        data = json.loads(path.read_text(encoding="utf-8"))      # follows the link
        weight_map = data["weight_map"]
    except (OSError, ValueError, KeyError, TypeError):
        fail("the new model's model.safetensors.index.json cannot be read, so the graft cannot add to it")
    if not isinstance(weight_map, dict):
        fail("the new model's model.safetensors.index.json has no weight_map object")
    weight_map.update({n: GRAFT_SHARD for n in names})
    meta = data.setdefault("metadata", {})
    if isinstance(meta, dict) and isinstance(meta.get("total_size"), int):
        meta["total_size"] += total
    os.unlink(path)             # the link, not the cached file it points to
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


ADDITIONAL = re.compile(r"--additional-config\s+'([^']*)'")


def drop_device_sampling(text: str, bundle: Path) -> tuple[str, str]:
    """Remove "sample_on_device_mode" from the single --additional-config of the command. With the drafter
    on, the DFlash path samples on its own. With it off, the model code asks for on-device sampling and
    refuses it unless the mesh is 1x4, 1x8 or 1x1 ("Unset sample_on_device_mode for host sampling"). Returns
    (new text, a note). A command without the option or the key, or with a config that is not JSON, is left
    alone and the note says so."""
    found = list(ADDITIONAL.finditer(text))
    if not found:
        return text, "--additional-config was absent; nothing to change for sampling"
    if len(found) > 1:
        fail(f"expected at most one --additional-config in {bundle / 'run.sh'}, found {len(found)}")
    m = found[0]
    try:
        cfg = json.loads(m.group(1))
    except ValueError:
        return text, "--additional-config is not valid JSON, so sample_on_device_mode was not removed"
    tt = cfg.get("tt") if isinstance(cfg, dict) else None
    if not isinstance(tt, dict) or "sample_on_device_mode" not in tt:
        return text, "sample_on_device_mode was absent from --additional-config; nothing to remove"
    del tt["sample_on_device_mode"]
    return (text[:m.start()] + f"--additional-config '{json.dumps(cfg)}'" + text[m.end():],
            'removed "sample_on_device_mode" from --additional-config (the drafter is off, so sampling '
            "moves to the host; on-device sampling is certified only on 1x4, 1x8 and 1x1 meshes)")


AUX_LINE = re.compile(r"^([ \t]*)export[ \t]+TT_AUXILIARY_WEIGHTS=(.*)$", re.MULTILINE)
DFLASH_LINE = re.compile(r"^[ \t]*export[ \t]+DFLASH_WEIGHTS=(.*)$", re.MULTILINE)


def drop_drafter_weights(text: str, bundle: Path) -> tuple[str, str]:
    """With the drafter off, remove the drafter's repo from `export TT_AUXILIARY_WEIGHTS=` (a comma-separated list the bundle's launcher
    downloads before vLLM starts). The drafter's repo is the one `export DFLASH_WEIGHTS=` names, at any revision; other entries stay, and a
    line left empty is removed. Returns (new text, a note). More than one TT_AUXILIARY_WEIGHTS line exits 2."""
    aux = list(AUX_LINE.finditer(text))
    if not aux:
        return text, "TT_AUXILIARY_WEIGHTS was absent; nothing to remove"
    if len(aux) > 1:
        fail(f"expected at most one export TT_AUXILIARY_WEIGHTS= line in {bundle / 'run.sh'}, found {len(aux)}")
    drafters = {m.group(1).strip().strip("\"'").split("@", 1)[0] for m in DFLASH_LINE.finditer(text)} - {""}
    if not drafters:
        return text, ("TT_AUXILIARY_WEIGHTS was left as it is: run.sh has no export DFLASH_WEIGHTS= line, so the drafter's entry cannot be "
                      "told apart")
    m = aux[0]
    items = [x.strip() for x in m.group(2).strip().strip("\"'").split(",") if x.strip()]
    keep = [x for x in items if x.split("@", 1)[0] not in drafters]
    gone = [x for x in items if x not in keep]
    if not gone:
        return text, "TT_AUXILIARY_WEIGHTS names no drafter repo; left as it is"
    new_line = f'{m.group(1)}export TT_AUXILIARY_WEIGHTS="{",".join(keep)}"' if keep else ""
    end = m.end() + (1 if not keep and text[m.end():m.end() + 1] == "\n" else 0)
    return (text[:m.start()] + new_line + text[end:],
            f"removed the drafter's {', '.join(gone)} from TT_AUXILIARY_WEIGHTS (the drafter is off, so it is not downloaded)")


# The two forms of bundle run.sh. See the module docstring.
OLDER_FORM, CURRENT_FORM = "older", "current"
HERE_MODEL = r'"?\$(?:\{HERE\}|HERE)/model-dir"?'                         # "$HERE/model-dir", "${HERE}/model-dir", $HERE/model-dir
CURRENT_MODEL_RE = r"--model\s+" + HERE_MODEL + r"(?=\s|$)"
PREPARE_LINE_RE = r'^[ \t]*"?\$PYBIN"?[ \t]+"?\$(?:\{HERE\}|HERE)/prepare_model_dir\.py"?[ \t]*$'
PREPARE_REPLACEMENT = ("# prepare_swap.py built model-dir. The bundle's prepare_model_dir.py is not run here: it would rebuild "
                       "the bundle's own model-dir with the nearest model's weights.")


def older_model_re(nearest: str) -> str:
    return r'--model\s+(?:"' + re.escape(nearest) + r'"|' + re.escape(nearest) + r')(?=\s|$)'


def run_script_form(text: str, nearest: str, where: Path) -> str:
    """OLDER_FORM or CURRENT_FORM, by whichever `--model` argument matches. Exits 2 when none or more than one does."""
    n_old = len(re.findall(older_model_re(nearest), text))
    n_cur = len(re.findall(CURRENT_MODEL_RE, text))
    if n_old + n_cur == 1:
        # Any other --model option (either spelling) could override the one this script rewrites when the server parses the command.
        every = re.findall(r"(?<![\w-])--model(?:=\S*|\s+\S+)", "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#")))
        if len(every) != 1:
            fail(f"{where} has {len(every)} --model options and the swap rewrites one; an additional --model option "
                 f"({', '.join(every)}) could override the model directory. Remove it or edit the script by hand")
        return OLDER_FORM if n_old else CURRENT_FORM
    if n_old and n_cur:
        fail(f'{where} has --model "{nearest}" ({n_old}) and --model "$HERE/model-dir" ({n_cur}), so it is in both forms; '
             "expected one --model argument")
    found = f"found {n_old + n_cur}" if n_old + n_cur else "found 0"
    fail(f'expected exactly one --model argument in {where}, either --model "{nearest}" (the older form) or '
         f'--model "$HERE/model-dir" (the current form, which builds model-dir with prepare_model_dir.py); {found}')


def edit_run_script(text: str, bundle: Path, nearest: str, model_dir: Path,
                    drafter_off: bool = False, explicit_drafter_off: bool = False,
                    drafter_reason: str = "the new model has no mtp.* tensors, so the speculative drafter "
                                          "that needs the MTP head is switched off",
                    form: str | None = None) -> tuple[str, list[str]]:
    """Apply the edits. Returns (new text, notes about optional edits that did not apply). `form` is run_script_form's
    answer; it is worked out here when not given."""
    where = bundle / "run.sh"
    form = form or run_script_form(text, nearest, where)
    text, n = re.subn(r"^HERE=.*$", lambda m: f'HERE="{bundle}"', text, flags=re.MULTILINE)
    if n != 1:
        fail(f"expected exactly one HERE= line in {where}, found {n}")
    notes = []
    if form == OLDER_FORM:
        text, n = re.subn(older_model_re(nearest), lambda m: f"--model {shlex.quote(str(model_dir))}", text)
    else:
        text, n = re.subn(CURRENT_MODEL_RE, lambda m: f"--model {shlex.quote(str(model_dir))}", text)
        text, p = re.subn(PREPARE_LINE_RE, lambda m: PREPARE_REPLACEMENT, text, flags=re.MULTILINE)
        if p != 1:
            fail(f'expected exactly one line that runs "$HERE/prepare_model_dir.py" in {where}, found {p}. A current-form '
                 "bundle rebuilds its own model-dir with it at each start, and the copy must not")
        notes.append("the line that runs prepare_model_dir.py was replaced by a comment (the bundle's own model-dir is not rebuilt)")
    if n != 1:
        fail(f'expected exactly one --model argument in {where}, found {n}')
    text, n = re.subn(r"^([ \t]*)export[ \t]+HF_MODEL=.*$",
                      lambda m: f'{m.group(1)}export HF_MODEL="{model_dir}"', text, flags=re.MULTILINE)
    if n == 0:
        notes.append("export HF_MODEL=... was absent; nothing to rewrite (serve_and_compare.py sets "
                     "HF_MODEL and MODEL_WEIGHTS_DIR itself)")
    else:
        notes.append(f'export HF_MODEL="{model_dir}" ({n} line(s) rewritten)')
    # The TT runtime reads MODEL_WEIGHTS_DIR before HF_MODEL, and an export in run.sh overrides what serve_and_compare.py puts in the environment.
    text, n = re.subn(r"^([ \t]*)export[ \t]+MODEL_WEIGHTS_DIR=.*$",
                      lambda m: f'{m.group(1)}export MODEL_WEIGHTS_DIR="{model_dir}"', text, flags=re.MULTILINE)
    if n:
        notes.append(f'export MODEL_WEIGHTS_DIR="{model_dir}" ({n} line(s) rewritten)')
    if form == CURRENT_FORM and re.search(HERE_MODEL, text):
        fail(f"{where} mentions the bundle's own $HERE/model-dir somewhere other than the --model argument, "
             "HF_MODEL and MODEL_WEIGHTS_DIR; a copy would still read the nearest model's weights from it")
    if drafter_off:
        text, n = re.subn(r"^([ \t]*)export[ \t]+QWEN36_DRAFTER=.*$",
                          lambda m: f'{m.group(1)}export QWEN36_DRAFTER=""', text, flags=re.MULTILINE)
        if n > 1:
            fail(f"expected at most one export QWEN36_DRAFTER= line in {bundle / 'run.sh'}, found {n}")
        if n == 0 and explicit_drafter_off:
            # The gate and the card read the serving record from this line; "not set" is not "off". Put the line where the command is built.
            text, placed = re.subn(r"^([ \t]*)CMD=\(", lambda m: f'{m.group(1)}export QWEN36_DRAFTER=""\n{m.group(0)}', text, count=1,
                                   flags=re.MULTILINE)
            if placed != 1:
                fail(f"{bundle / 'run.sh'} has no export QWEN36_DRAFTER= line and no CMD=( line to put one before, so the drafter "
                     "cannot be recorded as off")
            n = 1
        notes.append(f'export QWEN36_DRAFTER="" ({drafter_reason})' if n else
                     "export QWEN36_DRAFTER=... was absent; nothing to clear")
        text, note = drop_device_sampling(text, bundle)
        notes.append(note)
        text, note = drop_drafter_weights(text, bundle)
        notes.append(note)
    for flag in ("--revision", "--tokenizer-revision"):
        # The leading space keeps " --revision" from matching inside "--tokenizer-revision".
        text, n = re.subn(r" " + flag + r'\s+"?[0-9a-f]{40}"?(?=\s|$)', "", text)
        if n > 1:
            fail(f"expected at most one {flag} <40 hex> in {bundle / 'run.sh'}, found {n}")
        if n == 0:
            notes.append(f"{flag} <40 hex> was absent; nothing to delete")
    return text, notes


def main() -> int:
    cfg = load_config()
    base, new = Path(cfg["base_snapshot"]), Path(cfg["new_snapshot"])
    bundle = Path(cfg["bundle_dir"]).resolve() if cfg.get("bundle_dir") else None
    for label, d in (("bundle_dir", bundle), ("base_snapshot", base), ("new_snapshot", new)):
        if d is not None and not d.is_dir():
            fail(f"{label} {d} is not a directory")
    model_dir = HERE_DIR / "model-dir"
    graft = cfg.get("graft_mtp") is True
    plan = plan_mtp_graft(base, new) if graft else None
    config_dir, form = None, OLDER_FORM
    if bundle is not None:
        src_run = bundle / "run.sh"
        if not src_run.is_file():
            fail(f"{src_run} does not exist")
        form = run_script_form(src_run.read_text(encoding="utf-8"), cfg["nearest_model_id"], src_run)
        if form == CURRENT_FORM:
            config_dir = bundle / "base_config"
            if not (config_dir / "config.json").is_file():
                fail(f"{src_run} is a current-form run.sh, which builds model-dir from the bundle's base_config/, and "
                     f"{config_dir / 'config.json'} does not exist")
    # The vocabulary is compared with the configuration that is served: the bundle's base_config/ in the current form.
    vocab = vocab_facts(config_dir or base, new)
    if bundle is not None:
        # Edit in memory first, so a failed edit leaves no run.sh and no half-built model-dir.
        mtp = has_mtp_tensors(new)
        no_drafter_why = None
        if vocab["trimmed"]:
            no_drafter_why = (f"the new model's vocabulary ({vocab['model']}) differs from the nearest model's ({vocab['base']}), "
                              "and the speculative drafter works on the nearest model's token ids")
        kind = bundle_drafter(src_run.read_text(encoding="utf-8"))  # hardening: ok the bundle's run.sh, which this script reads to edit
        target_off = False
        if kind is not None and kind.startswith("dflash") and not graft and not no_drafter_why:
            differ = drafter_target_differences(base, new)
            if differ is None or differ:
                target_off = True
                no_drafter_why = ("the bundle's DFlash2 drafter reads the hidden states of the model it was trained on (the nearest model's), and "
                                  + ("the two configs cannot be compared" if differ is None else
                                     f"the new model's {', '.join(differ)} differ from the nearest model's")
                                  + ", so it is switched off whatever mtp.* tensors the new model has")
        forced_off = cfg.get("drafter") == "off"
        if forced_off and not no_drafter_why:
            no_drafter_why = ("swap_config.json says drafter off: this directory serves the way a model without mtp.* "
                              "tensors is served, so a control can be compared with such a run")
        text, notes = edit_run_script(src_run.read_text(encoding="utf-8"), bundle,
                                      cfg["nearest_model_id"], model_dir,
                                      drafter_off=((mtp is False or bool(vocab["trimmed"]) or forced_off or target_off) and not graft),
                                      explicit_drafter_off=bool(vocab["trimmed"]) or forced_off,
                                      form=form, **({"drafter_reason": no_drafter_why} if no_drafter_why else {}))
        if form == CURRENT_FORM and (bundle / "graft.json").is_file() and not graft:
            notes.append("the bundle's graft.json is not applied (its prepare_model_dir.py is not run); the drafter is set by "
                         "the rules above, and graft_mtp true in swap_config.json grafts the nearest model's mtp.* tensors")
        if graft:
            notes.append("the drafter and on-device sampling stay on: the nearest model's mtp.* tensors are "
                         "grafted into model-dir")
        elif mtp is None:
            notes.append("could not tell whether the new model has mtp.* tensors (no readable weight index or "
                         "shard headers), so the drafter setting is unchanged")
    model_dir, copied, linked, absent = build_model_dir(base, new, config_dir)
    if vocab["trimmed"] and "config.json" in copied:
        write_vocab(model_dir, vocab["model"])
    grafted = None
    if plan is not None:
        digest, total = write_graft_shard(plan, model_dir / GRAFT_SHARD)
        graft_index(model_dir, sorted(plan["entries"]), total)
        grafted = {"from": cfg["nearest_model_id"], "base_revision": snapshot_revision(base), "file": GRAFT_SHARD,
                   "source_file": plan["file"], "tensors": len(plan["entries"]), "bytes": total, "sha256": digest}
    eos = stop_token_ids(model_dir)
    record = sources_record(base, new, cfg["nearest_model_id"], copied, linked, absent, eos,
                            CONFIG_FROM_BUNDLE if config_dir else CONFIG_FROM_SNAPSHOT)
    if grafted:
        record["mtp_graft"] = grafted
        record["files"].append({"name": GRAFT_SHARD, "source": "base", "how": "grafted",
                                "reason": "the new model has no mtp.* tensors; these are the nearest model's"})
        record["files"].sort(key=lambda f: f["name"])
    record["vocab"] = vocab
    record.update(chat_facts(model_dir, base))
    tok_id = tokenizer_token_id(model_dir, record["special_tokens"]["eos_token"])
    record["tokenizer_eos_token_id"] = tok_id
    record["tokenizer_eos_served"] = None if tok_id is None or not eos else tok_id in eos
    (HERE_DIR / SOURCES_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"model-dir: {model_dir}")
    print(f"  copied from {'the bundle base_config/' if config_dir else 'base_snapshot'}: {', '.join(copied) or 'nothing'}")
    print(f"  linked from new_snapshot: {len(linked)} files ({', '.join(linked)})")
    if grafted:
        print(f"  grafted {grafted['tensors']} mtp.* tensors of {grafted['from']} as {GRAFT_SHARD} "
              f"(sha256 {digest[:12]}); the drafter's acceptance on the new model is not measured by this script")
    if vocab["trimmed"]:
        print(f"  config.json: vocab_size set to the new model's {vocab['model']} (the nearest model's is {vocab['base']}); "
              "the speculative drafter is switched off because it works on the nearest model's token ids")
    if FALLBACK_COPY in copied:
        origin = "the bundle's base_config/" if config_dir and (config_dir / FALLBACK_COPY).is_file() else "base_snapshot"
        print(f"  {FALLBACK_COPY}: new_snapshot has none; copied from {origin}")
    origin = {"model": "the new model's", "base": "the nearest model's", "none": None}[record["sampling_defaults_from"]]
    print(f"  {SOURCES_NAME}: " + (f"sampling defaults and stop tokens are {origin} generation_config.json"
                                   if origin else "no generation_config.json, so no sampling defaults or stop tokens"))
    ct = record["chat_template"]
    if ct["file"] is None:
        print("  chat template: the model directory has none" + (", and the nearest model has one" if ct["nearest_sha256"] else ""))
    else:
        same = {True: ", the same as the nearest model's", False: ", different from the nearest model's",
                None: ", and the nearest model has none"}[ct["same_as_nearest"]]
        print(f"  chat template: {ct['file']} (sha256 {ct['sha256'][:12]}){same}")
    if absent:
        print(f"  absent in new_snapshot, not linked: {', '.join(absent)}")
    if not eos and FALLBACK_COPY not in absent:
        print("  warning: no stop token (a top-level eos_token_id) in the model directory's generation_config.json or "
              "config.json. The Qwen3.8 bundle's serving code looks there and stops after loading the weights when it "
              "finds none, so the server will probably not start")
    if record["tokenizer_eos_served"] is False:
        print(f"  warning: the tokenizer's eos token {record['special_tokens']['eos_token']} (id {tok_id}) is not among "
              f"the stop tokens {eos} the server will stop on, so a reply that ends on it runs to the length limit. "
              "Check the model card for the stop tokens it recommends")
    if FALLBACK_COPY in absent:
        print(f"  warning: {FALLBACK_COPY} is in neither snapshot. The bundle's serving code reads eos_token_id from it or "
              "from config.json (here the base model's, which may nest it), so the server may not start")
    if bundle is None:
        print(f"run.sh: not built; {cfg['package']} is a container package, and "
              "serve_and_compare_container.py starts it")
        return 0
    run_out = HERE_DIR / "run.sh"
    run_out.write_text(text, encoding="utf-8")
    run_out.chmod(run_out.stat().st_mode | 0o111)
    print(f"run.sh: {run_out}")
    print(f'  HERE="{bundle}"; --model {model_dir}')
    for note in notes:
        print(f"  {note}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
