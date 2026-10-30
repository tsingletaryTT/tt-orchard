# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Is the served chat template, special tokens and stop ids the new model's, and did the chat replies start and stop like its chat format? (issue 17)

Stage 0 already compares the new model's chat template and special tokens with the nearest model's (`delta_triage.py`, hazard area
`chat_template`) and checks the stop tokens it predicts the server will use. This module checks what stage 2 and stage 4 actually served.
It is a pure check of three things already on disk, and it sends nothing to a server:

  * `check_files(served_dir, new_snapshot)`: the template, the eos, bos, pad and unk tokens and the stop ids of the served model directory
    against the new model's snapshot. The template is read the way the loader reads it: `chat_template.jinja` first, then the template
    inside `tokenizer_config.json` (a string, or a list of named templates of which `default` or else the first is used). A blank template
    is none. Files are read through their links (a served directory links its tokenizer files into the snapshot).
  * `check_replies(prompt_suite_record)`: the chat replies the swap check recorded. A reply that begins with a role header or a special
    token, one that holds an end token, and a run in which no reply ever stopped on its own (all hit the length limit) are problems.
  * `build_record` / `write_record` join the two into `served-chat.json` in the stage directory.

What it does not check, and the record says so: whether a reply begins and ends as the model card describes (a card is prose), and
whether a model fine-tuned on a different template answers worse. `ok` is true only when there is no problem and at least one chat reply
was checked.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from orchard import safeio

FILE = "served-chat.json"
MAX_JSON_BYTES = 64 << 20            # a tokenizer.json is tens of megabytes
MAX_TEMPLATE_BYTES = 1 << 20
TOKEN_NAMES = ("eos_token", "bos_token", "pad_token", "unk_token")
LABEL = "check of the served files and the recorded chat replies; this check sent no request to a server"
NOT_CHECKED = ("whether a reply begins and ends as the model's card describes (the card is prose); whether a model trained on a different "
               "template answers worse; the card of the model was not read")
# A role header at the start of a reply: the word `assistant` followed by a colon or a line end, with an optional start-of-turn token before it.
ROLE_HEADER = re.compile(r"\A\s*(?:<\|im_start\|>\s*)?[Aa]ssistant[ \t]*(?::|\n)")
END_MARKER = re.compile(r"end|eos|</s>", re.IGNORECASE)


def _real(path) -> str:
    # a served directory links its tokenizer files into the snapshot, and O_NOFOLLOW would refuse a link, so the path is resolved first
    return os.path.realpath(path)


def _json(path, notes: list, what: str):
    """The JSON object at `path`, or None with a note. Never raises."""
    try:
        return safeio.read_json(_real(path), max_bytes=MAX_JSON_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        if "ENOENT" not in str(exc.reason):            # a file the model does not ship is normal and needs no note
            notes.append(f"{what} could not be read: {exc.reason}")
        return None


def _text(path) -> str | None:
    try:
        return safeio.read_bytes(_real(path), max_bytes=MAX_TEMPLATE_BYTES).decode("utf-8", errors="replace")
    except safeio.SafeReadError:
        return None


def _inline(value) -> str | None:
    if isinstance(value, list):
        named = {t.get("name"): t.get("template") for t in value if isinstance(t, dict)}
        value = named.get("default") or next((t for t in named.values() if isinstance(t, str)), None)
    return value if isinstance(value, str) and value.strip() else None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_of(directory, notes: list | None = None) -> tuple[str | None, str | None]:
    """(file the template comes from, its sha256) in `directory`, the loader's way; (None, None) when there is none."""
    notes = [] if notes is None else notes
    d = Path(directory)
    jinja = _text(d / "chat_template.jinja")
    cfg = _json(d / "tokenizer_config.json", notes, f"{d.name}/tokenizer_config.json")
    inline = _inline(cfg.get("chat_template")) if cfg else None
    if jinja is not None and jinja.strip():
        if inline is not None and _sha(inline) != _sha(jinja):
            notes.append(f"{d.name}: the inline chat template in tokenizer_config.json differs from chat_template.jinja; the loader uses the file")
        return "chat_template.jinja", _sha(jinja)
    if inline is not None:
        return "tokenizer_config.json", _sha(inline)
    return None, None


def tokens_of(directory, notes: list) -> dict:
    cfg = _json(Path(directory) / "tokenizer_config.json", notes, f"{Path(directory).name}/tokenizer_config.json") or {}
    out = {}
    for name in TOKEN_NAMES:
        v = cfg.get(name)
        if isinstance(v, dict):
            v = v.get("content")
        out[name] = v if isinstance(v, str) else None
    return out


def _ids(v) -> list[int]:
    items = [v] if isinstance(v, int) and not isinstance(v, bool) else v if isinstance(v, list) else []
    return [x for x in items if isinstance(x, int) and not isinstance(x, bool)]


def stop_ids_of(directory, notes: list) -> list[int]:
    """The ids the bundle's serving code stops on: the top-level `eos_token_id` of generation_config.json, then of config.json (a value nested
    under text_config does not count), without repeats."""
    out: list[int] = []
    for name in ("generation_config.json", "config.json"):
        data = _json(Path(directory) / name, notes, f"{Path(directory).name}/{name}") or {}
        for x in _ids(data.get("eos_token_id")):
            if x not in out:
                out.append(x)
    return out


def token_id(directory, token, notes: list) -> int | None:
    """The id tokenizer.json gives `token`: added tokens first, then the vocabulary. None when unknown."""
    if not isinstance(token, str):
        return None
    data = _json(Path(directory) / "tokenizer.json", notes, f"{Path(directory).name}/tokenizer.json")
    if not data:
        return None
    for item in data.get("added_tokens") if isinstance(data.get("added_tokens"), list) else []:
        if isinstance(item, dict) and item.get("content") == token:
            return item["id"] if isinstance(item.get("id"), int) and not isinstance(item.get("id"), bool) else None
    model = data.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    v = vocab.get(token) if isinstance(vocab, dict) else None
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def check_files(served_dir, new_snapshot) -> dict:
    """The template, special tokens and stop ids of `served_dir` against the new model's `new_snapshot`. Never raises."""
    notes: list[str] = []
    problems: list[dict] = []
    s_file, s_sha = template_of(served_dir, notes)
    n_file, n_sha = template_of(new_snapshot, notes)
    if n_sha and not s_sha:
        problems.append({"name": "template_missing_in_served", "detail": "the new model ships a chat template and the served directory has none"})
    elif s_sha and not n_sha:
        problems.append({"name": "template_not_in_new", "detail": "the served directory has a chat template and the new model ships none (it is another model's)"})
    elif s_sha and n_sha and s_sha != n_sha:
        problems.append({"name": "template_differs_from_new",
                         "detail": f"the served template (sha256 {s_sha[:12]}) is not the new model's (sha256 {n_sha[:12]})"})
    elif not s_sha and not n_sha:
        notes.append("neither the served directory nor the new model has a chat template, so a chat request cannot be made")
    served_tokens, new_tokens = tokens_of(served_dir, notes), tokens_of(new_snapshot, notes)
    for name in TOKEN_NAMES:
        if served_tokens[name] != new_tokens[name]:
            problems.append({"name": "special_token_differs",
                             "detail": f"{name} is {served_tokens[name]!r} in the served directory and {new_tokens[name]!r} in the new model"})
    served_ids = stop_ids_of(served_dir, notes)
    ends = stop_ids_of(new_snapshot, notes)
    eos_id = token_id(new_snapshot, new_tokens["eos_token"], notes)
    if eos_id is not None and eos_id not in ends:
        ends.append(eos_id)
    missing = [i for i in ends if i not in served_ids]
    if missing:
        problems.append({"name": "stop_id_missing",
                         "detail": f"the new model ends a turn with id {', '.join(map(str, missing))}, which the served directory's stop ids {served_ids} lack"})
    return {"template": {"served_file": s_file, "served_sha256": s_sha, "new_file": n_file, "new_sha256": n_sha},
            "special_tokens": {"served": served_tokens, "new": new_tokens},
            "stop_ids": {"served": served_ids, "new_model_ends": ends}, "problems": problems, "notes": notes}


def check_replies(record) -> dict:
    """The recorded chat replies of a swap check's `prompt_suite` record. Never raises."""
    empty = {"mode": None, "checked": False, "why_not": "", "counts": {"checked": 0, "stopped": 0, "length": 0}, "problems": []}
    if not isinstance(record, dict) or record.get("ran") is not True:
        return {**empty, "why_not": "the prompt suite did not run"}
    mode = record.get("mode") if record.get("mode") in ("chat", "completion") else None
    if mode != "chat":
        return {**empty, "mode": mode, "why_not": "the prompt suite sent plain completions (the tokenizer has no chat template), so no chat reply exists"}
    entries = record.get("entries") if isinstance(record.get("entries"), list) else []
    specials = [t for t in (record.get("special_tokens") if isinstance(record.get("special_tokens"), list) else [])
                if isinstance(t, str) and len(t) >= 3 and not t.isalnum()]
    ends = [t for t in specials if END_MARKER.search(t)]
    problems: list[dict] = []
    seen = set()
    counts = {"checked": 0, "stopped": 0, "length": 0}

    def add(name, detail):
        if name not in seen:
            seen.add(name)
            problems.append({"name": name, "detail": detail})

    for e in entries:
        if not isinstance(e, dict) or e.get("error") or e.get("mode") not in (None, "chat"):
            continue
        content = e.get("content") if isinstance(e.get("content"), str) else ""
        finish = e.get("finish_reason") if isinstance(e.get("finish_reason"), str) else None
        if not content.strip() and not e.get("tool_calls"):
            continue
        counts["checked"] += 1
        counts["stopped"] += finish in ("stop", "tool_calls")
        counts["length"] += finish == "length"
        lead = content.lstrip()
        if ROLE_HEADER.match(content) or any(lead.startswith(t) for t in specials):
            add("role_header_start", f"the reply to {e.get('id')!r} begins with a role header or a special token: {lead[:30]!r}")
        for t in ends:
            if t in content:
                add("stop_token_in_reply", f"the reply to {e.get('id')!r} holds the end token {t!r}, so the server did not stop on it")
                break
    if counts["checked"] >= 2 and counts["stopped"] == 0 and counts["length"] == counts["checked"]:
        add("no_reply_stopped", f"all {counts['checked']} chat replies ran to the length limit, so the server may not be stopping on the model's end token")
    return {"mode": "chat", "checked": counts["checked"] > 0, "why_not": "" if counts["checked"] else "no chat reply was recorded without an error",
            "counts": counts, "problems": problems}


def build_record(served_dir, new_snapshot, swap_check_path) -> dict:
    """Both checks joined. `swap_check_path` is a swap-check.json; one that cannot be read means the replies were not checked."""
    files = check_files(served_dir, new_snapshot)
    try:
        report = safeio.read_json(_real(swap_check_path), max_bytes=MAX_JSON_BYTES, expect=dict)
        suite = report.get("prompt_suite")
    except safeio.SafeReadError:
        suite = None
    replies = check_replies(suite)
    if suite is None:
        replies = {**replies, "why_not": "the swap check could not be read"}
    problems = files["problems"] + replies["problems"]
    return {"label": LABEL, "not_checked": NOT_CHECKED, "ok": not problems and replies["checked"], "problems": problems,
            "files": files, "replies": replies}


def write_record(stage_dir, served_dir, new_snapshot, swap_check_path) -> dict:
    """Build the record and write it to `stage_dir/served-chat.json` (all or nothing). Returns it."""
    rec = build_record(served_dir, new_snapshot, swap_check_path)
    safeio.atomic_write(Path(stage_dir) / FILE, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec
