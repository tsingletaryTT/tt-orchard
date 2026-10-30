# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Is the served chat template, special tokens and stop ids the new model's, and did the chat replies start and stop like its chat format? (issue 17)

Stage 0 already compares the new model's chat template and special tokens with the nearest model's (`delta_triage.py`, hazard area
`chat_template`) and checks the stop tokens it predicts the server will use. This module checks what stage 2 and stage 4 actually served.
It is a pure check of three things already on disk, and it sends nothing to a server:

  * `check_files(served_dir, new_snapshot)`: the template, the eos, bos, pad and unk tokens, the id of the eos token and the stop ids of the
    served model directory against the new model's snapshot. The template is read the way the loader reads it: `chat_template.jinja` first,
    then the template inside `tokenizer_config.json` (a string, or a list of named templates of which `default` or else the first is used).
    A blank template is none. Files are read through their links (a served directory links its tokenizer files into the snapshot).
  * `check_replies(prompt_suite_record, end_tokens)`: the chat replies the swap check recorded, in both the answer and the reasoning text. A
    reply that begins with a role header or a special token, one that holds an end token (the strings of the ids the new model ends a turn
    with), and a run in which no reply ever stopped on its own (all hit the length limit) are problems.
  * `build_record` / `write_record` join the two into `served-chat.json` in the stage directory, and tie the replies to the files: the
    template digest the swap test recorded when it ran must equal the digest of the files compared now.

A check that could not finish is not a clean one. A file that exists and cannot be read (oversized, hostile JSON), an end token whose id
cannot be found and a recorded digest that is missing or different go in `incomplete`, and `ok` is then false. A file the model does not
ship is normal and is not incomplete.

What it does not check, and the record says so: whether a reply begins and ends as the model card describes (a card is prose), and
whether a model fine-tuned on a different template answers worse. `ok` is true only when there is no problem, nothing is incomplete and
at least one chat reply was checked.
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


def _real(path) -> str:
    # a served directory links its tokenizer files into the snapshot, and O_NOFOLLOW would refuse a link, so the path is resolved first
    try:
        return os.path.realpath(path)
    except (ValueError, OSError):          # an embedded null byte
        return str(path)


def _absent(exc) -> bool:
    return "ENOENT" in str(exc.reason)


def _json(path, incomplete: list, what: str):
    """The JSON object at `path`. None when the file is absent (normal) or cannot be read; an unreadable file is added to `incomplete`."""
    try:
        return safeio.read_json(_real(path), max_bytes=MAX_JSON_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        if not _absent(exc):
            incomplete.append(f"{what} could not be read: {exc.reason}")
        return None


def _text(path, incomplete: list, what: str) -> str | None:
    try:
        return safeio.read_bytes(_real(path), max_bytes=MAX_TEMPLATE_BYTES).decode("utf-8", errors="replace")
    except safeio.SafeReadError as exc:
        if not _absent(exc):
            incomplete.append(f"{what} could not be read: {exc.reason}")
        return None


def _inline(value) -> str | None:
    if isinstance(value, list):
        named = {t["name"]: t.get("template") for t in value if isinstance(t, dict) and isinstance(t.get("name"), str)}
        value = named.get("default") or next((t for t in named.values() if isinstance(t, str)), None)
    return value if isinstance(value, str) and value.strip() else None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_of(directory, notes: list, incomplete: list) -> tuple[str | None, str | None]:
    """(file the template comes from, its sha256) in `directory`, the loader's way; (None, None) when there is none. A template file that
    exists and cannot be read does not fall back to the inline one: it is `incomplete`."""
    d = Path(directory)
    unread = len(incomplete)
    jinja = _text(d / "chat_template.jinja", incomplete, f"{d.name}/chat_template.jinja")
    cfg = _json(d / "tokenizer_config.json", incomplete, f"{d.name}/tokenizer_config.json")
    inline = _inline(cfg.get("chat_template")) if cfg else None
    if len(incomplete) > unread and jinja is None and (d / "chat_template.jinja").exists():
        return None, None
    if jinja is not None and jinja.strip():
        if inline is not None and _sha(inline) != _sha(jinja):
            notes.append(f"{d.name}: the inline chat template in tokenizer_config.json differs from chat_template.jinja; the loader uses the file")
        return "chat_template.jinja", _sha(jinja)
    if inline is not None:
        return "tokenizer_config.json", _sha(inline)
    return None, None


def tokens_of(directory, incomplete: list) -> dict:
    cfg = _json(Path(directory) / "tokenizer_config.json", incomplete, f"{Path(directory).name}/tokenizer_config.json") or {}
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


def stop_ids_of(directory, incomplete: list) -> list[int]:
    """The ids the bundle's serving code stops on: the top-level `eos_token_id` of generation_config.json, then of config.json (a value nested
    under text_config does not count), without repeats."""
    out: list[int] = []
    for name in ("generation_config.json", "config.json"):
        data = _json(Path(directory) / name, incomplete, f"{Path(directory).name}/{name}") or {}
        for x in _ids(data.get("eos_token_id")):
            if x not in out:
                out.append(x)
    return out


def _vocab(directory, incomplete: list) -> tuple[dict, dict]:
    """(token -> id, id -> token) from tokenizer.json: added tokens first, then the vocabulary (a dict, or the list a unigram model uses,
    where an entry is a token or [token, score] and the position is the id)."""
    data = _json(Path(directory) / "tokenizer.json", incomplete, f"{Path(directory).name}/tokenizer.json") or {}
    to_id: dict = {}
    is_id = lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0      # noqa: E731
    for item in data.get("added_tokens") if isinstance(data.get("added_tokens"), list) else []:
        if isinstance(item, dict) and isinstance(item.get("content"), str) and is_id(item.get("id")):
            to_id.setdefault(item["content"], item["id"])
    model = data.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        for tok, i in vocab.items():
            if is_id(i):
                to_id.setdefault(tok, i)
    elif isinstance(vocab, list):
        for i, entry in enumerate(vocab):
            tok = entry[0] if isinstance(entry, list) and entry else entry
            if isinstance(tok, str):
                to_id.setdefault(tok, i)
    return to_id, {i: t for t, i in to_id.items()}


def check_files(served_dir, new_snapshot) -> dict:
    """The template, special tokens, eos id and stop ids of `served_dir` against the new model's `new_snapshot`. Never raises."""
    notes: list[str] = []
    incomplete: list[str] = []
    problems: list[dict] = []
    s_file, s_sha = template_of(served_dir, notes, incomplete)
    n_file, n_sha = template_of(new_snapshot, notes, incomplete)
    if n_sha and not s_sha:
        problems.append({"name": "template_missing_in_served", "detail": "the new model ships a chat template and the served directory has none"})
    elif s_sha and not n_sha:
        problems.append({"name": "template_not_in_new", "detail": "the served directory has a chat template and the new model ships none (it is another model's)"})
    elif s_sha and n_sha and s_sha != n_sha:
        problems.append({"name": "template_differs_from_new",
                         "detail": f"the served template (sha256 {s_sha[:12]}) is not the new model's (sha256 {n_sha[:12]})"})
    elif not s_sha and not n_sha:
        notes.append("neither the served directory nor the new model has a chat template, so a chat request cannot be made")
    served_tokens, new_tokens = tokens_of(served_dir, incomplete), tokens_of(new_snapshot, incomplete)
    for name in TOKEN_NAMES:
        if served_tokens[name] != new_tokens[name]:
            problems.append({"name": "special_token_differs",
                             "detail": f"{name} is {served_tokens[name]!r} in the served directory and {new_tokens[name]!r} in the new model"})
    s_to_id, _ = _vocab(served_dir, incomplete)
    n_to_id, n_to_tok = _vocab(new_snapshot, incomplete)
    ids = {}
    for label, tokens, to_id in (("served", served_tokens, s_to_id), ("new", new_tokens, n_to_id)):
        tok = tokens["eos_token"]
        ids[label] = to_id.get(tok) if tok is not None else None
        if tok is not None and ids[label] is None:
            incomplete.append(f"the id of the {label} directory's eos token {tok!r} could not be found in its tokenizer.json")
    if ids["served"] is not None and ids["new"] is not None and ids["served"] != ids["new"]:
        problems.append({"name": "eos_id_differs",
                         "detail": f"the served tokenizer gives the eos token {new_tokens['eos_token']!r} the id {ids['served']}; the new model's gives it {ids['new']}"})
    served_ids = stop_ids_of(served_dir, incomplete)
    ends = stop_ids_of(new_snapshot, incomplete)
    if ids["new"] is not None and ids["new"] not in ends:
        ends.append(ids["new"])
    missing = [i for i in ends if i not in served_ids]
    if missing:
        problems.append({"name": "stop_id_missing",
                         "detail": f"the new model ends a turn with id {', '.join(map(str, missing))}, which the served directory's stop ids {served_ids} lack"})
    extra = [i for i in served_ids if ends and i not in ends]
    if extra:
        problems.append({"name": "unexpected_stop_id",
                         "detail": f"the served directory stops on id {', '.join(map(str, extra))}, which the new model does not end a turn with "
                                   f"(its end ids are {ends}); an ordinary token with that id would stop a reply early"})
    return {"template": {"served_file": s_file, "served_sha256": s_sha, "new_file": n_file, "new_sha256": n_sha},
            "special_tokens": {"served": served_tokens, "new": new_tokens}, "eos_ids": ids,
            "stop_ids": {"served": served_ids, "new_model_ends": ends},
            "end_tokens": [n_to_tok[i] for i in ends if i in n_to_tok], "problems": problems, "incomplete": incomplete, "notes": notes}


def _texts(e: dict) -> list[str]:
    return [t for t in (e.get("content"), e.get("reasoning")) if isinstance(t, str) and t.strip()]


def check_replies(record, end_tokens=()) -> dict:
    """The recorded chat replies of a swap check's `prompt_suite` record, with `end_tokens` the strings of the new model's end-of-turn ids.
    Never raises."""
    empty = {"mode": None, "checked": False, "why_not": "", "counts": {"checked": 0, "stopped": 0, "length": 0}, "problems": []}
    if not isinstance(record, dict) or record.get("ran") is not True:
        return {**empty, "why_not": "the prompt suite did not run"}
    mode = record.get("mode") if record.get("mode") in ("chat", "completion") else None
    if mode != "chat":
        return {**empty, "mode": mode, "why_not": "the prompt suite sent plain completions (the tokenizer has no chat template), so no chat reply exists"}
    entries = record.get("entries") if isinstance(record.get("entries"), list) else []
    specials = [t for t in (record.get("special_tokens") if isinstance(record.get("special_tokens"), list) else [])
                if isinstance(t, str) and len(t) >= 3 and not t.isalnum()]
    ends = [t for t in end_tokens if isinstance(t, str) and t]
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
        texts = _texts(e)
        finish = e.get("finish_reason") if isinstance(e.get("finish_reason"), str) else None
        if not texts and not e.get("tool_calls"):
            continue
        counts["checked"] += 1
        counts["stopped"] += finish in ("stop", "tool_calls")
        counts["length"] += finish == "length"
        for text in texts:
            lead = text.lstrip()
            if ROLE_HEADER.match(text) or any(lead.startswith(t) for t in specials):
                add("role_header_start", f"the reply to {e.get('id')!r} begins with a role header or a special token: {lead[:30]!r}")
            for t in ends:
                if t in text:
                    add("stop_token_in_reply", f"the reply to {e.get('id')!r} holds the end token {t!r}, so the server did not stop on it")
                    break
    if counts["checked"] >= 2 and counts["stopped"] == 0 and counts["length"] == counts["checked"]:
        add("no_reply_stopped", f"all {counts['checked']} chat replies ran to the length limit, so the server may not be stopping on the model's end token")
    return {"mode": "chat", "checked": counts["checked"] > 0, "why_not": "" if counts["checked"] else "no chat reply was recorded without an error",
            "counts": counts, "problems": problems}


def build_record(served_dir, new_snapshot, swap_check_path) -> dict:
    """Both checks joined. `swap_check_path` is a swap-check.json; one that cannot be read means the replies were not checked."""
    files = check_files(served_dir, new_snapshot)
    incomplete = list(files["incomplete"])
    suite = None
    try:
        report = safeio.read_json(_real(swap_check_path), max_bytes=MAX_JSON_BYTES, expect=dict)
        suite = report.get("prompt_suite")
    except safeio.SafeReadError as exc:
        if not _absent(exc):
            incomplete.append(f"the swap check could not be read: {exc.reason}")
    replies = check_replies(suite, files["end_tokens"])
    if suite is None:
        replies = {**replies, "why_not": "the swap check could not be read"}
    problems = files["problems"] + replies["problems"]
    if replies["mode"] == "chat":
        # tie the replies to the files: the template the swap test saw when it ran must be the template compared now
        recorded = suite.get("chat_template") if isinstance(suite.get("chat_template"), dict) else {}
        digest, now = recorded.get("sha256"), files["template"]["served_sha256"]
        if not isinstance(digest, str):
            incomplete.append("the swap check recorded no chat template digest, so the replies cannot be tied to the files compared here")
        elif now is not None and digest != now:
            problems.append({"name": "template_changed_since_replies",
                             "detail": f"the swap test recorded a chat template with sha256 {digest[:12]}; the served directory now holds {now[:12]}"})
    return {"label": LABEL, "not_checked": NOT_CHECKED, "ok": not problems and not incomplete and replies["checked"], "problems": problems,
            "incomplete": incomplete, "files": files, "replies": replies}


def write_record(stage_dir, served_dir, new_snapshot, swap_check_path) -> dict:
    """Build the record and write it to `stage_dir/served-chat.json` (all or nothing). Returns it."""
    rec = build_record(served_dir, new_snapshot, swap_check_path)
    safeio.atomic_write(Path(stage_dir) / FILE, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec
