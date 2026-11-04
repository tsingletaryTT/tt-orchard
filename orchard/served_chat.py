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

  * `check_live_probe(record)` (issue 368): the live-probe record that serve_and_compare.py writes under `live_probe` in swap-check.json
    (the template cannot import orchard, so it records the raw answers and this module judges them with the rules of
    orchard/served_verify.py): the listing (the served name must be the name the template asked for, and the context must be the
    `--max-model-len` that run.sh passes), the two raw completions and the engine's token counter. A swap-check.json written before the
    probe existed has no record: the result is `not_checked`, never a pass. A failed probe is a problem and makes `ok` false; a probe the
    server does not offer (no /metrics, no completions route) is listed as not checked and does not.
  * the token counts of the chat replies (issue 361): completion tokens, reasoning tokens and answer tokens per reply, with their
    distribution, and the cap those counts support.

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

from orchard import safeio, scrub, served_verify

FILE = "served-chat.json"
MAX_JSON_BYTES = 64 << 20            # a tokenizer.json is tens of megabytes
MAX_TEMPLATE_BYTES = 1 << 20
TOKEN_NAMES = ("eos_token", "bos_token", "pad_token", "unk_token")
LABEL = "check of the served files, the recorded chat replies and the recorded live probe; this check sent no request to a server"
NOT_CHECKED = ("whether a reply begins and ends as the model's card describes (the card is prose); whether a model trained on a different "
               "template answers worse; the card of the model was not read")
# A role header at the start of a reply: a role word (`assistant`, `user`, `system` or `tool`) followed by a colon or a line end, with an
# optional start-of-turn token before it. A reply that starts with one of them is echoing the template.
# The files whose digests the swap test records when it runs the suite (serve_and_compare.py `served_files`). The replies are tied to them.
INPUT_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json", "chat_template.jinja")
MAX_INPUT_BYTES = 256 << 20
ROLE_HEADER = re.compile(r"\A\s*(?:<\|im_start\|>\s*)?(?:[Aa]ssistant|[Uu]ser|[Ss]ystem|[Tt]ool)[ \t]*(?::|\n)")


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
        # the swap test reads the file as text, which turns CRLF and CR line ends into LF before it hashes the template
        return safeio.read_bytes(_real(path), max_bytes=MAX_TEMPLATE_BYTES).decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
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


def file_digests(directory) -> dict:
    """{file name: sha256 of its bytes, or None when the file is absent or cannot be read} for the files in INPUT_FILES, read through links."""
    out = {}
    for name in INPUT_FILES:
        try:
            out[name] = hashlib.sha256(safeio.read_bytes(_real(Path(directory) / name), max_bytes=MAX_INPUT_BYTES)).hexdigest()
        except safeio.SafeReadError:
            out[name] = None
    return out


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


def _added_tokens(directory, incomplete: list) -> dict:
    """{content: (id, special)} for the `added_tokens` list of tokenizer.json. The first entry for a content wins, as in `_vocab`. An entry
    whose content is not a string or whose id is not a non-negative integer is skipped. `special` is True only for a JSON true."""
    data = _json(Path(directory) / "tokenizer.json", incomplete, f"{Path(directory).name}/tokenizer.json") or {}
    out: dict = {}
    for item in data.get("added_tokens") if isinstance(data.get("added_tokens"), list) else []:
        if (isinstance(item, dict) and isinstance(item.get("content"), str)
                and isinstance(item.get("id"), int) and not isinstance(item.get("id"), bool) and item["id"] >= 0):
            out.setdefault(item["content"], (item["id"], item.get("special") is True))
    return out


SHOWN_CHARS = 80               # a token's text in a problem is cut to this many characters (the cut says how long the whole text was)
TOKEN_TEXT_MAX = 200           # a token's text in the written record is cut to this many characters
RECORD_TEXT_MAX = 600          # one problem detail, incomplete line or note in the written record
RECORD_LIST_MAX = 50           # entries in each list of the written record


SCRUB_WINDOW = 256              # characters read past a cut so a credential that starts before the cut is matched whole (the patterns need 30)


def _scrubbed_prefix(text: str, limit: int) -> str:
    """The first `limit` characters of `text` with every credential in them replaced by a label, found in the text BEFORE it is cut.

    A token that starts before the cut and ends after it would leave a fragment (`hf_` and 17 payload characters at a cut of 500) that no
    pattern matches, because the patterns need 30 payload characters (issue 297, review of PR 466). The scan covers the cut plus
    SCRUB_WINDOW characters, so a token that starts inside the shown part is matched with at least SCRUB_WINDOW characters of its payload; the
    work stays bounded however long the text is. A replacement is shorter than any match, so the cut text is never longer than `limit`."""
    head = text[:limit + SCRUB_WINDOW]
    for _what, pattern in scrub.TOKEN_PATTERNS:
        head = pattern.sub("<token removed>", head)
    return head[:limit]


def _shown(value) -> str:
    """The repr of a token's text for a problem or an incomplete line, cut to SHOWN_CHARS characters. Token text comes from a tokenizer file
    (up to MAX_JSON_BYTES), and the bundle refuses a record over 8 MiB as unreadable, so one huge token must not reach a detail whole (issue 297,
    review of PR 466). A cut text says how many characters it had. Credentials are removed before the cut (`_scrubbed_prefix`)."""
    text = value if isinstance(value, str) else str(value)
    if len(text) <= SHOWN_CHARS:
        return repr(_scrubbed_prefix(text, SHOWN_CHARS))
    r = repr(_scrubbed_prefix(text, SHOWN_CHARS))
    return f"{r[:-1]}...{r[-1]} ({len(text)} characters)"


def _cut_text(value, limit: int):
    """`value` itself when it is not text or fits; otherwise its first `limit` characters (credentials removed before the cut) and the full
    length."""
    if isinstance(value, str) and len(value) > limit:
        return f"{_scrubbed_prefix(value, limit)}... ({len(value)} characters)"
    return value


def _bounded_list(items, limit: int = RECORD_LIST_MAX):
    """At most `limit` entries, the last one saying how many were left out. Not a list: returned as it is."""
    if not isinstance(items, list) or len(items) <= limit:
        return items
    left = len(items) - (limit - 1)
    kept = list(items[:limit - 1])
    kept.append({"name": "entries_not_listed", "detail": f"{left} more entries are not listed"} if items and isinstance(items[0], dict)
                else f"{left} more entries are not listed")
    return kept


def _bounded_entries(items):
    """A list of problems (dicts with a detail) or of lines, capped in count and in the length of each text."""
    out = []
    for item in _bounded_list(items) if isinstance(items, list) else []:
        out.append({**item, "detail": _cut_text(item.get("detail"), RECORD_TEXT_MAX)} if isinstance(item, dict) else _cut_text(item, RECORD_TEXT_MAX))
    return out


def _bounded_files(files: dict) -> dict:
    """`files` as written to the record: the token texts cut to TOKEN_TEXT_MAX. The checks use the whole texts and run before this."""
    out = dict(files)
    tokens = files.get("special_tokens")
    if isinstance(tokens, dict):
        out["special_tokens"] = {side: ({k: _cut_text(v, TOKEN_TEXT_MAX) for k, v in vals.items()} if isinstance(vals, dict) else vals)
                                 for side, vals in tokens.items()}
    if isinstance(files.get("end_tokens"), list):
        out["end_tokens"] = [_cut_text(v, TOKEN_TEXT_MAX) for v in _bounded_list(files["end_tokens"])]
    for key in ("problems", "incomplete"):
        out[key] = _bounded_entries(files.get(key))
    out["notes"] = _bounded_entries(files.get("notes"))
    return out


def bounded_record(rec: dict) -> dict:
    """The record with every list capped in count and every text in length, so its size is bounded whatever the input files hold."""
    out = dict(rec)
    for key in ("problems", "incomplete"):
        out[key] = _bounded_entries(rec.get(key))
    if isinstance(rec.get("files"), dict):
        out["files"] = _bounded_files(rec["files"])
    replies = rec.get("replies")
    if isinstance(replies, dict):
        out["replies"] = {**replies, "problems": _bounded_entries(replies.get("problems")), "incomplete": _bounded_entries(replies.get("incomplete"))}
    live = rec.get("live_probe")
    if isinstance(live, dict) and isinstance(live.get("problems"), list):
        out["live_probe"] = {**live, "problems": _bounded_entries(live["problems"])}
    return out


ADDED_TOKEN_SHOWN = 5          # a problem about added tokens names this many of them and counts the rest, so a hostile list gives a small record


def _added_token_problems(served: dict, new: dict) -> list[dict]:
    """One problem per kind of difference between the added tokens of the two tokenizers (issue 297: a token can keep its text and get another
    id, or lose its `special` flag, and the named-token checks above would not see it)."""
    kinds = {"added_token_id_differs": [], "added_token_special_differs": [], "added_token_missing_in_served": [], "added_token_extra_in_served": []}
    for content, (nid, nspecial) in new.items():
        if content not in served:
            kinds["added_token_missing_in_served"].append(f"{_shown(content)} (id {nid})")
            continue
        sid, sspecial = served[content]
        if sid != nid:
            kinds["added_token_id_differs"].append(f"{_shown(content)}: served {sid}, new {nid}")
        if sspecial != nspecial:
            kinds["added_token_special_differs"].append(f"{_shown(content)}: served special={sspecial}, new special={nspecial}")
    kinds["added_token_extra_in_served"] = [f"{_shown(c)} (id {i})" for c, (i, _s) in served.items() if c not in new]
    what = {"added_token_id_differs": "added tokens with another id in the served tokenizer",
            "added_token_special_differs": "added tokens whose special flag differs",
            "added_token_missing_in_served": "added tokens of the new model that the served tokenizer lacks",
            "added_token_extra_in_served": "added tokens in the served tokenizer that the new model does not have"}
    out = []
    for name, rows in kinds.items():
        if rows:
            more = f" and {len(rows) - ADDED_TOKEN_SHOWN} more" if len(rows) > ADDED_TOKEN_SHOWN else ""
            out.append({"name": name, "detail": f"{len(rows)} {what[name]}: " + "; ".join(rows[:ADDED_TOKEN_SHOWN]) + more})
    return out


# The stop-id rule of each serving backend: which files the server reads for the ids that end a turn. Only the bundle's rule is known
# (generation_config.json, then the top level of config.json). A backend that is not listed is compared on everything else and the record says the
# stop ids could not be judged (issue 297).
STOP_ID_RULES = {"bundle": "generation_config.json, then the top level of config.json"}


def check_files(served_dir, new_snapshot, backend: str | None = "bundle") -> dict:
    """The template, special tokens, eos id and stop ids of `served_dir` against the new model's `new_snapshot`. Never raises. `backend` names
    the serving backend whose stop-id rule applies (a key of STOP_ID_RULES); with an unknown one the stop ids are listed and not judged."""
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
                             "detail": f"{name} is {_shown(served_tokens[name])} in the served directory and {_shown(new_tokens[name])} in the new model"})
    s_to_id, s_to_tok = _vocab(served_dir, incomplete)
    n_to_id, n_to_tok = _vocab(new_snapshot, incomplete)
    ids = {}
    for label, tokens, to_id in (("served", served_tokens, s_to_id), ("new", new_tokens, n_to_id)):
        tok = tokens["eos_token"]
        ids[label] = to_id.get(tok) if tok is not None else None
        if tok is not None and ids[label] is None:
            incomplete.append(f"the id of the {label} directory's eos token {_shown(tok)} could not be found in its tokenizer.json")
    if ids["served"] is not None and ids["new"] is not None and ids["served"] != ids["new"]:
        problems.append({"name": "eos_id_differs",
                         "detail": f"the served tokenizer gives the eos token {_shown(new_tokens['eos_token'])} the id {ids['served']}; the new model's gives it {ids['new']}"})
    for name in TOKEN_NAMES[1:]:                       # bos, pad, unk: the named tokens must map to the same ids in both tokenizers
        tok = new_tokens[name]
        if tok is None or tok != served_tokens[name]:
            continue
        a, b = n_to_id.get(tok), s_to_id.get(tok)
        if a is None or b is None:
            incomplete.append(f"the id of the {name} {_shown(tok)} could not be found in the {'new' if a is None else 'served'} directory's tokenizer.json")
        elif a != b:
            problems.append({"name": "special_token_id_differs", "detail": f"{name} {_shown(tok)} has id {b} in the served tokenizer and {a} in the new model's"})
    problems += _added_token_problems(_added_tokens(served_dir, incomplete), _added_tokens(new_snapshot, incomplete))
    rule = STOP_ID_RULES.get(backend) if isinstance(backend, str) else None
    served_ids = stop_ids_of(served_dir, incomplete)
    ends = stop_ids_of(new_snapshot, incomplete)
    nested = (_json(Path(new_snapshot) / "config.json", incomplete, f"{Path(new_snapshot).name}/config.json") or {}).get("text_config")
    for x in _ids(nested.get("eos_token_id") if isinstance(nested, dict) else None):     # the new model's own config may nest its end ids
        if x not in ends:
            ends.append(x)
    if ids["new"] is not None and ids["new"] not in ends:
        ends.append(ids["new"])
    if not ends:
        incomplete.append("the new model's snapshot names no end id (no eos_token_id in its generation_config.json or config.json and no eos token the tokenizer knows)")
    if rule is None:
        incomplete.append(f"the stop id rule of the {backend if isinstance(backend, str) else 'unknown'} serving backend is not known, so the stop ids "
                          f"{served_ids} of the served directory were not compared with the new model's ({ends})")
    missing = [i for i in ends if i not in served_ids] if rule else []
    if missing:
        problems.append({"name": "stop_id_missing",
                         "detail": f"the new model ends a turn with id {', '.join(map(str, missing))}, which the served directory's stop ids {served_ids} lack"})
    extra = [i for i in served_ids if ends and i not in ends] if rule else []
    if extra:
        problems.append({"name": "unexpected_stop_id",
                         "detail": f"the served directory stops on id {', '.join(map(str, extra))}, which the new model does not end a turn with "
                                   f"(its end ids are {ends}); an ordinary token with that id would stop a reply early"})
    for i in ends:                                     # every end id must be the same token in the served tokenizer
        want, got = n_to_tok.get(i), s_to_tok.get(i)
        if want is None:
            incomplete.append(f"the token of the new model's end id {i} could not be found in its tokenizer.json")
        elif got is None:
            incomplete.append(f"the token of end id {i} ({_shown(want)}) could not be found in the served tokenizer.json")
        elif want != got:
            problems.append({"name": "end_id_token_differs", "detail": f"end id {i} is {_shown(got)} in the served tokenizer and {_shown(want)} in the new model's"})
    return {"template": {"served_file": s_file, "served_sha256": s_sha, "new_file": n_file, "new_sha256": n_sha},
            "special_tokens": {"served": served_tokens, "new": new_tokens}, "eos_ids": ids,
            "stop_ids": {"served": served_ids, "new_model_ends": ends}, "stop_id_rule": rule,
            "end_tokens": [n_to_tok[i] for i in ends if i in n_to_tok], "problems": problems, "incomplete": incomplete, "notes": notes}


def _texts(e: dict) -> list[str]:
    return [t for t in (e.get("content"), e.get("reasoning")) if isinstance(t, str) and t.strip()]


CAP_MARGIN = 1.5            # choice: a generation cap set from the replies seen is the largest one used, with half again as much room


def token_summary(totals, thinking, answers, max_tokens) -> dict:
    """Completion, reasoning and answer tokens of the checked chat replies (issue 361) and the cap they support. `thinking` holds the tokens of the
    reasoning text counted with the model's tokenizer (None for a reply with no count). When some reply has reasoning text, `cap_evidence` is the
    largest completion count with CAP_MARGIN on top, and says whether the suite's own limit (`max_tokens`) is below it. It is a figure to read
    beside the card, and nothing in the harness sets a cap from it."""
    reasoning = [t for t in thinking if isinstance(t, int) and not isinstance(t, bool) and t > 0]
    out = {"completion_tokens": served_verify.distribution(totals), "reasoning_tokens": served_verify.distribution(thinking),
           "answer_tokens": served_verify.distribution(answers), "replies_with_reasoning": len(reasoning)}
    biggest = max([t for t in totals if isinstance(t, int) and not isinstance(t, bool)], default=None)
    if reasoning and biggest is not None:
        cap = int(biggest * CAP_MARGIN + 0.999)
        limit = max_tokens if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) else None
        out["cap_evidence"] = {"largest_reply_tokens": biggest, "margin": CAP_MARGIN, "supported_cap": cap, "suite_limit": limit,
                               "suite_limit_below_supported_cap": limit is not None and limit < cap}
    return out


def _call_texts(e: dict) -> list[str]:
    """The name and arguments of each tool call of a reply, for the end-token search (a reply that is only a tool call has no content)."""
    out = []
    for call in e.get("tool_calls") if isinstance(e.get("tool_calls"), list) else []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        for key in ("name", "arguments"):
            v = fn.get(key)
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, (dict, list)):
                out.append(json.dumps(v, ensure_ascii=False))
    return out


def check_replies(record, end_tokens=(), start_tokens=()) -> dict:
    """The recorded chat replies of a swap check's `prompt_suite` record, with `end_tokens` the strings of the new model's end-of-turn ids
    and `start_tokens` named tokens (bos, pad, unk, eos) that the suite's `special_tokens` list may lack because they live in the
    vocabulary and not in `added_tokens`. Never raises."""
    empty = {"mode": None, "checked": False, "why_not": "", "counts": {"checked": 0, "stopped": 0, "length": 0}, "problems": [], "incomplete": []}
    if not isinstance(record, dict) or record.get("ran") is not True:
        return {**empty, "why_not": "the prompt suite did not run"}
    mode = record.get("mode") if record.get("mode") in ("chat", "completion") else None
    if mode != "chat":
        return {**empty, "mode": mode, "why_not": "the prompt suite sent plain completions (the tokenizer has no chat template), so no chat reply exists"}
    entries = record.get("entries") if isinstance(record.get("entries"), list) else []
    specials = [t for t in (record.get("special_tokens") if isinstance(record.get("special_tokens"), list) else []) + list(start_tokens)
                if isinstance(t, str) and len(t) >= 3 and not t.isalnum()]
    ends = [t for t in end_tokens if isinstance(t, str) and t]
    problems: list[dict] = []
    seen = set()
    counts = {"checked": 0, "stopped": 0, "length": 0}
    totals, thinking, answers = [], [], []

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
        totals.append(e.get("completion_tokens"))
        thinking.append(e.get("reasoning_tokens"))
        answers.append(e.get("answer_tokens"))
        for text in texts:
            lead = text.lstrip()
            if ROLE_HEADER.match(text) or any(lead.startswith(t) for t in specials):
                add("role_header_start", f"the reply to {_shown(e.get('id'))} begins with a role header or a special token: {_scrubbed_prefix(lead, 30)!r}")
        for text in texts + _call_texts(e):
            for t in ends:
                if t in text:
                    add("stop_token_in_reply", f"the reply to {_shown(e.get('id'))} holds the end token {_shown(t)}, so the server did not stop on it")
                    break
    if counts["checked"] >= 2 and counts["stopped"] == 0 and counts["length"] == counts["checked"]:
        add("no_reply_stopped", f"all {counts['checked']} chat replies ran to the length limit, so the server may not be stopping on the model's end token")
    unseen = []
    if counts["checked"] and not counts["stopped"]:
        unseen.append("no checked chat reply was seen to stop (a stop or tool-call finish), so the replies do not show the server stopping on the end token")
    return {"mode": "chat", "checked": counts["checked"] > 0, "why_not": "" if counts["checked"] else "no chat reply was recorded without an error",
            "counts": counts, "problems": problems, "incomplete": unseen, "tokens": token_summary(totals, thinking, answers, record.get("max_tokens"))}


def check_live_probe(record, *, native_context=None) -> dict:
    """Judge the `live_probe` record of a swap check (issue 368) with the rules of orchard/served_verify.py. Returns `{"state", "results",
    "problems", "not_checked"}`: `state` is `not_checked` when there is no record (a swap check from before the probe, or one that switched it off),
    `fail` when any probe failed, else `pass`. Never raises. The record's model name is the name the template asked for, and the listing must hold it."""
    results: list[dict] = []
    if not isinstance(record, dict) or record.get("ran") is not True:
        return {"state": "not_checked", "results": [], "problems": [],
                "not_checked": ["the swap check has no live probe record (it was written before the live probe existed, or the probe was switched off)"]}
    asked = record.get("model") if isinstance(record.get("model"), str) and record.get("model") else None
    declared_len = record.get("declared_max_model_len")
    declared_len = declared_len if isinstance(declared_len, int) and not isinstance(declared_len, bool) and declared_len > 0 else None

    def raw(entry):
        """(status, body bytes) of a recorded request; (None, error text) when it recorded no answer."""
        if not isinstance(entry, dict):
            return None, "the record has no entry for this request"
        status = entry.get("status")
        if status is None or isinstance(status, bool) or not isinstance(status, int):
            return None, str(entry.get("error") or "no answer was recorded")[:200]
        body = entry.get("body")
        return status, (json.dumps(body).encode("utf-8") if isinstance(body, dict) else b"")

    status, body = raw(record.get("listing"))
    if status is None:
        results.append({"name": "served_model", "state": "fail", "detail": f"GET /v1/models got no answer: {body}"})
    elif asked is None:
        results.append({"name": "served_model", "state": "not_checked", "detail": "the record does not say which model name the swap check asked for"})
    else:
        results.append(served_verify.check_listing(status, body, declared_name=asked, declared_context=declared_len, native_context=native_context))
    completions = record.get("completions") if isinstance(record.get("completions"), dict) else {}
    for name, _ in served_verify.COMPLETION_PROBES:
        status, body = raw(completions.get(name))
        results.append({"name": name, "state": "fail", "detail": f"no answer from /v1/completions: {body}"} if status is None
                       else served_verify.judge_completion(name, status, body))
    arithmetic = completions.get("arithmetic") if isinstance(completions.get("arithmetic"), dict) else {}
    usage = (arithmetic.get("body") or {}).get("usage") if isinstance(arithmetic.get("body"), dict) else None
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    text = lambda v: "\n".join(x for x in v if isinstance(x, str)) if isinstance(v, list) else None      # noqa: E731
    results.append(served_verify.check_counter(text(metrics.get("before")), text(metrics.get("after")), completion_tokens=tokens))
    problems = [{"name": r["name"], "detail": r["detail"]} for r in results if r["state"] == "fail"]
    missing = [f"{r['name']}: {r['detail']}" for r in results if r["state"] == "not_checked"]
    return {"state": "fail" if problems else "pass", "results": results, "problems": problems, "not_checked": missing}


def build_record(served_dir, new_snapshot, swap_check_path) -> dict:
    """Both checks joined. `swap_check_path` is a swap-check.json; one that cannot be read means the replies were not checked."""
    suite = None
    live_record = None
    backend: str | None = "bundle"                       # a swap check with no `kind` was written by the bundle's serve_and_compare.py
    try:
        report = safeio.read_json(_real(swap_check_path), max_bytes=MAX_JSON_BYTES, expect=dict)
        suite = report.get("prompt_suite")
        live_record = report.get("live_probe")
        if "kind" in report:                             # the container's script writes kind "container"; its stop-id rule is not in STOP_ID_RULES
            backend = report["kind"] if isinstance(report["kind"], str) else None
    except safeio.SafeReadError as exc:
        report = None
        if not _absent(exc):
            incomplete_early = f"the swap check could not be read: {exc.reason}"
        else:
            incomplete_early = None
    else:
        incomplete_early = None
    files = check_files(served_dir, new_snapshot, backend=backend)
    incomplete = list(files["incomplete"])
    if incomplete_early:
        incomplete.append(incomplete_early)
    served_named = files["special_tokens"].get("served") if isinstance(files.get("special_tokens"), dict) else None
    named = [v for v in served_named.values() if isinstance(v, str)] if isinstance(served_named, dict) else []
    replies = check_replies(suite, files["end_tokens"], start_tokens=named)
    if suite is None:
        replies = {**replies, "why_not": "the swap check could not be read"}
    live = check_live_probe(live_record)
    problems = files["problems"] + replies["problems"] + [{"name": f"live_probe_{p['name']}", "detail": p["detail"]} for p in live["problems"]]
    incomplete += replies.get("incomplete", [])
    if replies["mode"] == "chat":
        # tie the replies to the files: the template and the tokenizer and stop files the swap test saw when it ran must be what is compared now
        recorded = suite.get("chat_template") if isinstance(suite.get("chat_template"), dict) else {}
        digest, now = recorded.get("sha256"), files["template"]["served_sha256"]
        if not isinstance(digest, str):
            incomplete.append("the swap check recorded no chat template digest, so the replies cannot be tied to the files compared here")
        elif now is None:
            incomplete.append("the swap test recorded a chat template digest and the served directory has no template on disk now, so the "
                              "replies cannot be tied to a template")
        elif digest != now:
            problems.append({"name": "template_changed_since_replies",
                             "detail": f"the swap test recorded a chat template with sha256 {digest[:12]}; the served directory now holds {now[:12]}"})
        seen = suite.get("served_files")
        if not isinstance(seen, dict):
            incomplete.append("the swap check recorded no served_files digests, so the replies cannot be tied to the tokenizer and stop files compared here")
        else:
            here = file_digests(served_dir)
            after = suite.get("served_files_after")
            if not isinstance(after, dict):
                incomplete.append("the swap check recorded no served_files_after digests, so a file changed while the replies were made would not be seen")
            else:
                moved = sorted(n for n in INPUT_FILES if seen.get(n) != after.get(n))
                if moved:
                    problems.append({"name": "inputs_changed_during_replies",
                                     "detail": "these files differ between the start and the end of the swap test's replies: " + ", ".join(moved)})
            if seen.get("tokenizer.json") is None or (isinstance(after, dict) and after.get("tokenizer.json") is None):
                incomplete.append("tokenizer.json could not be hashed when the swap test ran (absent, refused or unreadable), so the replies cannot be "
                                  "tied to it")
            changed = sorted(n for n in INPUT_FILES if n != "chat_template.jinja" and n in seen and seen.get(n) != here.get(n))
            if changed:
                problems.append({"name": "inputs_changed_since_replies",
                                 "detail": "these files differ from the ones the swap test served: " + ", ".join(changed)})
            absent = [n for n in INPUT_FILES if n not in seen]
            if absent:
                incomplete.append("the swap check's served_files lacks " + ", ".join(absent))
    return bounded_record({"label": LABEL, "not_checked": NOT_CHECKED, "ok": not problems and not incomplete and replies["checked"],
                           "problems": problems, "incomplete": incomplete, "files": files, "replies": replies, "live_probe": live})


def write_record(stage_dir, served_dir, new_snapshot, swap_check_path) -> dict:
    """Build the record and write it to `stage_dir/served-chat.json` (all or nothing). Returns it."""
    rec = build_record(served_dir, new_snapshot, swap_check_path)
    safeio.atomic_write(Path(stage_dir) / FILE, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec
