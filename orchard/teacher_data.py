# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Teacher data: send prompts to a running model server and write the replies as `messages` JSONL (issue 97, first slice).

tt-orchard can serve a teacher, check a student and bring a same-shape student up as a weights-only swap. It does not train. The first thing any
distillation recipe needs is the teacher's answers to a set of prompts, in a form a trainer reads. This command collects them from a server that is
already running (the stage 2 or 4 server, or a shipped package's). It opens no device, so it needs no lease of its own; the board that serves the
model is the caller's.

    python3 -m orchard.teacher_data run --port <server port> --model <served model name> --prompts suite|agreement|FILE --out teacher.jsonl \\
        [--max-tokens 2048] [--thinking on|off] [--keep-truncated]

`--prompts` is one of the shipped sets by name (`suite` is prompt_suite.json, `agreement` is agreement_prompts.json, so the output is comparable
with the checks that use them), a prompt-set JSON file (`{"prompts": [{"id", "kind", "messages"}]}`) or a JSONL file with one prompt a line: an
object with `messages` (and optionally `id` and `kind`) or a bare list of messages. A message is a role (system, user, assistant or tool) and a
text; any other key is dropped. A prompt has to end with a user turn. Ids are unique; a line without one is named `p0001`, `p0002` and so on.

Each prompt is posted to /v1/chat/completions on the loopback address with temperature 0, the token limit given and a streamed reply. `--thinking`
sends `chat_template_kwargs: {"enable_thinking": true|false}`; without it nothing is sent and the template's own default applies. The
choice is recorded in the header.

The output file has one JSON object a line: a `header` (the model name the command sent, the settings, the prompt file's name and sha256), a
`sample` for each prompt that finished (`messages` with the assistant reply appended, the finish reason, the token counts the server reported or
null, the sha256 of the prompt's messages) and a closing `summary` (how many were written, which truncated replies were kept or refused, which
prompts failed and why). A reply cut off at the token limit (finish reason `length`) is refused: it is left out and listed, because a trainer
that learns from half an answer learns to stop early. `--keep-truncated` keeps it and marks the sample `truncated: true`. The file is written once,
at the end, with `safeio.atomic_write`, so a reader never sees a part of it. Two runs against a server that answers the same way write the same bytes
(the file holds no timestamps).

Exit codes: 0 when every prompt gave a sample; 1 when some prompt was refused as truncated or failed (the file is still written with the others);
2 when the arguments or the prompt file are refused before any request is sent.

Limits. The model name is the one the command sends: it does not check which weights the server loaded, and the header says so. The server must be
on this machine; ports 8000 and 8100 to 8103 are refused because other services use them; a redirect is a failure and is never followed. Nothing
here trains a student, computes a loss or needs a board: the logit distillation the issue also describes needs full logits that the bundles do
not return, and a trainer. Nothing has run against a real teacher: the tests use a fake server.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
from pathlib import Path

from orchard import safeio

LOOPBACK = ("127.0.0.1", "localhost", "::1")
FORBIDDEN_PORTS = frozenset({8000, 8100, 8101, 8102, 8103})
SHIPPED = {"suite": "prompt_suite.json", "agreement": "agreement_prompts.json"}
SHIPPED_DIR = Path(__file__).resolve().parent / "skills" / "weights-swap-templates"
ROLES = ("system", "user", "assistant", "tool")
MAX_FILE_BYTES = 8 << 20          # choice: a few thousand prompts fit in a few megabytes
MAX_PROMPTS = 10_000
MAX_MESSAGES = 200                # messages in one prompt
MAX_ID_CHARS = 200
MAX_TOKENS_LIMIT = 100_000
STREAM_EVENT_MAX_BYTES = 1 << 20  # one streamed event: a few hundred bytes in practice
DEFAULT_MAX_TOKENS = 2048
REQUEST_TIMEOUT_S = 3600.0        # choice: a long thinking reply on one board can take many minutes
NOTE = "the name sent in each request; this command does not check which weights the server loaded"


class TeacherDataError(ValueError):
    """A prompt file or an argument that is refused before any request is sent."""


# ---- the prompts -----------------------------------------------------------------------------------------------------------------

def _messages_problem(messages) -> str | None:
    """Why `messages` is not a usable prompt (None when it is): a non-empty list of at most MAX_MESSAGES role and text pairs that ends with a user turn."""
    if not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES:
        return f"messages must be a non-empty list of at most {MAX_MESSAGES} messages"
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str) or not isinstance(m.get("role"), str):
            return "every message needs a string role and a string content (messages)"
        if m["role"] not in ROLES:
            return f"a message has the role {m['role'][:30]!r}; the roles are {', '.join(ROLES)}"
    if messages[-1]["role"] != "user":
        return "the last message of a prompt must be a user turn"
    return None


def _prompt(row, index: int, where: str) -> dict:
    """One prompt {id, kind, messages} from a JSONL row or a prompt-set entry. Raises TeacherDataError with `where` in the message."""
    if isinstance(row, list):
        row = {"messages": row}
    if not isinstance(row, dict):
        raise TeacherDataError(f"{where}: a prompt is an object with messages, or a list of messages")
    problem = _messages_problem(row.get("messages"))
    if problem:
        raise TeacherDataError(f"{where}: {problem}")
    pid = row.get("id", f"p{index:04d}")
    if not isinstance(pid, str) or not pid or len(pid) > MAX_ID_CHARS:
        raise TeacherDataError(f"{where}: the id must be a non-empty string of at most {MAX_ID_CHARS} characters")
    kind = row.get("kind")
    if kind is not None and not isinstance(kind, str):
        raise TeacherDataError(f"{where}: the kind must be a string")
    return {"id": pid, "kind": kind, "messages": [{"role": m["role"], "content": m["content"]} for m in row["messages"]]}


def load_prompts(source: str) -> tuple[list[dict], str, str]:
    """(prompts, sha256 of the file's bytes, a label for the header) for `source`: a shipped set's name, a prompt-set JSON file or a JSONL file.
    The file is read through safeio (no link, no special file, at most MAX_FILE_BYTES) and parsed with its JSON limits. Raises TeacherDataError."""
    shipped = SHIPPED.get(source)
    path = SHIPPED_DIR / shipped if shipped else Path(source)
    label = source if shipped else path.name
    try:
        raw = safeio.read_bytes(path, max_bytes=MAX_FILE_BYTES)
    except safeio.SafeReadError as exc:
        raise TeacherDataError(f"the prompt file {source!r} cannot be read: {exc.reason}") from exc
    sha = hashlib.sha256(raw).hexdigest()
    prompts: list[dict] = []
    try:
        if path.suffix == ".jsonl":
            for n, line in enumerate(raw.split(b"\n"), start=1):
                if not line.strip():
                    continue
                try:
                    row = safeio.parse_json_bytes(line, max_bytes=MAX_FILE_BYTES)
                except safeio.SafeReadError as exc:
                    raise TeacherDataError(f"line {n}: {exc.reason}") from exc
                prompts.append(_prompt(row, len(prompts) + 1, f"line {n}"))
                if len(prompts) > MAX_PROMPTS:
                    raise TeacherDataError(f"the prompt file has more than {MAX_PROMPTS}: at most {MAX_PROMPTS} prompts are accepted")
        else:
            try:
                data = safeio.parse_json_bytes(raw, max_bytes=MAX_FILE_BYTES)
            except safeio.SafeReadError as exc:
                raise TeacherDataError(f"the prompt file is not usable JSON: {exc.reason}") from exc
            rows = data.get("prompts") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise TeacherDataError("a prompt-set file is an object with a prompts list")
            if len(rows) > MAX_PROMPTS:
                raise TeacherDataError(f"the prompt file has {len(rows)}: at most {MAX_PROMPTS} prompts are accepted")
            prompts = [_prompt(r, i, f"prompt {i}") for i, r in enumerate(rows, start=1)]
    except TeacherDataError:
        raise
    if not prompts:
        raise TeacherDataError("the prompt file holds no prompts")
    seen = set()
    for p in prompts:
        if p["id"] in seen:
            raise TeacherDataError(f"the prompt id {p['id']!r} appears more than once")
        seen.add(p["id"])
    return prompts, sha, label


def prompt_sha256(messages: list) -> str:
    """The sha256 of the prompt's messages as canonical JSON: ties a sample to the prompt it answers."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


# ---- the request -----------------------------------------------------------------------------------------------------------------

def http_complete(host: str, port: int, model: str, messages: list, max_tokens: int, thinking: bool | None, timeout: float = REQUEST_TIMEOUT_S) -> dict:
    """One streamed chat request at temperature 0. Returns {text, finish_reason, prompt_tokens, completion_tokens} (the counts are None when the server
    reports no usage). Raises OSError for a connection failure, a non-200 answer (a redirect included: it is never followed) and ValueError for an
    error event in the stream, a stream without `[DONE]`, no finish reason, or an empty reply."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    text, usage, finish, done = [], {}, None, False
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise OSError(f"the server answered HTTP {resp.status}" + (", a redirect; redirects are not followed" if 300 <= resp.status < 400 else ""))
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            try:
                obj = safeio.parse_json_bytes(payload, max_bytes=STREAM_EVENT_MAX_BYTES)       # the library's JSON limits: no NaN, no repeated key, bounded depth
            except safeio.SafeReadError as exc:
                raise ValueError(f"a stream event is not usable JSON: {exc.reason}") from exc
            if not isinstance(obj, dict):
                raise ValueError("a stream event is not an object")
            if obj.get("error") is not None:
                raise ValueError(f"the server sent an error event: {str(obj['error'])[:200]}")
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                piece = (choice.get("delta") or {}).get("content")
                if isinstance(piece, str):
                    text.append(piece)
                if isinstance(choice.get("finish_reason"), str):
                    finish = choice["finish_reason"]
    finally:
        conn.close()
    if not done:
        raise ValueError("the stream ended without the [DONE] marker")
    if finish is None:
        raise ValueError("the server sent no finish reason, so a reply cut off at the limit cannot be told from a finished one")
    reply = "".join(text)
    if not reply.strip():
        raise ValueError("the server returned an empty reply")

    def count(key):
        v = usage.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
    return {"text": reply, "finish_reason": finish, "prompt_tokens": count("prompt_tokens"), "completion_tokens": count("completion_tokens")}


# ---- collecting ------------------------------------------------------------------------------------------------------------------

def collect(send, prompts: list[dict], *, max_tokens: int, keep_truncated: bool, progress=None) -> dict:
    """Send every prompt with `send(messages, max_tokens)` (the dict of `http_complete`) and return
    {samples, truncated_kept, refused_truncated, errors}. A failing prompt is listed with its error and the run goes on; a reply cut off at the limit is
    left out and listed unless `keep_truncated`; a prompt that does not end with a user turn is not sent."""
    samples, kept, refused, errors = [], [], [], []
    for p in prompts:
        messages = p["messages"]
        if not messages or messages[-1].get("role") != "user":
            errors.append({"id": p["id"], "error": "the last message is not a user turn, so the prompt was not sent"})
            continue
        try:
            got = send(messages, max_tokens)
            text = got["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("the server returned an empty reply")
        except (OSError, ValueError, http.client.HTTPException, KeyError, TypeError) as exc:
            errors.append({"id": p["id"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        truncated = got.get("finish_reason") == "length"
        if truncated and not keep_truncated:
            refused.append(p["id"])
            continue
        if truncated:
            kept.append(p["id"])
        sample = {"type": "sample", "id": p["id"], "kind": p.get("kind"), "prompt_sha256": prompt_sha256(messages),
                  "messages": [*messages, {"role": "assistant", "content": text}], "finish_reason": got.get("finish_reason"),
                  "prompt_tokens": got.get("prompt_tokens"), "completion_tokens": got.get("completion_tokens"), "truncated": truncated}
        samples.append(sample)
        if progress:
            progress(sample)
    return {"samples": samples, "truncated_kept": kept, "refused_truncated": refused, "errors": errors}


def render(header: dict, done: dict) -> str:
    """The output file's text: the header, the samples and the summary, one JSON object a line."""
    summary = {"type": "summary", "written": len(done["samples"]), "truncated_kept": done["truncated_kept"],
               "refused_truncated": done["refused_truncated"], "errors": done["errors"]}
    return "".join(json.dumps(row) + "\n" for row in [header, *done["samples"], summary])


def _max_tokens_arg(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--max-tokens must be a whole number from 1 to {MAX_TOKENS_LIMIT}, got {text!r}") from exc
    if not 1 <= value <= MAX_TOKENS_LIMIT:
        raise argparse.ArgumentTypeError(f"--max-tokens must be from 1 to {MAX_TOKENS_LIMIT}, got {value}")
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m orchard.teacher_data", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="collect a running server's replies to a set of prompts")
    r.add_argument("--port", type=int, required=True, help="the server's port; never guessed, and 8000 and 8100 to 8103 are refused")
    r.add_argument("--host", default="127.0.0.1", help="the server's address; only a loopback address is accepted")
    r.add_argument("--model", required=True, help="the served model name sent in each request")
    r.add_argument("--prompts", required=True, help="suite, agreement, a prompt-set JSON file or a JSONL file of prompts")
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=_max_tokens_arg, default=DEFAULT_MAX_TOKENS)
    r.add_argument("--thinking", choices=("on", "off"), default=None, help="send enable_thinking true or false; leave it out for the template's default")
    r.add_argument("--keep-truncated", action="store_true", help="keep a reply cut off at the token limit, marked truncated, instead of leaving it out")
    args = ap.parse_args(argv)
    if args.host not in LOOPBACK:
        print("teacher_data: only a server on this machine's loopback address is accepted", file=sys.stderr)
        return 2
    if args.port in FORBIDDEN_PORTS:
        print(f"teacher_data: port {args.port} belongs to another service on this machine; the command refuses 8000 and 8100 to 8103", file=sys.stderr)
        return 2
    if not 1 <= args.port <= 65535:
        print(f"teacher_data: {args.port} is not a port", file=sys.stderr)
        return 2
    try:
        prompts, sha, label = load_prompts(args.prompts)
    except TeacherDataError as exc:
        print(f"teacher_data: {exc}", file=sys.stderr)
        return 2
    thinking = None if args.thinking is None else args.thinking == "on"
    header = {"type": "header", "tool": "orchard.teacher_data", "version": 1,
              "model": {"requested": args.model, "verified": False, "note": NOTE},
              "settings": {"temperature": 0, "max_tokens": args.max_tokens, "thinking": args.thinking or "server default", "api": "chat",
                           "keep_truncated": bool(args.keep_truncated)},
              "prompts": {"source": label, "sha256": sha, "count": len(prompts)}}
    done = collect(lambda messages, k: http_complete(args.host, args.port, args.model, messages, k, thinking), prompts,
                   max_tokens=args.max_tokens, keep_truncated=args.keep_truncated,
                   progress=lambda s: print(json.dumps({k: s[k] for k in ("id", "finish_reason", "completion_tokens")}), flush=True))
    try:
        safeio.atomic_write(args.out, render(header, done))
    except OSError as exc:
        print(f"teacher_data: cannot write {args.out}: {exc}", file=sys.stderr)
        return 2
    bad = len(done["refused_truncated"]) + len(done["errors"])
    print(f"wrote {args.out}: {len(done['samples'])} samples" + (f", {len(done['refused_truncated'])} truncated replies left out, "
                                                               f"{len(done['errors'])} failed" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
