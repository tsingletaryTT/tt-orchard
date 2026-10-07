# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Read qwen-code session transcripts as the watchdog's event stream.

This module owns the mapping from a qwen-code (0.24.7) chat file to `Event`s, and the sanitised
signature files the tests commit. A chat file is JSON lines. The records used:
- `system` with subtype `ui_telemetry` whose uiEvent `event.name` is `qwen-code.api_response`:
  one model call, with input, output and thinking token counts, the duration and the visible
  response text. Calls by memory subagents carry `subagent_name` and are skipped.
- `assistant`: message.parts holds thought text and `functionCall {name, args}` parts. It follows
  the api_response it belongs to. A call retried by qwen-code itself has an api_response and no
  assistant record.
- `tool_result`: message.parts[].functionResponse {name, response}.

A response is held back until the next api_response or tool result, so `had_tool_call` is known
when it is emitted. Only hashes, counts, tool names, kinds and times leave this module; no message
text does.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from orchard.watchdog import Event


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def iso(ts: float) -> str:
    dt = datetime.fromtimestamp(round(ts, 3), tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def content_hash(obj, salt: bytes = b"") -> str:
    """sha256 of the JSON form, after `salt`. A random salt that is then thrown away keeps
    equality within one extraction and stops anyone from recovering short content (a file path,
    a one-word reply) by hashing guesses."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(salt + blob.encode("utf-8")).hexdigest()


def _parts(record: dict) -> list:
    parts = (record.get("message") or {}).get("parts") or []
    return [p for p in parts if isinstance(p, dict)]


def qwen_events(lines: Iterable[str], agent: str | None = None,
                salt: bytes = b"") -> Iterator[Event]:
    pending: Event | None = None
    calls: list[Event] = []

    def flush() -> list[Event]:
        nonlocal pending, calls
        out = [replace(pending, had_tool_call=bool(calls))] if pending is not None else []
        out += calls
        pending, calls = None, []
        return out

    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                          # a torn last line, or a non-JSON line
        if not isinstance(r, dict) or "timestamp" not in r:
            continue
        if agent is None and r.get("sessionId"):
            agent = "qwen:" + str(r["sessionId"])[:8]
        who = agent or "qwen:unknown"
        typ = r.get("type")
        if typ == "system" and r.get("subtype") == "ui_telemetry":
            ev = (r.get("systemPayload") or {}).get("uiEvent") or {}
            if ev.get("event.name") != "qwen-code.api_response" or ev.get("subagent_name"):
                continue
            yield from flush()
            text = ev.get("response_text") or ""
            dur = ev.get("duration_ms")
            pending = Event(ts=parse_ts(r["timestamp"]), agent=who, kind="response",
                            input_tokens=ev.get("input_token_count"),
                            output_tokens=ev.get("output_token_count"),
                            thinking_tokens=ev.get("thoughts_token_count"),
                            duration_s=None if dur is None else dur / 1000,
                            text_hash=content_hash(text, salt) if text else None)
        elif typ == "assistant":
            ts = parse_ts(r["timestamp"])
            for p in _parts(r):
                fc = p.get("functionCall")
                if isinstance(fc, dict):
                    calls.append(Event(ts=ts, agent=who, kind="tool_call", tool=fc.get("name"),
                                       args_hash=content_hash(fc.get("args"), salt)))
            if pending is None:
                yield from flush()
        elif typ == "tool_result":
            yield from flush()
            ts = parse_ts(r["timestamp"])
            for p in _parts(r):
                fr = p.get("functionResponse")
                if isinstance(fr, dict):
                    yield Event(ts=ts, agent=who, kind="tool_result", tool=fr.get("name"),
                                output_hash=content_hash([fr.get("name"), fr.get("response")],
                                                            salt))
    yield from flush()


def write_signature(events, path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            d = {k: v for k, v in asdict(e).items() if v is not None}
            f.write(json.dumps(d, sort_keys=True) + "\n")
            n += 1
    return n


def load_signature(path) -> list[Event]:
    return [Event(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
