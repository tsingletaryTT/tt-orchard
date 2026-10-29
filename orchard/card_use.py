# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What a package card can say about how to use a model, read from the model's own files and the package's run.sh.

A card that says "text generation" tells a reader nothing. Whether a model thinks before it answers, whether it
calls tools, whether it accepts images and what its authors built it for decide how a person should call it.
Each of those is a fact in a file, so this module reads it there. No model is involved and nothing is guessed:

- the authors' own "Intended use" paragraph from the model's README.md, to be quoted as their words and never
  as something this repository measured;
- the model's chat template: whether thinking is on unless a request turns it off, which reasoning effort
  values it accepts, whether it renders tools;
- the flags in the package's run.sh: the reasoning parser (thinking comes back in `reasoning_content`), the tool
  call parser with automatic tool choice (tool calls come back in `tool_calls`), and a vision tower that run.sh
  skips (the package is text only even when the model is multimodal upstream);
- the run's own evidence: whether a tool-call prompt in the speed runs or the boot check returned a tool call.

Everything but the last is taken from a file and not tested on the served model; the card says so
(`UNTESTED_SENTENCE`). `package_card.py` writes the sections from these facts and `card_problems` checks them.

What this does not do: read a chat template as code. It looks for the few phrases real templates use for these
settings, and says nothing about a setting it cannot find.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from orchard import safeio
from orchard.stages import inside

# A number with a performance unit in prose. The card's Numbers table is the only place for one.
PERF_IN_PROSE = re.compile(r"\b\d[\d.,]*\s*(?:tok/s|tokens/s|t/s|tokens per second|ms\b|%)")
UNTESTED_SENTENCE = "taken from the chat template and run.sh, not tested here"
MAX_AUTHORS_CHARS = 1200
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
TOOL_EVIDENCE_FILES = ("stages/7/benchmark/speed.json", "stages/7/verify/evidence/verify.json")


@dataclass(frozen=True)
class UseFacts:
    authors_use: str | None           # the authors' "Intended use" paragraph, one line, quoted verbatim
    thinking_default: bool | None     # True: the template thinks unless a request turns it off; None: not stated
    effort_values: tuple[str, ...]    # the reasoning effort values the template accepts, in its order
    effort_default: str | None
    can_disable_thinking: bool        # the template has an `enable_thinking` switch
    tools_in_template: bool           # the template renders a `tools` list
    reasoning_parser: str | None      # run.sh: --reasoning_parser NAME
    tool_parser: str | None           # run.sh: --tool-call-parser NAME
    auto_tool_choice: bool            # run.sh: --enable-auto-tool-choice
    text_only: bool                   # run.sh skips the vision tower
    tool_evidence: tuple[str, tuple[str, ...]] | None   # (run-relative evidence file, tool names a prompt returned)

    @property
    def tools_ok(self) -> bool:
        """The server can return tool calls: the template renders tools and run.sh parses calls automatically."""
        return bool(self.tools_in_template and self.tool_parser and self.auto_tool_choice)


def has_facts(u: UseFacts | None) -> bool:
    """True when at least one fact was found, so the card has something specific to say."""
    return u is not None and bool(
        u.authors_use or u.thinking_default is not None or u.effort_values or u.tools_ok or u.reasoning_parser
        or u.tool_parser or u.text_only or u.tool_evidence)


# ---- the chat template --------------------------------------------------------------------------------------------

_THINK_ON = (r"enable_thinking\s+is\s+undefined\s+or\s+enable_thinking\s+is\s+true",
             r"enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+false")
_THINK_OFF = (r"enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+true",)
_EFFORT_LIST = re.compile(r"\w*reasoning_effort\s+not\s+in\s+[(\[]([^)\]]*)[)\]]")
_EFFORT_DEFAULT = re.compile(r"reasoning_effort\s*\|\s*default\(\s*['\"]([^'\"]+)['\"]")
_TOOLS = re.compile(r"\bif\s+tools\b|\btools\s+is\b|\bin\s+tools\b|\{\{-?\s*tools\b|\btools\s*\|")


def template_facts(text: str) -> dict:
    """The settings a chat template shows: `thinking_default`, `effort_values`, `effort_default`,
    `can_disable_thinking` and `tools_in_template`."""
    if any(re.search(p, text) for p in _THINK_ON):
        thinking = True
    elif any(re.search(p, text) for p in _THINK_OFF):
        thinking = False
    else:
        thinking = None
    listed = _EFFORT_LIST.search(text)
    values = tuple(a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", listed.group(1))) if listed else ()
    default = _EFFORT_DEFAULT.search(text)
    return {"thinking_default": thinking, "effort_values": values,
            "effort_default": default.group(1) if default and values else None,
            "can_disable_thinking": "enable_thinking" in text,
            "tools_in_template": bool(_TOOLS.search(text))}


# ---- run.sh -------------------------------------------------------------------------------------------------------

_SKIP = re.compile(r"""^\s*(?:export\s+)?\w*SKIP_VISION\w*\s*=\s*["']?(\w+)["']?\s*(?:#.*)?$""")


def _code(run_sh: str) -> list[str]:
    return [ln for ln in run_sh.splitlines() if not ln.lstrip().startswith("#")]


def _flag(code: str, *names: str) -> str | None:
    for name in names:
        m = re.search(rf"(?<![\w-]){re.escape(name)}[ =]([^\s\"')]+|\"[^\"]*\")", code)
        if m:
            return m.group(1).strip("\"'")
    return None


def flag_facts(run_sh: str) -> dict:
    """What run.sh passes to the server: `reasoning_parser`, `tool_parser`, `auto_tool_choice`, `text_only`."""
    lines = _code(run_sh)
    code = "\n".join(lines)
    skip = [m.group(1).lower() for ln in lines if (m := _SKIP.match(ln))]
    return {"reasoning_parser": _flag(code, "--reasoning_parser", "--reasoning-parser"),
            "tool_parser": _flag(code, "--tool-call-parser", "--tool_call_parser"),
            "auto_tool_choice": bool(re.search(r"(?<![\w-])--enable[-_]auto[-_]tool[-_]choice\b", code)),
            "text_only": bool(skip) and skip[-1] in ("1", "true", "yes", "on")}


# ---- the authors' words -------------------------------------------------------------------------------------------

_HEADING = re.compile(r"^#{1,6}\s")
_INTENDED = re.compile(r"^#{1,6}\s*intended\s+uses?\s*:?\s*$", re.I)


def authors_intended_use(readme: str) -> str | None:
    """The paragraph(s) under the README's "Intended use" heading as one line, up to MAX_AUTHORS_CHARS and ending at a
    paragraph boundary. None when there is no such section, when the first paragraph is longer than the limit, or when
    the text holds a performance number: the card's Numbers table is the only place for one, and these are the
    authors' words and have no evidence here."""
    body = readme
    if body.startswith("---\n"):
        body = body[4:].split("\n---", 1)[-1]
    lines = body.splitlines()
    start = next((i for i, ln in enumerate(lines) if _INTENDED.match(ln.strip())), None)
    if start is None:
        return None
    section = []
    for ln in lines[start + 1:]:
        if _HEADING.match(ln):
            break
        section.append(ln)
    paragraphs = [" ".join(" ".join(p.split()) for p in block.splitlines() if p.strip() and not re.fullmatch(r"\s*<[^>]*>\s*", p))
                  for block in re.split(r"\n\s*\n", "\n".join(section))]
    paragraphs = [p for p in paragraphs if p]
    taken: list[str] = []
    for p in paragraphs:
        if len(" ".join(taken + [p])) > MAX_AUTHORS_CHARS:
            break
        taken.append(p)
    text = " ".join(taken)
    if not text or PERF_IN_PROSE.search(text):
        return None
    return text


# ---- evidence that a tool-call prompt returned a call -------------------------------------------------------------

def _tool_names(node, depth=0) -> set[str]:
    names: set[str] = set()
    if depth > 12:
        return names
    if isinstance(node, dict):
        calls = node.get("tool_calls")
        if isinstance(calls, list):
            for c in calls:
                if not isinstance(c, dict):
                    continue
                fn = c.get("function")
                name = c.get("name") if isinstance(c.get("name"), str) else (fn.get("name") if isinstance(fn, dict) else None)
                if isinstance(name, str) and name.strip():
                    names.add(name.strip())
        for v in node.values():
            names |= _tool_names(v, depth + 1)
    elif isinstance(node, list):
        for v in node:
            names |= _tool_names(v, depth + 1)
    return names


def _entries(data) -> list[dict]:
    """The records in an evidence file that say which model they measured: each dict under `models` (a report that holds
    several models) and the file's own top level when it names a model (a single measure_serving or verifier record)."""
    if not isinstance(data, dict):
        return []
    out = [v for v in data["models"].values() if isinstance(v, dict)] if isinstance(data.get("models"), dict) else []
    if "model_id" in data:
        out.append({k: v for k, v in data.items() if k != "models"})
    return out


def _measured(entry: dict, identity: tuple[str, str | None]) -> bool:
    """The record names the package's model and its revision. A revision comes after `@` in `model_id` or in a separate
    `revision` field. A record that names neither the model nor the revision is not about this package."""
    repo, rev = identity
    named = entry.get("model_id")
    if not isinstance(named, str) or not rev:
        return False
    name, _, at = named.partition("@")
    found = at or entry.get("revision")
    return name == repo and isinstance(found, str) and found == rev


def tool_call_evidence(run_dir, identity: tuple[str, str | None] | None = None) -> tuple[str, tuple[str, ...]] | None:
    """(run-relative file, sorted tool names) from the first evidence file in which a prompt of THIS package's model
    returned a tool call, or None. `identity` is the package's (model repo, revision). Evidence that names another
    model, another revision or no model is not evidence for this package, and without an identity nothing counts.
    Only files inside the run count and a file over MAX_EVIDENCE_BYTES is not read."""
    if not identity:
        return None
    for rel in TOOL_EVIDENCE_FILES:
        path = inside(run_dir, rel)
        if path is None:
            continue
        try:
            data = safeio.read_json(path, max_bytes=MAX_EVIDENCE_BYTES)     # a larger file, a FIFO, a link or deep nesting is skipped
        except safeio.SafeReadError:
            continue
        names: set[str] = set()
        for entry in _entries(data):
            if _measured(entry, identity):
                names |= _tool_names(entry)
        if names:
            return rel, tuple(sorted(names))
    return None


# ---- all of it ----------------------------------------------------------------------------------------------------

def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _template(snapshot: Path) -> str:
    text = _read(snapshot / "chat_template.jinja")
    if text is not None:
        return text
    raw = _read(snapshot / "tokenizer_config.json")
    try:
        t = json.loads(raw)["chat_template"] if raw else None
    except (ValueError, KeyError, TypeError):
        return ""
    if isinstance(t, list):
        t = next((x.get("template") for x in t if isinstance(x, dict) and x.get("name") == "default"),
                 next((x.get("template") for x in t if isinstance(x, dict)), None))
    return t if isinstance(t, str) else ""


def read_use_facts(snapshot, run_sh: str, run_dir, *, identity: tuple[str, str | None] | None = None,
                   measured: bool = True) -> UseFacts | None:
    """The facts for one package: the model's snapshot (README.md and chat template), the package's run.sh text and the
    run directory (for the evidence files). `identity` is the package's (model repo, revision): tool-call evidence counts
    only when it names that model. `measured` is False for a profile the run did not measure (an optional one): the
    run-wide evidence is not about its server, so it is left out. None when none of them gives a fact."""
    snap = Path(snapshot)
    readme = _read(snap / "README.md") or ""
    t, f = template_facts(_template(snap)), flag_facts(run_sh or "")
    facts = UseFacts(authors_use=authors_intended_use(readme), tool_evidence=tool_call_evidence(run_dir, identity) if measured else None, **t, **f)
    return facts if has_facts(facts) else None
