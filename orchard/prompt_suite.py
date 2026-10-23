# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The prompt-suite check of the stage 2 and stage 4 swap tests (issue 15).

The 32-token check compares one prompt with the stage 1 CPU reference. It cannot tell a model that
serves badly on other kinds of prompt from one that serves well, and it cannot tell a wrong model
from a quantised right one when the two are close. This module holds the part that runs outside the
hardware test:

- the committed suite (`prompt_suite.json` next to the templates) and its checks,
- the checks that read a reply for the failure shapes the tenstorrent `qualitative-check` skill lists
  (added further down), and
- the comparison with a control, which decides what the stage gates fail.

The template (serve_and_compare.py) only sends the prompts and records the raw replies in
evidence/swap-check.json. The gates read those replies and decide here, so an agent cannot change a
verdict by editing its copy of the template. A copy that sends other prompts is caught by the
suite's sha256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from orchard.defaults import SUITE_MIN_PROMPTS

SUITE_FILE = Path(__file__).resolve().parent / "skills" / "weights-swap-templates" / "prompt_suite.json"
ROLES = ("system", "user", "assistant")
SCRIPTS = ("Latin", "Cyrillic", "Greek", "CJK", "Hangul", "Arabic", "Hebrew", "Thai", "Devanagari")


class SuiteError(Exception):
    """The suite file cannot be read or does not describe a usable suite."""


def check_suite(data) -> list[str]:
    """Every problem with a suite, one sentence each; an empty list means the suite is usable."""
    if not isinstance(data, dict):
        return ["the suite is not a JSON object"]
    out = []
    if not isinstance(data.get("version"), int) or isinstance(data.get("version"), bool):
        out.append("the suite has no whole-number version")
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or len(prompts) < SUITE_MIN_PROMPTS:
        return out + [f"the suite needs at least {SUITE_MIN_PROMPTS} prompts"]
    seen, markers = set(), {}
    for i, p in enumerate(prompts):
        where = f"prompt {i}"
        if not isinstance(p, dict):
            out.append(f"{where} is not an object")
            continue
        pid = p.get("id")
        where = f"prompt {pid!r}" if isinstance(pid, str) and pid else where
        if not isinstance(pid, str) or not pid:
            out.append(f"{where} has no id")
        elif pid in seen:
            out.append(f"{where}: duplicate id")
        seen.add(pid)
        text = p.get("text")
        if not isinstance(text, str) or not text.strip():
            out.append(f"{where} has no text (the plain form used when the model has no chat template)")
        msgs = p.get("messages")
        if not isinstance(msgs, list) or not msgs:
            out.append(f"{where} has no messages")
        else:
            for m in msgs:
                if not (isinstance(m, dict) and m.get("role") in ROLES and isinstance(m.get("content"), str)
                        and m["content"].strip()):
                    out.append(f"{where}: a message needs a role from {ROLES} and non-empty content")
                    break
            if isinstance(msgs[-1], dict) and msgs[-1].get("role") != "user":
                out.append(f"{where}: the last message must be the user's")
        if p.get("expect_script") not in SCRIPTS:
            out.append(f"{where}: expect_script must be one of {SCRIPTS}")
        has_tools = "tools" in p
        if p.get("needs_tools") is True and not (isinstance(p.get("tools"), list) and p["tools"]):
            out.append(f"{where}: needs_tools is set but the prompt has no tools")
        if has_tools and p.get("needs_tools") is not True:
            out.append(f"{where}: it has tools but needs_tools is not true, so a template without tools would still send them")
        if p.get("marker") is not None:
            if not isinstance(p["marker"], str) or not p["marker"]:
                out.append(f"{where}: a marker must be a non-empty string")
            else:
                markers[pid] = p["marker"]
    for pid, marker in markers.items():
        for p in prompts:
            if not isinstance(p, dict) or not isinstance(p.get("messages"), list):
                continue
            text = json.dumps(p["messages"], ensure_ascii=False) + str(p.get("text"))
            if (marker in text) != (p.get("id") == pid):
                where = "is missing from its own prompt" if p.get("id") == pid else f"also appears in prompt {p.get('id')!r}"
                out.append(f"marker {marker!r} of prompt {pid!r} {where}")
    return out


def load_suite(path=None) -> dict:
    """The suite from `path` (the committed file by default). Raises SuiteError with every problem."""
    path = Path(path) if path else SUITE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SuiteError(f"cannot read the prompt suite {path}: {exc}") from exc
    problems = check_suite(data)
    if problems:
        raise SuiteError(f"the prompt suite {path} has problems: " + "; ".join(problems))
    return data


def suite_sha256(path=None) -> str:
    """The sha256 of the suite file's bytes. The template records its copy's; the gate compares it with the shipped file's."""
    return hashlib.sha256(Path(path or SUITE_FILE).read_bytes()).hexdigest()
