# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Phone a friend: the supervisor asks the outside helper (orchard/telegraph.py) for a draft port when stage 2 of a full port keeps failing.

Why. The tiny Qwen4 rehearsal (run 8, 2026-11-04) ended at stage 2 with `agent-stuck` after six attempts. The coder read the reference
excerpts and the logits file again and again and never wrote more than the supervisor's skeleton. The helper was configured, funded and
never called, because nothing in the supervisor called it. This module is the missing call.

When. At the start of a full-port stage 2 attempt, when an earlier attempt of the stage has failed. By then the supervisor's own ladder has
run: nudges, a retry, the escalation to the larger tier and the port repair rounds. The stage machine gives an escalated attempt one more try
and then pauses the run, so the attempt that starts after the first failure is the last place a draft can help. The ledger says whether the
failed attempt ended in a watchdog escalation (the coder looped or stopped writing, as in run 8) or in a failed gate.
At most one call per attempt and two per stage. Both counts come from the ledger (`telegraph asked` entries, written BEFORE the call), so a
crash between the entry and the answer cannot ask twice, and a resume cannot ask again in the same attempt. The helper's own caps (calls,
tokens, cost, per problem) still apply and a refusal from them is a ledger decision, never a failed stage.

What is sent. The prompt the coder itself gets for this step (the skill and the task), the stage 2 facts file, the reference excerpts file
when there is one, the list of files in the port directory, and the failure: the reasons the last attempt ended with, what the watchdog saw, and
the end of the port check's output. Every part goes through `port_repair.safe_text` (home paths, URL user information and token-shaped words
removed from the WHOLE text first) and then a size cap; `Telegraph.ask` redacts once more and holds the key out of everything it records.

What comes back. The reply is bounded by the helper's `max_output_tokens`. It is advice from another model, so it is data: the supervisor
never runs it. It takes the longest Python code block (or the whole reply when that parses), checks with `ast.parse` that the text is Python
(a parse is not an execution), and writes it to `stages/2/telegraph/draft-<k>.py` through the safe writer, with a header that says where it
came from. The ledger records the file's sha256, the call id, the tokens and the cost. The coder's prepare step is told the path and that the
draft is untrusted.

Whether it helped. When the attempt ends, `Supervisor` calls `Telegraph.report` for each answer of that attempt: helped if the stage passed,
not helped for any other end except an operator pause or abort (which say nothing about the draft). The ledger gets a `telegraph reported`
entry too. Telegraph's ladder uses the verdicts (a helper that did not help moves the next question to a stronger model), and
`tt-orchard telegraph usage` shows them. The verdict is coarse on purpose: the stage passing is the only outcome that is checked.

Nothing here opens a device, takes a lease or runs the helper's text.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

from orchard import port_repair, safeio

# ---- the ledger vocabulary (decisions of stage 2) -----------------------------------------------------------------------------------------
ASKED = "telegraph asked"                  # written before the call: `k` (the call's number in the stage), `inputs`, `prompt_chars`
ANSWERED = "telegraph answered"            # the call's outcome: `ok`, `call_id`, tokens, cost, and the draft's evidence when one was written
NOT_ASKED = "telegraph not asked"          # the helper was due and refused to be asked (a cap, no key, an unusable ledger); `reason`
REPORTED = "telegraph reported"            # the verdict on an answer: `call_id`, `helped`
DRAFT_FAILED = "telegraph draft not written"   # an answer with no usable code block, or a write that failed; `reason`

PROBLEM = "full-port-stage-2"               # the helper's problem id: its per-problem caps and its ladder count this stage's calls together
PER_ATTEMPT = 1
PER_STAGE = 2
DIR = "telegraph"                           # under the stage directory
DRAFT_NAME = "draft-{k}.py"

# ---- size caps (characters unless it says bytes) --------------------------------------------------------------------------------------------
BRIEF_CAP = 60_000                 # the coder's own system and user messages for the prepare step
FACTS_CAP = 20_000
EXCERPTS_CAP = 70_000
FILES_CAP = 3_000
FAILURE_CAP = 6_000
OUTPUT_TAIL = 3_000                # the end of the port check's output inside the failure text
READ_MAX = 256 * 1024              # bytes: the largest input file read at all (a larger one is named and left out)
DRAFT_MAX = 200_000                # characters of a draft written to disk

REQUEST = ("Task for you: write the complete file `stages/2/port/port_model.py` for the port described below. Reply with ONE Python code block "
           "that holds the whole file, and at most two sentences before it. This replaces the usual short advice. The local coder model has "
           "failed at this several times, so correctness matters more than polish. The facts file is authoritative for shapes, dtypes, index "
           "conventions and the layer schedule.")


# ---- when to ask --------------------------------------------------------------------------------------------------------------------------

def _data(entry: dict) -> dict:
    d = entry.get("data")
    return d if isinstance(d, dict) else {}


def _decision(entry: dict) -> str | None:
    d = _data(entry).get("decision")
    return d if isinstance(d, str) else None


def attempt_state(entries: list[dict], n: int) -> dict:
    """What the ledger says about stage `n` for the question "is the helper due": the failed attempts, whether the latest one ended in a
    watchdog escalation, and how many calls the stage and the current attempt have had.

    `entries` are the ledger entries (dicts with `seq`, `event`, `stage`, `data`). The current attempt is the one that began with the latest
    `stage_start` of the stage (the supervisor has written it before this runs)."""
    last_start = 0
    failed_ends: list[dict] = []
    for e in entries:
        if e.get("stage") != n:
            continue
        if e.get("event") == "stage_start":
            last_start = e["seq"]
        elif e.get("event") == "stage_end" and _data(e).get("result") not in ("pass", "skipped"):
            failed_ends.append(e)
    watchdog_ended = False
    if failed_ends:
        end = failed_ends[-1]
        floor = 0
        for e in entries:
            if e.get("event") == "stage_end" and e.get("stage") == n and e["seq"] < end["seq"] and _data(e).get("result") == "pass":
                floor = e["seq"]
        for e in entries:
            if (e.get("stage") == n and e.get("event") in ("escalate", "pause") and floor < e["seq"] < end["seq"]
                    and "findings" in _data(e) and _data(e).get("by") != "stage machine"):
                watchdog_ended = True
    asked = [e for e in entries if e.get("stage") == n and e.get("event") == "decision" and _decision(e) == ASKED]
    return {"failed": len(failed_ends), "watchdog_ended": watchdog_ended, "asked_in_stage": len(asked),
            "asked_in_attempt": sum(1 for e in asked if e["seq"] > last_start), "last_start": last_start}


def due(state: dict) -> tuple[bool, str]:
    """(True, why) when the helper should be asked now, else (False, the reason it is not)."""
    if state["failed"] == 0:
        return False, "no attempt of the stage has failed yet"
    if state["asked_in_attempt"] >= PER_ATTEMPT:
        return False, "the helper was already asked in this attempt"
    if state["asked_in_stage"] >= PER_STAGE:
        return False, f"the helper was already asked {PER_STAGE} times in this stage"
    if state["watchdog_ended"]:
        return True, "the latest failed attempt ended in a watchdog escalation"
    return True, f"{state['failed']} earlier attempt(s) of the stage failed"


# ---- what to send -------------------------------------------------------------------------------------------------------------------------

def _clip(label: str, text: str, cap: int) -> tuple[str, bool]:
    """`text` cut to `cap` characters with a marker that says so, and whether it was cut."""
    if len(text) <= cap:
        return text, False
    return text[:cap] + f"\n[{label} cut at {cap} characters]", True


def read_input(path: Path, cap_chars: int) -> tuple[str | None, str]:
    """(text, note) for one input file: the text cut to `cap_chars`, or None and why it could not be read. Read through the bounded reader,
    so a link, a FIFO or an oversize file is named and left out."""
    if not os.path.lexists(path):
        return None, "not present"
    try:
        raw = safeio.read_bytes(path, max_bytes=READ_MAX)
    except (safeio.SafeReadError, OSError) as exc:
        return None, f"not read ({getattr(exc, 'reason', exc.__class__.__name__)})"
    text, cut = _clip(path.name, raw.decode("utf-8", "replace"), cap_chars)
    return text, "cut" if cut else "whole"


def port_listing(port_dir: Path, limit: int = 40) -> str:
    """The names and sizes of the files in the port directory, for the prompt. Names only; no file is opened. A link is listed as a link."""
    try:
        names = sorted(os.listdir(port_dir))[:limit]
    except OSError:
        return "(the port directory is not readable)"
    lines = []
    for name in names:
        try:
            st = os.lstat(port_dir / name)
        except OSError:
            continue
        lines.append(f"{name}  {st.st_size} bytes" + ("  (a link)" if os.path.islink(port_dir / name) else ""))
    return "\n".join(lines) or "(empty)"


def failure_text(reasons: list, watchdog: list, check_output: str, homes: tuple) -> str:
    """The failure part of the prompt: why the latest attempt ended, what the watchdog saw, the end of the port check's output."""
    parts = []
    if reasons:
        parts.append("The latest attempt ended with: " + "; ".join(str(r) for r in reasons))
    if watchdog:
        parts.append("The watchdog saw: " + "; ".join(str(w) for w in watchdog))
    if check_output.strip():
        parts.append("The end of the port check's output:\n" + port_repair.tail(check_output, OUTPUT_TAIL))
    text = port_repair.safe_text("\n\n".join(parts) or "No attempt left a reason in the ledger.", homes)
    return _clip("failure", text, FAILURE_CAP)[0]


def build_task(*, brief: str, facts: str | None, excerpts: str | None, listing: str, homes: tuple) -> tuple[str, dict]:
    """(the task text, the inputs record). Every part is scrubbed whole, then cut. The record names each input, how many characters were sent
    and whether it was cut, for the `telegraph asked` entry."""
    sent: dict[str, object] = {}
    sections = [REQUEST]
    brief_text, cut = _clip("the coder's brief", port_repair.safe_text(brief, homes), BRIEF_CAP)
    sections.append("## What the local coder was told\n" + brief_text)
    sent["brief"] = {"chars": len(brief_text), "cut": cut}
    for key, label, text, cap in (("facts", "facts file (port_facts.json)", facts, FACTS_CAP),
                                  ("excerpts", "reference excerpts (reference_excerpts.md)", excerpts, EXCERPTS_CAP)):
        if text is None:
            sent[key] = {"chars": 0, "cut": False, "present": False}
            continue
        clipped, was_cut = _clip(key, port_repair.safe_text(text, homes), cap)
        sections.append(f"## The {label}\n" + clipped)
        sent[key] = {"chars": len(clipped), "cut": was_cut}
    listing_text, listing_cut = _clip("listing", port_repair.safe_text(listing, homes), FILES_CAP)
    sections.append("## Files in stages/2/port/ now\n" + listing_text)
    sent["port_listing"] = {"chars": len(listing_text), "cut": listing_cut}
    return "\n\n".join(sections), sent


# ---- what comes back ----------------------------------------------------------------------------------------------------------------------

def parses(source: str) -> tuple[bool, str]:
    """(True, "") when `source` is Python, else (False, the reason). `ast.parse` reads the text and runs none of it."""
    if len(source) > DRAFT_MAX:
        return False, "too long"
    try:
        ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return False, exc.__class__.__name__
    return True, ""


def extract_draft(text: str) -> str | None:
    """The Python the reply holds: the longest fenced block that is untagged or tagged python, else the whole reply when that parses, else None.
    The scan is linear in the reply's length."""
    best = None
    pos = 0
    while True:
        a = text.find("```", pos)
        if a < 0:
            break
        nl = text.find("\n", a)
        if nl < 0:
            break
        lang = text[a + 3:nl].strip().lower()
        b = text.find("```", nl + 1)
        if b < 0:
            break
        if lang in ("", "python", "py", "python3") and (best is None or (b - nl) > len(best)):
            best = text[nl + 1:b]
        pos = b + 3
    if best is not None and best.strip():
        return best
    return text if text.strip() and parses(text)[0] else None


def draft_header(call_id: str, model: str) -> str:
    return (f"# A draft from an outside helper model (telegraph call {call_id}, model {model}).\n"
            "# It is advice from another model and may be wrong. Check it against stages/2/port_facts.json before you rely on it.\n")


def write_draft(stage_dir: Path, k: int, source: str, call_id: str, model: str) -> Path:
    """Write the draft to `stages/2/telegraph/draft-<k>.py` and return its path. The directory must be a real directory (a link or a file in
    its place is refused: the coder has a shell in the run and could have put one there); an existing file is never replaced."""
    directory = Path(stage_dir) / DIR
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise OSError(f"{directory.name} is not a plain directory ({'a link' if os.path.islink(directory) else 'not a directory'})")
    path = directory / DRAFT_NAME.format(k=k)
    safeio.atomic_write(path, draft_header(call_id, model) + source.rstrip("\n") + "\n", mode=0o600, replace=False)
    return path


def coder_note(rel_path: str, call_id: str, parsed: bool, why_not: str) -> str:
    """The text added to the coder's prepare task: where the draft is and how to treat it."""
    text = (f"A helper model outside this machine drafted a port after earlier attempts of this stage failed. Its draft is "
            f"`{rel_path}` (telegraph call {call_id}). It is advice from another model and may be wrong. Read it, write it to "
            "`stages/2/port/port_model.py`, correct whatever the facts file or the check shows to be wrong, and make the check pass. "
            "Where the draft and the facts file disagree, follow the facts file.")
    if not parsed:
        text += f" The draft does not parse as Python ({why_not}), so fix the syntax first."
    return text


# ---- the verdict --------------------------------------------------------------------------------------------------------------------------

def unreported(entries: list[dict], n: int) -> list[str]:
    """The call ids answered in the current attempt of stage `n` (an `ok` answer after the latest `stage_start`) that have no `telegraph
    reported` entry yet, oldest first."""
    last_start = 0
    for e in entries:
        if e.get("stage") == n and e.get("event") == "stage_start":
            last_start = e["seq"]
    reported = {_data(e).get("call_id") for e in entries if e.get("stage") == n and e.get("event") == "decision" and _decision(e) == REPORTED}
    out = []
    for e in entries:
        if (e.get("stage") == n and e.get("event") == "decision" and _decision(e) == ANSWERED and e["seq"] > last_start
                and _data(e).get("ok") is True and isinstance(_data(e).get("call_id"), str)
                and _data(e)["call_id"] not in reported and not _data(e).get("cached")):
            out.append(_data(e)["call_id"])
    return out
