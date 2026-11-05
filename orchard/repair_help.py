# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The outside helper inside the port repair rounds of a full-port stage 2 (orchard/telegraph.py, orchard/phone_a_friend.py).

Why. Run11 of the tiny Qwen4 rehearsal (2026-11-04) was the first run in which a helper draft was placed as the port and the supervisor's
check ran it on the chips. The helper was asked twice, both times at the start of an attempt, and the two asks were used up. After that the
coder alone repaired the port: it fixed a shape error in prefill (round 2), then the check failed at load, and in the rounds that followed the
coder looped on `read_file`, broke the file and ended the stage. A coder that cannot finish a port needs the helper at the point where the
port fails, with the port in front of it. The start of an attempt is too early and too rare.

When. In a repair round of stage 2 (orchard/supervisor.py `_port_repair_rounds`), whose check has just failed on the port itself:
- at the start of the round, in round 1 and in any round whose failure is no further along the request chain than the previous round's
  (`start_trigger`): the coder has had its chance or has not made progress;
- after the coder's repair step, when that step ended in a watchdog pause or escalation or in capped replies (the step's status), because
  the coder could not use the round (`coder_failed`).
At most one call per round (`PER_ROUND`) and six per stage 2 counting the asks at the start of attempts (`TOTAL_PER_STAGE`). Both counts come
from the ledger, and the ask is written before the call. The helper's own caps still apply and a refusal is a ledger decision.

What is sent. The facts file, the reference excerpts, the whole current `port_model.py` (never cut: an answer to a cut file would lose the
rest, so a port over the cap is not sent and the ledger says so), the names and sizes of the other files of the port, and the failure: the
round, where the check failed and the end of its output. Every part is cleaned of home paths, URL user information and token-shaped words
(`port_repair.safe_text`) and then cut to a cap, as the first draft's request is.

What comes back. The reply is data. It is read as Python with `ast` and run by nothing here. A reply is a stub when a required function is
missing, takes the wrong arguments, is empty or raises NotImplementedError, or when the file uses a name nothing defines
(`phone_a_friend.draft_problems`, `undefined_names`). A stub is recorded and not placed.

Placing it. The reply becomes the next `port_model.py` only when the file on disk is still the one that was sent (its sha256 is on the ask)
or the coder's step failed; both hashes go in the ledger. A coder edit made while the helper was answering is never overwritten otherwise.
The coder's repair step then starts from the placed file.

Whether it helped. After the check runs again, each answer of the round gets a verdict: helped when the check passed or now fails later in
the request chain (`OPS`: init, load, prefill, decode) than it did when the helper was asked. A different message at the same request is not
progress. The verdict goes to the helper's ledger (`Telegraph.report`, which moves its ladder to a stronger model after a not-helped verdict)
and to the run's ledger with the two requests (`telegraph repair progress`).

Nothing here opens a device, takes a lease or runs the helper's text.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from orchard import phone_a_friend as paf
from orchard import port_repair, safeio

# ---- the ledger vocabulary (decisions of stage 2) -----------------------------------------------------------------------------------------
ASKED = "telegraph repair asked"           # written before the call: `round`, `why`, `port_sha256` (the file that is sent), `failure_op`
ANSWERED = "telegraph repair answered"     # the call's outcome: `ok`, `call_id`, tokens, cost, the reply's file as evidence
NOT_ASKED = "telegraph repair not asked"   # the helper was due and could not be asked; `reason`
STUB = "telegraph repair reply is a stub"  # the reply is not a whole port; `problems`
PLACED = "telegraph repair placed"         # the reply is now port_model.py: `sent_sha256`, `found_sha256`, `port_sha256`, `why`
NOT_PLACED = "telegraph repair not placed"
PROGRESS = "telegraph repair progress"     # the verdict: `call_id`, `before_op`, `after_op`, `outcome`, `helped`

TOTAL_PER_STAGE = 6                        # asks in stage 2, at attempt starts and in repair rounds together
HELPER_ROUNDS = 10                         # rounds of the helper-driven loop (no coder step) a stage 2 attempt may run
HELPER_TOTAL = 16                          # asks in stage 2 when the helper drives (attempt starts and rounds together)
DRIVEN = "helper-driven round"             # the ledger entry of a round the helper drove: `round`, `op`, hashes, `cost_usd`, `change`
BEST = "best port recorded"                # the furthest a port has got along the request chain: `op`, `rank`, `port_sha256`, `attempt`
RESTORED = "best port restored"            # the best port put back at the start of an attempt: `op`, `port_sha256`, `skeleton_sha256`
BEST_DIR = "2.best"                        # stages/2.best/port_model.py: a copy that the move-aside of a failed attempt does not take
PASSED_RANK = 99                           # a check that passed is further along than any failure
PER_ROUND = 1
PORT_CAP = 90_000                          # characters of port_model.py sent (a larger file is not sent)
PORT_MAX = 400_000                         # bytes read at all
OPS = ("init", "load", "prefill", "decode")          # the order the check asks the port; a failure later in it is progress
PREFIX = "repair"                          # the reply's file: stages/2/telegraph/repair-<k>.py

REQUEST = ("Task for you: the port file below failed the supervisor's check on the chips. Reply with ONE Python code block that holds the "
           "COMPLETE corrected `stages/2/port/port_model.py` (the whole file, every function the check calls), and at most two sentences "
           "before it. Fix the failure shown under 'What failed' and anything else in the file that the facts file shows to be wrong; keep what "
           "already works. Do not leave a function empty or raising NotImplementedError. The local coder model has failed to repair this "
           "several times, so correctness matters more than polish.")


# ---- the request chain and progress --------------------------------------------------------------------------------------------------------

def rank(op) -> int | None:
    """How far along the check's request chain a failure at `op` is: 0 for a failure before any request, 1..4 for `OPS`, None for an op this
    module does not know (no claim of progress is made for it)."""
    if op is None:
        return 0
    return OPS.index(op) + 1 if op in OPS else None


def progressed(before_op, after_op) -> bool:
    """True when a failure at `after_op` is further along the chain than one at `before_op`. Unknown ops are never progress."""
    b, a = rank(before_op), rank(after_op)
    return b is not None and a is not None and a > b


# ---- the ledger ----------------------------------------------------------------------------------------------------------------------------

def _data(entry: dict) -> dict:
    d = entry.get("data")
    return d if isinstance(d, dict) else {}


def _decision(entry: dict) -> str | None:
    d = _data(entry).get("decision")
    return d if isinstance(d, str) else None


def _last_start(entries: list[dict], n: int) -> int:
    last = 0
    for e in entries:
        if e.get("stage") == n and e.get("event") == "stage_start":
            last = e["seq"]
    return last


def asks_in_stage(entries: list[dict], n: int) -> int:
    """The helper's asks in stage `n` so far: the attempt-start asks and the repair-round asks together."""
    names = (paf.ASKED, ASKED)
    return sum(1 for e in entries if e.get("stage") == n and e.get("event") == "decision" and _decision(e) in names)


def asked_in_round(entries: list[dict], n: int, k: int) -> bool:
    """True when a repair ask for round `k` was already written in the current attempt."""
    floor = _last_start(entries, n)
    return any(e.get("stage") == n and e.get("event") == "decision" and _decision(e) == ASKED and e["seq"] > floor
               and _data(e).get("round") == k for e in entries)


def due(entries: list[dict], n: int, k: int, total: int = TOTAL_PER_STAGE) -> tuple[bool, str]:
    """(True, "") when a repair ask for round `k` is within the limits, else (False, why). `total` is the stage's cap on asks (`HELPER_TOTAL` when
    the helper drives)."""
    if asked_in_round(entries, n, k):
        return False, f"the helper was already asked in round {k}"
    used = asks_in_stage(entries, n)
    if used >= total:
        return False, f"the helper was already asked {used} times in this stage (the limit is {total})"
    return True, ""


def previous_op(entries: list[dict], n: int, k: int):
    """(found, op): the request the check failed at when round `k - 1` of this attempt began, from its `port repair round` entry."""
    floor = _last_start(entries, n)
    for e in reversed(entries):
        if (e.get("stage") == n and e.get("event") == "decision" and _decision(e) == port_repair.DECISION and e["seq"] > floor
                and _data(e).get("round") == k - 1):
            op = _data(e).get("op")
            return True, op if isinstance(op, str) or op is None else None
    return False, None


def start_trigger(entries: list[dict], n: int, k: int, failure: dict) -> tuple[bool, str]:
    """Whether to ask at the start of round `k`: in round 1 (the port as it came out of the attempt's first step failed), and in a later round when
    the failure is no further along than in the round before (the coder did not move it). A round that moved the failure on lets the coder go
    on first; the helper is asked after its step if the step fails (`coder_failed`)."""
    found, before = previous_op(entries, n, k) if k > 1 else (False, None)
    after = failure.get("op")
    if not found:
        return True, ("the first repair round of the attempt: the port failed its first test" if k <= 1
                      else "the previous round is not in the ledger")
    if progressed(before, after):
        return False, "the failure moved further along since the previous round"
    return True, f"the failure did not move on since the previous round (it is at {after!r})"


def unreported(entries: list[dict], n: int) -> list[dict]:
    """The repair answers of stage `n` (ok, with a call id) that have no `telegraph repair progress` entry yet, oldest first, each with the ask's
    `round`, `failure_op` and the sequence number of the answer."""
    done = {_data(e).get("call_id") for e in entries if e.get("stage") == n and e.get("event") == "decision" and _decision(e) == PROGRESS}
    asks: dict[int, dict] = {}
    out = []
    for e in entries:
        if e.get("stage") != n or e.get("event") != "decision":
            continue
        d = _data(e)
        if _decision(e) == ASKED:
            asks[e["seq"]] = d
            last_ask = d
        elif (_decision(e) == ANSWERED and d.get("ok") is True and isinstance(d.get("call_id"), str) and d["call_id"] not in done
              and not d.get("cached")):
            out.append({"call_id": d["call_id"], "round": last_ask.get("round") if asks else None,
                        "failure_op": last_ask.get("failure_op") if asks else None, "seq": e["seq"]})
    return out


def tested_after(entries: list[dict], n: int, seq: int) -> bool:
    """True when the check ran again after the ledger entry `seq` (a `port repair tested` entry came later)."""
    return any(e.get("stage") == n and e.get("event") == "decision" and _decision(e) == port_repair.TESTED and e["seq"] > seq for e in entries)


# ---- what to send --------------------------------------------------------------------------------------------------------------------------

def read_port(stage_dir) -> tuple[bytes | None, str]:
    """(the bytes of stages/2/port/port_model.py, "") or (None, why). A link, a missing file and one over `PORT_MAX` are refused."""
    port = Path(stage_dir) / "port" / "port_model.py"
    try:
        if os.path.islink(port) or not os.path.isfile(port):
            return None, "port_model.py is missing or is a link"
        return safeio.read_bytes(port, max_bytes=PORT_MAX), ""
    except (OSError, safeio.SafeReadError) as exc:
        return None, f"port_model.py could not be read ({exc.__class__.__name__})"


def build_task(*, facts: str | None, excerpts: str | None, port_text: str, listing: str, homes: tuple, previous: str | None = None) -> tuple[str | None, dict]:
    """(the task text, the inputs record), or (None, the record) when the port is larger than `PORT_CAP` (the helper would answer a cut file)."""
    sent: dict[str, object] = {}
    clean_port = port_repair.safe_text(port_text, homes)
    if len(clean_port) > PORT_CAP:
        sent["port"] = {"chars": len(clean_port), "cut": True}
        return None, sent
    sections = [REQUEST]
    for key, label, text, cap in (("facts", "facts file (port_facts.json)", facts, paf.FACTS_CAP),
                                  ("excerpts", "reference excerpts (reference_excerpts.md)", excerpts, paf.EXCERPTS_CAP)):
        if text is None:
            sent[key] = {"chars": 0, "cut": False, "present": False}
            continue
        clipped, was_cut = paf._clip(key, port_repair.safe_text(text, homes), cap)
        sections.append(f"## The {label}\n" + clipped)
        sent[key] = {"chars": len(clipped), "cut": was_cut}
    if previous:
        sections.append("## What the previous round did\n" + port_repair.safe_text(previous, homes)[:1500])
        sent["previous"] = {"chars": min(len(previous), 1500)}
    sections.append("## The current stages/2/port/port_model.py (the version the check just failed on)\n```python\n" + clean_port.rstrip("\n") + "\n```")
    sent["port"] = {"chars": len(clean_port), "cut": False}
    listing_text, listing_cut = paf._clip("listing", port_repair.safe_text(listing, homes), paf.FILES_CAP)
    sections.append("## Files in stages/2/port/ now\n" + listing_text)
    sent["port_listing"] = {"chars": len(listing_text), "cut": listing_cut}
    return "\n\n".join(sections), sent


def failure(round_k: int, limit: int, op, output: str, homes: tuple) -> str:
    """The failure part of the request: the round, where the check failed and the end of its output."""
    where = f"at the {op!r} request" if op else "before any request reached the port"
    return paf.failure_text([f"Port repair round {round_k} of {limit}: the check failed {where}"], [], output, homes)


# ---- what comes back, and putting it in place ----------------------------------------------------------------------------------------------

def reply_problems(source: str) -> list[str]:
    """Why `source` is not a whole port: the draft checks of `phone_a_friend.draft_problems` plus names the file uses and nothing defines."""
    problems = paf.draft_problems(source)
    if not problems and (missing := paf.undefined_names(source)):
        problems.append("uses names nothing defines: " + ", ".join(missing[:5]))
    return problems


def header(call_id: str, model: str) -> str:
    return (f"# Replaced by the supervisor with a corrected version from an outside helper model (telegraph call {call_id}, model {model}).\n"
            "# It is advice from another model and may be wrong. Check it against stages/2/port_facts.json before you rely on it.\n")


def place(stage_dir, source: str, sent_sha: str, *, coder_failed: bool, call_id: str, model: str) -> tuple[bool, str, str]:
    """(placed, why, found_sha256). Writes `source` as `stages/2/port/port_model.py` when that file is still the one that was sent (`sent_sha`, a
    sha256 hex digest) or the coder's step failed; a file the coder changed while the helper answered is left alone. A reply that equals the port
    as it is changes nothing. `source` is not run. The caller has already refused a stub."""
    port = Path(stage_dir) / "port" / "port_model.py"
    current, why = read_port(stage_dir)
    if current is None:
        return False, why, ""
    found = hashlib.sha256(current).hexdigest()
    if found != sent_sha and not coder_failed:
        return False, "the coder changed port_model.py while the helper answered", found
    if hashlib.sha256(source.rstrip("\n").encode("utf-8")).digest() == hashlib.sha256(current.rstrip(b"\n")).digest():
        return False, "the reply is the port as it already is", found
    try:
        safeio.atomic_write(port, header(call_id, model) + source.rstrip("\n") + "\n", mode=0o644, replace=True)
    except (OSError, safeio.SafeReadError) as exc:
        return False, f"port_model.py could not be written ({exc.__class__.__name__})", found
    return True, "", found


def coder_note(call_id: str, placed: bool, rel_path: str) -> str:
    """The sentence added to the coder's repair message."""
    if placed:
        return (f"A helper model outside this machine has just replaced `stages/2/port/port_model.py` with a corrected version (telegraph call "
                f"{call_id}; its own file is `{rel_path}`). The failure above is the one the OLD file had. It is advice from another model and may "
                "be wrong: read the new file, correct whatever the facts file or the check shows to be wrong, and keep what works. If the file "
                "already looks right, change nothing that works.")
    return ""


# ---- the helper-driven loop: what the helper is told about the round before, and when to hand back to the coder ------------------------------

def change_summary(old: str, new: str) -> str:
    """One line on what `new` changes in `old`: the count of lines added and removed and the names of the functions whose text changed (at most
    six). Computed by the supervisor from the two files, never taken from the helper's words."""
    import difflib
    a, b = old.splitlines(), new.splitlines()
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    names = _changed_functions(old, new)
    where = ("; functions changed: " + ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")) if names else ""
    return f"{added} lines added, {removed} removed{where}"


def _changed_functions(old: str, new: str) -> list[str]:
    import ast
    def table(src: str) -> dict:
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return {}
        return {node.name: ast.dump(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    before, after = table(old), table(new)
    return sorted(name for name in after if before.get(name) != after[name])


def _round_entry(entries: list[dict], n: int, k: int):
    floor = _last_start(entries, n)
    for e in reversed(entries):
        if (e.get("stage") == n and e.get("event") == "decision" and _decision(e) == port_repair.DECISION and e["seq"] > floor
                and _data(e).get("round") == k):
            return e
    return None


def previous_round_section(entries: list[dict], n: int, k: int) -> str | None:
    """The text that tells the helper what round `k - 1` of this attempt did: the failure it was asked about, the change the supervisor computed
    between the file it sent and the file it placed, and the failure that followed (this round's). None in the first round or when the ledger
    holds no such round."""
    prev, cur = _round_entry(entries, n, k - 1) if k > 1 else None, _round_entry(entries, n, k)
    if prev is None or cur is None:
        return None
    pd, cd = _data(prev), _data(cur)
    change = ""
    for e in reversed(entries):
        if e.get("stage") == n and e.get("event") == "decision" and _decision(e) in (PLACED, DRIVEN) and e["seq"] > prev["seq"] and e["seq"] < cur["seq"]:
            change = str(_data(e).get("change") or "")
            if change:
                break
    lines = [f"Round {k - 1} failed at the {pd.get('op')!r} request: {pd.get('failure')}"]
    if change:
        lines.append(f"The file was then changed ({change}).")
    lines.append(f"This round the check failed at the {cd.get('op')!r} request: {cd.get('failure')}")
    lines.append("Do not repeat a change that did not move the failure.")
    return "\n".join(lines)


def same_failure_twice(entries: list[dict], n: int, k: int) -> bool:
    """True when round `k` and round `k - 1` of this attempt failed at the same request with the same message: the helper's change did not
    move anything, so the helper-driven loop hands the port back to the coder's repair rounds."""
    prev, cur = (_round_entry(entries, n, k - 1) if k > 1 else None), _round_entry(entries, n, k)
    if prev is None or cur is None:
        return False
    pd, cd = _data(prev), _data(cur)
    return pd.get("op") == cd.get("op") and bool(cd.get("failure")) and pd.get("failure") == cd.get("failure")


# ---- the best port so far, kept across attempts ----------------------------------------------------------------------------------------

def best_rank(op) -> int:
    """How far a check got: 0 for a failure before any request or an op this module does not know, 1..4 along `OPS`, `PASSED_RANK` for none."""
    r = rank(op)
    return r if r is not None else 0


def best_state(entries: list[dict], n: int):
    """The best port recorded so far in stage `n` (the `best port recorded` entry with the highest rank, the latest on a tie), or None."""
    best = None
    for e in entries:
        if e.get("stage") == n and e.get("event") == "decision" and _decision(e) == BEST:
            d = _data(e)
            if isinstance(d.get("rank"), int) and isinstance(d.get("port_sha256"), str) and (best is None or d["rank"] >= best["rank"]):
                best = d
    return best


def best_path(stage_dir) -> Path:
    """stages/2.best/port_model.py next to the stage directory: the move-aside of a failed attempt takes stages/2 and leaves this."""
    return Path(stage_dir).parent / BEST_DIR / "port_model.py"


def save_best(stage_dir, data: bytes) -> bool:
    """Write `data` as the best port's copy. True when it was written."""
    target = best_path(stage_dir)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        safeio.atomic_write(target, data, mode=0o644, replace=True)
    except (OSError, safeio.SafeReadError):
        return False
    return True


def restore_best(stage_dir, best: dict, skeleton: bytes) -> tuple[bool, str]:
    """Put the best port in place of the unchanged skeleton. (placed, why). The copy's hash must equal the one the ledger recorded, the file
    in the port directory must still be the skeleton byte for byte, and neither may be a link: a file the coder or the helper has written is
    never replaced."""
    port = Path(stage_dir) / "port" / "port_model.py"
    current, why = read_port(stage_dir)
    if current is None:
        return False, why
    if current.rstrip(b"\n") != skeleton.rstrip(b"\n"):
        return False, "port_model.py is not the unchanged skeleton"
    source = best_path(stage_dir)
    try:
        if os.path.islink(source) or not os.path.isfile(source):
            return False, "the best port's copy is missing or is a link"
        data = safeio.read_bytes(source, max_bytes=PORT_MAX)
    except (OSError, safeio.SafeReadError) as exc:
        return False, f"the best port's copy could not be read ({exc.__class__.__name__})"
    if hashlib.sha256(data).hexdigest() != best.get("port_sha256"):
        return False, "the best port's copy does not match the hash the ledger recorded"
    try:
        safeio.atomic_write(port, data, mode=0o644, replace=True)
    except (OSError, safeio.SafeReadError) as exc:
        return False, f"port_model.py could not be written ({exc.__class__.__name__})"
    return True, ""
