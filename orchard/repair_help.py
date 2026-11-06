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
BEST_DIR = "2.best"                        # stages/2.best/<snapshot>/: copies of the best port that the move-aside of a failed attempt does not take
PASSED_RANK = 99                           # a check that passed is further along than any failure
PER_ROUND = 1
PORT_CAP = 90_000                          # characters of port_model.py sent (a larger file is not sent)
PORT_PARTS = ("port", "port_model.py")         # the port file beneath the stage directory
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
    """The seq of the stage's latest attempt start. A resumed start belongs to the attempt it resumes, so it is not a boundary (the same rule as
    `port_repair.rounds_used`): after a restart in round 1 the helper's earlier progress is still in the history (review of PR 496)."""
    last = 0
    for e in entries:
        if e.get("stage") == n and e.get("event") == "stage_start" and _data(e).get("resumed") is not True:
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
    try:
        with safeio.pinned_root(stage_dir) as pin:
            return _read_in(pin), ""
    except safeio.SafeReadError as exc:
        return None, f"port_model.py is missing, is a link, or could not be read ({exc.reason})"


def _read_in(pin) -> bytes:
    """The port file beneath the pinned stage directory: a link at `port` or at `port_model.py` is refused, whoever planted it (review of PR 496)."""
    return safeio.read_beneath(pin.fd, PORT_PARTS, max_bytes=PORT_MAX)


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
    # The port directory is opened once, beneath the pinned stage directory, and the read and the write both go through that descriptor. A
    # directory swapped in for `port` after the open is not read or written, so the comparison and the placement are about one directory (review
    # of PR 496, round 3). The replacement is conditional on the hash of the bytes that were read: a coder file written after the comparison is
    # kept and the placement is refused.
    try:
        with safeio.pinned_root(stage_dir) as pin:
            port_fd = safeio.open_beneath(pin.fd, PORT_PARTS[:-1])
            try:
                try:
                    current = safeio.read_beneath(port_fd, PORT_PARTS[-1:], max_bytes=PORT_MAX)
                except safeio.SafeReadError as exc:
                    return False, f"port_model.py is missing, is a link, or could not be read ({exc.reason})", ""
                found = hashlib.sha256(current).hexdigest()
                if found != sent_sha and not coder_failed:
                    return False, "the coder changed port_model.py while the helper answered", found
                if hashlib.sha256(source.rstrip("\n").encode("utf-8")).digest() == hashlib.sha256(current.rstrip(b"\n")).digest():
                    return False, "the reply is the port as it already is", found
                try:
                    safeio.write_beneath_if(port_fd, PORT_PARTS[-1:], (header(call_id, model) + source.rstrip("\n") + "\n").encode("utf-8"),
                                            expect_sha256=found, mode=0o644)
                except safeio.ChangedWhileReading:
                    return False, "the coder changed port_model.py while the reply was being placed", found
            finally:
                os.close(port_fd)
    except safeio.SafeReadError as exc:
        return False, f"port_model.py could not be written ({exc.reason})", locals().get("found", "")
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


SNAPSHOT_MAX_FILES = 64                    # files of the port directory a best snapshot keeps (a port with more is not snapshotted)
SNAPSHOT_TOTAL_MAX = 4 * PORT_MAX          # bytes in one snapshot


def best_root(stage_dir) -> Path:
    """stages/2.best next to the stage directory: the move-aside of a failed attempt takes stages/2 and leaves this."""
    return Path(stage_dir).parent / BEST_DIR


def manifest_digest(files: list[dict]) -> str:
    """The sha256 of a snapshot's file list in canonical form. It names the snapshot's directory, so a directory's name says what it must hold."""
    import json
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def snapshot_port(stage_dir) -> tuple[list[tuple[str, bytes]] | None, str]:
    """(the files of stages/2/port as (relative path, bytes), "") or (None, why). Read beneath a pinned stage directory, so a link swapped in for
    `port` or for a directory under it is refused. A link or a special file in the tree refuses the snapshot, `__pycache__` is left out, and
    a tree over `SNAPSHOT_MAX_FILES` files or `SNAPSHOT_TOTAL_MAX` bytes is not kept. `port_model.py` must be among the files (the ports may import
    their neighbours, review of PR 497, so all of them are kept)."""
    try:
        with safeio.pinned_root(stage_dir) as pin:
            pfd = safeio.open_beneath(pin.fd, ("port",))
            port = safeio.Pin(pfd, "port")
            try:
                found: list[tuple[str, ...]] = []
                bad: list[str] = []

                def visit(entry) -> None:
                    if "__pycache__" in entry.parts:
                        return
                    if entry.kind == "file":
                        found.append(entry.parts)
                    elif entry.kind != "dir":
                        bad.append("/".join(entry.parts))

                scan = safeio.scan_beneath(port, visit, max_entries=4 * SNAPSHOT_MAX_FILES)
                if scan.incomplete:
                    return None, "the port directory could not be listed in full"
                if bad:
                    return None, f"the port directory holds a link or a special file ({bad[0]})"
                if len(found) > SNAPSHOT_MAX_FILES:
                    return None, f"the port directory holds more than {SNAPSHOT_MAX_FILES} files"
                files, total = [], 0
                for parts in sorted(found):
                    data = safeio.read_beneath(port.fd, parts, max_bytes=PORT_MAX)
                    total += len(data)
                    if total > SNAPSHOT_TOTAL_MAX:
                        return None, "the port directory is larger than a snapshot may be"
                    files.append(("/".join(parts), data))
            finally:
                port.close()
    except safeio.SafeReadError as exc:
        return None, f"the port directory could not be read ({exc.reason})"
    if not any(rel == "port_model.py" for rel, _ in files):
        return None, "port_model.py is not in the port directory"
    return files, ""


def save_best(stage_dir) -> dict | None:
    """Keep the port directory as it is now in `stages/2.best/<snapshot>/`, where `<snapshot>` is the digest of the file list, and return the record
    for the ledger: {"snapshot", "files": [{"path", "sha256", "bytes"}], "port_sha256"}. None when it could not be kept.

    A snapshot is written once and never replaced: its name is a digest of what it holds, so a new best port goes to a new directory and the
    one an earlier ledger entry names stays as it was. The caller writes the ledger entry after this returns; a stop between the two leaves a
    directory nothing refers to, and the earlier best port is still restorable (review of PR 497)."""
    files, _ = snapshot_port(stage_dir)
    if files is None:
        return None
    listing = [{"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for rel, data in files]
    digest = manifest_digest(listing)
    try:
        with safeio.pinned_root(Path(stage_dir).parent) as pin:
            for rel, data in files:
                safeio.write_beneath(pin.fd, (BEST_DIR, digest, *rel.split("/")), data, mode=0o644)
    except safeio.SafeReadError:
        return None
    port_sha = next(item["sha256"] for item in listing if item["path"] == "port_model.py")
    return {"snapshot": digest, "files": listing, "port_sha256": port_sha}


def _exists(pin, parts) -> bool:
    """Whether anything (even a link or a directory) stands at `parts` beneath the pin."""
    try:
        safeio.read_beneath(pin.fd, parts, max_bytes=PORT_MAX)
        return True
    except safeio.ReadFailed as exc:
        return not ("(ENOENT)" in exc.reason or "(ENOTDIR)" in exc.reason)
    except safeio.SafeReadError:
        return True


def restore_best(stage_dir, best: dict, skeleton: bytes) -> tuple[bool, str]:
    """Put the best port's snapshot in place of the unchanged skeleton. (placed, why). The ledger entry `best` names the snapshot and lists its
    files with their hashes; every file of the directory it names must match, and the list must hash to the name. The port directory must still
    hold the skeleton byte for byte and no other file of the snapshot's list: a file the coder or the helper wrote is never replaced. Reads
    and writes go beneath pinned directories, so a link swapped in is refused. The supporting files are written before `port_model.py`, so a
    failure part-way leaves the skeleton as the entry module and the next attempt can try again."""
    snapshot, listing = best.get("snapshot"), best.get("files")
    if not (isinstance(snapshot, str) and len(snapshot) == 64 and isinstance(listing, list) and listing):
        return False, "the ledger entry names no snapshot of the best port"
    try:
        files = [(str(item["path"]), str(item["sha256"])) for item in listing]
    except (KeyError, TypeError):
        return False, "the ledger entry's file list is malformed"
    if manifest_digest(listing) != snapshot:
        return False, "the ledger entry's file list does not match the snapshot's name"
    if len(files) > SNAPSHOT_MAX_FILES or not any(rel == "port_model.py" for rel, _ in files):
        return False, "the ledger entry's file list is not a port"
    current, why = read_port(stage_dir)
    if current is None:
        return False, why
    if current.rstrip(b"\n") != skeleton.rstrip(b"\n"):
        return False, "port_model.py is not the unchanged skeleton"
    blobs: list[tuple[str, bytes]] = []
    try:
        with safeio.pinned_root(Path(stage_dir).parent) as pin:
            for rel, want in files:
                data = safeio.read_beneath(pin.fd, (BEST_DIR, snapshot, *rel.split("/")), max_bytes=PORT_MAX)
                if hashlib.sha256(data).hexdigest() != want:
                    return False, f"the best port's copy of {rel} does not match the hash the ledger recorded"
                blobs.append((rel, data))
    except safeio.SafeReadError as exc:
        return False, f"the best port's copy is missing, is a link, or could not be read ({exc.reason})"
    try:
        with safeio.pinned_root(stage_dir) as pin:
            for rel, _ in blobs:
                if rel != "port_model.py" and _exists(pin, ("port", *rel.split("/"))):
                    return False, f"port/{rel} already exists, and a file written in the attempt is not replaced"
            for rel, data in sorted(blobs, key=lambda item: item[0] == "port_model.py"):
                safeio.write_beneath(pin.fd, ("port", *rel.split("/")), data, mode=0o644)
    except safeio.SafeReadError as exc:
        return False, f"the port could not be written ({exc.reason})"
    return True, ""
