# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the web page shows about the cans (orchard/can.py, README 5.9i, docs/superpowers/specs/2026-11-02-canning-tab-design.md).

Read only. A can is a spec file (config/can.*.toml, read with `can.load_spec`), a staged bundle directory and a Hugging Face repository.
This module reads the specs and the bundles and returns plain facts for `GET /api/cans`. It returns no path, writes no file, runs no
subprocess, opens no chip and sends no request (so it cannot say anything about the Hub except the repository id and a link).

What is read, and how:
  * the specs: each `can.<slug>.toml` in the specs folder, with the can module's own loader. A spec the loader refuses is a row of its own.
  * the staged bundle `<cans folder>/<spec name>`: opened beneath a pinned descriptor, so a link anywhere is refused and never followed.
    SHA256SUMS is recomputed here with bounds, and README.md is read for the card rows. The gate (`can.gate`) is NOT run: it walks and hashes
    with no total limit, compiles every Python file, imports tt-model's reader and follows links. Its result is read from the optional
    `<name>.can-state.json` that `tt-orchard can gate --state-dir` writes (`can.write_state`).
  * the recorded state: `<cans folder>/<name>.can-state.json`, checked field by field with `can.check_state`. A record carries the digest of
    SHA256SUMS at the time; when the bundle's digest differs, the page says the record is for an earlier bundle.

A missing or broken piece is shown as that piece, in words. Nothing here raises for bad input: a file that cannot be read is a state of that
piece. The whole answer is rebuilt at most once every CACHE_S seconds, one rebuild at a time.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
import time
from pathlib import Path

from orchard import can, safeio

CACHE_S = 30.0
MAX_SPECS = 60                       # specs listed; the rest are counted
MAX_SPEC_ENTRIES = 2000              # directory entries taken while listing the specs folder
SUMS_MAX_BYTES = 1024 * 1024
SUMS_MAX_LINES = 5000
HASH_MAX_BYTES = 1 << 30             # hashed per bundle
SCAN_MAX_ENTRIES = 5000
SCAN_MAX_DEPTH = 8
CARD_MAX_BYTES = 256 * 1024
CARD_MAX_ROWS = 12
CARD_MAX_LINES = 12
CELL_MAX = 200
PROBLEMS_SHOWN = 10
NAME_MAX = 120
SPEC_FILE = re.compile(r"^can\.([A-Za-z0-9][A-Za-z0-9._-]{0,80})\.toml$")
HUB_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")      # every control character, newline and tab included: one name is one line
_SUMS_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)

HUB_NOT_CHECKED = "not checked here"
HUB_NOTE = ("The page makes no request to the Hub, so it cannot say whether the pinned revision exists or whether a published copy "
            "matches. It shows the weights repository and a link.")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _piece(state: str, tone: str, text: str, **more) -> dict:
    return {"state": state, "tone": tone, "text": text, **more}


class CansView:
    """Builds the answer of `GET /api/cans`. `specs_dir` holds the can.*.toml files, `base` is the directory their paths are relative to (the
    checkout), `cans_dir` holds the staged bundles and the optional state files. Every string that came from a file goes through `redact`."""

    def __init__(self, specs_dir, base, cans_dir, *, redact=lambda t: t, clock=time.time):
        self.specs_dir, self.base, self.cans_dir = Path(specs_dir), Path(base), Path(cans_dir)
        self.redact, self.clock = redact, clock
        self._lock = threading.Lock()
        self._cache = None                 # (time, answer)

    # ---- text that came from a file ----
    def _clean(self, text, limit=CELL_MAX) -> str:
        return self.redact(_CONTROL.sub(" ", str(text)))[:limit]

    # ---- the specs ----
    def _spec_files(self) -> tuple[list[tuple[str, Path]], int, bool]:
        """(slug, path) for the first MAX_SPECS spec files by name, how many more there are, and whether the folder could not be listed."""
        found, taken = [], 0
        try:
            with os.scandir(self.specs_dir) as listing:
                for entry in listing:
                    taken += 1
                    if taken > MAX_SPEC_ENTRIES:
                        break
                    m = SPEC_FILE.match(entry.name)
                    try:
                        if m and entry.is_file(follow_symlinks=False):
                            found.append((m.group(1), Path(entry.path)))
                    except OSError:
                        continue
        except OSError:
            return [], 0, True
        found.sort()
        return found[:MAX_SPECS], max(0, len(found) - MAX_SPECS), False

    # ---- the recorded state ----
    def _state(self, cans_pin, name: str) -> tuple[dict | None, str | None]:
        """(the checked state file, None), (None, None) when there is no file, or (None, why) when the file cannot be used."""
        if cans_pin is None:
            return None, None
        try:
            raw = safeio.read_json(f"{name}{can.STATE_SUFFIX}", anchor=cans_pin.fd, max_bytes=can.STATE_MAX_BYTES, expect=dict)
            state = can.check_state(raw)
            if state["name"] != name:
                raise can.CanError("state: the record is for another can")
            return state, None
        except safeio.ReadFailed as exc:
            if "ENOENT" in exc.reason:
                return None, None
            return None, "the recorded state could not be read"
        except safeio.SafeReadError:
            return None, "the recorded state is not a regular JSON file the page accepts"
        except can.CanError:
            return None, "the recorded state has a field that is not valid"

    # ---- the bundle ----
    def _open_bundle(self, cans_pin, name: str):
        """(pin, None) for the bundle directory, or (None, (state, text)) for a bundle that is not there or is not read."""
        if cans_pin is None:
            return None, ("not-staged", "no staged bundle")
        try:
            st = os.stat(name, dir_fd=cans_pin.fd, follow_symlinks=False)
        except FileNotFoundError:
            return None, ("not-staged", "no staged bundle")
        except OSError:
            return None, ("unreadable", "the bundle folder cannot be examined")
        if stat.S_ISLNK(st.st_mode):
            return None, ("link", "the bundle is a link and was not read")
        if not stat.S_ISDIR(st.st_mode):
            return None, ("not-a-directory", "the bundle is not a directory")
        try:
            return cans_pin.child((name,)), None
        except safeio.SafeReadError:
            return None, ("unreadable", "the bundle folder cannot be opened")

    def _hash_file(self, pin, parts: list[str], budget: list) -> tuple[str, str | None]:
        """("ok", digest), ("skipped", None) when the hash budget is spent, or ("bad", reason). Opened beneath the pin, no link followed, no wait."""
        try:
            fd = safeio.open_beneath(pin.fd, parts, file_flags=_FILE_FLAGS)
        except safeio.SafeReadError:
            return "bad", "cannot be opened without following a link, or its path is not allowed"
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                return "bad", "is not a regular file"
            if st.st_size > budget[0]:
                return "skipped", None
            h = hashlib.sha256()
            left = st.st_size
            while left > 0:
                block = os.read(fd, min(1 << 20, left))
                if not block:
                    break
                h.update(block)
                left -= len(block)
            budget[0] -= st.st_size
            return "ok", h.hexdigest()
        except OSError:
            return "bad", "cannot be read"
        finally:
            os.close(fd)

    def _sums(self, pin) -> tuple[dict, str | None]:
        """The SHA256SUMS check of a bundle and the digest of the SHA256SUMS file (None when it was not read)."""
        try:
            data = safeio.read_bytes(can.SUMS_FILE, max_bytes=SUMS_MAX_BYTES, anchor=pin.fd)
        except safeio.TooLarge:
            return _piece("unreadable", "warn", f"{can.SUMS_FILE} is larger than 1 MB, so it was not read"), None
        except safeio.ReadFailed as exc:
            if "ENOENT" in exc.reason:
                return _piece("missing", "warn", f"{can.SUMS_FILE} is missing"), None
            return _piece("unreadable", "warn", f"{can.SUMS_FILE} cannot be read"), None
        except safeio.SafeReadError:
            return _piece("unreadable", "warn", f"{can.SUMS_FILE} is not a regular file, so it was not read"), None
        digest = hashlib.sha256(data).hexdigest()
        lines = data.decode("utf-8", errors="replace").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if len(lines) > SUMS_MAX_LINES:
            return _piece("unreadable", "warn", f"{can.SUMS_FILE} lists more than {SUMS_MAX_LINES} files, too many to check here"), digest
        problems: list[str] = []
        listed: dict[str, str] = {}
        for n, line in enumerate(lines, 1):
            m = _SUMS_LINE.match(line)
            if not m:
                problems.append(f"{can.SUMS_FILE} line {n} is not a digest and a path")
            elif m.group(2) in listed:
                problems.append(f"{can.SUMS_FILE} lists {self._clean(m.group(2), NAME_MAX)} twice")
            else:
                listed[m.group(2)] = m.group(1)
        on_disk: set[str] = set()
        counts = {"links": 0}

        def visit(entry):
            rel = "/".join(entry.parts)
            if entry.kind == "file":
                on_disk.add(rel)
            elif entry.kind in ("link", "other"):
                counts["links"] += 1
                problems.append(f"{self._clean(rel, NAME_MAX)} is {'a link' if entry.kind == 'link' else 'not a regular file or directory'}")
        walk = safeio.scan_beneath(pin, visit, max_entries=SCAN_MAX_ENTRIES, max_depth=SCAN_MAX_DEPTH)
        budget, hashed, skipped = [HASH_MAX_BYTES], 0, 0
        for rel, want in sorted(listed.items()):
            shown = self._clean(rel, NAME_MAX)
            if rel not in on_disk and not walk.incomplete:
                problems.append(f"{can.SUMS_FILE} lists {shown}, which is not in the bundle")
                continue
            outcome, got = self._hash_file(pin, rel.split("/"), budget)
            if outcome == "skipped":
                skipped += 1
            elif outcome == "bad":
                problems.append(f"{shown} {got}" if rel in on_disk or walk.incomplete else f"{can.SUMS_FILE} lists {shown}, which is not in the bundle")
            else:
                hashed += 1
                if got != want:
                    problems.append(f"{shown} changed after {can.SUMS_FILE} was written")
        if not walk.incomplete:
            problems += [f"{can.SUMS_FILE} does not list {self._clean(rel, NAME_MAX)}" for rel in sorted(on_disk - set(listed) - {can.SUMS_FILE})]
        partial = skipped or walk.incomplete
        why = []
        if skipped:
            why.append(f"{_plural(skipped, 'file')} not hashed because the {HASH_MAX_BYTES >> 30} GiB limit was reached")
        if walk.incomplete:
            why.append("the bundle listing was cut short")
        if problems:
            text = f"differs: {_plural(len(problems), 'problem')}"
            if partial:
                text += f" (partly checked: {'; '.join(why)})"
            return _piece("differs", "bad", text, problems=problems[:PROBLEMS_SHOWN], problem_count=len(problems), files=len(listed)), digest
        if partial:
            return _piece("partial", "warn", f"partly checked, no difference found so far: {'; '.join(why)}", problems=[], problem_count=0,
                          files=len(listed)), digest
        return _piece("ok", "good", f"all {_plural(hashed, 'file')} match", problems=[], problem_count=0, files=len(listed)), digest

    # ---- the card ----
    def _card(self, pin, profile) -> dict:
        try:
            raw = safeio.read_bytes("README.md", max_bytes=CARD_MAX_BYTES, anchor=pin.fd)
        except safeio.TooLarge:
            return _piece("unreadable", "warn", "the card (README.md) is larger than 256 KB, so it was not read", performance=None, limitations=[])
        except safeio.ReadFailed as exc:
            word = "is missing" if "ENOENT" in exc.reason else "cannot be read"
            return _piece("missing", "warn", f"the card (README.md) {word}", performance=None, limitations=[])
        except safeio.SafeReadError:
            return _piece("unreadable", "warn", "the card (README.md) is not a regular file, so it was not read", performance=None, limitations=[])
        text = raw.decode("utf-8", errors="replace")
        sections = can._sections(text)
        perf = None
        rows = [can._cells(line) for line in sections.get("Expected performance", []) if can._is_row(line)]
        rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
        if rows:
            perf = {"header": [self._clean(c) for c in rows[0]], "rows": [[self._clean(c) for c in r] for r in rows[1:1 + CARD_MAX_ROWS]],
                    "more": max(0, len(rows) - 1 - CARD_MAX_ROWS)}
        limits = [self._clean(line.strip()) for line in sections.get("Limitations", []) if line.strip()]
        problems = can.card_problems(text, profile)
        if problems:
            return _piece("problems", "bad", "the card lacks what the profile asks for: " + "; ".join(self._clean(p, 160) for p in problems[:3]),
                          performance=perf, limitations=limits[:CARD_MAX_LINES], limitations_more=max(0, len(limits) - CARD_MAX_LINES))
        if profile is None:
            return _piece("not-required", "dim", "the spec sets no card profile, so the card is not checked", performance=perf,
                          limitations=limits[:CARD_MAX_LINES], limitations_more=max(0, len(limits) - CARD_MAX_LINES))
        return _piece("ok", "good", f"the card has the sections the {profile} profile asks for", performance=perf,
                      limitations=limits[:CARD_MAX_LINES], limitations_more=max(0, len(limits) - CARD_MAX_LINES))

    # ---- records against the bundle ----
    def _gate(self, state, state_problem, digest) -> dict:
        if state_problem:
            return _piece("unreadable", "warn", state_problem)
        part = (state or {}).get("gate")
        if part is None:
            return _piece("none", "warn", "no gate result recorded (run tt-orchard can gate with --state-dir)")
        if digest is None or part["bundle_sums_sha256"] != digest:
            return _piece("stale", "warn", f"a gate result from {part['at']} is for an earlier bundle ("
                                           f"{'passed' if part['passed'] else 'failed'} then)", at=part["at"])
        if part["passed"]:
            return _piece("passed", "good", f"passed (recorded {part['at']})", at=part["at"], problems=[], problem_count=0)
        n = part["problem_count"]
        return _piece("failed", "bad", f"failed with {_plural(n, 'problem')} (recorded {part['at']})", at=part["at"], problem_count=n,
                      problems=[self._clean(p, 300) for p in part["problems"]])

    def _boot(self, state, state_problem, digest) -> dict:
        empty = {"decode_tok_s": None, "ready_s": None, "cache_state": None, "failed": []}
        if state_problem:
            return _piece("unreadable", "warn", state_problem, **empty)
        part = (state or {}).get("boot")
        if part is None:
            return _piece("none", "warn", "no boot recorded (a boot check needs a chip and a lease; run tt-orchard can boot with --state-dir)", **empty)
        facts = {"decode_tok_s": part["decode_tok_s"], "ready_s": part["ready_s"], "cache_state": part["cache_state"],
                 "failed": [self._clean(f, 300) for f in part["failed"]], "at": part["at"]}
        if digest is None or part["bundle_sums_sha256"] != digest:
            return _piece("stale", "warn", f"a boot from {part['at']} is for an earlier bundle ({'passed' if part['passed'] else 'failed'} then)", **facts)
        bits = []
        if part["decode_tok_s"] is not None:
            bits.append(f"decode {part['decode_tok_s']:g} tok/s")
        if part["ready_s"] is not None:
            bits.append(f"ready in {part['ready_s']:g} s")
        bits.append({"empty": "cache empty", "warm": "cache warm"}.get(part["cache_state"], "cache state not recorded"))
        if part["passed"]:
            return _piece("passed", "good", "passed: " + ", ".join(bits) + f" (recorded {part['at']})", **facts)
        return _piece("failed", "bad", "failed" + (f" ({', '.join(facts['failed'][:3])})" if facts["failed"] else "") + f" (recorded {part['at']})", **facts)

    # ---- one can ----
    def _row_for_spec(self, slug: str, path: Path, cans_pin) -> dict:
        try:
            spec = can.load_spec(path, base=self.base)
        except can.CanError as exc:
            msg = str(exc).replace(str(self.base), "<checkout>").replace(str(self.specs_dir), "<specs>")
            return {"name": self._clean(slug, NAME_MAX), "slug": self._clean(slug, NAME_MAX), "kind": None, "mesh": None, "device_count": None,
                    "weights": None, "card_profile": None, "tone": "bad", "summary": "spec refused",
                    "spec": _piece("refused", "bad", "spec refused: " + self._clean(msg, 300)),
                    "bundle": _piece("unknown", "dim", "not looked up while the spec is refused"),
                    "sums": _piece("none", "dim", "not checked"), "card": _piece("none", "dim", "not read", performance=None, limitations=[]),
                    "gate": _piece("none", "dim", "not looked up"), "boot": _piece("none", "dim", "not looked up", decode_tok_s=None,
                                                                                   ready_s=None, cache_state=None, failed=[])}
        name = spec.name
        pin, why = self._open_bundle(cans_pin, name)
        state, state_problem = self._state(cans_pin, name)
        digest = None
        try:
            if pin is None:
                bundle = _piece(why[0], "dim" if why[0] == "not-staged" else "warn", why[1])
                sums = _piece("none", "dim", "no bundle to check")
                card = _piece("none", "dim", "no bundle to read", performance=None, limitations=[])
            else:
                bundle = _piece("staged", "good", "staged")
                sums, digest = self._sums(pin)
                card = self._card(pin, spec.card_profile)
        finally:
            if pin is not None:
                pin.close()
        gate, boot = self._gate(state, state_problem, digest), self._boot(state, state_problem, digest)
        pieces = (bundle, sums, card, gate, boot)
        if bundle["state"] == "not-staged":
            tone, summary = "dim", "not staged"
        elif any(p["tone"] == "bad" for p in pieces):
            tone, summary = "bad", "failing"
        elif bundle["state"] == "staged" and sums["state"] == "ok" and gate["state"] == "passed" and boot["state"] == "passed" \
                and card["state"] in ("ok", "not-required"):
            tone, summary = "good", "booted"
        else:
            tone, summary = "warn", "incomplete"
        repo = spec.repo
        return {"name": self._clean(name, NAME_MAX), "slug": self._clean(slug, NAME_MAX), "kind": spec.kind, "mesh": self._clean(spec.mesh, 40),
                "device_count": spec.device_count,
                "weights": {"repo": self._clean(repo, 200), "revision": spec.revision,
                            "hub_url": f"https://huggingface.co/{repo}" if HUB_REPO.match(repo) else None,
                            "hub_text": HUB_NOT_CHECKED},
                "card_profile": spec.card_profile, "tone": tone, "summary": summary,
                "spec": _piece("ok", "good", "spec loaded"), "bundle": bundle, "sums": sums, "card": card, "gate": gate, "boot": boot}

    # ---- the answer ----
    def _build(self) -> dict:
        files, more, specs_unreadable = self._spec_files()
        pin_ctx = None
        folder = _piece("present", "good", "the cans folder is present")
        try:
            pin_ctx = safeio.pinned_root(self.cans_dir)
            cans_pin = pin_ctx.__enter__()
        except safeio.SafeReadError as exc:
            cans_pin = None
            if "ENOENT" in exc.reason:
                folder = _piece("missing", "warn", "the cans folder does not exist yet; set ORCHARD_CANS_DIR or stage into <runs folder>/cans")
            else:
                folder = _piece("unreadable", "warn", "the cans folder cannot be opened (a link, or no permission)")
        try:
            rows = [self._row_for_spec(slug, path, cans_pin) for slug, path in files]
        finally:
            if pin_ctx is not None:
                pin_ctx.__exit__(None, None, None)
        return {"label": "cans", "read_only": True, "cans": rows, "folder": folder,
                "specs": {"listed": len(rows), "more": more, "folder_unreadable": specs_unreadable},
                "hub": {"text": HUB_NOT_CHECKED, "note": HUB_NOTE}}

    def build(self) -> dict:
        with self._lock:
            if self._cache is not None and self.clock() - self._cache[0] < CACHE_S:
                return self._cache[1]
            answer = self._build()
            self._cache = (self.clock(), answer)           # dated when the rebuild ends, so a slow one is still reused for the full lifetime
            return answer
