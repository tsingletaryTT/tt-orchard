# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard db migrate telegraph`: move a telegraph directory's `telegraph.jsonl` ledger and `cache/*.json` index into the
project database, and cut the old files over so no old writer can start a second history.

This module and its parser are the only code left that reads the jsonl format: the importer, and the reader of an archived file
for `tt-orchard db export telegraph --from-archive`.

The order of steps follows the design note (docs/superpowers/specs/2026-10-28-project-database-design.md, I3 and I7 to I10).
The old lock is held from step 1 to step 7, so an old writer cannot append meanwhile.

  1  take the old `telegraph.lock`                                   nothing changed
  2  write `MIGRATING-TO-SQLITE` (scope, sha256 of the ledger), sync  marker; the old ledger is still live
  3  recover sidecars and a torn tail (I9); ONE transaction: import every record and the cache index, compare the totals,
     set the phase `imported` with the byte count and sha256 of what was imported
  4  rename `telegraph.jsonl` to `telegraph.jsonl.archived`, sync the directory; phase `archived`
  5  make the directory `telegraph.jsonl` with a README (the refusing sentinel); phase `sentinel`
  6  write `MIGRATED-TO-SQLITE`, sync; phase `complete`; remove `MIGRATING-TO-SQLITE`
  7  release the old lock

A rerun resumes from the recorded phase and from what is on disk. It never refuses because an earlier phase finished; it refuses
only when the directory has been migrated completely (`AlreadyMigrated`) or when something a person must decide has appeared (a
ledger whose already-imported prefix changed, a regular file named `telegraph.jsonl` next to the archive, a malformed line).

`hook(step)` is a test seam: it is called after each step with names like "after:2" and "inside:3" and may raise to simulate a
crash at that point.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass, field

from orchard import safeio
from orchard import telegraph as tg
from orchard import telegraph_store as ts
from orchard.db import Database, LedgerRefused, LedgerUnavailable, private_directory

MAX_LEDGER_BYTES = 16 * 1024 * 1024               # the page refuses a larger ledger too; a larger one needs a person to look
MAX_CACHE_BYTES = 4 * 1024 * 1024
MAX_RAW_LINE = 4_000_000                      # the database refuses a longer record; a longer line is reported as malformed
SIDECAR_PREFIX = f"{ts.LEDGER}.torn-"
CACHE_NAME = re.compile(r"^[0-9a-f]{64}\.json$")
README = ("This telegraph ledger moved to the project database (project.db in the parent directory).\n"
          "An old writer that finds this directory refuses to run. See MIGRATED-TO-SQLITE and the archived ledger\n"
          f"{ts.ARCHIVE}. `tt-orchard db export telegraph` writes the old lines back.\n")
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class MigrationRefused(LedgerRefused):
    """The migration will not go on until a person looks."""


class AlreadyMigrated(MigrationRefused):
    """This directory was migrated completely."""


class MigrationMismatch(MigrationRefused):
    """The database does not hold what the file holds. Nothing was changed."""


class MalformedLedger(MigrationRefused):
    """The ledger has lines that are not records. `lines` are their line numbers."""

    def __init__(self, message: str, lines: list[int]):
        super().__init__(message)
        self.lines = lines


# ---------------------------------------------------------------------------------------------------------- parsing

@dataclass
class Parsed:
    items: list = field(default_factory=list)          # [(record, raw line)] in order
    malformed: list = field(default_factory=list)      # 1-based line numbers of lines that are not records


def parse_jsonl(data: bytes) -> Parsed:
    """The records of a jsonl ledger exactly as the old reader saw them: the text is decoded with replacement, split with
    `splitlines`, and a line that is JSON and an object is a record. A blank line is neither a record nor malformed; any other
    line is `malformed` (and the old reader dropped it silently; the migration therefore refuses unless a person accepts)."""
    out = Parsed()
    for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        if len(line) > MAX_RAW_LINE:
            out.malformed.append(number)
            continue
        try:
            item = json.loads(line)
        except (ValueError, RecursionError):
            out.malformed.append(number)
            continue
        if isinstance(item, dict):
            out.items.append((item, line))
        else:
            out.malformed.append(number)
    return out


def read_archive_lines(directory) -> list[str]:
    """The record lines of the archived ledger (`telegraph.jsonl.archived`), for `db export --from-archive`."""
    with safeio.pinned_root(directory) as pin:
        data = safeio.read_bytes(ts.ARCHIVE, max_bytes=MAX_LEDGER_BYTES, anchor=pin.fd)
    return [raw for _rec, raw in parse_jsonl(data).items]


# ---------------------------------------------------------------------------------------------------------- the old files

def _read_ledger_bytes(dir_fd: int) -> bytes:
    try:
        return safeio.read_bytes(ts.LEDGER, max_bytes=MAX_LEDGER_BYTES, anchor=dir_fd)
    except safeio.SafeReadError as exc:
        raise MigrationRefused(f"cannot read {ts.LEDGER}: {exc}") from exc


def _ledger_data(db: Database) -> bytes:
    """The old ledger's bytes, or b'' when there is no such file (a missing file is an empty ledger)."""
    kind = db.exists_name(ts.LEDGER)
    if kind is None:
        return b""
    if kind != "file":
        raise MigrationRefused(f"{ts.LEDGER} is a {kind} and cannot be a ledger file")
    return _read_ledger_bytes(db._dirfd)


MAX_DIRECTORY_ENTRIES = 200_000


def _listdir(fd: int) -> list[str]:
    """The names in a directory, refusing a directory of more than MAX_DIRECTORY_ENTRIES (a hostile or runaway directory)."""
    names = os.listdir(fd)
    if len(names) > MAX_DIRECTORY_ENTRIES:
        raise MigrationRefused(f"a directory has more than {MAX_DIRECTORY_ENTRIES} entries; a person has to look at it")
    return names


def _sidecars(dir_fd: int) -> list[str]:
    names = []
    for name in sorted(_listdir(dir_fd)):
        if name.startswith(SIDECAR_PREFIX):
            try:
                st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                names.append(name)
    return names


def db_exists(dir_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _same_content(dir_fd: int, name: str, expected: bytes) -> bool:
    try:
        return safeio.read_bytes(name, max_bytes=len(expected) + 1, anchor=dir_fd) == expected
    except safeio.SafeReadError:
        return False


def recover_ledger(dir_fd: int, max_output_tokens: int) -> dict:
    """I9, under the old lock: the old recovery, so that the import sees every conservative charge the old code would have counted.

    A last line with no line break is a write that was cut short: it moves to a sidecar (written and synced first), the ledger is cut
    back to its last whole line, and then every sidecar that has no `torn` note gets one (tokens = `max_output_tokens`). Safe to
    run again after a crash at any point: the sidecar is complete before the cut, and a note is written for every sidecar that lacks one."""
    moved = notes = 0
    try:
        fd = os.open(ts.LEDGER, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | _O_CLOEXEC, dir_fd=dir_fd)
    except FileNotFoundError:
        fd = -1
    except OSError as exc:
        raise MigrationRefused(f"cannot open {ts.LEDGER}: {exc.strerror}") from exc
    try:
        data = b""
        if fd >= 0:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise MigrationRefused(f"{ts.LEDGER} is not a regular file")
            data = _read_ledger_bytes(dir_fd)
            if data and not data.endswith(b"\n"):
                cut = data.rfind(b"\n") + 1
                fragment = data[cut:]
                # The sidecar's name comes from the digest of the whole ledger it was cut from. A crash between publishing the sidecar
                # and cutting the ledger leaves a complete sidecar of this very ledger, and a rerun finds the same name and reuses it. An
                # older orphan sidecar has another name, so identical bytes in it are never taken for this tail.
                name = f"{SIDECAR_PREFIX}{hashlib.sha256(data).hexdigest()[:12]}"
                if db_exists(dir_fd, name):
                    if not _same_content(dir_fd, name, fragment):
                        raise MigrationRefused(f"{name} exists with other content than the tail of {ts.LEDGER}; a person has to look at it")
                else:
                    tmp = f".torn-tmp-{uuid.uuid4().hex[:12]}"        # not a countable name until it is whole and synced
                    sfd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _O_CLOEXEC, 0o600, dir_fd=dir_fd)
                    try:
                        try:
                            safeio._write_all(sfd, fragment)
                            os.fsync(sfd)
                        finally:
                            os.close(sfd)
                        os.rename(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                    except BaseException:
                        try:
                            os.unlink(tmp, dir_fd=dir_fd)
                        except OSError:
                            pass
                        raise
                    os.fsync(dir_fd)
                os.ftruncate(fd, cut)                  # only after the whole fragment is safe in the sidecar
                os.fsync(fd)
                data = data[:cut]
                moved = 1
        noted = {rec.get("torn") for rec, _raw in parse_jsonl(data).items if "torn" in rec}
        missing = [n for n in _sidecars(dir_fd) if n not in noted]
        if missing:
            if fd < 0:                                                  # no ledger yet: the notes need one
                fd = os.open(ts.LEDGER, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | _O_CLOEXEC, 0o600, dir_fd=dir_fd)
            else:
                os.close(fd)
                fd = os.open(ts.LEDGER, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | _O_CLOEXEC, dir_fd=dir_fd)
            for name in missing:
                safeio.append_line(fd, json.dumps({"torn": name, "tokens": int(max_output_tokens)}, sort_keys=True))
                notes += 1
            os.fsync(dir_fd)
    except OSError as exc:
        raise MigrationRefused(f"cannot recover {ts.LEDGER}: {exc.strerror or type(exc).__name__}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    return {"fragments_moved": moved, "notes_added": notes}


def _cache_items(dir_fd: int) -> tuple[list, int]:
    """([(identity, entry)], skipped): the entries of `cache/*.json` that hold an answer and a reply file name."""
    try:
        cfd = os.open("cache", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_CLOEXEC, dir_fd=dir_fd)
    except OSError:
        return [], 0                                   # no cache directory, or a link: nothing to import
    items, skipped = [], 0
    try:
        for name in sorted(_listdir(cfd)):
            if not CACHE_NAME.match(name):
                skipped += 1
                continue
            try:
                entry = safeio.read_json(name, max_bytes=MAX_CACHE_BYTES, anchor=cfd, expect=dict)
            except safeio.SafeReadError:
                skipped += 1
                continue
            if isinstance(entry.get("text"), str) and isinstance(entry.get("reply_file"), str) and entry["reply_file"]:
                items.append((name[:-5], entry))
            else:
                skipped += 1
    finally:
        os.close(cfd)
    return items, skipped


# ---------------------------------------------------------------------------------------------------------- the check

def _shape(records: list[dict]) -> dict:
    """The numbers that must be equal before and after: the summary and the accounting, with every per-problem row. They are
    compared exactly: both sides add in ledger order, so a difference is a difference."""
    summary = tg.summarise_records(records)
    acct = tg.accounting(records)
    return {"summary": {k: v for k, v in summary.items()},
            "accounting": {"calls": acct["calls"], "tokens": acct["tokens"], "cost": acct["cost"], "unknown_cost": acct["unknown_cost"],
                           "problems": {str(k): v for k, v in sorted(acct["problems"].items(), key=lambda kv: str(kv[0]))}}}


def _canon(value) -> str:
    """A text form of a nested value in which every key is text and NaN equals NaN, so two shapes can be compared exactly."""
    def norm(v):
        if isinstance(v, dict):
            return {(k if isinstance(k, str) else repr(k)): norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v
    return json.dumps(norm(value), sort_keys=True)


def compare(file_items: list, db_lines: list[str], db_records: list[dict], sidecars: list[str], cache_expected: int,
            cache_rows: int) -> dict:
    """Facts about the equality of the file and the database, and whether all of them hold."""
    file_records = [rec for rec, _raw in file_items]
    want, got = _shape(file_records), _shape(db_records)
    same_totals = _canon(want) == _canon(got)
    torn_names = [r.get("torn") for r in file_records if "torn" in r]
    checks = {
        "lines": [raw for _rec, raw in file_items] == db_lines,
        "totals": same_totals,
        "sidecars": len(sidecars) == len(set(torn_names)) == len(torn_names) and set(sidecars) == set(torn_names),
        "cache": cache_expected == cache_rows,
    }
    return {"equal": all(checks.values()), "checks": checks, "file": want, "database": got,
            "records": len(file_items), "torn_records": len(torn_names), "sidecars": len(sidecars)}


def _brief(result: dict) -> dict:
    """The totals a person reads: calls answered, tokens in and out, cost, and the problems."""
    s, a = result["file"]["summary"], result["file"]["accounting"]
    return {"calls": s["calls"], "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"], "cost_usd": s["cost_usd"],
            "accounted_calls": a["calls"], "accounted_tokens": a["tokens"], "unknown_cost": a["unknown_cost"],
            "problems": len(a["problems"]), "records": result["records"]}


# ---------------------------------------------------------------------------------------------------------- the command

def _noop(step: str) -> None:
    pass


def _take_old_lock(dir_fd: int, timeout_s: float = 30.0) -> int:
    """Step 1: the lock the old code takes. Waits for another holder (a process that is still running)."""
    try:
        fd = os.open(ts.LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | _O_CLOEXEC, 0o600, dir_fd=dir_fd)
    except OSError as exc:
        raise MigrationRefused(f"cannot open {ts.LOCK}: {exc.strerror}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise MigrationRefused(f"{ts.LOCK} is not a regular file of this user")
        os.fchmod(fd, 0o600)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except (BlockingIOError, PermissionError):
                if time.monotonic() >= deadline:
                    raise MigrationRefused(f"{ts.LOCK} is held by a running process; stop the harness and try again") from None
                time.sleep(0.05)
    except BaseException:
        os.close(fd)
        raise


def migrate(directory, *, scope: str = ts.DEFAULT_SCOPE, max_output_tokens: int = 1500, accept_malformed: int | None = None,
            dry_run: bool = False, hook=None) -> dict:
    """Migrate (or resume migrating) a telegraph directory. Returns a dict of what was done and the totals before and after.
    `dry_run` reads the directory and checks that an import would give equal totals; it writes nothing."""
    hook = hook or _noop
    if dry_run:
        return _dry_run(directory, scope=scope, max_output_tokens=max_output_tokens, accept_malformed=accept_malformed)
    if not os.path.isdir(directory):
        raise MigrationRefused(f"{directory} is not a directory")
    dir_fd = private_directory(directory)
    lock = -1
    db = None
    try:
        lock = _take_old_lock(dir_fd)                                                   # step 1
        hook("after:1")
        db = Database(directory)
        return _run(db, dir_fd, scope, max_output_tokens, accept_malformed, hook)
    finally:
        if db is not None:
            db.close()
        if lock >= 0:
            try:
                fcntl.flock(lock, fcntl.LOCK_UN)                                        # step 7
            except OSError:
                pass
            os.close(lock)
        os.close(dir_fd)


def _marker_body(scope: str, sha: str) -> str:
    return json.dumps({"scope": scope, "ledger_sha256": sha, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "pid": os.getpid()}, sort_keys=True) + "\n"


def _set_phase(db: Database, scope: str, phase: str, **extra) -> None:
    with db.write() as conn:
        db.meta_set(conn, ts.cutover_key(scope), phase)
        for k, v in extra.items():
            db.meta_set(conn, f"{k}:{scope}", str(v))


def _run(db: Database, dir_fd: int, scope: str, max_output_tokens: int, accept_malformed, hook) -> dict:
    store = ts.TelegraphStore(db, scope, guard=False)
    directory = db.directory
    phase = db.meta_get(ts.cutover_key(scope))
    migrating = db.exists_name(ts.MARKER_MIGRATING) is not None
    migrated = db.exists_name(ts.MARKER_MIGRATED) is not None
    live = db.exists_name(ts.LEDGER)
    resumed = phase is not None or migrating
    if phase == "complete":
        if migrating:                                                           # crashed between step 6b and 6c
            _finish_marker(db, dir_fd, hook)
            return {"status": "finished", "resumed": True, "phase": "complete"}
        raise AlreadyMigrated(f"{directory} was migrated already")
    if phase is None and migrated:
        raise MigrationRefused(f"{directory} was migrated, but the database does not record it (the file was lost or replaced); "
                               "restore a backup with `tt-orchard db restore` and run this again")
    if live in ("link", "other") or (live == "dir" and phase is None):
        raise MigrationRefused(f"{ts.LEDGER} is a {live}; it has to be a ledger file (or absent) before the migration starts")
    result: dict = {"status": "migrated", "resumed": resumed, "scope": scope}
    fresh = phase is None and not migrating
    if phase is None:
        data = _ledger_data(db)
        if fresh:
            _refuse_malformed(parse_jsonl(_without_fragment(data)), accept_malformed)      # before anything is written
            try:
                safeio.atomic_write(os.path.join(directory, ts.MARKER_MIGRATING),                # step 2
                                    _marker_body(scope, hashlib.sha256(data).hexdigest()), mode=0o600)
            except OSError as exc:
                raise MigrationRefused(f"cannot write {ts.MARKER_MIGRATING}: {exc.strerror}") from exc
        hook("after:2")
    if phase is None or (phase == "imported" and live == "file"):
        try:
            result.update(_step3(db, store, dir_fd, scope, max_output_tokens, accept_malformed, phase, hook))      # step 3
        except MigrationRefused:
            if fresh:                                                           # a refused start leaves no marker behind
                try:
                    os.unlink(ts.MARKER_MIGRATING, dir_fd=dir_fd)
                    os.fsync(dir_fd)
                except OSError:
                    pass
            raise
        phase = "imported"
        live = db.exists_name(ts.LEDGER)
    if phase == "imported":
        _step4(db, dir_fd, scope, hook)                                          # step 4
        phase = "archived"
    if phase == "archived":
        _step5(db, dir_fd, scope, hook)                                          # step 5
        phase = "sentinel"
    if phase == "sentinel":
        _step6(db, dir_fd, scope, hook)                                          # step 6
    # After the cutover: a backup of the finished database, and the second equality check against the archive.
    try:
        db.backup()
    except (LedgerUnavailable, LedgerRefused, OSError):
        pass
    result["export_matches_archive"] = _export_matches_archive(db, store, dir_fd)
    result["phase"] = "complete"
    return result


def _without_fragment(data: bytes) -> bytes:
    """The ledger without a last line that has no line break (recovery moves that fragment to a sidecar before the import)."""
    return data[:data.rfind(b"\n") + 1] if data and not data.endswith(b"\n") else data


def _refuse_malformed(parsed: Parsed, accept_malformed) -> None:
    """I9: a line that is not a record, and is not the last line without a line break, cannot be recovered. The old reader skipped it
    silently, so whatever it recorded was never counted. The migration refuses, names the lines, and goes on only when a person passes
    the count they inspected."""
    if parsed.malformed and accept_malformed != len(parsed.malformed):
        shown = ", ".join(str(n) for n in parsed.malformed[:50])
        raise MalformedLedger(
            f"{ts.LEDGER} has {len(parsed.malformed)} line(s) that are not records (line numbers: {shown}"
            f"{', ...' if len(parsed.malformed) > 50 else ''}). The old reader skipped them silently, so their spend was never counted. "
            f"Look at them; if you accept that they are dropped, run again with --accept-malformed {len(parsed.malformed)}",
            parsed.malformed)


def _step3(db, store, dir_fd, scope, max_output_tokens, accept_malformed, phase, hook) -> dict:
    """Recover the old ledger (I9), then import, compare and record the phase in ONE transaction."""
    recovery = recover_ledger(dir_fd, max_output_tokens)
    data = _ledger_data(db)
    parsed = parse_jsonl(data)
    _refuse_malformed(parsed, accept_malformed)
    imported_before = 0
    if phase == "imported":
        want_sha = db.meta_get(f"imported_sha256:{scope}")
        n = int(db.meta_get(f"imported_bytes:{scope}") or "0")
        if len(data) < n or hashlib.sha256(data[:n]).hexdigest() != want_sha:
            raise MigrationRefused("the ledger's imported part changed since it was imported (something edited it or removed lines, beyond adding "
                                   "new ones); a person has to decide what is true. Nothing was changed")
        imported_before = n
    cache_items, cache_skipped = _cache_items(dir_fd)
    sidecars = _sidecars(dir_fd)
    with db.write() as conn:
        if phase == "imported":
            already = store.count_events(conn)
            prefix = parse_jsonl(data[:imported_before]).items
            if already != len(prefix):
                raise MigrationRefused("the database does not hold the records it says it imported; nothing was changed")
            store.import_records(conn, parsed.items[len(prefix):])
        else:
            if store.count_events(conn):
                raise MigrationRefused("the database already holds records for this scope but no migration is recorded; "
                                       "nothing was imported (a second history would double the totals)")
            store.import_records(conn, parsed.items)
        store.import_cache(conn, cache_items)
        cache_rows = store.cache_count_in(conn)
        result = compare(parsed.items, store.lines_in(conn), store.records_in(conn), sidecars, len(cache_items), cache_rows)
        if not result["equal"]:
            bad = ", ".join(k for k, ok in result["checks"].items() if not ok)
            raise MigrationMismatch(f"the database does not equal the ledger ({bad}); nothing was changed. "
                                    f"file {json.dumps(_brief(result))}")
        db.meta_set(conn, ts.cutover_key(scope), "imported")
        for k, v in (("imported_bytes", len(data)), ("imported_sha256", hashlib.sha256(data).hexdigest()),
                     ("imported_records", len(parsed.items))):
            db.meta_set(conn, f"{k}:{scope}", str(v))
        hook("inside:3")
    hook("after:3")
    return {"totals": _brief(result), "recovery": recovery, "malformed_accepted": len(parsed.malformed),
            "cache_imported": len(cache_items), "cache_skipped": cache_skipped, "sidecars": len(sidecars),
            "checks": result["checks"], "ledger_sha256": hashlib.sha256(data).hexdigest()}


def _step4(db, dir_fd, scope, hook) -> None:
    live = db.exists_name(ts.LEDGER)
    archived = db.exists_name(ts.ARCHIVE)
    if live == "file":
        if archived is not None:
            raise MigrationRefused(f"a regular {ts.LEDGER} exists next to {ts.ARCHIVE}: an old writer got through after the "
                                   "archive step. Stop it and reconcile the late records before going on")
        data = _ledger_data(db)
        if hashlib.sha256(data).hexdigest() != db.meta_get(f"imported_sha256:{scope}"):
            raise MigrationRefused("the ledger changed after it was imported; run the migration again to import the change")
        os.rename(ts.LEDGER, ts.ARCHIVE, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        os.fsync(dir_fd)
        hook("inside:4")
    elif live not in (None, "dir"):
        raise MigrationRefused(f"{ts.LEDGER} is a {live}")
    _set_phase(db, scope, "archived")
    hook("after:4")


def _step5(db, dir_fd, scope, hook) -> None:
    live = db.exists_name(ts.LEDGER)
    if live is None:
        try:
            os.mkdir(ts.LEDGER, 0o700, dir_fd=dir_fd)
        except FileExistsError:
            pass
        live = db.exists_name(ts.LEDGER)
    if live != "dir":
        raise MigrationRefused(f"a {live or 'new file'} named {ts.LEDGER} appeared where the sentinel directory goes: an old writer got "
                               "through after the archive step. Stop it and reconcile the late records before going on")
    hook("inside:5")
    sfd = os.open(ts.LEDGER, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_CLOEXEC, dir_fd=dir_fd)
    try:
        rfd = os.open("README.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | _O_CLOEXEC, 0o600, dir_fd=sfd)
        try:
            safeio._write_all(rfd, README.encode("utf-8"))
            os.fsync(rfd)
        finally:
            os.close(rfd)
        os.fsync(sfd)
    finally:
        os.close(sfd)
    os.fsync(dir_fd)
    _set_phase(db, scope, "sentinel")
    hook("after:5")


def _step6(db, dir_fd, scope, hook) -> None:
    archived = db.exists_name(ts.ARCHIVE) is not None
    body = json.dumps({"migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "archive": ts.ARCHIVE if archived else None,
                       "sha256": db.meta_get(f"imported_sha256:{scope}"),
                       "records": int(db.meta_get(f"imported_records:{scope}") or "0"), "scope": scope}, sort_keys=True) + "\n"
    safeio.atomic_write(os.path.join(db.directory, ts.MARKER_MIGRATED), body, mode=0o600)
    hook("after:6a")
    _set_phase(db, scope, "complete")
    hook("after:6b")
    _finish_marker(db, dir_fd, hook)


def _finish_marker(db, dir_fd, hook) -> None:
    try:
        os.unlink(ts.MARKER_MIGRATING, dir_fd=dir_fd)
    except FileNotFoundError:
        pass
    os.fsync(dir_fd)
    hook("after:6c")


def _export_matches_archive(db, store, dir_fd) -> bool | None:
    """The second check: the export of the database equals the archived ledger's record lines, line by line. None when there is
    no archive (the directory had no ledger)."""
    if db.exists_name(ts.ARCHIVE) is None:
        return None
    data = safeio.read_bytes(ts.ARCHIVE, max_bytes=MAX_LEDGER_BYTES, anchor=dir_fd)
    return [raw for _rec, raw in parse_jsonl(data).items] == store.export_lines()


def _dry_run(directory, *, scope, max_output_tokens, accept_malformed) -> dict:
    """Read the directory and import into a throw-away in-memory database. The directory is not changed: no lock file, no marker,
    no recovery. A fragment and a sidecar with no note are counted as the real run would count them."""
    result: dict = {"status": "dry-run", "scope": scope}
    with safeio.pinned_root(directory) as pin:
        names_here = _listdir(pin.fd)
        if ts.MARKER_MIGRATED in names_here:
            raise AlreadyMigrated(f"{directory} was migrated already")
        try:
            data = safeio.read_bytes(ts.LEDGER, max_bytes=MAX_LEDGER_BYTES, anchor=pin.fd) if ts.LEDGER in names_here else b""
        except safeio.SafeReadError as exc:
            raise MigrationRefused(f"cannot read {ts.LEDGER}: {exc}") from exc
        sidecars = _sidecars(pin.fd)
        cache_items, cache_skipped = _cache_items(pin.fd)
        archived = ts.ARCHIVE in names_here
    fragment = bool(data) and not data.endswith(b"\n")
    if fragment:
        data = data[:data.rfind(b"\n") + 1]
    parsed = parse_jsonl(data)
    if parsed.malformed and accept_malformed != len(parsed.malformed):
        raise MalformedLedger(f"{ts.LEDGER} has {len(parsed.malformed)} line(s) that are not records (line numbers: "
                              f"{', '.join(str(n) for n in parsed.malformed[:50])}); a real run needs "
                              f"--accept-malformed {len(parsed.malformed)}", parsed.malformed)
    names = list(sidecars) + ([f"{SIDECAR_PREFIX}(would be made)"] if fragment else [])
    noted = {rec.get("torn") for rec, _raw in parsed.items if "torn" in rec}
    items = list(parsed.items)
    for name in names:
        if name not in noted:
            rec = {"torn": name, "tokens": int(max_output_tokens)}
            items.append((rec, json.dumps(rec, sort_keys=True)))
    mem = Database(None, memory=True)
    try:
        store = ts.TelegraphStore(mem, scope, guard=False)
        with mem.write() as conn:
            store.import_records(conn, items)
            store.import_cache(conn, cache_items)
            rows = store.cache_count_in(conn)
            checked = compare(items, store.lines_in(conn), store.records_in(conn), names, len(cache_items), rows)
    finally:
        mem.close()
    result.update({"equal": checked["equal"], "totals": _brief(checked), "checks": checked["checks"], "fragment_to_move": fragment,
                   "sidecars": len(sidecars), "cache_entries": len(cache_items), "cache_skipped": cache_skipped,
                   "malformed_accepted": len(parsed.malformed), "already_archived": archived})
    return result
