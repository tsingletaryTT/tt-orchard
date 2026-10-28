# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard db status | migrate telegraph | export telegraph | backup | restore | confirm-restore`: the project database
(orchard/db.py) and the telegraph ledger on it (orchard/telegraph_store.py).

  status                     what is in the directory: database, schema version, events, backups, markers, migration phase
  migrate telegraph          import telegraph.jsonl and cache/*.json into the database and cut the old files over
                             (--dry-run checks the import and writes nothing; --accept-malformed N accepts N unreadable lines)
  export telegraph           write the ledger as jsonl on standard output, the lines the old file held
                             (--from-archive reads telegraph.jsonl.archived instead of the database)
  backup                     an online backup into backups/
  restore BACKUP             put a backup in place of the database; the ledger is then unconfirmed
  confirm-restore --note T   a person says the history is accounted for; this is the only thing that lets calls go out again

The state directory is chosen as for `tt-orchard telegraph`: `--dir`, the config's `[telegraph] dir`, `$ORCHARD_TELEGRAPH_DIR`, then
the caller's home. Exit codes: 0 done, 2 a problem (the message says what to do).
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from orchard import db as dbm
from orchard import safeio
from orchard import telegraph as tg
from orchard import telegraph_cmd
from orchard import telegraph_migrate as mig
from orchard import telegraph_store as ts

EXIT_OK, EXIT_PROBLEM = 0, 2


def reply_ids(directory) -> list[str]:
    """The call ids of the reply files (`replies/<call_id>.txt`) in a state directory: regular files only, links never followed."""
    try:
        with safeio.pinned_root(Path(directory) / "replies") as pin:
            names = os.listdir(pin.fd)
            out = []
            for name in sorted(names):
                if not name.endswith(".txt"):
                    continue
                try:
                    if stat.S_ISREG(os.stat(name, dir_fd=pin.fd, follow_symlinks=False).st_mode):
                        out.append(name[:-4])
                except OSError:
                    continue
            return out
    except safeio.SafeReadError:
        return []


def main(args, *, out, environ=None, home=None) -> int:
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)

    def say(text=""):
        print(text, file=out)

    try:
        cfg = telegraph_cmd.load_config(telegraph_cmd._config_path(args, environ, home))
    except Exception:                                         # noqa: BLE001 - these commands work without a telegraph config
        cfg = tg.Config()
    try:
        directory = telegraph_cmd.resolve_directory(getattr(args, "dir", None), cfg, environ, home)
    except tg.ConfigError as exc:
        say(f"db: {exc}")
        return EXIT_PROBLEM
    scope = getattr(args, "scope", None) or ts.DEFAULT_SCOPE
    action = args.action
    try:
        if action == "status":
            return _status(directory, scope, say)
        if action == "migrate":
            return _migrate(directory, scope, cfg, args, say)
        if action == "export":
            return _export(directory, scope, args, out, say)
        if action == "backup":
            with dbm.Database(directory) as db:
                say(f"backup written: {db.backup()}")
            return EXIT_OK
        if action == "restore":
            replaced = dbm.restore_backup(directory, args.backup)
            say(f"restored from {args.backup}" + (f"; the file it replaced is {replaced}" if replaced else ""))
            say("The ledger is now unconfirmed: no outside call is allowed until a person runs `tt-orchard db confirm-restore --note \"...\"`.")
            return EXIT_OK
        if action == "confirm-restore":
            return _confirm(directory, scope, cfg, args, say)
    except (dbm.LedgerUnavailable, dbm.LedgerRefused) as exc:
        say(f"db: {exc}")
        return EXIT_PROBLEM
    say(f"db: unknown action {action!r}")
    return EXIT_PROBLEM


def _status(directory, scope, say) -> int:
    try:
        st = ts.status_report(directory, scope)
    except (dbm.LedgerUnavailable, dbm.LedgerRefused) as exc:
        say(f"db: {exc}")
        return EXIT_PROBLEM
    say(f"directory: {st['directory']}")
    say(f"database: {'present' if st['database'] else 'none yet'}")
    if st["database"]:
        say(f"schema version: {st['schema_version']}; telegraph records: {st['events']}; cached answers: {st['cache_rows']}")
        say(f"migration: {st['cutover'] or 'not recorded'}")
        say(f"unconfirmed: {st['unconfirmed'] or 'no'}")
    if st["recovery_marker"]:
        say(f"RECOVERY-REQUIRED: {st['recovery_marker'].get('reason')} (no call is allowed until `tt-orchard db confirm-restore`)")
    if st["legacy_ledger"]:
        say(f"telegraph.jsonl: {st['legacy_ledger']}")
    markers = [n for n, here in (("MIGRATING-TO-SQLITE", st["migrating_marker"]), ("MIGRATED-TO-SQLITE", st["migrated_marker"])) if here]
    say(f"archived ledger: {'yes' if st['archived_ledger'] else 'no'}" + (f"; markers: {', '.join(markers)}" if markers else ""))
    say(f"backups: {len(st['backups'])}" + (f" (newest {st['backups'][-1]})" if st["backups"] else ""))
    return EXIT_OK


def _migrate(directory, scope, cfg, args, say) -> int:
    if getattr(args, "what", None) != "telegraph":
        say("db migrate: say what to migrate: `tt-orchard db migrate telegraph`")
        return EXIT_PROBLEM
    accept = getattr(args, "accept_malformed", None)
    try:
        result = mig.migrate(directory, scope=scope, max_output_tokens=cfg.max_output_tokens, accept_malformed=accept,
                             dry_run=bool(getattr(args, "dry_run", False)))
    except mig.MalformedLedger as exc:
        say(f"db migrate: {exc}")
        return EXIT_PROBLEM
    except mig.AlreadyMigrated as exc:
        say(f"db migrate: {exc}; nothing to do")
        return EXIT_PROBLEM
    except (mig.MigrationRefused, dbm.LedgerUnavailable, dbm.LedgerRefused) as exc:
        say(f"db migrate: {exc}")
        return EXIT_PROBLEM
    totals = result.get("totals", {})
    if result["status"] == "dry-run":
        say("dry run: nothing was changed")
        say(f"records {totals['records']}, answered calls {totals['calls']}, tokens in {totals['tokens_in']} out {totals['tokens_out']}, "
            f"cost ${totals['cost_usd']}")
        say(f"checks: {', '.join(f'{k} {"equal" if v else "DIFFERENT"}' for k, v in result['checks'].items())}")
        if result["fragment_to_move"] or result["sidecars"]:
            say(f"torn records: {result['sidecars']} sidecar file(s)" + ("; a cut-off last line would be moved to a sidecar" if result["fragment_to_move"] else ""))
        say(f"cached answers to import: {result['cache_entries']} ({result['cache_skipped']} files skipped)")
        return EXIT_OK if result["equal"] else EXIT_PROBLEM
    say(f"migration {result['status']}" + (" (resumed from an interrupted run)" if result.get("resumed") else ""))
    if totals:
        say(f"records {totals['records']}, answered calls {totals['calls']}, tokens in {totals['tokens_in']} out {totals['tokens_out']}, "
            f"cost ${totals['cost_usd']}")
        say(f"checks: {', '.join(f'{k} {"equal" if v else "DIFFERENT"}' for k, v in result['checks'].items())}")
    if result.get("export_matches_archive") is not None:
        say(f"the export equals the archived ledger line by line: {'yes' if result['export_matches_archive'] else 'NO'}")
    return EXIT_OK if result.get("export_matches_archive") in (True, None) else EXIT_PROBLEM


def _export(directory, scope, args, out, say) -> int:
    if getattr(args, "what", None) != "telegraph":
        say("db export: say what to export: `tt-orchard db export telegraph`")
        return EXIT_PROBLEM
    if getattr(args, "from_archive", False):
        lines = mig.read_archive_lines(directory)
    else:
        lines = ts.read_export_lines(directory, scope)
    for line in lines:
        print(line, file=out)
    return EXIT_OK


def _confirm(directory, scope, cfg, args, say) -> int:
    note = (getattr(args, "note", None) or "").strip()
    if not note:
        say("db confirm-restore: --note is required: say in a sentence how you checked the history (for example, the provider's usage page)")
        return EXIT_PROBLEM
    with dbm.Database(directory) as db:                  # a full open: a damaged file is recovered (and marked) before anything is asked
        if db.exists_name(ts.MARKER_MIGRATING) is not None or db.meta_get(ts.cutover_key(scope)) not in (None, "complete"):
            say("db confirm-restore: a migration of this directory did not finish; run `tt-orchard db migrate telegraph` first")
            return EXIT_PROBLEM
        if not db.integrity_ok():
            say("db confirm-restore: the database fails its integrity check; restore a backup first")
            return EXIT_PROBLEM
        reason, marker = db.unreconciled(), db.recovery_marker()
        if reason is None and marker is None:
            say("nothing to confirm: the ledger is not marked unconfirmed")
            return EXIT_OK
        say(f"the ledger is unconfirmed: {reason or (marker or {}).get('reason')}")
        if not getattr(args, "no_replies", False):
            tokens, cost = tg.worst_case_call(cfg)
            store = ts.TelegraphStore(db, scope, guard=False)
            added = store.add_recovered_reservations(reply_ids(directory), tokens, cost)
            say(f"{added} reply file(s) had no ledger record; each was added as an unsettled reservation at the worst case "
                f"({tokens} tokens" + (f", up to ${cost:.4f}" if cost is not None else ", cost unknown") + "), so the counted spend only goes up")
        db.confirm_reconciled(note)
    say("confirmed. Calls are allowed again. Calls that failed or timed out leave no reply file; check the provider's usage page for those.")
    return EXIT_OK
