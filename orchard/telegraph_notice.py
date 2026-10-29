# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What a person must do about the Telegraph state directory, said once and in plain English.

The ledger moved from `telegraph.jsonl` to the SQLite database `project.db` (orchard/telegraph_store.py). Someone who runs
`git pull` and then `tt-orchard` with an old ledger in place has a directory that the new code refuses to use. This module looks
at the directory and says which of these states it is in, and what to type next:

  unmigrated      `telegraph.jsonl` is a live ledger file and the directory was never moved
  half-finished   `MIGRATING-TO-SQLITE` is there: a migration started and did not finish
  lost-database   the directory was migrated (or had a ledger) and `project.db` is gone
  old-writer      the directory was migrated, and a `telegraph.jsonl` file is there again (an old process wrote it)

A directory that is fine, absent or empty has no notice. A person who never used Telegraph sees nothing, because every state
above needs a file that only a used ledger leaves behind.

Nothing here opens a database, creates a file, follows a link or reads a file's content: it only asks the operating system what
each name in the directory is (`lstat`). That makes it safe to call from a page that refreshes, from setup and from the start of
any command, and it cannot disturb a migration that is running.

`telegraph_state_notice(config)` resolves the directory the usual way (`--dir`, the config's `[telegraph] dir`,
`$ORCHARD_TELEGRAPH_DIR`, the home default) and returns a `Notice` or None. `state_of(directory)` is the same look at a directory
that is already chosen.
"""
from __future__ import annotations

import os
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path

from orchard import db as dbm
from orchard import telegraph_store as ts

UNMIGRATED, HALF_FINISHED, LOST_DATABASE, OLD_WRITER = "unmigrated", "half-finished", "lost-database", "old-writer"
KINDS = (UNMIGRATED, HALF_FINISHED, LOST_DATABASE, OLD_WRITER)
# The two states in which a command that opens the ledger would create `project.db` and its helper files in a directory that is
# not ready for them. Such a command stops before it opens anything.
BLOCKS_OPENING = (UNMIGRATED, HALF_FINISHED)

DRY_RUN = "tt-orchard db migrate telegraph --dry-run"
MIGRATE = "tt-orchard db migrate telegraph"
GUIDED = "tt-orchard db migrate telegraph --dry-run-first"


@dataclass(frozen=True)
class Notice:
    """One state of one directory. `summary` is a single line that starts with what to do. `lines` say why and what is safe.
    `commands` are the exact commands, in order, ready to paste (they carry `--dir` when the caller chose a directory)."""
    kind: str
    directory: str
    summary: str
    lines: tuple[str, ...]
    commands: tuple[str, ...]

    @property
    def blocks_opening(self) -> bool:
        return self.kind in BLOCKS_OPENING

    def text(self) -> str:
        """The whole notice as lines of text: the summary first, then the reasons."""
        return "\n".join((self.summary, *self.lines))

    def as_dict(self) -> dict:
        """The notice as plain data for a page (the web view and `spend --json`)."""
        return {"kind": self.kind, "directory": self.directory, "summary": self.summary, "lines": list(self.lines),
                "commands": list(self.commands)}


def _kind_of(names: dict) -> str | None:
    """The state a directory is in, from what each name is (`file`, `dir`, `link`, `other`, or absent)."""
    live = names.get(ts.LEDGER) in ("file", "link", "other")             # a directory of that name is the cutover sentinel
    moved = names.get(ts.MARKER_MIGRATED) is not None or names.get(ts.ARCHIVE) is not None
    if names.get(ts.MARKER_MIGRATING) is not None:
        return HALF_FINISHED
    if live and moved:
        return OLD_WRITER
    if live:
        return UNMIGRATED
    database_gone = names.get(dbm.DB_NAME) is None
    if database_gone and (names.get(ts.MARKER_MIGRATED) is not None or names.get(dbm.INIT_NAME) is not None):
        return LOST_DATABASE
    return None


def _names(directory) -> dict:
    """What each name of interest is, by `lstat` (a link is a link, never followed). An unreadable name counts as absent."""
    out: dict = {}
    for name in (ts.LEDGER, ts.ARCHIVE, ts.MARKER_MIGRATING, ts.MARKER_MIGRATED, dbm.DB_NAME, dbm.INIT_NAME):
        try:
            mode = os.lstat(os.path.join(os.fspath(directory), name)).st_mode
        except OSError:
            continue
        out[name] = ("link" if stat.S_ISLNK(mode) else "dir" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode) else "other")
    return out


def state_of(directory, *, flags: str = "", migrate_flags: str = "") -> Notice | None:
    """The notice for a chosen directory, or None when it needs nothing.

    `flags` is text added to every command of the notice (for example `--dir /path --config c.toml --scope team`): each option that
    changes which ledger a command acts on or how. `migrate_flags` is added to the `db migrate` commands only (`--accept-malformed N`,
    which the other `db` commands do not take)."""
    kind = _kind_of(_names(directory))
    if kind is None:
        return None
    pad = lambda text: (" " + text.strip()) if text.strip() else ""             # noqa: E731
    return _notice(kind, str(directory), pad(flags), pad(migrate_flags))


def _notice(kind: str, where: str, flags: str, migrate_flags: str = "") -> Notice:
    dry, run, guided = DRY_RUN + flags + migrate_flags, MIGRATE + flags + migrate_flags, GUIDED + flags + migrate_flags
    if kind == UNMIGRATED:
        return Notice(
            kind, where,
            f"Run `{dry}`, then `{run}`: the Telegraph ledger is still the old telegraph.jsonl file.",
            (f"Directory: {where}",
             "Until it is moved into the SQLite database, Telegraph calls are refused, nothing is sent and the spend cannot be shown.",
             "Stop any running tt-orchard process first. The dry run checks the move and changes nothing.",
             "Nothing is deleted: the old file is kept as telegraph.jsonl.archived, and a backup of the new database is taken.",
             f"To do both steps with one confirmation: `{guided}`."),
            (dry, run))
    if kind == HALF_FINISHED:
        return Notice(
            kind, where,
            f"Run `{run}` again to finish moving the Telegraph ledger: an earlier run stopped part way.",
            (f"Directory: {where}",
             "Telegraph calls are refused until it finishes, and nothing is sent.",
             "The command resumes where it stopped. Stop any running tt-orchard process first. If the earlier run used a scope "
             "other than default, the command needs `--scope <that scope>`; it names the scope when it is missing.",
             "Nothing is deleted: the old file is kept as telegraph.jsonl.archived, and a backup of the new database is taken at the end."),
            (run,))
    if kind == LOST_DATABASE:
        status = "tt-orchard db status" + flags
        restore = "tt-orchard db restore <a file from backups/>" + flags
        return Notice(
            kind, where,
            f"Run `{status}`, then restore the newest backup with `{restore}`: the Telegraph database is missing.",
            (f"Directory: {where}",
             "This directory was moved to the database (or had a ledger), but project.db is gone. The spend history is not "
             "available and Telegraph calls are refused.",
             "After a restore the ledger is unconfirmed. Compare it with the provider's usage page, then run "
             "`tt-orchard db confirm-restore --note \"how I checked\"" + flags + "`.",
             "The restore keeps any file it replaces."),
            (status, restore))
    status = "tt-orchard db status" + flags
    return Notice(
        kind, where,
        f"Stop the old tt-orchard process that wrote telegraph.jsonl, then run `{status}`.",
        (f"Directory: {where}",
         "This directory was already moved to the database, and a telegraph.jsonl file is there again. An older tt-orchard "
         "(from before the update) is running or ran after the move.",
         "Telegraph calls are refused until that is reconciled. Do not run `db migrate` again: it refuses a directory that was moved.",
         "The README section \"The ledger database\" says how to reconcile."),
        (status,))


def telegraph_state_notice(config, *, directory=None, environ=None, home=None, flags: str = "",
                           migrate_flags: str = "") -> Notice | None:
    """The notice for the Telegraph state directory of `config` (a `telegraph.Config`), or None.

    The directory is chosen as `tt-orchard telegraph` chooses it: `directory` (the `--dir` flag), else `config.dir`, else
    `$ORCHARD_TELEGRAPH_DIR`, else the home default. A choice that cannot be made (an agent shell with no configured directory, a
    relative environment path) gives None: there is nothing to say about a directory nobody can name. The directory is looked at
    with `lstat` only, so nothing is opened, created or changed.

    Telegraph that is switched off still gets the notice when its directory holds an old ledger, because that ledger is what the
    person kept. A directory with no ledger files gives None whether Telegraph is on or off."""
    from orchard import telegraph as tg
    from orchard import telegraph_cmd
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    cfg = config if config is not None else tg.Config()
    try:
        chosen = telegraph_cmd.resolve_directory(directory, cfg, environ, home)
    except tg.ConfigError:
        return None
    if directory is None and "--dir" not in flags.split() and chosen != telegraph_cmd.default_dir(home):
        # A command pasted from the notice resolves its own directory (its config, its environment). When this one did not come from
        # the home default, the commands name it, so they act on the ledger the notice is about.
        flags = f"{flags} --dir {shlex.quote(str(chosen))}".strip()
    return state_of(chosen, flags=flags, migrate_flags=migrate_flags)


def flags_from_args(args) -> str:
    """The options of a command line that choose the ledger, as text to add to every command of a notice: `--dir`,
    `--telegraph-config`, `--config` and a `--scope` other than the default. (`--accept-malformed` is in `migrate_flags_from_args`:
    only `db migrate` takes it.)

    Every selector the caller passed is carried. A config that was not named is found again by the pasted command in the same way
    (`$ORCHARD_BRINGUP_CONFIG`, the checkout, the home folder), and `telegraph_state_notice` adds `--dir` whenever the directory it
    inspected is not the home default, so the commands act on the same ledger whichever config they find."""
    parts = []
    for flag, attr in (("--dir", "dir"), ("--telegraph-config", "telegraph_config"), ("--config", "config")):
        value = getattr(args, attr, None)
        if value:
            parts += [flag, shlex.quote(str(value))]
    scope = getattr(args, "scope", None)
    if scope and scope != ts.DEFAULT_SCOPE:
        parts += ["--scope", shlex.quote(str(scope))]
    return " ".join(parts)


def migrate_flags_from_args(args) -> str:
    """The options only `db migrate` takes: the count of unreadable lines a person accepted."""
    accept = getattr(args, "accept_malformed", None)
    return f"--accept-malformed {int(accept)}" if accept is not None else ""


def with_dir(flags: str, directory) -> str:
    """`flags` with `--dir <directory>` added when it names no directory yet."""
    if "--dir" in shlex.split(flags):
        return flags
    return f"{flags} --dir {shlex.quote(str(directory))}".strip()


def display_quote(shown: str) -> str:
    """A value for a command that is shown on a page after redaction: shell-quoted, except that a leading `~/` stays outside the
    quotes so that a shell expands it."""
    return "~/" + shlex.quote(shown[2:]) if shown.startswith("~/") else shlex.quote(shown)
