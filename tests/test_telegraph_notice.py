"""The notice a person sees when the Telegraph state directory still holds an old jsonl ledger (or a half-finished or lost move).

Every test builds its own temporary directory. No test touches a real state directory, opens a device or makes a network call."""
import io
import json
import os
import stat

import pytest

from ledger_io import tree_text
from orchard import telegraph as tg
from orchard import telegraph_migrate as mig
from orchard import telegraph_notice as tn

RECORD = {"call_id": "a1", "problem": "p", "model": "small", "outcome": "ok", "tokens_in": 10, "tokens_out": 5,
          "cost_usd": 0.5, "usage_known": True, "ts": 1.0}


def old_ledger(directory, lines=1):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "telegraph.jsonl").write_text("".join(json.dumps({**RECORD, "call_id": f"c{i}"}) + "\n" for i in range(lines)))
    return directory


def snapshot(directory):
    """Every name under a directory with its mode, size and modification time: a change of any kind shows."""
    out = {}
    for p in sorted(directory.rglob("*")):
        st = os.lstat(p)
        out[str(p.relative_to(directory))] = (stat.S_IFMT(st.st_mode), st.st_size, st.st_mtime_ns)
    return out


CFG_ON = tg.Config(enabled=True, endpoint="https://llm.example.org/v1", models=("small",), key_env="K")


# ------------------------------------------------------------------------------------------ the four states

def test_an_unmigrated_ledger_names_both_commands_and_says_nothing_is_deleted(tmp_path):
    d = old_ledger(tmp_path / "state")
    n = tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path)
    assert n.kind == "unmigrated"
    assert n.summary.startswith("Run `tt-orchard db migrate telegraph --dry-run`, then `tt-orchard db migrate telegraph`")
    assert n.commands == ("tt-orchard db migrate telegraph --dry-run", "tt-orchard db migrate telegraph")
    text = n.text()
    assert "Nothing is deleted" in text and "backup" in text and "telegraph.jsonl.archived" in text
    assert "--dry-run-first" in text and str(d) in text and str(d) not in n.summary


def test_a_half_finished_migration_says_to_run_the_command_again(tmp_path):
    d = old_ledger(tmp_path / "state")
    (d / "MIGRATING-TO-SQLITE").write_text("{}")
    n = tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path)
    assert n.kind == "half-finished" and n.summary.startswith("Run `tt-orchard db migrate telegraph` again")
    assert n.commands == ("tt-orchard db migrate telegraph",) and "Nothing is deleted" in n.text()


def test_a_lost_database_points_at_status_and_restore(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "MIGRATED-TO-SQLITE").write_text("{}")
    n = tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path)
    assert n.kind == "lost-database" and n.summary.startswith("Run `tt-orchard db status`")
    assert "db restore" in n.text() and "confirm-restore" in n.text()


def test_a_directory_that_had_a_ledger_and_lost_its_database_is_a_lost_database(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "INITIALIZED").write_text("")
    assert tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path).kind == "lost-database"


def test_a_jsonl_file_back_in_a_moved_directory_is_an_old_writer(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "project.db").write_bytes(b"x")
    (d / "MIGRATED-TO-SQLITE").write_text("{}")
    (d / "telegraph.jsonl").write_text("")
    n = tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path)
    assert n.kind == "old-writer" and n.summary.startswith("Stop the old tt-orchard process")
    assert "db migrate telegraph --dry-run" not in n.text().replace("Do not run `db migrate` again", "")


def test_a_link_named_like_the_ledger_counts_as_an_old_ledger_and_is_not_followed(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (tmp_path / "elsewhere").write_text("x")
    (d / "telegraph.jsonl").symlink_to(tmp_path / "elsewhere")
    assert tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path).kind == "unmigrated"


# ------------------------------------------------------------------------------------------ nothing to say

def test_a_migrated_directory_has_no_notice_and_the_notice_goes_away_after_the_real_migration(tmp_path):
    d = old_ledger(tmp_path / "state", lines=3)
    assert tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path) is not None
    mig.migrate(d)                                                     # the real migration, in a temporary directory
    assert (d / "telegraph.jsonl").is_dir()                            # the cutover sentinel is a directory and holds no ledger
    assert tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path) is None


def test_someone_who_never_used_telegraph_sees_nothing(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    assert tn.telegraph_state_notice(tg.Config(), environ={}, home=home) is None          # no directory at all, telegraph off
    assert not (home / ".cache").exists()                                                # and the call made nothing
    (home / ".cache" / "tt-orchard" / "telegraph").mkdir(parents=True)
    assert tn.telegraph_state_notice(tg.Config(), environ={}, home=home) is None          # an empty directory
    assert tn.telegraph_state_notice(CFG_ON, environ={}, home=home) is None                # on, but nothing was ever written


def test_a_directory_with_only_a_database_has_no_notice(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "project.db").write_bytes(b"x")
    (d / "INITIALIZED").write_text("")
    assert tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path) is None


def test_telegraph_that_is_off_still_warns_about_an_old_ledger_in_its_directory(tmp_path):
    d = old_ledger(tmp_path / "state")
    assert tn.telegraph_state_notice(tg.Config(), directory=d, environ={}, home=tmp_path).kind == "unmigrated"


# ------------------------------------------------------------------------------------------ the usual resolution

def test_the_directory_comes_from_the_flag_then_the_config_then_the_environment_then_the_home(tmp_path):
    flag, conf, env, home = (old_ledger(tmp_path / n) for n in ("flag", "conf", "env", "home/.cache/tt-orchard/telegraph"))
    environ = {"ORCHARD_TELEGRAPH_DIR": str(env)}
    both = tg.Config(enabled=True, endpoint="https://llm.example.org/v1", models=("small",), key_env="K", dir=conf)
    assert tn.telegraph_state_notice(both, directory=flag, environ=environ, home=tmp_path / "home").directory == str(flag)
    assert tn.telegraph_state_notice(both, environ=environ, home=tmp_path / "home").directory == str(conf)
    assert tn.telegraph_state_notice(CFG_ON, environ=environ, home=tmp_path / "home").directory == str(env)
    assert tn.telegraph_state_notice(CFG_ON, environ={}, home=tmp_path / "home").directory == str(home)


def test_an_agent_shell_with_no_configured_directory_gets_no_notice_and_no_exception(tmp_path):
    old_ledger(tmp_path / "home/.cache/tt-orchard/telegraph")
    assert tn.telegraph_state_notice(CFG_ON, environ={"ORCHARD_RUN_DIR": "/r"}, home=tmp_path / "home") is None


def test_flags_are_added_to_every_command_and_quoted(tmp_path):
    d = old_ledger(tmp_path / "a dir")
    n = tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path, flags=f"--dir '{d}'")
    assert all(c.endswith(f"--dir '{d}'") for c in n.commands)
    assert n.commands[0].startswith("tt-orchard db migrate telegraph --dry-run --dir ")
    ns = type("A", (), {"dir": str(d), "telegraph_config": None, "config": "/c/b.toml"})()
    assert tn.flags_from_args(ns) == f"--dir '{d}' --config /c/b.toml"


# ------------------------------------------------------------------------------------------ it only looks

@pytest.mark.parametrize("marker", [None, "MIGRATING-TO-SQLITE", "MIGRATED-TO-SQLITE"])
def test_the_notice_changes_nothing_in_the_directory(tmp_path, marker):
    d = old_ledger(tmp_path / "state")
    if marker:
        (d / marker).write_text("{}")
    before, text = snapshot(d), tree_text(d)
    for _ in range(2):
        tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path)
    assert snapshot(d) == before and tree_text(d) == text
    assert not (d / "project.db").exists() and not (d / "INITIALIZED").exists() and not (d / "telegraph.lock").exists()


def test_the_notice_does_not_create_a_missing_directory(tmp_path):
    d = tmp_path / "nope"
    assert tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path) is None and not d.exists()


@pytest.mark.parametrize("kind", ["unmigrated", "half-finished", "lost-database", "old-writer"])
def test_no_notice_text_uses_the_framing_phrases_the_project_avoids(tmp_path, kind):
    d = tmp_path / "state"
    d.mkdir()
    if kind in ("unmigrated", "half-finished", "old-writer"):
        (d / "telegraph.jsonl").write_text("")
    if kind == "half-finished":
        (d / "MIGRATING-TO-SQLITE").write_text("{}")
    if kind in ("lost-database", "old-writer"):
        (d / "MIGRATED-TO-SQLITE").write_text("{}")
    n = tn.telegraph_state_notice(CFG_ON, directory=d, environ={}, home=tmp_path)
    assert n.kind == kind
    for bad in (", not ", "rather than", "instead of", "reads as", "honest", "the one", "which is why", "This is what"):
        assert bad not in n.text()


