"""`tt-orchard db backup | restore | confirm-restore`, and where the telegraph state directory comes from (issue 133)."""
import io
import json
import os
from pathlib import Path

import pytest

from dbtools import make_history, reserve
from ledger_io import ledger_records
from orchard import agent, cli, db_cmd
from orchard import db as dbm
from orchard import telegraph as tg
from orchard import telegraph_cmd as cmd
from orchard import telegraph_store as ts

KEY = "k3y-VALUE-0123456789"


def run_cli(argv, tmp_path, env=None):
    out = io.StringIO()
    code = cli.main(argv, env={"HOME": str(tmp_path / "home"), **(env or {})}, stdout=out)
    return code, out.getvalue()


# =============================================================================================== backup, restore, confirm-restore

def test_backup_then_restore_then_confirm_is_the_recovery_a_person_follows(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c0")
    code, text = run_cli(["db", "backup", "--dir", str(d)], tmp_path)
    assert code == 0 and "backup written:" in text
    backup = text.split("backup written:")[1].strip()
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c1")                                                   # a call made after the backup
    code, text = run_cli(["db", "restore", "--dir", str(d), backup], tmp_path)
    assert code == 0 and "unconfirmed" in text and "confirm-restore" in text
    assert len(ledger_records(d)) == 1                                         # the history since the backup is not in it
    code, text = run_cli(["db", "status", "--dir", str(d)], tmp_path)
    assert "RECOVERY-REQUIRED" in text and "unconfirmed: restored" in text
    with pytest.raises(SystemExit):                                            # --note is required: argparse refuses before the command
        run_cli(["db", "confirm-restore", "--dir", str(d)], tmp_path)
    code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "compared with the provider usage page"], tmp_path)
    assert code == 0 and "confirmed" in text
    with ts.TelegraphStore.open(d) as store:
        assert reserve(store, "c2") is None
    assert not (d / "RECOVERY-REQUIRED").exists()


def test_confirm_restore_charges_a_reply_file_that_has_no_ledger_record_at_its_worst_case_and_does_not_confirm_by_itself(tmp_path):
    d = tmp_path / "state"
    make_history(d, 1)
    (d / "project.db").write_bytes(b"garbage" * 1000)                          # a recovery that lost the ledger
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    (d / "replies").mkdir()
    for name in ("aaa111", "bbb222"):
        (d / "replies" / f"{name}.txt").write_text("an answer")
    (d / "replies" / "link.txt").symlink_to(d / "replies" / "aaa111.txt")      # a link is not a reply file
    cfg = tmp_path / "t.toml"
    cfg.write_text('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["small"]\n'
                   'max_output_tokens = 100\nmax_prompt_chars = 500\n[telegraph.prices]\nsmall = [1.0, 3.0]\n')
    code, text = run_cli(["db", "status", "--dir", str(d), "--telegraph-config", str(cfg)], tmp_path)
    code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "checked", "--telegraph-config", str(cfg)], tmp_path)
    assert code == 0 and "2 reply file(s) had no ledger record" in text
    recovered = [r for r in ledger_records(d) if r.get("recovered")]
    assert sorted(r["reserve"] for r in recovered) == ["aaa111", "bbb222"] and all(r["tokens"] > 100 for r in recovered)
    assert all(r["cost_usd"] and r["cost_usd"] > 0 for r in recovered)


def test_confirm_restore_with_nothing_to_confirm_says_so(tmp_path):
    d = make_history(tmp_path / "state", 1)
    code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "x"], tmp_path)
    assert code == 0 and "nothing to confirm" in text


def test_confirm_restore_refuses_a_database_that_fails_its_integrity_check_or_a_migration_that_did_not_finish(tmp_path):
    d = tmp_path / "state"
    make_history(d, 1)
    with dbm.Database(d) as db:
        db._write_marker("restored")
        db._write_unreconciled("restored")
        with db.write() as conn:
            db.meta_set(conn, "cutover:default", "imported")
    code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "x"], tmp_path)
    assert code == 2 and "did not finish" in text and (d / "RECOVERY-REQUIRED").exists()


def test_restore_names_a_file_that_is_not_a_backup_and_changes_nothing(tmp_path):
    d = make_history(tmp_path / "state", 1)
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"x" * 100)
    code, text = run_cli(["db", "restore", "--dir", str(d), str(bad)], tmp_path)
    assert code == 2 and "not a usable backup" in text and not (d / "RECOVERY-REQUIRED").exists()


def test_reply_ids_ignores_links_directories_and_other_files(tmp_path):
    (tmp_path / "replies").mkdir()
    (tmp_path / "replies" / "a.txt").write_text("x")
    (tmp_path / "replies" / "b.md").write_text("x")
    (tmp_path / "replies" / "c.txt").mkdir()
    (tmp_path / "replies" / "d.txt").symlink_to(tmp_path / "replies" / "a.txt")
    assert db_cmd.reply_ids(tmp_path) == ["a"]
    assert db_cmd.reply_ids(tmp_path / "nowhere") == []


# =============================================================================================== issue 133: the directory is never the caller's home

TABLE = ('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["small"]\nkey_env = "TG_KEY"\n'
         'max_calls = 1\nmax_calls_per_problem = 9\n')


def ok_transport(url, headers, body, timeout):
    return 200, json.dumps({"choices": [{"message": {"content": "advice"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 10}}).encode()


def ask_from(home, cfg_path, tmp_path, environ, *extra, problem="p"):
    task = tmp_path / "task.txt"
    task.write_text("task")
    fail = tmp_path / "fail.txt"
    fail.write_text(f"failure {os.urandom(4).hex()}")
    out = io.StringIO()
    args = cli._parser().parse_args(["telegraph", "ask", "--telegraph-config", str(cfg_path), "--problem", problem,
                                     "--task-file", str(task), "--failure-file", str(fail), *extra])
    code = cmd.main(args, out=out, environ={"TG_KEY": KEY, **environ}, transport=ok_transport, home=home)
    return code, out.getvalue()


def test_an_agent_shell_with_its_own_home_uses_the_configured_ledger_and_shares_its_caps(tmp_path):
    operator_home, shell_home, ledger = tmp_path / "operator", tmp_path / "run" / "home", tmp_path / "shared-ledger"
    shell_home.mkdir(parents=True)
    cfg = tmp_path / "t.toml"
    cfg.write_text(TABLE + f'dir = "{ledger}"\n')
    code, text = ask_from(operator_home, cfg, tmp_path, {})                    # the operator makes the one call the cap allows
    assert code == 0
    code, text = ask_from(shell_home, cfg, tmp_path, {"ORCHARD_RUN_DIR": str(tmp_path / "run")})   # a call from the agent shell
    assert code == 3 and "call cap" in text                                    # counted against the same caps
    assert not (shell_home / ".cache").exists()                                # no second ledger under the run directory
    assert len([r for r in ledger_records(ledger) if "reserve" in r]) == 1


def test_an_agent_shell_takes_the_directory_from_the_variable_the_supervisor_set(tmp_path):
    shell_home, ledger = tmp_path / "run" / "home", tmp_path / "shared-ledger"
    shell_home.mkdir(parents=True)
    cfg = tmp_path / "t.toml"
    cfg.write_text(TABLE)                                                       # no `dir`: only the environment names it
    code, text = ask_from(shell_home, cfg, tmp_path, {"ORCHARD_RUN_DIR": str(tmp_path / "run"), "ORCHARD_TELEGRAPH_DIR": str(ledger)})
    assert code == 0 and len(ledger_records(ledger)) >= 1
    assert not (shell_home / ".cache").exists()


def test_an_agent_shell_with_no_configured_directory_stops_and_does_not_open_a_ledger_in_its_home(tmp_path):
    shell_home = tmp_path / "run" / "home"
    shell_home.mkdir(parents=True)
    cfg = tmp_path / "t.toml"
    cfg.write_text(TABLE)
    code, text = ask_from(shell_home, cfg, tmp_path, {"ORCHARD_RUN_DIR": str(tmp_path / "run")})
    assert code == 2 and "agent shell" in text and "ORCHARD_TELEGRAPH_DIR" in text
    assert list(shell_home.rglob("*")) == []


def test_an_operator_shell_without_a_configured_directory_still_uses_the_default_under_its_home(tmp_path):
    operator_home = tmp_path / "operator"
    cfg = tmp_path / "t.toml"
    cfg.write_text(TABLE)
    code, _text = ask_from(operator_home, cfg, tmp_path, {})
    assert code == 0 and (operator_home / ".cache" / "tt-orchard" / "telegraph" / "project.db").is_file()


def test_the_order_is_dir_then_config_then_variable_then_home(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    with_dir = tg.load({"enabled": False, "dir": str(b)}, base=tmp_path)
    without = tg.Config()
    home = tmp_path / "home"
    assert cmd.resolve_directory(str(a), with_dir, {"ORCHARD_TELEGRAPH_DIR": str(c)}, home) == a
    assert cmd.resolve_directory(None, with_dir, {"ORCHARD_TELEGRAPH_DIR": str(c)}, home) == b
    assert cmd.resolve_directory(None, without, {"ORCHARD_TELEGRAPH_DIR": str(c)}, home) == c
    assert cmd.resolve_directory(None, without, {}, home) == home / ".cache" / "tt-orchard" / "telegraph"


def test_a_relative_directory_in_the_variable_is_refused(tmp_path):
    with pytest.raises(tg.ConfigError, match="absolute"):
        cmd.resolve_directory(None, tg.Config(), {"ORCHARD_TELEGRAPH_DIR": "relative/dir"}, tmp_path)


def test_the_supervisors_agent_environment_passes_an_absolute_directory_and_nothing_else_of_the_kind(tmp_path):
    env = agent.agent_env(tmp_path / "run", source={"PATH": "/usr/bin", "ORCHARD_TELEGRAPH_DIR": "/srv/ledger", "HOME": "/home/op"})
    assert env["ORCHARD_TELEGRAPH_DIR"] == "/srv/ledger" and env["HOME"] == str(tmp_path / "run" / "home")
    for bad in ("relative", "", "/x\0y"):
        got = agent.agent_env(tmp_path / "run2", source={"ORCHARD_TELEGRAPH_DIR": bad})
        assert "ORCHARD_TELEGRAPH_DIR" not in got
    assert "ORCHARD_TELEGRAPH_DIR" not in agent.agent_env(tmp_path / "run3", source={"PATH": "/usr/bin"})


def test_the_report_command_a_shell_prints_carries_the_directory_so_the_verdict_goes_to_the_same_ledger(tmp_path):
    shell_home, ledger = tmp_path / "run" / "home", tmp_path / "shared-ledger"
    shell_home.mkdir(parents=True)
    cfg = tmp_path / "t.toml"
    cfg.write_text(TABLE + f'dir = "{ledger}"\n')
    code, text = ask_from(shell_home, cfg, tmp_path, {"ORCHARD_RUN_DIR": str(tmp_path / "run")})
    assert code == 0 and "--dir" in text and str(ledger) in text


def test_db_status_reports_an_unreadable_backups_folder_and_exits_with_the_problem_code(tmp_path):
    """Issue 157: a symlinked backups folder made status_report return backups=None and `db status` raised a TypeError."""
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c0")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (d / "backups").symlink_to(elsewhere)
    code, text = run_cli(["db", "status", "--dir", str(d)], tmp_path)
    assert code == db_cmd.EXIT_PROBLEM
    assert "backups: cannot be read:" in text
    assert "directory:" in text and "schema version:" in text          # the rest of the report is still printed
