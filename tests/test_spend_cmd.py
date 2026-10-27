# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard spend` (orchard/spend_cmd.py): the real helper spend and the virtual local spend, side by side.

The real spend comes from a ledger written in the producer's shape; the local figure from fake agent logs. The two totals are
on separate lines and no figure adds them."""
import io
import json
import os
from pathlib import Path

import pytest

from orchard import cli, spend_cmd
from test_virtualspend import turn, write_log

BASE = 'runs_root = "runs"\n[coder]\ntarget = "org/coder"\nkind = "bundle"\nprofile = "p300"\nport = 8001\nchips = 2\n'


def real_ledger(directory, cost=0.25):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "telegraph.jsonl").write_text(json.dumps(
        {"call_id": "c1", "outcome": "ok", "tokens_in": 100, "tokens_out": 50, "cost_usd": cost, "model": "m", "problem": "p"}) + "\n")


def run(tmp_path, argv, config_text=None):
    cfg = tmp_path / "bringup.toml"
    cfg.write_text(BASE + (config_text or ""))
    out = io.StringIO()
    code = cli.main(["spend", *argv, "--config", str(cfg), "--telegraph-config", str(cfg)], env={}, stdout=out)
    return code, out.getvalue()


@pytest.fixture
def probe(tmp_path):
    root = tmp_path / "probe"
    write_log(root, "op/attempt-1/log/probe-1.jsonl", [turn(1, 1_000_000, 100_000)])
    (root / "op" / "attempt-1" / "attempt.json").write_text(json.dumps({"agent_wall_s": 7200}))
    return root


def test_it_prints_the_real_and_the_virtual_spend_on_separate_lines(tmp_path, probe):
    real_ledger(tmp_path / "tg")
    code, text = run(tmp_path, [str(probe), "--dir", str(tmp_path / "tg")])
    assert code == 0
    real_at = text.index("REAL helper spend")
    local_at = text.index("LOCAL virtual spend")
    assert real_at < local_at
    real_block, local_block = text[real_at:local_at], text[local_at:]
    assert "$0.2500 · 1 calls · 100 in, 50 out" in real_block
    assert "1000000 prompt, 100000 completion tokens" in local_block and "2.00 h of agent time" in local_block
    assert "at claude-haiku-5-5 ($0.1 in, $0.5 out per million): $0.1500" in local_block
    assert "virtual" in local_block and "nobody was charged" in local_block
    assert "$0.2500" not in local_block and "$0.1500" not in real_block
    assert "$0.4000" not in text                                                         # nothing sums the two ($0.25 + $0.15)


def test_every_configured_reference_is_priced(tmp_path, probe):
    extra = '[virtual_spend]\nreference = "tiny"\nprice_in = 1\nprice_out = 2\n[[virtual_spend.extra]]\nname = "big"\nprice_in = 2\nprice_out = 10\n'
    code, text = run(tmp_path, [str(probe), "--dir", str(tmp_path / "nothing")], extra)
    assert code == 0
    assert "at tiny ($1 in, $2 out per million): $1.2000" in text
    assert "at big ($2 in, $10 out per million): $3.0000" in text


def test_the_json_keeps_the_two_spends_in_separate_keys_and_adds_no_total(tmp_path, probe):
    real_ledger(tmp_path / "tg")
    code, text = run(tmp_path, [str(probe), "--json", "--dir", str(tmp_path / "tg")])
    data = json.loads(text)
    assert code == 0 and set(data) == {"real_helper_spend", "local_virtual_spend"}
    assert data["real_helper_spend"]["cost_usd"] == 0.25 and data["real_helper_spend"]["calls"] == 1
    local = data["local_virtual_spend"]
    assert "virtual figure" in local["label"] and "comparison figure only" in local["note"]
    (row,) = local["directories"]
    assert row["dir"] == str(probe) and row["turns"] == 1 and row["agent_wall_s"] == 7200
    assert row["virtual"][0]["cost_usd"] == pytest.approx(0.15)
    assert "total" not in json.dumps(data).replace("total_tokens", "")                  # no combined figure anywhere
    assert set(local["all_directories"]) == {"turns", "prompt_tokens", "completion_tokens", "virtual"}


def test_several_directories_get_a_row_each_and_one_local_total(tmp_path, probe):
    other = tmp_path / "other"
    write_log(other, "log/run.jsonl", [turn(1, 2_000_000, 0)])
    code, text = run(tmp_path, [str(probe), str(other), "--dir", str(tmp_path / "none")])
    assert code == 0
    assert text.count(": 1 turns") == 2
    assert "all 2 directories: 2 turns · 3000000 prompt, 100000 completion tokens" in text
    assert "at claude-haiku-5-5: $0.3500" in text


def test_a_directory_with_no_logs_is_listed_when_named_and_says_what_was_left_out(tmp_path):
    empty = tmp_path / "empty"
    write_log(empty, "log/run.jsonl", ["{bad", turn(1, 5, 1).replace('"usage"', '"use"')])
    code, text = run(tmp_path, [str(empty), "--dir", str(tmp_path / "none")])
    assert code == 0 and "0 turns" in text
    assert "Left out: 1 lines that are not JSON, 1 turns with no token counts." in text


def test_with_no_directory_it_reads_every_run_under_runs_root(tmp_path):
    runs = tmp_path / "runs"
    write_log(runs, "org--a/stages/1/log/run-1.jsonl", [turn(1, 100, 10)])
    write_log(runs, "org--b/stages/1/log/run-1.jsonl", [turn(1, 200, 20)])
    (runs / "org--c").mkdir()
    code, text = run(tmp_path, ["--dir", str(tmp_path / "none")])
    assert code == 0
    assert "org--a: 1 turns" in text and "org--b: 1 turns" in text and "org--c" not in text      # a run with no turns is not listed
    assert "all 2 directories: 2 turns" in text


def test_a_runs_root_link_to_a_folder_is_not_followed(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    outside = tmp_path / "outside"
    write_log(outside, "log/run.jsonl", [turn(1, 100, 10)])
    (runs / "linked").symlink_to(outside)
    code, text = run(tmp_path, ["--dir", str(tmp_path / "none")])
    assert code == 0 and "linked" not in text and "no local agent logs" in text


def test_with_no_directory_and_no_config_it_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "no-checkout")
    out = io.StringIO()
    args = cli._parser().parse_args(["spend", "--dir", str(tmp_path / "none")])
    assert spend_cmd.main(args, out=out, environ={}, home=tmp_path / "home") == 2
    assert "name at least one directory" in out.getvalue()


def test_a_named_directory_works_without_any_config(tmp_path, monkeypatch, probe):
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "no-checkout")
    out = io.StringIO()
    args = cli._parser().parse_args(["spend", str(probe)])
    assert spend_cmd.main(args, out=out, environ={}, home=tmp_path / "home") == 0
    assert "at claude-haiku-5-5" in out.getvalue()                                       # the defaults


def test_a_bad_virtual_spend_table_is_a_problem_with_the_key_named(tmp_path, probe):
    code, text = run(tmp_path, [str(probe)], "[virtual_spend]\nprice_in = -1\n")
    assert code == 2 and "virtual_spend.price_in" in text and "REAL" not in text


def test_an_unreadable_real_ledger_does_not_stop_the_local_figure(tmp_path, probe, monkeypatch):
    def boom(_):
        raise ValueError("bad record")
    monkeypatch.setattr(spend_cmd.tg, "summarise_records", boom)
    code, text = run(tmp_path, [str(probe), "--dir", str(tmp_path / "tg")])
    assert code == 0 and "the telegraph ledger has a record that cannot be read" in text and "$0.1500" in text


def test_a_real_ledger_calls_with_no_price_are_said(tmp_path, probe):
    d = tmp_path / "tg"
    d.mkdir()
    (d / "telegraph.jsonl").write_text(json.dumps({"call_id": "c1", "outcome": "ok", "tokens_in": 1, "tokens_out": 1, "model": "m", "problem": "p"}) + "\n")
    code, text = run(tmp_path, [str(probe), "--dir", str(d)])
    assert "(+ 1 calls with no price)" in text


def test_the_command_says_the_lease_time_is_not_shown_and_the_figure_is_not_a_bill(tmp_path, probe):
    code, text = run(tmp_path, [str(probe), "--dir", str(tmp_path / "none")])
    assert "Chip lease time is not recorded" in text
    assert "prompt caching" in text and "comparison figure only" in text


def test_the_configured_telegraph_dir_is_the_real_ledger_when_dir_is_not_given(tmp_path, probe):
    real_ledger(tmp_path / "ledger-here", cost=1.5)
    code, text = run(tmp_path, [str(probe)], '[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["a"]\ndir = "ledger-here"\n')
    assert code == 0 and "$1.5000" in text


def test_the_command_writes_nothing(tmp_path, probe):
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    run(tmp_path, [str(probe), "--dir", str(tmp_path / "tg")])
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert set(after) - set(before) == {"bringup.toml"} and set(before) <= set(after)    # only the test's own config file is new



def test_one_directory_gets_no_all_directories_line(tmp_path, probe):
    code, text = run(tmp_path, [str(probe), "--dir", str(tmp_path / "none")])
    assert code == 0 and "all 1 directories" not in text and "directories:" not in text



# ---- review fix 1: a config that was named and is missing is a setup error ------------------------------------------

def test_a_named_config_that_is_missing_is_a_setup_error_not_a_silent_default(tmp_path, probe, monkeypatch):
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "no-checkout")
    missing = tmp_path / "gone.toml"
    for argv, env in ((["spend", str(probe), "--config", str(missing)], {}),
                      (["spend", str(probe)], {cli.CONFIG_ENV: str(missing)})):
        out = io.StringIO()
        code = cli.main(argv, env=env, stdout=out)
        text = out.getvalue()
        assert code == 2 and "does not exist" in text, (argv, text)
        assert "REAL" not in text and "claude-haiku" not in text, "no figures from default settings"


def test_a_missing_config_found_by_discovery_still_falls_back_to_the_defaults(tmp_path, probe, monkeypatch):
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "no-checkout")
    out = io.StringIO()
    args = cli._parser().parse_args(["spend", str(probe)])
    assert spend_cmd.main(args, out=out, environ={}, home=tmp_path / "home") == 0
    assert "at claude-haiku-5-5" in out.getvalue()



# ---- review fix 1 (second round): a telegraph table that fails validation is a setup error ---------------------------------

@pytest.mark.parametrize("table,needle", [
    ('[telegraph]\nenabled = true\nmodels = ["a"]\ndir = "ledger-here"\n', "telegraph.endpoint"),                       # no endpoint
    ('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["a"]\ndir = 7\n', "telegraph.dir"),
    ('[telegraph]\nenabled = true\nendpoint = "ftp://x"\nmodels = ["a"]\ndir = "ledger-here"\n', "telegraph.endpoint"),
    ('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["a"]\ndir = "ledger-here"\nbogus = 1\n', "unknown key"),
], ids=["no-endpoint", "bad-dir", "bad-scheme", "unknown-key"])
def test_an_invalid_telegraph_table_does_not_fall_back_to_the_default_ledger(tmp_path, probe, table, needle):
    real_ledger(tmp_path / "ledger-here", cost=1.5)                  # paid calls in the configured ledger
    code, text = run(tmp_path, [str(probe)], table)
    assert code == 2 and needle in text, text
    assert "REAL helper spend" not in text and "$0.00" not in text


def test_a_config_that_cannot_be_parsed_is_a_setup_error(tmp_path, probe):
    cfg = tmp_path / "bad.toml"
    cfg.write_text("[[[")
    out = io.StringIO()
    assert cli.main(["spend", str(probe), "--telegraph-config", str(cfg)], env={}, stdout=out) == 2
    assert "cannot parse" in out.getvalue() and "REAL" not in out.getvalue()


def test_a_disabled_telegraph_table_with_a_dir_still_shows_that_ledger(tmp_path, probe):
    real_ledger(tmp_path / "ledger-here", cost=1.5)
    code, text = run(tmp_path, [str(probe)], '[telegraph]\nenabled = false\ndir = "ledger-here"\n')
    assert code == 0 and "$1.5000" in text


# ---- review fix 2 (fourth round): the tables file and the bringup config are chosen separately ------------------------------

def test_a_prices_file_with_only_the_tables_does_not_hide_runs_root_of_the_bringup_config(tmp_path):
    runs = tmp_path / "runs"
    write_log(runs, "org--a/stages/1/log/run-1.jsonl", [turn(1, 1_000_000, 0)])
    bringup = tmp_path / "bringup.toml"
    bringup.write_text(BASE)
    prices = tmp_path / "prices.toml"
    prices.write_text('[virtual_spend]\nreference = "other-model"\nprice_in = 4\nprice_out = 8\n')
    out = io.StringIO()
    code = cli.main(["spend", "--config", str(bringup), "--telegraph-config", str(prices), "--dir", str(tmp_path / "none")],
                    env={}, stdout=out)
    text = out.getvalue()
    assert code == 0, text
    assert "org--a: 1 turns" in text, "runs come from the bringup config's runs_root"
    assert "at other-model ($4 in, $8 out per million): $4.0000" in text, "prices come from the tables file"


def test_the_tables_file_alone_still_serves_a_named_directory(tmp_path, probe, monkeypatch):
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "no-checkout")
    prices = tmp_path / "prices.toml"
    prices.write_text('[virtual_spend]\nreference = "other-model"\nprice_in = 4\nprice_out = 8\n')
    out = io.StringIO()
    assert cli.main(["spend", str(probe), "--telegraph-config", str(prices), "--dir", str(tmp_path / "none")], env={}, stdout=out) == 0
    assert "at other-model" in out.getvalue()


def test_the_runs_root_comes_from_the_environment_config_when_a_tables_file_is_given(tmp_path):
    runs = tmp_path / "runs"
    write_log(runs, "org--b/stages/1/log/run-1.jsonl", [turn(1, 10, 1)])
    bringup = tmp_path / "bringup.toml"
    bringup.write_text(BASE)
    prices = tmp_path / "prices.toml"
    prices.write_text('[virtual_spend]\nreference = "x"\n')
    out = io.StringIO()
    code = cli.main(["spend", "--telegraph-config", str(prices), "--dir", str(tmp_path / "none")],
                    env={cli.CONFIG_ENV: str(bringup)}, stdout=out)
    assert code == 0 and "org--b: 1 turns" in out.getvalue()


def test_a_tables_file_that_is_missing_is_still_a_setup_error(tmp_path, probe):
    out = io.StringIO()
    code = cli.main(["spend", str(probe), "--telegraph-config", str(tmp_path / "gone.toml")], env={}, stdout=out)
    assert code == 2 and "cannot read" in out.getvalue()


def test_a_full_bringup_config_given_only_as_the_tables_file_still_gives_runs_root(tmp_path, monkeypatch):
    """With no bringup config to be found, the tables file is tried as one (a single file that holds everything)."""
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "no-checkout")
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    runs = tmp_path / "runs"
    write_log(runs, "org--c/stages/1/log/run-1.jsonl", [turn(1, 10, 1)])
    cfg = tmp_path / "everything.toml"
    cfg.write_text(BASE + '[virtual_spend]\nreference = "x"\n')
    out = io.StringIO()
    code = cli.main(["spend", "--telegraph-config", str(cfg), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    assert code == 0 and "org--c: 1 turns" in out.getvalue() and "at x" in out.getvalue()


def test_runs_root_comes_from_the_bringup_config_even_when_the_tables_file_is_a_full_config_too(tmp_path):
    good, other = tmp_path / "good-runs", tmp_path / "other-runs"
    write_log(good, "org--good/stages/1/log/run-1.jsonl", [turn(1, 10, 1)])
    write_log(other, "org--other/stages/1/log/run-1.jsonl", [turn(1, 10, 1)])
    bringup = tmp_path / "bringup.toml"
    bringup.write_text(BASE.replace('runs_root = "runs"', 'runs_root = "good-runs"'))
    tables = tmp_path / "tables.toml"
    tables.write_text(BASE.replace('runs_root = "runs"', 'runs_root = "other-runs"') + '[virtual_spend]\nreference = "t"\n')
    out = io.StringIO()
    code = cli.main(["spend", "--config", str(bringup), "--telegraph-config", str(tables), "--dir", str(tmp_path / "none")],
                    env={}, stdout=out)
    text = out.getvalue()
    assert code == 0 and "org--good" in text and "org--other" not in text and "at t" in text


# ---- review (sixth round): a ledger that cannot be read is unknown spend, and unsettled spend is shown apart ----------------

def _write_records(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "telegraph.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_an_unreadable_ledger_is_unknown_helper_spend_not_zero(tmp_path, probe):
    ledger = tmp_path / "tg"
    ledger.mkdir()
    (ledger / "telegraph.jsonl").mkdir()                                  # a read raises IsADirectoryError
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    real = text[text.index("REAL helper spend"):text.index("LOCAL virtual spend")]
    assert code == 0
    assert "cannot be read" in real and "unknown" in real.lower()
    assert "$0.0000" not in real and "0 calls" not in real
    assert "$0.1500" in text, "the local figures stay usable"


def test_an_unreadable_ledger_is_unknown_in_the_json_too(tmp_path, probe):
    ledger = tmp_path / "tg"
    ledger.mkdir()
    (ledger / "telegraph.jsonl").mkdir()
    code, text = run(tmp_path, [str(probe), "--json", "--dir", str(ledger)])
    data = json.loads(text)
    assert code == 0 and "cannot be read" in data["real_helper_spend"]["error"]
    assert "cost_usd" not in data["real_helper_spend"] and data["local_virtual_spend"]["directories"][0]["turns"] == 1


def test_a_missing_ledger_is_still_an_empty_history(tmp_path, probe):
    code, text = run(tmp_path, [str(probe), "--dir", str(tmp_path / "never-written")])
    assert code == 0 and "$0.0000 · 0 calls" in text and "cannot be read" not in text


def test_pending_reservations_are_disclosed_apart_from_confirmed_charges(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [
        {"reserve": "r1", "tokens": 4000, "cost_usd": 0.08, "problem": "p"},             # a request whose result was never recorded
        {"reserve": "r2", "tokens": 1000, "cost_usd": 0.02, "problem": "p"},
        {"call_id": "r2", "outcome": "ok", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "model": "m", "problem": "p"}])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    real = text[text.index("REAL helper spend"):text.index("LOCAL virtual spend")]
    assert "confirmed" in real and "$0.0100" in real
    assert "1 unsettled reservation" in real and "up to $0.0900" in real, real
    assert "may still be billed" in real


def test_a_ledger_with_only_a_reservation_still_shows_its_exposure(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [{"reserve": "r1", "tokens": 4000, "cost_usd": 0.08, "problem": "p"}])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    real = text[text.index("REAL helper spend"):text.index("LOCAL virtual spend")]
    assert "1 unsettled reservation" in real and "up to $0.0800" in real
    assert "confirmed: $0.0000" in real, "the confirmed charge is still zero; the exposure is stated apart"


def test_quarantined_torn_records_and_unknown_usage_are_disclosed(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [
        {"torn": "abc123", "tokens": 500},
        {"reserve": "u1", "tokens": 800, "cost_usd": 0.03, "problem": "p"},
        {"call_id": "u1", "outcome": "ok", "usage_known": False, "tokens_in": 0, "tokens_out": 0, "model": "m", "problem": "p"}])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    real = text[text.index("REAL helper spend"):text.index("LOCAL virtual spend")]
    assert "1 torn record" in real and "1 call with unknown usage" in real
    assert "of unknown cost" in real


def test_the_json_keeps_confirmed_and_possible_spend_in_separate_keys(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [
        {"reserve": "r1", "tokens": 4000, "cost_usd": 0.08, "problem": "p"},
        {"call_id": "c1", "outcome": "ok", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "model": "m", "problem": "p"},
        {"torn": "t1", "tokens": 10}])
    code, text = run(tmp_path, [str(probe), "--json", "--dir", str(ledger)])
    real = json.loads(text)["real_helper_spend"]
    assert real["cost_usd"] == 0.01 and real["calls"] == 1                 # confirmed charges only
    assert real["unsettled_reservations"] == 1 and real["torn_records"] == 1
    assert real["worst_case_cost_usd"] == pytest.approx(0.09)
    assert real["worst_case_unknown_cost_calls"] == 1


def test_a_clean_ledger_has_no_pending_lines(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [{"call_id": "c1", "outcome": "ok", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "model": "m", "problem": "p"}])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    assert "unsettled" not in text and "torn" not in text and "unknown usage" not in text


# ---- review (sixth round, sweep): the default run listing is read beneath a pinned runs folder ----------------------------------

def test_the_default_listing_does_not_follow_a_run_folder_swapped_for_a_link_after_discovery(tmp_path, monkeypatch):
    from orchard import virtualspend_view as view
    runs = tmp_path / "runs"
    write_log(runs, "org--a/stages/1/log/run-1.jsonl", [turn(1, 5, 1)])
    outside = tmp_path / "outside"
    write_log(outside, "stages/1/log/run-1.jsonl", [turn(1, 999, 9)])
    real = view.discover

    def discover_then_swap(root, *a):
        found = real(root, *a)
        import shutil
        shutil.rmtree(runs / "org--a")
        (runs / "org--a").symlink_to(outside, target_is_directory=True)
        return found
    monkeypatch.setattr(view, "discover", discover_then_swap)
    bringup = tmp_path / "bringup.toml"
    bringup.write_text(BASE)
    out = io.StringIO()
    code = cli.main(["spend", "--config", str(bringup), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    text = out.getvalue()
    assert code == 0 and "999" not in text and "1 turns" not in text


def test_the_default_listing_includes_the_kernel_probe_outputs_and_reads_them_beneath_the_pin(tmp_path):
    runs = tmp_path / "runs"
    write_log(runs, "org--a/stages/1/log/run-1.jsonl", [turn(1, 5, 1)])
    write_log(runs, "kprobe/p1/op/attempt-1/log/probe-1.jsonl", [turn(1, 7, 1)])
    bringup = tmp_path / "bringup.toml"
    bringup.write_text(BASE)
    out = io.StringIO()
    code = cli.main(["spend", "--config", str(bringup), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    text = out.getvalue()
    assert code == 0 and "org--a: 1 turns" in text and "kprobe/p1: 1 turns" in text
    assert "all 2 directories: 2 turns" in text


def test_the_default_listing_says_when_the_folder_listing_was_cut_short(tmp_path, monkeypatch):
    from orchard import virtualspend_view as view
    monkeypatch.setattr(view, "MAX_NAMES", 2)
    runs = tmp_path / "runs"
    for i in range(6):
        write_log(runs, f"org--{i}/log/run.jsonl", [turn()])
    bringup = tmp_path / "bringup.toml"
    bringup.write_text(BASE)
    out = io.StringIO()
    cli.main(["spend", "--config", str(bringup), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    assert "The folder listing was cut short" in out.getvalue()
    out = io.StringIO()
    cli.main(["spend", "--json", "--config", str(bringup), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    assert json.loads(out.getvalue())["local_virtual_spend"]["listing_cut_short"] is True


def test_an_unreadable_runs_folder_is_a_setup_error_not_an_empty_listing(tmp_path):
    bringup = tmp_path / "bringup.toml"
    bringup.write_text(BASE)                                                # runs_root = "runs" does not exist
    out = io.StringIO()
    code = cli.main(["spend", "--config", str(bringup), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    assert code == 2 and "name at least one directory" in out.getvalue()


def test_an_incomplete_scan_is_marked_in_the_text_and_the_json(tmp_path, monkeypatch, probe):
    from orchard import virtualspend as virtual
    monkeypatch.setattr(virtual, "MAX_DEPTH", 1)
    write_log(probe, "op/attempt-2/log/probe-1.jsonl", [turn(1, 5, 1)])      # below the depth limit
    code, text = run(tmp_path, [str(probe), "--dir", str(tmp_path / "none")])
    assert "INCOMPLETE" in text and "folders below the depth limit" in text
    code, js = run(tmp_path, [str(probe), "--json", "--dir", str(tmp_path / "none")])
    assert json.loads(js)["local_virtual_spend"]["directories"][0]["complete"] is False


# ---- cases added after the sixth mutation table ----------------------------------------------------------------------------

def test_a_clean_ledger_prints_no_not_confirmed_line_at_all(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [{"call_id": "c1", "outcome": "ok", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "model": "m", "problem": "p"}])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    assert "not confirmed" not in text and "worst case" not in text


def _runs_config(tmp_path):
    bringup = tmp_path / "bringup.toml"
    bringup.write_text(BASE)
    return bringup


def test_a_probe_folder_swapped_for_a_link_after_discovery_is_not_read_by_the_default_listing(tmp_path, monkeypatch):
    import shutil
    from orchard import virtualspend_view as view
    runs = tmp_path / "runs"
    write_log(runs, "kprobe/p1/op/attempt-1/log/probe-1.jsonl", [turn(1, 5, 1)])
    outside = tmp_path / "outside"
    write_log(outside, "p1/op/attempt-1/log/probe-1.jsonl", [turn(1, 999, 9)])
    real = view.discover

    def discover_then_swap(root, *a):
        found = real(root, *a)
        shutil.rmtree(runs / "kprobe")
        (runs / "kprobe").symlink_to(outside, target_is_directory=True)
        return found
    monkeypatch.setattr(view, "discover", discover_then_swap)
    out = io.StringIO()
    cli.main(["spend", "--config", str(_runs_config(tmp_path)), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    assert "999" not in out.getvalue() and "1 turns" not in out.getvalue()


def test_the_default_listing_lists_the_pinned_original_when_the_runs_folder_is_replaced_after_pinning(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    write_log(runs, "org--a/log/run.jsonl", [turn(1, 5, 1)])
    other = tmp_path / "other-runs"
    write_log(other, "org--z/log/run.jsonl", [turn(1, 999, 9)])
    real_pin = spend_cmd.vs.open_pin

    def pin_then_swap(path):
        pin = real_pin(path)
        Path(path).rename(Path(path).parent / "moved-runs")
        Path(path).symlink_to(other, target_is_directory=True)
        return pin
    monkeypatch.setattr(spend_cmd.vs, "open_pin", pin_then_swap)
    out = io.StringIO()
    cli.main(["spend", "--config", str(_runs_config(tmp_path)), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    text = out.getvalue()
    assert "org--a: 1 turns" in text and "org--z" not in text and "999" not in text


def test_a_runs_folder_that_cannot_be_pinned_is_a_setup_error(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(spend_cmd.vs, "open_pin", lambda path: None)
    out = io.StringIO()
    code = cli.main(["spend", "--config", str(_runs_config(tmp_path)), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    assert code == 2 and "cannot be opened" in out.getvalue()


def test_a_folder_read_in_part_is_listed_even_with_no_turns(tmp_path, monkeypatch):
    from orchard import virtualspend as virtual
    monkeypatch.setattr(virtual, "MAX_DEPTH", 2)
    runs = tmp_path / "runs"
    write_log(runs, "org--deep/x/y/log/run.jsonl", [turn(1, 5, 1)])         # everything is below the depth limit
    out = io.StringIO()
    cli.main(["spend", "--config", str(_runs_config(tmp_path)), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    assert "org--deep: 0 turns" in out.getvalue() and "INCOMPLETE" in out.getvalue()


def test_the_default_listing_leaves_no_descriptor_open(tmp_path):
    runs = tmp_path / "runs"
    write_log(runs, "org--a/log/run.jsonl", [turn()])
    write_log(runs, "kprobe/p/op/attempt-1/log/p.jsonl", [turn()])
    config = _runs_config(tmp_path)
    before = set(os.listdir("/proc/self/fd"))
    cli.main(["spend", "--config", str(config), "--dir", str(tmp_path / "none")], env={}, stdout=io.StringIO())
    assert set(os.listdir("/proc/self/fd")) == before


def test_a_cost_that_overflows_is_reported_for_that_folder_and_left_out_of_the_totals(tmp_path):
    big, small = tmp_path / "big", tmp_path / "small"
    write_log(big, "log/run.jsonl", [turn(1, 1_000_000, 0)])
    write_log(small, "log/run.jsonl", [turn(1, 10, 0)])
    code, text = run(tmp_path, [str(big), str(small), "--dir", str(tmp_path / "none")], "[virtual_spend]\nprice_in = 1e307\n")
    assert code == 0
    assert "cannot be priced" in text and "not finite" in text
    assert "inf" not in text.lower().replace("information", "") and "nan" not in text.lower()
    code, js = run(tmp_path, [str(big), str(small), "--json", "--dir", str(tmp_path / "none")], "[virtual_spend]\nprice_in = 1e307\n")
    data = json.loads(js)                                                    # strict: no NaN or Infinity in the output
    rows = {Path(r["dir"]).name: r for r in data["local_virtual_spend"]["directories"]}
    assert "not finite" in rows["big"]["error"] and "error" not in rows["small"]
    assert data["local_virtual_spend"]["all_directories"]["turns"] == 1


def test_sound_folders_have_no_error_key(tmp_path, probe):
    code, js = run(tmp_path, [str(probe), "--json", "--dir", str(tmp_path / "none")])
    assert "error" not in json.loads(js)["local_virtual_spend"]["directories"][0]


def test_a_total_that_is_not_finite_is_unknown_in_the_text_and_the_json(tmp_path, probe, monkeypatch):
    from orchard import virtualspend as virtual
    real = virtual.cost
    calls = []

    def cost_then_nan(p, c, pi, po):
        calls.append(1)
        return real(p, c, pi, po) if len(calls) == 1 else float("nan")      # the folder's own cost is sound; the total's is not
    monkeypatch.setattr(virtual, "cost", cost_then_nan)
    code, text = run(tmp_path, [str(probe), "--dir", str(tmp_path / "none")])
    assert code == 0 and "$0.1500" in text and "nan" not in text.lower()
    calls.clear()
    code, js = run(tmp_path, [str(probe), "--json", "--dir", str(tmp_path / "none")])
    data = json.loads(js)                                                    # strict JSON
    assert data["local_virtual_spend"]["all_directories"]["virtual"][0]["cost_usd"] is None


# ---- review (seventh round): damaged ledger numbers are unknown spend, and estimates are not confirmed charges ---------------

def _write_raw(directory, lines):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "telegraph.jsonl").write_text("".join(l + "\n" for l in lines))


@pytest.mark.parametrize("cost", ["NaN", "Infinity", "-Infinity", "-0.5"], ids=["nan", "inf", "-inf", "negative"])
def test_a_damaged_cost_in_the_ledger_is_unknown_spend_not_a_number(tmp_path, probe, cost):
    ledger = tmp_path / "tg"
    _write_raw(ledger, ['{"call_id": "c1", "outcome": "ok", "tokens_in": 10, "tokens_out": 5, "cost_usd": %s, "model": "m", "problem": "p"}' % cost])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    real = text[text.index("REAL helper spend"):text.index("LOCAL virtual spend")]
    assert code == 0 and "unknown" in real.lower() and "cannot be read" in real
    assert "nan" not in real.lower() and "inf" not in real.lower() and "$-" not in real
    code, js = run(tmp_path, [str(probe), "--json", "--dir", str(ledger)])
    data = json.loads(js)                                                    # strict JSON: no NaN or Infinity literal
    assert "error" in data["real_helper_spend"] and "cost_usd" not in data["real_helper_spend"]


def test_a_damaged_token_count_in_the_reservation_is_unknown_spend(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_raw(ledger, ['{"reserve": "r1", "tokens": NaN, "cost_usd": 0.01, "problem": "p"}'])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    assert "cannot be read" in text and "nan" not in text.lower()


def test_a_sound_ledger_is_not_refused_by_the_check(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_raw(ledger, ['{"call_id": "c1", "outcome": "ok", "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.5, "model": "m", "problem": "p"}'])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    assert "confirmed: $0.5000" in text and "cannot be read" not in text


def test_an_estimate_for_a_call_with_unknown_usage_is_not_a_confirmed_charge(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [
        {"call_id": "u1", "outcome": "ok", "usage_known": False, "tokens_in": 7, "tokens_out": 3, "cost_usd": 0.0001, "model": "m", "problem": "p"}])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    real = text[text.index("REAL helper spend"):text.index("LOCAL virtual spend")]
    assert "confirmed: $0.0000 · 0 calls" in real, real
    assert "1 call with unknown usage" in real and "up to $0.0001" in real
    code, js = run(tmp_path, [str(probe), "--json", "--dir", str(ledger)])
    data = json.loads(js)["real_helper_spend"]
    assert data["cost_usd"] == 0 and data["calls"] == 0 and data["tokens_in"] == 0
    assert data["unknown_usage_calls"] == 1 and data["worst_case_cost_usd"] == pytest.approx(0.0001)


def test_an_estimate_beside_its_reservation_counts_at_the_reservation_and_not_as_confirmed(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [
        {"reserve": "u1", "tokens": 800, "cost_usd": 0.03, "problem": "p"},
        {"call_id": "u1", "outcome": "ok", "usage_known": False, "tokens_in": 7, "tokens_out": 3, "cost_usd": 0.0001, "model": "m", "problem": "p"},
        {"call_id": "c2", "outcome": "ok", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "model": "m", "problem": "p"}])
    code, text = run(tmp_path, [str(probe), "--dir", str(ledger)])
    real = text[text.index("REAL helper spend"):text.index("LOCAL virtual spend")]
    assert "confirmed: $0.0100 · 1 calls" in real
    assert "1 call with unknown usage" in real and "up to $0.0400" in real, real


def test_calls_with_known_usage_stay_confirmed_when_another_call_is_unknown(tmp_path, probe):
    ledger = tmp_path / "tg"
    _write_records(ledger, [
        {"call_id": "c1", "outcome": "ok", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "model": "m", "problem": "p"},
        {"call_id": "u1", "outcome": "ok", "usage_known": False, "cost_usd": 0.02, "model": "m", "problem": "p"}])
    code, js = run(tmp_path, [str(probe), "--json", "--dir", str(ledger)])
    data = json.loads(js)["real_helper_spend"]
    assert data["calls"] == 1 and data["cost_usd"] == 0.01 and data["tokens_in"] == 100


# ---- small virtual costs stay visible -----------------------------------------------------------------------------------------

def test_a_tiny_nonzero_virtual_cost_is_not_shown_as_zero(tmp_path):
    tiny = tmp_path / "tiny"
    write_log(tiny, "log/run.jsonl", [turn(1, 100, 0)])                       # 100 prompt tokens at $0.10 per million = $0.00001
    code, text = run(tmp_path, [str(tiny), "--dir", str(tmp_path / "none")])
    assert "<$0.0001" in text and "$0.0000" not in text.split("LOCAL virtual spend")[1]


def test_a_zero_virtual_cost_is_still_zero(tmp_path):
    empty = tmp_path / "empty"
    write_log(empty, "log/run.jsonl", ["{bad"])
    code, text = run(tmp_path, [str(empty), "--dir", str(tmp_path / "none")])
    assert "$0.0000" in text.split("LOCAL virtual spend")[1]


def test_a_folder_with_only_unreadable_logs_is_listed_and_marked_incomplete(tmp_path):
    runs = tmp_path / "runs"
    dead = write_log(runs, "org--dead/log/run.jsonl", [turn(1, 7, 1)])
    os.chmod(dead, 0)
    try:
        if os.access(dead, os.R_OK):
            pytest.skip("running as a user that can read mode 0 files")
        out = io.StringIO()
        cli.main(["spend", "--config", str(_runs_config(tmp_path)), "--dir", str(tmp_path / "none")], env={}, stdout=out)
    finally:
        os.chmod(dead, 0o644)
    text = out.getvalue()
    assert "org--dead: 0 turns" in text and "INCOMPLETE" in text and "files that could not be read" in text
