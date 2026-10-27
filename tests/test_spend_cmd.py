# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard spend` (orchard/spend_cmd.py): the real helper spend and the virtual local spend, side by side.

The real spend comes from a ledger written in the producer's shape; the local figure from fake agent logs. The two totals are
on separate lines and no figure adds them."""
import io
import json

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
    code = cli.main(["spend", *argv, "--telegraph-config", str(cfg)], env={}, stdout=out)
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
    monkeypatch.setattr(spend_cmd.tg, "summarise", boom)
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
