# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard reported show|accept RUN`: a person turns draft rows from the model card into reported-results.json (issue 19).

Stage 0 drafts rows (tests/test_reported_stage0.py). Nothing reaches a package card until `accept` copies rows into
`<run>/reported-results.json`, and the ledger records who accepted what. The command reads files in the run. It
opens no network connection and starts nothing.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import cli, reported_cmd
from orchard.ledger import Ledger, read_entries
from orchard.package import REPORTED_FILE, read_reported, read_reported_file

CARD = "# Wren\n\n| Model | MMLU | GSM8K |\n|---|---|---|\n| Wren-7B | 71.5 | 82.25 |\n"
ROWS = [
    {"name": "MMLU", "value": "71.5", "note": "model card table, line 5",
     "extracted_from": {"kind": "table", "line": 5, "text": "| Wren-7B | 71.5 | 82.25 |"}},
    {"name": "GSM8K", "value": "82.25", "note": "model card table, line 5",
     "extracted_from": {"kind": "table", "line": 5, "text": "| Wren-7B | 71.5 | 82.25 |"}},
]


def make_run(tmp_path, *, rows=None, card=CARD, snapshot=True, draft=True):
    run = tmp_path / "run"
    (run / "stages" / "0" / "evidence").mkdir(parents=True)
    with Ledger(run / "ledger.jsonl") as led:
        led.append("run_start", None, model="Acme/Wren-7B", package=None, paths={})
    snap = tmp_path / "snap"
    snap.mkdir()
    if card is not None:
        (snap / "README.md").write_text(card)
    if snapshot:
        (run / "stages" / "0" / "triage_config.json").write_text(json.dumps({"model_snapshot": str(snap)}))
    if draft:
        (run / "stages" / "0" / "evidence" / "reported-results.draft.json").write_text(json.dumps({
            "model": "Acme/Wren-7B", "source": "the Acme/Wren-7B model card (README.md)",
            "rows": ROWS if rows is None else rows, "skipped": [], "skipped_dropped": 0, "other_model_rows": 0}))
    return run


def args(run, action="accept", **kw):
    base = dict(action=action, run_dir=str(run), all=False, rows=None, row=None, replace=False)
    base.update(kw)
    return SimpleNamespace(**base)


def go(run, action="accept", **kw):
    out = io.StringIO()
    code = reported_cmd.main(args(run, action, **kw), out=out)
    return code, out.getvalue()


def decisions(run):
    return [e for e in read_entries(run / "ledger.jsonl")
            if e["event"] == "decision" and e["data"].get("decision") == "reported scores accepted"]


@pytest.fixture(autouse=True)
def who(monkeypatch):
    monkeypatch.setattr(reported_cmd.getpass, "getuser", lambda: "alice")


# ---- accept ------------------------------------------------------------------------------------------------------

def test_accept_all_writes_reported_results_that_the_card_reader_accepts(tmp_path):
    run = make_run(tmp_path)
    code, out = go(run, all=True)
    assert code == 0, out
    rows, source = read_reported(run)
    assert [(r.name, r.value) for r in rows] == [("MMLU", "71.5"), ("GSM8K", "82.25")]
    assert source == "the Acme/Wren-7B model card (README.md)"
    assert "model card table, line 5" in rows[0].note
    assert "nothing was published" in out.lower() or "nothing is published" in out.lower()
    assert "package-card" in out


def test_the_written_file_holds_only_what_the_card_reader_needs(tmp_path):
    run = make_run(tmp_path)
    go(run, all=True)
    data = json.loads((run / REPORTED_FILE).read_text())
    assert set(data) == {"source", "rows"}
    assert all(set(r) == {"name", "value", "note"} for r in data["rows"])


def test_accept_rows_copies_only_the_named_rows(tmp_path):
    run = make_run(tmp_path)
    code, _ = go(run, rows="GSM8K")
    assert code == 0
    assert [r.name for r in read_reported(run)[0]] == ["GSM8K"]


def test_accept_rows_takes_a_comma_list_and_row_can_repeat(tmp_path):
    run = make_run(tmp_path)
    assert go(run, rows="MMLU,GSM8K")[0] == 0
    assert [r.name for r in read_reported(run)[0]] == ["MMLU", "GSM8K"]          # the draft's order


def test_accept_row_flag_names_one_row_exactly_even_with_a_comma(tmp_path):
    rows = [{**ROWS[0], "name": "A, B", "extracted_from": {"kind": "table", "line": 5, "text": "| Wren-7B | 71.5 |"}}]
    run = make_run(tmp_path, rows=rows, card=None)
    assert go(run, row=["A, B"])[0] == 0
    assert [r.name for r in read_reported(run)[0]] == ["A, B"]


def test_an_unknown_row_name_is_refused_and_lists_the_names_there_are(tmp_path):
    run = make_run(tmp_path)
    code, out = go(run, rows="MMLU,Nope")
    assert code == 2
    assert "Nope" in out and "MMLU" in out and "GSM8K" in out
    assert not (run / REPORTED_FILE).exists() and decisions(run) == []


def test_without_a_draft_the_command_refuses(tmp_path):
    run = make_run(tmp_path, draft=False)
    code, out = go(run, all=True)
    assert code == 2 and "reported-results.draft.json" in out


def test_a_directory_with_no_ledger_is_refused(tmp_path):
    code, out = go(tmp_path / "nowhere", all=True)
    assert code == 2 and "ledger" in out


def test_an_empty_draft_has_nothing_to_accept(tmp_path):
    run = make_run(tmp_path, rows=[])
    code, out = go(run, all=True)
    assert code == 2 and "no rows" in out


def test_a_damaged_draft_is_refused(tmp_path):
    run = make_run(tmp_path)
    (run / "stages/0/evidence/reported-results.draft.json").write_text("{not json")
    assert go(run, all=True)[0] == 2
    (run / "stages/0/evidence/reported-results.draft.json").write_text(json.dumps({"rows": "x", "source": "s"}))
    assert go(run, all=True)[0] == 2


def test_the_ledger_records_who_accepted_what_and_which_draft(tmp_path):
    run = make_run(tmp_path)
    go(run, rows="MMLU")
    [d] = decisions(run)
    assert d["data"]["by"] == "alice"
    assert d["data"]["names"] == ["MMLU"]
    assert d["data"]["count"] == 1 and d["data"]["of"] == 2
    draft = run / "stages/0/evidence/reported-results.draft.json"
    assert d["data"]["draft_sha256"] == hashlib.sha256(draft.read_bytes()).hexdigest()
    assert d["data"]["verified_against_card"] is True
    assert d["data"]["replaced"] is False
    assert d["stage"] == 0 or d["stage"] is None


def test_an_existing_different_file_is_kept_unless_replace_is_given(tmp_path):
    run = make_run(tmp_path)
    (run / REPORTED_FILE).write_text(json.dumps({"source": "by hand", "rows": [{"name": "X", "value": "1"}]}))
    code, out = go(run, all=True)
    assert code == 2 and "--replace" in out
    assert read_reported(run)[1] == "by hand" and decisions(run) == []
    assert go(run, all=True, replace=True)[0] == 0
    assert read_reported(run)[1] == "the Acme/Wren-7B model card (README.md)"
    assert decisions(run)[-1]["data"]["replaced"] is True


def test_accepting_the_same_rows_again_changes_nothing(tmp_path):
    run = make_run(tmp_path)
    go(run, all=True)
    before = (run / REPORTED_FILE).read_text()
    assert go(run, all=True)[0] == 0
    assert (run / REPORTED_FILE).read_text() == before


def test_a_run_a_supervisor_still_holds_is_refused(tmp_path):
    run = make_run(tmp_path)
    with Ledger(run / "ledger.jsonl"):
        code, out = go(run, all=True)
    assert code == 2 and "supervisor" in out
    assert not (run / REPORTED_FILE).exists()


# ---- the draft is checked again, because the stage 0 agent can write to it --------------------------------------

def test_a_row_whose_quoted_card_text_does_not_hold_the_value_is_refused(tmp_path):
    forged = [{**ROWS[0], "value": "99.9"}]
    run = make_run(tmp_path, rows=forged)
    code, out = go(run, all=True)
    assert code == 2 and "MMLU" in out and "99.9" in out
    assert not (run / REPORTED_FILE).exists()


def test_a_value_that_is_not_a_plain_number_is_refused_whatever_the_quote_says(tmp_path):
    bad = [{**ROWS[0], "value": "81.8 (best of 3)",
            "extracted_from": {"kind": "table", "line": 5, "text": "| Wren-7B | 81.8 (best of 3) |"}}]
    run = make_run(tmp_path, rows=bad, card="# x\n\n| Model | MMLU |\n|---|---|\n| Wren-7B | 81.8 (best of 3) |\n")
    assert go(run, all=True)[0] == 2


@pytest.mark.parametrize("value", ["nan", "inf", "1e999", "", "n/a"])
def test_non_finite_and_empty_values_are_refused(tmp_path, value):
    bad = [{**ROWS[0], "value": value, "extracted_from": {"kind": "table", "line": 5, "text": f"| Wren-7B | {value} |"}}]
    run = make_run(tmp_path, rows=bad, card=None)           # no card on disk: the number rule alone refuses it
    code, out = go(run, all=True)
    assert code == 2 and "not a plain number" in out


def test_a_row_the_card_on_disk_does_not_give_is_refused(tmp_path):
    run = make_run(tmp_path, card="# Wren\n\nnothing here\n")
    code, out = go(run, all=True)
    assert code == 2 and "model card on disk does not give" in out and "MMLU" in out


def test_a_model_index_row_is_checked_against_its_line_too(tmp_path):
    card = "---\nmodel-index:\n  - name: x\n    results:\n      - metrics:\n          - type: a\n            value: 84.2\n---\n"
    row = {"name": "unnamed dataset / a", "value": "84.2", "note": "model card model-index, line 7",
           "extracted_from": {"kind": "model-index", "line": 7, "text": "value: 84.2"}}
    run = make_run(tmp_path, rows=[row], card=card)
    assert go(run, all=True)[0] == 0
    run2 = make_run(tmp_path / "two", rows=[{**row, "value": "85.2",
                                             "extracted_from": {**row["extracted_from"], "text": "value: 85.2"}}], card=card)
    assert go(run2, all=True)[0] == 2


def test_without_the_card_on_disk_the_rows_are_accepted_and_the_ledger_says_they_were_not_checked(tmp_path):
    run = make_run(tmp_path, card=None)
    code, out = go(run, all=True)
    assert code == 0 and "not checked against it" in out
    assert decisions(run)[0]["data"]["verified_against_card"] is False


def test_a_draft_with_a_repeated_name_or_too_many_rows_is_refused_by_the_card_rules(tmp_path):
    run = make_run(tmp_path, rows=[ROWS[0], ROWS[0]])
    code, out = go(run, all=True)
    assert code == 2 and "appears twice" in out
    many = [{**ROWS[0], "name": f"r{i}"} for i in range(61)]
    run2 = make_run(tmp_path / "two", rows=many)
    assert go(run2, all=True)[0] == 2


def test_a_refusal_writes_nothing_and_leaves_no_temp_file(tmp_path):
    run = make_run(tmp_path, rows=[{**ROWS[0], "value": "99.9"}])
    before = sorted(p.name for p in run.iterdir())
    go(run, all=True)
    assert sorted(p.name for p in run.iterdir()) == before


def test_the_command_opens_no_network_connection(tmp_path, monkeypatch):
    import socket

    def refuse(*a, **k):
        raise RuntimeError("network use is not allowed")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    run = make_run(tmp_path)
    assert go(run, all=True)[0] == 0 and go(run, "show")[0] == 0


# ---- show --------------------------------------------------------------------------------------------------------

def test_show_lists_the_rows_with_their_lines_and_changes_nothing(tmp_path):
    run = make_run(tmp_path)
    before = (run / "ledger.jsonl").read_bytes()
    code, out = go(run, "show")
    assert code == 0
    assert "MMLU" in out and "71.5" in out and "line 5" in out and "| Wren-7B | 71.5 | 82.25 |" in out
    assert "not confirmed" in out
    assert (run / "ledger.jsonl").read_bytes() == before and not (run / REPORTED_FILE).exists()


def test_show_works_while_a_supervisor_holds_the_run(tmp_path):
    run = make_run(tmp_path)
    with Ledger(run / "ledger.jsonl"):
        assert go(run, "show")[0] == 0


def test_show_says_how_many_entries_were_skipped_and_why(tmp_path):
    run = make_run(tmp_path)
    p = run / "stages/0/evidence/reported-results.draft.json"
    d = json.loads(p.read_text())
    d["skipped"] = [{"reason": "the cell is not a plain number (58.0*)", "line": 9, "text": "| x |"}]
    p.write_text(json.dumps(d))
    code, out = go(run, "show")
    assert "1 entry was left out" in out and "58.0*" in out


def test_show_marks_the_rows_a_package_card_already_uses(tmp_path):
    run = make_run(tmp_path)
    go(run, rows="MMLU")
    out = go(run, "show")[1]
    assert "accepted" in out.lower()


# ---- the command line --------------------------------------------------------------------------------------------

def test_the_cli_passes_the_flags_through(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(reported_cmd, "main", lambda a, **kw: seen.update(vars(a)) or 0)
    monkeypatch.setenv("ORCHARD_BRINGUP_CONFIG", str(tmp_path / "none.toml"))
    code = cli.main(["reported", "accept", str(tmp_path / "run"), "--rows", "A,B", "--replace"],
                    stdout=io.StringIO())
    assert code == 0
    assert (seen["action"], seen["run_dir"], seen["rows"], seen["row"], seen["replace"], seen["all"]) == \
        ("accept", str(tmp_path / "run"), "A,B", None, True, False)
    cli.main(["reported", "accept", str(tmp_path / "run"), "--row", "C", "--row", "D"], stdout=io.StringIO())
    assert seen["row"] == ["C", "D"] and seen["rows"] is None and seen["replace"] is False


def test_accept_needs_all_or_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHARD_BRINGUP_CONFIG", str(tmp_path / "none.toml"))
    with pytest.raises(SystemExit):
        cli.main(["reported", "accept", str(tmp_path / "run")], stdout=io.StringIO())
    with pytest.raises(SystemExit):
        cli.main(["reported", "accept", str(tmp_path / "run"), "--all", "--rows", "A"], stdout=io.StringIO())


def test_the_accepted_file_is_read_by_the_package_card_reader_unchanged(tmp_path):
    run = make_run(tmp_path)
    go(run, all=True)
    rows, source = read_reported_file(run / REPORTED_FILE)
    assert len(rows) == 2 and source


# ---- more guards -------------------------------------------------------------------------------------------------

def test_a_row_whose_line_is_not_the_cards_line_is_refused(tmp_path):
    moved = [{**ROWS[0], "extracted_from": {**ROWS[0]["extracted_from"], "line": 9}}]
    run = make_run(tmp_path, rows=moved)
    code, out = go(run, all=True)
    assert code == 2 and "line 9" in out


def test_without_a_user_name_the_ledger_says_unknown(tmp_path, monkeypatch):
    def boom():
        raise KeyError("no user")
    monkeypatch.setattr(reported_cmd.getpass, "getuser", boom)
    run = make_run(tmp_path)
    assert go(run, all=True)[0] == 0
    assert decisions(run)[0]["data"]["by"] == "unknown"


def test_show_says_how_many_comparison_rows_were_ignored(tmp_path):
    run = make_run(tmp_path)
    p = run / "stages/0/evidence/reported-results.draft.json"
    d = json.loads(p.read_text())
    d["other_model_rows"] = 42
    p.write_text(json.dumps(d))
    assert "42 comparison-table rows were about other models" in go(run, "show")[1]
