# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`python3 -m orchard.prompt_suite swap-check.json` prints what the gates read in a swap-check.json: the reasons that fail a
stage, the advisory findings (recorded only) and the summary. Operators and agents use it to see the findings a gate
does not fail on."""
import json
import subprocess
import sys

from orchard import prompt_suite as ps
from test_prompt_suite_verdict import LOOP, make_record, make_report, with_entry


def run(path):
    return subprocess.run([sys.executable, "-m", "orchard.prompt_suite", str(path)], capture_output=True, text=True, timeout=60)


def write(tmp_path, report):
    p = tmp_path / "swap-check.json"
    p.write_text(json.dumps(report))
    return p


def test_a_healthy_record_prints_no_reasons_and_exits_0(tmp_path):
    r = run(write(tmp_path, make_report(make_record())))
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout)
    assert out["reasons"] == [] and out["advisory"] == [] and out["summary"]["prompts"] == len(ps.load_suite()["prompts"])


def test_a_failing_record_prints_the_reasons_and_exits_1(tmp_path):
    r = run(write(tmp_path, make_report(with_entry(make_record(), "arithmetic", content=LOOP))))
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert len(out["reasons"]) == 1 and "arithmetic" in out["reasons"][0]


def test_advisory_findings_are_printed_and_do_not_fail(tmp_path):
    r = run(write(tmp_path, make_report(with_entry(make_record(), "chat-capital", content=""))))
    assert r.returncode == 0
    assert any("empty_output" in a for a in json.loads(r.stdout)["advisory"])


def test_a_report_without_a_record_says_so_and_exits_0(tmp_path):
    r = run(write(tmp_path, make_report()))
    assert r.returncode == 0 and json.loads(r.stdout)["summary"] == {"ran": False}


def test_a_file_that_cannot_be_read_exits_2_with_a_message(tmp_path):
    r = run(tmp_path / "missing.json")
    assert r.returncode == 2 and "cannot read" in r.stderr
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert run(bad).returncode == 2


def test_main_returns_the_exit_code_without_exiting(tmp_path, capsys):
    assert ps.main([str(write(tmp_path, make_report(make_record())))]) == 0
    assert json.loads(capsys.readouterr().out)["reasons"] == []
