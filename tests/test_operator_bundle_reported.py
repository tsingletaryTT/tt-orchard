# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 8 bundle and the scores the model's authors report (issue 19).

RESULTS.md gets one section for the rows a person accepted into `reported-results.json`, marked as the authors' own claims
and not measured here. A draft nobody accepted gets one line and no rows. A run with neither gets the same RESULTS.md as
before. A reported row can never be a measured one: the section's label is never `measured`, the gate does not read RESULTS.md
numbers, and the operator view's verdict reads only measured result files (tests below).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_template import (GOLDEN, build, bundle_of, make_run, section, write, write_json)  # noqa: E402

from orchard import run_view, verdict  # noqa: E402
from orchard.stages import gate_bundle  # noqa: E402

HEADING = "## Reported by the authors (not measured here)"
REPORTED = {"source": "the Acme/Wren-7B model card (README.md)",
            "rows": [{"name": "MLE-Lite", "value": "81.8", "note": "model card table, line 85"},
                     {"name": "HLE", "value": "52.6", "note": "model card table, line 136"}]}
DRAFT = {"model": "Acme/Wren-7B", "source": "s", "rows": [{"name": "MLE-Lite", "value": "81.8", "note": ""},
                                                          {"name": "HLE", "value": "52.6", "note": ""},
                                                          {"name": "GAIA", "value": "92.2", "note": ""}]}


def results_of(run) -> str:
    r = build(run)
    assert r.returncode == 0, r.stdout + r.stderr
    return (bundle_of(run) / "RESULTS.md").read_text()


def test_accepted_rows_get_their_own_section_labelled_as_reported(tmp_path):
    run = make_run(tmp_path)
    write_json(run / "reported-results.json", REPORTED)
    text = results_of(run)
    body = section(text, HEADING)
    assert "the Acme/Wren-7B model card (README.md)" in body
    assert "| MLE-Lite | 81.8 |" in body and "| HLE | 52.6 |" in body
    assert "model card table, line 85" in body
    assert "not measured" in body.lower()
    assert "measured |" not in body.replace("not measured", "")          # no row carries the label `measured`
    assert body.count("| reported, not measured here |") == 2


def test_the_section_says_no_pass_bar_or_verdict_uses_the_rows(tmp_path):
    run = make_run(tmp_path)
    write_json(run / "reported-results.json", REPORTED)
    body = section(results_of(run), HEADING)
    assert "no pass bar and no verdict" in " ".join(body.split()).lower()


def test_the_rows_are_not_counted_as_measured_anywhere_else_in_the_bundle(tmp_path):
    run = make_run(tmp_path)
    before = results_of(run)
    write_json(run / "reported-results.json", REPORTED)
    after = results_of(run)
    start = after.index(HEADING)
    assert after[:start] + after[after.index("## Stage 0", start):] == before      # the section is the only change
    for name in ("RISKS.md", "card.md"):
        assert "81.8" not in (bundle_of(run) / name).read_text()


def test_a_draft_nobody_accepted_gets_one_line_and_no_rows(tmp_path):
    run = make_run(tmp_path)
    write_json(run / "stages" / "0" / "evidence" / "reported-results.draft.json", DRAFT)
    text = results_of(run)
    body = section(text, HEADING)
    assert "3 rows" in body and "nobody accepted" in body.lower()
    assert "81.8" not in text and "92.2" not in text
    assert "tt-orchard reported accept" in body


def test_accepted_rows_and_a_larger_draft_say_how_many_were_accepted(tmp_path):
    run = make_run(tmp_path)
    write_json(run / "reported-results.json", REPORTED)
    write_json(run / "stages" / "0" / "evidence" / "reported-results.draft.json", DRAFT)
    body = section(results_of(run), HEADING)
    assert "drafted 3 rows" in body and "2 were accepted" in body
    assert "92.2" not in body                                             # the third row was not accepted


def test_a_run_with_no_reported_scores_gets_the_same_results_as_before(tmp_path):
    run = make_run(tmp_path)
    assert HEADING not in results_of(run)


def test_an_invalid_reported_file_is_named_and_no_row_is_listed(tmp_path):
    run = make_run(tmp_path)
    write(run / "reported-results.json", '{"source": "x", "rows": []}')
    body = section(results_of(run), HEADING)
    assert "reported-results.json" in body and "not valid" in body
    assert "| MLE-Lite" not in body
    write(run / "reported-results.json", "{not json")
    assert "not valid" in section(results_of(run), HEADING)


def test_a_symlinked_reported_file_is_not_followed(tmp_path):
    run = make_run(tmp_path)
    outside = write_json(tmp_path / "outside.json", REPORTED)
    (run / "reported-results.json").symlink_to(outside)
    text = results_of(run)
    assert "81.8" not in text


def test_copied_text_goes_through_the_redactor(tmp_path):
    run = make_run(tmp_path)
    rows = {"source": "the card at /home/someone/.cache/x/README.md",
            "rows": [{"name": "MLE-Lite", "value": "81.8", "note": "see /home/someone/y; token hf_abcdefghijklmnopqrstuvwxyz0123456789"}]}
    write_json(run / "reported-results.json", rows)
    text = results_of(run)
    assert "/home/someone" not in text and "hf_abcdefghijklmnopqrstuvwxyz" not in text


def test_a_cell_cannot_break_the_table_or_add_a_heading(tmp_path):
    run = make_run(tmp_path)
    rows = {"source": "s\n# Injected", "rows": [{"name": "a | b", "value": "1", "note": "n\n## Injected too"}]}
    write_json(run / "reported-results.json", rows)
    text = results_of(run)
    assert "\n# Injected" not in text and "\n## Injected too" not in text
    body = section(text, HEADING)
    assert body.count("\n|") >= 3
    assert all(line.count("|") == 5 for line in body.splitlines() if line.startswith("| a "))


def test_the_gate_passes_the_bundle_that_has_the_section(tmp_path):
    run = make_run(tmp_path)
    write_json(run / "reported-results.json", REPORTED)
    results_of(run)
    assert gate_bundle(run / "stages" / "8", run).ok


def test_the_golden_results_text_is_unchanged_for_a_run_with_no_reported_scores(tmp_path):
    run = make_run(tmp_path)
    assert HEADING not in results_of(run)
    assert GOLDEN.is_file()


# ---- a reported row can never be a verdict or a yardstick ------------------------------------------------------------

def test_the_operator_views_results_and_verdict_ignore_reported_scores(tmp_path):
    run = make_run(tmp_path)
    entries = []
    swap = {"result_draft": {"top1_agreement": 0.97, "n_tokens": 32, "server_ready_s": 100.0, "coherent": True}}
    write_json(run / "stages" / "2" / "evidence" / "swap-check.json", swap)
    before = run_view.results(run, entries)
    items_before = verdict.build(before, [])
    lure = {"source": "s", "rows": [{"name": "top1_agreement", "value": "0.01"}, {"name": "server_ready_s", "value": "1"},
                                    {"name": "pass", "value": "0"}]}
    write_json(run / "reported-results.json", lure)
    write_json(run / "stages" / "0" / "evidence" / "reported-results.draft.json", {**lure, "model": "m"})
    after = run_view.results(run, entries)
    assert after == before
    assert verdict.build(after, []) == items_before
    assert "0.01" not in json.dumps(after) + json.dumps(items_before)


def test_the_other_runs_yardstick_does_not_read_reported_files(tmp_path):
    src = Path(verdict.__file__).read_text() + Path(run_view.__file__).read_text()
    assert "reported-results" not in src


def test_a_reported_file_over_the_size_cap_is_not_read(tmp_path):
    run = make_run(tmp_path)
    big = {**REPORTED, "rows": [{"name": "MLE-Lite", "value": "81.8", "note": "n" * 100}], "pad": "x" * (300 * 1024)}
    write_json(run / "reported-results.json", big)
    body = section(results_of(run), HEADING)
    assert "not valid" in body and "KiB" in body and "81.8" not in body


def test_a_draft_that_is_a_link_or_over_the_cap_is_not_counted(tmp_path):
    run = make_run(tmp_path)
    draft = run / "stages" / "0" / "evidence" / "reported-results.draft.json"
    outside = write_json(tmp_path / "outside.json", DRAFT)
    draft.symlink_to(outside)
    assert HEADING not in results_of(run)
    draft.unlink()
    write_json(draft, {**DRAFT, "pad": "x" * (300 * 1024)})
    assert HEADING not in results_of(run)


def test_a_draft_with_one_row_says_row_not_rows(tmp_path):
    run = make_run(tmp_path)
    write_json(run / "stages" / "0" / "evidence" / "reported-results.draft.json", {**DRAFT, "rows": DRAFT["rows"][:1]})
    assert "drafted 1 row from" in " ".join(section(results_of(run), HEADING).split())
