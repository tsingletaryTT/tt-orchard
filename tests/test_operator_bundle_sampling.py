# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 8 bundle and the sampling probe (issue 107): a RESULTS.md section that says, for stage 2 and for each stage 4 configuration, what the
server did with a request that has temperature above 0, from the recorded numbers (orchard/sampling_probe.py). A run with no probe record gets the
same RESULTS.md as before, which the golden text test pins."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_template import build, bundle_of, make_run, section, write_json  # noqa: E402
from test_sampling_probe import GREEDY, LINE, OTHER, STORY, entry, record, refused  # noqa: E402

HEAD = "## Sampling"


def results(run):
    assert build(run).returncode == 0
    return (bundle_of(run) / "RESULTS.md").read_text()


def put(run, stage_path, rec):
    write_json(run / stage_path / "swap-check.json", {"sampling_probe": rec})


def test_a_run_without_a_probe_record_has_no_sampling_section(tmp_path):
    run = make_run(tmp_path)
    assert HEAD not in results(run)


def test_the_stage_2_record_gives_a_row_with_speeds_the_verdict_and_the_evidence_file(tmp_path):
    run = make_run(tmp_path)
    put(run, "stages/2/evidence", record(greedy=entry(seconds=2.0, log=LINE), a=entry(text=STORY, seconds=2.0), b=entry(text=OTHER, seconds=3.0)))
    text = results(run)
    block = section(text, HEAD)
    assert "stage 2" in block and "sampled-full-speed" in block
    assert "48.0" in block and "40.0" in block                           # greedy tokens/s and the mean of the two sampled requests
    assert "0.536" in block                                              # the drafter's accepted share on the greedy request, 15 of 28
    assert "stages/2/evidence/swap-check.json" in block and "measured" in block
    assert "Sampled requests decoded at 40.0 tokens/s against 48.0 tokens/s for greedy requests" in block


def test_ignored_sampling_is_stated_in_a_sentence(tmp_path):
    run = make_run(tmp_path)
    put(run, "stages/2/evidence", record(a=entry(text=STORY), b=entry(text=STORY)))
    block = section(results(run), HEAD)
    assert "sampled-ignored" in block and "Sampling was not applied, so decoding was greedy" in block


def test_ignored_sampling_found_by_the_greedy_text_says_so_in_the_bundle(tmp_path):
    run = make_run(tmp_path)
    put(run, "stages/2/evidence", record(a=entry(text=GREEDY), b=entry(text=OTHER)))
    block = section(results(run), HEAD)
    assert "sampled-ignored" in block and "returned the greedy text" in block and "differ only in seed" not in block


def test_a_refused_sampled_request_is_stated_with_its_error(tmp_path):
    run = make_run(tmp_path)
    put(run, "stages/2/evidence", record(a=refused("HTTP 400: sampling is not supported"), b=refused("HTTP 400: sampling is not supported")))
    block = section(results(run), HEAD)
    assert "sampled-refused" in block and "sampling is not supported" in block


def test_each_stage_4_configuration_has_its_own_row(tmp_path):
    run = make_run(tmp_path)
    put(run, "stages/2/evidence", record())
    put(run, "stages/4/configs/2/evidence", record())
    put(run, "stages/4/configs/4/evidence", record(a=entry(text=STORY), b=entry(text=STORY)))
    block = section(results(run), HEAD)
    rows = [ln for ln in block.splitlines() if ln.startswith("| ") and "swap-check.json" in ln]
    assert len(rows) == 3
    assert any(ln.startswith("| stage 4, 4 chips ") and "sampled-ignored" in ln for ln in rows)
    assert any(ln.startswith("| stage 4, 2 chips ") and "sampled-full-speed" in ln for ln in rows)


def test_a_malformed_record_is_shown_as_malformed_and_its_numbers_are_not_copied(tmp_path):
    run = make_run(tmp_path)
    rec = record()
    rec["requests"]["greedy"]["seconds"] = "fast"
    put(run, "stages/2/evidence", rec)
    block = section(results(run), HEAD)
    assert "malformed" in block and "fast" not in block.replace("Sampling was", "")


def test_a_record_that_is_not_an_object_or_a_file_that_is_not_json_is_left_out(tmp_path):
    run = make_run(tmp_path)
    write_json(run / "stages/2/evidence/swap-check.json", {"sampling_probe": "yes"})
    (run / "stages/4/configs/2/evidence").mkdir(parents=True, exist_ok=True)
    (run / "stages/4/configs/2/evidence/swap-check.json").write_text("{not json")
    text = results(run)
    assert "sampling_probe" not in text or "malformed" in section(text, HEAD)


def test_a_path_or_home_directory_in_an_error_text_is_redacted(tmp_path):
    run = make_run(tmp_path)
    put(run, "stages/2/evidence", record(a=refused("HTTP 400: /home/someone/secret/path failed"), b=refused()))
    text = results(run)
    assert "/home/someone" not in text


def test_a_probe_with_a_missing_timing_is_shown_as_inconclusive(tmp_path):
    run = make_run(tmp_path)
    rec = record()
    rec["requests"]["sampled_b"]["seconds"] = None
    put(run, "stages/2/evidence", rec)
    block = section(results(run), HEAD)
    assert "inconclusive" in block and "sampled-full-speed" not in block


def test_a_record_that_did_not_run_has_no_row(tmp_path):
    run = make_run(tmp_path)
    put(run, "stages/2/evidence", {"ran": False})
    assert HEAD not in results(run)


def test_a_malformed_record_row_is_labelled_todo(tmp_path):
    run = make_run(tmp_path)
    rec = record()
    rec["requests"]["greedy"]["seconds"] = "fast"
    put(run, "stages/2/evidence", rec)
    row = [ln for ln in section(results(run), HEAD).splitlines() if ln.startswith("| stage 2")][0]
    assert "| TODO | `stages/2" in row and "measured" not in row


def test_a_drafter_share_of_zero_is_shown_and_not_called_todo(tmp_path):
    run = make_run(tmp_path)
    zero = LINE.replace("accept 3.75/7", "accept 0.00/7").replace("histogram [0, 2, 0, 0, 0, 0, 1, 1]", "histogram [4, 0, 0, 0, 0, 0, 0, 0]")
    put(run, "stages/2/evidence", record(greedy=entry(log=zero), a=entry(text=STORY, log=zero), b=entry(text=OTHER)))
    row = [ln for ln in section(results(run), HEAD).splitlines() if ln.startswith("| stage 2")][0]
    assert "| 0.0 / 0.0 |" in row


def test_stage_4_rows_come_with_the_larger_chip_count_first(tmp_path):
    run = make_run(tmp_path)
    for n in (2, 4, 1):
        put(run, f"stages/4/configs/{n}/evidence", record())
    block = section(results(run), HEAD)
    order = [block.index(f"| stage 4, {n} chips") for n in (4, 2, 1)]
    assert order == sorted(order)
