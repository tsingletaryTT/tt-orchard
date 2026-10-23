# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Re-running cut-off items (orchard/bench_rerun.py): find the items whose reply came back empty, and merge a re-run's results into the first run's.

Sample files are small fabrications in the layout lm-eval writes (`<phase>/results/<model>/samples_<task>_<time>.jsonl`, one JSON object per item).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import bench_rerun as br


def write_samples(root: Path, phase: str, rows: list[dict], stamp="2026-10-19T21-17-43.912713", task=None):
    d = root / phase / "results" / "served"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"samples_{task or phase + '_chat'}_{stamp}.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


def row(doc_id, text, **metrics):
    return {"doc_id": doc_id, "resps": [[text]], "filtered_resps": [text], "doc": {}, "target": "t", **metrics}


# ---- finding the items ----------------------------------------------------------------------------------------------------------------

def test_items_with_an_empty_reply_are_listed_by_document_id(tmp_path):
    write_samples(tmp_path, "mbpp", [row(0, "ok", pass_at_1=1), row(1, "", pass_at_1=0), row(2, "  \n", pass_at_1=0), row(3, "x", pass_at_1=0)])
    assert br.cutoff_items(tmp_path, ["mbpp"]) == {"mbpp": [1, 2]}


def test_a_phase_with_no_empty_reply_is_left_out(tmp_path):
    write_samples(tmp_path, "mbpp", [row(0, "ok", pass_at_1=1)])
    write_samples(tmp_path, "ifeval", [row(5, "", prompt_level_strict_acc=False)])
    assert br.cutoff_items(tmp_path, ["mbpp", "ifeval"]) == {"ifeval": [5]}


def test_the_ids_come_back_sorted_and_without_repeats(tmp_path):
    write_samples(tmp_path, "mbpp", [row(9, ""), row(2, ""), row(2, "")])
    assert br.cutoff_items(tmp_path, ["mbpp"]) == {"mbpp": [2, 9]}


def test_the_newest_samples_file_is_read(tmp_path):
    write_samples(tmp_path, "mbpp", [row(1, "")], stamp="2026-10-19T10-00-00.000000")
    write_samples(tmp_path, "mbpp", [row(2, "")], stamp="2026-10-19T11-00-00.000000")
    assert br.cutoff_items(tmp_path, ["mbpp"]) == {"mbpp": [2]}


def test_a_phase_with_no_samples_file_is_an_error_that_names_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="mbpp"):
        br.cutoff_items(tmp_path, ["mbpp"])


def test_a_reply_that_is_not_a_string_is_not_counted_as_empty(tmp_path):
    write_samples(tmp_path, "mbpp", [{"doc_id": 1, "resps": [[None]]}, {"doc_id": 2, "resps": []}, {"doc_id": 3, "resps": [["x"]]}])
    assert br.cutoff_items(tmp_path, ["mbpp"]) == {}


# ---- merging ------------------------------------------------------------------------------------------------------------------------

def test_a_rerun_replaces_the_first_runs_result_for_the_same_item(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "a", pass_at_1=1), row(1, "", pass_at_1=0), row(2, "", pass_at_1=0), row(3, "d", pass_at_1=0)])
    write_samples(again, "mbpp", [row(1, "fixed", pass_at_1=1), row(2, "still wrong", pass_at_1=0)])
    got = br.merge(first, again, ["mbpp"])["mbpp"]
    assert got["n"] == 4 and got["rerun_n"] == 2
    assert got["metrics"]["pass_at_1"] == {"before": 0.25, "after": 0.5}
    assert got["flips"] == {"pass_at_1": {"gained": [1], "lost": []}}


def test_an_item_the_rerun_did_not_cover_keeps_the_first_runs_result(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "a", pass_at_1=1), row(1, "", pass_at_1=0)])
    write_samples(again, "mbpp", [row(1, "x", pass_at_1=0)])
    assert br.merge(first, again, ["mbpp"])["mbpp"]["metrics"]["pass_at_1"] == {"before": 0.5, "after": 0.5}


def test_a_rerun_item_that_got_worse_is_reported_as_lost(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "", pass_at_1=1)])
    write_samples(again, "mbpp", [row(0, "x", pass_at_1=0)])
    got = br.merge(first, again, ["mbpp"])["mbpp"]
    assert got["flips"]["pass_at_1"] == {"gained": [], "lost": [0]} and got["metrics"]["pass_at_1"]["after"] == 0.0


def test_a_rerun_item_missing_from_the_first_run_is_an_error(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "a", pass_at_1=1)])
    write_samples(again, "mbpp", [row(7, "x", pass_at_1=1)])
    with pytest.raises(ValueError, match="7"):
        br.merge(first, again, ["mbpp"])


def test_every_numeric_per_item_metric_is_merged_and_text_fields_are_not(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    a = row(0, "", prompt_level_strict_acc=False, prompt_level_loose_acc=False, inst_level_strict_acc=[False, False])
    write_samples(first, "ifeval", [a, row(1, "y", prompt_level_strict_acc=True, prompt_level_loose_acc=True, inst_level_strict_acc=[True])])
    write_samples(again, "ifeval", [row(0, "x", prompt_level_strict_acc=True, prompt_level_loose_acc=False, inst_level_strict_acc=[True, True])])
    got = br.merge(first, again, ["ifeval"])["ifeval"]["metrics"]
    assert set(got) == {"prompt_level_strict_acc", "prompt_level_loose_acc"}
    assert got["prompt_level_strict_acc"] == {"before": 0.5, "after": 1.0} and got["prompt_level_loose_acc"] == {"before": 0.5, "after": 0.5}


def test_the_merge_does_not_touch_either_input(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    f = write_samples(first, "mbpp", [row(0, "", pass_at_1=0)])
    g = write_samples(again, "mbpp", [row(0, "x", pass_at_1=1)])
    before = (f.read_bytes(), g.read_bytes())
    br.merge(first, again, ["mbpp"])
    assert (f.read_bytes(), g.read_bytes()) == before


# ---- the command line -----------------------------------------------------------------------------------------------------------------

def test_the_items_command_writes_a_file_the_session_accepts(tmp_path, capsys):
    from orchard import bench_session as bs
    write_samples(tmp_path / "run", "mbpp", [row(4, ""), row(1, ""), row(2, "ok")])
    out = tmp_path / "items.json"
    assert br.main(["items", "--run", str(tmp_path / "run"), "--phases", "mbpp", "--out", str(out)]) == 0
    assert bs.load_items(out) == {"mbpp": [1, 4]}
    assert "mbpp: 2 items" in capsys.readouterr().out


def test_the_items_command_with_nothing_to_rerun_writes_no_file_and_says_so(tmp_path, capsys):
    write_samples(tmp_path / "run", "mbpp", [row(2, "ok")])
    out = tmp_path / "items.json"
    assert br.main(["items", "--run", str(tmp_path / "run"), "--phases", "mbpp", "--out", str(out)]) == 3
    assert not out.exists() and "nothing to re-run" in capsys.readouterr().out


def test_the_merge_command_prints_the_before_and_after(tmp_path, capsys):
    write_samples(tmp_path / "first", "mbpp", [row(0, "", pass_at_1=0)])
    write_samples(tmp_path / "again", "mbpp", [row(0, "x", pass_at_1=1)])
    out = tmp_path / "merged.json"
    assert br.main(["merge", "--first", str(tmp_path / "first"), "--again", str(tmp_path / "again"), "--phases", "mbpp", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["mbpp"]["metrics"]["pass_at_1"] == {"before": 0.0, "after": 1.0}
    assert "mbpp" in capsys.readouterr().out


# ---- finding replies cut off by a stop string in a finished run ------------------------------------------------------------------------------
# lm-eval's gsm8k task stops at "Question:", which vLLM also matches inside a thinking model's reasoning. A run made before the proxy recorded
# `content_chars` has only token counts and reasoning length, so the check reads each phase's requests.jsonl with the short-reply rule.

def write_requests(root: Path, phase: str, recs: list[dict]):
    d = root / phase
    d.mkdir(parents=True, exist_ok=True)
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def old_rec(tokens, reasoning=70, finish="stop"):
    return {"status": 200, "error": False, "finish_reason": finish, "completion_tokens": tokens, "reasoning_chars": reasoning}


def test_the_suspects_are_counted_for_each_phase(tmp_path):
    write_requests(tmp_path, "gsm8k", [old_rec(20), old_rec(24), old_rec(300), old_rec(8192, finish="length")])
    write_requests(tmp_path, "ifeval", [old_rec(500), old_rec(600)])
    got = br.stop_suspects(tmp_path, ["gsm8k", "ifeval"])
    assert got == {"gsm8k": {"requests": 4, "suspects": 2}, "ifeval": {"requests": 2, "suspects": 0}}


def test_a_phase_with_no_request_log_is_an_error_that_names_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="gsm8k"):
        br.stop_suspects(tmp_path, ["gsm8k"])


def test_the_suspects_command_prints_each_phase_and_exits_4_when_any_are_found(tmp_path, capsys):
    write_requests(tmp_path, "gsm8k", [old_rec(20), old_rec(300)])
    assert br.main(["suspects", "--run", str(tmp_path), "--phases", "gsm8k"]) == 4
    out = capsys.readouterr().out
    assert "gsm8k: 1 of 2 requests" in out and "stop string" in out


def test_one_phase_with_suspects_is_enough_for_exit_4_when_another_has_none(tmp_path, capsys):
    write_requests(tmp_path, "gsm8k", [old_rec(20)])
    write_requests(tmp_path, "ifeval", [old_rec(500)])
    assert br.main(["suspects", "--run", str(tmp_path), "--phases", "ifeval,gsm8k"]) == 4
    out = capsys.readouterr().out
    assert "gsm8k: 1 of 1 requests look cut off" in out and "ifeval: 0 of 1 requests" in out and "look cut off" not in out.split("ifeval")[1].split("\n")[0]


def test_the_suspects_command_exits_0_and_says_none_when_there_are_none(tmp_path, capsys):
    write_requests(tmp_path, "gsm8k", [old_rec(300), old_rec(400)])
    assert br.main(["suspects", "--run", str(tmp_path), "--phases", "gsm8k"]) == 0
    out = capsys.readouterr().out
    assert "gsm8k: 0 of 2 requests" in out and "stop string" not in out


def test_the_suspects_command_with_no_log_exits_2_and_names_the_file(tmp_path, capsys):
    assert br.main(["suspects", "--run", str(tmp_path), "--phases", "gsm8k"]) == 2
    assert "requests.jsonl" in capsys.readouterr().err


# ---- refusing a re-run that does not fit the first run ---------------------------------------------------------------------------------

def test_a_task_with_two_rows_per_document_is_refused(tmp_path):
    """lm-eval writes one row per document per filter (GSM8K has two); keying by document id would keep only one of them."""
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "gsm8k", [{**row(0, "a", exact_match=1), "filter": "strict-match"}, {**row(0, "a", exact_match=0), "filter": "flexible-extract"}])
    write_samples(again, "gsm8k", [{**row(0, "a", exact_match=1), "filter": "strict-match"}])
    with pytest.raises(ValueError, match="gsm8k.*more than one row.*0"):
        br.merge(first, again, ["gsm8k"])


def test_a_repeated_document_in_the_rerun_is_refused(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "", pass_at_1=0)])
    write_samples(again, "mbpp", [row(0, "x", pass_at_1=1), row(0, "y", pass_at_1=0)])
    with pytest.raises(ValueError, match="more than one row"):
        br.merge(first, again, ["mbpp"])


def test_a_rerun_made_with_a_different_prompt_is_refused(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [{**row(0, "", pass_at_1=0), "prompt_hash": "aaa", "doc_hash": "d"}])
    write_samples(again, "mbpp", [{**row(0, "x", pass_at_1=1), "prompt_hash": "bbb", "doc_hash": "d"}])
    with pytest.raises(ValueError, match="mbpp: item 0.*prompt_hash"):
        br.merge(first, again, ["mbpp"])


def test_a_rerun_with_a_different_document_is_refused(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [{**row(0, "", pass_at_1=0), "prompt_hash": "p", "doc_hash": "aaa"}])
    write_samples(again, "mbpp", [{**row(0, "x", pass_at_1=1), "prompt_hash": "p", "doc_hash": "bbb"}])
    with pytest.raises(ValueError, match="doc_hash"):
        br.merge(first, again, ["mbpp"])


def test_matching_hashes_merge_and_rows_with_no_hashes_are_not_compared(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [{**row(0, "", pass_at_1=0), "prompt_hash": "p", "doc_hash": "d"}, row(1, "", pass_at_1=0)])
    write_samples(again, "mbpp", [{**row(0, "x", pass_at_1=1), "prompt_hash": "p", "doc_hash": "d"}, {**row(1, "x", pass_at_1=1), "prompt_hash": "q"}])
    assert br.merge(first, again, ["mbpp"])["mbpp"]["metrics"]["pass_at_1"]["after"] == 1.0


def test_a_rerun_that_is_not_exactly_the_requested_items_is_refused(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "", pass_at_1=0), row(1, "", pass_at_1=0), row(2, "a", pass_at_1=1)])
    write_samples(again, "mbpp", [row(0, "x", pass_at_1=1)])
    with pytest.raises(ValueError, match="mbpp.*missing.*1"):
        br.merge(first, again, ["mbpp"], expected={"mbpp": [0, 1]})
    write_samples(again, "mbpp", [row(0, "x", pass_at_1=1), row(1, "x", pass_at_1=1), row(2, "x", pass_at_1=1)], stamp="2026-10-20T00-00-00.000000")
    with pytest.raises(ValueError, match="mbpp.*not requested.*2"):
        br.merge(first, again, ["mbpp"], expected={"mbpp": [0, 1]})


def test_a_rerun_that_is_exactly_the_requested_items_merges(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "", pass_at_1=0), row(1, "a", pass_at_1=1)])
    write_samples(again, "mbpp", [row(0, "x", pass_at_1=1)])
    assert br.merge(first, again, ["mbpp"], expected={"mbpp": [0]})["mbpp"]["rerun_n"] == 1


def test_a_phase_missing_from_the_requested_items_is_refused(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "", pass_at_1=0)])
    write_samples(again, "mbpp", [row(0, "x", pass_at_1=1)])
    with pytest.raises(ValueError, match="mbpp.*not in the requested"):
        br.merge(first, again, ["mbpp"], expected={"ifeval": [0]})


def test_a_sample_missing_a_metric_is_an_error_not_a_zero(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "", pass_at_1=0), row(1, "a", pass_at_1=1)])
    write_samples(again, "mbpp", [row(0, "x")])
    with pytest.raises(ValueError, match="mbpp: item 0 of the re-run has no pass_at_1"):
        br.merge(first, again, ["mbpp"])


def test_a_first_run_sample_missing_a_metric_is_an_error(tmp_path):
    first, again = tmp_path / "first", tmp_path / "again"
    write_samples(first, "mbpp", [row(0, "", pass_at_1=0), row(1, "a")])
    write_samples(again, "mbpp", [row(0, "x", pass_at_1=1)])
    with pytest.raises(ValueError, match="item 1 of the first run has no pass_at_1"):
        br.merge(first, again, ["mbpp"])


def test_the_merge_command_checks_the_requested_items_when_given_them(tmp_path):
    write_samples(tmp_path / "first", "mbpp", [row(0, "", pass_at_1=0), row(1, "", pass_at_1=0)])
    write_samples(tmp_path / "again", "mbpp", [row(0, "x", pass_at_1=1)])
    items = tmp_path / "items.json"
    items.write_text('{"mbpp": [0, 1]}')
    out = tmp_path / "merged.json"
    with pytest.raises(ValueError, match="missing"):
        br.main(["merge", "--first", str(tmp_path / "first"), "--again", str(tmp_path / "again"), "--phases", "mbpp", "--items", str(items), "--out", str(out)])
    assert not out.exists()


# ---- one record per line, whatever characters the text holds ------------------------------------------------------------------------------

def write_samples_raw(root: Path, phase: str, rows: list[dict]):
    """lm-eval writes samples with ensure_ascii off, so U+2028 and the like stay raw inside a JSON string."""
    d = root / phase / "results" / "served"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"samples_{phase}_chat_2026-10-22T10-18-57.902126.jsonl"
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return f


@pytest.mark.parametrize("sep", [" ", " ", "\x85", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e"])
def test_a_separator_character_inside_a_reply_does_not_split_a_record(tmp_path, sep):
    write_samples_raw(tmp_path, "gsm8k", [row(0, f"first{sep}second", exact_match=1), row(1, "", exact_match=0), row(2, f"x{sep}", exact_match=0)])
    assert br.read_samples(tmp_path, "gsm8k")[0]["resps"][0][0] == f"first{sep}second"
    assert [r["doc_id"] for r in br.read_samples(tmp_path, "gsm8k")] == [0, 1, 2]
    assert br.cutoff_items(tmp_path, ["gsm8k"]) == {"gsm8k": [1]}
