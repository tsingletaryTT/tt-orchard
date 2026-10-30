# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/served_chat.py: is the served chat template, the special tokens and the stop ids the new model's, and did the recorded chat replies
start and stop like the model's chat format (issue 17). A pure check of the served directory, the new model's snapshot and the swap check's
recorded prompt-suite replies. Nothing is sent to a server."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from orchard import served_chat as sc

TEMPLATE_A = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
TEMPLATE_B = "{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}"
TOK = {"eos_token": "<|im_end|>", "bos_token": None, "pad_token": "<|endoftext|>", "unk_token": None}
TOKENIZER_JSON = {"added_tokens": [{"id": 7, "content": "<|im_end|>", "special": True}, {"id": 8, "content": "<|endoftext|>", "special": True},
                                    {"id": 9, "content": "<|im_start|>", "special": True}], "model": {"vocab": {"a": 0}}}


def make(d: Path, *, template=TEMPLATE_A, inline=None, tokens=TOK, gen=None, cfg=None, tokenizer=TOKENIZER_JSON, jinja=True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    if template is not None and jinja:
        (d / "chat_template.jinja").write_text(template)
    tc = dict(tokens)
    if inline is not None:
        tc["chat_template"] = inline
    (d / "tokenizer_config.json").write_text(json.dumps(tc))
    if tokenizer is not None:
        (d / "tokenizer.json").write_text(json.dumps(tokenizer))
    if gen is not None:
        (d / "generation_config.json").write_text(json.dumps(gen))
    if cfg is not None:
        (d / "config.json").write_text(json.dumps(cfg))
    return d


def names(rec):
    return {p["name"] for p in rec["problems"]}


GOOD_GEN = {"eos_token_id": [7, 8]}


def pair(tmp_path, **served_kw):
    new = make(tmp_path / "new", gen=GOOD_GEN)
    served = make(tmp_path / "served", **{"gen": GOOD_GEN, **served_kw})
    return served, new


# ---- the files -------------------------------------------------------------------------------------------------------------

def test_the_same_template_tokens_and_stop_ids_pass_with_the_hashes_recorded(tmp_path):
    served, new = pair(tmp_path)
    rec = sc.check_files(served, new)
    assert rec["problems"] == [] and rec["template"]["served_sha256"] == rec["template"]["new_sha256"] and len(rec["template"]["new_sha256"]) == 64
    assert rec["template"]["served_file"] == "chat_template.jinja"
    assert rec["special_tokens"]["served"] == rec["special_tokens"]["new"]
    assert rec["stop_ids"]["served"] == [7, 8] and rec["stop_ids"]["new_model_ends"] == [7, 8]


def test_a_served_template_that_is_not_the_new_models_is_a_problem(tmp_path):
    served, new = pair(tmp_path, template=TEMPLATE_B)
    assert "template_differs_from_new" in names(sc.check_files(served, new))


def test_a_new_model_template_missing_from_the_served_directory_is_a_problem(tmp_path):
    served, new = pair(tmp_path, template=None)
    assert "template_missing_in_served" in names(sc.check_files(served, new))


def test_a_served_template_the_new_model_does_not_ship_is_a_problem(tmp_path):
    new = make(tmp_path / "new", template=None, gen=GOOD_GEN)
    served = make(tmp_path / "served", gen=GOOD_GEN)
    assert "template_not_in_new" in names(sc.check_files(served, new))


def test_neither_side_shipping_a_template_is_a_note_and_not_a_problem(tmp_path):
    new = make(tmp_path / "new", template=None, gen=GOOD_GEN)
    served = make(tmp_path / "served", template=None, gen=GOOD_GEN)
    rec = sc.check_files(served, new)
    assert rec["problems"] == [] and any("neither" in n and "chat template" in n for n in rec["notes"])


def test_an_inline_template_is_read_the_way_the_loader_does_and_the_file_wins(tmp_path):
    new = make(tmp_path / "new", template=None, inline=TEMPLATE_A, gen=GOOD_GEN)
    served = make(tmp_path / "served", template=TEMPLATE_A, inline=TEMPLATE_B, gen=GOOD_GEN)       # the file wins over the inline copy
    rec = sc.check_files(served, new)
    assert rec["template"]["new_file"] == "tokenizer_config.json" and rec["template"]["served_file"] == "chat_template.jinja"
    assert rec["problems"] == []                                                                     # same text, read from different files


def test_a_template_read_through_a_link_is_the_linked_files_text(tmp_path):
    new = make(tmp_path / "new", gen=GOOD_GEN)
    served = make(tmp_path / "served", template=None, gen=GOOD_GEN)
    os.symlink(os.path.realpath(new / "chat_template.jinja"), served / "chat_template.jinja")        # how prepare_swap.py links it
    assert sc.check_files(served, new)["problems"] == []


@pytest.mark.parametrize("field", ["eos_token", "pad_token"])
def test_a_special_token_that_differs_is_a_problem_naming_both_values(tmp_path, field):
    served, new = pair(tmp_path, tokens={**TOK, field: "<other>"})
    rec = sc.check_files(served, new)
    assert "special_token_differs" in names(rec)
    assert any(field in p["detail"] and "<other>" in p["detail"] for p in rec["problems"])


def test_a_token_written_as_an_object_with_content_equals_the_plain_string(tmp_path):
    served, new = pair(tmp_path, tokens={**TOK, "eos_token": {"content": "<|im_end|>", "special": True}})
    assert sc.check_files(served, new)["problems"] == []


def test_a_stop_id_of_the_new_models_that_the_served_directory_lacks_is_a_problem(tmp_path):
    served, new = pair(tmp_path, gen={"eos_token_id": [8]})
    rec = sc.check_files(served, new)
    assert "stop_id_missing" in names(rec) and "7" in next(p["detail"] for p in rec["problems"] if p["name"] == "stop_id_missing")


def test_the_tokenizers_own_eos_counts_as_an_end_token_of_the_new_model(tmp_path):
    new = make(tmp_path / "new")                                   # no generation_config at all: the tokenizer's eos is the only end token
    served = make(tmp_path / "served", gen={"eos_token_id": 8})
    rec = sc.check_files(served, new)
    assert rec["stop_ids"]["new_model_ends"] == [7] and "stop_id_missing" in names(rec)


def test_stop_ids_come_from_the_generation_config_then_the_top_level_of_config_json(tmp_path):
    new = make(tmp_path / "new", gen={"eos_token_id": 7})
    served = make(tmp_path / "served", gen={"eos_token_id": 8}, cfg={"eos_token_id": [7, 8]})
    assert sc.check_files(served, new)["stop_ids"]["served"] == [8, 7]


def test_a_stop_id_nested_under_text_config_does_not_count_because_the_bundle_reads_only_the_top_level(tmp_path):
    new = make(tmp_path / "new", gen={"eos_token_id": 7})
    served = make(tmp_path / "served", cfg={"text_config": {"eos_token_id": 7}})
    assert "stop_id_missing" in names(sc.check_files(served, new))


@pytest.mark.parametrize("bad", ["{nope", "[]", "null", '{"a": NaN}'])
def test_unreadable_or_hostile_json_files_make_the_check_incomplete_and_never_raise(tmp_path, bad):
    served, new = pair(tmp_path)
    (served / "tokenizer_config.json").write_text(bad)
    (served / "generation_config.json").write_text(bad)
    rec = sc.check_files(served, new)
    assert len(rec["incomplete"]) >= 2 and isinstance(rec["problems"], list)
    assert sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))["ok"] is False


def test_a_missing_served_directory_is_a_problem_not_an_exception(tmp_path):
    new = make(tmp_path / "new", gen=GOOD_GEN)
    rec = sc.check_files(tmp_path / "nowhere", new)
    assert "template_missing_in_served" in names(rec)


# ---- the replies -----------------------------------------------------------------------------------------------------------

def reply(content="Paris, with the Eiffel Tower.", finish="stop", error=None, mode="chat"):
    return {"id": "x", "mode": mode, "content": content, "reasoning": "", "tool_calls": [], "finish_reason": finish, "error": error}


def suite(entries, mode="chat", specials=("<|im_end|>", "<|im_start|>", "<|endoftext|>"), template=TEMPLATE_A, files=None):
    """A prompt_suite record the way serve_and_compare.py writes it: the digest of the template the test saw is in it, and `files` (the digests
    of the served tokenizer and stop-id files, or None to leave them out)."""
    rec = {"ran": True, "mode": mode, "max_tokens": 256, "special_tokens": list(specials), "entries": entries,
           "chat_template": {"present": mode == "chat", "sha256": sc._sha(template)}}
    if files is not None:
        rec["served_files"] = files
    return rec


def test_replies_that_stop_and_start_cleanly_pass(tmp_path):
    rec = sc.check_replies(suite([reply(), reply("def f(): pass"), reply(finish="length")]))
    assert rec["problems"] == [] and rec["mode"] == "chat" and rec["counts"] == {"checked": 3, "stopped": 2, "length": 1}


def test_when_no_reply_ever_stops_on_its_own_the_stop_token_is_suspected(tmp_path):
    rec = sc.check_replies(suite([reply(finish="length"), reply(finish="length"), reply(finish="length")]))
    assert "no_reply_stopped" in names(rec)


def test_one_reply_that_hit_the_limit_is_not_enough_to_say_so(tmp_path):
    assert "no_reply_stopped" not in names(sc.check_replies(suite([reply(finish="length")])))


@pytest.mark.parametrize("content", ["assistant\nParis.", "Assistant: Paris.", "<|im_start|>assistant\nParis.", "  <|im_end|> Paris.",
                                     "assistant: Paris"])
def test_a_reply_that_starts_with_a_role_header_or_a_special_token_is_a_problem(content):
    assert "role_header_start" in names(sc.check_replies(suite([reply(content), reply()])))


@pytest.mark.parametrize("content", ["Assistants help people.", "The assistant said hello.", "assist me", "Paris."])
def test_ordinary_text_that_merely_mentions_an_assistant_is_not_flagged(content):
    assert "role_header_start" not in names(sc.check_replies(suite([reply(content)])))


def test_a_stop_token_inside_a_reply_is_a_problem(tmp_path):
    assert "stop_token_in_reply" in names(sc.check_replies(suite([reply("Paris.<|im_end|>"), reply()]), end_tokens=["<|im_end|>"]))


def test_replies_with_errors_are_left_out_of_the_counts(tmp_path):
    rec = sc.check_replies(suite([reply(error="HTTP 500", finish=None, content=""), reply()]))
    assert rec["counts"]["checked"] == 1 and rec["problems"] == []


@pytest.mark.parametrize("record", [None, {"ran": False}, {"ran": True, "mode": "completion", "entries": [reply(mode="completion")]},
                                    {"ran": True, "mode": "chat", "entries": []}, {"ran": True, "mode": "chat", "entries": "x"}, 5, []])
def test_a_suite_that_sent_no_chat_request_has_nothing_to_check_and_says_so(record):
    rec = sc.check_replies(record)
    assert rec["problems"] == [] and rec["counts"]["checked"] == 0 and rec["checked"] is False and rec["why_not"]


def test_a_hostile_entry_shape_never_raises():
    rec = sc.check_replies(suite([None, 5, "x", {"content": 7, "finish_reason": []}, reply()]))
    assert rec["counts"]["checked"] == 1


# ---- the record ------------------------------------------------------------------------------------------------------------

def swap_check(tmp_path, entries, served=None, **kw):
    """A swap-check.json whose prompt suite recorded the digests of the files of `served` (default: tmp_path/served, the layout of pair())."""
    p = tmp_path / "swap-check.json"
    served = Path(served) if served else tmp_path / "served"
    p.write_text(json.dumps({"prompt_suite": suite(entries, files=sc.file_digests(served) if served.is_dir() else None, **kw)}))
    return p


def test_the_record_joins_the_file_check_and_the_reply_check_and_says_it_sent_no_request(tmp_path):
    served, new = pair(tmp_path)
    rec = sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))
    assert rec["ok"] is True and rec["problems"] == [] and rec["files"]["problems"] == [] and rec["replies"]["checked"] is True
    assert "sent no request" in rec["label"] and "card" in rec["not_checked"]


def test_any_problem_makes_the_record_not_ok_and_lists_it_once(tmp_path):
    served, new = pair(tmp_path, template=TEMPLATE_B)
    rec = sc.build_record(served, new, swap_check(tmp_path, [reply("assistant\nhi"), reply()]))
    assert rec["ok"] is False and {p["name"] for p in rec["problems"]} == {"template_differs_from_new", "role_header_start",
                                                                              "template_changed_since_replies"}


def test_a_record_with_no_replies_checked_is_not_ok_for_the_bundle(tmp_path):
    served, new = pair(tmp_path)
    rec = sc.build_record(served, new, swap_check(tmp_path, []))
    assert rec["problems"] == [] and rec["ok"] is False and rec["replies"]["checked"] is False


def test_write_record_is_atomic_and_round_trips(tmp_path):
    served, new = pair(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    rec = sc.write_record(out, served, new, swap_check(tmp_path, [reply()]))
    assert json.loads((out / sc.FILE).read_text()) == rec


def test_an_unreadable_swap_check_makes_a_record_that_says_the_replies_were_not_checked(tmp_path):
    served, new = pair(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert sc.build_record(served, new, bad)["replies"]["checked"] is False
    assert sc.build_record(served, new, tmp_path / "absent.json")["replies"]["checked"] is False


# ---- a real fine-tune whose template differs (the issue's open question) ----------------------------------------------------

def test_the_two_real_template_shapes_of_issue_17_are_told_apart(tmp_path):
    """Qwen3-Coder-Next carries two different templates, one inside tokenizer_config.json and one in chat_template.jinja; the loader uses
    the file (project log, 2026-10-22). A served directory built from the new model's files holds the file's template, so a check that
    read the inline one would call a correct directory wrong."""
    new = make(tmp_path / "new", template=TEMPLATE_A, inline=TEMPLATE_B, gen=GOOD_GEN)
    served = make(tmp_path / "served", template=TEMPLATE_A, inline=TEMPLATE_B, gen=GOOD_GEN)
    rec = sc.check_files(served, new)
    assert rec["problems"] == [] and rec["template"]["served_file"] == "chat_template.jinja"
    assert any("inline" in n and "differs" in n for n in rec["notes"])


def test_a_file_the_model_does_not_ship_adds_no_note_but_a_broken_one_does(tmp_path):
    new = make(tmp_path / "new")                      # no generation_config.json, no config.json
    served = make(tmp_path / "served")
    assert sc.check_files(served, new)["notes"] == []
    (served / "generation_config.json").write_text("{nope")
    rec = sc.check_files(served, new)
    assert any("generation_config.json could not be read" in n for n in rec["incomplete"]) and rec["notes"] == []


# ---- the supervisor writes it when stage 2 or stage 4 passes ---------------------------------------------------------------

def stage_files(run: Path, rel: str, tmp_path: Path, *, entries, served_kw=None, name="s"):
    """A stage 2 (rel "stages/2") or one stage 4 configuration (rel "stages/4/configs/2") directory with its served model-dir, swap config
    and swap check, over a new snapshot."""
    d = run / rel
    new = make(tmp_path / f"new-{name}", gen=GOOD_GEN)
    make(d / "model-dir", gen=GOOD_GEN, **(served_kw or {}))
    (d / "swap_config.json").write_text(json.dumps({"new_snapshot": str(new)}))
    (d / "evidence").mkdir(exist_ok=True)
    (d / "evidence" / "swap-check.json").write_text(json.dumps({"prompt_suite": suite(entries, files=sc.file_digests(d / "model-dir"))}))
    return d


def end_stage(tmp_path, n, prepare):
    from orchard.stages import GateResult, spec_for
    from test_supervisor_stock import Rig, _supervisor
    r = Rig(tmp_path)
    try:
        r.run()
        prepare(r.run_dir)
        sup, led = _supervisor(r)
        try:
            spec = spec_for(n, "weights-only")
            stage = r.run_dir / "stages" / str(n)
            stage.mkdir(parents=True, exist_ok=True)
            if spec.gate_file and not (stage / spec.gate_file).exists():
                (stage / spec.gate_file).write_text("{}")
            sup._end_stage(spec, stage, "pass", [], GateResult(True, (), ()), False)
            return r.run_dir, [e for e in led.read() if e["event"] == "evidence" and e["data"].get("what") == "served chat check"]
        finally:
            led.__exit__(None, None, None)
    finally:
        r.close()


def test_stage_2_passing_writes_the_record_and_the_ledger_names_the_problems(tmp_path):
    def prepare(run):
        stage_files(run, "stages/2", tmp_path, entries=[reply("assistant\nhi"), reply()], served_kw={"template": TEMPLATE_B})
    run, ev = end_stage(tmp_path, 2, prepare)
    rec = json.loads((run / "stages/2" / sc.FILE).read_text())
    assert rec["ok"] is False and {p["name"] for p in rec["problems"]} == {"template_differs_from_new", "role_header_start",
                                                                              "template_changed_since_replies"}
    assert len(ev) == 1 and ev[0]["stage"] == 2 and ev[0]["data"]["ok"] is False and ev[0]["data"]["path"] == "stages/2/served-chat.json"
    assert sorted(ev[0]["data"]["problems"]) == ["role_header_start", "template_changed_since_replies", "template_differs_from_new"]


def test_stage_4_passing_writes_one_record_per_configuration(tmp_path):
    def prepare(run):
        for n in (2, 4):
            stage_files(run, f"stages/4/configs/{n}", tmp_path, entries=[reply(), reply()], name=str(n))
    run, ev = end_stage(tmp_path, 4, prepare)
    for n in (2, 4):
        assert json.loads((run / f"stages/4/configs/{n}" / sc.FILE).read_text())["ok"] is True
    assert sorted(e["data"]["path"] for e in ev) == ["stages/4/configs/2/served-chat.json", "stages/4/configs/4/served-chat.json"]


def test_a_stage_without_a_swap_config_writes_nothing(tmp_path):
    run, ev = end_stage(tmp_path, 2, lambda run: None)
    assert ev == [] and not (run / "stages/2" / sc.FILE).exists()


def test_a_swap_config_with_a_hostile_new_snapshot_writes_nothing_and_never_stops_the_run(tmp_path):
    def prepare(run):
        d = stage_files(run, "stages/2", tmp_path, entries=[reply()])
        (d / "swap_config.json").write_text(json.dumps({"new_snapshot": 5}))
    run, ev = end_stage(tmp_path, 2, prepare)
    assert ev == [] and not (run / "stages/2" / sc.FILE).exists()


# ---- the bundle ------------------------------------------------------------------------------------------------------------

def passing_chips(run: Path) -> list[int]:
    result = json.loads((run / "stages/4/result.json").read_text())
    return [c["chips"] for c in result["configs"] if c.get("pass") is True]


def chat_block(tmp_path, records, *, ledger=True, mutate=None, failed_write=False):
    """The chat_template block of RISKS.md for a run whose stage 2 record is `records["stage2"]` and whose passing stage 4 configurations each
    have `records[chips]`. With `ledger` the supervisor's evidence entry (path and sha256) is appended for each record file written;
    `mutate(run)` runs after that, so a file can change after the ledger saw it."""
    from orchard.ledger import Ledger
    from orchard.stages import evidence_record
    from test_operator_bundle_template import CHAT_HAZARD, build, bundle_of, make_run, section, write_json
    run = make_run(tmp_path)
    p = run / "stages/0/delta.json"
    d = json.loads(p.read_text())
    d["hazards"] = [*d.get("hazards", []), CHAT_HAZARD]
    write_json(p, d)
    paths = {}
    for key, rec in records.items():
        rel = f"stages/2/{sc.FILE}" if key == "stage2" else f"stages/4/configs/{key}/{sc.FILE}"
        write_json(run / rel, rec)
        paths[rel] = rec
    if ledger:
        with Ledger(run / "ledger.jsonl") as led:
            for rel in paths:
                led.append("evidence", int(rel.split("/")[1]), what="served chat check", ok=True, problems=[], incomplete=0,
                           **evidence_record(run, run / rel))
            if failed_write:
                led.append("decision", 2, decision="served chat check not written", reason="OSError: disk full")
    if mutate:
        mutate(run)
    assert build(run).returncode == 0
    risks = (bundle_of(run) / "RISKS.md").read_text()
    return [b for b in section(risks, "## Stage 0 hazards").split("### ") if b.startswith("chat_template")][0], run


def good_record(tmp_path):
    served, new = pair(tmp_path)
    return sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))


def all_good(tmp_path):
    """Records for stage 2 and for every stage 4 configuration the fake run passed."""
    from test_operator_bundle_template import make_run
    rec = good_record(tmp_path / "rec")
    chips = passing_chips(make_run(tmp_path / "probe"))
    return {"stage2": rec, **{c: rec for c in chips}}


def test_clean_records_make_the_chat_template_hazard_dealt_with_and_name_what_they_did_not_check(tmp_path):
    block, _ = chat_block(tmp_path / "run", all_good(tmp_path))
    assert "Dealt with: yes" in block and "stages/2/served-chat.json" in block
    assert "chat requests" in block and "model card" in block          # what was checked and what was not


def test_a_record_with_a_problem_leaves_the_hazard_not_shown_and_names_the_problem(tmp_path):
    served, new = pair(tmp_path / "rec", template=TEMPLATE_B)
    bad = sc.build_record(served, new, swap_check(tmp_path / "rec", [reply(), reply()]))
    recs = {**all_good(tmp_path), "stage2": bad}
    block, _ = chat_block(tmp_path / "run", recs)
    assert "Dealt with: Not shown" in block and "template_differs_from_new" in block


def test_a_record_that_checked_no_chat_reply_leaves_the_hazard_not_shown(tmp_path):
    served, new = pair(tmp_path / "rec")
    none_checked = sc.build_record(served, new, swap_check(tmp_path / "rec", []))
    block, _ = chat_block(tmp_path / "run", {**all_good(tmp_path), "stage2": none_checked})
    assert "Dealt with: Not shown" in block and "checked no chat reply" in block


@pytest.mark.parametrize("bad", [[], 5, {"ok": "yes"}, {"ok": True, "problems": "x"}, {"ok": True}, {"ok": True, "problems": [], "replies": {}}])
def test_a_served_chat_file_of_the_wrong_shape_is_never_believed(tmp_path, bad):
    block, _ = chat_block(tmp_path, {"stage2": bad})
    assert "Dealt with: Not shown" in block


def test_a_record_overwritten_after_the_supervisor_wrote_it_is_not_believed(tmp_path):
    def forge(run):
        (run / "stages/2" / sc.FILE).write_text(json.dumps({**good_record(tmp_path / "forged"), "problems": []}) + "\n")
    block, _ = chat_block(tmp_path / "run", all_good(tmp_path), mutate=forge)
    assert "Dealt with: Not shown" in block and "does not match the sha256 the ledger recorded" in block


def test_a_record_with_no_ledger_entry_is_not_believed(tmp_path):
    block, _ = chat_block(tmp_path / "run", all_good(tmp_path), ledger=False)
    assert "Dealt with: Not shown" in block and "no ledger entry" in block


def test_a_passing_stage_4_configuration_without_a_record_leaves_the_hazard_not_shown(tmp_path):
    recs = all_good(tmp_path)
    dropped = [k for k in recs if k != "stage2"][0]
    del recs[dropped]
    block, _ = chat_block(tmp_path / "run", recs)
    assert "Dealt with: Not shown" in block and f"stages/4/configs/{dropped}/served-chat.json" in block and "missing or unreadable" in block


def test_a_stage_4_configuration_with_a_problem_leaves_the_hazard_not_shown_even_when_stage_2_is_clean(tmp_path):
    recs = all_good(tmp_path)
    served, new = pair(tmp_path / "bad", template=TEMPLATE_B)
    key = [k for k in recs if k != "stage2"][0]
    recs[key] = sc.build_record(served, new, swap_check(tmp_path / "bad", [reply(), reply()]))
    block, _ = chat_block(tmp_path / "run", recs)
    assert "Dealt with: Not shown" in block and f"stages/4/configs/{key}/served-chat.json" in block and "template_differs_from_new" in block


def test_an_incomplete_record_leaves_the_hazard_not_shown(tmp_path):
    served, new = pair(tmp_path / "rec")
    (served / "tokenizer_config.json").write_text("{nope")
    bad = sc.build_record(served, new, swap_check(tmp_path / "rec", [reply(), reply()]))
    block, _ = chat_block(tmp_path / "run", {**all_good(tmp_path), "stage2": bad})
    assert "Dealt with: Not shown" in block and "incomplete" in block


def test_a_failed_write_the_ledger_names_leaves_the_hazard_not_shown(tmp_path):
    block, _ = chat_block(tmp_path / "run", all_good(tmp_path), failed_write=True)
    assert "Dealt with: Not shown" in block and "failed to write" in block


def test_a_variant_record_the_ledger_names_is_required_and_a_missing_one_leaves_the_hazard_not_shown(tmp_path):
    from orchard.ledger import Ledger
    from orchard.stages import evidence_record

    def name_a_variant(run):
        rel = f"stages/2/graft-trial/variant/{sc.FILE}"
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text("{}")
        with Ledger(run / "ledger.jsonl") as led:
            led.append("evidence", 2, what="served chat check", ok=True, problems=[], incomplete=0, **evidence_record(run, run / rel))
        (run / rel).unlink()
    block, _ = chat_block(tmp_path / "run", all_good(tmp_path), mutate=name_a_variant)
    assert "Dealt with: Not shown" in block and "graft-trial/variant/served-chat.json" in block


def test_a_reply_that_ended_in_a_tool_call_counts_as_stopped():
    calls = [{**reply(finish="tool_calls"), "tool_calls": [{"name": "f"}]} for _ in range(3)]
    rec = sc.check_replies(suite(calls))
    assert rec["counts"]["stopped"] == 3 and "no_reply_stopped" not in names(rec)


def test_a_reply_with_an_error_is_not_counted_even_when_it_has_text():
    rec = sc.check_replies(suite([reply("assistant\nhi", finish="length", error="HTTP 500"), reply(finish="length", error="HTTP 500")]))
    assert rec["counts"]["checked"] == 0 and rec["problems"] == []


def test_a_swap_config_naming_a_snapshot_that_is_not_a_directory_writes_nothing(tmp_path):
    def prepare(run):
        d = stage_files(run, "stages/2", tmp_path, entries=[reply()])
        (d / "swap_config.json").write_text(json.dumps({"new_snapshot": str(tmp_path / "gone")}))
    run, ev = end_stage(tmp_path, 2, prepare)
    assert ev == [] and not (run / "stages/2" / sc.FILE).exists()


def test_stage_2_also_checks_the_graft_and_sampling_trial_variants_that_served(tmp_path):
    def prepare(run):
        stage_files(run, "stages/2", tmp_path, entries=[reply(), reply()])
        for trial, name in (("graft-trial", "g"), ("sampling-trial", "s")):
            v = run / "stages/2" / trial / "variant"
            make(v / "model-dir", gen=GOOD_GEN)
            (v / "evidence").mkdir(parents=True)
            (v / "evidence" / "swap-check.json").write_text(json.dumps({"prompt_suite": suite([reply("assistant\nhi"), reply()],
                                                                                                files=sc.file_digests(v / "model-dir"))}))
        make(run / "stages/2" / "sampling-trial" / "variant2" / "model-dir", gen=GOOD_GEN)       # not a variant the trials write: ignored
    run, ev = end_stage(tmp_path, 2, prepare)
    paths = sorted(e["data"]["path"] for e in ev)
    assert paths == ["stages/2/graft-trial/variant/served-chat.json", "stages/2/sampling-trial/variant/served-chat.json", "stages/2/served-chat.json"]
    bad = json.loads((run / "stages/2/graft-trial/variant" / sc.FILE).read_text())
    assert "role_header_start" in {p["name"] for p in bad["problems"]}


def test_a_trial_variant_that_left_no_swap_check_is_not_checked(tmp_path):
    def prepare(run):
        stage_files(run, "stages/2", tmp_path, entries=[reply(), reply()])
        make(run / "stages/2/graft-trial/variant/model-dir", gen=GOOD_GEN)
    run, ev = end_stage(tmp_path, 2, prepare)
    assert [e["data"]["path"] for e in ev] == ["stages/2/served-chat.json"]


def test_a_failure_to_write_the_record_is_a_ledger_decision_and_the_stage_still_passes(tmp_path, monkeypatch):
    from orchard.ledger import Ledger
    from orchard.stages import GateResult, spec_for
    from test_supervisor_stock import Rig, _supervisor

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    r = Rig(tmp_path)
    try:
        r.run()
        stage_files(r.run_dir, "stages/2", tmp_path, entries=[reply(), reply()])
        monkeypatch.setattr(sc, "write_record", boom)
        sup, led = _supervisor(r)
        try:
            spec = spec_for(2, "weights-only")
            stage = r.run_dir / "stages/2"
            if spec.gate_file and not (stage / spec.gate_file).exists():
                (stage / spec.gate_file).write_text("{}")
            assert sup._end_stage(spec, stage, "pass", [], GateResult(True, (), ()), False) == "pass"
            entries = led.read()
        finally:
            led.__exit__(None, None, None)
        assert any(e["data"].get("decision") == "served chat check not written" and "No space" in e["data"]["reason"] for e in entries)
        assert any(e["event"] == "stage_end" and e["stage"] == 2 and e["data"]["result"] == "pass" for e in entries)
    finally:
        r.close()


# ---- review of PR 294: incomplete checks, end ids, and evidence tied to the ledger -------------

def test_an_oversized_template_file_is_incomplete_and_does_not_fall_back_to_the_inline_template(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "MAX_TEMPLATE_BYTES", 10)
    new = make(tmp_path / "new", template=TEMPLATE_A, gen=GOOD_GEN, inline=TEMPLATE_A)
    served = make(tmp_path / "served", template=TEMPLATE_B, gen=GOOD_GEN, inline=TEMPLATE_A)      # a different file, the same inline copy
    rec = sc.check_files(served, new)
    assert any("chat_template.jinja could not be read" in i for i in rec["incomplete"]) and rec["template"]["served_sha256"] is None
    assert sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))["ok"] is False


def test_an_eos_token_whose_id_cannot_be_found_is_incomplete(tmp_path):
    new = make(tmp_path / "new", gen=GOOD_GEN, tokenizer={"added_tokens": [], "model": {"vocab": {}}})
    served = make(tmp_path / "served", gen=GOOD_GEN)
    rec = sc.check_files(served, new)
    assert any("eos token" in i and "new directory" in i for i in rec["incomplete"])
    assert sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))["ok"] is False


def test_a_served_tokenizer_that_gives_the_eos_string_another_id_is_a_problem(tmp_path):
    other = {"added_tokens": [{"id": 99, "content": "<|im_end|>", "special": True}, {"id": 8, "content": "<|endoftext|>", "special": True}], "model": {"vocab": {}}}
    new = make(tmp_path / "new", gen=GOOD_GEN)
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=other)
    rec = sc.check_files(served, new)
    assert "eos_id_differs" in names(rec) and rec["eos_ids"] == {"served": 99, "new": 7}


def test_a_served_stop_id_the_new_model_does_not_end_a_turn_with_is_a_problem(tmp_path):
    new = make(tmp_path / "new", gen={"eos_token_id": [7]})
    served = make(tmp_path / "served", gen={"eos_token_id": [7, 99]})
    rec = sc.check_files(served, new)
    assert "unexpected_stop_id" in names(rec) and "99" in next(p["detail"] for p in rec["problems"] if p["name"] == "unexpected_stop_id")


def test_a_served_stop_id_set_that_matches_exactly_has_no_unexpected_id(tmp_path):
    served, new = pair(tmp_path)
    assert "unexpected_stop_id" not in names(sc.check_files(served, new))


def test_a_unigram_list_vocabulary_gives_the_eos_id_by_position(tmp_path):
    uni = {"added_tokens": [], "model": {"vocab": [["<unk>", 0.0], ["</s>", 0.0], ["a", -1.0]]}}
    tok = {**TOK, "eos_token": "</s>", "pad_token": None}
    new = make(tmp_path / "new", tokens=tok, tokenizer=uni, gen=None)
    served = make(tmp_path / "served", tokens=tok, tokenizer=uni, gen={"eos_token_id": 1})
    rec = sc.check_files(served, new)
    assert rec["eos_ids"] == {"served": 1, "new": 1} and rec["problems"] == [] and rec["incomplete"] == []
    assert rec["end_tokens"] == ["</s>"]


def test_a_template_list_with_an_entry_whose_name_is_not_a_string_is_ignored_not_fatal(tmp_path):
    inline = [{"name": ["x"], "template": "bad"}, {"name": {"a": 1}, "template": "worse"}, {"name": "default", "template": TEMPLATE_A}]
    new = make(tmp_path / "new", template=None, inline=inline, gen=GOOD_GEN)
    assert sc.check_files(new, new)["template"]["new_sha256"] == sc._sha(TEMPLATE_A)


def test_the_end_tokens_are_the_strings_of_the_new_models_end_ids_not_a_spelling_guess(tmp_path):
    tk = {"added_tokens": [{"id": 7, "content": "<|eot_id|>", "special": True}, {"id": 8, "content": "<|vision_end|>", "special": True}],
          "model": {"vocab": {}}}
    new = make(tmp_path / "new", tokens={**TOK, "eos_token": "<|eot_id|>"}, tokenizer=tk, gen={"eos_token_id": 7})
    served = make(tmp_path / "served", tokens={**TOK, "eos_token": "<|eot_id|>"}, tokenizer=tk, gen={"eos_token_id": 7})
    rec = sc.build_record(served, new, swap_check(tmp_path, [reply("Done.<|eot_id|>"), reply("A <|vision_end|> marker.")]))
    problems = {p["name"]: p["detail"] for p in rec["problems"]}
    assert "stop_token_in_reply" in problems and "<|eot_id|>" in problems["stop_token_in_reply"] and "vision_end" not in problems["stop_token_in_reply"]
    assert rec["files"]["end_tokens"] == ["<|eot_id|>"]


def test_reasoning_only_replies_count_and_their_text_is_checked(tmp_path):
    thinking = [{**reply(content="", finish="length"), "reasoning": "assistant\nlet me think"} for _ in range(3)]
    rec = sc.check_replies(suite(thinking))
    assert rec["counts"]["checked"] == 3 and {"role_header_start", "no_reply_stopped"} <= names(rec)


def test_a_recorded_template_digest_that_differs_from_the_files_is_a_problem_and_a_missing_one_is_incomplete(tmp_path):
    served, new = pair(tmp_path)
    p = tmp_path / "swap-check.json"
    stale = suite([reply(), reply()], template=TEMPLATE_B)
    p.write_text(json.dumps({"prompt_suite": stale}))
    rec = sc.build_record(served, new, p)
    assert "template_changed_since_replies" in {x["name"] for x in rec["problems"]} and rec["ok"] is False
    nodigest = suite([reply(), reply()])
    del nodigest["chat_template"]["sha256"]
    p.write_text(json.dumps({"prompt_suite": nodigest}))
    rec = sc.build_record(served, new, p)
    assert rec["ok"] is False and any("digest" in i for i in rec["incomplete"]) and rec["problems"] == []


def test_an_incomplete_record_is_never_ok_even_with_no_problem_and_checked_replies(tmp_path):
    served, new = pair(tmp_path)
    (served / "tokenizer_config.json").write_text("{nope")
    rec = sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))
    assert rec["replies"]["checked"] is True and rec["incomplete"] and rec["ok"] is False


# ---- review of PR 294, round 2 ---------------------------------------------------------------------------------------------------

def test_a_new_model_whose_config_nests_its_end_ids_under_text_config_has_them_counted(tmp_path):
    new = make(tmp_path / "new", cfg={"text_config": {"eos_token_id": [7, 8]}})
    served = make(tmp_path / "served", gen={"eos_token_id": [7]})
    rec = sc.check_files(served, new)
    assert rec["stop_ids"]["new_model_ends"] == [7, 8] and "stop_id_missing" in names(rec)


def test_a_secondary_end_id_that_maps_to_another_token_in_the_served_tokenizer_is_a_problem(tmp_path):
    other = {"added_tokens": [{"id": 7, "content": "<|im_end|>", "special": True}, {"id": 8, "content": "<|ordinary|>", "special": True}],
             "model": {"vocab": {}}}
    new = make(tmp_path / "new", gen=GOOD_GEN)
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=other)
    rec = sc.check_files(served, new)
    assert "end_id_token_differs" in names(rec) and "8" in next(p["detail"] for p in rec["problems"] if p["name"] == "end_id_token_differs")


def test_a_named_special_token_with_another_id_in_the_served_tokenizer_is_a_problem_and_an_unresolved_one_is_incomplete(tmp_path):
    other = {"added_tokens": [{"id": 7, "content": "<|im_end|>", "special": True}, {"id": 11, "content": "<|endoftext|>", "special": True}],
             "model": {"vocab": {}}}
    new = make(tmp_path / "new", gen=GOOD_GEN)
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=other)
    assert "special_token_id_differs" in names(sc.check_files(served, new))
    gone = {"added_tokens": [{"id": 7, "content": "<|im_end|>", "special": True}], "model": {"vocab": {}}}
    served2 = make(tmp_path / "served2", gen=GOOD_GEN, tokenizer=gone)
    new2 = make(tmp_path / "new2", gen=GOOD_GEN, tokenizer=gone)
    assert any("pad_token" in i for i in sc.check_files(served2, new2)["incomplete"])


def test_a_new_model_with_no_known_end_id_is_incomplete_not_clean(tmp_path):
    tokens = {**TOK, "eos_token": None}
    new = make(tmp_path / "new", tokens=tokens)
    served = make(tmp_path / "served", tokens=tokens, gen={"eos_token_id": [123]})
    rec = sc.check_files(served, new)
    assert any("no end id" in i for i in rec["incomplete"]) and "unexpected_stop_id" not in names(rec)


def test_the_replies_are_tied_to_the_tokenizer_and_stop_files_the_swap_test_saw(tmp_path):
    served, new = pair(tmp_path)
    p = swap_check(tmp_path, [reply(), reply()])
    assert sc.build_record(served, new, p)["ok"] is True
    (served / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 8, 9]}))      # changed after the test
    rec = sc.build_record(served, new, p)
    assert "inputs_changed_since_replies" in {x["name"] for x in rec["problems"]} and rec["ok"] is False
    assert "generation_config.json" in next(x["detail"] for x in rec["problems"] if x["name"] == "inputs_changed_since_replies")


def test_a_swap_check_that_recorded_no_input_digests_makes_the_record_incomplete(tmp_path):
    served, new = pair(tmp_path)
    p = tmp_path / "swap-check.json"
    p.write_text(json.dumps({"prompt_suite": suite([reply(), reply()])}))
    rec = sc.build_record(served, new, p)
    assert rec["ok"] is False and any("served_files" in i for i in rec["incomplete"])


def test_file_digests_are_sha256_of_the_raw_bytes_and_none_for_a_file_that_is_absent(tmp_path):
    served, _ = pair(tmp_path)
    d = sc.file_digests(served)
    assert d["tokenizer.json"] == __import__("hashlib").sha256((served / "tokenizer.json").read_bytes()).hexdigest()
    assert d["config.json"] is None and set(d) == set(sc.INPUT_FILES)


def test_a_recorded_template_digest_with_no_template_on_disk_now_is_incomplete(tmp_path):
    new = make(tmp_path / "new", template=None, gen=GOOD_GEN)
    served = make(tmp_path / "served", template=None, gen=GOOD_GEN)
    rec = sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))
    assert rec["ok"] is False and any("template" in i and "on disk now" in i for i in rec["incomplete"])


@pytest.mark.parametrize("entries", [[reply(finish="length")], [reply(finish=None), reply(finish=None)], [reply(finish="weird")]])
def test_replies_of_which_none_was_seen_to_stop_make_the_record_incomplete(tmp_path, entries):
    served, new = pair(tmp_path)
    rec = sc.build_record(served, new, swap_check(tmp_path, entries))
    assert rec["ok"] is False and any("seen to stop" in i for i in rec["incomplete"])


def test_a_template_with_windows_line_endings_matches_the_digest_the_suite_computes(tmp_path):
    crlf = TEMPLATE_A.replace("\n", "\r\n")
    new = make(tmp_path / "new", template=None, gen=GOOD_GEN)
    served = make(tmp_path / "served", template=None, gen=GOOD_GEN)
    for d in (new, served):
        (d / "chat_template.jinja").write_bytes(crlf.encode())
    suite_digest = sc._sha((served / "chat_template.jinja").read_text(encoding="utf-8"))          # what serve_and_compare.py computes
    p = tmp_path / "swap-check.json"
    p.write_text(json.dumps({"prompt_suite": {**suite([reply(), reply()], files=sc.file_digests(served)),
                                              "chat_template": {"present": True, "sha256": suite_digest}}}))
    rec = sc.build_record(served, new, p)
    assert "template_changed_since_replies" not in {x["name"] for x in rec["problems"]} and rec["ok"] is True


def test_the_swap_template_records_the_digests_of_the_files_it_served(tmp_path):
    from test_weights_swap_templates import load_swap_template
    mod = load_swap_template()
    served, _ = pair(tmp_path)
    assert mod.served_files(served) == sc.file_digests(served)


def test_a_snapshot_path_in_a_swap_config_is_resolved_against_the_run_directory(tmp_path):
    def prepare(run):
        d = stage_files(run, "stages/2", tmp_path, entries=[reply(), reply()])
        new = Path(json.loads((d / "swap_config.json").read_text())["new_snapshot"])
        rel = os.path.relpath(new, run)
        (d / "swap_config.json").write_text(json.dumps({"new_snapshot": rel}))
    run, ev = end_stage(tmp_path, 2, prepare)
    assert [e["data"]["path"] for e in ev] == ["stages/2/served-chat.json"]


def test_a_stage_4_configuration_with_no_swap_check_is_skipped(tmp_path):
    def prepare(run):
        stage_files(run, "stages/4/configs/2", tmp_path, entries=[reply(), reply()], name="a")
        d = stage_files(run, "stages/4/configs/4", tmp_path, entries=[reply(), reply()], name="b")
        (d / "evidence" / "swap-check.json").unlink()
    run, ev = end_stage(tmp_path, 4, prepare)
    assert [e["data"]["path"] for e in ev] == ["stages/4/configs/2/served-chat.json"]


def test_a_write_failure_decision_names_the_record_path(tmp_path, monkeypatch):
    from orchard.stages import GateResult, spec_for
    from test_supervisor_stock import Rig, _supervisor

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    r = Rig(tmp_path)
    try:
        r.run()
        stage_files(r.run_dir, "stages/2", tmp_path, entries=[reply(), reply()])
        monkeypatch.setattr(sc, "write_record", boom)
        sup, led = _supervisor(r)
        try:
            spec = spec_for(2, "weights-only")
            stage = r.run_dir / "stages/2"
            if spec.gate_file and not (stage / spec.gate_file).exists():
                (stage / spec.gate_file).write_text("{}")
            sup._end_stage(spec, stage, "pass", [], GateResult(True, (), ()), False)
            entries = led.read()
        finally:
            led.__exit__(None, None, None)
        assert any(e["data"].get("decision") == "served chat check not written" and e["data"].get("path") == "stages/2/served-chat.json" for e in entries)
    finally:
        r.close()


# ---- the bundle, round 2 ---------------------------------------------------------------------------------------------------------

def test_a_write_failure_with_no_record_file_is_reported_and_not_hidden_by_the_old_fallback(tmp_path):
    from orchard.ledger import Ledger
    from test_operator_bundle_template import CHAT_HAZARD, build, bundle_of, make_run, section, write_json
    run = make_run(tmp_path)
    p = run / "stages/0/delta.json"
    d = json.loads(p.read_text())
    d["hazards"] = [*d.get("hazards", []), CHAT_HAZARD]
    write_json(p, d)
    with Ledger(run / "ledger.jsonl") as led:
        led.append("decision", 2, decision="served chat check not written", path="stages/2/served-chat.json", reason="OSError: disk full")
    assert build(run).returncode == 0
    block = [b for b in section((bundle_of(run) / "RISKS.md").read_text(), "## Stage 0 hazards").split("### ") if b.startswith("chat_template")][0]
    assert "Dealt with: Not shown" in block and "failed to write" in block and "stages/2/served-chat.json" in block


def test_a_write_failure_that_a_later_successful_record_for_the_same_path_supersedes_is_not_counted(tmp_path):
    from orchard.ledger import Ledger
    from orchard.stages import evidence_record

    def retry(run):
        with Ledger(run / "ledger.jsonl") as led:
            led.append("decision", 2, decision="served chat check not written", path="stages/2/served-chat.json", reason="OSError: disk full")
            led.append("evidence", 2, what="served chat check", ok=True, problems=[], incomplete=0,
                       **evidence_record(run, run / "stages/2" / sc.FILE))
    block, _ = chat_block(tmp_path / "run", all_good(tmp_path), ledger=True, mutate=retry)
    assert "Dealt with: yes" in block


def test_a_record_that_is_a_fifo_or_a_link_to_a_device_is_unverified_and_never_blocks_the_build(tmp_path):
    import threading
    from test_operator_bundle_template import build, bundle_of

    def swap_in_fifo(run):
        f = run / "stages/2" / sc.FILE
        f.unlink()
        os.mkfifo(f)
    box = {}

    def go():
        box["block"], box["run"] = chat_block(tmp_path / "run", all_good(tmp_path), mutate=swap_in_fifo)
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(40)
    assert "block" in box, "the bundle build blocked on a FIFO"
    assert "Dealt with: Not shown" in box["block"]


def test_the_record_that_is_parsed_is_the_record_that_was_hashed(tmp_path, monkeypatch):
    """One read of the bytes serves both: the verdict reads the file once through safeio and hashes and parses that buffer."""
    import re
    text = (Path(__file__).parent.parent / "orchard/skills/operator-bundle-templates/build_bundle.py").read_text()
    body = text[text.index("def served_chat_verdict"):text.index("def chat_template_dealt_with")]
    assert body.count("read_bytes(") == 1 and "path.read_bytes()" not in body and "parse_json_bytes(" in body
