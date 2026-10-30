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
def test_unreadable_or_hostile_json_files_become_notes_never_exceptions(tmp_path, bad):
    served, new = pair(tmp_path)
    (served / "tokenizer_config.json").write_text(bad)
    (served / "generation_config.json").write_text(bad)
    rec = sc.check_files(served, new)
    assert isinstance(rec["problems"], list) and rec["notes"] is not None


def test_a_missing_served_directory_is_a_problem_not_an_exception(tmp_path):
    new = make(tmp_path / "new", gen=GOOD_GEN)
    rec = sc.check_files(tmp_path / "nowhere", new)
    assert "template_missing_in_served" in names(rec)


# ---- the replies -----------------------------------------------------------------------------------------------------------

def reply(content="Paris, with the Eiffel Tower.", finish="stop", error=None, mode="chat"):
    return {"id": "x", "mode": mode, "content": content, "reasoning": "", "tool_calls": [], "finish_reason": finish, "error": error}


def suite(entries, mode="chat", specials=("<|im_end|>", "<|im_start|>", "<|endoftext|>")):
    return {"ran": True, "mode": mode, "max_tokens": 256, "special_tokens": list(specials), "entries": entries,
            "chat_template": {"present": mode == "chat"}}


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
    assert "stop_token_in_reply" in names(sc.check_replies(suite([reply("Paris.<|im_end|>"), reply()])))


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

def swap_check(tmp_path, entries):
    p = tmp_path / "swap-check.json"
    p.write_text(json.dumps({"prompt_suite": suite(entries)}))
    return p


def test_the_record_joins_the_file_check_and_the_reply_check_and_says_it_sent_no_request(tmp_path):
    served, new = pair(tmp_path)
    rec = sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))
    assert rec["ok"] is True and rec["problems"] == [] and rec["files"]["problems"] == [] and rec["replies"]["checked"] is True
    assert "sent no request" in rec["label"] and "card" in rec["not_checked"]


def test_any_problem_makes_the_record_not_ok_and_lists_it_once(tmp_path):
    served, new = pair(tmp_path, template=TEMPLATE_B)
    rec = sc.build_record(served, new, swap_check(tmp_path, [reply("assistant\nhi"), reply()]))
    assert rec["ok"] is False and {p["name"] for p in rec["problems"]} == {"template_differs_from_new", "role_header_start"}


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
    assert any("generation_config.json could not be read" in n for n in sc.check_files(served, new)["notes"])


# ---- the supervisor writes it when stage 2 or stage 4 passes ---------------------------------------------------------------

def stage_files(run: Path, rel: str, tmp_path: Path, *, entries, served_kw=None, name="s"):
    """A stage 2 (rel "stages/2") or one stage 4 configuration (rel "stages/4/configs/2") directory with its served model-dir, swap config
    and swap check, over a new snapshot."""
    d = run / rel
    new = make(tmp_path / f"new-{name}", gen=GOOD_GEN)
    make(d / "model-dir", gen=GOOD_GEN, **(served_kw or {}))
    (d / "swap_config.json").write_text(json.dumps({"new_snapshot": str(new)}))
    (d / "evidence").mkdir(exist_ok=True)
    (d / "evidence" / "swap-check.json").write_text(json.dumps({"prompt_suite": suite(entries)}))
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
    assert rec["ok"] is False and {p["name"] for p in rec["problems"]} == {"template_differs_from_new", "role_header_start"}
    assert len(ev) == 1 and ev[0]["stage"] == 2 and ev[0]["data"]["ok"] is False and ev[0]["data"]["path"] == "stages/2/served-chat.json"
    assert sorted(ev[0]["data"]["problems"]) == ["role_header_start", "template_differs_from_new"]


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

def chat_block(tmp_path, record):
    from test_operator_bundle_template import CHAT_HAZARD, build, bundle_of, make_run, section, write_json
    run = make_run(tmp_path)
    p = run / "stages/0/delta.json"
    d = json.loads(p.read_text())
    d["hazards"] = [*d.get("hazards", []), CHAT_HAZARD]
    write_json(p, d)
    if record is not None:
        write_json(run / "stages/2" / sc.FILE, record)
    assert build(run).returncode == 0
    risks = (bundle_of(run) / "RISKS.md").read_text()
    return [b for b in section(risks, "## Stage 0 hazards").split("### ") if b.startswith("chat_template")][0]


def good_record(tmp_path):
    served, new = pair(tmp_path)
    return sc.build_record(served, new, swap_check(tmp_path, [reply(), reply()]))


def test_a_clean_record_makes_the_chat_template_hazard_dealt_with_and_names_what_it_did_not_check(tmp_path):
    block = chat_block(tmp_path / "run", good_record(tmp_path / "rec"))
    assert "Dealt with: yes" in block and "stages/2/served-chat.json" in block
    assert "chat requests" in block and "model card" in block          # what was checked and what was not


def test_a_record_with_a_problem_leaves_the_hazard_not_shown_and_names_the_problem(tmp_path):
    served, new = pair(tmp_path / "rec", template=TEMPLATE_B)
    rec = sc.build_record(served, new, swap_check(tmp_path / "rec", [reply(), reply()]))
    block = chat_block(tmp_path / "run", rec)
    assert "Dealt with: Not shown" in block and "template_differs_from_new" in block


def test_a_record_that_checked_no_chat_reply_leaves_the_hazard_not_shown(tmp_path):
    served, new = pair(tmp_path / "rec")
    rec = sc.build_record(served, new, swap_check(tmp_path / "rec", []))
    assert "Dealt with: Not shown" in chat_block(tmp_path / "run", rec)


@pytest.mark.parametrize("bad", [[], 5, {"ok": "yes"}, {"ok": True, "problems": "x"}, {"ok": True}])
def test_a_served_chat_file_of_the_wrong_shape_is_never_believed(tmp_path, bad):
    assert "Dealt with: Not shown" in chat_block(tmp_path, bad)


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


def test_a_stage_4_configuration_with_a_problem_leaves_the_hazard_not_shown_even_when_stage_2_is_clean(tmp_path):
    from test_operator_bundle_template import write_json
    good = good_record(tmp_path / "rec")
    served, new = pair(tmp_path / "bad", template=TEMPLATE_B)
    bad = sc.build_record(served, new, swap_check(tmp_path / "bad", [reply(), reply()]))
    from test_operator_bundle_template import CHAT_HAZARD, build, bundle_of, make_run, section
    run = make_run(tmp_path / "run")
    p = run / "stages/0/delta.json"
    d = json.loads(p.read_text())
    d["hazards"] = [*d.get("hazards", []), CHAT_HAZARD]
    write_json(p, d)
    write_json(run / "stages/2" / sc.FILE, good)
    write_json(run / "stages/4/configs/2" / sc.FILE, bad)
    assert build(run).returncode == 0
    risks = (bundle_of(run) / "RISKS.md").read_text()
    block = [b for b in section(risks, "## Stage 0 hazards").split("### ") if b.startswith("chat_template")][0]
    assert "Dealt with: Not shown" in block and "stages/4/configs/2/served-chat.json" in block and "template_differs_from_new" in block
