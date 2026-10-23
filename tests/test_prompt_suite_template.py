# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""serve_and_compare.py and serve_and_compare_container.py send the prompt suite after the 32-token check (issue 15).

Everything runs on the fake server of tests/fake_swap_server.py. The suite is opt-in: only a swap_config.json with
`"prompt_suite": true` makes the templates send it (the supervisor writes that key into the configs it drafts), so older
configs, the stage 7 boot check and the benchmark scripts behave as before."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from orchard import defaults, prompt_suite as ps  # noqa: E402
from test_container_template import crig  # noqa: E402,F401  (a fixture)
from test_weights_swap_templates import NEAREST, TEMPLATES, load_swap_template, report, swap  # noqa: E402,F401  (a fixture)

SUITE = ps.load_suite()
JINJA = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}{% if tools %}tools available{% endif %}"
JINJA_NO_TOOLS = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"


def stage_for_suite(swap, template=JINJA, **tokenizer_config):
    """The stage directory gets prompt_suite.json and a chat template next to the tokenizer."""
    shutil.copy(TEMPLATES / "prompt_suite.json", swap["stage"])
    md = swap["stage"] / "model-dir"
    if template is not None:
        (md / "chat_template.jinja").write_text(template)
    if tokenizer_config:
        (md / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return md


def requests(tmp_path):
    log = tmp_path / "requests.jsonl"
    return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


def run(swap, tmp_path, **kw):
    kw.setdefault("prompt_suite", True)
    extra = kw.pop("server_extra", {})
    return swap["start"]("perfect", server_extra={"request_log": str(tmp_path / "requests.jsonl"), **extra}, **kw)


# ---- opt-in ---------------------------------------------------------------------------------------------------------

def test_without_the_key_nothing_is_sent_and_nothing_is_recorded(swap, tmp_path):
    stage_for_suite(swap)
    r = run(swap, tmp_path, prompt_suite=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "prompt_suite" not in report(swap)
    assert not any(q["path"] == "/v1/chat/completions" for q in requests(tmp_path))


def test_the_default_config_of_the_older_tests_has_no_prompt_suite_key(swap):
    assert "prompt_suite" not in swap["cfg"]


# ---- chat mode ------------------------------------------------------------------------------------------------------

def test_a_model_with_a_chat_template_gets_the_suite_through_the_chat_endpoint(swap, tmp_path):
    md = stage_for_suite(swap)
    r = run(swap, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["prompt_suite"]
    assert rec["ran"] is True and rec["mode"] == "chat"
    assert rec["model_id"] == swap["cfg"]["new_model_id"]            # the gate needs it to tell a control from a copy
    assert [e["id"] for e in rec["entries"]] == [p["id"] for p in SUITE["prompts"]]
    assert rec["skipped"] == []
    assert rec["suite_sha256"] == ps.suite_sha256()
    assert rec["max_tokens"] == defaults.SUITE_MAX_TOKENS
    tpl = rec["chat_template"]
    assert tpl == {"present": True, "source": "chat_template.jinja", "mentions_tools": True,
                   "sha256": hashlib.sha256(JINJA.encode()).hexdigest()}
    chat = [q for q in requests(tmp_path) if q["path"] == "/v1/chat/completions"]
    assert len(chat) == len(SUITE["prompts"])
    for q, p in zip(chat, SUITE["prompts"]):
        body = q["body"]
        assert body["model"] == str(md) and body["messages"] == p["messages"]
        assert body["temperature"] == 0 and body["max_tokens"] == defaults.SUITE_MAX_TOKENS
        assert set(body) <= {"model", "messages", "max_tokens", "temperature", "tools"}
        assert ("tools" in body) == bool(p.get("needs_tools"))
        if "tools" in body:
            assert body["tools"] == p["tools"]


def test_the_requests_of_the_32_token_check_are_the_same_with_the_suite_on(swap, tmp_path):
    stage_for_suite(swap)
    r = run(swap, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    comps = [q for q in requests(tmp_path) if q["path"] == "/v1/completions"]
    assert len(comps) == 33 and all(isinstance(q["body"]["prompt"], list) for q in comps)
    draft = report(swap)["result_draft"]
    assert set(draft) == {"serves", "server_ready_s", "coherent", "free_run_text", "top1_agreement", "n_tokens",
                          "cache_dir", "evidence"}
    assert draft["top1_agreement"] == 1.0


def test_the_special_tokens_come_from_the_tokenizer_file(swap, tmp_path):
    md = stage_for_suite(swap)
    tok = json.loads((md / "tokenizer.json").read_text())
    flags = {"single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
    tok["added_tokens"] = [{"id": 900, "content": "<|im_end|>", "special": True, **flags},
                           {"id": 901, "content": "<|im_start|>", "special": True, **flags},
                           {"id": 902, "content": "<tool_call>", "special": False, **flags}]
    (md / "tokenizer.json").write_text(json.dumps(tok))
    assert run(swap, tmp_path).returncode == 0
    assert report(swap)["prompt_suite"]["special_tokens"] == ["<|im_end|>", "<|im_start|>"]


def test_the_reply_is_recorded_with_reasoning_content_and_tool_calls(swap, tmp_path):
    stage_for_suite(swap)
    replies = [["capital of France", {"content": "Paris.", "reasoning_content": "We need the capital.", "finish_reason": "stop"}],
               ["weather in Lisbon", {"content": None, "reasoning": "Use the tool.", "finish_reason": "tool_calls",
                                      "tool_calls": [{"name": "get_weather", "arguments": '{"city": "Lisbon"}'}]}]]
    assert run(swap, tmp_path, server_extra={"chat_replies": replies}).returncode == 0
    entries = {e["id"]: e for e in report(swap)["prompt_suite"]["entries"]}
    assert entries["chat-capital"]["content"] == "Paris." and entries["chat-capital"]["reasoning"] == "We need the capital."
    assert entries["chat-capital"]["finish_reason"] == "stop" and entries["chat-capital"]["error"] is None
    tool = entries["tool-call"]
    assert tool["content"] == "" and tool["reasoning"] == "Use the tool." and tool["finish_reason"] == "tool_calls"
    assert tool["tool_calls"] == [{"name": "get_weather", "arguments": '{"city": "Lisbon"}'}]
    assert entries["chat-capital"]["mode"] == "chat"


def test_a_refused_request_is_recorded_as_an_error_and_the_run_goes_on(swap, tmp_path):
    stage_for_suite(swap)
    r = run(swap, tmp_path, server_extra={"chat_error_for": "Lisbon"})
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    entries = {e["id"]: e for e in rep["prompt_suite"]["entries"]}
    assert "HTTP 400" in entries["tool-call"]["error"] and entries["tool-call"]["content"] == ""
    assert entries["summarize"]["error"] is None and entries["summarize"]["content"]
    assert rep["result_draft"]["top1_agreement"] == 1.0


# ---- the chat template decides the format ---------------------------------------------------------------------------

def test_a_model_without_a_chat_template_gets_plain_continuation_prompts(swap, tmp_path):
    md = stage_for_suite(swap, template=None)
    r = run(swap, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["prompt_suite"]
    assert rec["mode"] == "completion"
    assert rec["chat_template"] == {"present": False, "source": None, "sha256": None, "mentions_tools": False}
    assert [s["id"] for s in rec["skipped"]] == ["tool-call"]
    assert {e["id"] for e in rec["entries"]} == {p["id"] for p in SUITE["prompts"]} - {"tool-call"}
    assert all(e["mode"] == "completion" for e in rec["entries"])
    plain = [q for q in requests(tmp_path) if q["path"] == "/v1/completions" and isinstance(q["body"]["prompt"], str)]
    assert [q["body"]["prompt"] for q in plain] == [p["text"] for p in SUITE["prompts"] if not p.get("needs_tools")]
    assert not any(q["path"] == "/v1/chat/completions" for q in requests(tmp_path))
    assert plain[0]["body"]["model"] == str(md)
    assert ps.evaluate(report(swap)).reasons == []


def test_a_template_in_tokenizer_config_counts(swap, tmp_path):
    stage_for_suite(swap, template=None, chat_template=JINJA)
    assert run(swap, tmp_path).returncode == 0
    tpl = report(swap)["prompt_suite"]["chat_template"]
    assert tpl["present"] is True and tpl["source"] == "tokenizer_config.json" and tpl["mentions_tools"] is True


def test_a_list_of_named_templates_uses_the_default_one(swap, tmp_path):
    stage_for_suite(swap, template=None,
                    chat_template=[{"name": "rag", "template": "rag template with tools"},
                                   {"name": "default", "template": JINJA_NO_TOOLS}])
    assert run(swap, tmp_path).returncode == 0
    tpl = report(swap)["prompt_suite"]["chat_template"]
    assert tpl["sha256"] == hashlib.sha256(JINJA_NO_TOOLS.encode()).hexdigest() and tpl["mentions_tools"] is False


def test_an_empty_template_is_no_template(swap, tmp_path):
    stage_for_suite(swap, template="   \n")
    assert run(swap, tmp_path).returncode == 0
    assert report(swap)["prompt_suite"]["mode"] == "completion"


def test_the_jinja_file_wins_over_tokenizer_config(swap, tmp_path):
    stage_for_suite(swap, template=JINJA, chat_template="old template")
    assert run(swap, tmp_path).returncode == 0
    assert report(swap)["prompt_suite"]["chat_template"]["source"] == "chat_template.jinja"


def test_a_template_that_does_not_mention_tools_skips_the_tool_prompt_and_the_gate_accepts_it(swap, tmp_path):
    stage_for_suite(swap, template=JINJA_NO_TOOLS)
    assert run(swap, tmp_path).returncode == 0
    rec = report(swap)["prompt_suite"]
    assert rec["mode"] == "chat" and [s["id"] for s in rec["skipped"]] == ["tool-call"]
    assert rec["chat_template"]["mentions_tools"] is False
    assert ps.evaluate(report(swap)).reasons == []
    chat = [q for q in requests(tmp_path) if q["path"] == "/v1/chat/completions"]
    assert not any("tools" in q["body"] for q in chat)


def test_a_healthy_run_passes_the_gate_reading_and_a_looping_one_does_not(swap, tmp_path):
    stage_for_suite(swap)
    assert run(swap, tmp_path).returncode == 0
    assert ps.evaluate(report(swap)).reasons == []
    loop = "Sure. " + "the quick brown fox jumps over " * 5
    assert run(swap, tmp_path, server_extra={"chat_replies": [["pens", loop]]}).returncode == 0
    reasons = ps.evaluate(report(swap)).reasons
    assert len(reasons) == 1 and "arithmetic" in reasons[0] and "mechanical_repetition" in reasons[0]


def test_the_token_limit_is_the_default_and_the_config_can_lower_it(swap, tmp_path):
    stage_for_suite(swap)
    assert load_swap_template().SUITE_MAX_TOKENS == defaults.SUITE_MAX_TOKENS
    assert run(swap, tmp_path, suite_max_tokens=7).returncode == 0
    assert report(swap)["prompt_suite"]["max_tokens"] == 7
    assert {q["body"]["max_tokens"] for q in requests(tmp_path) if q["path"] == "/v1/chat/completions"} == {7}


# ---- failing early --------------------------------------------------------------------------------------------------

def test_a_stage_without_the_suite_file_exits_2_before_any_server_starts(swap, tmp_path):
    md = swap["stage"] / "model-dir"
    (md / "chat_template.jinja").write_text(JINJA)
    r = run(swap, tmp_path)
    assert r.returncode == 2 and "prompt_suite.json" in (r.stdout + r.stderr)
    assert not swap["pid_file"].exists()


def test_a_suite_file_with_a_problem_exits_2_before_any_server_starts(swap, tmp_path):
    stage_for_suite(swap)
    bad = json.loads((swap["stage"] / "prompt_suite.json").read_text())
    bad["prompts"][1]["id"] = bad["prompts"][0]["id"]
    (swap["stage"] / "prompt_suite.json").write_text(json.dumps(bad))
    r = run(swap, tmp_path)
    assert r.returncode == 2 and "duplicate" in (r.stdout + r.stderr) and not swap["pid_file"].exists()


def test_the_recorded_sha256_is_that_of_the_copy_in_the_stage(swap, tmp_path):
    # An edited copy still runs (the template does not know the shipped file); the gate sees the sha256 differ.
    stage_for_suite(swap)
    edited = json.loads((swap["stage"] / "prompt_suite.json").read_text())
    edited["prompts"][0]["messages"][0]["content"] += " Be brief."
    (swap["stage"] / "prompt_suite.json").write_text(json.dumps(edited))
    assert run(swap, tmp_path).returncode == 0
    rec = report(swap)["prompt_suite"]
    assert rec["suite_sha256"] == hashlib.sha256((swap["stage"] / "prompt_suite.json").read_bytes()).hexdigest()
    assert any("not the committed prompt suite" in r for r in ps.evaluate(report(swap)).reasons)


# ---- a control ------------------------------------------------------------------------------------------------------

def control_file(swap, tmp_path, **change):
    """The swap-check.json of an earlier run on the nearest model: the record this template wrote, with the nearest model's id (a
    control of any other model is refused, tests/test_prompt_suite_control_identity.py). `change` overrides record fields."""
    stage_for_suite(swap)
    assert run(swap, tmp_path).returncode == 0
    rep = report(swap)
    rep["prompt_suite"].update({"model_id": NEAREST, **change})
    path = swap["run"] / "stages" / "2" / "control" / "swap-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep))
    return path


def test_a_control_file_is_embedded_with_its_sha256_and_a_run_relative_source(swap, tmp_path):
    path = control_file(swap, tmp_path)
    assert run(swap, tmp_path, suite_control="stages/2/control/swap-check.json").returncode == 0
    ctl = report(swap)["prompt_suite"]["control"]
    assert ctl["source"] == "stages/2/control/swap-check.json"
    assert ctl["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert ctl["record"]["entries"] and ctl["record"]["suite_sha256"] == ps.suite_sha256()
    assert ctl["record"].get("control") is None


def test_an_absolute_control_path_works_and_a_control_of_a_control_is_not_nested(swap, tmp_path):
    path = control_file(swap, tmp_path)
    first = json.loads(path.read_text())
    first["prompt_suite"]["control"] = {"source": "x", "sha256": "0" * 64, "record": {"entries": []}}
    path.write_text(json.dumps(first))
    assert run(swap, tmp_path, suite_control=str(path)).returncode == 0
    assert "control" not in report(swap)["prompt_suite"]["control"]["record"]


def test_without_a_control_the_record_says_so(swap, tmp_path):
    stage_for_suite(swap)
    assert run(swap, tmp_path).returncode == 0
    assert report(swap)["prompt_suite"]["control"] is None


@pytest.mark.parametrize("how", ["missing", "no record", "not json"])
def test_a_control_that_cannot_be_used_exits_2_before_any_server_starts(swap, tmp_path, how):
    stage_for_suite(swap)
    bad = swap["run"] / "bad-control.json"
    if how == "no record":
        bad.write_text(json.dumps({"top1_agreement": 1.0}))
    elif how == "not json":
        bad.write_text("{nope")
    r = run(swap, tmp_path, suite_control=str(bad))
    assert r.returncode == 2 and "control" in (r.stdout + r.stderr) and not swap["pid_file"].exists()


def test_a_shared_finding_between_the_model_and_its_control_does_not_fail(swap, tmp_path):
    loop = "Sure. " + "the quick brown fox jumps over " * 5
    stage_for_suite(swap)
    assert run(swap, tmp_path, server_extra={"chat_replies": [["pens", loop]]}).returncode == 0
    # The control is an earlier run on the nearest model: another model id, and replies that differ on some prompt.
    control = json.loads((swap["stage"] / "evidence" / "swap-check.json").read_text())
    control["prompt_suite"]["model_id"] = "Qwen/Qwen3.8-27B"
    next(e for e in control["prompt_suite"]["entries"] if e["id"] == "list-format")["content"] = "Red\nGreen\nBlue"
    path = swap["run"] / "stages" / "2" / "control" / "swap-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(control))
    assert run(swap, tmp_path, suite_control=str(path), server_extra={"chat_replies": [["pens", loop]]}).returncode == 0
    assert ps.evaluate(report(swap)).reasons == []


def test_a_control_that_is_a_copy_of_the_run_is_refused(swap, tmp_path):
    # The same model served twice: same model id, same replies. It is no control of the nearest model: the template refuses it
    # before a server starts, and a gate that is not told the nearest model still says "same model id".
    path = control_file(swap, tmp_path, model_id=swap["cfg"]["new_model_id"])
    shutil.rmtree(swap["stage"] / "evidence", ignore_errors=True)
    r = run(swap, tmp_path, suite_control=str(path))
    assert r.returncode == 2 and "nearest model" in r.stdout + r.stderr
    rep = json.loads(path.read_text())
    rep["prompt_suite"]["control"] = {"source": "x", "sha256": "0" * 64, "record": dict(rep["prompt_suite"], control=None)}
    assert any("same model id" in x for x in ps.evaluate(rep).reasons)


# ---- the container template runs the same suite ---------------------------------------------------------------------

def test_the_container_template_sends_the_suite_and_records_the_control(crig):
    shutil.copy(TEMPLATES / "prompt_suite.json", crig["cdir"])
    (crig["cdir"] / "model-dir" / "chat_template.jinja").write_text(JINJA)
    log = crig["cdir"].parent.parent.parent / "container-requests.jsonl"
    r = crig["start"](prompt_suite=True, server_extra={"request_log": str(log)})
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())
    rec = rep["prompt_suite"]
    assert rec["ran"] is True and rec["mode"] == "chat" and len(rec["entries"]) == len(SUITE["prompts"])
    assert rep["kind"] == "container" and rep["result_draft"]["top1_agreement"] == 1.0
    sent = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert sum(1 for q in sent if q["path"] == "/v1/chat/completions") == len(SUITE["prompts"])


def test_the_container_template_without_the_key_sends_no_chat_request(crig):
    shutil.copy(TEMPLATES / "prompt_suite.json", crig["cdir"])
    log = crig["cdir"].parent.parent.parent / "container-requests.jsonl"
    r = crig["start"](server_extra={"request_log": str(log)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "prompt_suite" not in json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())
    assert "/v1/chat/completions" not in log.read_text()


def test_the_container_template_exits_2_when_the_suite_file_is_missing(crig):
    r = crig["start"](prompt_suite=True)
    assert r.returncode == 2 and "prompt_suite.json" in (r.stdout + r.stderr)
    assert not (crig["cdir"] / "evidence" / "docker-argv.json").exists()


# ---- cases added with the first mutation run ------------------------------------------------------------------------

def test_a_suite_file_that_is_not_json_exits_2_before_any_server_starts(swap, tmp_path):
    stage_for_suite(swap)
    (swap["stage"] / "prompt_suite.json").write_text("{not json")
    r = run(swap, tmp_path)
    assert r.returncode == 2 and "not JSON" in (r.stdout + r.stderr) and not swap["pid_file"].exists()


def test_the_skip_reason_says_why(swap, tmp_path):
    stage_for_suite(swap, template=JINJA_NO_TOOLS)
    assert run(swap, tmp_path).returncode == 0
    assert "does not mention tools" in report(swap)["prompt_suite"]["skipped"][0]["reason"]
    stage_for_suite(swap, template="  ")
    assert run(swap, tmp_path).returncode == 0
    assert "no chat template" in report(swap)["prompt_suite"]["skipped"][0]["reason"]


def test_a_blank_template_in_tokenizer_config_is_no_template(swap, tmp_path):
    stage_for_suite(swap, template=None, chat_template="   ")
    assert run(swap, tmp_path).returncode == 0
    assert report(swap)["prompt_suite"]["chat_template"]["present"] is False


def test_a_list_of_templates_with_none_named_default_uses_the_first_string(swap, tmp_path):
    stage_for_suite(swap, template=None, chat_template=[{"name": "a", "template": "first one"}, {"name": "b", "template": "second"}])
    assert run(swap, tmp_path).returncode == 0
    assert report(swap)["prompt_suite"]["chat_template"]["sha256"] == hashlib.sha256(b"first one").hexdigest()
