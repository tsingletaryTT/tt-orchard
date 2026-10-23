# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The committed prompt suite and its loader (issue 15). The checks that read a model's replies are in
tests/test_prompt_suite_checks.py; the template that sends the prompts is tested in
tests/test_prompt_suite_template.py."""
import copy
import hashlib
import json

import pytest

from orchard import prompt_suite as ps


def suite():
    return json.loads(ps.SUITE_FILE.read_text(encoding="utf-8"))


def test_the_committed_suite_loads_and_has_the_prompts_the_issue_asks_for():
    data = ps.load_suite()
    prompts = data["prompts"]
    assert len(prompts) >= 5            # issue 15: the gate uses at least five prompts
    ids = [p["id"] for p in prompts]
    assert len(set(ids)) == len(ids)
    kinds = {p["kind"] for p in prompts}
    # chat, code, a tool call and a non-English prompt (issue 15), plus a multi-turn one.
    assert {"chat", "code", "tool", "non-ascii", "multi-turn"} <= kinds


def test_the_multi_turn_prompt_has_an_assistant_turn_and_ends_with_the_user():
    p = next(p for p in ps.load_suite()["prompts"] if p["kind"] == "multi-turn")
    roles = [m["role"] for m in p["messages"]]
    assert "assistant" in roles and roles[-1] == "user" and len(roles) >= 3


def test_the_non_english_prompt_is_not_ascii_and_expects_its_own_script():
    p = next(p for p in ps.load_suite()["prompts"] if p["kind"] == "non-ascii")
    assert not p["messages"][-1]["content"].isascii()
    assert p["expect_script"] != "Latin"


def test_only_the_tool_prompt_needs_tools_and_it_carries_them():
    prompts = ps.load_suite()["prompts"]
    needs = [p for p in prompts if p.get("needs_tools")]
    assert [p["kind"] for p in needs] == ["tool"]
    assert needs[0]["tools"] and needs[0]["tools"][0]["function"]["name"]


def test_every_marker_appears_in_its_own_prompt_and_in_no_other():
    prompts = ps.load_suite()["prompts"]
    markers = [(p["id"], p["marker"]) for p in prompts if p.get("marker")]
    assert len(markers) >= 2
    for pid, marker in markers:
        for p in prompts:
            text = json.dumps(p["messages"], ensure_ascii=False) + p["text"]
            assert (marker in text) == (p["id"] == pid), (marker, p["id"])


def test_the_sha256_is_that_of_the_file_bytes():
    assert ps.suite_sha256() == hashlib.sha256(ps.SUITE_FILE.read_bytes()).hexdigest()


def problems_of(mutate):
    data = copy.deepcopy(suite())
    mutate(data)
    return ps.check_suite(data)


def test_the_committed_suite_has_no_problems():
    assert ps.check_suite(suite()) == []


@pytest.mark.parametrize("label,mutate,expected", [
    ("a duplicate id", lambda d: d["prompts"][1].update(id=d["prompts"][0]["id"]), "duplicate"),
    ("no text", lambda d: d["prompts"][0].pop("text"), "text"),
    ("an empty text", lambda d: d["prompts"][0].update(text="  "), "text"),
    ("no messages", lambda d: d["prompts"][0].update(messages=[]), "messages"),
    ("the last message is not the user's",
     lambda d: d["prompts"][0]["messages"].append({"role": "assistant", "content": "x"}), "last message"),
    ("an unknown role", lambda d: d["prompts"][0]["messages"][0].update(role="robot"), "role"),
    ("an unknown script", lambda d: d["prompts"][0].update(expect_script="Klingon"), "script"),
    ("a marker that also sits in another prompt",
     lambda d: d["prompts"][0].update(text=d["prompts"][0]["text"] + " Zephyrine"), "marker"),
    ("a marker that sits in no prompt", lambda d: d["prompts"][0].update(marker="Nowhereword"), "marker"),
    ("tools without needs_tools", lambda d: d["prompts"][0].update(tools=[{"type": "function"}]), "tools"),
    ("needs_tools without tools", lambda d: d["prompts"][5].pop("tools"), "tools"),
    ("fewer than five prompts", lambda d: d.update(prompts=d["prompts"][:4]), "at least"),
    ("no version", lambda d: d.pop("version"), "version"),
])
def test_a_suite_with_a_problem_is_refused_and_the_reason_names_it(label, mutate, expected):
    found = problems_of(mutate)
    assert found, label
    assert any(expected in f for f in found), (label, found)


def test_load_suite_raises_with_every_problem_in_the_message(tmp_path):
    bad = suite()
    bad["prompts"][1]["id"] = bad["prompts"][0]["id"]
    bad["prompts"][2].pop("text")
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ps.SuiteError) as exc:
        ps.load_suite(path)
    assert "duplicate" in str(exc.value) and "text" in str(exc.value)


def test_a_file_that_is_not_json_is_a_suite_error(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text("{not json")
    with pytest.raises(ps.SuiteError):
        ps.load_suite(path)


def test_true_is_not_a_version_number():
    # bool is an int in Python, so isinstance alone would accept it.
    assert any("version" in f for f in problems_of(lambda d: d.update(version=True)))


def test_a_message_with_blank_content_is_refused():
    found = problems_of(lambda d: d["prompts"][0]["messages"][0].update(content="   "))
    assert any("non-empty content" in f for f in found)


@pytest.mark.parametrize("not_a_suite", [[], "text", 7, None])
def test_a_suite_that_is_not_an_object_has_a_problem_and_does_not_crash(not_a_suite):
    assert ps.check_suite(not_a_suite) == ["the suite is not a JSON object"]
