"""The role-fit test (orchard/rolefit.py): can a candidate model do the agent role in this loop?

It replays recorded agent turns (the run logs hold each turn's full prompt and the model's reply) against a
candidate server and judges the replies. The server here is the repo's fake model, so nothing touches a
network or a chip."""
import json
from pathlib import Path

import pytest

from fake_model import FakeModel, call, empty, final, truncated
from orchard import rolefit

SYSTEM = {"role": "system", "content": "You are the agent for stage 1."}
USER = {"role": "user", "content": "Start."}


def log_line(turn, *, finish="tool_calls", reply=None, chars=0):
    reply = reply if reply is not None else call("shell", command="ls")
    return json.dumps({"turn": turn, "sent": [SYSTEM, {**USER, "content": "Start." + "x" * chars}],
                       "received": reply, "finish_reason": finish, "usage": {"completion_tokens": 20}})


def write_log(root: Path, name: str, lines: list[str]) -> Path:
    p = root / name / "log" / "run-00001.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


# ---- reading the logs -------------------------------------------------------------------------

def test_turns_are_read_with_their_prompt_reply_and_finish_reason(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [log_line(1), log_line(2, finish="stop", reply={"content": "done"})])
    turns = rolefit.collect_turns([tmp_path], limit=10)
    assert [t.turn for t in turns] == [1, 2]
    assert turns[0].messages[0] == SYSTEM and turns[0].finish_reason == "tool_calls"
    assert turns[1].recorded == {"content": "done"}


def test_selection_is_deterministic_and_spread_across_the_logs(tmp_path):
    for r in "abcd":
        write_log(tmp_path, f"run-{r}/stages/1", [log_line(i) for i in range(1, 11)])
    a = rolefit.collect_turns([tmp_path], limit=8)
    b = rolefit.collect_turns([tmp_path], limit=8)
    assert [(t.source, t.turn) for t in a] == [(t.source, t.turn) for t in b]
    assert len(a) == 8 and len({t.source for t in a}) == 4          # every log contributes


def test_a_prompt_over_the_size_limit_is_left_out(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [log_line(1), log_line(2, chars=100_000)])
    assert [t.turn for t in rolefit.collect_turns([tmp_path], limit=10, max_prompt_chars=50_000)] == [1]


def test_unreadable_lines_and_missing_directories_are_skipped(tmp_path):
    p = write_log(tmp_path, "run-a/stages/1", [log_line(1), "{not json", '{"turn": 3}', log_line(4)])
    assert [t.turn for t in rolefit.collect_turns([tmp_path, tmp_path / "nope"], limit=10)] == [1, 4]
    assert p.exists()


def test_a_recorded_truncated_or_empty_reply_marks_a_failure_turn(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [
        log_line(1), log_line(2, finish="length", reply={"content": None}),
        log_line(3, finish="stop", reply={"content": ""}), log_line(4, finish="stop", reply={"content": "ok"})])
    assert [(t.turn, t.recorded_failure) for t in rolefit.collect_failures([tmp_path])] == [(2, "truncated"),
                                                                                           (3, "empty")]
    assert [t.turn for t in rolefit.collect_turns([tmp_path], limit=10)] == [1, 4]


def test_failure_turns_are_collected_separately_and_all_of_them_are_kept(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [log_line(i) for i in range(1, 30)]
              + [log_line(30, finish="length", reply={"content": None})])
    fails = rolefit.collect_failures([tmp_path])
    assert [t.turn for t in fails] == [30] and fails[0].recorded_failure == "truncated"
    assert all(t.recorded_failure is None for t in rolefit.collect_turns([tmp_path], limit=50))


# ---- judging a reply --------------------------------------------------------------------------

def msg(**kw):
    return {"content": "", "tool_calls": None, **kw}


def tc(name="shell", arguments='{"command": "ls"}'):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.mark.parametrize("message,finish,verdict", [
    (msg(tool_calls=[tc()]), "tool_calls", "ok"),
    (msg(tool_calls=[tc("write_file", '{"path": "a", "content": "b"}')]), "tool_calls", "ok"),
    (msg(content="The stage is finished."), "stop", "ok"),
    (msg(content=""), "stop", "empty"),
    (msg(content=None), "stop", "empty"),
    (msg(content=""), "length", "truncated"),
    (msg(content="some text that was cut", tool_calls=None), "length", "truncated"),
    (msg(tool_calls=[tc("browse")]), "tool_calls", "unknown_tool"),
    (msg(tool_calls=[tc(arguments="{not json")]), "tool_calls", "bad_arguments"),
    (msg(tool_calls=[tc(arguments='["ls"]')]), "tool_calls", "bad_arguments"),
    (msg(tool_calls=[tc(arguments="{}")]), "tool_calls", "bad_arguments"),                 # command missing
    (msg(tool_calls=[tc("write_file", '{"path": "a"}')]), "tool_calls", "bad_arguments"),  # content missing
    (msg(tool_calls=[tc(), tc("browse")]), "tool_calls", "unknown_tool"),
])
def test_each_reply_gets_one_verdict(message, finish, verdict):
    assert rolefit.classify(message, finish) == verdict


def test_a_tool_call_inside_a_length_stop_is_still_truncated():
    """A reply cut at max_tokens may hold half a tool call; it is not usable."""
    assert rolefit.classify(msg(tool_calls=[tc()]), "length") == "truncated"


# ---- replaying against a server ---------------------------------------------------------------

def turns_of(tmp_path, n=6, **kw):
    write_log(tmp_path, "run-a/stages/1", [log_line(i, **kw) for i in range(1, n + 1)])
    return rolefit.collect_turns([tmp_path], limit=n)


def test_a_replay_sends_the_recorded_prompt_with_the_agents_tools_and_settings(tmp_path):
    with FakeModel(lambda r: call("shell", command="ls")) as server:
        out = rolefit.replay(server.endpoint, "fake", turns_of(tmp_path, 2), temperature=0.0)
        sent = server.requests[0]
    assert [r["verdict"] for r in out] == ["ok", "ok"]
    assert sent["messages"][0] == SYSTEM and sent["temperature"] == 0.0
    assert {t["function"]["name"] for t in sent["tools"]} == {"shell", "write_file"}
    assert sent["max_tokens"] == rolefit.defaults.AGENT_MAX_TOKENS and sent["stream"] is False


def test_a_server_error_is_a_result_not_a_crash(tmp_path):
    with FakeModel(lambda r: 500) as server:
        out = rolefit.replay(server.endpoint, "fake", turns_of(tmp_path, 2))
    assert [r["verdict"] for r in out] == ["error", "error"] and "500" in out[0]["detail"]


def test_a_truncated_reply_is_recorded_as_truncated(tmp_path):
    with FakeModel(lambda r: truncated()) as server:
        out = rolefit.replay(server.endpoint, "fake", turns_of(tmp_path, 1))
    assert out[0]["verdict"] == "truncated"


def test_the_replay_records_tokens_and_time(tmp_path):
    with FakeModel(lambda r: call("shell", command="ls")) as server:
        out = rolefit.replay(server.endpoint, "fake", turns_of(tmp_path, 1))
    assert out[0]["completion_tokens"] == 20 and out[0]["seconds"] >= 0


def test_failure_turns_are_replayed_several_times_and_each_repeat_is_counted(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [log_line(1, finish="length", reply={"content": None})])
    fails = rolefit.collect_failures([tmp_path])
    replies = iter([truncated(), call("shell", command="ls"), empty(), call("shell", command="ls"),
                    truncated()])
    with FakeModel(lambda r: next(replies)) as server:
        out = rolefit.replay_failures(server.endpoint, "fake", fails, repeats=5, temperature=0.7)
    assert out[0]["repeats"] == 5 and out[0]["failed_again"] == 3 and out[0]["recorded_failure"] == "truncated"


# ---- the verdict ------------------------------------------------------------------------------

def results(ok, bad):
    return [{"verdict": "ok"}] * ok + [{"verdict": "empty"}] * bad


def test_the_format_bar_is_95_percent():
    assert rolefit.judge_format(results(95, 5))["passed"] is True
    assert rolefit.judge_format(results(94, 6))["passed"] is False
    assert rolefit.judge_format([])["passed"] is False                  # nothing replayed is not a pass


def test_any_repeated_failure_fails_the_failure_replays():
    assert rolefit.judge_failures([{"repeats": 5, "failed_again": 0}])["passed"] is True
    assert rolefit.judge_failures([{"repeats": 5, "failed_again": 1}])["passed"] is False
    assert rolefit.judge_failures([])["passed"] is True                  # no recorded failures to repeat
    assert rolefit.judge_failures([])["note"]


def test_the_overall_verdict_needs_every_part_and_a_booted_canary():
    good = {"canary": {"passed": True}, "format": {"passed": True}, "failures": {"passed": True},
            "throughput": {"decode_tok_s": {"8192": 30.0, "32768": 20.0}}}
    assert rolefit.verdict(good)["passed"] is True
    for part in ("canary", "format", "failures"):
        assert rolefit.verdict({**good, part: {"passed": False}})["passed"] is False


def test_the_verdict_lists_what_failed():
    v = rolefit.verdict({"canary": {"passed": True}, "format": {"passed": False}, "failures": {"passed": False},
                         "throughput": {}})
    assert v["failed"] == ["format", "failures"]


# ---- throughput -------------------------------------------------------------------------------

def test_throughput_separates_prefill_from_decode_at_each_prompt_size():
    # Per size, two timed requests: one token (prefill only) and 65 tokens (prefill plus 64 decoded).
    clock = iter([0, 1, 1, 4,        # 8K: prefill 1 s, full 3 s, so 64 tokens in 2 s
                  4, 6, 6, 16])      # 32K: prefill 2 s, full 10 s, so 64 tokens in 8 s

    def post(url, body, timeout):
        return {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 8000, "completion_tokens": 65}}
    out = rolefit.throughput("http://x", "m", sizes=(8192, 32768), out_tokens=65, http=post,
                             clock=lambda: next(clock))
    assert out["decode_tok_s"] == {"8192": 32.0, "32768": 8.0}
    assert out["prefill_tok_s"] == {"8192": 8000.0, "32768": 4000.0}
    assert out["out_tokens"] == 65


def test_a_throughput_request_asks_for_a_fixed_output_length_and_a_prompt_of_the_requested_size():
    seen = []

    def post(url, body, timeout):
        seen.append(body)
        return {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 8192, "completion_tokens": 64}}
    rolefit.throughput("http://x", "m", sizes=(8192,), out_tokens=64, http=post,
                       clock=iter([0, 1, 1, 4]).__next__)
    assert [b["max_tokens"] for b in seen] == [1, 64] and all(b["temperature"] == 0 for b in seen)
    assert len(seen[0]["messages"][0]["content"]) >= 8192 * 3           # roughly 4 characters a token


def test_a_failed_throughput_request_leaves_a_null_number():
    def post(url, body, timeout):
        raise OSError("refused")
    out = rolefit.throughput("http://x", "m", sizes=(8192,), out_tokens=64, http=post)
    assert out["decode_tok_s"] == {"8192": None} and "refused" in out["errors"]["8192"]


# ---- the command ------------------------------------------------------------------------------

def test_the_command_writes_a_result_file_with_every_part_and_exits_by_the_verdict(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [log_line(i) for i in range(1, 21)]
              + [log_line(21, finish="length", reply={"content": None})])
    out = tmp_path / "result.json"
    answers = iter([final("42")])

    def script(request):
        if request["messages"][0]["content"].startswith("What is 7 times 6"):
            return final("7 times 6 is 42.")
        return call("shell", command="ls")
    with FakeModel(script) as server:
        code = rolefit.main(["--endpoint", server.endpoint, "--model", "fake", "--logs", str(tmp_path),
                             "--out", str(out), "--turns", "10", "--repeats", "2", "--sizes", "128"])
    res = json.loads(out.read_text())
    assert code == 0 and res["verdict"]["passed"] is True
    assert set(res) >= {"model", "endpoint", "canary", "format", "failures", "throughput", "verdict", "when"}
    assert res["format"]["n"] == 10 and res["failures"]["replayed"][0]["repeats"] == 2


def test_the_command_exits_1_when_the_verdict_fails(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [log_line(i) for i in range(1, 11)])
    out = tmp_path / "result.json"
    with FakeModel(lambda r: empty()) as server:
        code = rolefit.main(["--endpoint", server.endpoint, "--model", "fake", "--logs", str(tmp_path),
                             "--out", str(out), "--turns", "5", "--repeats", "1", "--sizes", "128"])
    assert code == 1 and json.loads(out.read_text())["verdict"]["failed"]


def test_the_command_refuses_a_remote_endpoint(tmp_path, capsys):
    code = rolefit.main(["--endpoint", "http://example.com:8000", "--model", "m", "--logs", str(tmp_path),
                         "--out", str(tmp_path / "r.json")])
    assert code == 2 and "local" in capsys.readouterr().err


def test_a_wrong_canary_answer_fails_the_whole_test_even_when_every_replay_is_fine(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [log_line(i) for i in range(1, 6)])
    out = tmp_path / "r.json"

    def script(request):
        if request["messages"][0]["content"].startswith("What is 7 times 6"):
            return final("I am not sure.")
        return call("shell", command="ls")
    with FakeModel(script) as server:
        code = rolefit.main(["--endpoint", server.endpoint, "--model", "fake", "--logs", str(tmp_path),
                             "--out", str(out), "--turns", "5", "--repeats", "1", "--sizes", "128"])
    res = json.loads(out.read_text())
    assert code == 1 and res["verdict"]["failed"] == ["canary"] and res["canary"]["answer"] == "I am not sure."


def test_thinking_off_sends_enable_thinking_false_in_every_request_and_only_then(tmp_path):
    write_log(tmp_path, "run-a/stages/1", [log_line(1)])
    for flag, expected in (([], None), (["--thinking-off"], {"enable_thinking": False})):
        with FakeModel(lambda r: call("shell", command="ls")) as server:
            rolefit.main(["--endpoint", server.endpoint, "--model", "fake", "--logs", str(tmp_path),
                          "--out", str(tmp_path / "r.json"), "--turns", "1", "--repeats", "1",
                          "--sizes", "128", *flag])
            sent = list(server.requests)
        assert sent and all(r.get("chat_template_kwargs") == expected for r in sent), flag
