# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/teacher_data.py (issue 97, first slice): serve a teacher, send prompts, write `messages` JSONL.

The tests run against a small streaming HTTP server on a loopback port the test chooses, as tests/test_long_sweep.py does. No board is used."""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from orchard import teacher_data as td

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / "orchard" / "skills" / "weights-swap-templates"


class Teacher(BaseHTTPRequestHandler):
    """A fake chat server. `mode` picks what it does; `seen` keeps every request body. The reply to a prompt is 'echo: ' plus the last user
    message, cut to max_tokens words, so a test can see which prompt a reply answers."""
    mode = "ok"
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, "body": req})
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        last = req["messages"][-1]["content"]
        if self.mode == "http-500-on-boom" and "boom" in last:
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        words = ("echo: " + last).split()
        limit = req["max_tokens"]
        cut = len(words) > limit
        words = words[:limit]
        if self.mode == "empty":
            words, cut = [], False
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for w in words:
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": w + " "}}]}).encode() + b"\n\n")
        if self.mode == "error-event":
            self.wfile.write(b"data: " + json.dumps({"error": {"message": "backend died"}}).encode() + b"\n\n")
            return
        finish = None if self.mode == "no-finish-reason" else ("length" if cut or self.mode == "always-length" else "stop")
        if finish:
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]}).encode() + b"\n\n")
        if self.mode != "no-usage":
            self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": len(words)}}).encode() + b"\n\n")
        if self.mode != "no-done":
            self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():
    Teacher.mode, Teacher.seen = "ok", []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Teacher)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def jsonl(tmp_path, rows, name="prompts.jsonl"):
    p = tmp_path / name
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
    return p


def user(text):
    return [{"role": "user", "content": text}]


def lines_of(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


# ---- the prompts ---------------------------------------------------------------------------------------------------------------

def test_the_shipped_sets_load_by_name_with_the_sha256_of_their_files():
    for name, file, count in (("suite", "prompt_suite.json", 8), ("agreement", "agreement_prompts.json", 6)):
        prompts, sha, label = td.load_prompts(name)
        assert len(prompts) == count and label == name
        assert sha == hashlib.sha256((SKILLS / file).read_bytes()).hexdigest()
        assert all(isinstance(p["id"], str) and p["messages"][-1]["role"] == "user" for p in prompts)


def test_a_prompt_set_file_and_a_jsonl_file_load(tmp_path):
    setfile = tmp_path / "set.json"
    setfile.write_text(json.dumps({"prompts": [{"id": "a", "kind": "chat", "messages": user("hi")}]}))
    prompts, sha, label = td.load_prompts(str(setfile))
    assert [p["id"] for p in prompts] == ["a"] and label == "set.json" and sha == hashlib.sha256(setfile.read_bytes()).hexdigest()
    path = jsonl(tmp_path, [{"messages": user("one")}, {"id": "named", "messages": user("two")}, user("three")])
    prompts, _, _ = td.load_prompts(str(path))
    assert [p["id"] for p in prompts] == ["p0001", "named", "p0003"] and prompts[2]["messages"] == user("three")


@pytest.mark.parametrize("row,word", [
    ({"messages": []}, "messages"),
    ({"messages": [{"role": "user", "content": 5}]}, "messages"),
    ({"messages": [{"role": "robot", "content": "x"}]}, "role"),
    ({"messages": [{"role": "assistant", "content": "x"}]}, "last message"),
    ({"messages": user("x"), "id": ""}, "id"),
    ({"messages": user("x"), "id": 7}, "id"),
    ("not json", "line 1"),
    ('{"messages": [{"role": "user", "content": "x"}], "extra": NaN}', "line 1"),
    ("[1, 2]", "messages"),
])
def test_a_bad_prompt_line_is_refused_with_its_line_number(tmp_path, row, word):
    with pytest.raises(td.TeacherDataError, match=word):
        td.load_prompts(str(jsonl(tmp_path, [row])))


def test_duplicate_ids_and_an_empty_file_and_too_many_prompts_are_refused(tmp_path):
    with pytest.raises(td.TeacherDataError, match="more than once"):
        td.load_prompts(str(jsonl(tmp_path, [{"id": "a", "messages": user("x")}, {"id": "a", "messages": user("y")}])))
    with pytest.raises(td.TeacherDataError, match="no prompts"):
        td.load_prompts(str(jsonl(tmp_path, [], "empty.jsonl")))
    rows = [{"messages": user("x")}] * (td.MAX_PROMPTS + 1)
    with pytest.raises(td.TeacherDataError, match="at most"):
        td.load_prompts(str(jsonl(tmp_path, rows, "many.jsonl")))


def test_a_missing_or_oversized_file_and_an_unknown_name_are_refused(tmp_path):
    with pytest.raises(td.TeacherDataError, match="cannot be read"):
        td.load_prompts(str(tmp_path / "nope.jsonl"))
    big = tmp_path / "big.jsonl"
    big.write_bytes(b"x" * (td.MAX_FILE_BYTES + 1))
    with pytest.raises(td.TeacherDataError, match="cannot be read|larger"):
        td.load_prompts(str(big))


def test_a_link_is_not_followed(tmp_path):
    real = jsonl(tmp_path, [{"messages": user("x")}], "real.jsonl")
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)
    with pytest.raises(td.TeacherDataError, match="cannot be read"):
        td.load_prompts(str(link))


# ---- the request ---------------------------------------------------------------------------------------------------------------

def test_a_request_is_a_streamed_chat_at_temperature_zero_with_the_stated_limit(server):
    r = td.http_complete("127.0.0.1", server, "m", user("hello there"), 50, None, timeout=20)
    body = Teacher.seen[0]["body"]
    assert Teacher.seen[0]["path"] == "/v1/chat/completions"
    assert body["temperature"] == 0 and body["max_tokens"] == 50 and body["stream"] is True and body["model"] == "m"
    assert body["messages"] == user("hello there") and "chat_template_kwargs" not in body
    assert r["text"].strip() == "echo: hello there" and r["finish_reason"] == "stop"
    assert r["prompt_tokens"] == 11 and r["completion_tokens"] == 3


@pytest.mark.parametrize("choice", [True, False])
def test_the_thinking_choice_is_sent_only_when_named(server, choice):
    td.http_complete("127.0.0.1", server, "m", user("x"), 20, choice, timeout=20)
    assert Teacher.seen[0]["body"]["chat_template_kwargs"] == {"enable_thinking": choice}


@pytest.mark.parametrize("mode,word", [("error-event", "error event"), ("no-done", "DONE"), ("no-finish-reason", "finish reason"), ("empty", "empty")])
def test_a_broken_stream_or_an_empty_reply_is_a_failure(server, mode, word):
    Teacher.mode = mode
    with pytest.raises(ValueError, match=word):
        td.http_complete("127.0.0.1", server, "m", user("x"), 20, None, timeout=20)


def test_a_server_that_reports_no_usage_gives_no_counts(server):
    Teacher.mode = "no-usage"
    r = td.http_complete("127.0.0.1", server, "m", user("x"), 20, None, timeout=20)
    assert r["prompt_tokens"] is None and r["completion_tokens"] is None


# ---- collecting ----------------------------------------------------------------------------------------------------------------

def fake_send(**kw):
    def send(messages, max_tokens):
        text = messages[-1]["content"]
        if text == "boom":
            raise OSError("connection refused")
        if text == "cut":
            return {"text": "partial", "finish_reason": "length", "prompt_tokens": 3, "completion_tokens": max_tokens}
        return {"text": f"answer to {text}", "finish_reason": "stop", "prompt_tokens": 3, "completion_tokens": 4}
    return send


def prompts_of(*texts):
    return [{"id": f"p{i}", "kind": "chat", "messages": user(t)} for i, t in enumerate(texts)]


def test_a_sample_is_the_prompt_messages_with_the_reply_appended_and_the_server_numbers():
    done = td.collect(fake_send(), prompts_of("a"), max_tokens=100, keep_truncated=False)
    s = done["samples"][0]
    assert s["messages"] == user("a") + [{"role": "assistant", "content": "answer to a"}]
    assert (s["id"], s["kind"], s["finish_reason"], s["prompt_tokens"], s["completion_tokens"], s["truncated"]) == ("p0", "chat", "stop", 3, 4, False)
    assert s["prompt_sha256"] == hashlib.sha256(json.dumps(user("a"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def test_the_system_turn_and_earlier_turns_are_kept_unchanged():
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
    done = td.collect(fake_send(), [{"id": "m", "kind": "multi", "messages": msgs}], max_tokens=100, keep_truncated=False)
    assert done["samples"][0]["messages"][:4] == msgs and done["samples"][0]["messages"][4]["role"] == "assistant"


def test_a_reply_cut_off_at_the_limit_is_refused_unless_asked_for():
    done = td.collect(fake_send(), prompts_of("a", "cut", "b"), max_tokens=100, keep_truncated=False)
    assert [s["id"] for s in done["samples"]] == ["p0", "p2"] and done["refused_truncated"] == ["p1"]
    kept = td.collect(fake_send(), prompts_of("a", "cut"), max_tokens=100, keep_truncated=True)
    assert [s["id"] for s in kept["samples"]] == ["p0", "p1"] and kept["samples"][1]["truncated"] is True and kept["refused_truncated"] == []


def test_a_failing_prompt_is_recorded_and_the_rest_go_on():
    done = td.collect(fake_send(), prompts_of("a", "boom", "b"), max_tokens=100, keep_truncated=False)
    assert [s["id"] for s in done["samples"]] == ["p0", "p2"]
    assert done["errors"] == [{"id": "p1", "error": "OSError: connection refused"}]


def test_a_prompt_that_is_not_a_user_turn_at_the_end_is_not_sent():
    sent = []
    td.collect(lambda m, k: sent.append(m) or {"text": "x", "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1},
               [{"id": "x", "kind": "k", "messages": [{"role": "assistant", "content": "hi"}]}], max_tokens=10, keep_truncated=False)
    assert sent == []


# ---- the command ---------------------------------------------------------------------------------------------------------------

def run_main(server, tmp_path, *extra, prompts=None, out="teacher.jsonl"):
    p = prompts or jsonl(tmp_path, [{"id": "one", "messages": user("first question")}, {"id": "two", "messages": user("second question")}])
    return td.main(["run", "--port", str(server), "--model", "m", "--prompts", str(p), "--out", str(tmp_path / out), *extra]), tmp_path / out


def test_the_command_writes_a_header_the_samples_and_a_summary(server, tmp_path, capsys):
    rc, out = run_main(server, tmp_path, "--max-tokens", "100")
    assert rc == 0
    rows = lines_of(out)
    assert [r["type"] for r in rows] == ["header", "sample", "sample", "summary"]
    h, s, summary = rows[0], rows[1:3], rows[3]
    assert h["model"] == {"requested": "m", "verified": False, "note": h["model"]["note"]}
    assert h["settings"] == {"temperature": 0, "max_tokens": 100, "thinking": "server default", "api": "chat", "keep_truncated": False}
    assert h["prompts"]["count"] == 2 and h["prompts"]["sha256"] == hashlib.sha256((tmp_path / "prompts.jsonl").read_bytes()).hexdigest()
    assert [x["id"] for x in s] == ["one", "two"] and s[0]["messages"][-1]["content"].strip() == "echo: first question"
    assert summary == {"type": "summary", "written": 2, "truncated_kept": [], "refused_truncated": [], "errors": []}
    assert capsys.readouterr().out.count("one") >= 1


def test_the_thinking_choice_and_the_truncation_choice_are_recorded_in_the_header(server, tmp_path):
    rc, out = run_main(server, tmp_path, "--thinking", "off", "--keep-truncated", "--max-tokens", "3")
    h = lines_of(out)[0]
    assert h["settings"]["thinking"] == "off" and h["settings"]["keep_truncated"] is True and h["settings"]["max_tokens"] == 3
    assert all(r["body"]["chat_template_kwargs"] == {"enable_thinking": False} for r in Teacher.seen)


def test_a_truncated_reply_makes_the_command_exit_1_but_the_good_samples_are_written(server, tmp_path):
    Teacher.mode = "always-length"
    rc, out = run_main(server, tmp_path)
    rows = lines_of(out)
    assert rc == 1 and [r["type"] for r in rows] == ["header", "summary"] and rows[-1]["refused_truncated"] == ["one", "two"]
    rc, out = run_main(server, tmp_path, "--keep-truncated")
    assert rc == 0 and [r["type"] for r in lines_of(out)] == ["header", "sample", "sample", "summary"]


def test_a_failing_request_makes_the_command_exit_1_and_is_listed(server, tmp_path):
    Teacher.mode = "http-500-on-boom"
    p = jsonl(tmp_path, [{"id": "ok", "messages": user("fine")}, {"id": "bad", "messages": user("boom")}])
    rc, out = run_main(server, tmp_path, prompts=p)
    summary = lines_of(out)[-1]
    assert rc == 1 and summary["written"] == 1 and summary["errors"][0]["id"] == "bad" and "500" in summary["errors"][0]["error"]


def test_two_runs_against_the_same_server_write_the_same_bytes(server, tmp_path):
    _, a = run_main(server, tmp_path, out="a.jsonl")
    _, b = run_main(server, tmp_path, out="b.jsonl")
    assert a.read_bytes() == b.read_bytes()


def test_the_shipped_sets_are_accepted_by_name(server, tmp_path):
    rc = td.main(["run", "--port", str(server), "--model", "m", "--prompts", "agreement", "--out", str(tmp_path / "o.jsonl"), "--max-tokens", "5000"])
    rows = lines_of(tmp_path / "o.jsonl")
    assert rc == 0 and rows[0]["prompts"]["source"] == "agreement" and len([r for r in rows if r["type"] == "sample"]) == 6


def test_an_existing_output_file_is_replaced_whole_and_a_link_is_replaced_not_written_through(server, tmp_path):
    target = tmp_path / "teacher.jsonl"
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("keep me")
    target.symlink_to(elsewhere)
    rc, out = run_main(server, tmp_path)
    assert rc == 0 and elsewhere.read_text() == "keep me" and not out.is_symlink() and out.read_text().startswith('{"type": "header"')


# ---- what the command refuses ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["10.0.0.5", "example.com", "0.0.0.0", ""])
def test_only_a_loopback_address_is_accepted(tmp_path, host, capsys):
    p = jsonl(tmp_path, [{"messages": user("x")}])
    rc = td.main(["run", "--host", host, "--port", "20000", "--model", "m", "--prompts", str(p), "--out", str(tmp_path / "o.jsonl")])
    assert rc == 2 and "loopback" in capsys.readouterr().err and not (tmp_path / "o.jsonl").exists()


@pytest.mark.parametrize("port", [8000, 8100, 8101, 8102, 8103])
def test_the_ports_other_services_use_are_refused(tmp_path, port, capsys):
    p = jsonl(tmp_path, [{"messages": user("x")}])
    rc = td.main(["run", "--port", str(port), "--model", "m", "--prompts", str(p), "--out", str(tmp_path / "o.jsonl")])
    assert rc == 2 and "8000 and 8100 to 8103" in capsys.readouterr().err


def test_the_port_is_required_and_never_guessed(tmp_path):
    with pytest.raises(SystemExit):
        td.main(["run", "--model", "m", "--prompts", "suite", "--out", str(tmp_path / "o.jsonl")])


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "1.5", "100001"])
def test_a_token_limit_that_is_not_sensible_is_refused(tmp_path, bad):
    with pytest.raises(SystemExit):
        td.main(["run", "--port", "20000", "--model", "m", "--prompts", "suite", "--out", str(tmp_path / "o.jsonl"), "--max-tokens", bad])


def test_a_redirect_is_a_failure_and_is_not_followed(tmp_path):
    seen = []

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(307)
            self.send_header("Location", "http://127.0.0.1:1/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with pytest.raises(OSError):
            td.http_complete("127.0.0.1", httpd.server_address[1], "m", user("x"), 10, None, timeout=20)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert seen == ["/v1/chat/completions"]


@pytest.mark.parametrize("event", ["not json", '{"choices": [{"delta": {"content": NaN}}]}', '{"a": 1, "a": 2}', "[1]"])
def test_a_stream_event_that_is_not_usable_json_is_a_failure(event):
    class Bad(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: " + event.encode() + b"\n\ndata: [DONE]\n\n")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Bad)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ValueError, match="stream event"):
            td.http_complete("127.0.0.1", httpd.server_address[1], "m", user("x"), 10, None, timeout=20)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_collect_leaves_out_a_blank_reply_and_lists_it():
    done = td.collect(lambda m, k: {"text": "  \n", "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}, prompts_of("a"),
                      max_tokens=10, keep_truncated=False)
    assert done["samples"] == [] and "empty reply" in done["errors"][0]["error"]


def test_a_system_turn_in_a_prompt_file_survives_loading(tmp_path):
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q"}]
    prompts, _, _ = td.load_prompts(str(jsonl(tmp_path, [{"messages": msgs}])))
    assert prompts[0]["messages"] == msgs


def test_a_prompt_set_file_with_too_many_prompts_is_refused(tmp_path):
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"prompts": [{"id": f"i{n}", "messages": user("x")} for n in range(td.MAX_PROMPTS + 1)]}))
    with pytest.raises(td.TeacherDataError, match="at most"):
        td.load_prompts(str(big))
