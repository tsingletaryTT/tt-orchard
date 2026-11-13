"""The thin prompt path of stage 2 (issue 677, arm E): one chat request per stub, no tool loop, behind `--thin-port`.

The coder is a fake endpoint (a function with the signature of the agent loop's HTTP client), the inputs are the tiny Qwen4 fixtures of the
port generator, and nothing opens a device or a network connection.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from orchard import op_cheats, port_template as pt, stub_prompts as sp, thin_port as tp

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "port-generator"
SHEET = op_cheats.load_sheet(op_cheats.TTNN_SHEET)
SENTINEL = "SENTINEL_5b1e9c"


@pytest.fixture(autouse=True)
def _no_block_families(monkeypatch):
    """Code under test that generates a port itself (the thin port step) also sees the plan without block families (see _generate_without_families)."""
    monkeypatch.setattr(pt, "BLOCK_FAMILIES", {})



def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Ledger:
    """What the supervisor's ledger offers this module: append(kind, stage, **data)."""

    def __init__(self):
        self.entries = []

    def append(self, kind, stage, **data):
        self.entries.append({"event": kind, "stage": stage, "data": data})

    def decisions(self, name=None):
        return [e["data"] for e in self.entries if e["event"] == "decision" and (name is None or e["data"].get("decision") == name)]


def inputs():
    return tp.Inputs(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"),
                     tensors=_load("tiny-qwen4-tensors.json"), inventory=_load("tiny-qwen4-ops-inventory.json"), excerpts=None,
                     construction_text=None)


def target(user: str):
    """(function, arguments) the request asks for: the last line of the user text says it."""
    m = re.findall(r"Write `(\w+)` with these arguments: ([\w, ]+)\.", user)
    return m[-1][0], m[-1][1]


class Coder:
    """A fake OpenAI-compatible endpoint. `reply(n, request, function, args)` returns the content of the n-th request."""

    def __init__(self, reply=None, error=None):
        self.bodies, self.urls, self.timeouts = [], [], []
        self.reply, self.error = reply, error

    def good(self, n, body, function, args):
        return f"```python\ndef {function}({args}):\n    return x  # {SENTINEL}\n```"

    def __call__(self, url, body, timeout):
        self.urls.append(url)
        self.bodies.append(body)
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        function, args = target(body["messages"][-1]["content"])
        content = (self.reply or self.good)(len(self.bodies), body, function, args)
        return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 42}}


def run(tmp_path, coder, **kw):
    stage_dir = tmp_path / "stages" / "2"
    stage_dir.mkdir(parents=True, exist_ok=True)
    ledger = kw.pop("ledger", None) or Ledger()
    out = tp.run_with_inputs(stage_dir=stage_dir, ledger=ledger, stage=2, post=coder, endpoint="http://127.0.0.1:1/v1", model="m",
                             sampling=kw.pop("sampling", {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096}),
                             inputs=kw.pop("given", None) or inputs(), sheet=SHEET, **kw)
    return out, ledger, stage_dir


def port_text(stage_dir):
    return (stage_dir / "port" / "port_model.py").read_text(encoding="utf-8")


# ---- the flow --------------------------------------------------------------------------------------------------------------------------

def test_every_stub_is_requested_in_turn_and_the_port_ends_with_no_stub(tmp_path):
    coder = Coder()
    out, ledger, stage_dir = run(tmp_path, coder)
    text = port_text(stage_dir)
    result = pt.check_sources({"port_model.py": text})
    assert out.status == "done" and out.stubs_left == []
    assert result["ok"] and result["stubs"] == [] and result["matmul_calls"] >= 1
    assert len(coder.bodies) == out.requests == 15
    assert len(ledger.decisions("thin port request")) == 15
    assert [d["outcome"] for d in ledger.decisions("thin port request")] == ["spliced"] * 15


def test_the_request_is_a_plain_chat_completion_with_the_tiers_sampling_and_no_tools(tmp_path):
    coder = Coder()
    run(tmp_path, coder, sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096, "repetition_penalty": 1.05})
    body, url, timeout = coder.bodies[0], coder.urls[0], coder.timeouts[0]
    assert url == "http://127.0.0.1:1/v1/chat/completions" and timeout == tp.REQUEST_TIMEOUT_S
    assert body["model"] == "m" and body["stream"] is False
    assert "tools" not in body and "tool_choice" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert (body["temperature"], body["top_p"], body["top_k"], body["max_tokens"], body["repetition_penalty"]) == (0.6, 0.95, 20, 4096, 1.05)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_on_sends_no_thinking_switch(tmp_path):
    coder = Coder()
    run(tmp_path, coder, thinking=True)
    assert "chat_template_kwargs" not in coder.bodies[0]


def test_a_refused_reply_is_retried_once_with_the_reason_and_then_spliced(tmp_path):
    def reply(n, body, function, args):
        if n == 1:
            return "I cannot write that."
        return Coder().good(n, body, function, args)
    coder = Coder(reply)
    out, ledger, _ = run(tmp_path, coder)
    assert out.stubs_left == [] and len(coder.bodies) == 16
    assert "was refused" in coder.bodies[1]["messages"][-1]["content"] and "does not parse" in coder.bodies[1]["messages"][-1]["content"]
    outcomes = [d["outcome"] for d in ledger.decisions("thin port request")]
    assert outcomes[:2] == ["refused", "spliced"] and len(outcomes) == 16


def test_a_stub_refused_twice_is_left_as_a_stub_and_the_rest_go_on(tmp_path):
    first = {}

    def reply(n, body, function, args):
        first.setdefault("name", function)
        if function == first["name"]:
            return "no code at all"
        return Coder().good(n, body, function, args)
    coder = Coder(reply)
    out, ledger, stage_dir = run(tmp_path, coder)
    assert out.status == "done" and out.stubs_left == [first["name"]]
    assert len(coder.bodies) == 16
    outcomes = [d["outcome"] for d in ledger.decisions("thin port request")]
    assert outcomes[:2] == ["refused", "left as a stub"]
    assert sp.stub_names(port_text(stage_dir)) == [first["name"]]
    done = ledger.decisions("thin port finished")[0]
    assert done["left"] == [first["name"]] and len(done["filled"]) == 14


def test_the_splice_guard_is_used_on_every_reply(tmp_path):
    def reply(n, body, function, args):
        return f"```python\ndef {function}({args}):\n    exec('x = 1')\n    return x\n```"
    out, ledger, _ = run(tmp_path, Coder(reply), max_requests=4)
    assert out.stubs_left and any("may not call exec" in d.get("reason", "") for d in ledger.decisions("thin port request"))


# ---- the endpoint ----------------------------------------------------------------------------------------------------------------------

def test_an_endpoint_that_fails_ends_the_step_with_an_error_after_one_retry(tmp_path):
    coder = Coder(error=OSError("connection refused"))
    out, ledger, _ = run(tmp_path, coder)
    assert out.status == "error" and "retried once" in out.detail
    assert len(coder.bodies) == 2
    assert ledger.decisions("thin port request")[-1]["outcome"] == "error"


def test_a_reply_cut_off_at_the_token_cap_is_refused(tmp_path):
    class Capped(Coder):
        def __call__(self, url, body, timeout):
            data = super().__call__(url, body, timeout)
            data["choices"][0]["finish_reason"] = "length"
            return data
    out, ledger, _ = run(tmp_path, Capped(), max_requests=2)
    assert out.stubs_left and "cut off" in ledger.decisions("thin port request")[0]["reason"]


def test_a_reply_with_no_content_is_refused(tmp_path):
    class Empty(Coder):
        def __call__(self, url, body, timeout):
            super().__call__(url, body, timeout)
            return {"choices": [{"message": {"content": None}, "finish_reason": "stop"}], "usage": {}}
    out, ledger, _ = run(tmp_path, Empty(), max_requests=2)
    assert out.stubs_left and "no content" in ledger.decisions("thin port request")[0]["reason"]


# ---- the bounds ------------------------------------------------------------------------------------------------------------------------

def test_the_request_cap_is_two_per_stub_plus_two_by_default():
    assert tp.request_cap(15) == 32 and tp.request_cap(0) == 2


def test_the_request_cap_stops_the_step_and_hands_the_rest_over(tmp_path):
    coder = Coder()
    out, ledger, _ = run(tmp_path, coder, max_requests=3)
    assert len(coder.bodies) == 3 and out.status == "done" and len(out.stubs_left) == 12
    assert ledger.decisions("thin port stopped")[0]["why"] == "request cap"


def test_the_prompt_size_cap_stops_the_step_before_a_request_goes_out(tmp_path):
    coder = Coder()
    out, ledger, _ = run(tmp_path, coder, max_prompt_chars=10)
    assert coder.bodies == [] and out.status == "done" and len(out.stubs_left) == 15
    assert ledger.decisions("thin port stopped")[0]["why"] == "prompt size cap"


def test_the_default_caps_are_finite_and_a_request_has_a_timeout():
    assert 0 < tp.PROMPT_CHARS_CAP < 10_000_000 and 0 < tp.REQUEST_TIMEOUT_S <= 3600


# ---- the ledger ------------------------------------------------------------------------------------------------------------------------

def test_the_ledger_holds_no_reply_text_and_no_machine_path(tmp_path):
    def reply(n, body, function, args):
        if n % 2:
            return f"```python\ndef {function}({args}):\n    exec('x')  # {SENTINEL}\n    return x\n```"
        return Coder().good(n, body, function, args)
    out, ledger, _ = run(tmp_path, Coder(reply), max_requests=6)
    text = json.dumps(ledger.entries)
    assert SENTINEL not in text and str(tmp_path) not in text
    for d in ledger.decisions("thin port request"):
        assert set(d) <= {"decision", "function", "prompt_chars", "completion_tokens", "seconds", "outcome", "reason", "attempt", "retry_sampling"}
        assert isinstance(d["prompt_chars"], int) and d["completion_tokens"] == 42


# ---- the port directory ----------------------------------------------------------------------------------------------------------------

def test_a_port_directory_with_another_file_is_not_used(tmp_path):
    stage_dir = tmp_path / "stages" / "2"
    (stage_dir / "port").mkdir(parents=True)
    (stage_dir / "port" / "other.py").write_text("x = 1\n")
    coder = Coder()
    out, ledger, _ = run(tmp_path, coder)
    assert out.status == "skipped" and "other files" in out.detail and coder.bodies == []
    assert not (stage_dir / "port" / "port_model.py").exists()


def test_the_step_writes_only_the_port_file(tmp_path):
    _, _, stage_dir = run(tmp_path, Coder())
    assert sorted(p.name for p in (stage_dir / "port").iterdir()) == ["port_model.py"]


def test_a_resumed_port_keeps_its_filled_functions_and_asks_only_for_the_rest(tmp_path):
    first = Coder()
    out1, _, stage_dir = run(tmp_path, first, max_requests=5)
    assert len(out1.stubs_left) == 10
    filled = port_text(stage_dir)
    second = Coder()
    out2, _, _ = run(tmp_path, second)
    assert len(second.bodies) == 10 and out2.stubs_left == []
    assert all(f in port_text(stage_dir) for f in re.findall(r"^def (_\w+)\(", filled, re.M))


# ---- inputs that cannot be read ------------------------------------------------------------------------------------------------------------

def test_a_snapshot_that_cannot_be_read_gives_up_so_the_agent_prepare_step_runs(tmp_path):
    stage_dir = tmp_path / "stages" / "2"
    stage_dir.mkdir(parents=True)
    out = tp.run(run_dir=tmp_path, stage_dir=stage_dir, snapshot=str(tmp_path / "missing"), python="/nonexistent/python", ledger=Ledger(),
                 stage=2, post=Coder(), endpoint="http://127.0.0.1:1/v1", model="m", sampling={})
    assert out.status == "skipped" and out.detail


def test_the_names_of_the_stubs_are_recorded_in_the_stage_directory_for_the_repair_step(tmp_path):
    """Issue 698: the repair step finds the stub that called a failing generated helper among the functions that were stubs; a function the
    template wrote (`_mlp`, `_forward`) also calls `_linear` and is not one. The file is in the stage directory, not the port directory
    (the step skips a port directory that holds other files)."""
    out, ledger, stage_dir = run(tmp_path, Coder())
    names = json.loads((stage_dir / "thin_stubs.json").read_text())
    assert names and all(isinstance(n, str) for n in names) and len(names) == 15 and "_mlp" not in names
    assert not (stage_dir / "port" / "thin_stubs.json").exists()


def _hub_snapshot(tmp_path, *, config_target_outside=False):
    """A Hugging Face cache snapshot: every file of the snapshot is a link into the model's blobs directory."""
    import struct

    model = tmp_path / "hub" / "models--org--m"
    (model / "blobs").mkdir(parents=True)
    snap = model / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    header = json.dumps({"w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    (model / "blobs" / "h1").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    if config_target_outside:
        (tmp_path / "elsewhere.json").write_text('{"hidden_size": 4}')
        (model / "blobs" / "h2").symlink_to(tmp_path / "elsewhere.json")
    else:
        (model / "blobs" / "h2").write_text('{"hidden_size": 4}')
    (snap / "model.safetensors").symlink_to("../../blobs/h1")
    (snap / "config.json").symlink_to("../../blobs/h2")
    return snap


def test_a_hugging_face_snapshot_whose_files_are_links_into_blobs_is_read(tmp_path):
    # Issue 733: the config read refused every link, so a real hub snapshot always skipped the generator (Gemma-4 run).
    snap = _hub_snapshot(tmp_path)
    got = tp.gather_inputs(run_dir=tmp_path, stage_dir=tmp_path, snapshot=str(snap), python="py", construct=lambda p, s: ({"ok": 1}, ""))
    assert got.config == {"hidden_size": 4}
    assert [t.name for t in got.tensors] == ["w"]


def test_a_config_link_that_leaves_the_snapshot_and_its_blobs_is_still_refused(tmp_path):
    snap = _hub_snapshot(tmp_path, config_target_outside=True)
    with pytest.raises(tp.Skip, match="config.json"):
        tp.gather_inputs(run_dir=tmp_path, stage_dir=tmp_path, snapshot=str(snap), python="py", construct=lambda p, s: ({"ok": 1}, ""))


def test_the_generator_command_line_reads_a_hub_snapshot_config_too(tmp_path):
    snap = _hub_snapshot(tmp_path)
    assert pt.read_snapshot_config(snap) == {"hidden_size": 4}
    outside = _hub_snapshot(tmp_path / "o", config_target_outside=True)
    with pytest.raises(Exception, match="config.json"):
        pt.read_snapshot_config(outside)


# ---- the coder is reached through `connect` only when a stub is sent (issue 713) ---------------------------------------------------------

def test_connect_is_called_once_before_the_first_stub_request_and_its_endpoint_is_used(tmp_path):
    coder, calls = Coder(), []

    def connect(n_stubs):
        calls.append((n_stubs, len(coder.bodies)))
        return "http://127.0.0.1:2/v1", "m2", {"temperature": 0.1, "max_tokens": 4096}
    stage_dir = tmp_path / "stages" / "2"
    stage_dir.mkdir(parents=True)
    out = tp.run_with_inputs(stage_dir=stage_dir, ledger=Ledger(), stage=2, post=coder, endpoint=None, model=None, sampling=None,
                             inputs=inputs(), sheet=SHEET, connect=connect)
    assert out.status == "done" and out.requests == 15
    assert calls == [(15, 0)]                                              # once, with the stub count, before any request
    assert set(coder.urls) == {"http://127.0.0.1:2/v1/chat/completions"} and {b["model"] for b in coder.bodies} == {"m2"}
    assert {b["temperature"] for b in coder.bodies} == {0.1}


def test_a_plan_with_no_stub_never_calls_connect_and_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.undo()                                  # the real block families: the tiny Qwen4 plan has no stub (issue 696)
    coder, calls = Coder(), []
    stage_dir = tmp_path / "stages" / "2"
    stage_dir.mkdir(parents=True)
    out = tp.run_with_inputs(stage_dir=stage_dir, ledger=Ledger(), stage=2, post=coder, endpoint=None, model=None, sampling=None,
                             inputs=inputs(), sheet=SHEET, connect=lambda n: calls.append(n))
    assert out.status == "done" and out.requests == 0 and out.stubs_left == []
    assert calls == [] and coder.bodies == []
    assert pt.stub_lines(port_text(stage_dir)) == []
