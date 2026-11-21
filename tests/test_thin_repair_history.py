"""Thin repair keeps what its earlier rounds did (issue 783).

In the Aion run one function was spliced in three rounds in a row. Each round fixed the error it was shown and caused another (RuntimeError, then
TypeError, then AttributeError), because a round saw only the current function and the latest error. Two things change:

1. a request after the first carries the earlier rounds: the error each one was asked to fix, the failing line, and the diff it spliced;
2. when the last two rounds spliced the same function and the error changed between them, a third round on that function is not sent. The
   round goes to the agent loop with a note that names the function and the functions that call it.

The unit tests run `thin_repair.run_round` with a fake chat endpoint. The wiring tests run the supervisor's repair rounds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orchard import thin_repair as tr

PORT = ("import ttnn\n\n\ndef _A_linear_attn(model, i, x, ids):\n    return x.shape\n\n\n"
        "def _B_mlp(model, i, x, ids):\n    return _A_linear_attn(model, i, x, ids)\n")
LINE_A = 5            # `return x.shape` in _A_linear_attn
LINE_B = 9            # the call in _B_mlp


def _stage(tmp_path: Path, text: str = PORT) -> Path:
    d = tmp_path / "stage"
    (d / "port").mkdir(parents=True)
    (d / "port" / "port_model.py").write_text(text)
    return d


def _raw(line: int, code: str, etype: str, message: str = "boom") -> str:
    return f"FAILED: the port failed at 'prefill': \"[port_model.py:{line}: {code}] {etype}: {message}\"\n"


def _reply(code: str) -> dict:
    return {"choices": [{"message": {"content": "```python\n" + code + "```"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 7}}


class Endpoint:
    """A fake chat endpoint: it records each request body and answers with the next scripted function source."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.bodies: list = []

    def __call__(self, url, body, timeout=None):
        self.bodies.append(body)
        return _reply(self.replies.pop(0))

    def user(self, n: int = -1) -> str:
        return self.bodies[n]["messages"][-1]["content"]


def go(stage: Path, k: int, raw: str, endpoint, **kw):
    return tr.run_round(stage_dir=stage, k=k, raw_output=raw, post=endpoint, endpoint="http://x/v1", model="m", sampling={}, **kw)


def fn_a(body: str) -> str:
    return f"def _A_linear_attn(model, i, x, ids):\n    {body}\n"


def history(stage: Path):
    return json.loads((stage / tr.HISTORY_NAME).read_text())


# ---- the history is written and sent -------------------------------------------------------------------------------------------------

def test_a_spliced_round_records_the_error_the_failing_line_and_the_diff(tmp_path):
    stage, ep = _stage(tmp_path), Endpoint(fn_a("return x.dims"))
    r = go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    assert r.status == "spliced"
    [rec] = history(stage)
    assert rec["round"] == 1 and rec["function"] == "_A_linear_attn" and rec["error_type"] == "KeyError"
    assert rec["code"] == "return x.shape"
    assert "-    return x.shape" in rec["diff"] and "+    return x.dims" in rec["diff"]
    assert r.diff_sha256 == hashlib.sha256(rec["diff"].encode("utf-8")).hexdigest()    # the ledger carries this hash for the evidence file


def test_the_first_request_has_no_history_section(tmp_path):
    ep = Endpoint(fn_a("return x.dims"))
    go(_stage(tmp_path), 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    assert "earlier thin repair rounds" not in ep.user()


def test_the_next_request_carries_the_earlier_error_line_and_diff(tmp_path):
    stage, ep = _stage(tmp_path), Endpoint(fn_a("return x.dims"), fn_a("return x.dims()"))
    go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    go(stage, 2, _raw(LINE_A, "return x.dims", "TypeError"), ep)
    user = ep.user(1)
    assert "## What earlier thin repair rounds did" in user
    assert "Round 1" in user and "KeyError" in user and "return x.shape" in user
    assert "+    return x.dims" in user and "-    return x.shape" in user
    assert "Do not repeat a change" in user


def test_a_round_that_was_refused_leaves_no_history(tmp_path):
    stage = _stage(tmp_path)
    ep = Endpoint("not python at all (", "still not python (")
    r = go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    assert r.status == "refused" and not (stage / tr.HISTORY_NAME).exists()


def test_history_in_a_request_is_bounded(tmp_path):
    stage = _stage(tmp_path)
    big = [{"round": n, "function": "_A_linear_attn", "error_type": "KeyError", "line": 5, "code": "c" * 50_000, "message": "m" * 50_000,
            "diff": "+" + "d" * 50_000} for n in range(1, 5)]
    (stage / tr.HISTORY_NAME).write_text(json.dumps(big))
    ep = Endpoint(fn_a("return 1"))
    go(stage, 5, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    section = ep.user().split("## What earlier thin repair rounds did")[1].split("## ")[0]
    assert len(section) <= tr.HISTORY_SECTION_MAX
    assert "d" * (tr.HISTORY_DIFF_MAX + 1) not in section                # each diff is cut at its own cap


GOOD_RECORD = {"round": 1, "function": "_A_linear_attn", "error_type": "KeyError", "line": 5, "code": "c", "message": "m", "diff": "d"}


@pytest.mark.parametrize("field,value", [("code", 5), ("message", None), ("diff", ["d"]), ("function", "not an identifier!"), ("error_type", "a b"),
                                          ("round", True), ("round", -1), ("line", 10**9), ("line", "5")])
def test_one_record_with_a_wrong_field_voids_the_whole_file(tmp_path, field, value):
    stage = _stage(tmp_path)
    (stage / tr.HISTORY_NAME).write_text(json.dumps([GOOD_RECORD, {**GOOD_RECORD, "round": 2, field: value}]))
    assert tr.load_history(stage) == []


def test_a_good_history_file_is_read_back(tmp_path):
    stage = _stage(tmp_path)
    (stage / tr.HISTORY_NAME).write_text(json.dumps([GOOD_RECORD, {**GOOD_RECORD, "round": 2}]))
    assert [r["round"] for r in tr.load_history(stage)] == [1, 2]


def test_a_valid_history_file_above_the_size_cap_counts_as_no_history(tmp_path):
    stage = _stage(tmp_path)
    (stage / tr.HISTORY_NAME).write_text(json.dumps([{**GOOD_RECORD, "message": "m" * (tr.HISTORY_FILE_MAX + 10)}]))
    assert tr.load_history(stage) == []


def test_a_malformed_history_file_counts_as_no_history(tmp_path):
    for n, bad in enumerate(("not json", json.dumps({"a": 1}), json.dumps([{"round": "x"}]), json.dumps([1, 2]), "[" * 5000, "x" * 400_000)):
        stage = tmp_path / f"s{n}"
        (stage / "port").mkdir(parents=True)
        (stage / "port" / "port_model.py").write_text(PORT)
        (stage / tr.HISTORY_NAME).write_text(bad)
        ep = Endpoint(fn_a("return x.dims"))
        r = go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
        assert r.status == "spliced" and "earlier thin repair rounds" not in ep.user()
        assert len(history(stage)) == 1                              # the new record replaces the unreadable file


def test_a_long_diff_is_cut_when_it_is_recorded(tmp_path):
    stage = _stage(tmp_path)
    body = "\n".join(f"    v{n} = {n}" for n in range(300))
    ep = Endpoint(f"def _A_linear_attn(model, i, x, ids):\n{body}\n    return x\n")
    r = go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    assert r.status == "spliced"
    [rec] = history(stage)
    assert len(rec["diff"]) <= tr.HISTORY_DIFF_MAX and rec["diff"].endswith("(cut)")


# ---- the stop rule ---------------------------------------------------------------------------------------------------------------------

def two_rounds_on_a(tmp_path, first="KeyError", second="TypeError"):
    stage, ep = _stage(tmp_path), Endpoint(fn_a("return x.dims"), fn_a("return x.dims()"))
    go(stage, 1, _raw(LINE_A, "return x.shape", first), ep)
    go(stage, 2, _raw(LINE_A, "return x.dims", second), ep)
    return stage, ep


def test_a_third_round_on_the_same_function_after_a_changed_error_is_not_sent(tmp_path):
    stage, ep = two_rounds_on_a(tmp_path)
    sent = len(ep.bodies)
    r = go(stage, 3, _raw(LINE_A, "return x.dims()", "AttributeError"), ep)
    assert len(ep.bodies) == sent and r.requests == 0
    assert r.status == "fallback" and r.function == "_A_linear_attn"
    assert "_A_linear_attn" in r.reason and "KeyError" in r.reason and "TypeError" in r.reason
    assert "_B_mlp" in r.note and "_A_linear_attn" in r.note and "together" in r.note


def test_two_rounds_with_the_same_error_keep_the_old_behaviour(tmp_path):
    stage, ep = two_rounds_on_a(tmp_path, first="KeyError", second="KeyError")
    ep.replies.append(fn_a("return x.other"))
    r = go(stage, 3, _raw(LINE_A, "return x.dims()", "KeyError"), ep)
    assert r.status == "spliced" and len(ep.bodies) == 3


def test_one_earlier_round_is_not_enough_to_stop(tmp_path):
    stage, ep = _stage(tmp_path), Endpoint(fn_a("return x.dims"), fn_a("return x.dims()"))
    go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    r = go(stage, 2, _raw(LINE_A, "return x.dims", "TypeError"), ep)
    assert r.status == "spliced"


def test_a_different_function_is_still_repaired_after_the_stop(tmp_path):
    stage, ep = two_rounds_on_a(tmp_path)
    ep.replies.append("def _B_mlp(model, i, x, ids):\n    return x\n")
    r = go(stage, 3, _raw(LINE_B, "return _A_linear_attn(model, i, x, ids)", "ValueError"), ep)
    assert r.status == "spliced" and r.function == "_B_mlp" and len(ep.bodies) == 3


def test_the_stop_holds_for_every_later_round_on_that_function(tmp_path):
    stage, ep = two_rounds_on_a(tmp_path)
    sent = len(ep.bodies)
    for k in (3, 4):
        r = go(stage, k, _raw(LINE_A, "return x.dims()", "AttributeError"), ep)
        assert r.status == "fallback" and r.requests == 0
    assert len(ep.bodies) == sent


def test_rounds_that_are_not_consecutive_do_not_count_as_two_in_a_row(tmp_path):
    """Round 2 went to the agent loop, so no record exists for it: rounds 1 and 3 on one function are not two rounds in a row."""
    stage, ep = _stage(tmp_path), Endpoint(fn_a("return x.dims"), fn_a("return x.dims()"), fn_a("return x.rank"))
    go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    go(stage, 3, _raw(LINE_A, "return x.dims", "TypeError"), ep)
    r = go(stage, 4, _raw(LINE_A, "return x.dims()", "AttributeError"), ep)
    assert r.status == "spliced" and len(ep.bodies) == 3


def test_two_rounds_on_different_functions_do_not_stop_a_third(tmp_path):
    stage, ep = _stage(tmp_path), Endpoint(fn_a("return x.dims"), "def _B_mlp(model, i, x, ids):\n    return x\n", fn_a("return x.rank"))
    go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    go(stage, 2, _raw(LINE_B, "return _A_linear_attn(model, i, x, ids)", "TypeError"), ep)
    r = go(stage, 3, _raw(LINE_A, "return x.dims", "AttributeError"), ep)
    assert r.status == "spliced"


def test_the_caller_note_names_no_caller_when_none_exists(tmp_path):
    stage = _stage(tmp_path, "import ttnn\n\n\ndef _A_linear_attn(model, i, x, ids):\n    return x.shape\n")
    ep = Endpoint(fn_a("return x.dims"), fn_a("return x.dims()"))
    go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    go(stage, 2, _raw(LINE_A, "return x.dims", "TypeError"), ep)
    r = go(stage, 3, _raw(LINE_A, "return x.dims()", "AttributeError"), ep)
    assert r.status == "fallback" and "no other function in the port calls it" in r.note


def test_a_history_record_holds_only_the_named_fields_and_no_time(tmp_path):
    """The records hold no time, so a replayed run writes the same file."""
    stage, ep = _stage(tmp_path), Endpoint(fn_a("return x.dims"))
    go(stage, 1, _raw(LINE_A, "return x.shape", "KeyError"), ep)
    [rec] = history(stage)
    assert set(rec) == {"round", "function", "error_type", "line", "code", "message", "diff"}


# ---- the supervisor's repair rounds -----------------------------------------------------------------------------------------------------
# The chips, the coder and the model are fakes; the supervisor, the stage machine and the check template are the real ones.

from orchard.supervisor import EXIT_ABORTED                                                    # noqa: E402
from run_fakes import where                                                                      # noqa: E402
from test_thin_repair_wiring import (agent_repair_steps, fenced, rig, scripted, setup, thin_rounds)    # noqa: E402,F401  (rig is a fixture)

RAISES = "def load(model_dir, mesh_device):\n    raise {}('{}')\n"


def capture_agent_repairs(rig):
    """Wrap the rig's script so the text of every stage 2 agent repair request is kept."""
    seen, inner = [], rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "repair"):
            seen.append(json.dumps(request["messages"]))
        return inner(request)
    rig.script = script
    return seen


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_third_round_on_one_function_after_two_changed_errors_goes_to_the_agent_loop_with_a_note(rig):
    setup(rig)
    thin = scripted(rig, lambda n, r: fenced(RAISES.format("ValueError", "second") if n == 1 else RAISES.format("TypeError", "third")))
    seen = capture_agent_repairs(rig)
    assert rig.run() == EXIT_ABORTED
    assert len(thin) == 2                                                                        # no request for round 3
    spliced = thin_rounds(rig)
    assert [d["round"] for d in spliced] == [1, 2]
    recorded = json.loads((rig.run_dir / "stages" / "2" / tr.HISTORY_NAME).read_text())              # the history file the requests were built from
    assert [d["diff_sha256"] for d in spliced] == [hashlib.sha256(r["diff"].encode("utf-8")).hexdigest() for r in recorded[:2]]
    assert "## What earlier thin repair rounds did" in thin[1]["messages"][-1]["content"] and "KeyError" in thin[1]["messages"][-1]["content"]
    handed = [d for d in rig.decisions() if d.get("decision") == "thin repair handed to the coder"]
    assert handed and handed[0]["round"] == 3 and "thin repair stopped for load" in handed[0]["reason"]
    assert len(agent_repair_steps(rig)) == 1 and seen                                               # one step, one request per turn
    assert all("Read `load` together with" in t and "KeyError to ValueError and then to TypeError" in t for t in seen)


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_round_the_history_did_not_stop_gets_no_note(rig):
    setup(rig)
    scripted(rig, lambda n, r: {"role": "assistant", "content": "I cannot do that"})              # two refusals: the agent loop takes round 1
    seen = capture_agent_repairs(rig)
    rig.run()
    assert seen and all("Thin repair changed" not in t for t in seen)


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_the_note_of_one_round_does_not_reach_the_next_round(rig):
    """A note kept from a stopped round would be sent to the agent step of a later round that the history did not stop."""
    from orchard.ledger import Ledger
    from orchard.stages import FULL_PORT_STAGE_2
    from orchard.supervisor import build
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, thin_port=True)
        sup = build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=rig.coder_cls(rig.m), home=rig.home, containers=rig.containers)
        sup._driven_stop = lambda spec: None
        sup._thin_handover = "a note from an earlier round"
        assert sup._thin_repair_round(FULL_PORT_STAGE_2, rig.run_dir / "stages" / "2", 4, {"kind": "numerics", "op": "numerics"}, "raw", False) == (False, None)
        assert sup._thin_handover == ""
