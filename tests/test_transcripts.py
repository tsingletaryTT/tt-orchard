"""The qwen-code transcript reader, the committed signatures and what the detectors find in them."""
import json
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from orchard.transcripts import iso, load_signature, parse_ts, qwen_events, write_signature
from orchard.watchdog import KINDS, Event, IdenticalResponses, replay, transcript_detectors

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
SESSION = "abcdef12-0000-0000-0000-000000000000"

LOOP_FINDINGS = [
    ("thinking_without_action", "2026-10-01T02:05:05.444Z"),
    ("thinking_without_action", "2026-10-01T02:16:01.025Z"),
    ("identical_responses", "2026-10-01T02:26:58.738Z"),
    ("thinking_without_action", "2026-10-01T02:26:58.738Z"),
    ("thinking_without_action", "2026-10-01T02:37:57.977Z"),
    ("thinking_without_action", "2026-10-01T02:48:58.884Z"),
    ("identical_responses", "2026-10-01T03:46:56.001Z"),
]


def api(ts, i, o, think, text="", sub=None):
    ev = {"event.name": "qwen-code.api_response", "input_token_count": i, "output_token_count": o,
          "thoughts_token_count": think, "duration_ms": 1500, "response_text": text}
    if sub:
        ev["subagent_name"] = sub
    return {"type": "system", "subtype": "ui_telemetry", "timestamp": ts, "sessionId": SESSION,
            "systemPayload": {"uiEvent": ev}}


def assistant(ts, *calls):
    parts = [{"text": "private reasoning", "thought": True}]
    parts += [{"functionCall": {"name": n, "args": a}} for n, a in calls]
    return {"type": "assistant", "timestamp": ts, "sessionId": SESSION,
            "message": {"role": "model", "parts": parts}}


def tool_result(ts, name, response):
    return {"type": "tool_result", "timestamp": ts, "sessionId": SESSION,
            "message": {"role": "user", "parts": [{"functionResponse": {"id": "x", "name": name,
                                                                         "response": response}}]}}


def lines(*records):
    return [json.dumps(r) for r in records]


def test_a_response_learns_whether_a_tool_call_followed():
    evs = list(qwen_events(lines(
        api("2026-10-01T00:00:00.000Z", 100, 10, 5, text="Looking."),
        assistant("2026-10-01T00:00:00.010Z", ("read_file", {"path": "a"})),
        tool_result("2026-10-01T00:00:00.020Z", "read_file", {"output": "x"}),
        api("2026-10-01T00:00:01.000Z", 200, 20, 30000),
    )))
    assert [e.kind for e in evs] == ["response", "tool_call", "tool_result", "response"]
    assert evs[0].had_tool_call is True and evs[3].had_tool_call is False
    assert evs[0].agent == "qwen:abcdef12" and evs[0].duration_s == 1.5
    assert evs[1].tool == "read_file" and len(evs[1].args_hash) == 64
    assert evs[3].text_hash is None and evs[3].thinking_tokens == 30000


def test_subagent_responses_and_other_records_are_skipped():
    evs = list(qwen_events(lines(
        api("2026-10-01T00:00:00.000Z", 100, 10, 5, sub="managed-auto-memory-dreamer"),
        {"type": "system", "subtype": "attribution_snapshot", "timestamp": "2026-10-01T00:00:00.000Z"},
        {"type": "user", "timestamp": "2026-10-01T00:00:00.000Z", "message": {"parts": []}},
    )))
    assert evs == []


def test_a_torn_last_line_is_ignored():
    good = lines(api("2026-10-01T00:00:00.000Z", 1, 1, 1))
    assert len(list(qwen_events(good + ['{"type": "assis']))) == 1


def test_times_round_trip_to_the_millisecond():
    assert iso(parse_ts("2026-10-01T02:26:58.738Z")) == "2026-10-01T02:26:58.738Z"


def found(path, detectors=None):
    return [(f.detector, iso(f.ts)) for f in replay(load_signature(path),
                                                    detectors or transcript_detectors())]


def test_the_loop_signature_has_the_recorded_shape():
    evs = load_signature(FIX / "qwen_loop_signature.jsonl")
    kinds = [e.kind for e in evs]
    assert (kinds.count("response"), kinds.count("tool_call"), kinds.count("tool_result")) == (74, 76, 76)
    loop = [e for e in evs if e.kind == "response" and e.input_tokens == 145299]
    assert len(loop) == 5 and {(e.output_tokens, e.thinking_tokens) for e in loop} == {(33348, 27939)}
    assert all(640 < e.duration_s < 660 for e in loop)


def test_the_detectors_fire_on_both_repeats_in_the_loop_chat():
    assert found(FIX / "qwen_loop_signature.jsonl") == LOOP_FINDINGS


def test_the_detectors_stay_quiet_on_a_chat_that_worked():
    assert found(FIX / "qwen_quiet_signature.jsonl") == []


def test_n_of_two_would_also_fire_on_a_two_call_retry():
    # The loop chat also has two identical calls at 01:47:42 and 01:50:03. N=3 ignores them.
    hits = found(FIX / "qwen_loop_signature.jsonl", [IdenticalResponses(n=2)])
    assert ("identical_responses", "2026-10-01T01:50:03.988Z") in hits


HEX64 = re.compile(r"[0-9a-f]{64}")
ALLOWED = {f.name for f in fields(Event)}


def test_the_fixtures_carry_no_message_text():
    for name in ("qwen_loop_signature.jsonl", "qwen_quiet_signature.jsonl"):
        for line in (FIX / name).read_text().splitlines():
            d = json.loads(line)
            assert set(d) <= ALLOWED, d
            for k, v in d.items():
                if not isinstance(v, str):
                    continue
                ok = ((k in ("text_hash", "args_hash", "output_hash") and HEX64.fullmatch(v))
                      or (k == "kind" and v in KINDS)
                      or (k == "agent" and re.fullmatch(r"qwen:[0-9a-f]{8}", v))
                      or (k == "tool" and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", v)))
                assert ok, (name, k, v)


def test_the_extract_script_writes_a_loadable_signature(tmp_path):
    src = tmp_path / "chat.jsonl"
    src.write_text("\n".join(lines(api("2026-10-01T00:00:00.000Z", 1, 2, 3, text="Hello there"),
                                   assistant("2026-10-01T00:00:00.001Z", ("glob", {"p": "*"})))) + "\n")
    out = tmp_path / "sig.jsonl"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "extract_qwen_signature.py"),
                        str(src), str(out)], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "2"
    evs = load_signature(out)
    assert [e.kind for e in evs] == ["response", "tool_call"] and evs[0].had_tool_call
    text = out.read_text()
    assert "Hello" not in text and "private reasoning" not in text and '"*"' not in text


def equality_structure(events):
    """Each hash replaced by the index of its first occurrence: what the detectors compare."""
    first = {}
    out = []
    for e in events:
        row = []
        for k in ("text_hash", "args_hash", "output_hash"):
            h = getattr(e, k)
            row.append(None if h is None else first.setdefault(h, len(first)))
        out.append(tuple(row))
    return out


def test_two_extractions_differ_in_hashes_and_agree_on_equality(tmp_path):
    src = tmp_path / "chat.jsonl"
    src.write_text("\n".join(lines(
        api("2026-10-01T00:00:00.000Z", 1, 2, 3, text="same"),
        assistant("2026-10-01T00:00:00.001Z", ("read_file", {"path": "a"})),
        tool_result("2026-10-01T00:00:00.002Z", "read_file", {"output": "x"}),
        api("2026-10-01T00:00:01.000Z", 1, 2, 3, text="same"),
        assistant("2026-10-01T00:00:01.001Z", ("read_file", {"path": "a"})),
        tool_result("2026-10-01T00:00:01.002Z", "read_file", {"output": "x"}))) + "\n")
    script = str(REPO / "scripts" / "extract_qwen_signature.py")
    outs = []
    for name in ("one.jsonl", "two.jsonl"):
        subprocess.run([sys.executable, script, str(src), str(tmp_path / name)], check=True,
                       capture_output=True)
        outs.append(load_signature(tmp_path / name))
    hashes = [{getattr(e, k) for e in evs for k in ("text_hash", "args_hash", "output_hash")} - {None}
              for evs in outs]
    assert hashes[0].isdisjoint(hashes[1])
    assert equality_structure(outs[0]) == equality_structure(outs[1])
    # Within one file, repeated content still hashes equal.
    assert outs[0][0].text_hash == outs[0][3].text_hash


def test_write_then_load_round_trips(tmp_path):
    evs = [Event(ts=1.5, agent="qwen:abcdef12", kind="response", input_tokens=3)]
    assert write_signature(evs, tmp_path / "s.jsonl") == 1
    assert load_signature(tmp_path / "s.jsonl") == evs
