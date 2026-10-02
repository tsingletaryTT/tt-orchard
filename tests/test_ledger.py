"""Ledger tests. The fault-injection test stops the writer at every point in a scripted run
and checks that the resumed run ends in the same state as an uninterrupted one."""
import pytest

from orchard.ledger import (GENESIS, Ledger, LedgerCorrupt, LedgerLocked,
                            file_evidence, replay_state)

SCRIPT = [
    ("run_start", None, {"tt_metal": "abc123"}),
    ("stage_start", 0, {}),
    ("stage_end", 0, {"result": "pass"}),
    ("stage_start", 4, {}),
    ("park", 4, {"board": "0000:01:00.0"}),
    ("restore", 4, {"canary": "ok"}),
    ("stage_end", 4, {"result": "pass"}),
]


def run_script(ledger, events):
    for event, stage, data in events:
        ledger.append(event, stage, **data)


def comparable(ledger):
    """Entries without timestamps, which differ between runs."""
    return [(e["event"], e["stage"], e["data"]) for e in ledger.read()]


def test_round_trip_and_chain(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        run_script(led, SCRIPT)
        entries = led.read()
    assert [e["seq"] for e in entries] == list(range(1, len(SCRIPT) + 1))
    assert entries[0]["prev"] == GENESIS
    assert comparable_from(entries) == SCRIPT


def comparable_from(entries):
    return [(e["event"], e["stage"], e["data"]) for e in entries]


def test_missing_file_is_an_empty_ledger(tmp_path):
    with Ledger(tmp_path / "new" / "ledger.jsonl") as led:
        assert led.read() == []


def test_unknown_event_is_rejected(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        with pytest.raises(ValueError):
            led.append("made_up_event")


def test_measurement_needs_a_label(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        with pytest.raises(ValueError):
            led.append("measurement", 6, tok_s=40)
        led.append("measurement", 6, tok_s=40, label="measured")
        led.append("measurement", 6, tok_s=None, label="TODO")
        assert len(led.read()) == 2


def test_tampering_with_an_earlier_line_is_detected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with Ledger(path) as led:
        run_script(led, SCRIPT)
    path.write_text(path.read_text().replace('"result":"pass"', '"result":"fail"', 1))
    with pytest.raises(LedgerCorrupt):
        Ledger(path)


def test_torn_tail_is_set_aside_not_deleted(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with Ledger(path) as led:
        run_script(led, SCRIPT[:3])
    with open(path, "ab") as f:
        f.write(b'{"seq":4,"prev":"abc","ev')  # a write cut off mid-line
    with Ledger(path) as led:
        assert len(led.read()) == 3
    # The cut-off bytes must be gone from the ledger itself, not just ignored by the reader.
    assert path.read_bytes().endswith(b"\n") and b'"seq":4' not in path.read_bytes()
    sidecars = list(tmp_path.glob("ledger.jsonl.torn-*"))
    assert len(sidecars) == 1
    assert sidecars[0].read_bytes().startswith(b'{"seq":4')


def test_second_opener_is_refused(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with Ledger(path):
        with pytest.raises(LedgerLocked):
            Ledger(path)
    Ledger(path).close()  # the lock is released on close


def test_replay_state_tracks_stage_and_parking():
    entries = [
        {"event": "stage_start", "stage": 4, "data": {}},
        {"event": "park", "stage": 4, "data": {}},
    ]
    state = replay_state(entries)
    assert state["stage"] == 4 and state["stage_status"] == "running" and state["parked"]
    entries += [
        {"event": "restore", "stage": 4, "data": {}},
        {"event": "stage_end", "stage": 4, "data": {"result": "pass"}},
    ]
    state = replay_state(entries)
    assert not state["parked"] and state["completed"] == [4] and state["stage_status"] == "pass"


@pytest.mark.parametrize("torn_tail", [False, True])
@pytest.mark.parametrize("crash_after", range(len(SCRIPT) + 1))
def test_crash_at_every_point_resumes_to_the_same_state(tmp_path, crash_after, torn_tail):
    # Reference: the whole script, never interrupted.
    with Ledger(tmp_path / "ref" / "ledger.jsonl") as ref:
        run_script(ref, SCRIPT)
        want = comparable(ref)
        want_state = replay_state(ref.read())

    # Crash: write part of the script, optionally leave half a line, drop the writer.
    path = tmp_path / "run" / "ledger.jsonl"
    led = Ledger(path)
    run_script(led, SCRIPT[:crash_after])
    led.close()
    if torn_tail:
        with open(path, "ab") as f:
            f.write(b'{"seq":99,"pr')

    # Resume: reopen and finish the script.
    with Ledger(path) as led:
        run_script(led, SCRIPT[crash_after:])
        assert comparable(led) == want
        assert replay_state(led.read()) == want_state


def test_file_evidence_hashes_content(tmp_path):
    f = tmp_path / "e.txt"
    f.write_bytes(b"abc")
    ev = file_evidence(f)
    assert ev["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert ev["path"] == str(f)


# ---- fix round 1: lock hygiene, failed appends, use after close, sidecar names ----

def test_failed_open_releases_the_lock(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("not json\n")
    # Keep the first exception alive: its traceback holds the half-built Ledger, which would
    # otherwise be garbage-collected and hide a leaked lock.
    with pytest.raises(LedgerCorrupt) as first:
        Ledger(path)
    # A retry in the same process must see the corruption again, not a stale lock.
    with pytest.raises(LedgerCorrupt):
        Ledger(path)


def test_failed_append_leaves_the_ledger_writable(tmp_path, monkeypatch):
    import os
    path = tmp_path / "ledger.jsonl"
    real_fsync = os.fsync
    calls = {"n": 0}

    def flaky_fsync(fd):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")  # the line is already in the file when this fires
        real_fsync(fd)

    with Ledger(path) as led:
        led.append("run_start")
        monkeypatch.setattr(os, "fsync", flaky_fsync)
        with pytest.raises(OSError):
            led.append("stage_start", 0)
        led.append("stage_start", 0)  # must not merge with the failed line
        assert [e["seq"] for e in led.read()] == [1, 2]
    Ledger(path).close()  # and the file still verifies on reopen


def test_append_after_close_is_refused(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.close()
    led.close()  # closing twice is harmless
    with pytest.raises(ValueError):
        led.append("run_start")


def test_two_recoveries_in_one_instant_keep_both_sidecars(tmp_path, monkeypatch):
    import orchard.ledger as mod
    monkeypatch.setattr(mod.time, "time_ns", lambda: 1234)
    path = tmp_path / "ledger.jsonl"
    for tail in (b"first-cut", b"second-cut"):
        with open(path, "ab") as f:
            f.write(tail)
        Ledger(path).close()
    contents = sorted(p.read_bytes() for p in tmp_path.glob("ledger.jsonl.torn-*"))
    assert contents == [b"first-cut", b"second-cut"]


# ---- final review: malformed content must raise LedgerCorrupt, never another exception ----

@pytest.mark.parametrize("content,needle", [
    (b'{"seq":1,"prev":"x","event":"run_start","data":"\xff"}\n', "not valid UTF-8"),
    (b"[]\n", "line 1 is not a JSON object"),
    (b"1\n", "line 1 is not a JSON object"),
    (b'"x"\n', "line 1 is not a JSON object"),
    (b"null\n", "line 1 is not a JSON object"),
])
def test_malformed_content_raises_ledger_corrupt_and_frees_the_lock(tmp_path, content, needle):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(content)
    with pytest.raises(LedgerCorrupt, match=needle) as first:  # held so a leaked lock is not hidden
        Ledger(path)
    # A second open must fail for the same reason, not with LedgerLocked.
    with pytest.raises(LedgerCorrupt, match=needle):
        Ledger(path)
    path.write_bytes(b"")
    Ledger(path).close()


def test_non_utf8_error_names_the_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b"\xff\n")
    with pytest.raises(LedgerCorrupt, match="ledger.jsonl"):
        Ledger(path)


@pytest.mark.parametrize("line,needle", [
    ('{"seq":1,"prev":"%s","event":5,"stage":null,"data":{}}', "event must be a string"),
    ('{"seq":1,"prev":"%s","event":"run_start","stage":null,"data":[]}', "data must be an object"),
    ('{"seq":1,"prev":"%s","event":"run_start","data":{}}', "has no stage"),
])
def test_entry_fields_with_the_wrong_type_are_corrupt(tmp_path, line, needle):
    path = tmp_path / "ledger.jsonl"
    path.write_text((line % GENESIS) + "\n")
    with pytest.raises(LedgerCorrupt, match=needle):
        Ledger(path)


def test_boolean_sequence_number_is_corrupt(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"seq":true,"prev":"%s","event":"run_start","stage":null,"data":{}}\n' % GENESIS)
    with pytest.raises(LedgerCorrupt, match="expected sequence 1, found True"):
        Ledger(path)


def test_a_stage_that_passes_twice_is_completed_once():
    def end(stage, result):
        return {"event": "stage_end", "stage": stage, "data": {"result": result}}
    state = replay_state([end(2, "pass"), end(3, "pass"), end(2, "pass")])
    assert state["completed"] == [2, 3]
    # A stage that failed and then passed on retry is completed once too.
    state = replay_state([end(2, "fail"), end(2, "pass")])
    assert state["completed"] == [2] and state["stage_status"] == "pass"
