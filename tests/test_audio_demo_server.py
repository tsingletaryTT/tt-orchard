# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Stable Audio job server (orchard/audio_demo/audio_server.py): one clip at a time, a short queue, a pluggable generator.

Every test runs a real HTTP server on an ephemeral local port with a scripted generator, so nothing here needs a chip or a model.
"""
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchard" / "audio_demo"))
import audio_server as srv  # noqa: E402


def wav_bytes(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * int(8000 * seconds))
    return buf.getvalue()


class Scripted(srv.Generator):
    """A generator the test controls: `gate` holds a clip until the test sets it; `fail` makes a clip raise; `calls` records the requests."""
    model_name = "scripted"
    device_name = "none"

    def __init__(self, gate=None, fail=None, load_gate=None, load_error=None):
        self.gate, self.fail, self.load_gate, self.load_error = gate, fail, load_gate, load_error
        self.calls = []
        self.loads = 0
        self.closed = 0

    def close(self):
        self.closed += 1

    def load(self):
        self.loads += 1
        if self.load_gate is not None:
            self.load_gate.wait(10)
        if self.load_error:
            raise RuntimeError(self.load_error)

    def generate(self, req, on_stage):
        self.calls.append(req)
        on_stage("sampling")
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.fail:
            raise RuntimeError(self.fail)
        return srv.Clip(wav_bytes(req.seconds), "wav", float(req.seconds), {"sampling_s": 0.5})


class Live:
    """A running server on an ephemeral port, with small HTTP helpers that return (status, parsed body) and never raise on an HTTP error."""
    def __init__(self, gen, **kw):
        self.js = srv.JobServer(gen, **kw)
        self.http = srv.make_http_server(self.js, "127.0.0.1", 0)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"
        self.t = threading.Thread(target=lambda: self.http.serve_forever(0.05), daemon=True)
        self.t.start()
        self.js.start()

    def call(self, method, path, body=None, raw_body=None, raw=False):
        data = raw_body if raw_body is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                payload = r.read()
                return r.status, (payload if raw else json.loads(payload))
        except urllib.error.HTTPError as e:
            payload = e.read()
            return e.code, (payload if raw else json.loads(payload))

    def wait_state(self, job_id, states, timeout=10):
        end = time.monotonic() + timeout
        st = None
        while time.monotonic() < end:
            _, st = self.call("GET", f"/v1/audio/jobs/{job_id}")
            if st.get("status") in states:
                return st
            time.sleep(0.01)
        raise AssertionError(f"job {job_id} never reached {states}; last {st}")

    def wait_ready(self):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            if self.call("GET", "/v1/health")[1]["status"] != "loading":
                return
            time.sleep(0.01)
        raise AssertionError("the server stayed in loading")

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        self.js.stop()


@pytest.fixture
def live():
    made = []

    def make(gen=None, **kw):
        s = Live(gen or Scripted(), **kw)
        made.append(s)
        s.wait_ready()
        return s
    yield make
    for s in made:
        s.close()


JOB = {"prompt": "A warm acoustic guitar melody with soft percussion, 100 BPM", "duration": 10, "steps": 8, "cfg_scale": 1.0, "seed": 42}


def test_a_clip_is_made_and_the_settings_come_back(live):
    s = live()
    code, body = s.call("POST", "/v1/audio/jobs", JOB)
    assert code == 202
    st = s.wait_state(body["id"], {"done"})
    assert st["settings"] == JOB
    assert st["stats"]["audio_s"] == 10.0 and st["stats"]["audio_format"] == "wav" and st["stats"]["stages"] == {"sampling_s": 0.5}
    assert st["stats"]["total_s"] >= 0
    code, audio = s.call("GET", f"/v1/audio/jobs/{body['id']}/audio", raw=True)
    assert code == 200 and audio[:4] == b"RIFF"


def test_defaults_come_from_the_generator_and_the_seed_is_chosen_and_echoed(live):
    s = live()
    _, body = s.call("POST", "/v1/audio/jobs", {"prompt": "x"})
    st = s.wait_state(body["id"], {"done"})
    assert st["settings"]["steps"] == 8 and st["settings"]["cfg_scale"] == 1.0 and st["settings"]["duration"] == 10
    assert isinstance(st["settings"]["seed"], int) and st["settings"]["seed"] >= 0


def test_health_says_what_the_server_is_and_what_it_accepts(live):
    s = live()
    h = s.call("GET", "/v1/health")[1]
    assert h["status"] == "ok" and h["model"] == "scripted" and h["device_name"] == "none"
    assert h["queue_depth"] == 0 and h["queue_max"] == 2
    assert h["limits"] == {"min_seconds": 1, "max_seconds": 120, "max_steps": 100, "min_cfg_scale": 1.0, "max_cfg_scale": 20.0}


@pytest.mark.parametrize("patch", [
    {"prompt": ""}, {"prompt": "   "}, {"prompt": 5}, {"prompt": "x" * 3000},
    {"duration": 0}, {"duration": 121}, {"duration": -1}, {"duration": "10"}, {"duration": float("nan")}, {"duration": True},
    {"steps": 0}, {"steps": 101}, {"steps": 1.5}, {"steps": True},
    {"cfg_scale": 0.5}, {"cfg_scale": 21}, {"cfg_scale": "1"},
    {"seed": -1}, {"seed": 1.5}, {"seed": True}, {"seed": 2**40},
    {"surprise": 1}])
def test_bad_input_is_a_400_and_never_reaches_the_generator(live, patch):
    gen = Scripted()
    s = live(gen)
    raw = json.dumps({**JOB, **patch}).encode()          # json.dumps writes NaN, which a strict reader must refuse
    code, err = s.call("POST", "/v1/audio/jobs", raw_body=raw)
    assert code == 400 and err["detail"]
    time.sleep(0.05)
    assert gen.calls == []


def test_a_body_that_is_not_json_or_too_large_is_refused(live):
    s = live()
    assert s.call("POST", "/v1/audio/jobs", raw_body=b"{not json")[0] == 400
    assert s.call("POST", "/v1/audio/jobs", raw_body=b"[1]")[0] == 400
    assert s.call("POST", "/v1/audio/jobs", raw_body=b"{" + b" " * (srv.MAX_BODY_BYTES + 10) + b"}")[0] == 413


def test_zero_waiting_slots_still_allow_one_running_clip(live):
    gate = threading.Event()
    s = live(Scripted(gate=gate), max_waiting=0)
    first = s.call("POST", "/v1/audio/jobs", JOB)
    assert first[0] == 202                                   # idle: the one clip is accepted
    second = s.call("POST", "/v1/audio/jobs", JOB)
    assert second[0] == 429                                  # busy, and nothing may wait
    gate.set()
    s.wait_state(first[1]["id"], {"done"})
    assert s.call("POST", "/v1/audio/jobs", JOB)[0] == 202   # idle again


def test_a_job_that_has_not_been_picked_up_yet_counts_against_the_capacity(live):
    """With one waiting slot, two quick posts are accepted whether or not the worker has dequeued the first; a third is refused."""
    gate = threading.Event()
    s = live(Scripted(gate=gate), max_waiting=1)
    codes = [s.call("POST", "/v1/audio/jobs", JOB)[0] for _ in range(3)]
    gate.set()
    assert codes == [202, 202, 429]


def test_negative_waiting_slots_are_refused():
    with pytest.raises(ValueError):
        srv.JobServer(Scripted(), max_waiting=-1)
    with pytest.raises(SystemExit):
        srv.parse_args(["--generator", "fake", "--max-waiting", "-1"])


def test_stop_closes_the_generator(live):
    gen = Scripted()
    s = live(gen)
    s.js.stop()
    assert gen.closed == 1


def test_a_negative_content_length_is_refused_and_does_not_hang_the_server(live):
    import http.client
    s = live()
    conn = http.client.HTTPConnection("127.0.0.1", s.http.server_address[1], timeout=5)
    conn.putrequest("POST", "/v1/audio/jobs")
    conn.putheader("Content-Length", "-5")
    conn.endheaders()
    assert conn.getresponse().status == 400                  # a read of -5 bytes would wait for the client to close the connection
    conn.close()


def test_nan_and_infinity_are_not_json(live):
    s = live()
    for token in (b"NaN", b"Infinity", b"-Infinity"):
        code, err = s.call("POST", "/v1/audio/jobs", raw_body=b'{"prompt": "x", "duration": ' + token + b"}")
        assert code == 400 and "not valid JSON" in err["detail"], token


def test_full_queue_is_a_429_and_the_waiting_jobs_still_finish(live):
    gate = threading.Event()
    s = live(Scripted(gate=gate), max_waiting=1)
    first = s.call("POST", "/v1/audio/jobs", JOB)[1]["id"]
    s.wait_state(first, {"running"})
    second = s.call("POST", "/v1/audio/jobs", JOB)
    assert second[0] == 202
    assert s.call("GET", f"/v1/audio/jobs/{second[1]['id']}")[1]["queue_position"] == 1
    third = s.call("POST", "/v1/audio/jobs", JOB)
    assert third[0] == 429 and "queue is full" in third[1]["detail"]
    assert s.call("GET", "/v1/health")[1]["queue_depth"] == 1
    gate.set()
    s.wait_state(first, {"done"})
    s.wait_state(second[1]["id"], {"done"})


def test_one_clip_runs_at_a_time(live):
    gate = threading.Event()
    gen = Scripted(gate=gate)
    s = live(gen, max_waiting=3)
    ids = [s.call("POST", "/v1/audio/jobs", {**JOB, "seed": i})[1]["id"] for i in range(3)]
    s.wait_state(ids[0], {"running"})
    time.sleep(0.1)
    assert len(gen.calls) == 1                                  # the others wait; the generator is never entered twice at once
    gate.set()
    for i in ids:
        s.wait_state(i, {"done"})
    assert [c.seed for c in gen.calls] == [0, 1, 2]


def test_a_generator_error_ends_that_job_only(live):
    gen = Scripted(fail="the chip hung")
    s = live(gen)
    bad = s.call("POST", "/v1/audio/jobs", JOB)[1]["id"]
    st = s.wait_state(bad, {"error"})
    assert "the chip hung" in st["error"]
    gen.fail = None
    good = s.call("POST", "/v1/audio/jobs", JOB)[1]["id"]
    s.wait_state(good, {"done"})
    assert s.call("GET", f"/v1/audio/jobs/{bad}/audio", raw=True)[0] == 409


def test_a_queued_job_is_cancelled_and_never_runs(live):
    gate = threading.Event()
    gen = Scripted(gate=gate)
    s = live(gen, max_waiting=2)
    a = s.call("POST", "/v1/audio/jobs", JOB)[1]["id"]
    s.wait_state(a, {"running"})
    b = s.call("POST", "/v1/audio/jobs", {**JOB, "seed": 7})[1]["id"]
    assert s.call("DELETE", f"/v1/audio/jobs/{b}")[1]["status"] == "cancelled"
    gate.set()
    s.wait_state(a, {"done"})
    time.sleep(0.1)
    assert [c.seed for c in gen.calls] == [42]


def test_a_running_job_cannot_be_stopped_and_its_result_is_dropped(live):
    gate = threading.Event()
    s = live(Scripted(gate=gate))
    a = s.call("POST", "/v1/audio/jobs", JOB)[1]["id"]
    s.wait_state(a, {"running"})
    code, body = s.call("DELETE", f"/v1/audio/jobs/{a}")
    assert code == 200 and body["cancel_requested"] is True and "keeps running" in body["detail"]
    gate.set()
    s.wait_state(a, {"cancelled"})
    assert s.call("GET", f"/v1/audio/jobs/{a}/audio", raw=True)[0] == 409


def test_not_ready_while_loading_and_after_a_load_failure():
    gate = threading.Event()
    s = Live(Scripted(load_gate=gate))
    try:
        assert s.call("GET", "/v1/health")[1]["status"] == "loading"
        code, err = s.call("POST", "/v1/audio/jobs", JOB)
        assert code == 503 and "loading" in err["detail"]
        gate.set()
        s.wait_ready()
        assert s.call("POST", "/v1/audio/jobs", JOB)[0] == 202
    finally:
        s.close()
    bad = Live(Scripted(load_error="no chip"))
    try:
        bad.wait_ready()
        h = bad.call("GET", "/v1/health")[1]
        assert h["status"] == "error" and "no chip" in h["error"]
        assert bad.call("POST", "/v1/audio/jobs", JOB)[0] == 503
    finally:
        bad.close()


def test_the_model_is_loaded_once_for_many_clips(live):
    gen = Scripted()
    s = live(gen)
    for i in range(3):
        s.wait_state(s.call("POST", "/v1/audio/jobs", {**JOB, "seed": i})[1]["id"], {"done"})
    assert gen.loads == 1


def test_unknown_job_and_bad_paths(live):
    s = live()
    assert s.call("GET", "/v1/audio/jobs/" + "0" * 32)[0] == 404
    assert s.call("GET", "/v1/audio/jobs/../../etc/passwd")[0] == 404
    assert s.call("GET", "/nope")[0] == 404
    assert s.call("DELETE", "/v1/audio/jobs/" + "0" * 32)[0] == 404


def test_old_finished_jobs_are_forgotten(live):
    s = live(keep=2, max_waiting=8)
    ids = [s.call("POST", "/v1/audio/jobs", {**JOB, "seed": i})[1]["id"] for i in range(5)]
    s.wait_state(ids[-1], {"done"})
    time.sleep(0.1)
    codes = [s.call("GET", f"/v1/audio/jobs/{i}")[0] for i in ids]
    assert codes.count(200) == 2 and codes[-1] == 200          # the two newest finished jobs are kept


def test_the_default_bind_is_loopback_and_off_the_reserved_ports():
    args = srv.parse_args(["--generator", "fake"])
    assert args.host == "127.0.0.1" and args.ui_host == "127.0.0.1"
    for port in (args.port, args.ui_port):
        assert port not in (8000, 8100, 8101, 8102, 8103, 8780)


def test_fake_generator_makes_a_playable_clip_and_says_it_is_fake():
    g = srv.FakeGenerator(delay_s=0)
    g.load()
    clip = g.generate(srv.JobRequest("p", 3.0, 8, 1.0, 1), lambda s: None)
    with wave.open(io.BytesIO(clip.audio)) as w:
        assert abs(w.getnframes() / w.getframerate() - 3.0) < 0.01
    assert clip.audio_format == "wav" and abs(clip.audio_s - 3.0) < 0.01
    assert "fake" in g.model_name and g.device_name == "no chip"
