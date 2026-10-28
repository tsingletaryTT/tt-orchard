# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The client (audio_client.py) and the page's generation function (`run_generation`) against a real local server with a scripted generator."""
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "orchard" / "audio_demo"))
sys.path.insert(0, str(HERE))
import audio_client as cl  # noqa: E402
import audio_server as srv  # noqa: E402
from test_audio_demo_server import JOB, Live, Scripted, wav_bytes  # noqa: E402,F401


@pytest.fixture
def server():
    made = []

    def make(gen=None, **kw):
        s = Live(gen or Scripted(), **kw)
        made.append(s)
        s.wait_ready()
        return s, cl.AudioClient(s.base, timeout=5)
    yield make
    for s in made:
        s.close()


class ShortClip(Scripted):
    """A generator whose clip ends before the length asked for."""
    def generate(self, req, on_stage):
        clip = super().generate(req, on_stage)
        return srv.Clip(wav_bytes(req.seconds / 2), "wav", req.seconds / 2, clip.stages)


def test_generate_returns_the_clip_the_settings_and_the_timings(server):
    s, c = server()
    r = c.generate("a tune", 10, seed=5, steps=4, cfg_scale=2.0, poll_s=0.01)
    assert r.audio[:4] == b"RIFF" and r.audio_format == "wav"
    assert r.settings == {"prompt": "a tune", "duration": 10, "steps": 4, "cfg_scale": 2.0, "seed": 5}
    assert r.stats["audio_s"] == 10.0 and r.stats["stages"] == {"sampling_s": 0.5}


def test_health_is_read_and_a_dead_port_is_server_down():
    gen = Scripted()
    s = Live(gen)
    s.wait_ready()
    c = cl.AudioClient(s.base)
    assert c.health()["status"] == "ok"
    s.close()
    with pytest.raises(cl.ServerDown):
        c.health()


def test_a_full_queue_is_server_busy(server):
    gate = threading.Event()
    s, c = server(Scripted(gate=gate), max_waiting=1)
    c.submit("a", 10)
    s.wait_state(next(iter(s.js.jobs)), {"running"})
    c.submit("b", 10)
    with pytest.raises(cl.ServerBusy):
        c.submit("c", 10)
    gate.set()


def test_a_loading_server_is_not_ready():
    gate = threading.Event()
    s = Live(Scripted(load_gate=gate))
    try:
        with pytest.raises(cl.ServerNotReady):
            cl.AudioClient(s.base).submit("a", 10)
    finally:
        gate.set()
        s.close()


@pytest.mark.parametrize("kw", [{"prompt": ""}, {"prompt": " "}, {"seconds": 0}, {"seconds": 500}, {"seconds": float("nan")}, {"seconds": True},
                                {"steps": 0}, {"steps": 1000}, {"steps": 2.5}, {"cfg_scale": 0.1}, {"cfg_scale": 99}, {"seed": -3}, {"seed": 1.5}])
def test_bad_input_is_refused_before_a_request_is_made(kw):
    args = {"prompt": "a", "seconds": 10, "steps": 8, "cfg_scale": 1.0, "seed": 1, **kw}
    c = cl.AudioClient("http://127.0.0.1:1")                      # nothing listens: a request would raise ServerDown instead
    with pytest.raises(ValueError):
        c.submit(args["prompt"], args["seconds"], seed=args["seed"], steps=args["steps"], cfg_scale=args["cfg_scale"])


def test_a_job_that_errors_raises_with_the_server_message(server):
    s, c = server(Scripted(fail="the chip hung"))
    with pytest.raises(cl.AudioServerError, match="the chip hung"):
        c.generate("a", 10, poll_s=0.01)


def test_progress_is_reported_while_waiting(server):
    gate = threading.Event()
    s, c = server(Scripted(gate=gate))
    seen = []

    def on_progress(state, stage, position):
        seen.append((state, stage))
        if state == "running":
            gate.set()
    c.generate("a", 10, poll_s=0.01, on_progress=on_progress)
    assert ("running", "sampling") in seen


def test_a_timed_out_wait_cancels_the_job(server):
    gate = threading.Event()
    s, c = server(Scripted(gate=gate), max_waiting=2)
    c.submit("hog", 10)
    s.wait_state(next(iter(s.js.jobs)), {"running"})
    with pytest.raises(cl.AudioServerError, match="Timed out"):
        c.generate("waiting", 10, poll_s=0.01, timeout_s=0.1)
    waiting = [j for j in s.js.jobs.values() if j.req.prompt == "waiting"][0]
    assert waiting.status == "cancelled"                          # the abandoned job does not keep its place in the queue
    gate.set()


# ---- run_generation: what the page shows --------------------------------------------------------------------

def test_run_generation_done(server, tmp_path):
    s, c = server()
    out = cl.run_generation(c, "A warm acoustic guitar melody with soft percussion, 100 BPM", 10, 8, 1.0, 42, poll_s=0.01, out_dir=tmp_path)
    assert out.ok and out.audio_path.endswith(".wav") and Path(out.audio_path).read_bytes()[:4] == b"RIFF"
    assert out.seed == 42
    assert "Done: 10.0 s of audio, seed 42" in out.status
    for line in ("A warm acoustic guitar melody", "Seconds asked for: 10", "Steps: 8", "Guidance scale: 1", "Seed: 42", "Job `"):
        assert line in out.details
    assert "sampling 0.5 s" in out.details


def test_run_generation_blank_seed_is_chosen_and_shown(server, tmp_path):
    s, c = server()
    out = cl.run_generation(c, "x", 10, 8, 1.0, None, poll_s=0.01, out_dir=tmp_path)
    assert out.ok and isinstance(out.seed, int) and f"Seed: {out.seed}" in out.details
    again = cl.run_generation(c, "x", 10, 8, 1.0, "", poll_s=0.01, out_dir=tmp_path)
    assert again.ok and isinstance(again.seed, int)


@pytest.mark.parametrize("field,value", [
    ("seed", "abc"), ("seed", float("inf")), ("seed", float("nan")), ("seed", 1.5), ("seed", -1), ("seed", 2**70),
    ("steps", "abc"), ("steps", float("inf")), ("steps", 2.5), ("steps", 0),
    ("cfg_scale", "abc"), ("cfg_scale", float("inf")), ("cfg_scale", float("nan")),
    ("seconds", "abc"), ("seconds", float("inf")), ("seconds", float("nan")), ("seconds", None)])
def test_run_generation_turns_bad_control_values_into_a_status_and_never_raises(field, value, tmp_path):
    """The page gets whatever its boxes hold. A value that cannot be converted or is not finite is a message, and nothing is sent."""
    args = {"seconds": 10, "steps": 8, "cfg_scale": 1.0, "seed": 1, field: value}
    out = cl.run_generation(cl.AudioClient("http://127.0.0.1:1"), "x", args["seconds"], args["steps"], args["cfg_scale"], args["seed"], out_dir=tmp_path)
    assert not out.ok and out.audio_path is None
    assert "not reachable" not in out.status, "the value should have been refused before a request was made"
    assert "unexpected" not in out.status.lower() and "must be" in out.status    # a message about the control, not a crash caught at the last step


def test_run_generation_never_raises_for_an_unexpected_failure(tmp_path):
    class Broken(cl.AudioClient):
        def generate(self, *a, **k):
            raise KeyError("id")
    out = cl.run_generation(Broken("http://127.0.0.1:1"), "x", 10, 8, 1.0, 1, out_dir=tmp_path)
    assert not out.ok and "unexpected" in out.status.lower() and "KeyError" in out.status


def raw_server(reply: bytes):
    """A server that answers every request with `reply` as it is and closes the connection."""
    import socketserver

    class H(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(65536)
            self.request.sendall(reply)
    srv_ = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv_.daemon_threads = True
    threading.Thread(target=srv_.serve_forever, daemon=True).start()
    return srv_


def test_a_reply_cut_short_is_server_down_and_a_reply_that_is_not_json_is_an_error():
    cut = raw_server(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n{\"status\": ")
    junk = raw_server(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 6\r\n\r\n<html>")
    wrong = raw_server(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n[]")
    try:
        with pytest.raises(cl.ServerDown):
            cl.AudioClient(f"http://127.0.0.1:{cut.server_address[1]}", timeout=3).health()
        with pytest.raises(cl.AudioServerError, match="not a JSON"):
            cl.AudioClient(f"http://127.0.0.1:{junk.server_address[1]}", timeout=3).health()
        with pytest.raises(cl.AudioServerError, match="unexpected"):
            cl.AudioClient(f"http://127.0.0.1:{wrong.server_address[1]}", timeout=3).health()
        # through run_generation, each is a status
        for bad in (cut, junk, wrong):
            out = cl.run_generation(cl.AudioClient(f"http://127.0.0.1:{bad.server_address[1]}", timeout=3), "x", 10, 8, 1.0, 1)
            assert not out.ok and out.status
    finally:
        for x in (cut, junk, wrong):
            x.shutdown()
            x.server_close()


def test_an_error_reply_with_a_cut_short_body_keeps_its_meaning():
    """A 429 whose body stops halfway is still a busy server, and a 500 with a cut body is an error, not an exception from the HTTP library."""
    busy = raw_server(b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n{\"detail\": ")
    broken = raw_server(b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n{")
    try:
        with pytest.raises(cl.ServerBusy):
            cl.AudioClient(f"http://127.0.0.1:{busy.server_address[1]}", timeout=3).health()
        with pytest.raises(cl.AudioServerError, match="500"):
            cl.AudioClient(f"http://127.0.0.1:{broken.server_address[1]}", timeout=3).health()
    finally:
        for x in (busy, broken):
            x.shutdown()
            x.server_close()


@pytest.mark.parametrize("field", ["seconds", "cfg_scale", "seed", "steps"])
def test_a_huge_whole_number_is_a_message_on_the_client_too(field, tmp_path):
    args = {"seconds": 10, "steps": 8, "cfg_scale": 1.0, "seed": 1, field: 10**400}
    out = cl.run_generation(cl.AudioClient("http://127.0.0.1:1"), "x", args["seconds"], args["steps"], args["cfg_scale"], args["seed"], out_dir=tmp_path)
    assert not out.ok and "must be" in out.status and "unexpected" not in out.status.lower()
    with pytest.raises(ValueError):
        cl.check_settings("x", args["seconds"], args["seed"], args["steps"], args["cfg_scale"])


def test_pruning_touches_only_the_clips_this_page_made(server, tmp_path):
    (tmp_path / "settings.json").write_text("{}")
    (tmp_path / "recording.wav").write_bytes(b"x")
    (tmp_path / "clip-notes.txt").write_text("mine")
    s, c = server()
    for i in range(5):
        cl.run_generation(c, "x", 1, 8, 1.0, i, poll_s=0.01, out_dir=tmp_path, keep=2)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"settings.json", "recording.wav", "clip-notes.txt"} <= names
    assert len([n for n in names if n.startswith("clip-seed")]) == 2


def test_a_job_reply_without_an_id_is_an_error():
    no_id = raw_server(b"HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}")
    try:
        with pytest.raises(cl.AudioServerError, match="no job id"):
            cl.AudioClient(f"http://127.0.0.1:{no_id.server_address[1]}", timeout=3).submit("x", 10)
    finally:
        no_id.shutdown()
        no_id.server_close()


def test_saved_clips_are_kept_to_a_bounded_number(server, tmp_path):
    """Each clip the page saves is a file on the local disk. Only the newest `keep` stay."""
    s, c = server()
    paths = [cl.run_generation(c, "x", 1, 8, 1.0, i, poll_s=0.01, out_dir=tmp_path, keep=3).audio_path for i in range(6)]
    left = sorted(p.name for p in tmp_path.iterdir())
    assert len(left) == 3 and all(Path(p).name in left for p in paths[-3:])


def test_without_an_output_directory_the_clips_share_one_managed_directory(server):
    s, c = server()
    a = cl.run_generation(c, "x", 1, 8, 1.0, 1, poll_s=0.01)
    b = cl.run_generation(c, "x", 1, 8, 1.0, 2, poll_s=0.01)
    assert Path(a.audio_path).parent == Path(b.audio_path).parent          # not a new temporary directory for every request


def test_run_generation_says_when_the_clip_ended_early(server, tmp_path):
    s, c = server(ShortClip())
    out = cl.run_generation(c, "x", 10, 8, 1.0, 1, poll_s=0.01, out_dir=tmp_path)
    assert out.ok and "ended at 5.0 s, before the 10 s asked for" in out.status


def test_run_generation_busy_down_notready_and_bad_input_are_plain_statuses(server, tmp_path):
    gate = threading.Event()
    s, c = server(Scripted(gate=gate), max_waiting=1)
    c.submit("a", 10)
    s.wait_state(next(iter(s.js.jobs)), {"running"})
    c.submit("b", 10)
    busy = cl.run_generation(c, "c", 10, 8, 1.0, 1, poll_s=0.01, out_dir=tmp_path)
    assert not busy.ok and busy.audio_path is None and "busy" in busy.status.lower() and "queue is full" in busy.status
    gate.set()
    bad = cl.run_generation(c, "", 10, 8, 1.0, 1, out_dir=tmp_path)
    assert not bad.ok and "prompt" in bad.status
    dead = cl.run_generation(cl.AudioClient("http://127.0.0.1:1"), "x", 10, 8, 1.0, 1, out_dir=tmp_path)
    assert not dead.ok and "not reachable" in dead.status
    g = threading.Event()
    loading = Live(Scripted(load_gate=g))
    try:
        r = cl.run_generation(cl.AudioClient(loading.base), "x", 10, 8, 1.0, 1, out_dir=tmp_path)
        assert not r.ok and "loading" in r.status
    finally:
        g.set()
        loading.close()


def test_run_generation_reports_a_failed_job(server, tmp_path):
    s, c = server(Scripted(fail="the chip hung"))
    out = cl.run_generation(c, "x", 10, 8, 1.0, 1, poll_s=0.01, out_dir=tmp_path)
    assert not out.ok and out.audio_path is None and "the chip hung" in out.status


def test_server_status_line(server):
    s, c = server()
    line = cl.server_status(c)
    assert "ready" in line and "scripted" in line and "0 waiting" in line
    assert "not reachable" in cl.server_status(cl.AudioClient("http://127.0.0.1:1"))


def test_random_seed_is_a_valid_seed():
    seen = {cl.random_seed() for _ in range(50)}
    assert len(seen) > 1 and all(isinstance(x, int) and 0 <= x < 2**31 for x in seen)


def test_examples_include_the_benchmark_prompt_and_all_fit_the_limits():
    prompts = [e[0] for e in cl.EXAMPLES]
    assert "A warm acoustic guitar melody with soft percussion, 100 BPM" in prompts
    for prompt, seconds, seed in cl.EXAMPLES:
        assert prompt.strip() and cl.MIN_SECONDS <= seconds <= cl.MAX_SECONDS and isinstance(seed, int) and seed >= 0
