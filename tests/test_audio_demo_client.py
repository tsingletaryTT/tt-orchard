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
