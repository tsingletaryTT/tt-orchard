"""The vendored MiniMax-Music3 engine (pkgsrc/music3/code/.../server/engine.py) with a fake generator and no chip (issue 306).

The engine is the upstream file with one local patch, recorded in pkgsrc/music3/PROVENANCE.md. The two behaviours checked here:
- a device fault ends the worker; every job that was accepted but not started must end in status `error` with its `done` event set, and a submit
  that races with the fault must either be refused or be failed too (no accepted job is left `queued`);
- a request that is refused because the queue is full must not stay in `engine.jobs`.
"""
import importlib
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

MUSIC = Path(__file__).resolve().parent.parent / "pkgsrc" / "music3"


@pytest.fixture
def eng_mod(monkeypatch):
    for name in [m for m in sys.modules if m.split(".")[0] == "models"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(MUSIC / "code"))
    return importlib.import_module("models.autoports.minimaxai_minimax_music3.server.engine")


def _req(eng_mod, n=5):
    from models.autoports.minimaxai_minimax_music3.server.schemas import SpeechRequest
    return SpeechRequest(input="la", instructions="pop", max_new_tokens=n)


class FakeGen:
    """generate() blocks until `release` is set, then raises `fault` (or returns a stats-bearing result)."""

    def __init__(self, release, fault=None):
        self.release, self.fault, self.entered = release, fault, threading.Event()

    def generate(self, *a, **k):
        self.entered.set()
        assert self.release.wait(10)
        if self.fault:
            raise RuntimeError(self.fault)
        raise AssertionError("not used")


def _engine(eng_mod, monkeypatch, gen, maxq=8):
    monkeypatch.setenv("MUSIC3_MAX_QUEUE", str(maxq))
    e = eng_mod.Music3Engine(log=lambda *_: None)
    e._load = lambda: setattr(e, "gen", gen)  # no device, no model
    return e


def test_device_fault_fails_every_queued_job(eng_mod, monkeypatch):
    release = threading.Event()
    gen = FakeGen(release, fault="TT_THROW device timeout")
    e = _engine(eng_mod, monkeypatch, gen)
    e.start()
    assert e.wait_ready(5)
    running = e.submit(_req(eng_mod))
    assert gen.entered.wait(5)
    waiting = [e.submit(_req(eng_mod)) for _ in range(3)]
    release.set()  # the running job now raises a device fault
    for j in [running, *waiting]:
        assert j.done.wait(5), f"job {j.id} was left {j.status}"
        assert j.status == "error" and j.error
    assert e.dead and not e.ready.is_set()
    with pytest.raises(RuntimeError):  # a later submit is refused with the fault
        e.submit(_req(eng_mod))
    assert e.q.empty()


def test_submit_that_straddles_the_fault_is_not_left_queued(eng_mod, monkeypatch):
    """A submit that has passed its readiness check when the device fault lands. The queue is hooked so the fault happens (and the worker
    has time to drain) between the check and the put; the job must still end in `error`, not stay `queued`."""
    release = threading.Event()
    gen = FakeGen(release, fault="TT_THROW boom")
    e = _engine(eng_mod, monkeypatch, gen, maxq=64)
    e.start()
    assert e.wait_ready(5)
    first = e.submit(_req(eng_mod))
    assert gen.entered.wait(5)
    real_put, hooked = e.q.put_nowait, []

    def put_after_fault(job):
        if not hooked:
            hooked.append(1)
            release.set()  # the running job faults now
            time.sleep(0.4)  # long enough for an unlocked worker to finish its drain
        return real_put(job)

    e.q.put_nowait = put_after_fault
    try:
        late = e.submit(_req(eng_mod))
    except RuntimeError:
        late = None  # refusing the job is also correct
    assert first.done.wait(5)
    if late is not None:
        assert late.done.wait(5), f"job {late.id} was left {late.status}"
        assert late.status == "error"


def test_rejected_job_is_not_registered(eng_mod, monkeypatch):
    e = _engine(eng_mod, monkeypatch, FakeGen(threading.Event()), maxq=1)
    e.ready.set()  # no worker: the queue only fills
    kept = e.submit(_req(eng_mod))
    for _ in range(5):
        with pytest.raises(queue.Full):
            e.submit(_req(eng_mod))
    assert list(e.jobs) == [kept.id]
