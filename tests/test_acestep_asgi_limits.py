"""Issue 271, the two server parts that need no board: a per-request time limit for generation, and exact start-up stamps in the ready line.
Driven the same way as tests/test_acestep_asgi.py (a fake session, no chip)."""
import json
import threading
import time

import pytest

import test_acestep_asgi as base
from test_acestep_asgi import FakeSession, REQ, asgi, call, music, run  # noqa: F401  (the autouse fixture `state` comes with the import)

state = base.state  # reuse the module's autouse fixture


def test_the_timeout_setting_comes_from_the_environment_and_has_a_default():
    assert asgi.settings({}).request_timeout_s == asgi.DEFAULT_REQUEST_TIMEOUT_S > 0
    assert asgi.settings({"ACESTEP_REQUEST_TIMEOUT_S": "12.5"}).request_timeout_s == 12.5
    for bad in ("0", "-1", "nan", "inf", "abc", ""):
        with pytest.raises(ValueError, match="ACESTEP_REQUEST_TIMEOUT_S"):
            asgi.settings({"ACESTEP_REQUEST_TIMEOUT_S": bad})


def test_a_generation_over_the_limit_gets_504_and_the_slot_stays_taken_until_the_chip_call_returns(monkeypatch):
    sess = FakeSession()
    sess.release = threading.Event()
    asgi.STATE["session"] = sess
    monkeypatch.setitem(asgi.STATE, "timeout_s", 0.3)

    async def both():
        t0 = time.time()
        first = await music(REQ)
        waited = time.time() - t0
        second = await music(REQ)          # the chip call of the first request is still running
        sess.release.set()                 # now it ends
        return first, waited, second

    (status, _, body), waited, (status2, _, body2) = run(both())
    assert status == 504 and "time limit" in json.loads(body)["error"] and 0.25 <= waited < 5
    assert status2 == 503 and "being made" in json.loads(body2)["error"] and len(sess.calls) == 1
    deadline = time.time() + 5
    while asgi.LOCK.locked() and time.time() < deadline:
        time.sleep(0.02)
    assert not asgi.LOCK.locked()          # the slot freed itself when the chip call returned
    sess.release = None
    assert run(music(REQ))[0] == 200


def test_a_generation_inside_the_limit_is_unchanged(monkeypatch):
    monkeypatch.setitem(asgi.STATE, "timeout_s", 30.0)
    assert run(music(REQ))[0] == 200


def test_startup_puts_an_exact_ready_time_in_the_ready_line(tmp_path, monkeypatch, capsys):
    captured = {}
    base.install_fake_runtime(monkeypatch, captured)
    code, snap = base.make_code(tmp_path), base.make_snapshot(tmp_path, "r1")
    monkeypatch.setattr(asgi, "resolve_inputs", lambda a: (code, snap))
    before = time.time()
    asgi.startup(asgi.settings({"ACESTEP_WORK": str(tmp_path / "w"), "ACESTEP_FAST": "none"}))
    after = time.time()
    ready = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert before - 0.001 <= ready["ready_unix"] <= after + 0.001      # the stamp is rounded to a millisecond
    assert 0 <= ready["since_import_s"] <= after - asgi.IMPORTED_AT + 1
    assert asgi.STATE["timeout_s"] > 0
