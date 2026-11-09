# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The refresh button of the Models view's Updates panel (issue 231): orchard/updates_refresh.py and the routes in orchard/webui.py.

The button makes the web server run `tt-orchard updates --refresh`, a process that queries public repositories and the Hugging Face Hub.
So the job is held to a fixed argument list, no shell, a scrubbed environment, no captured output, a timeout, one run at a time and a
minimum interval, and the route is a POST that needs the page's token and origin, and is refused when the UI listens on the network.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from orchard import agent, updates_refresh as ur
from test_webui import Served, TOKEN, make_app, root  # noqa: F401  (root is a fixture)


class FakeProc:
    """What subprocess.Popen returns, with a wait() the test controls."""
    pid = 4242

    def __init__(self, *, code=0, hang=False, gate=None):
        self.code, self.hang, self.gate, self.killed = code, hang, gate, False

    def wait(self, timeout=None):
        if self.gate is not None:
            self.gate.wait(5)
        if self.hang and not self.killed:
            raise subprocess.TimeoutExpired("x", timeout)
        return self.code

    def kill(self):
        self.killed = True


class Popen:
    """A recording stand-in for subprocess.Popen."""

    def __init__(self, proc=None, error=None):
        self.calls, self.proc, self.error = [], proc or FakeProc(), error

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        if self.error:
            raise self.error
        return self.proc


@pytest.fixture(autouse=True)
def no_real_signals(monkeypatch):
    """A fake process has an invented pid: nothing in these tests may signal a real process group."""
    sent = []
    monkeypatch.setattr(ur.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    return sent


def wait_done(job, limit=5.0):
    end = time.time() + limit
    while job.state()["running"] and time.time() < end:
        time.sleep(0.01)
    return job.state()


def make_job(popen, now=None, **kw):
    clock = now if now is not None else (lambda: 1000.0)
    return ur.RefreshJob(popen=popen, clock=clock, environ=kw.pop("environ", {"PATH": "/usr/bin", "HOME": "/home/x"}), **kw)


# ---- what is run ---------------------------------------------------------------------------------------------------------------

def test_the_argument_list_is_fixed_and_no_shell_is_used():
    p = Popen()
    job = make_job(p)
    assert job.start() == "started"
    wait_done(job)
    argv, kw = p.calls[0]
    assert argv == [sys.executable, "-m", "orchard.cli", "updates", "--refresh"]
    assert not kw.get("shell")
    assert kw["start_new_session"] is True


def test_nothing_the_caller_sends_reaches_the_command():
    import inspect
    assert list(inspect.signature(ur.RefreshJob.start).parameters) == ["self"]       # start() takes no argument at all


def test_no_output_is_captured_and_stdin_is_closed():
    p = Popen()
    job = make_job(p)
    job.start()
    wait_done(job)
    kw = p.calls[0][1]
    assert kw["stdin"] is subprocess.DEVNULL and kw["stdout"] is subprocess.DEVNULL and kw["stderr"] is subprocess.DEVNULL


def test_the_environment_holds_only_allowed_names():
    secrets = {"HF_TOKEN": "hf_x", "GITHUB_TOKEN": "gh_x", "SSH_AUTH_SOCK": "/s", "ANTHROPIC_API_KEY": "k", "AWS_SECRET_ACCESS_KEY": "a",
               "HTTPS_PROXY": "http://user:pw@proxy", "TT_VISIBLE_DEVICES": "0"}
    env = {"PATH": "/usr/bin", "HOME": "/home/x", "LANG": "C", **secrets}
    p = Popen()
    job = make_job(p, environ=env)
    job.start()
    wait_done(job)
    child = p.calls[0][1]["env"]
    assert set(child) <= set(ur.ENV_ALLOW)
    assert {"PATH", "HOME", "LANG"} <= set(child)
    assert not (set(child) & set(secrets))
    assert "pw" not in json.dumps(child)


def test_the_allow_list_contains_the_agent_shells_and_nothing_secret_looking():
    assert set(agent.ENV_ALLOW) <= set(ur.ENV_ALLOW)
    assert not [k for k in ur.ENV_ALLOW if any(w in k.upper() for w in ("TOKEN", "KEY", "SECRET", "PASS", "AUTH", "PROXY"))]


# ---- how it ends ---------------------------------------------------------------------------------------------------------------

def test_a_timeout_kills_the_process_group_and_is_reported_in_fixed_words():
    sent = []
    proc = FakeProc(hang=True)
    p = Popen(proc)
    job = make_job(p, timeout_s=0.05)
    job.start()
    st = wait_done(job)
    assert proc.killed
    assert st["outcome"] == "timed-out" and st["running"] is False
    assert st["message"] == ur.MESSAGES["timed-out"]


def test_a_failing_exit_is_reported_with_its_code_only():
    job = make_job(Popen(FakeProc(code=3)))
    job.start()
    st = wait_done(job)
    assert st["outcome"] == "failed" and st["exit"] == 3
    assert "3" in st["message"]


def test_a_start_that_cannot_happen_is_reported_without_the_error_text():
    job = make_job(Popen(error=OSError("secret path /home/x/.config")))
    job.start()
    st = wait_done(job)
    assert st["outcome"] == "could-not-start"
    assert "secret" not in json.dumps(st) and "/home/x" not in json.dumps(st)


def test_the_state_uses_a_closed_vocabulary():
    assert set(ur.OUTCOMES) == {"ok", "failed", "timed-out", "could-not-start"}
    job = make_job(Popen())
    job.start()
    st = wait_done(job)
    assert st["outcome"] in ur.OUTCOMES and st["message"] in set(ur.MESSAGES.values()) | {ur.MESSAGES["failed"].format(code=0)}


# ---- one at a time, and not too often --------------------------------------------------------------------------------------------

def test_a_second_start_while_one_runs_starts_nothing():
    gate = threading.Event()
    p = Popen(FakeProc(gate=gate))
    job = make_job(p)
    assert job.start() == "started"
    assert job.start() == "already-running"
    assert job.start() == "already-running"
    gate.set()
    wait_done(job)
    assert len(p.calls) == 1


def test_a_start_within_the_minimum_interval_is_refused_with_the_wait():
    now = [1000.0]
    p = Popen()
    job = make_job(p, now=lambda: now[0])
    assert job.start() == "started"
    wait_done(job)
    now[0] += ur.MIN_INTERVAL_S - 5
    assert job.start() == "too-soon"
    assert 4 <= job.state()["retry_in_s"] <= 6
    assert len(p.calls) == 1
    now[0] += 10
    assert job.start() == "started"
    wait_done(job)
    assert len(p.calls) == 2


def test_the_interval_counts_from_the_start_of_the_last_run():
    now = [1000.0]
    job = make_job(Popen(), now=lambda: now[0])
    job.start()
    wait_done(job)
    assert job.state()["retry_in_s"] == ur.MIN_INTERVAL_S


# ---- the web routes ------------------------------------------------------------------------------------------------------------

def served_with(root, job, *, lan=False):
    app = make_app(root)
    app.updates_refresh = job
    app.lan = lan
    return Served(app)


def test_get_reports_the_state_and_starts_nothing(root):
    p = Popen()
    with served_with(root, make_job(p)) as s:
        code, _, body = s.request("GET", "/api/updates/refresh")
    st = json.loads(body)
    assert code == 200 and st["available"] is True and st["running"] is False
    assert p.calls == []


def test_a_post_needs_the_token_and_the_origin(root):
    p = Popen()
    with served_with(root, make_job(p)) as s:
        assert s.post("/api/updates/refresh", {}, token="")[0] == 403
        assert s.post("/api/updates/refresh", {}, token="wrong")[0] == 403
        assert s.post("/api/updates/refresh", {}, origin="http://evil.example")[0] == 403
        assert s.request("POST", "/api/updates/refresh", {}, {"Host": "evil.example"})[0] == 403
    assert p.calls == []


def test_a_post_starts_the_refresh_and_a_second_is_refused(root):
    gate = threading.Event()
    p = Popen(FakeProc(gate=gate))
    with served_with(root, make_job(p)) as s:
        code, _, body = s.post("/api/updates/refresh", {})
        assert code == 200 and json.loads(body)["running"] is True
        assert s.post("/api/updates/refresh", {})[0] == 409
        gate.set()
    assert len(p.calls) == 1


def test_a_post_too_soon_after_a_run_is_429(root):
    job = make_job(Popen())
    with served_with(root, job) as s:
        assert s.post("/api/updates/refresh", {})[0] == 200
        wait_done(job)
        code, _, body = s.post("/api/updates/refresh", {})
    assert code == 429 and "wait" in json.loads(body)["error"]


def test_the_body_of_the_post_changes_nothing(root):
    p = Popen()
    with served_with(root, make_job(p)) as s:
        s.post("/api/updates/refresh", {"argv": ["rm", "-rf", "/"], "offline": False, "refresh": "x", "env": {"A": "b"}})
    wait_done(s.server.app.updates_refresh)
    assert p.calls[0][0] == [sys.executable, "-m", "orchard.cli", "updates", "--refresh"]


def test_the_button_is_off_while_the_ui_listens_on_the_network(root):
    p = Popen()
    with served_with(root, make_job(p), lan=True) as s:
        assert s.post("/api/updates/refresh", {})[0] == 403
        code, _, body = s.request("GET", "/api/updates/refresh")
    assert code == 200 and json.loads(body)["available"] is False
    assert p.calls == []


def test_the_read_only_route_still_refuses_a_post_and_still_serves_the_rows(root):
    with served_with(root, make_job(Popen())) as s:
        assert s.post("/api/updates", {})[0] >= 400
        assert s.request("GET", "/api/updates")[0] == 200


def test_the_answer_holds_no_path_and_no_environment(root):
    job = make_job(Popen(error=OSError("/home/x/secret")), environ={"PATH": "/usr/bin", "HOME": "/home/x", "HF_TOKEN": "hf_abc"})
    with served_with(root, job) as s:
        s.post("/api/updates/refresh", {})
        wait_done(job)
        _, _, body = s.request("GET", "/api/updates/refresh")
    text = body.decode()
    assert "/home/x" not in text and "hf_abc" not in text and "/usr/bin" not in text


# ---- the page ------------------------------------------------------------------------------------------------------------------

def test_the_page_has_the_button_and_says_what_it_does(root):
    with Served(make_app(root)) as s:
        _, _, html = s.request("GET", "/")
        _, _, js = s.request("GET", "/static/app.js")
    html, js = html.decode(), js.decode()
    assert 'id="updates-refresh"' in html
    tip = html.split('id="updates-refresh"', 1)[1].split(">", 1)[0]
    assert "title=" in tip and "GitHub" in tip and "Hugging Face" in tip and "network" in tip.lower()
    assert "/api/updates/refresh" in js
