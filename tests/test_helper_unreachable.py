# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 650: an unreachable helper endpoint fails fast and stops the run from asking.

On 2026-11-09 the internal network behind the helper endpoint went away. Each helper call waited about 7 minutes for the TCP connect to
time out, the run kept asking, and hardware time went on coder-only repair rounds. Now: (1) a short connect timeout separate from the read
timeout; (2) a per-run circuit breaker that stops asking after a few connection failures and says so once; (3) a block code
`helper-unreachable` for unattended runs, retried by the auto-retry; (4) a preflight row. Nothing here opens a network connection: the
connect function and the transport are fakes, and time comes from an injected clock."""
import json
import socket
import types
import urllib.error

import pytest

from orchard import blocked, handoff, port_repair, preflight
from orchard import telegraph as tg
from orchard import supervisor as sup
from test_telegraph import KEY, Wire, make, openai_reply, records, table  # noqa: F401 - the shared fakes


class Down:
    """A transport whose connection always fails the way an unreachable network does."""

    def __init__(self):
        self.calls = 0

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        raise tg.Unreachable("connect to llm.example.org:443 timed out after 10 s")


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ---- config ------------------------------------------------------------------------------------------------------------------------

def test_the_new_keys_have_defaults_and_are_checked(tmp_path):
    cfg = tg.load(table(), base=tmp_path)
    assert (cfg.connect_timeout_s, cfg.unreachable_after, cfg.unreachable_action, cfg.required, cfg.unreachable_probe_s) == (10, 2, "block", False, 300)
    cfg = tg.load(table(connect_timeout_s=3, unreachable_after=1, unreachable_action="continue", required=True, unreachable_probe_s=60), base=tmp_path)
    assert (cfg.connect_timeout_s, cfg.unreachable_after, cfg.unreachable_action, cfg.required, cfg.unreachable_probe_s) == (3, 1, "continue", True, 60)
    for bad in ({"connect_timeout_s": 0}, {"connect_timeout_s": "10"}, {"unreachable_after": 0}, {"unreachable_after": True},
                {"unreachable_action": "ignore"}, {"required": "yes"}, {"unreachable_probe_s": 0}):
        with pytest.raises(tg.ConfigError):
            tg.load(table(**bad), base=tmp_path)


# ---- the connect timeout -----------------------------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status=200, body=b"{}", stall=False):
        self.status, self._body, self.stall, self.read_calls = status, body, stall, 0

    def read(self, n=None):
        self.read_calls += 1
        if self.stall:
            raise AssertionError("the body was read although only the status was needed")
        return self._body


class FakeConnection:
    """A stand-in for http.client's connection classes: it records the constructor arguments, the timeouts and what was sent, and never opens a
    socket. `fail_connect` and `fail_request` choose the phase in which it fails."""
    made: list = []
    fail_connect = None
    fail_request = None
    response = None

    def __init__(self, host, port=None, timeout=None, context=None):
        self.host, self.port, self.timeout, self.tunnel, self.sent, self.closed = host, port, timeout, None, None, False
        self.sock = types.SimpleNamespace(timeouts=[], settimeout=lambda t: self.sock.timeouts.append(t))
        FakeConnection.made.append(self)

    def set_tunnel(self, host, port=None, headers=None):
        self.tunnel = (host, port, dict(headers or {}))

    def connect(self):
        if FakeConnection.fail_connect is not None:
            raise FakeConnection.fail_connect

    def request(self, method, target, body=None, headers=None):
        if FakeConnection.fail_request is not None:
            raise FakeConnection.fail_request
        self.sent = (method, target, body, dict(headers or {}))

    def getresponse(self):
        return FakeConnection.response or FakeResponse()

    def close(self):
        self.closed = True


@pytest.fixture
def fake_http(monkeypatch):
    FakeConnection.made, FakeConnection.fail_connect, FakeConnection.fail_request, FakeConnection.response = [], None, None, None
    monkeypatch.setattr(tg.http.client, "HTTPSConnection", FakeConnection)
    monkeypatch.setattr(tg.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(tg.urllib.request, "getproxies", lambda: {})
    return FakeConnection


def test_the_connection_is_made_with_the_short_timeout_and_the_reply_timeout_applies_once_connected(fake_http):
    status, body = tg.http_post("https://llm.example.org/v1/chat/completions", {"Authorization": "Bearer " + KEY}, b"{}", 900, connect_timeout=10)
    conn = fake_http.made[0]
    assert (conn.host, conn.port, conn.timeout) == ("llm.example.org", 443, 10)          # the connect used 10 s, not the 900 s reply timeout
    assert conn.sock.timeouts == [900]                                                      # and once connected the reply gets 900 s
    assert conn.sent[:2] == ("POST", "/v1/chat/completions") and status == 200 and conn.closed


def test_a_failure_to_connect_is_unreachable_and_never_quotes_the_headers(fake_http):
    fake_http.fail_connect = socket.timeout("timed out")
    with pytest.raises(tg.Unreachable) as err:
        tg.http_post("https://llm.example.org/v1/chat/completions", {"Authorization": "Bearer " + KEY}, b"{}", 900, connect_timeout=10)
    assert KEY not in str(err.value) and "Authorization" not in str(err.value) and "timed out after 10 s" in str(err.value)
    assert fake_http.made[0].closed


def test_a_failure_after_the_connection_was_made_is_not_unreachable(fake_http):
    """The provider may have the request: a timeout while sending or waiting is an ordinary error, with unknown cost (review of PR 651)."""
    fake_http.fail_request = socket.timeout("timed out while sending")
    with pytest.raises(socket.timeout):
        tg.http_post("https://llm.example.org/v1/chat/completions", {}, b"{}", 900, connect_timeout=10)


def test_such_a_failure_keeps_the_usage_unknown_and_does_not_count_toward_the_breaker(tmp_path):
    def sent_then_stalled(url, headers, body, timeout):
        raise socket.timeout("timed out waiting for the reply")

    tel, _ = make(tmp_path, wire=sent_then_stalled, unreachable_after=1)
    ans = tel.ask("p", "t1", "f1")
    assert not ans.ok and not ans.unreachable and not tel.tripped
    rec = records(tmp_path)[-1]
    assert rec["usage_known"] is False and rec["cost_usd"] is None and "unreachable" not in rec


def test_a_proxy_gets_the_same_short_connect_timeout_and_the_target_is_tunnelled(fake_http, monkeypatch):
    monkeypatch.setattr(tg.urllib.request, "getproxies", lambda: {"https": "http://user:p%40ss@proxy.example.org:3128"})
    monkeypatch.setattr(tg.urllib.request, "proxy_bypass", lambda host: False)
    tg.http_post("https://llm.example.org/v1/chat/completions", {}, b"{}", 900, connect_timeout=10)
    conn = fake_http.made[0]
    assert (conn.host, conn.port, conn.timeout) == ("proxy.example.org", 3128, 10)
    assert conn.tunnel[:2] == ("llm.example.org", 443) and conn.tunnel[2]["Proxy-Authorization"].startswith("Basic ")
    assert conn.sock.timeouts == [900] and conn.sent[1] == "/v1/chat/completions"


def test_a_plain_http_proxy_gets_the_absolute_url_and_the_short_connect_timeout(fake_http, monkeypatch):
    monkeypatch.setattr(tg.urllib.request, "getproxies", lambda: {"http": "http://proxy.example.org:3128"})
    monkeypatch.setattr(tg.urllib.request, "proxy_bypass", lambda host: False)
    tg.http_post("http://10.1.2.3:8000/v1/chat/completions", {}, b"{}", 900, connect_timeout=10)
    conn = fake_http.made[0]
    assert (conn.host, conn.port, conn.timeout) == ("proxy.example.org", 3128, 10) and conn.sent[1] == "http://10.1.2.3:8000/v1/chat/completions"


def test_a_failure_to_connect_through_a_proxy_is_unreachable(fake_http, monkeypatch):
    monkeypatch.setattr(tg.urllib.request, "getproxies", lambda: {"https": "http://proxy.example.org:3128"})
    monkeypatch.setattr(tg.urllib.request, "proxy_bypass", lambda host: False)
    fake_http.fail_connect = OSError("Tunnel connection failed: 403 Forbidden")
    with pytest.raises(tg.Unreachable):
        tg.http_get("https://llm.example.org/v1/models", {}, 900, connect_timeout=10)


def test_the_status_probe_returns_as_soon_as_the_headers_arrive_and_never_reads_the_body(fake_http):
    fake_http.response = FakeResponse(status=401, stall=True)
    assert tg.http_status("https://llm.example.org/v1/models", {}, 10, connect_timeout=10) == 401
    assert fake_http.response.read_calls == 0 and fake_http.made[0].closed


def test_the_telegraph_passes_its_connect_timeout_to_the_default_transport(tmp_path, monkeypatch):
    got = {}

    def fake_post(url, headers, body, timeout, connect_timeout=None):
        got.update(timeout=timeout, connect=connect_timeout)
        return 200, json.dumps(openai_reply("ok text", 10, 5)).encode()

    monkeypatch.setattr(tg, "http_post", fake_post)
    cfg = tg.load(table(timeout_s=900, connect_timeout_s=7), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", environ={"TG_KEY": KEY}, clock=lambda: 1.0, hostname="n", home="/h")
    assert tel.ask("p", "task", "failure").ok
    assert got == {"timeout": 900, "connect": 7}


# ---- the breaker -------------------------------------------------------------------------------------------------------------------

def test_a_connection_failure_is_an_unreachable_answer_that_costs_nothing(tmp_path):
    down = Down()
    tel, _ = make(tmp_path, wire=down)
    ans = tel.ask("p", "task", "failure")
    assert not ans.ok and ans.unreachable and "timed out" in ans.reason
    rec = records(tmp_path)[-1]
    assert rec["outcome"] == "error" and rec["cost_usd"] == 0.0 and rec["usage_known"] is True and rec.get("unreachable") is True


def test_after_n_failures_the_breaker_opens_and_asking_makes_no_request(tmp_path):
    down = Down()
    tel, _ = make(tmp_path, wire=down, unreachable_after=2)
    tel.ask("p", "t1", "f1")
    assert not tel.tripped
    tel.ask("p", "t2", "f2")
    assert tel.tripped and down.calls == 2
    later = tel.ask("p", "t3", "f3")
    assert down.calls == 2                                            # no third request
    assert not later.ok and later.unreachable and "not asking" in later.reason


def test_a_success_resets_the_count(tmp_path):
    calls = {"n": 0}

    def flaky(url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise tg.Unreachable("down once")
        return 200, json.dumps(openai_reply("fine text", 10, 5)).encode()

    tel, _ = make(tmp_path, wire=flaky, unreachable_after=2)
    assert tel.ask("p", "t1", "f1").unreachable
    assert tel.ask("p", "t2", "f2").ok                               # a different prompt, so not a cache hit
    assert not tel.tripped and tel._conn_failures == 0
    assert tel.ask("p", "t3", "f3").ok


def test_after_the_probe_interval_one_ask_goes_out_and_a_success_closes_the_breaker(tmp_path):
    clock = Clock()
    state = {"up": False, "calls": 0}

    def wire(url, headers, body, timeout):
        state["calls"] += 1
        if not state["up"]:
            raise tg.Unreachable("down")
        return 200, json.dumps(openai_reply("back again", 10, 5)).encode()

    tel, _ = make(tmp_path, wire=wire, now=clock, unreachable_after=1, unreachable_probe_s=300)
    tel.ask("p", "t1", "f1")
    assert tel.tripped and state["calls"] == 1
    clock.t += 299
    assert tel.ask("p", "t2", "f2").unreachable and state["calls"] == 1      # still waiting
    clock.t += 2
    state["up"] = True
    assert tel.ask("p", "t3", "f3").ok and state["calls"] == 2              # the probe went out and worked
    assert not tel.tripped


def test_a_failed_probe_keeps_the_breaker_open_for_another_interval(tmp_path):
    clock = Clock()
    down = Down()
    tel, _ = make(tmp_path, wire=down, now=clock, unreachable_after=1, unreachable_probe_s=300)
    tel.ask("p", "t1", "f1")
    clock.t += 301
    tel.ask("p", "t2", "f2")
    assert down.calls == 2
    clock.t += 10
    tel.ask("p", "t3", "f3")
    assert down.calls == 2                                           # the wait restarted at the failed probe


def test_an_http_error_is_not_a_connection_failure(tmp_path):
    tel, _ = make(tmp_path, wire=Wire(status=500, reply=lambda body: {"error": "overloaded"}), unreachable_after=1)
    ans = tel.ask("p", "t1", "f1")
    assert not ans.ok and not ans.unreachable and not tel.tripped


def test_the_error_text_never_quotes_the_key_or_the_header(tmp_path):
    def leaky(url, headers, body, timeout):
        raise tg.Unreachable(f"cannot connect, headers were Authorization: Bearer {KEY}")

    tel, _ = make(tmp_path, wire=leaky)
    ans = tel.ask("p", "t1", "f1")
    assert KEY not in ans.reason and KEY not in json.dumps(records(tmp_path))


def test_the_call_caps_still_apply_to_a_helper_that_answers(tmp_path):
    tel, wire = make(tmp_path, max_calls=1)
    assert tel.ask("p", "t1", "f1").ok
    second = tel.ask("p", "t2", "f2")
    assert not second.ok and not second.unreachable and "call cap" in second.reason


# ---- the supervisor ----------------------------------------------------------------------------------------------------------------

class FakeLedger:
    def __init__(self):
        self.rows = []

    def append(self, kind, stage, **data):
        self.rows.append((kind, stage, data))

    def read(self):
        """The rows in the shape the supervisor reads from the real ledger."""
        return [{"event": k, "stage": s, "data": d} for k, s, d in self.rows]


def supervisor_stub(tel, unattended=True):
    ns = types.SimpleNamespace(telegraph=tel, unattended=unattended, ledger=FakeLedger(), _helper_down_noted=False)
    ns._block = lambda stage, reason, **ev: sup.Supervisor._block(ns, stage, reason, **ev)
    ns._helper_block_text = lambda failures: sup.Supervisor._helper_block_text(ns, failures)
    ns._helper_outage_open = lambda: sup.Supervisor._helper_outage_open(ns)
    return ns


def trip(tmp_path, **kw):
    tel, _ = make(tmp_path, wire=Down(), unreachable_after=1, **kw)
    return tel, tel.ask("p", "t", "f")


def test_an_unattended_run_blocks_with_the_helper_unreachable_code_once_the_breaker_opens(tmp_path):
    tel, answer = trip(tmp_path)
    ns = supervisor_stub(tel)
    with pytest.raises(handoff.Blocked) as err:
        sup.Supervisor._helper_unreachable(ns, 2, answer)
    reason = err.value.reason
    assert reason.startswith(blocked.HELPER_UNREACHABLE_MARK)
    notes = [r for r in ns.ledger.rows if r[0] == "decision"]
    assert len(notes) == 1 and notes[0][2]["decision"] == "helper unreachable, not asking"
    assert blocked.block_code({"reason": f"blocked: {reason}"}) == "helper-unreachable"
    assert blocked.retryable("helper-unreachable", f"blocked: {reason}")


def test_the_decision_is_written_once_even_if_the_run_goes_on(tmp_path):
    tel, answer = trip(tmp_path, unreachable_action="continue")
    ns = supervisor_stub(tel)
    for _ in range(3):
        sup.Supervisor._helper_unreachable(ns, 2, answer)                  # continue: never raises
    assert len([r for r in ns.ledger.rows if r[0] == "decision"]) == 1
    assert any(r[0] == "notice" for r in ns.ledger.rows)


def test_an_attended_run_does_not_block(tmp_path):
    tel, answer = trip(tmp_path)
    ns = supervisor_stub(tel, unattended=False)
    sup.Supervisor._helper_unreachable(ns, 2, answer)                      # the operator is watching: a notice, no block
    assert any(r[0] == "notice" for r in ns.ledger.rows)


def test_a_helper_that_answered_or_failed_another_way_does_nothing(tmp_path):
    tel, _ = make(tmp_path)
    ns = supervisor_stub(tel)
    sup.Supervisor._helper_unreachable(ns, 2, tg.Answer(False, reason="HTTP 500"))
    sup.Supervisor._helper_unreachable(ns, 2, tg.Answer(True, "text"))
    assert ns.ledger.rows == []


def test_an_answered_request_closes_the_outage_in_the_ledger_and_an_unanswered_one_does_not(tmp_path):
    tel, _ = make(tmp_path)
    ns = supervisor_stub(tel)
    ns._helper_down_noted = True
    sup.Supervisor._helper_reached(ns, 2, tg.Answer(False, reason="HTTP 500"))              # no call id: nothing was answered
    assert ns.ledger.rows == []
    sup.Supervisor._helper_reached(ns, 2, tg.Answer(True, "text", call_id="abc"))
    sup.Supervisor._helper_reached(ns, 2, tg.Answer(True, "text", call_id="def"))           # the second is a no-op: the outage is already closed
    assert [r[2]["decision"] for r in ns.ledger.rows] == ["helper reachable again"] and ns._helper_down_noted is False


# ---- the gate at the start of a restarted process (review of PR 651, finding 3) -----------------------------------------------------

def open_outage(ns):
    ns.ledger.append("decision", 2, decision="helper unreachable, not asking", failures=2)


class Prober:
    """A stand-in for the models-route probe: counts calls and either answers or fails to connect."""
    def __init__(self, up):
        self.up, self.calls = up, 0

    def __call__(self, url, headers, timeout, connect_timeout=None):
        self.calls += 1
        if not self.up:
            raise tg.Unreachable("cannot connect")


def gate_rig(tmp_path, up, **kw):
    prober = Prober(up)
    tel, _ = make(tmp_path, **kw)
    tel.prober = prober
    ns = supervisor_stub(tel)
    ns._helper_gate_done = False
    return tel, ns, prober


def test_the_gate_probes_once_and_blocks_a_restart_while_the_endpoint_is_still_down(tmp_path):
    tel, ns, prober = gate_rig(tmp_path, up=False)
    open_outage(ns)
    with pytest.raises(handoff.Blocked) as err:
        sup.Supervisor._helper_gate(ns, 2, round=3)
    assert err.value.reason.startswith(blocked.HELPER_UNREACHABLE_MARK)
    assert prober.calls == 1 and tel.tripped                                  # probed, and the breaker is held open without an ask
    withdrawn = [r[2] for r in ns.ledger.rows if r[2].get("decision") == port_repair.WITHDRAWN]
    assert len(withdrawn) == 1 and withdrawn[0]["round"] == 3
    sup.Supervisor._helper_gate(ns, 2, round=3)                               # once per process: the second call does nothing
    assert prober.calls == 1


def test_the_gate_at_an_attempt_start_blocks_without_withdrawing_a_round(tmp_path):
    tel, ns, prober = gate_rig(tmp_path, up=False)
    open_outage(ns)
    with pytest.raises(handoff.Blocked):
        sup.Supervisor._helper_gate(ns, 2)
    assert not [r for r in ns.ledger.rows if r[2].get("decision") == port_repair.WITHDRAWN]


def test_the_gate_clears_the_outage_when_the_probe_is_answered(tmp_path):
    tel, ns, prober = gate_rig(tmp_path, up=True)
    open_outage(ns)
    sup.Supervisor._helper_gate(ns, 2, round=1)                               # no block
    assert prober.calls == 1 and not tel.tripped
    assert ns.ledger.rows[-1][2]["decision"] == "helper reachable again"
    assert sup.Supervisor._helper_outage_open(ns) is False


def test_the_gate_does_nothing_when_the_ledger_holds_no_outage(tmp_path):
    tel, ns, prober = gate_rig(tmp_path, up=False)
    sup.Supervisor._helper_gate(ns, 2, round=1)
    assert prober.calls == 0 and ns.ledger.rows == []


def test_an_attended_or_continue_run_probes_and_holds_the_breaker_but_does_not_block(tmp_path):
    tel, ns, prober = gate_rig(tmp_path, up=False)
    ns.unattended = False
    open_outage(ns)
    sup.Supervisor._helper_gate(ns, 2, round=1)
    assert tel.tripped and not [r for r in ns.ledger.rows if r[2].get("decision") == port_repair.WITHDRAWN]


def test_the_block_code_is_listed_and_read_before_the_hardware_words():
    assert "helper-unreachable" in blocked.REASONS and "helper-unreachable" in blocked.AUTO_RETRY
    text = f"blocked: {blocked.HELPER_UNREACHABLE_MARK}: 2 connection attempts failed (connect to the host timed out; chip and board and device)"
    assert blocked.block_code({"reason": text}) == "helper-unreachable"


# ---- the preflight row -------------------------------------------------------------------------------------------------------------

def test_the_preflight_row_is_ok_when_the_endpoint_answers():
    row = preflight.check_helper(None, "https://llm.example.org/v1", required=False)
    assert row.status == preflight.OK and "llm.example.org" in row.detail


def test_the_preflight_row_warns_when_the_helper_is_optional_and_blocks_when_required():
    soft = preflight.check_helper("timed out after 10 s", "https://llm.example.org/v1", required=False)
    hard = preflight.check_helper("timed out after 10 s", "https://llm.example.org/v1", required=True)
    assert soft.status == preflight.WARN and hard.status == preflight.BLOCK and hard.reason == "helper-unreachable"
    assert "timed out" in hard.detail and "without help" in soft.detail


def test_the_preflight_probe_treats_any_http_status_as_reachable_and_reads_no_body(fake_http):
    fake_http.response = FakeResponse(status=401, stall=True)            # a stalled body must not make the probe fail (review of PR 651)
    assert preflight.helper_problem("https://llm.example.org/v1", "openai", 10) is None
    conn = fake_http.made[0]
    assert conn.sent[:2] == ("GET", "/v1/models") and "Authorization" not in conn.sent[3] and conn.timeout == 10


def test_the_preflight_probe_names_the_cause_when_the_connection_cannot_be_made(fake_http):
    fake_http.fail_connect = socket.timeout("timed out")
    assert "timed out" in preflight.helper_problem("https://llm.example.org/v1", "openai", 10)
    fake_http.fail_connect = None
    fake_http.fail_request = socket.timeout("stalled")
    assert "after the connection was made" in preflight.helper_problem("https://llm.example.org/v1", "openai", 10)


# ---- the preflight wiring ----------------------------------------------------------------------------------------------------------

def _preflight(tmp_path, tele, probe):
    import dataclasses
    from test_preflight import cfg, signals
    config = dataclasses.replace(cfg(tmp_path), telegraph=tele)
    return preflight.run_preflight(config, "Cloudflare/clef", accept_credentials=False, signals=signals(helper_probe=probe))


def test_the_helper_row_is_asked_only_when_the_helper_is_enabled_and_comes_with_the_configs_endpoint(tmp_path):
    asked = []
    enabled = tg.Config(enabled=True, endpoint="https://llm.example.org/v1", models=("m",), api="openai", connect_timeout_s=7)
    out = _preflight(tmp_path, enabled, lambda e, a, t: asked.append((e, a, t)) or None)
    assert [c.status for c in out if c.name == "helper"] == [preflight.OK] and asked == [("https://llm.example.org/v1", "openai", 7)]
    asked.clear()
    out = _preflight(tmp_path, tg.Config(), lambda e, a, t: asked.append(1) or None)
    assert "helper" not in [c.name for c in out] and asked == []


def test_an_unreachable_helper_warns_by_default_and_blocks_when_required(tmp_path):
    soft = tg.Config(enabled=True, endpoint="https://llm.example.org/v1", models=("m",))
    hard = tg.Config(enabled=True, endpoint="https://llm.example.org/v1", models=("m",), required=True)
    out = _preflight(tmp_path, soft, lambda e, a, t: "timed out after 10 s")
    assert next(c for c in out if c.name == "helper").status == preflight.WARN and not [c for c in preflight.blocked(out) if c.name == "helper"]
    out = _preflight(tmp_path, hard, lambda e, a, t: "timed out after 10 s")
    assert [c.reason for c in preflight.blocked(out) if c.name == "helper"] == ["helper-unreachable"]


def test_a_probe_that_raises_is_a_row_and_never_an_exception(tmp_path):
    def boom(e, a, t):
        raise RuntimeError("odd")

    out = _preflight(tmp_path, tg.Config(enabled=True, endpoint="https://llm.example.org/v1", models=("m",)), boom)
    assert "could not check" in next(c for c in out if c.name == "helper").detail


def test_the_real_preflight_signals_carry_the_helper_probe(tmp_path):
    from test_preflight import cfg
    assert preflight.default_signals(cfg(tmp_path)).helper_probe is preflight.helper_problem
