# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The telegraph spend view for the web page (orchard/telegraph_view.py, GET /api/telegraph).

The ledgers here are written by the real `telegraph.Telegraph` with a fake transport, so the records have the shape the
producer writes. The view is read only: no test sends a request to a helper model, and one test proves the route spends
nothing. The key, the key file's path and the key variable's value never appear in a response."""
import dataclasses
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import telegraph as tg
from orchard import telegraph_view as view
from test_webui import Served, make_app, TOKEN, write_ledger

KEY = "k3y-VALUE-0123456789abcdef"
HOST = "quietbox"


def reply(tin=1000, tout=500):
    def send(url, headers, body, timeout):
        return 200, json.dumps({"choices": [{"message": {"content": "use fill for constants"}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": tin, "completion_tokens": tout}}).encode()
    return send


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        self.t += 7
        return self.t


def table(**kw):
    base = {"enabled": True, "endpoint": "https://llm.example.org:8443/v1", "models": ["small", "big"],
            "key_env": "TG_KEY", "prices": {"small": [1.0, 5.0], "big": [10.0, 50.0]}}
    base.update(kw)
    return base


def setup(tmp_path, **kw):
    d = tmp_path / "ledger"
    cfg = tg.load(table(dir=str(d), **kw), base=tmp_path)
    tel = tg.Telegraph(cfg, d, transport=reply(), environ={"TG_KEY": KEY}, clock=Clock(), hostname=HOST, home="/home/x")
    return cfg, d, tel


def app_for(root, cfg, environ=None):
    app = make_app(root)
    app.cfg = SimpleNamespace(runs_root=root, coder=SimpleNamespace(port=8001, target="org/coder", chips=2), telegraph=cfg)
    app.environ = {"TG_KEY": KEY} if environ is None else environ
    return app


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


# ---- off, and nothing to show -------------------------------------------------------------------------------------

def test_telegraph_off_is_a_quiet_but_visible_state(root, tmp_path):
    out = app_for(root, tg.Config()).telegraph()
    assert out["state"] == "off" and out["enabled"] is False and out["tone"] == "dim"
    assert out["meter"]["text"] == "Helper: off"
    assert out["summary"]["calls"] == 0 and out["recent"] == []


def test_enabled_with_no_calls_shows_zero_spend_and_the_caps(root, tmp_path):
    cfg, d, _ = setup(tmp_path, max_calls=5)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "ok" and out["tone"] == "good"
    assert out["meter"]["text"].startswith("Helper: $0.00 · 0 calls")
    assert "calls 0 of 5" in out["meter"]["text"]


def test_a_missing_ledger_directory_gives_an_empty_summary_not_an_error(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    assert not d.exists()
    out = app_for(root, cfg).telegraph()
    assert out["error"] is None and out["summary"]["calls"] == 0


def test_an_unreadable_directory_gives_an_empty_summary(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    tel.ask("p", "task", "failure")
    os.chmod(d, 0)
    try:
        out = app_for(root, cfg).telegraph()
    finally:
        os.chmod(d, 0o700)
    assert out["summary"]["calls"] in (0, 1)                 # root can still read; the page must not fail either way
    assert out["state"] in ("ok", "problem")


def test_the_default_directory_is_under_the_home_cache_when_the_config_has_no_dir(root, tmp_path):
    cfg = tg.load({k: v for k, v in table().items()}, base=tmp_path)
    app = app_for(root, cfg)
    app.home = str(tmp_path / "home")
    d = Path(app.home) / ".cache" / "tt-orchard" / "telegraph"
    tg.Telegraph(cfg, d, transport=reply(), environ={"TG_KEY": KEY}, clock=Clock()).ask("p", "t", "f")
    assert app.telegraph()["summary"]["calls"] == 1


# ---- spend and counts ---------------------------------------------------------------------------------------------

def test_spend_is_the_known_cost_and_calls_are_sent_requests(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=10, max_calls_per_problem=9)
    tel.ask("a", "task one " * 80, "failed one")
    tel.ask("b", "task two " * 80, "failed two")
    tel.ask("b", "task two " * 80, "failed two")           # same question: answered from the cache, costs nothing
    out = app_for(root, cfg).telegraph()
    s = out["summary"]
    assert s["calls"] == 2 and s["cache_hits"] == 1
    assert s["cost_usd"] == pytest.approx(2 * (1000 * 1.0 + 500 * 5.0) / 1e6)
    assert out["meter"]["text"].startswith("Helper: $0.0070 · 2 calls")


def test_calls_with_no_price_say_the_cost_is_unknown_and_never_show_zero_dollars(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={})
    tel.ask("a", "task " * 80, "failed")
    out = app_for(root, cfg).telegraph()
    assert out["summary"]["cost_unknown_calls"] == 1
    assert "unknown cost" in out["meter"]["text"] and "$0.00" not in out["meter"]["text"]


def test_a_mix_of_priced_and_unpriced_calls_shows_the_known_part_and_the_unknown_count(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={"small": [1.0, 5.0]}, max_calls=10, max_calls_per_problem=9)
    tel.ask("a", "task " * 80, "failed")
    tel.report(next(r["call_id"] for r in tg.read_ledger(d) if "call_id" in r), helped=False)
    tel.ask("a", "task " * 80, "failed")                        # the second model has no price
    out = app_for(root, cfg).telegraph()
    assert out["summary"]["cost_unknown_calls"] == 1
    assert "+ 1 unknown" in out["meter"]["text"]


# ---- caps: near and capped ---------------------------------------------------------------------------------------

def fill(tel, n, problem_prefix="p"):
    for i in range(n):
        tel.ask(f"{problem_prefix}{i}", f"task {i} " * 80, "failed")


def test_below_eighty_percent_of_every_cap_the_state_is_ok(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=5)
    fill(tel, 3)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "ok" and out["tone"] == "good"


def test_at_eighty_percent_of_a_cap_the_state_is_near(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=5)
    fill(tel, 4)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "near" and out["tone"] == "warn"
    assert "calls 4 of 5" in out["meter"]["text"] and "near" in out["meter"]["text"]
    calls_cap = next(c for c in out["caps"] if c["id"] == "calls")
    assert calls_cap["percent"] == 80 and calls_cap["tone"] == "warn"


def test_the_token_cap_and_the_cost_cap_count_too(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=50, max_total_tokens=1900, max_cost_usd=100.0)
    fill(tel, 1)                                               # 1500 tokens of 1900 is 79 percent: ok
    assert app_for(root, cfg).telegraph()["state"] == "ok"
    cfg2, d2, tel2 = setup(tmp_path / "b", max_calls=50, max_total_tokens=1800, max_cost_usd=100.0)
    fill(tel2, 1)                                              # 1500 of 1800 is 83 percent
    out = app_for(root, cfg2).telegraph()
    assert out["state"] == "near" and "tokens 1500 of 1800" in out["meter"]["text"]
    # The producer would refuse a call whose worst case passes a small cap, so the cap is lowered after the call was made.
    cfg3, d3, tel3 = setup(tmp_path / "c", max_calls=50, max_cost_usd=100.0)
    fill(tel3, 1)                                              # the call cost $0.0035
    assert app_for(root, dataclasses.replace(cfg3, max_cost_usd=0.0045)).telegraph()["state"] == "ok"      # 77 percent
    out = app_for(root, dataclasses.replace(cfg3, max_cost_usd=0.0040)).telegraph()                        # 87 percent
    assert out["state"] == "near" and any(c["id"] == "cost" and c["percent"] == 87 for c in out["caps"])
    assert "cost $0.0035 of $0.0040" in out["meter"]["text"]
    assert app_for(root, dataclasses.replace(cfg3, max_cost_usd=0.0035)).telegraph()["state"] == "capped"


def test_a_per_problem_cap_at_eighty_percent_makes_the_state_near(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=50, max_calls_per_problem=5)
    for _ in range(4):
        tel.ask("same", "task " * 80, "failed")                  # the same text is a cache hit after the first
    fill_ok = [tel.ask("same", f"task variant {i} " * 80, "failed") for i in range(3)]
    assert all(a.ok for a in fill_ok)
    out = app_for(root, cfg).telegraph()
    prob = next(p for p in out["problems"] if p["name"] == "same")
    assert prob["calls"] == 4 and prob["percent"] == 80 and out["state"] == "near"


def test_a_cap_that_is_used_up_is_capped_with_its_reason(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=2, max_calls_per_problem=5)
    fill(tel, 2)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "capped" and out["tone"] == "bad"
    assert "call cap is reached (2 of 2)" in out["capped"]["reason"]
    assert "CAPPED" in out["meter"]["text"] and "call cap is reached" in out["meter"]["text"]


def test_a_refusal_as_the_latest_call_is_capped_and_a_later_success_clears_it(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=50, max_calls_per_problem=1)
    tel.ask("x", "task " * 80, "failed")
    refused = tel.ask("x", "task other " * 80, "failed")
    assert not refused.ok
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "capped" and "problem x" in out["capped"]["reason"]
    tel.ask("y", "task y " * 80, "failed")                      # another problem is still answered
    assert app_for(root, cfg).telegraph()["state"] != "capped"


def test_capped_wins_over_near_and_a_missing_key(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=1)
    fill(tel, 1)
    out = app_for(root, cfg, environ={}).telegraph()
    assert out["state"] == "capped"


def test_a_cost_cap_with_no_price_is_still_shown_as_a_cap_that_cannot_be_measured(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=9, max_cost_usd=1.0)
    out = app_for(root, cfg).telegraph()
    assert any(c["id"] == "cost" and c["limit"] == 1.0 for c in out["caps"])


# ---- the key --------------------------------------------------------------------------------------------------------

def file_key_cfg(tmp_path, d, key_file):
    t = {k: v for k, v in table(dir=str(d)).items() if k != "key_env"}
    return tg.load({**t, "key_file": str(key_file)}, base=tmp_path)


def test_a_missing_key_is_a_problem_state_and_says_so(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    out = app_for(root, cfg, environ={}).telegraph()
    assert out["state"] == "problem" and out["tone"] == "warn"
    assert out["key"]["state"] == "missing" and "key missing" in out["meter"]["text"]


def test_the_key_is_present_from_a_variable_or_a_file_and_neither_value_nor_path_is_returned(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    out = app_for(root, cfg).telegraph()
    assert out["key"] == {"state": "present", "source": "variable", "reason": None}
    kf = tmp_path / "secret-dir" / "the-key-file"
    kf.parent.mkdir()
    kf.write_text(KEY + "\n")
    cfg2 = file_key_cfg(tmp_path, d, kf)
    out2 = app_for(root, cfg2).telegraph()
    assert out2["key"]["state"] == "present" and out2["key"]["source"] == "file"
    blob = json.dumps(out) + json.dumps(out2)
    assert KEY not in blob and str(kf) not in blob and "secret-dir" not in blob and "TG_KEY" not in blob


def test_an_empty_key_file_is_missing(root, tmp_path):
    kf = tmp_path / "k"
    kf.write_text("  \n")
    cfg = file_key_cfg(tmp_path, tmp_path / "ledger", kf)
    assert app_for(root, cfg).telegraph()["key"]["state"] == "missing"


def test_a_keyless_endpoint_says_no_key_is_needed(root, tmp_path):
    d = tmp_path / "ledger"
    cfg = tg.load({k: v for k, v in table(dir=str(d)).items() if k != "key_env"}, base=tmp_path)
    out = app_for(root, cfg).telegraph()
    assert out["key"]["state"] == "not-needed" and out["state"] == "ok"


def test_a_key_file_that_is_a_directory_or_unreadable_is_missing_and_the_page_still_answers(root, tmp_path):
    kd = tmp_path / "kdir"
    kd.mkdir()
    cfg = file_key_cfg(tmp_path, tmp_path / "ledger", kd)
    out = app_for(root, cfg).telegraph()
    assert out["key"]["state"] == "missing" and out["state"] == "problem"
    gone = file_key_cfg(tmp_path, tmp_path / "ledger", tmp_path / "no-such-file")
    assert app_for(root, gone).telegraph()["key"]["state"] == "missing"


# ---- the configuration as shown -------------------------------------------------------------------------------------

def test_the_endpoint_is_shown_as_a_host_only_and_models_keep_their_ladder_order_and_prices(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    out = app_for(root, cfg).telegraph()
    c = out["config"]
    assert c["endpoint_host"] == "llm.example.org"
    assert "8443" not in json.dumps(c) and "/v1" not in json.dumps(c)
    assert [m["name"] for m in c["models"]] == ["small", "big"]
    assert c["models"][0]["input_per_mtok"] == 1.0 and c["models"][1]["output_per_mtok"] == 50.0
    assert c["after_failures"] == 2 and c["api"] == "openai"


def test_the_endpoints_host_name_is_redacted_when_it_is_this_machine(root, tmp_path):
    cfg, d, _ = setup(tmp_path, endpoint=f"http://{HOST}/v1")
    app = app_for(root, cfg)
    app.hostname = HOST
    assert app.telegraph()["config"]["endpoint_host"] == "<host>"


def test_a_model_without_a_price_shows_none(root, tmp_path):
    cfg, d, _ = setup(tmp_path, prices={"small": [1.0, 2.0]})
    models = app_for(root, cfg).telegraph()["config"]["models"]
    assert models[1]["input_per_mtok"] is None and models[1]["output_per_mtok"] is None


def test_the_ledger_location_is_shown_with_the_home_replaced(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    app = app_for(root, cfg)
    app.home = str(tmp_path)
    assert app.telegraph()["config"]["ledger_dir"].startswith("~/")


# ---- recent calls, breakdowns -------------------------------------------------------------------------------------

def test_recent_calls_are_newest_first_with_every_outcome_and_whether_they_helped(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=3, max_calls_per_problem=9)
    a = tel.ask("one", "task one " * 80, "failed")
    tel.report(a.call_id, helped=True)
    b = tel.ask("two", "task two " * 80, "failed")
    tel.ask("two", "task two " * 80, "failed")                   # cache hit
    tel.ask("three", "task three " * 80, "failed")
    refused = tel.ask("four", "task four " * 80, "failed")
    assert not refused.ok
    out = app_for(root, cfg).telegraph()
    rows = out["recent"]
    assert [r["outcome"] for r in rows] == ["budget", "ok", "cache hit", "ok", "ok"]
    assert rows[0]["reason"].startswith("the call cap is reached")
    assert rows[-1]["helped"] is True and rows[-2]["helped"] is None
    first = rows[-1]
    assert set(first) >= {"ts", "call_id", "problem", "model", "tokens_in", "tokens_out", "latency_s", "cost_usd", "outcome"}
    assert first["tokens_in"] == 1000 and first["tokens_out"] == 500 and first["model"] == "small"
    assert rows[2]["cost_usd"] == 0.0 and rows[2]["tokens_in"] == 0


def test_a_call_reported_as_not_helping_says_so(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    a = tel.ask("one", "task " * 80, "failed")
    tel.report(a.call_id, helped=False)
    assert app_for(root, cfg).telegraph()["recent"][0]["helped"] is False


def test_the_recent_list_is_bounded(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    d.mkdir()
    with (d / "telegraph.jsonl").open("w") as fh:
        for i in range(500):
            fh.write(json.dumps({"call_id": f"c{i:04d}", "problem": "p", "model": "m", "ts": 1.0 + i, "outcome": "ok",
                                 "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.001, "latency_s": 0.1}) + "\n")
    out = app_for(root, cfg).telegraph()
    assert len(out["recent"]) == view.RECENT_MAX == 50
    assert out["recent"][0]["call_id"] == "c0499"
    assert out["summary"]["calls"] == 500


def test_breakdowns_by_model_and_by_problem_are_lists_with_tokens_and_cost(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=9, max_calls_per_problem=9)
    a = tel.ask("p1", "task " * 80, "failed")
    tel.report(a.call_id, helped=False)
    tel.ask("p1", "task " * 80, "failed")                        # second model after a not-helped report
    out = app_for(root, cfg).telegraph()
    models = {m["model"]: m for m in out["by_model"]}
    assert models["small"]["calls"] == 1 and models["big"]["calls"] == 1
    assert models["big"]["cost_usd"] == pytest.approx((1000 * 10.0 + 500 * 50.0) / 1e6)
    (p,) = out["problems"]
    assert p["name"] == "p1" and p["calls"] == 2 and p["tokens"] == 3000 and p["not_helped"] == 1


# ---- hostile or broken ledgers -----------------------------------------------------------------------------------

def test_a_malformed_line_never_breaks_the_page(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    tel.ask("p", "task " * 80, "failed")
    with (d / "telegraph.jsonl").open("ab") as fh:
        fh.write(b"{not json\n\xff\xfe\n[1,2]\n")
    out = app_for(root, cfg).telegraph()
    assert out["summary"]["calls"] == 1 and out["error"] is None


def test_a_record_with_values_of_the_wrong_type_gives_an_error_and_not_a_crash(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    d.mkdir(exist_ok=True)
    (d / "telegraph.jsonl").write_text(json.dumps({"call_id": "x", "outcome": "ok", "tokens_in": "lots", "model": "m"}) + "\n")
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem" and "ledger" in out["error"]
    assert out["recent"] == []


def test_a_huge_ledger_is_not_read(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    d.mkdir()
    with (d / "telegraph.jsonl").open("wb") as fh:
        fh.truncate(view.LEDGER_MAX_BYTES + 1)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem" and "too large" in out["error"]


def test_strings_from_the_ledger_are_redacted(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    d.mkdir()
    rec = {"call_id": "c1", "problem": "/home/someone/secret sk-abcdefghijklmnopqrstuvwx", "model": "m", "ts": 1.0,
           "outcome": "error", "reason": "Bearer abcdefghijkl failed at /home/someone/x on quietbox",
           "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_s": 0.1,
           "reply_file": "replies/c1.txt", "prompt_sha256": "ab" * 32}
    (d / "telegraph.jsonl").write_text(json.dumps(rec) + "\n")
    app = app_for(root, cfg)
    app.hostname = "quietbox"
    app.home = "/home/someone"
    blob = json.dumps(app.telegraph())
    assert "sk-abcdefghij" not in blob and "Bearer abcdefghijkl" not in blob
    assert "/home/someone" not in blob and "quietbox" not in blob


def test_reply_files_cache_and_prompts_are_never_returned(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    tel.ask("p", "task " * 80, "failed")
    blob = json.dumps(app_for(root, cfg).telegraph())
    assert "use fill for constants" not in blob and "replies/" not in blob and "prompt_sha256" not in blob
    assert '"reply_file"' not in blob and "/cache/" not in blob


def test_a_ledger_that_changes_is_read_again(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=9, max_calls_per_problem=9)
    app = app_for(root, cfg)
    tel.ask("a", "task a " * 80, "failed")
    assert app.telegraph()["summary"]["calls"] == 1
    tel.ask("b", "task b " * 80, "failed")
    assert app.telegraph()["summary"]["calls"] == 2


# ---- the route ----------------------------------------------------------------------------------------------------

def test_the_route_answers_json_to_a_loopback_host_and_refuses_another_host(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    tel.ask("p", "task " * 80, "failed")
    with Served(app_for(root, cfg)) as s:
        code, headers, body = s.request("GET", "/api/telegraph")
        assert code == 200 and headers["Content-Type"].startswith("application/json")
        assert json.loads(body)["summary"]["calls"] == 1
        assert s.request("GET", "/api/telegraph", host="evil.example.com")[0] == 403


def test_there_is_no_way_to_post_to_it_and_the_view_makes_no_request(root, tmp_path, monkeypatch):
    cfg, d, tel = setup(tmp_path)
    sent = []
    monkeypatch.setattr(tg, "http_post", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(tg, "http_get", lambda *a, **k: sent.append(a))
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: sent.append(a))
    with Served(app_for(root, cfg)) as s:
        s.request("GET", "/api/telegraph")
        code, _, _ = s.post("/api/telegraph", {})
        assert code in (404, 405)
    assert sent == []
    assert not (d / "telegraph.jsonl").exists()                # nothing was written either


# ---- one run's own spend ----------------------------------------------------------------------------------------

def test_a_run_with_a_telegraph_ledger_shows_its_spend(root, tmp_path):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    cfg, d, tel = setup(root / "org--a")
    tel2 = tg.Telegraph(cfg, run, transport=reply(), environ={"TG_KEY": KEY}, clock=Clock(), hostname=HOST, home="/home/x")
    tel2.ask("p", "task " * 80, "failed")
    detail = app_for(root, cfg).run_detail("org--a")
    assert detail["telegraph"]["calls"] == 1 and detail["telegraph"]["cost_usd"] == pytest.approx(0.0035)
    assert detail["telegraph"]["unknown_cost"] == 0


def test_a_run_without_a_ledger_has_no_telegraph_field_value(root, tmp_path):
    write_ledger(root / "org--b", ("stage_start", 0, {}))
    cfg, d, _ = setup(tmp_path)
    assert app_for(root, cfg).run_detail("org--b")["telegraph"] is None


def test_a_linked_run_ledger_is_not_followed(root, tmp_path):
    run = write_ledger(root / "org--c", ("stage_start", 0, {}))
    outside = tmp_path / "outside.jsonl"
    outside.write_text(json.dumps({"call_id": "z", "outcome": "ok", "tokens_in": 1, "tokens_out": 1}) + "\n")
    (run / "telegraph.jsonl").symlink_to(outside)
    cfg, d, _ = setup(tmp_path)
    assert app_for(root, cfg).run_detail("org--c")["telegraph"] is None


# ---- money formatting ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,text", [(0, "$0.00"), (0.0035, "$0.0035"), (0.004, "$0.0040"), (0.5, "$0.50"),
                                        (12.349, "$12.35"), (0.01, "$0.01"), (0.0001, "$0.0001"),
                                        (0.00009999, "<$0.0001"), (0.00004, "<$0.0001"), (1e-9, "<$0.0001")])
def test_dollars_are_written_with_enough_digits_to_be_nonzero(value, text):
    assert view.usd(value) == text


def test_only_an_answered_call_shows_a_helped_verdict_even_if_a_refusal_was_reported(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=1, max_calls_per_problem=9)
    fill(tel, 1)
    refused = tel.ask("later", "another task " * 80, "failed")
    tel.report(refused.call_id, helped=True)                    # the producer accepts a report for any sent or refused call
    rows = app_for(root, cfg).telegraph()["recent"]
    assert rows[0]["outcome"] == "budget" and rows[0]["helped"] is None
