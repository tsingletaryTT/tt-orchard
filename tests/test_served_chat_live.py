# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 368 and 361: the stage 2 and 4 served check judges the live-probe record that serve_and_compare.py writes, and summarises reply tokens.

The template cannot import orchard, so it records raw answers (`live_probe` in swap-check.json) and orchard/served_chat.py judges them with the
rules of orchard/served_verify.py. The template runs against a local fake server here; no chip and no model.
"""
import copy
import http.client
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from orchard import served_chat, served_verify
from test_weights_swap_templates import report, swap  # noqa: E402,F401  (a fixture)

TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates" / "serve_and_compare.py"
MODEL = "/stage/model-dir"


def good():
    """A live_probe record from a healthy server."""
    return {"ran": True, "model_id": "org/new", "model": MODEL, "declared_max_model_len": 4096,
            "listing": {"status": 200, "body": {"object": "list", "data": [{"id": MODEL, "max_model_len": 4096}]}},
            "completions": {"arithmetic": {"status": 200, "body": {"choices": [{"text": "408"}], "usage": {"completion_tokens": 3}}},
                            "sequence": {"status": 200, "body": {"choices": [{"text": " 9 10 11"}], "usage": {"completion_tokens": 4}}}},
            "metrics": {"before": ['vllm:generation_tokens_total{engine="0"} 10.0'], "after": ['vllm:generation_tokens_total{engine="0"} 13.0']}}


def states(rec):
    return {r["name"]: r["state"] for r in served_chat.check_live_probe(rec)["results"]}


def test_a_healthy_record_passes_every_probe():
    out = served_chat.check_live_probe(good())
    assert out["state"] == "pass" and out["problems"] == [] and out["not_checked"] == [], out
    assert states(good()) == {"served_model": "pass", "arithmetic": "pass", "sequence": "pass", "engine_counter": "pass"}


@pytest.mark.parametrize("record", [None, {}, {"ran": False}, "x", [], 3])
def test_a_swap_check_from_before_the_probe_is_not_checked_and_never_a_pass(record):
    out = served_chat.check_live_probe(record)
    assert out["state"] == "not_checked" and out["results"] == [] and out["problems"] == [] and "before the live probe" in out["not_checked"][0]


def test_a_served_name_that_differs_from_the_name_asked_for_fails():
    rec = good()
    rec["listing"]["body"]["data"][0]["id"] = "/stage/other-model"
    out = served_chat.check_live_probe(rec)
    assert out["state"] == "fail" and out["problems"][0]["name"] == "served_model" and MODEL in out["problems"][0]["detail"], out


def test_a_context_that_differs_from_the_declared_max_model_len_fails():
    rec = good()
    rec["listing"]["body"]["data"][0]["max_model_len"] = 2048
    out = served_chat.check_live_probe(rec)
    assert out["state"] == "fail" and "2048" in out["problems"][0]["detail"] and "4096" in out["problems"][0]["detail"], out


def test_no_declared_context_reports_the_context_and_does_not_compare_it():
    rec = good()
    rec["declared_max_model_len"] = None
    r = served_chat.check_live_probe(rec)["results"][0]
    assert r["state"] == "pass" and "not compared" in r["detail"]


@pytest.mark.parametrize("bad", [0, -1, True, "4096", 1.5, [4096]])
def test_a_declared_length_of_the_wrong_type_is_ignored_not_trusted(bad):
    rec = good()
    rec["declared_max_model_len"] = bad
    assert states(rec)["served_model"] == "pass"


def test_a_wrong_product_and_a_repeated_token_fail_in_the_recorded_answers_too():
    rec = good()
    rec["completions"]["arithmetic"]["body"]["choices"][0]["text"] = "407"
    rec["completions"]["sequence"]["body"]["choices"][0]["text"] = " 9 9 10"
    assert states(rec)["arithmetic"] == "fail" and states(rec)["sequence"] == "fail"


def test_a_request_with_no_recorded_answer_fails_rather_than_passes():
    rec = good()
    rec["listing"] = {"status": None, "error": "URLError: refused"}
    rec["completions"]["arithmetic"] = {"status": None, "error": "timed out"}
    del rec["completions"]["sequence"]
    s = states(rec)
    assert (s["served_model"], s["arithmetic"], s["sequence"]) == ("fail", "fail", "fail")


def test_a_route_the_server_lacks_is_not_checked_and_does_not_fail():
    rec = good()
    rec["completions"]["arithmetic"] = {"status": 404, "body": None}
    rec["completions"]["sequence"] = {"status": 404, "body": None}
    rec["metrics"] = {"before": None, "after": None}
    out = served_chat.check_live_probe(rec)
    assert out["state"] == "pass" and len(out["not_checked"]) == 3, out


def test_a_counter_that_did_not_rise_fails():
    rec = good()
    rec["metrics"]["after"] = rec["metrics"]["before"]
    assert states(rec)["engine_counter"] == "fail"


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(listing="x"), lambda r: r.update(completions="x"), lambda r: r.update(metrics="x"),
    lambda r: r.update(model=3), lambda r: r.update(listing={"status": "200", "body": {}}), lambda r: r["listing"].update(body=[1]),
    lambda r: r["completions"]["arithmetic"].update(body="x"), lambda r: r["metrics"].update(before=[None, 3]),
])
def test_a_record_of_the_wrong_shape_gives_a_result_and_never_an_exception(mutate):
    rec = good()
    mutate(rec)
    out = served_chat.check_live_probe(rec)
    assert out["state"] in ("pass", "fail", "not_checked")


def test_a_failed_live_probe_makes_the_stage_record_not_ok_and_a_missing_one_does_not(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    swap = tmp_path / "swap-check.json"
    base = {"prompt_suite": {"ran": False}}
    swap.write_text(json.dumps(base))
    rec = served_chat.build_record(served, served, swap)
    assert rec["live_probe"]["state"] == "not_checked" and all(not p["name"].startswith("live_probe_") for p in rec["problems"])
    bad = good()
    bad["listing"]["body"]["data"][0]["id"] = "other"
    swap.write_text(json.dumps({**base, "live_probe": bad}))
    rec = served_chat.build_record(served, served, swap)
    assert rec["live_probe"]["state"] == "fail" and rec["ok"] is False
    assert [p["name"] for p in rec["problems"] if p["name"].startswith("live_probe_")] == ["live_probe_served_model"]


# ---- 361: reply tokens ------------------------------------------------------------------------------------------------------------

def reply(i, completion, reasoning=None, answer=None, finish="stop", text="fine"):
    return {"id": f"p{i}", "mode": "chat", "content": text, "reasoning": "r" if reasoning else "", "tool_calls": [], "finish_reason": finish,
            "error": None, "completion_tokens": completion, "reasoning_tokens": reasoning, "answer_tokens": answer}


def test_the_replies_report_a_distribution_and_a_cap_the_counts_support():
    suite = {"ran": True, "mode": "chat", "max_tokens": 512, "entries": [reply(1, 100, 80, 20), reply(2, 400, 380, 20), reply(3, 50, 0, 50)]}
    t = served_chat.check_replies(suite)["tokens"]
    assert t["completion_tokens"] == {"n": 3, "min": 50, "median": 100, "max": 400}
    assert t["reasoning_tokens"]["max"] == 380 and t["answer_tokens"]["max"] == 50 and t["replies_with_reasoning"] == 2
    cap = t["cap_evidence"]
    assert cap["largest_reply_tokens"] == 400 and cap["supported_cap"] == 600 and cap["suite_limit"] == 512
    assert cap["suite_limit_below_supported_cap"] is True


def test_a_suite_with_no_reasoning_gives_no_cap_evidence():
    suite = {"ran": True, "mode": "chat", "entries": [reply(1, 10, 0, 10), reply(2, 12, None, None)]}
    t = served_chat.check_replies(suite)["tokens"]
    assert "cap_evidence" not in t and t["replies_with_reasoning"] == 0


def test_a_record_from_before_the_counts_gives_empty_distributions():
    old = {"id": "a", "mode": "chat", "content": "x", "reasoning": "", "tool_calls": [], "finish_reason": "stop", "error": None}
    t = served_chat.check_replies({"ran": True, "mode": "chat", "entries": [old]})["tokens"]
    assert t["completion_tokens"] is None and t["reasoning_tokens"] is None and "cap_evidence" not in t


# ---- the template writes the record without importing orchard ---------------------------------------------------------------------

def load_template():
    spec = importlib.util.spec_from_file_location("sac_live_probe", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Server:
    """A local fake: /v1/models, /v1/completions, /metrics. `mode` switches what it does."""

    def __init__(self, mode="ok"):
        self.mode, self.tokens = mode, 0.0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self.send(200, {"data": [{"id": MODEL, "max_model_len": 4096}]})
                if self.path == "/metrics":
                    if outer.mode == "no_metrics":
                        return self.send(404, b"no")
                    return self.send(200, ("# HELP x\nvllm:other 5\nvllm:generation_tokens_total{engine=\"0\"} %s\n" % outer.tokens).encode())
                self.send(404, b"no")

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/v1/completions" or outer.mode == "no_completions":
                    return self.send(404, b"no")
                if outer.mode == "error500":
                    return self.send(500, b"boom")
                outer.tokens += 3
                text = "408" if req["prompt"].startswith("17") else " 9 10 11"
                self.send(200, {"choices": [{"text": text}], "usage": {"completion_tokens": 3}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(request):
    s = Server(getattr(request, "param", "ok"))
    yield s
    s.close()


def test_the_template_records_the_raw_answers_and_orchard_judges_them_as_a_pass(server):
    sac = load_template()
    rec = sac.run_live_probe({"new_model_id": "org/new"}, server.port, MODEL, "exec vllm serve --max-model-len 4096 --port 1\n")
    assert rec["ran"] is True and rec["model"] == MODEL and rec["declared_max_model_len"] == 4096
    assert rec["listing"]["status"] == 200 and set(rec["completions"]) == {"arithmetic", "sequence"}
    assert rec["metrics"] == {"before": ['vllm:generation_tokens_total{engine="0"} 0.0'], "after": ['vllm:generation_tokens_total{engine="0"} 3.0']}
    out = served_chat.check_live_probe(json.loads(json.dumps(rec)))
    assert out["state"] == "pass", out


@pytest.mark.parametrize("server,name", [("no_metrics", "engine_counter"), ("no_completions", "arithmetic")], indirect=["server"])
def test_a_route_the_server_lacks_is_recorded_and_judged_not_checked(server, name):
    rec = load_template().run_live_probe({}, server.port, MODEL, None)
    out = served_chat.check_live_probe(json.loads(json.dumps(rec)))
    assert out["state"] == "pass" and any(n.startswith(name) for n in out["not_checked"]), out


@pytest.mark.parametrize("server", ["error500"], indirect=True)
def test_a_server_error_is_recorded_and_judged_a_failure(server):
    rec = load_template().run_live_probe({}, server.port, MODEL, None)
    assert served_chat.check_live_probe(rec)["state"] == "fail"


def test_a_dead_server_gives_a_record_of_errors_never_an_exception(server):
    server.close()
    rec = load_template().run_live_probe({}, server.port, MODEL, None)
    assert rec["listing"]["status"] is None and "error" in rec["listing"] and rec["metrics"]["before"] is None
    assert served_chat.check_live_probe(rec)["state"] == "fail"


def test_the_template_asks_the_same_two_prompts_as_the_judge():
    assert load_template().LIVE_COMPLETIONS == served_verify.COMPLETION_PROBES
    assert load_template().LIVE_COUNTER == served_verify.COUNTER


@pytest.mark.parametrize("text,want", [("vllm serve --max-model-len 4096", 4096), ("--max_model_len=8192 --x", 8192), ("--max-model-len 1 --max-model-len 2", 2),
                                       ("--max-model-len abc", None), ("", None), (None, None), ("--max-model-lens 5", None)])
def test_the_declared_context_is_read_from_run_sh_text(text, want):
    assert load_template().declared_max_model_len(text) == want


def test_a_body_larger_than_the_bound_is_cut_not_buffered(server, monkeypatch):
    sac = load_template()
    monkeypatch.setattr(sac, "LIVE_BODY_BYTES", 10)
    out = sac.live_request(server.port, "/v1/models")
    assert out["status"] == 200 and out["body"] is None          # ten bytes of JSON do not parse; nothing larger was read


def test_the_template_imports_nothing_from_orchard():
    import re
    assert not re.search(r"^\s*(?:from|import)\s+orchard\b", TEMPLATE.read_text(), re.MULTILINE)


def test_a_suite_reply_records_its_tokens_with_the_tokenizer(server):
    sac = load_template()

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return type("E", (), {"ids": list(range(len(text.split())))})()
    assert sac.token_count(Tok(), "a b c") == 3 and sac.token_count(Tok(), "") == 0 and sac.token_count(None, "a b") is None

    class Broken:
        def encode(self, *a, **k):
            raise RuntimeError("x")
    assert sac.token_count(Broken(), "a") is None


# ---- the whole template, on the repo's fake swap server -----------------------------------------------------------------------------

def test_the_whole_template_writes_a_live_probe_record_by_default_and_none_when_switched_off(swap):
    r = swap["start"]("perfect")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["live_probe"]
    assert rec["ran"] is True and rec["model"] == str(swap["stage"] / "model-dir") and rec["listing"]["status"] == 200
    assert set(rec["completions"]) == {"arithmetic", "sequence"} and rec["metrics"]["before"] is None      # the fake has no /metrics
    out = served_chat.check_live_probe(rec)
    assert out["results"][0]["name"] == "served_model" and out["results"][0]["state"] == "pass", out         # the fake lists the model dir it was given
    r = swap["start"]("perfect", live_probe=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "live_probe" not in report(swap)
