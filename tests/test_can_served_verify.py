# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`can boot` runs the live-server probes of orchard/served_verify.py (listing, reply shape, raw completions, engine counter, mesh line).

The stand-in server below talks like vLLM: it lists a model with a context length, answers chat and raw completions, counts the tokens it
generates on /metrics and logs its mesh. An environment variable makes it fail in one way at a time, so each test shows that the probe
turns a real fault into a failed check and that a healthy server passes with no `not_checked` entry for what it offers. No chip is used.
"""
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import can

SERVER = textwrap.dedent('''
    import json, os, sys, threading, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    LOCK = threading.Lock()
    FAULT = os.environ.get("FAULT", "")
    TOKENS = [0]
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == "/health":
                return self._send(200, b"", "text/plain")
            if self.path == "/metrics":
                if FAULT == "no_metrics":
                    return self._send(404, b"no")
                return self._send(200, ('vllm:generation_tokens_total{engine="0"} %d\\n' % TOKENS[0]).encode(), "text/plain")
            if self.path == "/v1/models":
                if FAULT == "empty_listing":
                    return self._send(200, {"object": "list", "data": []})
                length = 2048 if FAULT == "wrong_context" else 4096
                return self._send(200, {"object": "list", "data": [{"id": "served-name", "max_model_len": length}]})
            self._send(404, {"error": "no"})
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
            if self.path == "/v1/completions":
                if FAULT == "no_completions":
                    return self._send(404, {"error": "no"})
                text = {"17*24=": "408", "1 2 3 4 5 6 7 8": " 9 10 11"}[req["prompt"]]
                if FAULT == "bad_math" and req["prompt"] == "17*24=":
                    text = "407"
                if FAULT == "repeat" and req["prompt"].startswith("1 2"):
                    text = " 9 9 10 10 1"
                return self._send(200, {"choices": [{"text": text, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}})
            if req.get("stream"):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for i in range(40):
                    self.wfile.write(b'data: {"choices": [{"delta": {"content": "w"}}]}\\n\\n'); self.wfile.flush(); time.sleep(0.01)
                self.wfile.write(b'data: {"choices": [], "usage": {"completion_tokens": 40}}\\n\\ndata: [DONE]\\n\\n'); self.wfile.flush()
                return
            prompt = req["messages"][0]["content"]
            if req.get("ignore_eos"):                      # the batching burst: 32 tokens at the stream's rate of about 100 tokens a second
                if FAULT == "serialise":
                    with LOCK:
                        time.sleep(0.32)
                else:
                    time.sleep(0.32)
                return self._send(200, {"choices": [{"message": {"content": "1 2 3"}, "finish_reason": "length"}], "usage": {"completion_tokens": 32, "prompt_tokens": 12}})
            if "is_prime" in prompt:
                code = "def is_prime(n" if FAULT == "bad_code" else "def is_prime(n):\\n    return n > 1 and all(n % d for d in range(2, n))\\n"
                return self._send(200, {"choices": [{"message": {"content": code}, "finish_reason": "stop"}], "usage": {"completion_tokens": 20, "prompt_tokens": 30}})
            if "secret word" in prompt:
                time.sleep(0.15)                           # a long prompt takes the same time every time here: no cache (the prefix probe judges by timing)
                return self._send(200, {"choices": [{"message": {"content": "tangerine"}, "finish_reason": "stop"}],
                                        "usage": {"completion_tokens": 2, "prompt_tokens": int(len(prompt.split()) * 1.1)}})
            if req.get("max_tokens") == 1:
                return self._send(200, {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}], "usage": {"completion_tokens": 1, "prompt_tokens": 12}})
            if FAULT != "frozen_counter":
                TOKENS[0] += 5
            content = "" if FAULT == "spent_budget" else None if FAULT == "null_content" else "Paris"
            finish = "length" if FAULT in ("spent_budget", "null_content") else "stop"
            self._send(200, {"model": "other-model" if FAULT == "wrong_echo" else req.get("model"),
                             "choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {"completion_tokens": 5}})
        def log_message(self, *a): pass
    print("INFO loaded %s layers on 2 devices" % ("12" if FAULT == "wrong_layers" else "36"), flush=True)
    print("INFO GPU KV cache size: %s tokens" % ("1,024" if FAULT == "small_kv" else "100,000"), flush=True)
    print("INFO Maximum concurrency for 4,096 tokens per request: %sx" % ("0.25" if FAULT == "small_kv" else "24.41"), flush=True)
    print("INFO Avg generation throughput: 0.0 tokens/s, Running: 0 reqs", flush=True)
    if FAULT == "warn":
        print("WARNING Unrecognised device name P150b; falling back to the generic profile", flush=True)
    if FAULT != "no_mesh_line":
        print("multidevice with %s devices and grid (1, 2) is created" % ("4" if FAULT == "wrong_mesh" else "2"), flush=True)
    ThreadingHTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
''')


def bundle(tmp_path, fault=""):
    b = tmp_path / "bundle-src"
    b.mkdir()
    (b / "server.py").write_text(SERVER)
    (b / "base_config").mkdir()
    (b / "base_config" / "config.json").write_text('{"num_hidden_layers": 36}')
    (b / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (b / "run.sh").write_text(f'#!/usr/bin/env bash\nexport FAULT="{fault}"\nexec "{sys.executable}" "$(dirname "$0")/server.py"\n')
    return b


def spec_of(kind="vllm", max_model_len=4096, **more):
    check = more.pop("check", can.Check(expect="Paris", stream=False))
    return can.Spec(name="toy", kind=kind, model_py=Path("m.py"), readme=Path("r.md"), repo="someone/toy", revision="0" * 40, device_count=2,
                    max_model_len=max_model_len, check=check, **more)


def boot(tmp_path, fault="", **kw):
    return can.boot_check(bundle(tmp_path, fault), tmp_path / "work", spec_of(**kw), install=False, health_timeout=30)


def test_a_healthy_server_passes_every_probe_and_leaves_nothing_unchecked(tmp_path):
    ev = boot(tmp_path)
    assert ev["passed"], ev
    names = set(ev["checks"])
    assert {"served_model", "reply_shape", "engine_counter", "arithmetic", "sequence", "mesh_devices", "code_parse", "long_prefill",
            "layers_loaded", "kv_cache"} <= names
    assert [n["name"] for n in ev["not_checked"]] == ["batching"], ev["not_checked"]      # the stand-in's can declares no max_num_seqs above 1
    assert ev["log_warnings"]["lines"] == [] and ev["log_warnings"]["dropped_as_noise"] == 0 or ev["log_warnings"]["lines"] == []


@pytest.mark.parametrize("fault,failed", [
    ("empty_listing", "served_model"), ("wrong_context", "served_model"), ("wrong_echo", "reply_shape"), ("spent_budget", "reply_shape"),
    ("frozen_counter", "engine_counter"), ("bad_math", "arithmetic"), ("repeat", "sequence"), ("wrong_mesh", "mesh_devices"),
])
def test_each_fault_fails_its_own_check_and_the_boot(tmp_path, fault, failed):
    ev = boot(tmp_path, fault)
    assert not ev["passed"]
    assert ev["checks"][failed]["ok"] is False, ev["checks"]
    others = [n for n in ("served_model", "reply_shape", "engine_counter", "arithmetic", "sequence", "mesh_devices") if n != failed]
    assert all(ev["checks"][n]["ok"] for n in others if n in ev["checks"]), ev["checks"]


@pytest.mark.parametrize("fault,missing", [("no_metrics", "engine_counter"), ("no_completions", "arithmetic"), ("no_mesh_line", "mesh_devices")])
def test_a_route_the_server_does_not_offer_is_listed_as_not_checked_and_does_not_fail_the_boot(tmp_path, fault, missing):
    ev = boot(tmp_path, fault)
    assert ev["passed"], ev
    assert missing not in ev["checks"]
    assert missing in [n["name"] for n in ev["not_checked"]]


def test_the_probes_do_not_run_for_the_dit_kind_which_serves_no_model_list(tmp_path):
    ev = boot(tmp_path, "empty_listing", kind="tt-dit-server")
    assert not {"served_model", "reply_shape", "engine_counter", "arithmetic", "sequence", "mesh_devices"} & set(ev["checks"]), ev["checks"]


def test_a_can_that_declares_no_context_length_reports_it_was_not_compared(tmp_path):
    ev = boot(tmp_path, "wrong_context", max_model_len=None)
    assert ev["checks"]["served_model"]["ok"]
    assert "not compared" in ev["checks"]["served_model"]["detail"]


def test_null_content_with_finish_reason_length_writes_a_failed_record_and_names_the_budget(tmp_path):
    ev = boot(tmp_path, "null_content")
    assert not ev["passed"] and (tmp_path / "work" / "boot-check.json").is_file()
    assert "token budget" in ev["checks"]["reply_shape"]["detail"]


def test_a_request_extra_model_name_is_the_name_the_listing_is_checked_for(tmp_path):
    from dataclasses import replace
    sp = spec_of()
    sp = replace(sp, check=replace(sp.check, request_extra={"model": "not-served"}))
    ev = can.boot_check(bundle(tmp_path), tmp_path / "work", sp, install=False, health_timeout=30)
    assert ev["checks"]["served_model"]["ok"] is False and "not-served" in ev["checks"]["served_model"]["detail"]
