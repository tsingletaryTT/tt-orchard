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
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
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
                if FAULT == "two_names":
                    return self._send(200, {"object": "list", "data": [{"id": "listed-first", "max_model_len": length}, {"id": "alias-name", "max_model_len": length}]})
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
                return self._send(200, b'data: {"choices": [{"delta": {"content": "w"}}]}\\n\\ndata: [DONE]\\n\\n', "text/event-stream")
            if FAULT != "frozen_counter":
                TOKENS[0] += 5
            content = "" if FAULT == "spent_budget" else None if FAULT == "null_content" else "Paris"
            if FAULT == "two_names" and content:
                content = "Paris asked-for=" + str(req.get("model"))       # lets a test see which name the chat request carried
            finish = "length" if FAULT in ("spent_budget", "null_content") else "stop"
            self._send(200, {"model": "other-model" if FAULT == "wrong_echo" else req.get("model"),
                             "choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {"completion_tokens": 5}})
        def log_message(self, *a): pass
    if FAULT != "no_mesh_line":
        print("multidevice with %s devices and grid (1, 2) is created" % ("4" if FAULT == "wrong_mesh" else "2"), flush=True)
    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
''')


def bundle(tmp_path, fault=""):
    b = tmp_path / "bundle-src"
    b.mkdir()
    (b / "server.py").write_text(SERVER)
    (b / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (b / "run.sh").write_text(f'#!/usr/bin/env bash\nexport FAULT="{fault}"\nexec "{sys.executable}" "$(dirname "$0")/server.py"\n')
    return b


def spec_of(kind="vllm", max_model_len=4096):
    return can.Spec(name="toy", kind=kind, model_py=Path("m.py"), readme=Path("r.md"), repo="someone/toy", revision="0" * 40, device_count=2,
                    max_model_len=max_model_len, check=can.Check(expect="Paris", stream=False))


def boot(tmp_path, fault="", **kw):
    return can.boot_check(bundle(tmp_path, fault), tmp_path / "work", spec_of(**kw), install=False, health_timeout=30)


def test_a_healthy_server_passes_every_probe_and_leaves_nothing_unchecked(tmp_path):
    ev = boot(tmp_path)
    assert ev["passed"], ev
    names = set(ev["checks"])
    assert {"served_model", "reply_shape", "engine_counter", "arithmetic", "sequence", "mesh_devices"} <= names
    assert "not_checked" not in ev, ev["not_checked"]


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


# ---- issue 385: a --served-model-name alias that a spec sets only through extra_args ---------------------------------------------------------------

from dataclasses import replace

from orchard import served_verify


def with_args(*args, **check):
    sp = replace(spec_of(), extra_args=tuple(args))
    return replace(sp, check=replace(sp.check, **check)) if check else sp


def boot_spec(tmp_path, sp, fault="two_names"):
    return can.boot_check(bundle(tmp_path, fault), tmp_path / "work", sp, install=False, health_timeout=30)


@pytest.mark.parametrize("args,names", [
    (("--served-model-name", "a"), ["a"]),
    (("--served-model-name", "a", "b", "--max-num-seqs", "4"), ["a", "b"]),                 # argparse nargs="+": every word up to the next option
    (("--served_model_name=a", "b"), ["a"]),                                                # an `=` value is the only value; `b` is not a name
    (("--served-model-name", "a", "--other", "--served-model-name", "c"), ["c"]),            # the last occurrence wins, as argparse does
    (("--served-model-name", "a", "a", "b"), ["a", "b"]),                                   # a repeated name is listed once
    (("--served-model-name",), []),
    (("--max-model-len", "4096"), []),
    ((), []),
])
def test_served_names_reads_the_flag_the_way_vllm_does(args, names):
    assert served_verify.served_names(args) == names


def test_served_names_ignores_words_that_are_not_strings():
    assert served_verify.served_names(("--served-model-name", 5, None)) == []


def test_an_alias_set_only_in_extra_args_is_the_name_the_chat_request_carries(tmp_path):
    ev = boot_spec(tmp_path, with_args("--served-model-name", "alias-name"))
    assert ev["passed"], ev
    assert ev["checks"]["served_model"]["ok"] and "alias-name" in ev["checks"]["served_model"]["detail"]
    assert "alias-name" in ev["answer"] and "listed-first" not in ev["answer"], ev["answer"]


def test_without_a_declared_name_the_request_still_carries_the_first_listed_one(tmp_path):
    ev = boot_spec(tmp_path, with_args())
    assert ev["passed"], ev
    assert "listed-first" in ev["answer"], ev["answer"]


def test_a_listing_with_none_of_the_declared_names_fails_and_names_them(tmp_path):
    ev = boot_spec(tmp_path, with_args("--served-model-name", "gone-a", "gone-b"))
    check = ev["checks"]["served_model"]
    assert not ev["passed"] and check["ok"] is False
    assert "gone-a" in check["detail"] and "gone-b" in check["detail"] and "served-model-name" in check["detail"]


def test_a_request_extra_model_name_beats_the_alias_in_extra_args(tmp_path):
    ev = boot_spec(tmp_path, with_args("--served-model-name", "alias-name", request_extra={"model": "listed-first"}))
    assert ev["passed"], ev
    assert "listed-first" in ev["answer"] and "alias-name" not in ev["answer"], ev["answer"]


def listing(*ids):
    import json
    return json.dumps({"object": "list", "data": [{"id": i, "max_model_len": 4096} for i in ids]}).encode()


def test_some_declared_names_listed_and_some_not_passes_and_says_which_are_missing():
    r = served_verify.check_listing(200, listing("x", "alias-b"), declared_names=["alias-a", "alias-b"], declared_context=4096)
    assert r["state"] == "pass" and r["served"]["id"] == "alias-b"
    assert "alias-a" in r["detail"] and "not listed" in r["detail"]


def test_the_first_declared_name_that_is_listed_is_chosen_whatever_the_listing_order():
    r = served_verify.check_listing(200, listing("second", "first"), declared_names=["first", "second"], declared_context=4096)
    assert r["state"] == "pass" and r["served"]["id"] == "first"


def test_a_request_extra_name_is_used_alone_when_both_are_given():
    r = served_verify.check_listing(200, listing("a", "b"), declared_name="b", declared_names=["a"], declared_context=4096)
    assert r["state"] == "pass" and r["served"]["id"] == "b"
