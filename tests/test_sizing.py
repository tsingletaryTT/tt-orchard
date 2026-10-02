import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from orchard import sizing
from orchard.ledger import Ledger
from orchard.sizing import ceiling_tok_s, main, measure, theoretical_gb_s


class Fake(BaseHTTPRequestHandler):
    """Stands in for ollama's /api/generate and /api/ps."""

    def log_message(self, *a):
        pass

    def _send(self, obj, status=200, raw=None):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.server.seen = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.server.misbehave == "truncate":
            # Promise 1000 bytes, send 4, close: the client's read raises IncompleteRead.
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"a"')
            self.close_connection = True
            return
        if self.server.misbehave == "badstatus":
            self.wfile.write(b"NOT HTTP AT ALL\r\n\r\n")  # a status line http.client rejects
            self.close_connection = True
            return
        if self.server.hang_up:
            self.close_connection = True  # no reply at all: the client sees the connection drop
            return
        time.sleep(self.server.delay)
        self._send(self.server.generate_reply, self.server.status, self.server.raw)

    def do_GET(self):
        time.sleep(self.server.ps_delay)
        self._send(self.server.ps_reply)


@pytest.fixture
def fake():
    server = HTTPServer(("127.0.0.1", 0), Fake)
    server.generate_reply = {
        "load_duration": 3_000_000_000,
        "prompt_eval_count": 50, "prompt_eval_duration": 500_000_000,
        "eval_count": 100, "eval_duration": 2_000_000_000,
    }
    server.ps_reply = {"models": [{"name": "m:1", "size": 18_000_000_000}]}
    server.status = 200   # HTTP status for /api/generate
    server.raw = None     # raw bytes to send instead of JSON
    server.delay = 0      # seconds to wait before answering /api/generate
    server.ps_delay = 0   # seconds to wait before answering /api/ps
    server.misbehave = None  # "truncate" or "badstatus" for malformed /api/generate replies
    server.hang_up = False  # close the connection without answering /api/generate
    # A short poll interval keeps server.shutdown() from waiting half a second per test.
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    yield server
    server.shutdown()


def host(server):
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_ceiling_arithmetic():
    assert theoretical_gb_s(3600) == pytest.approx(57.6)
    assert theoretical_gb_s(5600) == pytest.approx(89.6)
    assert ceiling_tok_s(57.6, 1.8) == pytest.approx(32.0)


def test_measure_parses_ollama_timings(fake):
    r = measure(host(fake), "m:1", "hello", num_predict=100)
    assert r["decode_tok_s"] == pytest.approx(50.0)
    assert r["prefill_tok_s"] == pytest.approx(100.0)
    assert r["load_s"] == pytest.approx(3.0)
    assert r["resident_bytes"] == 18_000_000_000
    assert fake.seen["options"]["temperature"] == 0  # the same prompt gives the same tokens


def test_cached_prompt_reply_has_no_prefill_fields(fake):
    del fake.generate_reply["prompt_eval_count"], fake.generate_reply["prompt_eval_duration"]
    r = measure(host(fake), "m:1", "hello")
    assert r["prefill_tok_s"] is None and r["decode_tok_s"] == pytest.approx(50.0)


def test_model_missing_from_ps_gives_unknown_size(fake):
    fake.ps_reply = {"models": []}
    assert measure(host(fake), "m:1", "hello")["resident_bytes"] is None


def test_cli_records_a_measured_entry(fake, tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello")
    ledger = tmp_path / "ledger.jsonl"
    rc = main(["--host", host(fake), "--model", "m:1", "--prompt-file", str(prompt),
               "--ledger", str(ledger), "--gb-per-token", "1.8", "--note", "coder idle"])
    assert rc == 0
    with Ledger(ledger) as led:
        (entry,) = led.read()
    assert entry["event"] == "measurement" and entry["data"]["label"] == "measured"
    assert entry["data"]["decode_tok_s"] == pytest.approx(50.0)
    assert entry["data"]["ceiling_tok_s"] == pytest.approx(32.0)
    assert entry["data"]["note"] == "coder idle"


# ---- measure(): request body, field handling, guards ----

def test_request_body_pins_model_prompt_seed_and_num_predict(fake):
    measure(host(fake), "m:1", "hello", num_predict=77)
    body = fake.seen
    assert body["model"] == "m:1" and body["prompt"] == "hello" and body["stream"] is False
    assert body["options"] == {"num_predict": 77, "temperature": 0, "seed": 1}


def test_default_num_predict_is_128(fake):
    measure(host(fake), "m:1", "hello")
    assert fake.seen["options"]["num_predict"] == 128


def test_token_counts_and_model_are_reported(fake):
    r = measure(host(fake), "m:1", "hello")
    assert r["model"] == "m:1" and r["prompt_tokens"] == 50 and r["decode_tokens"] == 100


def test_zero_durations_give_no_rate_instead_of_dividing_by_zero(fake):
    fake.generate_reply["prompt_eval_duration"] = 0
    fake.generate_reply["eval_duration"] = 0
    r = measure(host(fake), "m:1", "hello")
    assert r["prefill_tok_s"] is None and r["decode_tok_s"] is None


def test_zero_decode_count_with_a_duration_is_a_rate_of_zero(fake):
    # Nothing was generated in a measured time: that is 0.0 tokens per second, not "unknown".
    fake.generate_reply["eval_count"] = 0
    assert measure(host(fake), "m:1", "hello")["decode_tok_s"] == 0.0


def test_zero_prefill_count_gives_no_prefill_rate(fake):
    fake.generate_reply["prompt_eval_count"] = 0
    assert measure(host(fake), "m:1", "hello")["prefill_tok_s"] is None


def test_missing_eval_fields_give_no_decode_rate(fake):
    del fake.generate_reply["eval_count"], fake.generate_reply["eval_duration"]
    r = measure(host(fake), "m:1", "hello")
    assert r["decode_tok_s"] is None and r["decode_tokens"] is None


def test_missing_load_duration_gives_unknown_load_time(fake):
    del fake.generate_reply["load_duration"]
    assert measure(host(fake), "m:1", "hello")["load_s"] is None


def test_resident_size_found_by_model_key_when_name_differs(fake):
    fake.ps_reply = {"models": [{"name": "other", "size": 1},
                                {"name": "alias", "model": "m:1", "size": 7}]}
    assert measure(host(fake), "m:1", "hello")["resident_bytes"] == 7


def test_resident_size_skips_other_models(fake):
    fake.ps_reply = {"models": [{"name": "other:2", "size": 5}]}
    assert measure(host(fake), "m:1", "hello")["resident_bytes"] is None


def test_ps_reply_without_models_key_gives_unknown_size(fake):
    fake.ps_reply = {}
    assert measure(host(fake), "m:1", "hello")["resident_bytes"] is None


# ---- failure paths in measure() ----

def test_http_error_raises_sizing_error_naming_the_status(fake):
    fake.status = 500
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello")
    assert str(exc.value).endswith("/api/generate answered HTTP 500")


def test_invalid_json_raises_sizing_error(fake):
    fake.raw = b"not json"
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello")
    assert str(exc.value).endswith("/api/generate did not return valid JSON")


def test_json_that_is_not_an_object_raises_sizing_error(fake):
    fake.raw = b"[1, 2]"
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello")
    assert str(exc.value).endswith("/api/generate did not return a JSON object")


def test_error_field_in_reply_raises_sizing_error(fake):
    fake.generate_reply = {"error": "model 'm:1' not found"}
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello")
    assert str(exc.value) == "ollama reported an error for m:1: model 'm:1' not found"


def test_unreachable_server_raises_sizing_error():
    # Port 1 on loopback has nothing listening, so the connection is refused at once.
    with pytest.raises(sizing.SizingError) as exc:
        measure("http://127.0.0.1:1", "m:1", "hello")
    assert str(exc.value).startswith("cannot reach http://127.0.0.1:1/api/generate: ")


def test_slow_server_raises_sizing_error_on_timeout(fake):
    fake.delay = 1.0
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello", timeout=0.2)
    assert str(exc.value).endswith("did not answer within 0.2 seconds")


# ---- main(): CLI behavior ----

def run_cli(fake, tmp_path, *extra, models=("m:1",)):
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello")
    argv = ["--host", host(fake), "--prompt-file", str(prompt), *extra]
    for m in models:
        argv += ["--model", m]
    return main(argv)


def test_cli_prints_the_result_as_json(fake, tmp_path, capsys):
    assert run_cli(fake, tmp_path) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["model"] == "m:1" and printed["decode_tok_s"] == pytest.approx(50.0)
    assert printed["theoretical_gb_s"] == pytest.approx(57.6)


def test_cli_leaves_the_ceiling_out_when_gb_per_token_is_not_given(fake, tmp_path, capsys):
    run_cli(fake, tmp_path)
    assert "ceiling_tok_s" not in json.loads(capsys.readouterr().out)


def test_cli_mt_s_changes_the_theoretical_bandwidth(fake, tmp_path, capsys):
    run_cli(fake, tmp_path, "--mt-s", "5600", "--gb-per-token", "1.12")
    printed = json.loads(capsys.readouterr().out)
    assert printed["theoretical_gb_s"] == pytest.approx(89.6)
    assert printed["ceiling_tok_s"] == pytest.approx(80.0)


def test_cli_note_defaults_to_empty_and_num_predict_is_passed_on(fake, tmp_path, capsys):
    run_cli(fake, tmp_path, "--num-predict", "9")
    assert json.loads(capsys.readouterr().out)["note"] == ""
    assert fake.seen["options"]["num_predict"] == 9


def test_cli_ledger_entry_has_no_ceiling_without_gb_per_token(fake, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    assert run_cli(fake, tmp_path, "--ledger", str(ledger)) == 0
    with Ledger(ledger) as led:
        (entry,) = led.read()
    assert "ceiling_tok_s" not in entry["data"]


def test_cli_measures_each_model_and_records_each(fake, tmp_path):
    fake.ps_reply = {"models": [{"name": "m:1", "size": 1}, {"name": "m:2", "size": 2}]}
    ledger = tmp_path / "ledger.jsonl"
    assert run_cli(fake, tmp_path, "--ledger", str(ledger), models=("m:1", "m:2")) == 0
    with Ledger(ledger) as led:
        entries = led.read()
    assert [e["data"]["model"] for e in entries] == ["m:1", "m:2"]
    assert [e["data"]["resident_bytes"] for e in entries] == [1, 2]


def test_cli_creates_the_ledger_directory_when_it_does_not_exist(fake, tmp_path):
    ledger = tmp_path / "new" / "deeper" / "ledger.jsonl"
    assert run_cli(fake, tmp_path, "--ledger", str(ledger)) == 0
    with Ledger(ledger) as led:
        assert len(led.read()) == 1


def test_cli_without_ledger_writes_nothing(fake, tmp_path):
    run_cli(fake, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.txt"]


def test_cli_unreachable_server_prints_one_line_and_exits_1(tmp_path, capsys):
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello")
    rc = main(["--host", "http://127.0.0.1:1", "--model", "m:1", "--prompt-file", str(prompt)])
    captured = capsys.readouterr()
    assert rc == 1 and captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1 and lines[0].startswith("sizing: cannot reach http://127.0.0.1:1")


def test_cli_http_error_exits_1_with_the_status(fake, tmp_path, capsys):
    fake.status = 404
    assert run_cli(fake, tmp_path) == 1
    assert capsys.readouterr().err.endswith("/api/generate answered HTTP 404\n")


def test_cli_missing_prompt_file_exits_1(fake, tmp_path, capsys):
    rc = main(["--host", host(fake), "--model", "m:1",
               "--prompt-file", str(tmp_path / "absent.txt")])
    err = capsys.readouterr().err
    assert rc == 1 and err.startswith("sizing: [Errno 2] No such file or directory")


def test_cli_locked_ledger_exits_1(fake, tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    with Ledger(ledger):  # another holder has the lock
        assert run_cli(fake, tmp_path, "--ledger", str(ledger)) == 1
    assert capsys.readouterr().err.startswith("sizing: another process holds ")


def test_cli_corrupt_ledger_exits_1(fake, tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("garbage\n")
    assert run_cli(fake, tmp_path, "--ledger", str(ledger)) == 1
    assert capsys.readouterr().err == "sizing: line 1 is not JSON\n"


def test_cli_timeout_flag_reaches_the_request(fake, tmp_path, capsys):
    fake.delay = 1.0
    assert run_cli(fake, tmp_path, "--timeout", "0.2") == 1
    assert "did not answer within 0.2 seconds" in capsys.readouterr().err


def test_cli_closes_the_ledger_when_a_measurement_fails(fake, tmp_path, monkeypatch):
    # Keep a reference to the Ledger main() opens: dropping the last reference would let the
    # garbage collector close it and hide a missing close(). While main()'s Ledger stays alive,
    # reopening the path only works if main() released the lock.
    opened = []
    real = Ledger

    def tracking(path):
        led = real(path)
        opened.append(led)
        return led

    orig = sizing.measure
    calls = []

    def flaky(h, model, *a, **k):
        calls.append(model)
        if len(calls) == 2:
            raise sizing.SizingError("boom")
        return orig(h, model, *a, **k)

    monkeypatch.setattr(sizing, "Ledger", tracking)
    monkeypatch.setattr(sizing, "measure", flaky)
    ledger = tmp_path / "l.jsonl"
    assert run_cli(fake, tmp_path, "--ledger", str(ledger), models=("m:1", "m:2")) == 1
    assert len(opened) == 1
    with real(ledger) as led:  # raises LedgerLocked if main() left the lock held
        (entry,) = led.read()
    assert entry["data"]["model"] == "m:1"


def test_cli_keeps_earlier_results_when_a_later_model_fails(fake, tmp_path, capsys, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    # The fake answers every model, so make the second measurement fail on purpose.
    orig = sizing.measure
    calls = []

    def flaky(h, model, *a, **k):
        calls.append(model)
        if len(calls) == 2:
            raise sizing.SizingError("boom")
        return orig(h, model, *a, **k)

    monkeypatch.setattr(sizing, "measure", flaky)
    rc = run_cli(fake, tmp_path, "--ledger", str(ledger), "--num-predict", "100",
                 models=("m:1", "m:2"))
    assert rc == 1 and capsys.readouterr().err == "sizing: boom\n"
    with Ledger(ledger) as led:
        (entry,) = led.read()
    assert entry["data"]["model"] == "m:1"


def test_slow_ps_reply_raises_sizing_error_on_timeout(fake):
    # /api/generate answers at once, so only the /api/ps request can hit the timeout.
    fake.ps_delay = 1.0
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello", timeout=0.2)
    assert str(exc.value).endswith("/api/ps did not answer within 0.2 seconds")


def test_dropped_connection_raises_sizing_error(fake):
    fake.hang_up = True
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello")
    assert str(exc.value).startswith("connection to ") and "/api/generate failed: " in str(exc.value)


def test_cli_sends_the_prompt_file_text(fake, tmp_path):
    run_cli(fake, tmp_path)
    assert fake.seen["prompt"] == "hello"


def test_cli_num_predict_defaults_to_128(fake, tmp_path):
    run_cli(fake, tmp_path)
    assert fake.seen["options"]["num_predict"] == 128


def test_cli_does_not_open_a_ledger_when_none_is_asked_for(fake, tmp_path, monkeypatch):
    def forbidden(path):
        raise AssertionError("a ledger was opened without --ledger")

    monkeypatch.setattr(sizing, "Ledger", forbidden)
    assert run_cli(fake, tmp_path) == 0


# ---- resident-size name matching ----

def test_name_without_tag_matches_a_latest_entry(fake, capsys):
    fake.ps_reply = {"models": [{"name": "qwen3:latest", "size": 5}]}
    assert measure(host(fake), "qwen3", "hello", num_predict=100)["resident_bytes"] == 5
    assert capsys.readouterr().err == ""


def test_name_with_latest_matches_an_untagged_entry(fake):
    fake.ps_reply = {"models": [{"name": "qwen3", "size": 6}]}
    assert measure(host(fake), "qwen3:latest", "hello")["resident_bytes"] == 6


def test_latest_normalisation_also_applies_to_the_model_field(fake):
    fake.ps_reply = {"models": [{"name": "alias", "model": "qwen3:latest", "size": 8}]}
    assert measure(host(fake), "qwen3", "hello")["resident_bytes"] == 8


def test_absent_model_warns_naming_what_ollama_lists(fake, capsys):
    fake.ps_reply = {"models": [{"name": "other:2", "size": 5}, {"model": "third:1", "size": 6}]}
    r = measure(host(fake), "m:1", "hello", num_predict=100)
    assert r["resident_bytes"] is None
    assert capsys.readouterr().err == (
        "sizing: warning: model m:1 is not in ollama's /api/ps list (lists: other:2, third:1); "
        "resident_bytes recorded as null\n")


def test_absent_model_with_an_empty_list_says_nothing_is_loaded(fake, capsys):
    fake.ps_reply = {"models": []}
    measure(host(fake), "m:1", "hello", num_predict=100)
    assert "(lists: nothing)" in capsys.readouterr().err


# ---- decode completeness ----

def test_done_reason_length_marks_the_decode_complete(fake, capsys):
    fake.generate_reply["done_reason"] = "length"  # 100 tokens < num_predict 128, but length-capped
    r = measure(host(fake), "m:1", "hello", num_predict=128)
    assert r["done_reason"] == "length" and r["decode_complete"] is True
    assert capsys.readouterr().err == ""


def test_stop_with_fewer_tokens_than_num_predict_is_a_short_decode(fake, capsys):
    fake.generate_reply["done_reason"] = "stop"
    r = measure(host(fake), "m:1", "hello", num_predict=128)
    assert r["done_reason"] == "stop" and r["decode_complete"] is False
    assert capsys.readouterr().err == (
        "sizing: warning: decode rate came from a short run (100 of 128 tokens, done_reason stop); "
        "the prompt may need to be longer\n")


def test_token_count_reaching_num_predict_marks_the_decode_complete(fake):
    r = measure(host(fake), "m:1", "hello", num_predict=100)
    assert r["done_reason"] is None and r["decode_complete"] is True


def test_missing_decode_count_is_not_a_complete_decode(fake):
    del fake.generate_reply["eval_count"]
    assert measure(host(fake), "m:1", "hello", num_predict=100)["decode_complete"] is False


# ---- cold load and cached prefill labels ----

def test_load_over_the_threshold_is_cold(fake):
    assert measure(host(fake), "m:1", "hello")["load_cold"] is True


def test_load_under_the_threshold_is_not_cold(fake):
    fake.generate_reply["load_duration"] = 100_000_000  # 0.1 s: the model was already resident
    r = measure(host(fake), "m:1", "hello")
    assert r["load_cold"] is False and r["load_s"] == pytest.approx(0.1)


def test_load_exactly_at_the_threshold_is_cold(fake):
    assert sizing.COLD_LOAD_THRESHOLD_S == 0.5
    fake.generate_reply["load_duration"] = 500_000_000
    assert measure(host(fake), "m:1", "hello")["load_cold"] is True


def test_unknown_load_time_leaves_load_cold_unknown(fake):
    del fake.generate_reply["load_duration"]
    assert measure(host(fake), "m:1", "hello")["load_cold"] is None


def test_prompt_size_and_hash_are_recorded(fake):
    import hashlib
    r = measure(host(fake), "m:1", "hello")
    assert r["prompt_chars"] == 5
    assert r["prompt_sha256"] == hashlib.sha256(b"hello").hexdigest()


def test_prompt_hash_counts_bytes_not_characters(fake):
    import hashlib
    r = measure(host(fake), "m:1", "h\u00e9")
    assert r["prompt_chars"] == 2
    assert r["prompt_sha256"] == hashlib.sha256("h\u00e9".encode("utf-8")).hexdigest()


LONG_PROMPT = "x" * 400  # estimate: 400 / 4 = 100 tokens, so half is 50


def test_prefill_count_near_the_estimate_is_a_real_prefill(fake):
    fake.generate_reply["prompt_eval_count"] = 50  # exactly half of the estimate: still real
    r = measure(host(fake), "m:1", LONG_PROMPT)
    assert r["prefill_cached"] is False and r["prefill_tok_s"] == pytest.approx(100.0)


def test_prefill_count_under_half_the_estimate_is_cached(fake):
    fake.generate_reply["prompt_eval_count"] = 49
    r = measure(host(fake), "m:1", LONG_PROMPT)
    assert r["prefill_cached"] is True and r["prefill_tok_s"] is None
    assert r["prompt_tokens"] == 49  # the count is still reported


def test_missing_prefill_count_is_cached(fake):
    del fake.generate_reply["prompt_eval_count"]
    r = measure(host(fake), "m:1", LONG_PROMPT)
    assert r["prefill_cached"] is True and r["prefill_tok_s"] is None


# ---- MemAvailable ----

def test_mem_available_is_read_in_bytes(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  100 kB\nMemAvailable:   2048 kB\nBuffers: 1 kB\n")
    monkeypatch.setattr(sizing, "MEMINFO_PATH", str(meminfo))
    assert sizing.mem_available_bytes() == 2048 * 1024


def test_mem_available_is_none_when_the_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sizing, "MEMINFO_PATH", str(tmp_path / "absent"))
    assert sizing.mem_available_bytes() is None


def test_mem_available_is_none_when_the_line_is_missing(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  100 kB\n")
    monkeypatch.setattr(sizing, "MEMINFO_PATH", str(meminfo))
    assert sizing.mem_available_bytes() is None


def test_mem_available_is_none_when_the_value_is_malformed(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: lots kB\n")
    monkeypatch.setattr(sizing, "MEMINFO_PATH", str(meminfo))
    assert sizing.mem_available_bytes() is None


def test_mem_available_is_none_when_the_value_is_missing(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:\n")
    monkeypatch.setattr(sizing, "MEMINFO_PATH", str(meminfo))
    assert sizing.mem_available_bytes() is None


def test_measure_records_mem_available(fake, tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 10 kB\n")
    monkeypatch.setattr(sizing, "MEMINFO_PATH", str(meminfo))
    assert measure(host(fake), "m:1", "hello")["mem_available_bytes"] == 10 * 1024


# ---- malformed replies (http.client exceptions) ----

def test_truncated_body_raises_sizing_error(fake):
    fake.misbehave = "truncate"
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello")
    assert str(exc.value).endswith("/api/generate sent a malformed or truncated reply (IncompleteRead)")


def test_malformed_status_line_raises_sizing_error(fake):
    fake.misbehave = "badstatus"
    with pytest.raises(sizing.SizingError) as exc:
        measure(host(fake), "m:1", "hello")
    assert str(exc.value).endswith("/api/generate sent a malformed or truncated reply (BadStatusLine)")


def test_cli_truncated_reply_exits_1_without_a_traceback(fake, tmp_path, capsys):
    fake.misbehave = "truncate"
    assert run_cli(fake, tmp_path) == 1
    err = capsys.readouterr().err
    assert err.startswith("sizing: ") and err.count("\n") == 1 and "Traceback" not in err


# ---- CLI: ledger fields and help text ----

def test_cli_ledger_entry_records_prompt_identity_and_free_memory(fake, tmp_path):
    import hashlib
    ledger = tmp_path / "ledger.jsonl"
    assert run_cli(fake, tmp_path, "--ledger", str(ledger)) == 0
    with Ledger(ledger) as led:
        (entry,) = led.read()
    d = entry["data"]
    assert d["prompt_chars"] == 5
    assert d["prompt_sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert "mem_available_bytes" in d and d["load_cold"] is True and d["prefill_cached"] is False


def test_help_says_load_time_needs_a_cold_run_and_timeout_is_per_request(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "load_s is only meaningful for a cold first run" in text
    assert "applies to each request" in text and "longer in total" in text
