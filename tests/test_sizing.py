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


def test_zero_token_counts_give_no_rate(fake):
    fake.generate_reply["prompt_eval_count"] = 0
    fake.generate_reply["eval_count"] = 0
    r = measure(host(fake), "m:1", "hello")
    assert r["prefill_tok_s"] is None and r["decode_tok_s"] is None


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


def test_cli_closes_the_ledger_when_a_measurement_fails(tmp_path, monkeypatch):
    # Hold a reference to the Ledger main() opens: dropping the last reference would let the
    # garbage collector close it, which would hide a missing close().
    opened = []
    real = Ledger

    def tracking(path):
        led = real(path)
        opened.append(led)
        return led

    monkeypatch.setattr(sizing, "Ledger", tracking)
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello")
    rc = main(["--host", "http://127.0.0.1:1", "--model", "m:1", "--prompt-file", str(prompt),
               "--ledger", str(tmp_path / "l.jsonl")])
    assert rc == 1
    (led,) = opened
    assert led._closed is True


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
    rc = run_cli(fake, tmp_path, "--ledger", str(ledger), models=("m:1", "m:2"))
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
