# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 407: a scoring record says where the scores were computed. A port module can run the model in the check's worker process (the
default) or forward the prompts to a server on the loopback interface. The record says which, what address the port declared, and what
the template's own probe of that address returned. The gate asks a server-backed port for a loopback address and an answering server and
does not ask it for a mention of ttnn; it keeps asking an in-process port for the mention. The record's `chips` and `load_s` keep their meaning
and now say it. Nothing here touches a chip: the fake ttnn of tests/port_check_fakes.py and a fake server on 127.0.0.1."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from orchard import scoring_check as sc
from scoring_check_fakes import HONEST, make_run, run_template

SERVER_PORT_SRC = '''
COMPUTES_ON = "server"
ADDRESS = "{address}"
def load(model_dir, mesh_device):
    return {{"dir": model_dir}}
def server_address(model):
    return ADDRESS
def score_pairs(model, pairs):
    return [(ids[-1] - 100) / 10 for ids in pairs]
def close(model):
    pass
'''                                   # computes the scores itself, so the test needs no real server behind the declared address


class FakeServer:
    """A server on 127.0.0.1 that answers /health and /v1/models. `health` is the status it gives /health."""

    def __init__(self, health=200, models=("fake-reranker",)):
        outer = self
        self.health, self.models, self.paths = health, list(models), []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.paths.append(self.path)
                code, obj = (outer.health, {"status": "ok"}) if self.path == "/health" else (200, {"data": [{"id": m} for m in outer.models]})
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.address = f"127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def go(tmp_path, port_src):
    run, stage, _ = make_run(tmp_path, port_src)
    done = run_template(tmp_path, run, stage)
    return run, stage, done


def record(stage):
    return json.loads((stage / "evidence" / "scoring-check.json").read_text())


def gate(run, stage):
    return sc.check(stage, run)[0]


def edit(stage, change):
    path = stage / "evidence" / "scoring-check.json"
    rec = json.loads(path.read_text())
    change(rec)
    path.write_text(json.dumps(rec))


# ---- what the record says ---------------------------------------------------------------------------------------------------------------

def test_an_in_process_port_is_recorded_as_computed_in_the_worker_and_passes(tmp_path):
    run, stage, done = go(tmp_path, HONEST)
    assert done.returncode == 0, done.stdout + done.stderr
    c = record(stage)["computed_on"]
    assert c["where"] == "in-process" and c["server"] is None
    assert "opened" in c["chips"] and "load" in c["load_s"]
    assert gate(run, stage) == []


def test_a_server_backed_port_is_recorded_with_its_address_and_the_templates_own_probe(tmp_path, server):
    run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address=server.address))
    assert done.returncode == 0, done.stdout + done.stderr
    c = record(stage)["computed_on"]
    assert c["where"] == "server" and c["server"]["address"] == server.address
    assert c["server"]["probe"] == {"health_status": 200, "models": ["fake-reranker"], "error": None}
    assert "/health" in server.paths and "/v1/models" in server.paths
    assert "server" in c["chips"] and "server" in c["load_s"]


def test_the_gate_passes_a_server_backed_port_that_never_mentions_ttnn(tmp_path, server):
    run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address=server.address))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ttnn" not in (stage / "port" / "port_model.py").read_text()
    assert gate(run, stage) == []


def test_a_port_that_does_not_declare_the_server_still_needs_the_ttnn_mention(tmp_path, server):
    src = SERVER_PORT_SRC.format(address=server.address).replace('COMPUTES_ON = "server"\n', "")
    run, stage, done = go(tmp_path, src)
    assert done.returncode == 0, done.stdout + done.stderr
    assert record(stage)["computed_on"]["where"] == "in-process"
    assert any("mentions ttnn" in r for r in gate(run, stage))


def test_a_declared_server_that_does_not_answer_is_recorded_and_refused(tmp_path):
    s = FakeServer(health=503)
    try:
        run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address=s.address))
    finally:
        s.close()
    assert done.returncode == 0, done.stdout + done.stderr
    assert record(stage)["computed_on"]["server"]["probe"]["health_status"] == 503
    assert any("did not answer" in r for r in gate(run, stage))


def test_a_declared_server_nobody_listens_on_is_recorded_and_refused(tmp_path):
    s = FakeServer()
    address = s.address
    s.close()
    run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address=address))
    assert done.returncode == 0, done.stdout + done.stderr
    probe = record(stage)["computed_on"]["server"]["probe"]
    assert probe["health_status"] is None and probe["error"]
    assert any("did not answer" in r for r in gate(run, stage))


@pytest.mark.parametrize("address", ["10.0.0.5:8000", "example.com:80", "127.0.0.1", "127.0.0.1:0", "127.0.0.1:70000", "127.0.0.1:80/x",
                                     "http://127.0.0.1:80", "localhost:80:80", ""])
def test_a_declared_address_that_is_not_a_loopback_host_and_port_fails_the_run(tmp_path, address):
    run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address=address))
    assert done.returncode == 2 and "loopback" in done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


def test_an_unknown_computes_on_value_fails_the_run(tmp_path):
    run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address="127.0.0.1:1").replace('"server"', '"cloud"', 1))
    assert done.returncode == 2 and "COMPUTES_ON" in done.stdout


# ---- the gate's reading of the record ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change,word", [
    (lambda r: r.pop("computed_on"), "computed_on"),
    (lambda r: r.update(computed_on=None), "computed_on"),
    (lambda r: r.update(computed_on=["server"]), "computed_on"),
    (lambda r: r["computed_on"].update(where="cloud"), "must be one of"),
    (lambda r: r["computed_on"].update(where=1), "must be one of"),
    (lambda r: r["computed_on"].update(server={"address": "127.0.0.1:1", "probe": None}), "server"),
    (lambda r: r["computed_on"].update(chips=None), "chips"),
    (lambda r: r["computed_on"].update(load_s=3), "load_s"),
])
def test_a_record_with_a_malformed_computed_on_is_refused(tmp_path, change, word):
    run, stage, done = go(tmp_path, HONEST)
    assert done.returncode == 0, done.stdout + done.stderr
    edit(stage, change)
    r = gate(run, stage)
    assert any("computed_on" in x and word in x for x in r), r


def test_a_record_that_says_server_with_a_non_loopback_address_is_refused(tmp_path, server):
    run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address=server.address))
    assert done.returncode == 0, done.stdout + done.stderr
    edit(stage, lambda r: r["computed_on"]["server"].update(address="10.0.0.5:8000"))
    assert any("loopback" in x for x in gate(run, stage))


def test_a_record_that_says_in_process_but_names_a_server_is_refused(tmp_path, server):
    run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address=server.address))
    assert done.returncode == 0, done.stdout + done.stderr
    edit(stage, lambda r: r["computed_on"].update(where="in-process"))
    assert any("in-process" in x and "server" in x for x in gate(run, stage))


def test_a_record_that_says_server_without_a_server_is_refused(tmp_path):
    run, stage, done = go(tmp_path, HONEST)
    assert done.returncode == 0, done.stdout + done.stderr
    edit(stage, lambda r: r["computed_on"].update(where="server"))
    assert any("server" in x and "computed_on" in x for x in gate(run, stage))


# ---- the shipped HTTP port --------------------------------------------------------------------------------------------------------------

def test_the_shipped_http_port_declares_that_a_server_computes_and_names_its_address():
    path = Path(__file__).resolve().parent.parent / "scripts" / "scoring_check" / "port_model.py"
    spec = importlib.util.spec_from_file_location("http_port_declares", path)
    port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port)
    assert port.COMPUTES_ON == "server"
    assert port.server_address({"host": "127.0.0.1", "port": 20000}) == "127.0.0.1:20000"
    assert port.server_address({"host": "localhost", "port": 7}) == "localhost:7"
    assert port.server_address({"host": "::1", "port": 20000}) == "[::1]:20000"


# ---- Copilot round 1 on PR 423 ----------------------------------------------------------------------------------------------------------

def test_the_time_the_port_takes_to_name_its_server_is_not_part_of_load_s(tmp_path, server):
    """`load_s` is the time of the port's own load call. A server_address callback that takes two seconds must not be counted in it."""
    src = SERVER_PORT_SRC.format(address=server.address).replace(
        "def server_address(model):\n", "def server_address(model):\n    import time\n    time.sleep(2.0)\n")
    run, stage, done = go(tmp_path, src)
    assert done.returncode == 0, done.stdout + done.stderr
    rec = record(stage)
    assert rec["computed_on"]["server"]["address"] == server.address
    assert rec["load_s"] < 1.5, rec["load_s"]
    assert gate(run, stage) == []


def test_a_port_whose_load_is_slow_still_has_that_time_in_load_s(tmp_path, server):
    src = SERVER_PORT_SRC.format(address=server.address).replace(
        "def load(model_dir, mesh_device):\n", "def load(model_dir, mesh_device):\n    import time\n    time.sleep(1.0)\n")
    run, stage, done = go(tmp_path, src)
    assert done.returncode == 0, done.stdout + done.stderr
    assert record(stage)["load_s"] >= 1.0


def test_a_server_address_callback_that_raises_fails_the_run_with_its_error(tmp_path, server):
    src = SERVER_PORT_SRC.format(address=server.address).replace(
        "def server_address(model):\n", "def server_address(model):\n    raise RuntimeError('no address yet')\n")
    run, stage, done = go(tmp_path, src)
    assert done.returncode == 2 and "no address yet" in done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


def test_the_load_label_describes_the_timed_call_and_does_not_claim_the_server_was_preloaded(tmp_path, server):
    """A server-backed port may start or load the server's model inside `load()`, and the health probe cannot say when the model was loaded.
    The label says what is timed and that the server's own model-load time is not measured separately."""
    run, stage, done = go(tmp_path, SERVER_PORT_SRC.format(address=server.address))
    assert done.returncode == 0, done.stdout + done.stderr
    label = record(stage)["computed_on"]["load_s"]
    assert "before this check began" not in label and "had loaded" not in label
    assert "load call" in label and "not measured separately" in label


def test_the_in_process_load_label_is_unchanged_in_meaning(tmp_path):
    run, stage, done = go(tmp_path, HONEST)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "load call" in record(stage)["computed_on"]["load_s"]
