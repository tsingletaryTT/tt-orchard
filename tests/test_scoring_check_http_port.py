# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The HTTP port module for the scoring check (scripts/scoring_check/port_model.py, issue 375): it turns the prompt ids the template gives a
port back into the text the model's own reranker server takes, sends them to a server on the loopback interface, and returns the server's
raw scores. The tests run it against a fake server on 127.0.0.1 and a small byte-level tokenizer, on the CPU. Nothing here touches a chip."""
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from orchard import scoring_check as sc
from orchard import scoring_reference as sr
from port_check_fakes import CHIPS, python_path
from scoring_check_fakes import make_run, write_marker

PORT_FILE = Path(__file__).resolve().parent.parent / "scripts" / "scoring_check" / "port_model.py"
spec = importlib.util.spec_from_file_location("http_port_under_test", PORT_FILE)
port = importlib.util.module_from_spec(spec)
spec.loader.exec_module(port)


def fake_score(query: str, document: str) -> float:
    """What the fake server answers: a number that depends on both texts, so a swapped or truncated text changes it."""
    return round((len(document) % 17) / 3.0 - (len(query) % 5) / 7.0, 4)


class FakeServer:
    """A reranker server on 127.0.0.1. `mode` bends one answer: ok, http500, short, nan, tokens, bad_index."""

    def __init__(self, tokenizer=None):
        self.requests, self.mode, self.tokenizer = [], "ok", tokenizer
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200 if self.path == "/health" else 404, {"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append((self.path, body))
                if outer.mode == "http500":
                    return self._send(500, {"error": "chip on fire"})
                qs, ds = body["text_1"], body["text_2"]
                qs = qs * len(ds) if len(qs) == 1 else qs
                scores = [fake_score(q, d) for q, d in zip(qs, ds)]
                if outer.mode == "short":
                    scores = scores[:-1]
                rows = [{"index": i, "object": "score", "score": 0.5, "raw_score": s} for i, s in enumerate(scores)]
                if outer.mode == "nan":
                    rows[0]["raw_score"] = float("nan")
                if outer.mode == "bad_index":
                    rows[0]["index"] = 7
                tokens = sum(len(outer.tokenizer.encode(sr.PREFIX + _body(body, i) + sr.SUFFIX, add_special_tokens=False).ids)
                             for i in range(len(ds))) if outer.tokenizer else 0
                if outer.mode == "tokens":
                    tokens += 1
                self._send(200, {"id": "x", "data": rows, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _body(request: dict, i: int) -> str:
    qs, ds = request["text_1"], request["text_2"]
    q = qs[0] if len(qs) == 1 else qs[i]
    return f"<Instruct>: {request.get('instruction') or sr.INSTRUCTION}\n<Query>: {q}\n<Document>: {ds[i]}"


def train_tokenizer(path: Path):
    """A byte-level BPE tokenizer trained on the pair set, so that decode(encode(text)) is the text, as with the Qwen tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    texts = [sr.PREFIX, sr.SUFFIX, sr.INSTRUCTION]
    for row in sr.pair_set():
        texts += [row["query"], row["document"]]
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(path))
    return tok


@pytest.fixture
def snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    return snap, train_tokenizer(snap / "tokenizer.json")


@pytest.fixture
def server(snapshot):
    srv = FakeServer(snapshot[1])
    yield srv
    srv.close()


@pytest.fixture
def model(snapshot, server, monkeypatch):
    monkeypatch.setenv(port.ENV_URL, server.url)
    m = port.load(str(snapshot[0]), mesh_device=None)
    yield m
    port.close(m)


def pair_ids(tok):
    pairs = sr.build_pairs(tok)["pairs"]
    return pairs, [p["input_ids"] for p in pairs]


# ---- what the port sends and returns ----------------------------------------------------------------------------------------

def test_the_scores_come_back_in_order_and_the_texts_are_the_pair_set_texts(model, server, snapshot):
    pairs, ids = pair_ids(snapshot[1])
    group = ids[:4]
    got = port.score_pairs(model, group)
    assert got == [fake_score(p["query"], p["document"]) for p in pairs[:4]]
    path, body = server.requests[-1]
    assert path == "/v1/score" and body["raw_scores"] is True
    assert body["text_2"] == [p["document"] for p in pairs[:4]] and set(body["text_1"]) == {pairs[0]["query"]}
    assert body["instruction"] == sr.INSTRUCTION


def test_every_pair_of_the_shipped_set_survives_the_text_round_trip(model, server, snapshot):
    pairs, ids = pair_ids(snapshot[1])
    got = port.score_pairs(model, ids)                       # all 32 at once, several groups with different queries
    assert got == [fake_score(p["query"], p["document"]) for p in pairs]


def test_a_large_request_is_cut_into_chunks(model, server, snapshot, monkeypatch):
    monkeypatch.setattr(port, "MAX_PER_REQUEST", 5)
    pairs, ids = pair_ids(snapshot[1])
    port.score_pairs(model, ids)
    assert all(len(b["text_2"]) <= 5 for _, b in server.requests) and len(server.requests) >= 7


def test_the_port_ignores_proxy_variables(model, server, snapshot, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    pairs, ids = pair_ids(snapshot[1])
    assert len(port.score_pairs(model, ids[:2])) == 2


# ---- what it refuses -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://example.com:8080", "http://10.1.2.3:8080", "https://127.0.0.1:8080", "ftp://127.0.0.1:8080",
                                 "http://127.0.0.1.evil.example:8080", "http://user@evil.example:80", "http://127.0.0.1", "", "127.0.0.1:8080"])
def test_load_refuses_a_server_that_is_not_on_loopback_with_a_port(snapshot, monkeypatch, url):
    monkeypatch.setenv(port.ENV_URL, url)
    with pytest.raises(RuntimeError, match=port.ENV_URL):
        port.load(str(snapshot[0]), None)


def test_load_refuses_a_missing_url_variable(snapshot, monkeypatch):
    monkeypatch.delenv(port.ENV_URL, raising=False)
    with pytest.raises(RuntimeError, match=port.ENV_URL):
        port.load(str(snapshot[0]), None)


def test_load_refuses_a_server_that_is_not_healthy(snapshot, monkeypatch):
    monkeypatch.setenv(port.ENV_URL, "http://127.0.0.1:9")            # nothing listens on the discard port
    with pytest.raises(RuntimeError, match="health"):
        port.load(str(snapshot[0]), None)


def test_ids_that_are_not_a_prompt_are_refused_and_nothing_is_sent(model, server):
    with pytest.raises(RuntimeError, match="prompt"):
        port.score_pairs(model, [[1, 2, 3]])
    assert server.requests == []


def test_a_body_without_the_system_prefix_and_suffix_is_not_a_prompt(model, server, snapshot):
    tok = snapshot[1]
    body = tok.encode("<Instruct>: x\n<Query>: q\n<Document>: d", add_special_tokens=False).ids
    with pytest.raises(RuntimeError, match="system prefix"):
        port.score_pairs(model, [body])
    assert server.requests == []


def test_a_prompt_that_does_not_come_back_as_the_same_ids_is_refused(model, server, snapshot):
    """The ids decode to the right text but are not the ids the text gives: one word is spelled out letter by letter."""
    tok = snapshot[1]
    pairs, ids = pair_ids(tok)
    one = ids[0]
    body_start = len(model["prefix"])
    at = next(k for k in range(body_start, len(one) - len(model["suffix"]))
              if len(tok.id_to_token(one[k])) >= 2 and tok.id_to_token(one[k]).isalpha())
    spelled = one[:at] + [tok.token_to_id(c) for c in tok.id_to_token(one[at])] + one[at + 1:]
    assert tok.decode(spelled, skip_special_tokens=False) == tok.decode(one, skip_special_tokens=False)
    with pytest.raises(RuntimeError, match="same ids"):
        port.score_pairs(model, [spelled])
    assert server.requests == []


@pytest.mark.parametrize("mode,word", [("http500", "500"), ("short", "scores"), ("nan", "finite"), ("bad_index", "index"),
                                       ("tokens", "tokens")])
def test_a_bad_server_answer_is_refused(model, server, snapshot, mode, word):
    server.mode = mode
    pairs, ids = pair_ids(snapshot[1])
    with pytest.raises(RuntimeError, match=word):
        port.score_pairs(model, ids[:4])


# ---- the template runs it -----------------------------------------------------------------------------------------------

rg_spec = importlib.util.spec_from_file_location("run_gate_under_test", PORT_FILE.parent / "run_gate.py")
run_gate = importlib.util.module_from_spec(rg_spec)
rg_spec.loader.exec_module(run_gate)


def build_run(tmp_path, snapshot, server, *, skew=0.0):
    """A run directory made by run_gate.setup from a pair file and a reference whose scores are the fake server's (plus `skew`)."""
    built = sr.build_pairs(snapshot[1])
    rows = built["pairs"]
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    pairs_bytes = sr.encode_json(built)
    (ref_dir / "scoring-pairs.json").write_bytes(pairs_bytes)
    scores = [fake_score(p["query"], p["document"]) + skew for p in rows]
    import hashlib
    ref = {"schema": sr.REFERENCE_SCHEMA, "model_kind": "reranker", "precision": "fp32", "scores": scores,
           "pairs": {"file": "scoring-pairs.json", "sha256": hashlib.sha256(pairs_bytes).hexdigest(), "count": len(rows)}}
    (ref_dir / "scoring-reference.json").write_bytes(sr.encode_json(ref))
    run = tmp_path / "run"
    run_gate.setup(run, ref_dir, snapshot[0])
    from port_check_fakes import install_fake_ttnn
    install_fake_ttnn(tmp_path)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": python_path(tmp_path)}
    return run, scores, env


def test_the_template_runs_this_port_against_the_fake_server_and_the_gate_passes(tmp_path, snapshot, server):
    """The port source the supervisor would copy to stages/2/port/port_model.py, run by the shipped template in a subprocess with the fake
    runtime and the fake server. The reference scores are the fake server's, so the parity is exact and the gate passes."""
    run, scores, env = build_run(tmp_path, snapshot, server)
    done, reasons = run_gate.run(run, CHIPS[1:], server.url, timeout_s=120, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert reasons == []
    rec = json.loads((run / "stages" / "2" / "evidence" / "scoring-check.json").read_text())
    assert rec["chip_scores"] == scores and rec["metrics"]["pearson"] == pytest.approx(1.0) and rec["chips"] == CHIPS[1:]


def test_a_reference_the_server_does_not_match_fails_the_gate(tmp_path, snapshot, server):
    """A constant offset of 3 logits on the reference: the template still runs, and the gate reports the absolute-difference bounds."""
    run, scores, env = build_run(tmp_path, snapshot, server, skew=3.0)
    done, reasons = run_gate.run(run, CHIPS[1:], server.url, timeout_s=120, env=env)
    assert done.returncode == 0
    assert any("mean" in r for r in reasons) and any("max" in r for r in reasons)


def test_run_gate_refuses_an_existing_run_directory(tmp_path, snapshot):
    (tmp_path / "run").mkdir()
    with pytest.raises(SystemExit):
        run_gate.setup(tmp_path / "run", tmp_path, snapshot[0])
