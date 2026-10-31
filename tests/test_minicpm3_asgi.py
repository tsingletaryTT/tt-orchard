"""The MiniCPM3-4B package's ASGI front end (pkgsrc/server/minicpm3_asgi.py), driven without a server process or a chip.

A small ASGI driver calls `app` with a scope, a body and a `receive`/`send` pair. The model is a fake that returns scripted logits, so these
tests check the HTTP behaviour (routes, status codes, streaming, busy and unusable answers, client disconnect), the settings, the weights
resolution and the start-up and shutdown, not the model. The request checks and the generation loop are scripts/minicpm3/serve.py's, which the
adapter imports; the tests of `parse_request`, `encode_prompt` and `generate_ids` are here too because the adapter is their second caller."""
import asyncio
import importlib.util
import json
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("minicpm3_asgi_under_test", ROOT / "pkgsrc" / "server" / "minicpm3_asgi.py")
# serve.py is at scripts/minicpm3/serve.py, which is where `_core()` finds it when `minicpm3_serve` is not on the path
asgi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asgi)
serve = asgi.serve
EOS = 2                                    # serve.EOS holds 2 and 73440
PROMPT = 3


class FakeTok:
    """A chat template that gives `n_prompt` ids, and a decoder that maps id i to the letter chr(97 + i)."""
    n_prompt = PROMPT

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        return torch.tensor([list(range(1, self.n_prompt + 1))])

    def decode(self, ids):
        return "".join(chr(97 + i) for i in ids)


class FakeModel:
    """After the prompt, predicts the scripted tokens in order. `steps` records every (token, position) the chip was asked for."""

    def __init__(self, script, delay=0.0):
        self.script, self.delay, self.steps = list(script), delay, []

    def traced_step(self, tok, pos):
        self.steps.append((tok, pos))
        if self.delay:
            time.sleep(self.delay)
        v = torch.zeros(16)
        v[self.script[min(max(pos - (FakeTok.n_prompt - 1), 0), len(self.script) - 1)]] = 1.0
        return v


@pytest.fixture(autouse=True)
def state():
    FakeTok.n_prompt = PROMPT
    serve.STATE.clear()
    serve.STATE.update(tok=FakeTok(), model=FakeModel([4, 5, 6, EOS]), max_len=128, batched_prefill=False)
    yield
    if serve.LOCK.locked():
        serve.LOCK.release()
    serve.STATE.clear()


async def call(method, path, body=b"", disconnect_after=None, disconnect_at_start=False):
    """Returns (status, headers, body bytes). `disconnect_after`: send http.disconnect once this many body chunks have been sent.
    `disconnect_at_start`: the client is gone as soon as the body has been read."""
    sent, parts = [], []
    inbox = [{"type": "http.request", "body": body, "more_body": False}]
    gate = asyncio.Event()
    if disconnect_at_start:
        gate.set()

    async def receive():
        if inbox:
            return inbox.pop(0)
        await gate.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)
        if msg["type"] == "http.response.body":
            parts.append(msg.get("body", b""))
            if disconnect_after is not None and len(parts) >= disconnect_after:
                gate.set()

    await asyncio.wait_for(asgi.app({"type": "http", "method": method, "path": path}, receive, send), 30)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}, b"".join(parts)


def post(payload, **kw):
    return call("POST", "/v1/chat/completions", json.dumps(payload).encode() if not isinstance(payload, bytes) else payload, **kw)


def run(coro):
    return asyncio.run(coro)


MSG = {"messages": [{"role": "user", "content": "hi"}]}


# ---- routes and answers --------------------------------------------------------------------------------------------------------------

def test_health_models_and_unknown_path():
    status, _, body = run(call("GET", "/health"))
    assert status == 200 and json.loads(body) == {"status": "ok", "model": serve.MODEL_ID, "context": 128}
    status, _, body = run(call("GET", "/v1/models"))
    assert status == 200 and json.loads(body)["data"][0]["id"] == serve.MODEL_ID == "openbmb/MiniCPM3-4B"
    assert run(call("GET", "/nope"))[0] == 404
    assert run(call("GET", "/v1/chat/completions"))[0] == 404
    assert run(call("POST", "/v1/models", b"{}"))[0] == 404


def test_chat_completion_gives_text_usage_and_finish_stop():
    status, headers, body = run(post(MSG))
    out = json.loads(body)
    assert status == 200 and headers["content-type"] == "application/json"
    assert out["choices"][0] == {"index": 0, "message": {"role": "assistant", "content": "efg"}, "finish_reason": "stop"}
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
    assert not serve.LOCK.locked()


def test_the_answer_carries_timings_the_boot_check_reads():
    """The can's boot check reads `timings.decode_tok_s` from the response. With 10 ms per chip step the rate is under 100 and the first piece comes after
    the three prompt steps."""
    serve.STATE["model"] = FakeModel([4] * 30, delay=0.01)
    out = json.loads(run(post({**MSG, "max_tokens": 20}))[2])
    t = out["timings"]
    assert 0 < t["decode_tok_s"] <= 100 and t["first_token_s"] >= 0.03
    assert out["usage"]["completion_tokens"] == 20


def test_max_tokens_gives_finish_length_and_prompt_tokens_go_through_the_chip_from_position_zero():
    out = json.loads(run(post({**MSG, "max_tokens": 2}))[2])
    assert out["choices"][0]["message"]["content"] == "ef" and out["choices"][0]["finish_reason"] == "length"
    assert [p for _, p in serve.STATE["model"].steps][:4] == [0, 1, 2, 3]      # every request restarts at position 0; no batched prefill


@pytest.mark.parametrize("payload", [b"not json", b"[]", b"123", {"messages": []}, {"messages": "hi"}, {"messages": [{"role": "user", "content": 5}]},
                                     {"messages": ["hi"]}, {**MSG, "max_tokens": "many"}, {**MSG, "max_tokens": [1]}, {}])
def test_bad_requests_get_a_json_400_and_free_the_slot(payload):
    status, _, body = run(post(payload))
    assert status == 400 and "error" in json.loads(body)
    assert not serve.LOCK.locked()
    assert serve.STATE["model"].steps == []             # nothing reached the chip


def test_empty_and_oversized_bodies_are_refused():
    assert run(post(b""))[0] == 400
    assert run(post(b"x" * (serve.MAX_BODY + 1)))[0] == 400


@pytest.mark.parametrize("raw,words", [(b"[]", "JSON object"), (b"123", "JSON object"), (b'{"messages":[{"content":"x"}],"max_tokens":[1]}', "TypeError"),
                                       (b"", "1 MiB"), (b"x" * ((1 << 20) + 1), "1 MiB"), (b"{}", "KeyError")])
def test_parse_request_raises_value_error_with_a_message_for_every_bad_body(raw, words):
    """serve.py's own handler calls this too and catches only ValueError, so a TypeError or KeyError must not escape."""
    with pytest.raises(ValueError, match=words):
        serve.parse_request(raw)


def test_a_tokenizer_failure_is_the_clients_400_and_does_not_touch_the_chip():
    class BadTok(FakeTok):
        def apply_chat_template(self, *a, **k):
            raise TypeError("template could not render this message")
    serve.STATE["tok"] = BadTok()
    status, _, body = run(post(MSG))
    assert status == 400 and b"TypeError" in body and not serve.LOCK.locked() and serve.STATE["model"].steps == []


def test_max_tokens_is_clamped_to_4096_and_defaults_to_256():
    assert serve.parse_request(json.dumps({**MSG, "max_tokens": 10 ** 9}).encode())[1] == 4096
    assert serve.parse_request(json.dumps(MSG).encode())[1] == 256
    assert serve.parse_request(json.dumps({**MSG, "max_tokens": -5}).encode())[1] == 1


def test_streaming_sends_a_role_chunk_content_chunks_a_finish_chunk_and_done():
    status, headers, body = run(post({**MSG, "stream": True}))
    assert status == 200 and headers["content-type"] == "text/event-stream"
    events = [e[len("data: "):] for e in body.decode().split("\n\n") if e]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "efg"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert not serve.LOCK.locked()


@pytest.mark.parametrize("stream", [False, True])
def test_a_prompt_with_no_room_gets_a_json_400_before_any_event(stream):
    FakeTok.n_prompt = 127                              # the context is 128: no room for an answer
    status, headers, body = run(post({**MSG, "stream": stream}))
    assert status == 400 and headers["content-type"] == "application/json" and b"context" in body
    assert not serve.LOCK.locked() and serve.STATE["model"].steps == []


# ---- one request at a time, an unusable chip -------------------------------------------------------------------------------------------

def test_a_second_request_while_one_runs_gets_503_with_retry_after():
    assert serve.LOCK.acquire(blocking=False)          # stands in for the running request
    t0 = time.time()
    status, headers, _ = run(post(MSG))
    assert status == 503 and headers["retry-after"] == "5"
    assert time.time() - t0 < 5                         # it did not wait for the lock


def test_unusable_chip_gives_503_on_health_and_on_chat():
    serve.STATE["unusable"] = "trace replay failed"
    status, _, body = run(call("GET", "/health"))
    assert status == 503 and json.loads(body)["status"] == "unusable"
    status, _, body = run(post(MSG))
    assert status == 503 and b"unusable" in body
    assert not serve.LOCK.locked() and serve.STATE["model"].steps == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_failure_on_the_chip_marks_the_server_unusable_and_frees_the_slot(stream):
    class Boom(FakeModel):
        def traced_step(self, tok, pos):
            raise RuntimeError("MMIO per-op timeout")
    serve.STATE["model"] = Boom([4])
    status, _, body = run(post({**MSG, "stream": stream}))
    if not stream:
        assert status == 500 and b"RuntimeError" in body
    assert not serve.LOCK.locked()
    assert "MMIO" in serve.STATE["unusable"]
    assert run(call("GET", "/health"))[0] == 503      # a supervisor that polls /health restarts the server
    assert run(post(MSG))[0] == 503


def test_a_request_looks_at_unusable_again_after_it_takes_the_lock(monkeypatch):
    """The request that held the lock may have failed in between the first look and the lock, so the chip must not be touched."""
    real = serve.LOCK

    class FailsWhileWaiting:
        def acquire(self, blocking=True):
            got = real.acquire(blocking=blocking)
            serve.STATE["unusable"] = "failed while this request was taking the slot"
            return got

        def release(self):
            real.release()

        def locked(self):
            return real.locked()
    monkeypatch.setattr(serve, "LOCK", FailsWhileWaiting())
    status, _, body = run(post(MSG))
    assert status == 503 and b"unusable" in body
    assert not real.locked() and serve.STATE["model"].steps == []


# ---- client disconnect, concurrency ------------------------------------------------------------------------------------------------------

def test_client_disconnect_mid_stream_frees_the_slot_and_stops_the_chip():
    model = FakeModel([4] * 50, delay=0.03)           # the delay gives the event loop time to deliver the disconnect, even on a busy host
    serve.STATE["model"] = model
    status, _, _ = run(post({**MSG, "stream": True, "max_tokens": 40}, disconnect_after=3))
    assert status == 200
    assert not serve.LOCK.locked()
    assert len(model.steps) < 20                        # it stopped soon after the client left; it did not run its 40 tokens


def test_client_disconnect_during_the_prompt_stops_the_chip_before_the_answer():
    """A long prompt takes one chip step per token (about 30 ms each); a client that leaves must not keep the slot for the whole prompt."""
    FakeTok.n_prompt = 100
    model = FakeModel([4] * 50, delay=0.03)
    serve.STATE["model"] = model
    status, _, _ = run(post({**MSG, "stream": True}, disconnect_after=1))        # the first chunk is the role chunk, sent before any step
    assert status == 200 and not serve.LOCK.locked()
    assert len(model.steps) < 60                        # fewer than the 100 prompt steps


def test_client_disconnect_on_a_non_streaming_request_also_stops_the_chip():
    model = FakeModel([4] * 200, delay=0.03)
    serve.STATE["model"] = model
    run(post({**MSG, "max_tokens": 150}, disconnect_at_start=True))
    assert not serve.LOCK.locked()
    assert len(model.steps) < 60


def test_health_answers_while_a_request_generates():
    """The chip work is in a worker thread, so the event loop is free."""
    release, entered = threading.Event(), threading.Event()

    class Slow(FakeModel):
        def traced_step(self, tok, pos):
            entered.set()
            release.wait(10)
            return super().traced_step(tok, pos)
    serve.STATE["model"] = Slow([4, EOS])

    async def both():
        chat = asyncio.ensure_future(post(MSG))
        await asyncio.get_running_loop().run_in_executor(None, entered.wait, 10)
        health = await call("GET", "/health")
        busy = await post(MSG)
        release.set()
        return health, busy, await chat
    health, busy, chat = run(both())
    assert health[0] == 200 and busy[0] == 503 and chat[0] == 200


# ---- serve.py's pieces the adapter shares ----------------------------------------------------------------------------------------------------

def test_generate_ids_stops_when_asked_and_then_reports_nothing():
    gen = serve.generate_ids([1, 2, 3], 10, should_stop=lambda: True)
    assert list(gen) == [] and serve.STATE["model"].steps == [] and "finish" not in serve.STATE


def test_generate_and_generate_ids_agree():
    a = "".join(serve.generate(MSG["messages"], 10))
    b = "".join(serve.generate_ids([1, 2, 3], 10))
    assert a == b == "efg"


# ---- settings, weights, start-up ------------------------------------------------------------------------------------------------------------

def test_settings_come_from_the_environment_and_have_the_measured_defaults():
    d = asgi.settings({})
    assert (d.max_len, d.mlp_dtype, d.mlp_fidelity, d.model_dir, d.weights) == (2048, "bf16", "hifi2", None, None)
    a = asgi.settings({"MINICPM3_MAX_LEN": "512", "MINICPM3_MLP_DTYPE": "bfp8", "MINICPM3_MLP_FIDELITY": "lofi", "MINICPM3_DIR": "/m", "MINICPM3_WEIGHTS": "/w"})
    assert (a.max_len, a.mlp_dtype, a.mlp_fidelity, a.model_dir, a.weights) == (512, "bfp8", "lofi", "/m", "/w")


@pytest.mark.parametrize("env", [{"MINICPM3_MLP_DTYPE": "fp32"}, {"MINICPM3_MLP_FIDELITY": "hifi9"}, {"MINICPM3_MAX_LEN": "lots"}, {"MINICPM3_MAX_LEN": "64"},
                                 {"MINICPM3_MAX_LEN": "40000"}])
def test_a_setting_that_the_chip_model_cannot_take_is_refused_with_a_message(env):
    with pytest.raises(ValueError, match="MINICPM3_"):
        asgi.settings(env)


def test_the_default_revision_is_the_one_the_spec_and_the_card_pin():
    cfg = tomllib.loads((ROOT / "config" / "can.minicpm3.toml").read_text())
    assert asgi.REVISION == cfg["weights"]["revision"] == "d6b14ddaefdb11c624dd75c3c779549bc90b08cb"
    assert cfg["weights"]["repo"] == serve.MODEL_ID


def test_weights_in_the_model_directory_are_used_as_they_are(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert asgi.weights_file(tmp_path, tmp_path / "cache") == (tmp_path / "model.safetensors").resolve()


def test_the_pickle_is_converted_once_into_the_cache_and_reused(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    d = tmp_path / "snap"
    d.mkdir()
    torch.save({"model.embed.weight": torch.arange(6.0).reshape(2, 3)}, d / "pytorch_model.bin")
    calls = []
    real = asgi.convert_weights.convert
    monkeypatch.setattr(asgi.convert_weights, "convert", lambda *a: (calls.append(a), real(*a))[1])
    w1 = asgi.weights_file(d, tmp_path / "cache")
    w2 = asgi.weights_file(d, tmp_path / "cache")
    assert w1 == w2 and w1.parent == (tmp_path / "cache") and w1.is_file() and len(calls) == 1
    from safetensors.torch import load_file
    assert torch.equal(load_file(str(w1))["model.embed.weight"], torch.arange(6.0).reshape(2, 3))
    other = tmp_path / "other" / "snap"                  # a second directory with the same name does not share the file
    other.mkdir(parents=True)
    torch.save({"model.embed.weight": torch.zeros(1)}, other / "pytorch_model.bin")
    assert asgi.weights_file(other, tmp_path / "cache") != w1


def test_the_cache_directory_follows_hf_home_and_the_override():
    assert asgi.cache_dir({"HF_HOME": "/hf"}) == Path("/hf") / "tt-orchard-minicpm3"
    assert asgi.cache_dir({"HF_HOME": "/hf", "MINICPM3_CACHE": "/c"}) == Path("/c")
    assert asgi.cache_dir({"HOME": "/h"}) == Path.home() / ".cache" / "huggingface" / "tt-orchard-minicpm3"


def test_startup_downloads_the_pinned_revision_loads_the_chip_and_prints_a_ready_line(tmp_path, monkeypatch, capsys):
    import huggingface_hub
    got = {}
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw: (got.update(repo=repo, **kw), str(snap))[1])
    monkeypatch.setenv("HF_MODEL", "openbmb/MiniCPM3-4B")
    monkeypatch.setenv("TT_MODEL_WEIGHTS_REVISION", "a" * 40)
    monkeypatch.delenv("MINICPM3_DIR", raising=False)
    closed = []
    monkeypatch.setattr(asgi, "load_tokenizer", lambda d: FakeTok())
    monkeypatch.setattr(asgi, "load_chip", lambda a, weights: (SimpleNamespace(name="dev"), FakeModel([4]), 1.5))
    monkeypatch.setattr(asgi, "close_chip", lambda dev, model: closed.append(dev.name))
    close = asgi.startup(asgi.settings({}))
    assert got["repo"] == "openbmb/MiniCPM3-4B" and got["revision"] == "a" * 40
    assert isinstance(serve.STATE["model"], FakeModel) and serve.STATE["max_len"] == 2048 and serve.STATE["batched_prefill"] is False
    line = capsys.readouterr().out.strip().splitlines()[-1]
    ready = json.loads(line)
    assert ready["event"] == "ready" and ready["context"] == 2048 and ready["mlp_dtype"] == "bf16"
    assert '"traced": true' in line                      # the boot check looks for this text in the log
    close()
    assert closed == ["dev"]


def test_startup_does_not_download_when_a_model_directory_is_given(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("downloaded"))
    (tmp_path / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(asgi, "load_tokenizer", lambda d: FakeTok())
    monkeypatch.setattr(asgi, "load_chip", lambda a, weights: (SimpleNamespace(name="dev"), FakeModel([4]), 0.1))
    asgi.startup(asgi.settings({"MINICPM3_DIR": str(tmp_path)}))


def test_the_lifespan_loads_before_it_says_started_and_closes_the_chip_at_shutdown(monkeypatch):
    order = []
    monkeypatch.setattr(asgi, "startup", lambda a: (order.append("startup"), lambda: order.append("close"))[1])

    async def drive():
        inbox = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
        sent = []

        async def receive():
            return inbox.pop(0)

        async def send(msg):
            order.append(msg["type"])
            sent.append(msg)
        await asgi.app({"type": "lifespan"}, receive, send)
    run(drive())
    assert order == ["startup", "lifespan.startup.complete", "close", "lifespan.shutdown.complete"]


def test_a_failed_start_is_reported_to_uvicorn_so_the_process_exits(monkeypatch):
    def boom(a):
        raise RuntimeError("no chip")
    monkeypatch.setattr(asgi, "startup", boom)
    sent = []

    async def drive():
        async def receive():
            return {"type": "lifespan.startup"}

        async def send(msg):
            sent.append(msg)
        await asgi.app({"type": "lifespan"}, receive, send)
    run(drive())
    assert sent[0]["type"] == "lifespan.startup.failed" and "no chip" in sent[0]["message"]


def test_importing_the_adapter_loads_neither_ttnn_nor_the_chip_model():
    """The bundle's install check imports the adapter and must not open a device."""
    code = ("import sys; sys.path.insert(0, %r); import minicpm3_asgi; "
            "assert 'ttnn' not in sys.modules and 'chip_model' not in sys.modules, sorted(m for m in sys.modules if m in ('ttnn', 'chip_model'))")
    # the server module is next to the adapter in a bundle's extra_code/, under another name
    r = subprocess.run([sys.executable, "-I", "-c", code % str(ROOT / "pkgsrc" / "server")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


# ---- the spec ------------------------------------------------------------------------------------------------------------------------------

def test_the_minicpm3_spec_loads_and_names_files_that_exist_and_an_adapter_inside_extra_code():
    from orchard import can
    s = can.load_spec(ROOT / "config" / "can.minicpm3.toml", base=ROOT)
    assert (s.kind, s.app, s.mesh, s.device_count, s.name) == ("tt-dit-server", "minicpm3_asgi:app", "P150", 1, "minicpm3-4b-blackhole")
    assert 20130 <= s.port <= 20139 and s.repo == serve.MODEL_ID
    assert all(src.is_file() for src, _ in s.extra_code + s.bundle_files) and s.model_py.is_file() and s.readme.is_file()
    dsts = {dst for _, dst in s.extra_code}
    assert {"minicpm3_asgi.py", "minicpm3_serve.py", "chip_model.py", "ref_model.py", "convert_weights.py"} <= dsts
    assert can.requirement_problems("\n".join(s.requirements)) == []
    assert "import minicpm3_asgi" in s.verify               # the install runs the import with only extra_code/ on the path
    assert s.check.min_decode_tok_s and any('"traced": true' in n for n in s.check.ready_contains)


# ---- review of PR 308 ----------------------------------------------------------------------------------------------------------------------

def test_max_len_is_rounded_up_once_so_the_model_and_the_advertised_context_agree(tmp_path, monkeypatch):
    """The chip model builds its rotary tables from the length it is given; the server advertises STATE["max_len"]. Both must hold the rounded value."""
    assert asgi.settings({"MINICPM3_MAX_LEN": "129"}).max_len == 256
    assert asgi.settings({"MINICPM3_MAX_LEN": "256"}).max_len == 256 and asgi.settings({"MINICPM3_MAX_LEN": "128"}).max_len == 128
    (tmp_path / "model.safetensors").write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(asgi, "load_tokenizer", lambda d: FakeTok())
    monkeypatch.setattr(asgi, "load_chip", lambda a, w: (seen.update(max_len=a.max_len), (SimpleNamespace(name="d"), FakeModel([4]), 0.1))[1])
    asgi.startup(asgi.settings({"MINICPM3_DIR": str(tmp_path), "MINICPM3_MAX_LEN": "129"}))
    assert seen["max_len"] == 256 == serve.STATE["max_len"]


def test_the_answer_uses_the_counts_and_finish_of_its_own_generation_not_of_a_request_that_took_the_slot_next(monkeypatch):
    """The slot is released in the worker; a request that takes it next overwrites STATE["last_counts"] and ["finish"]. The response must be built from values read before the release."""
    real = serve.LOCK

    class NextRequestWritesOnRelease:
        def acquire(self, blocking=True):
            return real.acquire(blocking=blocking)

        def release(self):
            real.release()
            serve.STATE["last_counts"], serve.STATE["finish"] = (999, 999), "length"       # what the next request would leave behind

        def locked(self):
            return real.locked()
    monkeypatch.setattr(serve, "LOCK", NextRequestWritesOnRelease())
    out = json.loads(run(post(MSG))[2])
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
    assert out["choices"][0]["finish_reason"] == "stop"
