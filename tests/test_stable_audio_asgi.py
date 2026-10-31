"""The Stable Audio 3 package's ASGI front end (pkgsrc/server/stable_audio_asgi.py), driven without a server process or a chip.

A small ASGI driver calls `app` with a scope, a body and a `receive`/`send` pair. The generators are fakes (a gated one that waits for the test, and the job
server's own FakeGenerator), so these tests check the HTTP behaviour, the life cycle, the settings and the resident generator's use of its runtime, and
not the model."""
import asyncio
import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("stable_audio_asgi_under_test", ROOT / "pkgsrc" / "server" / "stable_audio_asgi.py")
# stable_audio_jobs.py is next to the file under test, which is where `_core()` finds it when it is not on the path
asgi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asgi)
jobs = asgi.jobs


class GatedGenerator(jobs.Generator):
    """Waits for `release` inside generate, so a test can hold a clip 'running'. Records the thread that loaded and the closes."""
    model_name = "gated test generator"
    device_name = "no chip"

    def __init__(self, fail=False):
        self.release = threading.Event()
        self.started = threading.Event()
        self.loaded_in = None
        self.loads = self.closes = 0
        self.closed_while_running = False
        self.running = False
        self.fail = fail

    def load(self):
        if self.loads:                      # the job server's worker calls load again after the front end did; a generator makes that a no-op
            return
        self.loads += 1
        self.loaded_in = threading.current_thread()

    def generate(self, req, on_stage):
        self.running = True
        self.started.set()
        on_stage("sampling")
        assert self.release.wait(10), "the test never released the clip"
        self.running = False
        if self.fail:
            raise RuntimeError("the chip run failed")
        return jobs.FakeGenerator(delay_s=0).generate(req, on_stage)

    def close(self):
        self.closes += 1
        self.closed_while_running = self.closed_while_running or self.running


def make_state(gen=None, max_waiting=2, host="127.0.0.1"):
    gen = gen or jobs.FakeGenerator(delay_s=0)
    js = jobs.JobServer(gen, max_waiting=max_waiting)
    js.start()
    t0 = time.time()
    while js.health()["status"] == "loading" and time.time() - t0 < 10:
        time.sleep(0.01)
    asgi.STATE.update(js=js, gen=gen, host=host)
    return js


@pytest.fixture(autouse=True)
def state():
    js = make_state()
    yield js
    for g in (asgi.STATE.get("gen"),):
        if hasattr(g, "release"):
            g.release.set()
    asgi.STATE["js"].stopping = True
    with asgi.STATE["js"].lock:
        asgi.STATE["js"].wake.notify_all()


async def call(method, path, body=b"", headers=None, chunks=None):
    """Returns (status, headers, body bytes). Default headers are a loopback Host and a JSON content type."""
    sent, parts = [], []
    h = {"host": "127.0.0.1:20170", "content-type": "application/json", **(headers or {})}
    if body and "content-length" not in h and chunks is None:
        h["content-length"] = str(len(body))
    pieces = chunks if chunks is not None else [body]
    inbox = [{"type": "http.request", "body": p, "more_body": i < len(pieces) - 1} for i, p in enumerate(pieces)]

    async def receive():
        return inbox.pop(0) if inbox else {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)
        if msg["type"] == "http.response.body":
            parts.append(msg.get("body", b""))

    scope = {"type": "http", "method": method, "path": path, "headers": [(k.encode(), v.encode()) for k, v in h.items()]}
    await asyncio.wait_for(asgi.app(scope, receive, send), 20)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}, b"".join(parts)


def run(coro):
    return asyncio.run(coro)


def submit(payload=None):
    body = json.dumps({"prompt": "a tune", "duration": 2, "steps": 8, "seed": 7, **(payload or {})}).encode()
    return run(call("POST", "/v1/audio/jobs", body))


def wait_done(job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        status, _, body = run(call("GET", f"/v1/audio/jobs/{job_id}"))
        view = json.loads(body)
        if view["status"] in ("done", "error", "cancelled"):
            return view
        time.sleep(0.02)
    raise AssertionError("the job did not finish")


# ---- the same code as the stand-alone server

def test_the_adapter_uses_the_job_servers_own_classes_and_defines_none_of_them():
    assert asgi.jobs.__name__ == "stable_audio_jobs"
    src = (ROOT / "pkgsrc" / "server" / "stable_audio_asgi.py").read_text()
    for name in ("class JobServer", "class Job:", "def validate", "def submit", "class QueueFull", "def host_allowed", "def openapi_document"):
        assert name not in src, f"{name} is the job server's; the adapter imports it"


# ---- routes

def test_health_says_ok_when_the_model_is_loaded_and_503_when_it_failed():
    status, _, body = run(call("GET", "/health"))
    assert status == 200 and json.loads(body)["status"] == "ok"
    asgi.STATE["js"].state, asgi.STATE["js"].load_error = "error", "no chip"
    status, _, body = run(call("GET", "/health"))
    assert status == 503 and json.loads(body)["status"] == "error"
    status, _, body = run(call("GET", "/v1/health"))                    # the job server's own route answers 200 with the state in the body
    assert status == 200 and json.loads(body)["error"] == "no chip"


def test_health_is_503_while_the_model_loads():
    asgi.STATE["js"].state = "loading"
    assert run(call("GET", "/health"))[0] == 503
    assert submit()[0] == 503


def test_a_job_runs_to_done_and_its_clip_can_be_fetched():
    status, _, body = submit()
    assert status == 202 and set(json.loads(body)) == {"id", "status"}
    jid = json.loads(body)["id"]
    view = wait_done(jid)
    assert view["status"] == "done" and view["settings"]["seed"] == 7 and view["stats"]["audio_s"] == 2.0
    status, headers, audio = run(call("GET", f"/v1/audio/jobs/{jid}/audio"))
    assert status == 200 and headers["content-type"] == "audio/wav" and audio[:4] == b"RIFF"


def test_a_clip_that_is_not_done_gives_409_and_an_unknown_job_404():
    gen = GatedGenerator()
    make_state(gen)
    jid = json.loads(submit()[2])["id"]
    assert gen.started.wait(10)
    assert run(call("GET", f"/v1/audio/jobs/{jid}/audio"))[0] == 409
    assert run(call("GET", f"/v1/audio/jobs/{'0' * 32}"))[0] == 404
    assert run(call("GET", "/v1/audio/jobs/not-an-id"))[0] == 404
    gen.release.set()


@pytest.mark.parametrize("payload", [{"prompt": ""}, {"prompt": 5}, {"duration": "ten"}, {"duration": 0}, {"duration": 1000}, {"steps": 0}, {"steps": True},
                                     {"cfg_scale": 99}, {"seed": -1}, {"extra": 1}])
def test_a_bad_job_gets_400_with_a_detail(payload):
    status, _, body = submit(payload)
    assert status == 400 and "detail" in json.loads(body)


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"prompt": NaN}'])
def test_a_body_that_is_not_a_json_object_gets_400(raw):
    assert run(call("POST", "/v1/audio/jobs", raw))[0] == 400


def test_a_content_type_other_than_json_gets_415():
    status, _, _ = run(call("POST", "/v1/audio/jobs", b'{"prompt": "x"}', {"content-type": "text/plain"}))
    assert status == 415


def test_a_body_over_the_limit_gets_413_by_its_length_header_and_when_it_arrives_in_pieces():
    big = b"x" * (jobs.MAX_BODY_BYTES + 1)
    assert run(call("POST", "/v1/audio/jobs", b"", {"content-length": str(len(big))}))[0] == 413
    assert run(call("POST", "/v1/audio/jobs", chunks=[big[:40000], big[40000:]]))[0] == 413
    assert run(call("POST", "/v1/audio/jobs", b"", {"content-length": "abc"}))[0] == 400


def test_a_body_that_arrives_in_pieces_is_assembled():
    body = json.dumps({"prompt": "a tune", "duration": 2}).encode()
    assert run(call("POST", "/v1/audio/jobs", chunks=[body[:10], body[10:]]))[0] == 202


def test_unknown_paths_and_methods_get_the_job_servers_404():
    for method, path in (("GET", "/nope"), ("POST", "/v1/health"), ("GET", "/v1/audio/jobs"), ("PUT", "/v1/audio/jobs"), ("POST", "/v1/audio/jobs/" + "0" * 32)):
        status, _, body = run(call(method, path))
        assert (status, json.loads(body)) == (404, {"detail": "not found"}), (method, path)


def test_the_openapi_document_and_the_docs_page_are_served():
    status, _, body = run(call("GET", "/openapi.json"))
    doc = json.loads(body)
    assert status == 200 and doc["openapi"].startswith("3.1") and "/v1/audio/jobs" in doc["paths"]
    status, headers, body = run(call("GET", "/docs"))
    assert status == 200 and headers["content-type"].startswith("text/html") and b"/openapi.json" in body


# ---- the one clip at a time rule

def test_a_full_queue_gets_429_at_once_and_a_slot_frees_when_the_clip_ends():
    gen = GatedGenerator()
    make_state(gen, max_waiting=0)
    first = json.loads(submit()[2])["id"]
    assert gen.started.wait(10)
    t0 = time.time()
    status, _, body = submit()
    assert status == 429 and time.time() - t0 < 2 and "queue is full" in json.loads(body)["detail"]
    gen.release.set()
    assert wait_done(first)["status"] == "done"
    assert submit()[0] == 202


def test_a_failed_clip_is_an_error_job_and_the_next_job_is_accepted():
    gen = GatedGenerator(fail=True)
    make_state(gen)
    jid = json.loads(submit()[2])["id"]
    gen.release.set()
    view = wait_done(jid)
    assert view["status"] == "error" and "the chip run failed" in view["error"]
    gen.fail = False
    assert submit()[0] == 202


def test_health_answers_while_a_clip_runs():
    gen = GatedGenerator()
    make_state(gen)
    submit()
    assert gen.started.wait(10)
    t0 = time.time()
    status, _, body = run(call("GET", "/health"))
    assert status == 200 and json.loads(body)["busy"] is True and time.time() - t0 < 2
    gen.release.set()


def test_deleting_a_queued_job_drops_it_and_deleting_a_running_one_asks_for_it_to_be_discarded():
    gen = GatedGenerator()
    make_state(gen)
    running = json.loads(submit()[2])["id"]
    assert gen.started.wait(10)
    queued = json.loads(submit()[2])["id"]
    status, _, body = run(call("DELETE", f"/v1/audio/jobs/{queued}"))
    assert status == 200 and json.loads(body)["status"] == "cancelled"
    status, _, body = run(call("DELETE", f"/v1/audio/jobs/{running}"))
    assert status == 200 and json.loads(body)["cancel_requested"] is True
    assert run(call("DELETE", f"/v1/audio/jobs/{'0' * 32}"))[0] == 404
    gen.release.set()
    assert wait_done(running)["status"] == "cancelled"


# ---- the Host check

def test_a_loopback_server_refuses_another_host_name_and_a_server_bound_elsewhere_does_not():
    assert run(call("GET", "/v1/health", headers={"host": "evil.example:20170"}))[0] == 403
    assert run(call("GET", "/v1/health", headers={"host": "localhost:20170"}))[0] == 200
    assert run(call("GET", "/health", headers={"host": "evil.example"}))[0] == 403
    assert run(call("POST", "/v1/audio/jobs", b'{"prompt": "x"}', {"host": "evil.example"}))[0] == 403
    asgi.STATE["host"] = "0.0.0.0"
    assert run(call("GET", "/v1/health", headers={"host": "evil.example:20170"}))[0] == 200


# ---- life cycle

def lifespan_driver(settings, factory):
    """Runs `app` as a lifespan: returns (messages sent, the state at startup complete) after a startup and a shutdown."""
    async def go():
        inbox = asyncio.Queue()
        sent = []
        seen = {}

        async def receive():
            return await inbox.get()

        async def send(msg):
            sent.append(msg["type"])
            if msg["type"] == "lifespan.startup.complete":
                seen["js"] = asgi.STATE.get("js")
                seen["loop_thread"] = threading.current_thread()

        task = asyncio.ensure_future(asgi.app({"type": "lifespan"}, receive, send))
        seen["main"] = threading.current_thread()
        await inbox.put({"type": "lifespan.startup"})
        for _ in range(500):
            if sent:
                break
            await asyncio.sleep(0.01)
        yield_state = dict(seen)
        started = list(sent)
        await inbox.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(task, 20)
        return started, yield_state, sent
    return asyncio.run(go())


def test_startup_loads_the_generator_off_the_event_loop_before_it_says_complete_and_shutdown_closes_it(monkeypatch):
    gen = GatedGenerator()
    gen.release.set()
    monkeypatch.setattr(asgi, "build_generator", lambda a: gen)
    started, seen, sent = lifespan_driver(asgi.settings({"AUDIO_GENERATOR": "fake"}), None)
    assert started == ["lifespan.startup.complete"] and gen.loads == 1
    assert gen.loaded_in is not seen["main"]                     # the load ran in a worker thread, so /health could answer
    assert seen["js"].health()["status"] == "ok"                 # the job server was ready when startup completed
    assert gen.closes == 1 and sent[-1] == "lifespan.shutdown.complete"


def test_a_failed_load_fails_the_startup_with_the_reason(monkeypatch):
    class Bad(GatedGenerator):
        def load(self):
            raise RuntimeError("no chip here")
    monkeypatch.setattr(asgi, "build_generator", lambda a: Bad())
    async def go():
        sent = []
        inbox = asyncio.Queue()
        await inbox.put({"type": "lifespan.startup"})

        async def receive():
            return await inbox.get()

        async def send(msg):
            sent.append(msg)
        await asyncio.wait_for(asgi.app({"type": "lifespan"}, receive, send), 20)
        return sent
    sent = asyncio.run(go())
    assert sent[0]["type"] == "lifespan.startup.failed" and "no chip here" in sent[0]["message"]


def test_shutdown_waits_for_a_running_clip_before_it_closes_the_device(monkeypatch):
    gen = GatedGenerator()
    monkeypatch.setattr(asgi, "build_generator", lambda a: gen)

    async def go():
        inbox = asyncio.Queue()
        sent = []

        async def receive():
            return await inbox.get()

        async def send(msg):
            sent.append(msg["type"])
        task = asyncio.ensure_future(asgi.app({"type": "lifespan"}, receive, send))
        await inbox.put({"type": "lifespan.startup"})
        while not sent:
            await asyncio.sleep(0.01)
        run_in_thread = threading.Thread(target=lambda: submit())
        run_in_thread.start()
        while not gen.started.is_set():
            await asyncio.sleep(0.01)
        await inbox.put({"type": "lifespan.shutdown"})
        await asyncio.sleep(0.3)
        assert not task.done() and gen.closes == 0               # the clip is still running, so the device stays open
        gen.release.set()
        await asyncio.wait_for(task, 20)
        run_in_thread.join()
    asyncio.run(go())
    assert gen.closes == 1 and not gen.closed_while_running


# ---- settings

def test_settings_come_from_the_environment_with_documented_defaults():
    a = asgi.settings({})
    assert (a.generator, a.model, a.max_waiting, a.max_seconds, a.host, a.port, a.host_decoder, a.warmup_seconds) == ("chip", "small-music", 2, 120, "127.0.0.1", 20170, False, 0)
    b = asgi.settings({"AUDIO_GENERATOR": "fake", "AUDIO_MAX_WAITING": "0", "AUDIO_MAX_SECONDS": "30", "HOST": "0.0.0.0", "PORT": "20172",
                       "AUDIO_HOST_DECODER": "1", "AUDIO_WARMUP_SECONDS": "5"})
    assert (b.generator, b.max_waiting, b.max_seconds, b.host, b.port, b.host_decoder, b.warmup_seconds) == ("fake", 0, 30, "0.0.0.0", 20172, True, 5)


@pytest.mark.parametrize("env", [{"AUDIO_GENERATOR": "gpu"}, {"AUDIO_MAX_WAITING": "-1"}, {"AUDIO_MAX_SECONDS": "0"}, {"AUDIO_MAX_SECONDS": "500"},
                                 {"AUDIO_MAX_WAITING": "x"}, {"AUDIO_WARMUP_SECONDS": "-2"}])
def test_a_bad_setting_is_refused_with_its_name(env):
    with pytest.raises(ValueError, match="AUDIO_"):
        asgi.settings(env)


def test_the_fake_generator_is_chosen_by_the_setting_and_the_chip_one_by_default():
    assert isinstance(asgi.build_generator(asgi.settings({"AUDIO_GENERATOR": "fake"})), jobs.FakeGenerator)
    g = asgi.build_generator(asgi.settings({}))
    assert isinstance(g, asgi.ResidentChipGenerator) and g.max_seconds == 120 and g.chip_decoder is True
    assert asgi.build_generator(asgi.settings({"AUDIO_HOST_DECODER": "1"})).chip_decoder is False


# ---- the resident generator, with a fake runtime

class FakeRuntime:
    def __init__(self):
        self.calls, self.loads, self.closed = [], 0, 0

    def load(self):
        self.loads += 1
        return {"model_load_s": 1.5}

    def generate(self, settings, out_dir=None, extra_timings=None):
        self.calls.append((dict(settings), out_dir))
        out = Path(out_dir)
        (out / "clips").mkdir(parents=True)
        (out / "clips" / "audio.flac").write_bytes(FLAC_10S)
        (out / "chip.json").write_text(json.dumps({"timings": {"generate_and_decode_s": 0.8, "chip_passes": 8, "host_loadavg_1m": 3.0, "note": "x"}}))
        return {"timings": {}, "output_sha256": "x", "dir": str(out)}

    def close(self):
        self.closed += 1


def flac_bytes(seconds, rate=44100):
    import struct
    packed = (rate << 44) | (1 << 41) | (15 << 36) | int(seconds * rate)
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + b"\0" * 10 + struct.pack(">Q", packed) + b"\0" * 16


FLAC_10S = flac_bytes(10)


def resident(tmp_path, **kw):
    rt = FakeRuntime()
    g = asgi.ResidentChipGenerator(runtime_factory=lambda: rt, work_dir=str(tmp_path / "work"), **kw)
    return g, rt


def test_the_resident_generator_loads_once_and_passes_the_job_settings_to_the_runtime(tmp_path):
    g, rt = resident(tmp_path)
    g.load()
    g.load()
    assert rt.loads == 1
    clip = g.generate(jobs.JobRequest("a tune", 10, 8, 1.0, 42), lambda s: None)
    settings, out = rt.calls[0]
    assert settings == {"prompt": "a tune", "duration": 10, "steps": 8, "cfg_scale": 1.0, "seed": 42, "sample_size": asgi.SAMPLE_SIZE}
    assert clip.audio == FLAC_10S and clip.audio_format == "flac" and clip.audio_s == 10.0


def test_only_the_timings_that_are_times_reach_the_clip_and_the_run_directory_is_removed(tmp_path):
    g, rt = resident(tmp_path)
    g.load()
    clip = g.generate(jobs.JobRequest("a tune", 10, 8, 1.0, 42), lambda s: None)
    assert clip.stages == {"generate_and_decode_s": 0.8}
    assert not Path(rt.calls[0][1]).exists() and list((tmp_path / "work").iterdir()) == []


def test_generating_before_the_load_is_an_error(tmp_path):
    g, _ = resident(tmp_path)
    with pytest.raises(RuntimeError, match="not loaded"):
        g.generate(jobs.JobRequest("a", 10, 8, 1.0, 1), lambda s: None)


def test_a_runtime_that_writes_no_clip_is_an_error_and_still_cleans_up(tmp_path):
    g, rt = resident(tmp_path)
    g.load()
    rt.generate = lambda settings, out_dir=None, extra_timings=None: Path(out_dir).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="without writing a clip"):
        g.generate(jobs.JobRequest("a", 10, 8, 1.0, 1), lambda s: None)
    assert list((tmp_path / "work").iterdir()) == []


def test_close_closes_the_runtime_once_and_a_second_close_does_nothing(tmp_path):
    g, rt = resident(tmp_path)
    g.load()
    g.close()
    g.close()
    assert rt.closed == 1


def test_a_warm_up_runs_one_unrecorded_generation_of_the_asked_length_after_the_load(tmp_path):
    g, rt = resident(tmp_path, warmup_seconds=3)
    g.load()
    assert len(rt.calls) == 1 and rt.calls[0][0]["duration"] == 3
    g2, rt2 = resident(tmp_path)
    g2.load()
    assert rt2.calls == []


def test_the_generator_limits_are_the_job_servers(tmp_path):
    g, _ = resident(tmp_path, max_seconds=30)
    assert g.max_seconds == 30 and g.min_seconds == jobs.MIN_SECONDS and g.default_steps == 8 and g.default_cfg_scale == 1.0
    with pytest.raises(ValueError, match="max_seconds"):
        resident(tmp_path, max_seconds=121)


# ---- the pinned weights

REV = "8bdfb53c6b0fbc62113cd571fd37373efe15fa14"


def fake_download(tmp_path, name=REV):
    def download(repo, revision):
        snap = tmp_path / "hub" / "models--stabilityai--stable-audio-3-small-music" / "snapshots" / name
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "model.safetensors").write_text("x")
        download.seen = (repo, revision)
        return str(snap)
    return download


def test_the_pinned_revision_is_downloaded_and_a_private_cache_resolves_main_to_it(tmp_path):
    env = {"AUDIO_PKG_CACHE": str(tmp_path / "pkgcache")}
    download = fake_download(tmp_path)
    private = asgi.resolve_weights(env, download, repo="stabilityai/stable-audio-3-small-music", revision=REV)
    assert download.seen == ("stabilityai/stable-audio-3-small-music", REV)
    assert Path(private).is_dir() and str(private).startswith(str(tmp_path / "pkgcache"))
    repo = Path(private) / "models--stabilityai--stable-audio-3-small-music"
    assert (repo / "refs" / "main").read_text() == REV and (repo / "snapshots" / REV).is_symlink()
    assert env["HF_HUB_CACHE"] == str(private)


def test_a_download_that_lands_on_another_revision_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="revision"):
        asgi.resolve_weights({"AUDIO_PKG_CACHE": str(tmp_path / "c")}, fake_download(tmp_path, "f" * 40), repo="a/b", revision=REV)


def test_the_environment_can_override_the_repo_and_revision_the_launcher_sets(tmp_path):
    env = {"AUDIO_PKG_CACHE": str(tmp_path / "c"), "HF_MODEL": "other/model", "TT_MODEL_WEIGHTS_REVISION": "1" * 40}
    download = fake_download(tmp_path, "1" * 40)
    asgi.resolve_weights(env, download)
    assert download.seen == ("other/model", "1" * 40)
