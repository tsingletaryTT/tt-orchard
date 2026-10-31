"""The ACE-Step ASGI front end (pkgsrc/server/acestep_asgi.py), driven without a server process or a chip.

A small ASGI driver calls `app` with a scope, a body and a `receive`/`send` pair. The session is a fake with the surface of
`acestep_run.Session` (`one(save_dir, duration, prompt, lyrics, seed) -> (wav path, timings)` and `worker.failed`), so these tests check the
HTTP behaviour (routes, status codes, the one-request slot, an unusable chip, a client that leaves) and the choice of inputs at start-up. They do
not check the music."""
import asyncio
import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("acestep_asgi_under_test", ROOT / "pkgsrc" / "server" / "acestep_asgi.py")
asgi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asgi)      # acestep_serve.py is found in scripts/acestep, where `_core()` looks when `acestep_serve` is not importable
serve = asgi.serve

WAV = b"RIFF....WAVEfmt fake audio bytes"


class FakeSession:
    """What `acestep_run.Session` gives the server."""

    def __init__(self, fail=None):
        self.calls, self.fail, self.worker = [], fail, SimpleNamespace(failed=False)
        self.release, self.entered, self.dirs = None, threading.Event(), []

    def one(self, save_dir, duration, prompt, lyrics="", seed=42):
        self.calls.append((duration, prompt, lyrics, seed))
        self.dirs.append(Path(save_dir))
        self.entered.set()
        if self.release is not None:
            self.release.wait(10)
        if self.fail is not None:
            raise self.fail
        wav = Path(save_dir) / "clip.wav"
        wav.write_bytes(WAV)
        return wav, {"generate_and_decode_s": 0.9}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def state(tmp_path):
    work = tmp_path / "clips-in-flight"
    work.mkdir()
    asgi.STATE.update(session=FakeSession(), work=work)
    yield
    if asgi.LOCK.locked():
        asgi.LOCK.release()
    asgi.STATE.clear()


async def call(method, path, body=b"", disconnect_after_start=False):
    """Returns (status, headers, body bytes). `disconnect_after_start`: the client leaves as soon as the request body was read."""
    sent, parts = [], []
    inbox = [{"type": "http.request", "body": body, "more_body": False}]
    gate = asyncio.Event()
    if disconnect_after_start:
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

    await asyncio.wait_for(asgi.app({"type": "http", "method": method, "path": path}, receive, send), 20)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}, b"".join(parts)


def music(payload, **kw):
    return call("POST", "/v1/music", json.dumps(payload).encode() if not isinstance(payload, bytes) else payload, **kw)


def run(coro):
    return asyncio.run(coro)


REQ = {"prompt": "calm lofi piano", "duration": 12}


def test_health_and_unknown_paths():
    status, _, body = run(call("GET", "/health"))
    assert status == 200 and json.loads(body) == {"status": "ok"}
    assert run(call("GET", "/nope"))[0] == 404
    assert run(call("GET", "/v1/music"))[0] == 404          # only POST makes a clip
    assert run(call("POST", "/nope", b"{}"))[0] == 404


def test_a_request_gives_the_wav_its_headers_and_removes_its_work_directory():
    status, headers, body = run(music({**REQ, "lyrics": "la", "seed": 7}))
    assert status == 200 and body == WAV and headers["content-type"] == "audio/wav"
    assert headers["x-sample-rate"] == "48000"
    timings = json.loads(headers["x-timings"])
    assert timings["generate_and_decode_s"] == 0.9 and timings["request_s"] >= 0
    session = asgi.STATE["session"]
    assert session.calls == [(12.0, "calm lofi piano", "la", 7)]
    assert not session.dirs[0].exists()                    # the clip is in memory; its directory is gone
    assert not asgi.LOCK.locked()


def test_defaults_match_the_standard_library_server():
    run(music({"prompt": "x"}))
    assert asgi.STATE["session"].calls == [(30.0, "x", "", 42)]


@pytest.mark.parametrize("payload", [b"not json", b"[]", {}, {"prompt": ""}, {"prompt": "x", "duration": 4}, {"prompt": "x", "duration": 481},
                                     {"prompt": "x", "duration": True}, {"prompt": "x", "seed": -1}, {"prompt": "x", "bogus": 1},
                                     {"prompt": "x" * 2001}, {"prompt": "x", "lyrics": "y" * 8001}])
def test_bad_requests_get_a_json_400_and_never_reach_the_chip(payload):
    status, _, body = run(music(payload))
    assert status == 400 and "error" in json.loads(body)
    assert asgi.STATE["session"].calls == []
    assert not asgi.LOCK.locked()


def test_empty_and_oversized_bodies_are_refused():
    status, _, body = run(music(b""))
    assert status == 400 and b"1 to" in body               # refused as an empty body, not as bad JSON
    valid_but_large = json.dumps(REQ).encode() + b" " * serve.MAX_BODY       # valid JSON that is over the limit
    status, _, body = run(music(valid_but_large))
    assert status == 400 and b"1 to" in body
    assert asgi.STATE["session"].calls == []


def test_a_second_request_while_one_runs_gets_503_with_retry_after_and_does_not_wait():
    assert asgi.LOCK.acquire(blocking=False)               # stands in for the running request
    t0 = time.perf_counter()
    status, headers, _ = run(music(REQ))
    assert status == 503 and headers["retry-after"] == "5" and time.perf_counter() - t0 < 1.0
    assert asgi.STATE["session"].calls == []
    assert asgi.LOCK.locked()                              # the refused request did not release the other one's slot


def test_an_unusable_chip_is_reported_even_while_the_slot_is_taken():
    asgi.STATE["session"].worker.failed = True
    assert asgi.LOCK.acquire(blocking=False)
    status, headers, body = run(music(REQ))
    assert status == 503 and b"unusable" in body and "retry-after" not in headers      # retrying shortly would not help


def test_a_chip_that_fails_while_the_request_waits_for_the_slot_is_caught_under_the_lock(monkeypatch):
    """The request that held the slot may have failed a chip call after this one's first look; the second look under the lock sees it."""
    looks = iter([None, "a chip call failed"])
    monkeypatch.setattr(asgi, "_unusable", lambda: next(looks))
    status, _, body = run(music(REQ))
    assert status == 503 and b"unusable" in body
    assert asgi.STATE["session"].calls == [] and not asgi.LOCK.locked()


def test_an_unusable_chip_gives_503_on_health_and_on_music():
    asgi.STATE["session"].worker.failed = True
    status, _, body = run(call("GET", "/health"))
    assert status == 503 and json.loads(body)["status"] == "unusable"
    status, _, body = run(music(REQ))
    assert status == 503 and b"unusable" in body
    assert asgi.STATE["session"].calls == []
    assert not asgi.LOCK.locked()


def test_a_failed_generation_gives_500_with_the_type_only_and_frees_the_slot():
    asgi.STATE["session"] = FakeSession(fail=RuntimeError("secret model text and /home/x/path"))
    status, _, body = run(music(REQ))
    assert status == 500 and b"RuntimeError" in body and b"secret" not in body and b"/home" not in body
    assert not asgi.LOCK.locked()
    assert not asgi.STATE["session"].dirs[0].exists()


def test_health_answers_while_a_clip_generates():
    """The chip work is in a worker thread, so the event loop is free."""
    session = asgi.STATE["session"]
    session.release = threading.Event()

    async def both():
        clip = asyncio.ensure_future(music(REQ))
        await asyncio.get_running_loop().run_in_executor(None, session.entered.wait, 10)
        health = await call("GET", "/health")
        busy = await music(REQ)
        session.release.set()
        return health, busy, await clip
    health, busy, clip = run(both())
    assert health[0] == 200 and busy[0] == 503 and clip[0] == 200 and clip[2] == WAV
    assert not asgi.LOCK.locked()


def test_a_client_that_leaves_during_generation_does_not_leak_the_slot_or_the_directory():
    """ACE-Step cannot stop a clip half way, so the slot is taken until the generation ends; the worker releases it then."""
    session = asgi.STATE["session"]
    session.release = threading.Event()

    async def leave():
        task = asyncio.ensure_future(music(REQ, disconnect_after_start=True))
        await asyncio.get_running_loop().run_in_executor(None, session.entered.wait, 10)
        assert asgi.LOCK.locked()
        session.release.set()
        return await task
    run(leave())
    assert not asgi.LOCK.locked()
    assert not session.dirs[0].exists()


# ---- settings and start-up -------------------------------------------------------------------------------------------------

def test_settings_come_from_the_environment():
    a = asgi.settings({"ACESTEP_FAST": "none", "ACESTEP_WORK": "/w", "ACESTEP_BUNDLE_CODE": "/c", "ACESTEP_SNAPSHOT": "/s", "HF_MODEL": "o/m",
                       "TT_MODEL_WEIGHTS_REVISION": "r"})
    assert (a.fast, str(a.work), a.bundle_code, a.snapshot, a.model, a.revision) == ("none", "/w", "/c", "/s", "o/m", "r")
    d = asgi.settings({"HOME": "/h"})
    assert (d.fast, d.bundle_code, d.snapshot) == ("all", None, None)
    assert str(d.work).startswith("/h/") and d.model == asgi.MODEL_ID and d.revision == asgi.PINS["model_revision"]


def test_the_default_work_directory_follows_xdg_cache_home():
    assert str(asgi.settings({"XDG_CACHE_HOME": "/x", "HOME": "/h"}).work).startswith("/x/")


def make_code(tmp_path):
    code = tmp_path / "code"
    (code / "acestep").mkdir(parents=True)
    (code / "acestep" / "a.py").write_text("x = 1\n")
    return code


def make_snapshot(tmp_path, revision):
    snap = tmp_path / "snapshots" / revision
    snap.mkdir(parents=True)
    return snap


def test_resolve_inputs_uses_directories_it_is_given_and_downloads_nothing(tmp_path, monkeypatch):
    code, snap = make_code(tmp_path), make_snapshot(tmp_path, "r1")
    monkeypatch.setattr(asgi, "problems_of", lambda c, s: [])
    monkeypatch.setattr(asgi, "download_code", lambda *a, **k: pytest.fail("downloaded the code"))
    monkeypatch.setattr(asgi, "download_weights", lambda *a, **k: pytest.fail("downloaded the weights"))
    a = asgi.settings({"ACESTEP_BUNDLE_CODE": str(code), "ACESTEP_SNAPSHOT": str(snap), "ACESTEP_WORK": str(tmp_path / "w")})
    assert asgi.resolve_inputs(a) == (code, snap)


def test_resolve_inputs_downloads_the_pinned_revisions_into_the_work_directory(tmp_path, monkeypatch):
    asked = {}

    def fake_code(work, repo, revision):
        asked["code"] = (repo, revision)
        return make_code(tmp_path)

    def fake_weights(model, revision):
        asked["weights"] = (model, revision)
        return make_snapshot(tmp_path, revision)
    monkeypatch.setattr(asgi, "download_code", fake_code)
    monkeypatch.setattr(asgi, "download_weights", fake_weights)
    monkeypatch.setattr(asgi, "problems_of", lambda c, s: [])
    asgi.resolve_inputs(asgi.settings({"ACESTEP_WORK": str(tmp_path / "w"), "HF_MODEL": "ACE-Step/Ace-Step1.5", "TT_MODEL_WEIGHTS_REVISION": "abc"}))
    assert asked["code"] == (asgi.PINS["bundle_repo"], asgi.PINS["bundle_revision"])
    assert asked["weights"] == ("ACE-Step/Ace-Step1.5", "abc")


def test_resolve_inputs_refuses_inputs_that_do_not_match_the_pins(tmp_path, monkeypatch):
    code, snap = make_code(tmp_path), make_snapshot(tmp_path, "r1")
    monkeypatch.setattr(asgi, "problems_of", lambda c, s: ["bundle code holds 2 link(s)"])
    a = asgi.settings({"ACESTEP_BUNDLE_CODE": str(code), "ACESTEP_SNAPSHOT": str(snap), "ACESTEP_WORK": str(tmp_path / "w")})
    with pytest.raises(RuntimeError, match="holds 2 link"):
        asgi.resolve_inputs(a)


def test_problems_of_uses_the_pins_and_refuses_a_code_directory_of_links(tmp_path):
    code = make_code(tmp_path)
    (code / "link.py").symlink_to(code / "acestep" / "a.py")
    snap = make_snapshot(tmp_path, asgi.PINS["model_revision"])
    found = asgi.problems_of(code, snap)
    assert any("link" in p for p in found) and any("pinned" in p for p in found)


def install_fake_runtime(monkeypatch, captured):
    """Stand-ins for `acestep_run` (the Session) and `acestep_fast` (the steps), so start-up runs without torch or a chip."""
    run_mod, fast_mod = types.ModuleType("acestep_run"), types.ModuleType("acestep_fast")

    class Session(FakeSession):
        WEIGHTS_REVISION = "rev"

        def __init__(self, **kw):
            super().__init__()
            captured.update(kw)
            self.fast = tuple(kw["fast"])
            self.load = {}
    run_mod.Session = Session
    fast_mod.parse_steps = lambda text: () if text == "none" else tuple(text.split(","))
    monkeypatch.setitem(sys.modules, "acestep_run", run_mod)
    monkeypatch.setitem(sys.modules, "acestep_fast", fast_mod)


def test_startup_opens_a_chip_session_on_the_resolved_inputs_and_prints_the_ready_line(tmp_path, monkeypatch, capsys):
    captured = {}
    install_fake_runtime(monkeypatch, captured)
    code, snap = make_code(tmp_path), make_snapshot(tmp_path, "r1")
    monkeypatch.setattr(asgi, "resolve_inputs", lambda a: (code, snap))
    a = asgi.settings({"ACESTEP_WORK": str(tmp_path / "w"), "ACESTEP_FAST": "cond_cache,gc_freeze"})
    close = asgi.startup(a)
    assert captured["device"] == "chip" and captured["bundle_code"] == code and captured["snapshot"] == snap
    assert captured["fast"] == ("cond_cache", "gc_freeze") and Path(captured["root"]).is_dir()
    assert asgi.STATE["session"] is not None and asgi.STATE["work"].is_dir()
    assert Path(captured["root"]).parent == tmp_path / "w" and asgi.STATE["work"].parent == tmp_path / "w"      # the project root is private to the work directory
    ready = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert ready["event"] == "ready" and ready["fast"] == ["cond_cache", "gc_freeze"] and ready["model"] == "ACE-Step/Ace-Step1.5"
    close()
    assert asgi.STATE["session"].closed


def test_an_unknown_fast_step_stops_start_up_before_the_chip_opens(tmp_path, monkeypatch):
    captured = {}
    install_fake_runtime(monkeypatch, captured)
    sys.modules["acestep_fast"].parse_steps = lambda text: (_ for _ in ()).throw(ValueError("unknown step"))
    monkeypatch.setattr(asgi, "resolve_inputs", lambda a: pytest.fail("fetched inputs before the steps were checked"))
    with pytest.raises(ValueError):
        asgi.startup(asgi.settings({"ACESTEP_WORK": str(tmp_path / "w"), "ACESTEP_FAST": "nope"}))
    assert captured == {}


def lifespan(messages):
    """Drives the app's lifespan with `messages` and returns what it sent."""
    sent, inbox = [], list(messages)

    async def receive():
        return inbox.pop(0)

    async def send(msg):
        sent.append(msg["type"])
    run(asgi.app({"type": "lifespan"}, receive, send))
    return sent


def test_lifespan_reports_a_start_up_failure(monkeypatch):
    monkeypatch.setattr(asgi, "startup", lambda a: (_ for _ in ()).throw(RuntimeError("bundle code is not the pinned revision")))
    assert lifespan([{"type": "lifespan.startup"}]) == ["lifespan.startup.failed"]


def test_lifespan_closes_the_chip_on_shutdown(monkeypatch):
    closed = []
    monkeypatch.setattr(asgi, "startup", lambda a: lambda: closed.append(1))
    assert lifespan([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]) == \
        ["lifespan.startup.complete", "lifespan.shutdown.complete"] and closed == [1]


# ---- the fetch of the model code and the weights -----------------------------------------------------------------------------

def fake_hub(monkeypatch, calls):
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo, **kw):
        calls.append((repo, kw))
        if kw.get("local_dir"):
            (Path(kw["local_dir"]) / "code" / "acestep").mkdir(parents=True, exist_ok=True)
            (Path(kw["local_dir"]) / "code" / "acestep" / "a.py").write_text("x = 1\n")
        return kw.get("local_dir") or "/cache/snapshots/" + kw["revision"]
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_the_code_is_fetched_as_real_files_at_the_pinned_revision_and_only_the_code_directory(tmp_path, monkeypatch):
    calls = []
    fake_hub(monkeypatch, calls)
    code = asgi.download_code(tmp_path, "o/bundle", "rev1")
    repo, kw = calls[0]
    assert repo == "o/bundle" and kw["revision"] == "rev1" and kw["allow_patterns"] == ["code/*"]
    assert Path(kw["local_dir"]) == tmp_path / "bundle" and code == tmp_path / "bundle" / "code"   # local_dir, so the files are not links into a cache


def test_a_code_directory_that_matches_the_pin_is_kept_and_a_different_one_is_fetched_again(tmp_path, monkeypatch):
    calls = []
    fake_hub(monkeypatch, calls)
    code = tmp_path / "bundle" / "code"
    (code / "acestep").mkdir(parents=True)
    (code / "acestep" / "a.py").write_text("x = 1\n")
    digest, n = asgi.check_bundle.tree_hash(code)
    monkeypatch.setitem(asgi.PINS, "bundle_tree_sha256", digest)
    monkeypatch.setitem(asgi.PINS, "bundle_files", n)
    asgi.download_code(tmp_path, "o/b", "r")
    assert calls == []                                            # it matches: nothing fetched
    (code / "acestep" / "a.py").write_text("x = 2\n")            # now it does not
    asgi.download_code(tmp_path, "o/b", "r")
    assert len(calls) == 1 and (code / "acestep" / "a.py").read_text() == "x = 1\n"


def test_a_matching_code_directory_with_a_link_in_it_is_fetched_again(tmp_path, monkeypatch):
    calls = []
    fake_hub(monkeypatch, calls)
    code = tmp_path / "bundle" / "code"
    (code / "acestep").mkdir(parents=True)
    (code / "acestep" / "a.py").write_text("x = 1\n")
    (code / "acestep" / "b.py").symlink_to(code / "acestep" / "a.py")
    monkeypatch.setattr(asgi.check_bundle, "tree_hash", lambda c: (asgi.PINS["bundle_tree_sha256"], asgi.PINS["bundle_files"]))
    asgi.download_code(tmp_path, "o/b", "r")
    assert len(calls) == 1 and not (code / "acestep" / "b.py").is_symlink()


def test_the_weights_fetch_names_the_revision_and_only_the_turbo_files(monkeypatch):
    calls = []
    fake_hub(monkeypatch, calls)
    snap = asgi.download_weights("ACE-Step/Ace-Step1.5", "abc")
    repo, kw = calls[0]
    assert repo == "ACE-Step/Ace-Step1.5" and kw["revision"] == "abc" and "local_dir" not in kw
    assert set(kw["allow_patterns"]) == {"acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*", "config.json"} and snap == Path("/cache/snapshots/abc")
