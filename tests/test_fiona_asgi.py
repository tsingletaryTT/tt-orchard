"""The ASGI app of the minimax-music3-fiona-crapple bundle (pkgsrc/music3/fiona_asgi.py), driven without a server process or a chip.

The app is the model's own FastAPI server (code copied from the Hub package, pkgsrc/music3/code) with the Gradio page mounted at /ui. Importing it
starts no thread and opens no device; these tests call it through httpx's ASGI transport, which does not run the lifespan. One test runs the
lifespan with a fake engine to check that the mount keeps the server's own startup."""
import asyncio
import importlib
import sys
from pathlib import Path

import httpx
import pytest


def _gradio_usable():
    """The page needs gradio, and the gradio that some hosts have is too old for the installed huggingface_hub. Those tests then skip; they
    run in the bundle's own venv, where install.sh pins gradio 6.20.0."""
    try:
        import gradio  # noqa: F401
    except Exception:  # noqa: BLE001  any import failure means the page cannot be built here
        return False
    return True


needs_gradio = pytest.mark.skipif(not _gradio_usable(), reason="gradio is not importable in this environment")

ROOT = Path(__file__).resolve().parent.parent
MUSIC = ROOT / "pkgsrc" / "music3"
PATHS = [str(MUSIC), str(MUSIC / "code"), str(MUSIC / "demo")]
OWN = ("fiona_asgi", "music_page", "music_client", "models")
REV = "53fc7557ea3101d3efbe5e50598bd0bea7f44344"
ENV_KEYS = ("TT_DIT_CACHE_DIR", "TT_CACHE_PATH", "HF_MODEL", "MUSIC3_SNAPSHOT", "TT_MODEL_WEIGHTS_REVISION", "MUSIC3_WEIGHTS_REVISION", "MUSIC3_UI", "MUSIC_URL", "PORT",
            "HF_MODEL_ID", "GRADIO_ANALYTICS_ENABLED", "MUSIC3_API_KEY")


@pytest.fixture
def load(monkeypatch):
    """load(**env) imports a fresh copy of the adapter and the model's server under that environment and returns the adapter module.
    The page is off unless a test sets MUSIC3_UI=1 (those tests carry `needs_gradio`)."""
    def go(**env):
        env = {"MUSIC3_UI": "0", **env}
        for k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        for name in [m for m in sys.modules if m.split(".")[0] in OWN]:
            monkeypatch.delitem(sys.modules, name)
        for p in PATHS:
            monkeypatch.syspath_prepend(p)
        return importlib.import_module("fiona_asgi")
    yield go
    for name in [m for m in sys.modules if m.split(".")[0] in OWN]:
        sys.modules.pop(name, None)


def call(app, method, path, **kw):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", follow_redirects=False) as c:
            return await c.request(method, path, **kw)
    return asyncio.run(go())


def test_the_model_servers_routes_are_still_served(load):
    mod = load(HF_MODEL="org/weights")
    r = call(mod.app, "GET", "/v1/music/jobs/nope")
    assert r.status_code == 404 and r.json()["detail"] == "job not found"       # the model server's own answer, not the page's
    r = call(mod.app, "GET", "/v1/health")
    assert r.status_code == 503 and r.json()["status"] != "ok"                   # no model is loaded in a test
    assert call(mod.app, "GET", "/v1/models").status_code == 200


@needs_gradio
def test_the_page_is_served_at_ui_and_carries_the_required_name(load):
    mod = load(HF_MODEL="org/weights", MUSIC3_UI="1")
    r = call(mod.app, "GET", "/ui/", headers={"accept": "text/html"})
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "MiniMax-Music3 on Tenstorrent" in r.text      # the license asks products to display "MiniMax-Music3"


@needs_gradio
def test_the_root_redirects_to_the_page(load):
    mod = load(HF_MODEL="org/weights", MUSIC3_UI="1")
    r = call(mod.app, "GET", "/")
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui"


def test_the_page_can_be_turned_off(load):
    mod = load(HF_MODEL="org/weights", MUSIC3_UI="0")
    assert call(mod.app, "GET", "/ui/").status_code == 404
    assert call(mod.app, "GET", "/").status_code == 404
    assert call(mod.app, "GET", "/v1/models").status_code == 200        # the API stays


@needs_gradio
def test_the_page_talks_to_this_server_on_the_port_uvicorn_was_given(load):
    mod = load(HF_MODEL="org/weights", PORT="20155", MUSIC3_UI="1")
    assert mod.app.state.music_client.base == "http://127.0.0.1:20155"


@needs_gradio
def test_the_page_url_defaults_to_the_bundles_port_and_music_url_overrides_it(load):
    assert load(HF_MODEL="org/weights", MUSIC3_UI="1").app.state.music_client.base == "http://127.0.0.1:20150"
    assert load(HF_MODEL="org/weights", MUSIC3_UI="1", MUSIC_URL="http://elsewhere:1234/").app.state.music_client.base == "http://elsewhere:1234"


def test_it_refuses_to_start_without_weights_and_says_what_to_do(load):
    with pytest.raises(RuntimeError, match="run.sh"):
        load()


def test_a_snapshot_directory_counts_as_naming_the_weights(load, tmp_path):
    mod = load(MUSIC3_SNAPSHOT=str(tmp_path))
    assert call(mod.app, "GET", "/v1/models").status_code == 200


def test_the_weights_revision_from_tt_model_reaches_the_model_code(load, monkeypatch):
    load(HF_MODEL="org/weights", TT_MODEL_WEIGHTS_REVISION=REV)
    import os
    assert os.environ["MUSIC3_WEIGHTS_REVISION"] == REV


def test_an_explicit_music3_revision_is_not_overridden(load):
    load(HF_MODEL="org/weights", TT_MODEL_WEIGHTS_REVISION=REV, MUSIC3_WEIGHTS_REVISION="b" * 40)
    import os
    assert os.environ["MUSIC3_WEIGHTS_REVISION"] == "b" * 40


def test_the_models_route_names_the_weights_and_not_the_base_model(load):
    mod = load(HF_MODEL="org/merged-weights")
    assert call(mod.app, "GET", "/v1/models").json()["data"][0]["id"] == "org/merged-weights"


def test_a_directory_is_named_by_its_last_part(load, tmp_path):
    d = tmp_path / "fiona-merged"
    d.mkdir()
    mod = load(HF_MODEL=str(d))
    assert call(mod.app, "GET", "/v1/models").json()["data"][0]["id"] == "fiona-merged"


def test_the_page_makes_no_analytics_call(load):
    load(HF_MODEL="org/weights")
    import os
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"


def test_importing_the_app_starts_no_worker_and_opens_no_device(load):
    mod = load(HF_MODEL="org/weights")
    engine = sys.modules["models.autoports.minimaxai_minimax_music3.server.app"].engine
    assert not engine._thread.is_alive() and not engine.ready.is_set()


@needs_gradio
def test_the_mount_keeps_the_servers_own_lifespan_startup(load):
    mod = load(HF_MODEL="org/weights", MUSIC3_UI="1")
    engine = sys.modules["models.autoports.minimaxai_minimax_music3.server.app"].engine
    calls = []
    engine.start = lambda: calls.append("start")
    engine.wait_ready = lambda timeout=None: calls.append("wait") or True

    async def run():
        async with mod.app.router.lifespan_context(mod.app):
            calls.append("serving")
    asyncio.run(run())
    assert calls[:3] == ["start", "wait", "serving"]


@needs_gradio
def test_mounting_twice_does_not_add_a_second_page(load):
    mod = load(HF_MODEL="org/weights", MUSIC3_UI="1")
    mod.make_app()
    assert sum(1 for r in mod.app.routes if getattr(r, "path", None) == "/ui") == 1
    assert sum(1 for r in mod.app.routes if getattr(r, "path", None) == "/" and "GET" in getattr(r, "methods", ())) == 1


def cache_dirs(load, **env):
    import os
    load(HF_MODEL="org/weights", TT_CACHE_PATH="/bundle/.tt_cache", **env)
    return os.environ["TT_CACHE_PATH"], os.environ["TT_DIT_CACHE_DIR"]


def test_the_caches_go_under_the_directories_run_sh_chose(load):
    tensor, dit = cache_dirs(load)
    assert tensor.startswith("/bundle/.tt_cache/") and dit.startswith("/bundle/.tt_cache-dit/")      # the model code would otherwise write under the home directory
    assert tensor.rsplit("/", 1)[1] == dit.rsplit("/", 1)[1]


def test_two_revisions_never_share_a_cache_directory(load):
    a = cache_dirs(load, TT_MODEL_WEIGHTS_REVISION="a" * 40)
    b = cache_dirs(load, TT_MODEL_WEIGHTS_REVISION="b" * 40)
    assert a[0] != b[0] and a[1] != b[1]


def test_two_weight_repositories_never_share_a_cache_directory(load):
    import os
    a = cache_dirs(load, TT_MODEL_WEIGHTS_REVISION=REV)
    load(HF_MODEL="org/other-weights", TT_CACHE_PATH="/bundle/.tt_cache", TT_MODEL_WEIGHTS_REVISION=REV)
    assert os.environ["TT_CACHE_PATH"] != a[0] and os.environ["TT_DIT_CACHE_DIR"] != a[1]


def test_a_local_snapshot_directory_and_a_repository_never_share_a_cache_directory(load, tmp_path):
    import os
    a = cache_dirs(load, TT_MODEL_WEIGHTS_REVISION=REV)
    load(MUSIC3_SNAPSHOT=str(tmp_path), HF_MODEL="org/weights", TT_CACHE_PATH="/bundle/.tt_cache", TT_MODEL_WEIGHTS_REVISION=REV)
    assert os.environ["TT_CACHE_PATH"] != a[0]


def test_two_model_directories_never_share_a_cache_directory(load, tmp_path):
    import os
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    load(HF_MODEL=str(tmp_path / "one"), TT_CACHE_PATH="/bundle/.tt_cache")
    one = os.environ["TT_CACHE_PATH"]
    load(HF_MODEL=str(tmp_path / "two"), TT_CACHE_PATH="/bundle/.tt_cache")
    assert os.environ["TT_CACHE_PATH"] != one


def test_two_snapshot_directories_never_share_a_cache_directory(load, tmp_path):
    import os
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    load(MUSIC3_SNAPSHOT=str(tmp_path / "one"), TT_CACHE_PATH="/bundle/.tt_cache")
    one = os.environ["TT_CACHE_PATH"]
    load(MUSIC3_SNAPSHOT=str(tmp_path / "two"), TT_CACHE_PATH="/bundle/.tt_cache")
    assert os.environ["TT_CACHE_PATH"] != one


def test_the_same_weights_and_revision_share_a_cache_directory_across_restarts(load):
    assert cache_dirs(load, TT_MODEL_WEIGHTS_REVISION=REV) == cache_dirs(load, TT_MODEL_WEIGHTS_REVISION=REV)


def test_an_explicit_music3_revision_decides_the_cache_directory_over_the_tt_model_one(load):
    a = cache_dirs(load, TT_MODEL_WEIGHTS_REVISION=REV, MUSIC3_WEIGHTS_REVISION="b" * 40)
    b = cache_dirs(load, TT_MODEL_WEIGHTS_REVISION="c" * 40, MUSIC3_WEIGHTS_REVISION="b" * 40)
    assert a == b


def test_preparing_the_environment_twice_does_not_nest_the_key(load):
    import os
    mod = load(HF_MODEL="org/weights", TT_CACHE_PATH="/bundle/.tt_cache")
    once = os.environ["TT_CACHE_PATH"]
    mod.prepare_environment()
    assert os.environ["TT_CACHE_PATH"] == once


def test_an_explicit_weight_cache_is_kept(load):
    load(HF_MODEL="org/weights", TT_CACHE_PATH="/bundle/.tt_cache", TT_DIT_CACHE_DIR="/elsewhere")
    import os
    assert os.environ["TT_DIT_CACHE_DIR"] == "/elsewhere"


def test_without_a_tensor_cache_path_the_caches_are_left_to_the_model_code(load):
    load(HF_MODEL="org/weights")
    import os
    assert "TT_DIT_CACHE_DIR" not in os.environ and "TT_CACHE_PATH" not in os.environ
