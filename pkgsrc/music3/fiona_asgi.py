# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ASGI app of the minimax-music3-fiona-crapple thin bundle (kind `tt-dit-server`): the MiniMax-Music3 server and a Gradio page.

`tt-model` starts this kind with `uvicorn <module>:app`. The model's own server is already a FastAPI app
(`models.autoports.minimaxai_minimax_music3.server.app:app`, code copied from jashansinghTT/minimax-music3-blackhole). It loads the model in its
lifespan startup, runs the chip in one worker thread, and answers 429 when its queue of jobs is full. Nothing of that is changed or copied here.
This file adds three things:

  * the Gradio page of episod/minimax-music3-fiona-crapple-blackhole, mounted at /ui with `gradio.mount_gradio_app`. The page calls this
    server's own job API over HTTP, as it does in the companion repository (`music_page.py` and `music_client.py` are that repository's
    `demo/app.py` and `demo/music_client.py`, unchanged);
  * `GET /` redirects to /ui when the page is on;
  * settings from the environment, because uvicorn passes no flags.

Settings (all optional; `run.sh` sets HF_MODEL, TT_MODEL_WEIGHTS_REVISION, HOST and PORT):

    HF_MODEL                     the weights: a Hugging Face repository id (downloaded at TT_MODEL_WEIGHTS_REVISION) or a local directory.
                                 Required. The model code falls back to the base MiniMax-Music3 weights when it is empty, and a bundle
                                 named for the merged weights must not start on the base weights without anyone noticing.
    MUSIC3_SNAPSHOT              a local directory with the weights; it takes precedence over HF_MODEL
    TT_MODEL_WEIGHTS_REVISION    the commit of the weights (becomes MUSIC3_WEIGHTS_REVISION unless that is set)
    MUSIC3_UI                    0 turns the page off (default 1)
    MUSIC_URL                    the server the page talks to (default http://127.0.0.1:$PORT, which is this server)
    TT_CACHE_PATH                the tensor cache run.sh chose; the app adds a subdirectory named for the weights (repository and revision, or the
                                 snapshot directory), so that two different weights never share a cache
    TT_DIT_CACHE_DIR             where the model code keeps its weight views and caches (default: TT_CACHE_PATH with "-dit" added, plus the same
                                 subdirectory; without TT_CACHE_PATH the model code's own default). An explicit value is kept as it is.
    HF_MODEL_ID                  the id /v1/models reports (default: HF_MODEL, or its last path part when that is a directory)
    MUSIC3_API_KEY               the server's own setting; the page does not send a key, so the page stops working when it is set

and the model code's own MUSIC3_* variables (MAX_SEQ_LEN, MAX_FRAMES, MAX_QUEUE, WARMUP, MESH_SHAPE, FABRIC_CONFIG, TRACE_REGION_MB and others).
The page has no login. Bind it to 127.0.0.1 (run.sh does) unless the network is trusted.
"""
import hashlib
import importlib
import os
from pathlib import Path

DEFAULT_PORT = "20150"
UI_PATH = "/ui"


def ui_enabled(environ=None) -> bool:
    e = os.environ if environ is None else environ
    return e.get("MUSIC3_UI", "1") != "0"


def ui_url(environ=None) -> str:
    """The server the page talks to: MUSIC_URL, or this server on the port uvicorn was started with."""
    e = os.environ if environ is None else environ
    return e.get("MUSIC_URL") or f"http://127.0.0.1:{e.get('PORT', DEFAULT_PORT)}"


def cache_key(environ=None) -> str:
    """A short name for the weights a server runs: the snapshot directory if one is named, else the repository id, and the revision. Two different
    weights never get the same name, so they never share a cache directory."""
    e = os.environ if environ is None else environ
    snapshot = e.get("MUSIC3_SNAPSHOT")
    model = e.get("HF_MODEL", "")
    if snapshot:
        identity = ["dir", os.path.realpath(snapshot)]
    elif os.path.isdir(model):
        identity = ["dir", os.path.realpath(model)]
    else:
        identity = ["repo", model, e.get("MUSIC3_WEIGHTS_REVISION") or e.get("TT_MODEL_WEIGHTS_REVISION") or ""]
    return hashlib.sha256("\0".join(identity).encode()).hexdigest()[:16]


def prepare_environment(environ=None) -> None:
    """Set what the model code reads before it is imported. Raises RuntimeError when no weights are named."""
    e = os.environ if environ is None else environ
    model = e.get("MUSIC3_SNAPSHOT") or e.get("HF_MODEL")
    if not model:
        raise RuntimeError("HF_MODEL names no weights. Start the server with the bundle's run.sh (it sets HF_MODEL and TT_MODEL_WEIGHTS_REVISION), "
                           "or set HF_MODEL to a repository id or a directory. Without it the model code loads the base MiniMax-Music3 weights.")
    if e.get("TT_MODEL_WEIGHTS_REVISION"):
        e.setdefault("MUSIC3_WEIGHTS_REVISION", e["TT_MODEL_WEIGHTS_REVISION"])
    e.setdefault("HF_MODEL_ID", Path(model).name if os.path.isdir(model) else model)
    if e.get("TT_CACHE_PATH"):
        # run.sh keeps every cache inside the bundle; the model code's weight views and caches would go under the home directory. Both caches get a
        # subdirectory named for the weights, because the model code reuses a language-model view without checking where it came from.
        key = cache_key(e)
        if os.path.basename(e["TT_CACHE_PATH"].rstrip("/")) != key:
            e["TT_CACHE_PATH"] = e["TT_CACHE_PATH"].rstrip("/") + "/" + key
            if "TT_DIT_CACHE_DIR" not in e:
                e["TT_DIT_CACHE_DIR"] = e["TT_CACHE_PATH"].rsplit("/", 2)[0] + "/" + e["TT_CACHE_PATH"].rsplit("/", 2)[1] + "-dit/" + key
    e.setdefault("GRADIO_ANALYTICS_ENABLED", "False")      # the page makes no call to a service outside this machine


def make_app(environ=None):
    """The server's FastAPI app, with the page mounted at /ui unless MUSIC3_UI=0. Importing the server does not touch a device: the chip opens in
    its lifespan startup, which the mount keeps."""
    prepare_environment(environ)
    server = importlib.import_module("models.autoports.minimaxai_minimax_music3.server.app")
    app = server.app
    if ui_enabled(environ) and not any(getattr(r, "path", None) == UI_PATH for r in app.routes):
        import gradio as gr
        from fastapi.responses import RedirectResponse
        from music_client import MusicClient
        from music_page import build_ui
        client = MusicClient(ui_url(environ))
        page = build_ui(client).queue(max_size=8, default_concurrency_limit=1)
        app = gr.mount_gradio_app(app, page, path=UI_PATH)
        app.state.music_client = client

        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse(UI_PATH)
    return app


app = make_app()
