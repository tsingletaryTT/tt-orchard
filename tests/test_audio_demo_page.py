# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Gradio page (audio_app.py). Gradio is an optional dependency: these tests are skipped where it cannot be imported, and the rest of the suite
(tests/test_audio_demo_client.py runs the page's whole generation logic) does not need it.

To run them: `pip install "gradio>=6.20,<7"` in the interpreter that runs pytest."""
import socket
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "orchard" / "audio_demo"))
sys.path.insert(0, str(HERE))

gr = pytest.importorskip("gradio", exc_type=ImportError)
gradio_client = pytest.importorskip("gradio_client", exc_type=ImportError)

import audio_app as app  # noqa: E402
import audio_client as cl  # noqa: E402
from test_audio_demo_server import Live, Scripted  # noqa: E402


@pytest.fixture
def page():
    server = Live(Scripted())
    server.wait_ready()
    demo = app.build_ui(cl.AudioClient(server.base, timeout=5), poll_s=0.01).queue()
    with socket.socket() as probe:                       # a free high port chosen by the system, never a fixed one
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    _, url, _ = demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
    yield gradio_client.Client(url, verbose=False), demo, server
    demo.close()
    server.close()


def labels(demo):
    return {getattr(b, "label", None) for b in demo.blocks.values()}


def test_the_page_has_every_control_the_model_takes(page):
    _, demo, _ = page
    for label in ("Prompt", "Length (seconds)", "Steps", "Guidance scale (cfg)", "Seed", "Result"):
        assert label in labels(demo), label
    buttons = {b.value for b in demo.blocks.values() if isinstance(b, gr.Button)}
    assert {"Generate", "Random seed"} <= buttons


def test_the_benchmark_prompt_is_the_default_and_an_example(page):
    _, demo, _ = page
    box = next(b for b in demo.blocks.values() if getattr(b, "label", None) == "Prompt")
    assert box.value == cl.BENCHMARK_PROMPT
    examples = [b for b in demo.blocks.values() if isinstance(b, gr.Dataset)]
    assert any(cl.BENCHMARK_PROMPT in [row[0] for row in ex.samples] for ex in examples)


def test_the_slider_limits_come_from_the_server(page):
    _, demo, _ = page
    length = next(b for b in demo.blocks.values() if getattr(b, "label", None) == "Length (seconds)")
    assert (length.minimum, length.maximum) == (1, 120)


def test_generate_through_the_page_returns_a_clip_and_the_echo(page):
    client, _, _ = page
    path, status, details, server_line = client.predict("a tune", 10, 8, 1.0, 5, api_name="/generate")
    assert "Server ready" in server_line                       # the line at the top is refreshed after each clip
    assert Path(path if isinstance(path, str) else path["value"]).exists() or path
    assert "Done: 10.0 s of audio, seed 5" in status
    assert "Seed: 5" in details and "Steps: 8" in details


def test_the_random_seed_button_gives_a_seed(page):
    client, _, _ = page
    a, b = client.predict(api_name="/random_seed"), client.predict(api_name="/random_seed")
    assert isinstance(a, (int, float)) and a != b


def test_a_busy_server_is_a_status_not_a_crash(page):
    client, _, server = page
    server.close()
    _, status, _, server_line = client.predict("a tune", 10, 8, 1.0, 5, api_name="/generate")
    assert "not reachable" in status and "not reachable" in server_line


def test_the_page_module_is_the_only_one_that_imports_gradio():
    for name in ("audio_client.py", "audio_server.py"):
        text = (HERE.parent / "orchard" / "audio_demo" / name).read_text()
        assert "import gradio" not in text and "from gradio" not in text


def test_the_page_fits_its_controls_and_examples_to_a_server_with_a_short_limit():
    """With --max-seconds 5 the starting length is 5 and no example asks for a length the server would refuse."""
    class Short(Scripted):
        max_seconds = 5
    server = Live(Short())
    try:
        server.wait_ready()
        demo = app.build_ui(cl.AudioClient(server.base, timeout=5), poll_s=0.01)
        length = next(b for b in demo.blocks.values() if getattr(b, "label", None) == "Length (seconds)")
        assert length.maximum == 5 and length.value <= 5
        for ex in (b for b in demo.blocks.values() if isinstance(b, gr.Dataset)):
            assert all(row[1] <= 5 for row in ex.samples)
        assert app.fitting_examples({"min_seconds": 1, "max_seconds": 5}) == [e for e in cl.EXAMPLES if e[1] <= 5]
        assert app.fitting_examples({"min_seconds": 100, "max_seconds": 120}) == []        # none fit: the page omits the examples
    finally:
        server.close()
