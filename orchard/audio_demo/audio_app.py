# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A Gradio page for Stable Audio 3 small music running on a Tenstorrent chip.

    python audio_app.py                          # talks to AUDIO_URL, default http://127.0.0.1:20100
    python audio_server.py --generator chip --ui  # loads the model on the leased chip, then runs this page

The server makes one clip at a time and keeps a short queue; when the queue is full the page says so. Nothing here signs anyone in:
the page binds 127.0.0.1 unless told otherwise, so bind 0.0.0.0 only on a network you trust.

Gradio is an optional dependency (`pip install "tt-orchard[audio-demo]"` or `pip install gradio`). The generation logic is in audio_client.py and has
no Gradio import; this file only builds the page and calls it.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

import audio_client as ac  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:20100"
PAGE_NOTE = ("Describe the music and the page asks the model for a clip. One clip at a time. "
             "Only 10 second clips have been checked on the chip; longer ones are allowed up to the model's own limit and have not been checked.")


def limits_from(client: ac.AudioClient) -> dict:
    """The limits the server reports, or this client's own when the server cannot be reached yet (the page still opens)."""
    own = {"min_seconds": ac.MIN_SECONDS, "max_seconds": ac.MAX_SECONDS, "max_steps": ac.MAX_STEPS, "min_cfg_scale": ac.MIN_CFG, "max_cfg_scale": ac.MAX_CFG}
    try:
        got = client.health().get("limits") or {}
    except ac.AudioServerError:
        return own
    return {k: got[k] if isinstance(got.get(k), (int, float)) and not isinstance(got.get(k), bool) else v for k, v in own.items()}


def build_ui(client: ac.AudioClient, poll_s: float = 1.0) -> gr.Blocks:
    lim = limits_from(client)
    with gr.Blocks(title="Stable Audio 3 small music on Tenstorrent") as demo:
        gr.Markdown("# Stable Audio 3 small music on Tenstorrent Blackhole\n" + PAGE_NOTE)
        status_line = gr.Markdown(ac.server_status(client))
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=3, value=ac.BENCHMARK_PROMPT, placeholder="genre, instruments, mood, tempo",
                                    info="Describe the music: instruments, mood and tempo work best.")
                seconds = gr.Slider(lim["min_seconds"], lim["max_seconds"], value=10, step=1, label="Length (seconds)")
                steps = gr.Slider(1, min(int(lim["max_steps"]), 50), value=8, step=1, label="Steps",
                                  info="8 steps with guidance scale 1 is the setting the model's authors use for small music.")
                cfg = gr.Slider(lim["min_cfg_scale"], min(lim["max_cfg_scale"], 10.0), value=1.0, step=0.5, label="Guidance scale (cfg)")
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0, minimum=0, scale=3)
                    dice = gr.Button("Random seed", scale=1)
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                audio = gr.Audio(label="Result", type="filepath", interactive=False)
                status = gr.Markdown()
                details = gr.Markdown()
        gr.Examples(examples=[list(e) for e in ac.EXAMPLES], inputs=[prompt, seconds, seed], label="Examples")

        def roll():
            return ac.random_seed()

        def generate(prompt_, seconds_, steps_, cfg_, seed_):
            """Runs the request in a thread and yields a status line on each poll. The server cannot say how far along a clip is, so the line gives
            the stage and the seconds waited, and no percentage."""
            started, seen, box = time.monotonic(), {"line": "Sending the request."}, {}

            def on_progress(state: str, stage: str, position: int) -> None:
                seen["line"] = f"Waiting: place {position} in the queue." if state == "queued" else f"Making the clip: {stage or state}."

            def work() -> None:
                box["out"] = ac.run_generation(client, prompt_, seconds_, steps_, cfg_, seed_, on_progress=on_progress, poll_s=poll_s)
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            while worker.is_alive():
                yield None, f"{seen['line']} {int(time.monotonic() - started)} s so far.", "", gr.skip()
                worker.join(poll_s)
            out = box["out"]
            yield out.audio_path, out.status, out.details, ac.server_status(client)

        dice.click(roll, None, seed, api_name="random_seed")
        go.click(generate, [prompt, seconds, steps, cfg, seed], [audio, status, details, status_line], api_name="generate", concurrency_limit=1)
        demo.load(lambda: ac.server_status(client), None, status_line)
    return demo


def launch(server_name: str = "127.0.0.1", server_port: int = 7872, audio_url: Optional[str] = None) -> None:
    url = audio_url or os.environ.get("AUDIO_URL", DEFAULT_URL)
    demo = build_ui(ac.AudioClient(url)).queue(max_size=8, default_concurrency_limit=1)
    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch(os.environ.get("AUDIO_UI_HOST", "127.0.0.1"), int(os.environ.get("AUDIO_UI_PORT", "7872")))
