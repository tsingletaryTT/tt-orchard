# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""A Gradio page for the MiniMax-Music3 server running on a Tenstorrent chip.

    python demo/app.py                       # talks to MUSIC_URL, default http://127.0.0.1:20000
    python serve/serve_music.py --ui         # boots the model on a leased chip, then runs this page

The server makes one song at a time (about 3.2 s of compute per second of audio on one p150 chip) and keeps a short
queue; when the queue is full the page says so. Nothing here signs anyone in: bind to 0.0.0.0 only on a network you trust.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr

from music_client import (MAX_SECONDS, MIN_SECONDS, MusicClient, MusicServerError, ServerBusy,
                          ServerDown)

DEFAULT_URL = "http://127.0.0.1:20000"
TRIGGER = "fiona crapple"

# (lyrics, caption, seconds, seed): clips this page was checked with
EXAMPLES = [
    ("[Verse 1]\nThe kettle sings at half past three\nThe rain keeps time on the window glass\n[Chorus]\nStay a little longer\nThe coffee is still warm",
     f"{TRIGGER}, jazzy piano, raw female vocal, confessional singer-songwriter, upright bass, brushed drums, intimate", 20, 11),
    ("[Verse 1]\nSunday morning, no one's rushing\nEggs are burning, we are laughing\n[Chorus]\nDance in the kitchen",
     f"{TRIGGER}, swinging brushed drums, walking bass, bright piano, playful vocal", 20, 14),
    ("[Verse 1]\nNeon signs along the highway\nWe are chasing summer's end\n[Chorus]\nTurn it up, turn it up",
     "upbeat synth pop, clean male vocal, four on the floor, bright chorus", 20, 21),
    ("[Verse 1]\nThe river knows the way back home\nI follow where the water goes\n[Chorus]\nCarry me down",
     "warm acoustic folk, fingerpicked guitar, soft harmonies, gentle", 20, 23),
    ("[Verse 1]\nEarly train and coffee steam\nNotebook full of half a dream\n[Chorus]\nKeep it moving",
     "boom bap hip hop, dusty drums, mellow keys, rhythmic male rap", 20, 25),
]


def server_status(client: MusicClient) -> str:
    try:
        h = client.health()
    except ServerDown as e:
        return f"**{e}.** Start the model (`python serve/serve_music.py --ui`) or check MUSIC_URL."
    if h.get("status") != "ok":
        return f"The server answered but is not ready: {h.get('error') or h.get('status')}."
    queue = h.get("queue_depth") or 0
    return (f"Server ready: **{h.get('model', 'unknown model')}** on {h.get('device_name', 'a Tenstorrent chip')} "
            f"(mesh {h.get('mesh', '?')}), {queue} waiting.")


def _int_or_none(v) -> Optional[int]:
    if v is None or v == "":
        return None
    f = float(v)
    return int(f) if f == int(f) else None


def run_generation(client: MusicClient, lyrics, caption, seconds, seed, steps, progress, poll_s: float = 2.0):
    """Generate one clip. Returns (path to a FLAC file, status text, details text). Raises gr.Error with a plain message."""
    seed_i = _int_or_none(seed)
    if seed_i is None or seed_i < 0:
        seed_i = random.randint(0, 2**31 - 1)          # a seed is always shown, so a clip can be asked for again
    steps_i = _int_or_none(steps)
    if steps_i == 0:
        steps_i = None

    def on_progress(done: int, total: int, stage: str) -> None:
        progress(min(1.0, done / total) if total else 0.0, desc=f"{stage}: {done}/{total} frames")

    try:
        result = client.generate(lyrics, caption, seconds, seed=seed_i, steps=steps_i, poll_s=poll_s, on_progress=on_progress)
    except ValueError as e:
        raise gr.Error(str(e)) from None
    except (ServerBusy, ServerDown) as e:
        raise gr.Error(str(e)) from None
    except MusicServerError as e:
        raise gr.Error(f"The server could not make the clip: {e}") from None
    out = Path(tempfile.mkdtemp(prefix="music-")) / f"clip-seed{seed_i}.flac"
    out.write_bytes(result.audio)
    s = result.stats
    audio_s, total_s = s.get("audio_s") or seconds, s.get("total_s")
    status = f"Done: {audio_s:.1f} s of audio, seed {seed_i}."
    if s.get("ended") and audio_s < 0.95 * seconds:
        # the server stops a song when its lyrics are finished, which can be before the length asked for
        status += (f" The song ended at {audio_s:.1f} s, before the {seconds:g} s asked for: the lyrics were finished. "
                   "Longer lyrics make a longer song.")
    lines = []
    if total_s:
        lines.append(f"Compute **{total_s:.0f} s** for **{audio_s:.1f} s** of audio: {total_s / audio_s:.1f} s per second of audio.")
        stages = [(k, s.get(f"{k}_s")) for k in ("llm", "depth", "denoise", "vocoder", "decode")]
        lines.append("Stage times: " + ", ".join(f"{k} {v:.1f} s" for k, v in stages if v is not None) + ".")
    lines.append(f"Job `{result.job_id}`, caption: {caption}")
    return str(out), status, "\n\n".join(lines)


def build_ui(client: MusicClient, poll_s: float = 2.0) -> gr.Blocks:
    with gr.Blocks(title="MiniMax-Music3 on Tenstorrent") as demo:
        gr.Markdown("# MiniMax-Music3 on Tenstorrent Blackhole\n"
                    "Write lyrics and describe the music. One song at a time; about 3.2 s of compute per second of audio on one chip.")
        status_line = gr.Markdown(server_status(client))
        with gr.Row():
            with gr.Column(scale=1):
                lyrics = gr.Textbox(label="Lyrics", lines=8, placeholder="[Verse 1]\nYour lines here\n[Chorus]\n...",
                                    info="Structure tags such as [Verse], [Chorus] go on their own lines.")
                caption = gr.Textbox(label="Music description", lines=3,
                                     placeholder="genre, instruments, vocals, mood, tempo",
                                     info=f'This build has an adapter that answers to the trigger phrase "{TRIGGER}".')
                seconds = gr.Slider(MIN_SECONDS, MAX_SECONDS, value=20, step=5, label="Length (seconds)")
                with gr.Accordion("More", open=False):
                    seed = gr.Number(label="Seed (blank for random; the seed used is shown)", value=None, precision=0, minimum=0)
                    steps = gr.Number(label="Diffusion steps, 1 to 100 (blank or 0 for the server's default)", value=None, precision=0)   # no minimum here: a blank box arrives as 0
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                audio = gr.Audio(label="Result", type="filepath", interactive=False)
                status = gr.Markdown()
                details = gr.Markdown()
        gr.Examples(examples=[list(e) for e in EXAMPLES], inputs=[lyrics, caption, seconds, seed], label="Examples")

        def generate(lyrics_, caption_, seconds_, seed_, steps_, progress=gr.Progress()):
            return run_generation(client, lyrics_, caption_, seconds_, seed_, steps_, progress, poll_s=poll_s)

        go.click(generate, [lyrics, caption, seconds, seed, steps], [audio, status, details], api_name="generate",
                 concurrency_limit=1)
        demo.load(lambda: server_status(client), None, status_line)
    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7871, music_url: Optional[str] = None) -> None:
    url = music_url or os.environ.get("MUSIC_URL", DEFAULT_URL)
    demo = build_ui(MusicClient(url)).queue(max_size=8, default_concurrency_limit=1)
    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch(os.environ.get("MUSIC_UI_HOST", "0.0.0.0"), int(os.environ.get("MUSIC_UI_PORT", "7871")))
