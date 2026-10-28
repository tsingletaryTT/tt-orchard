# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A client for the Stable Audio job server's API, and the page's generation logic (standard library only, so the page's behaviour is testable without Gradio).

`AudioClient` submits a job, polls it and returns the clip with the server's own settings and timings. A job that is abandoned (timeout, Ctrl-C) is
cancelled so it does not keep its place in the queue. `run_generation` wraps one request into what the page shows: the file, a status line and the details,
with a plain message for a busy, unreachable or still loading server, bad input, a failed job, and a clip that ended before the length asked for.
"""
from __future__ import annotations

import json
import math
import random
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MIN_SECONDS = 1
MAX_SECONDS = 120           # the model's own limit (sample_size 5,292,032 at 44,100 Hz). Only 10 s clips have been checked on the chip.
MAX_STEPS = 100
MIN_CFG, MAX_CFG = 1.0, 20.0
EARLY_END_SHARE = 0.95      # a clip shorter than this share of the length asked for is reported as ended early

BENCHMARK_PROMPT = "A warm acoustic guitar melody with soft percussion, 100 BPM"
# (prompt, seconds, seed): the first is the prompt of the harness's benchmark run
EXAMPLES = [
    (BENCHMARK_PROMPT, 10, 42),
    ("Lo-fi hip hop beat with mellow piano chords and vinyl crackle, 80 BPM", 10, 7),
    ("Upbeat synth pop with a driving bassline and bright arpeggios, 120 BPM", 10, 21),
    ("Slow cinematic strings with a soft piano, spacious and warm", 20, 3),
]


class AudioServerError(Exception):
    """The server refused a request or a job ended in error."""


class ServerBusy(AudioServerError):
    """HTTP 429: the queue is full."""


class ServerNotReady(AudioServerError):
    """HTTP 503: the model is loading or did not load."""


class ServerDown(AudioServerError):
    """The server did not answer."""


@dataclass
class Result:
    audio: bytes
    audio_format: str
    job_id: str
    settings: dict
    stats: dict


def check_settings(prompt, seconds, seed, steps, cfg_scale) -> None:
    """Raise ValueError, in plain words, for input the server would refuse."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The prompt must not be empty.")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError(f"The length must be between {MIN_SECONDS} and {MAX_SECONDS} seconds, got {seconds!r}.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1):
        raise ValueError(f"The seed must be a whole number of 0 or more, got {seed!r}.")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS):
        raise ValueError(f"Steps must be a whole number from 1 to {MAX_STEPS}, got {steps!r}.")
    if cfg_scale is not None and (isinstance(cfg_scale, bool) or not isinstance(cfg_scale, (int, float)) or not math.isfinite(cfg_scale)
                                  or not MIN_CFG <= cfg_scale <= MAX_CFG):
        raise ValueError(f"The guidance scale must be between {MIN_CFG:g} and {MAX_CFG:g}, got {cfg_scale!r}.")


class AudioClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Optional[dict] = None, raw: bool = False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = r.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode()).get("detail", "")
            except (ValueError, OSError, AttributeError):
                pass
            if e.code == 429:
                raise ServerBusy(f"The audio server is busy: {detail or 'its queue is full'}. Try again in a minute.") from None
            if e.code == 503:
                raise ServerNotReady(f"The audio server is not ready: {detail or 'the model is loading'}.") from None
            raise AudioServerError(f"The audio server answered {e.code}: {detail or e.reason}") from None
        except (urllib.error.URLError, OSError) as e:
            raise ServerDown(f"The audio server at {self.base} is not reachable ({getattr(e, 'reason', e)}).") from None
        return payload if raw else json.loads(payload)

    def health(self) -> dict:
        return self._call("GET", "/v1/health")

    def submit(self, prompt: str, seconds: float, seed: Optional[int] = None, steps: Optional[int] = None,
               cfg_scale: Optional[float] = None) -> str:
        """Queue a job and return its id. Raises ValueError for input the server would refuse, before any request is made."""
        check_settings(prompt, seconds, seed, steps, cfg_scale)
        body = {"prompt": prompt, "duration": seconds}
        for key, value in (("seed", seed), ("steps", steps), ("cfg_scale", cfg_scale)):
            if value is not None:
                body[key] = value
        return self._call("POST", "/v1/audio/jobs", body)["id"]

    def status(self, job_id: str) -> dict:
        return self._call("GET", f"/v1/audio/jobs/{job_id}")

    def audio(self, job_id: str) -> bytes:
        return self._call("GET", f"/v1/audio/jobs/{job_id}/audio", raw=True)

    def cancel(self, job_id: str) -> None:
        try:
            self._call("DELETE", f"/v1/audio/jobs/{job_id}")
        except AudioServerError:
            pass                                       # finished already, or the server is gone: nothing left to cancel

    def generate(self, prompt: str, seconds: float, seed: Optional[int] = None, steps: Optional[int] = None, cfg_scale: Optional[float] = None,
                 poll_s: float = 1.0, timeout_s: float = 900.0,
                 on_progress: Optional[Callable[[str, str, int], None]] = None) -> Result:
        """Submit, wait and return the clip. `on_progress(state, stage, queue_position)` is called on each poll."""
        job_id = self.submit(prompt, seconds, seed=seed, steps=steps, cfg_scale=cfg_scale)
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                st = self.status(job_id)
                state = st.get("status")
                if state == "done":
                    stats = st.get("stats") or {}
                    return Result(self.audio(job_id), stats.get("audio_format", "wav"), job_id, st.get("settings") or {}, stats)
                if state in ("error", "cancelled"):
                    raise AudioServerError(f"The job {state}: {st.get('error') or 'no reason given'}")
                if on_progress:
                    on_progress(str(state), str(st.get("stage") or ""), int(st.get("queue_position") or 0))
                if time.monotonic() > deadline:
                    raise AudioServerError(f"Timed out after {timeout_s:.0f} s waiting for job {job_id}.")
                time.sleep(poll_s)
        except (AudioServerError, KeyboardInterrupt):
            self.cancel(job_id)
            raise


# ---- what the page shows ------------------------------------------------------------------------------------

def random_seed() -> int:
    return random.randint(0, 2**31 - 1)


def _whole_or_none(v) -> Optional[int]:
    """A Gradio Number box gives None, "" or a float; a whole number comes back as an int and anything else as None."""
    if v is None or v == "":
        return None
    f = float(v)
    return int(f) if f == int(f) else None


@dataclass
class Outcome:
    ok: bool
    status: str
    details: str = ""
    audio_path: Optional[str] = None
    seed: Optional[int] = None


def server_status(client: AudioClient) -> str:
    """One line for the top of the page: what the server is, or why it cannot be used."""
    try:
        h = client.health()
    except ServerDown as e:
        return f"**{e}** Start it with `python audio_server.py --ui`, or check AUDIO_URL."
    except AudioServerError as e:
        return f"**{e}**"
    if h.get("status") == "loading":
        return "The server is up and the model is loading. Try again in a minute."
    if h.get("status") != "ok":
        return f"The server answered but is not ready: {h.get('error') or h.get('status')}."
    return (f"Server ready: **{h.get('model', 'unknown model')}** on {h.get('device_name', 'an unknown device')}, "
            f"{h.get('queue_depth', 0)} waiting (room for {h.get('queue_max', '?')}).")


def run_generation(client: AudioClient, prompt, seconds, steps, cfg_scale, seed, on_progress=None, poll_s: float = 1.0,
                   out_dir: Optional[Path] = None) -> Outcome:
    """Make one clip for the page. Never raises for a problem the user can read about: the Outcome's status says what happened."""
    seed_i = _whole_or_none(seed)
    if seed_i is None or seed_i < 0:
        seed_i = random_seed()                          # a seed is always shown, so a clip can be asked for again
    steps_i = _whole_or_none(steps)
    try:
        sec = float(seconds)
        sec = int(sec) if sec == int(sec) else sec
    except (TypeError, ValueError):
        return Outcome(False, f"The length must be a number of seconds, got {seconds!r}.", seed=seed_i)
    cfg = None if cfg_scale in (None, "") else float(cfg_scale)
    try:
        result = client.generate(prompt, sec, seed=seed_i, steps=steps_i, cfg_scale=cfg, poll_s=poll_s, on_progress=on_progress)
    except ValueError as e:
        return Outcome(False, str(e), seed=seed_i)
    except ServerBusy as e:
        return Outcome(False, str(e), seed=seed_i)
    except (ServerDown, ServerNotReady) as e:
        return Outcome(False, str(e), seed=seed_i)
    except AudioServerError as e:
        return Outcome(False, f"The server could not make the clip. {e}", seed=seed_i)
    folder = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="audio-"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"clip-seed{seed_i}-{result.job_id[:8]}.{result.audio_format}"
    path.write_bytes(result.audio)
    s, used = result.stats, result.settings
    audio_s = float(s.get("audio_s") or sec)
    status = f"Done: {audio_s:.1f} s of audio, seed {seed_i}."
    if audio_s < EARLY_END_SHARE * sec:
        status += f" The clip ended at {audio_s:.1f} s, before the {sec:g} s asked for."
    lines = [f"Prompt: {used.get('prompt', prompt)}",
             f"Seconds asked for: {used.get('duration', sec):g}. Steps: {used.get('steps')}. Guidance scale: {used.get('cfg_scale'):g}. Seed: {used.get('seed')}."
             if isinstance(used.get("cfg_scale"), (int, float)) else f"Settings: {used}"]
    total = s.get("total_s")
    if total:
        lines.append(f"The server took {total:.1f} s for {audio_s:.1f} s of audio.")
    stages = {k: v for k, v in (s.get("stages") or {}).items() if isinstance(v, (int, float))}
    if stages:
        lines.append("Stage times: " + ", ".join(f"{k.removesuffix('_s').replace('_', ' ')} {v:.1f} s" for k, v in stages.items()) + ".")
    lines.append(f"Job `{result.job_id}`.")
    return Outcome(True, status, "\n\n".join(lines), str(path), seed_i)
