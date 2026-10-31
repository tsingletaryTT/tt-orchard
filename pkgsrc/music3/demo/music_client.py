# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""A small client for the MiniMax-Music3 server's job API (standard library only).

The server makes one song at a time and keeps a short queue; a request that finds the queue full gets HTTP 429,
which this client reports as `ServerBusy`. `generate` submits a job, polls it, and returns the audio and the
server's own timing figures. A job that is abandoned (timeout, Ctrl-C) is cancelled so it does not keep the chip.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

MIN_SECONDS = 5          # shortest clip the demo asks for
MAX_SECONDS = 60         # about 3.5 minutes of compute at the measured 3.2 s per second of audio
MAX_STEPS = 100          # the server's own limit on num_inference_steps


class MusicServerError(Exception):
    """The server refused a request or a job ended in error."""


class ServerBusy(MusicServerError):
    """HTTP 429: the server's queue is full."""


class ServerDown(MusicServerError):
    """The server did not answer."""


@dataclass
class Result:
    audio: bytes
    job_id: str
    stats: dict


class MusicClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # ---- one HTTP call ----------------------------------------------------------------------------------

    def _call(self, method: str, path: str, body: Optional[dict] = None, raw: bool = False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
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
                raise ServerBusy("the music server is busy: its queue is full; try again in a minute") from None
            raise MusicServerError(f"the music server answered {e.code}: {detail or e.reason}") from None
        except (urllib.error.URLError, OSError) as e:
            raise ServerDown(f"the music server at {self.base} is not reachable ({getattr(e, 'reason', e)})") from None
        return payload if raw else json.loads(payload)

    # ---- the API ----------------------------------------------------------------------------------------

    def health(self) -> dict:
        return self._call("GET", "/v1/health")

    def submit(self, lyrics: str, caption: str, seconds: float, seed: Optional[int] = None,
               steps: Optional[int] = None) -> str:
        """Queue a job and return its id. Raises ValueError for input the server would refuse."""
        if not isinstance(lyrics, str) or not lyrics.strip():
            raise ValueError("lyrics must not be empty")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("caption must not be empty")
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(f"seconds must be between {MIN_SECONDS} and {MAX_SECONDS}, got {seconds}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError(f"seed must be a whole number of 0 or more, got {seed!r}")
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS):
            raise ValueError(f"steps must be a whole number from 1 to {MAX_STEPS}, got {steps!r}")
        body = {"input": lyrics, "instructions": caption, "audio_duration": seconds, "response_format": "flac"}
        if seed is not None:
            body["seed"] = seed
        if steps is not None:
            body["num_inference_steps"] = steps
        return self._call("POST", "/v1/music/jobs", body)["id"]

    def status(self, job_id: str) -> dict:
        return self._call("GET", f"/v1/music/jobs/{job_id}")

    def audio(self, job_id: str) -> bytes:
        return self._call("GET", f"/v1/music/jobs/{job_id}/audio", raw=True)

    def cancel(self, job_id: str) -> None:
        try:
            self._call("DELETE", f"/v1/music/jobs/{job_id}")
        except MusicServerError:
            pass                                     # already finished or the server is gone: nothing left to cancel

    def generate(self, lyrics: str, caption: str, seconds: float, seed: Optional[int] = None,
                 steps: Optional[int] = None, poll_s: float = 2.0, timeout_s: float = 900.0,
                 on_progress: Optional[Callable[[int, int, str], None]] = None) -> Result:
        """Submit, wait for the clip and return it. `on_progress(frames_done, frames_max, stage)` is called on each poll."""
        job_id = self.submit(lyrics, caption, seconds, seed=seed, steps=steps)
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                st = self.status(job_id)
                state = st.get("status")
                if state == "done":
                    return Result(self.audio(job_id), job_id, st.get("stats") or {})
                if state in ("error", "cancelled"):
                    raise MusicServerError(f"the job {state}: {st.get('error') or 'no reason given'}")
                if on_progress:
                    on_progress(int(st.get("frames_done") or 0), int(st.get("frames_max") or 0), str(st.get("stage") or ""))
                if time.monotonic() > deadline:
                    raise MusicServerError(f"timed out after {timeout_s:.0f} s waiting for job {job_id}")
                time.sleep(poll_s)
        except (MusicServerError, KeyboardInterrupt):
            self.cancel(job_id)
            raise
