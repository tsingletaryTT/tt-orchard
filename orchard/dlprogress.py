# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A progress line for a model download.

`hf download` run without a terminal prints nothing, and with one it prints only a count of files (measured with
huggingface_hub 1.33: "Fetching 10 files"), never bytes. So the meter measures the cache itself while the
download runs in another call: the bytes that have arrived in the repo's `blobs` directory, and the files that
are complete in the snapshot directory. It reads files and prints; it never starts, stops or touches the
download.

What the numbers mean:
- Bytes come from the blobs directory, which `hf` fills while it works (`*.incomplete` files grow, then are
  renamed; a newer huggingface_hub then moves the finished file to a shared store and leaves a link, which counts
  as the file it reaches). With Xet they grow in bursts, so on one large file the bar can sit still and then jump. Eight files
  download at once, so a big model averages out. A line that sits still says "waiting for data" instead of
  looking dead, and the elapsed time keeps counting.
- Only growth since the start counts, so blobs from other revisions are not read as progress. The total is what
  the hub says the repo holds, less the files already complete in the snapshot. A partly downloaded file left by
  an earlier attempt is in the starting size, so a resumed download starts a little low.

On a terminal the line is redrawn in place. On anything else (a pipe, a log file, `--style plain`) it prints one
line each time the percentage passes a step of 10, and at least one a minute.
"""
from __future__ import annotations

import os
import shutil
import stat
import threading
import time
from collections import deque
from dataclasses import dataclass

from orchard import ui

BAR_FILLED, BAR_EMPTY = "█", "░"
BAR_WIDTH = 20
WINDOW_S = 15.0        # the speed is the bytes that arrived over this many seconds
LOG_EVERY_S = 60.0     # a log gets a line at least this often
LOG_STEP = 10          # and one each time the percentage passes a multiple of this
WAITING_S = 15.0       # with no bytes for this long (and since the start) the line says it is waiting
ETA_AFTER_S = 10.0     # an ETA from the first seconds is a wild guess (14 minutes for a 36 second download)
JOIN_S = 5.0
ERASE = "\r\x1b[K"


@dataclass(frozen=True)
class Reading:
    label: str
    done: int            # bytes that arrived since the start
    need: int            # bytes to fetch; 0 when the hub gave no total
    files_done: int
    files_total: int
    rate: float          # bytes a second over the last WINDOW_S
    eta: float | None    # seconds, None without a speed
    elapsed: float
    waiting: bool        # no bytes lately


def human_bytes(n) -> str:
    n = max(float(n), 0.0)
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{n:.0f} B"


def human_duration(seconds) -> str:
    if seconds is None:
        return "?"
    s = max(round(seconds), 0)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def bar(fraction: float, width: int) -> str:
    filled = min(max(round(fraction * width), 0), width)
    return BAR_FILLED * filled + BAR_EMPTY * (width - filled)


def dir_bytes(path) -> int:
    """The bytes of the regular files that the entries directly in `path` resolve to (the blobs directory is flat).
    A finished blob in a newer huggingface_hub is a link into the shared store (`hub/blobs/<2 hex>/<hash>`), so a
    link counts as the file it reaches. Each file counts once, however many names it has. A link that goes nowhere
    or to a directory adds nothing."""
    total = 0
    seen = set()
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    st = entry.stat(follow_symlinks=True)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) in seen:
                    continue
                seen.add((st.st_dev, st.st_ino))
                total += st.st_size
    except OSError:
        return 0
    return total


def snapshot_done(snapshot, files) -> tuple[int, int]:
    """(files, bytes) of the listed files that are complete: a path that resolves to a regular file."""
    count = size = 0
    for name in files:
        try:
            st = os.stat(os.path.join(snapshot, name))
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode):
            count += 1
            size += st.st_size
    return count, size


def render_line(r: Reading, width: int, paint=None) -> str:
    """One line no wider than `width` - 1 (a full line would wrap, and a redrawn line must not). Parts are dropped
    from the end: elapsed time, files, ETA, speed; then the bar shrinks and goes."""
    paint = paint or (lambda text, role: text)
    name = r.label.rsplit("/", 1)[-1]
    known = r.need > 0
    fraction = min(r.done / r.need, 1.0) if known else 0.0
    sizes = f"{human_bytes(r.done)} / {human_bytes(r.need)}" if known else f"{human_bytes(r.done)} downloaded"
    extras = []
    if r.rate > 0:
        extras.append(f"{human_bytes(r.rate)}/s")
    elif r.waiting:
        extras.append("waiting for data")
    if r.eta is not None and r.rate > 0 and r.elapsed >= ETA_AFTER_S:
        extras.append(f"ETA {human_duration(r.eta)}")
    if r.files_total:
        extras.append(f"{r.files_done} of {r.files_total} files")
    extras.append(f"elapsed {human_duration(r.elapsed)}")

    def build(bar_width: int, keep: int, painter) -> str:
        head = f"{name}  {sizes}"
        if known and bar_width:
            head += f"  {painter(bar(fraction, bar_width), 'accent')} {int(fraction * 100)}%"
        return "  ".join([head] + extras[:keep])

    layouts = [(BAR_WIDTH, k) for k in range(len(extras), -1, -1)] + ([(10, 0), (0, 0)] if known else [])
    for bar_width, keep in layouts:
        if len(build(bar_width, keep, lambda text, role: text)) <= width - 1:
            return build(bar_width, keep, paint)
    return build(0, 0, lambda text, role: text)[:max(width - 1, 0)]


class DownloadMeter:
    """Use as a context manager around the download. `begin`, `tick` and `finish` are the same steps without the
    thread, for tests."""

    def __init__(self, label: str, *, blob_dir, snapshot, files, total_bytes: int, out, style,
                 clock=time.monotonic, interval: float = 1.0, width: int | None = None):
        self.label, self.blob_dir, self.snapshot, self.files = label, blob_dir, snapshot, list(files)
        self.total_bytes, self.out, self.style = int(total_bytes or 0), out, style
        self.clock, self.interval, self._width = clock, interval, width
        self.animate = ui.stream_is_tty(out) and bool(getattr(style, "pretty", True))
        self._thread = None
        self._stop = threading.Event()
        self._begun = self._finished = self._dead = False
        self._last = None

    # -- measuring --------------------------------------------------------------------------------------

    def begin(self) -> None:
        self._t0 = self.clock()
        self._baseline = dir_bytes(self.blob_dir)
        _, have = snapshot_done(self.snapshot, self.files)
        self._need = max(self.total_bytes - have, 0) if self.total_bytes else 0
        self._samples = deque()
        self._best = 0
        self._last_step = 0
        self._last_log = self._t0
        self._begun = True
        r = self.reading()
        if self.animate:
            self._draw(r)
        else:
            tail = f" to fetch ({r.files_done} of {r.files_total} files)" if r.files_total else " to fetch"
            self._write(f"downloading {self._name}: {human_bytes(r.need)}{tail}\n" if r.need
                        else f"downloading {self._name}\n")

    @property
    def _name(self) -> str:
        return self.label.rsplit("/", 1)[-1]

    def reading(self) -> Reading:
        now = self.clock()
        grown = max(dir_bytes(self.blob_dir) - self._baseline, 0)
        self._best = max(self._best, min(grown, self._need) if self._need else grown)
        self._samples.append((now, self._best))
        while len(self._samples) > 1 and self._samples[0][0] < now - WINDOW_S:
            self._samples.popleft()
        then, then_bytes = self._samples[0]
        rate = (self._best - then_bytes) / (now - then) if now > then else 0.0
        eta = (self._need - self._best) / rate if rate > 0 and self._need else None
        files_done, _ = snapshot_done(self.snapshot, self.files)
        elapsed = now - self._t0
        self._last = Reading(self.label, self._best, self._need, files_done, len(self.files), rate, eta, elapsed,
                             rate == 0 and elapsed >= WAITING_S)
        return self._last

    # -- showing ----------------------------------------------------------------------------------------

    def _write(self, text: str) -> None:
        """A closed or broken stream ends the display; it never reaches the download."""
        if self._dead:
            return
        try:
            self.out.write(text)
            self.out.flush()
        except (OSError, ValueError):
            self._dead = True

    def _terminal_width(self) -> int:
        if self._width:
            return self._width
        try:
            return os.get_terminal_size(self.out.fileno()).columns
        except (OSError, ValueError, AttributeError):
            return shutil.get_terminal_size((100, 24)).columns

    def _draw(self, r: Reading) -> None:
        self._write(ERASE + render_line(r, self._terminal_width(), paint=self.style.paint))

    def _log_line(self, r: Reading) -> str:
        if not r.need:
            return f"downloading {self._name}: {human_bytes(r.done)} so far"
        parts = [f"{human_bytes(r.done)} of {human_bytes(r.need)} ({int(min(r.done / r.need, 1.0) * 100)}%)"]
        if r.rate > 0:
            parts.append(f"{human_bytes(r.rate)}/s")
            if r.eta is not None and r.elapsed >= ETA_AFTER_S:
                parts.append(f"ETA {human_duration(r.eta)}")
        elif r.waiting:
            parts.append("waiting for data")
        if r.files_total:
            parts.append(f"{r.files_done} of {r.files_total} files")
        return f"downloading {self._name}: " + ", ".join(parts)

    def tick(self) -> None:
        r = self.reading()
        if self.animate:
            self._draw(r)
            return
        step = int(min(r.done / r.need, 1.0) * 100) // LOG_STEP if r.need else 0
        if step > self._last_step or self.clock() - self._last_log >= LOG_EVERY_S:
            self._last_step = max(step, self._last_step)
            self._last_log = self.clock()
            self._write(self._log_line(r) + "\n")

    def finish(self, ok: bool) -> None:
        if not self._begun or self._finished:
            return
        self._finished = True
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(JOIN_S)
        if self.animate:
            self._write(ERASE)
        if ok:
            try:
                r = self.reading()                   # one last look, so bytes since the final tick count
            except Exception:                        # a measurement that fails must not hide the summary
                r = self._last
            elapsed = self.clock() - self._t0
            if r.done <= 0:
                self._write(f"downloaded nothing new in {human_duration(elapsed)}\n")
                return
            # the bytes that arrived in this call: a resumed file counts only what it still needed
            average = f" ({human_bytes(r.done / elapsed)}/s average)" if elapsed > 0 else ""
            self._write(f"downloaded {human_bytes(r.done)} in {human_duration(elapsed)}{average}\n")

    # -- the thread -------------------------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception as exc:                 # a bug here must not reach the download
                if self.animate:
                    self._write(ERASE)
                self._write(f"(progress display stopped: {type(exc).__name__})\n")
                return

    def __enter__(self) -> "DownloadMeter":
        self.begin()
        self._thread = threading.Thread(target=self._loop, name="download-meter", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(exc_type is None)
        return False
