# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The download meter (orchard/dlprogress.py): what it measures, what it prints on a terminal and in a log."""
import io
import threading
import time

import pytest

from orchard import dlprogress as dp
from orchard import ui


class Out(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def style(pretty=True):
    return ui.Style("none", False, pretty=pretty)


def grow(path, name, size):
    """A blob that is `size` bytes long (a sparse file: only its length matters to the meter)."""
    with open(path / name, "wb") as f:
        f.truncate(size)


def make(tmp_path, *, total=1000, files=("a", "b"), tty=False, pretty=True, width=100, interval=0.01):
    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir(exist_ok=True)
    snap.mkdir(exist_ok=True)
    out, clock = Out(tty), Clock()
    m = dp.DownloadMeter("org/model", blob_dir=blobs, snapshot=snap, files=list(files), total_bytes=total, out=out,
                         style=style(pretty), clock=clock, interval=interval, width=width)
    return m, blobs, snap, out, clock


# ---- formatting -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,text", [(0, "0 B"), (-5, "0 B"), (999, "999 B"), (1_000, "1 KB"), (1_000_000, "1 MB"), (12_345, "12 KB"), (1_500_000, "2 MB"),
                                    (118_400_000, "118 MB"), (999_999_999, "1000 MB"), (1_000_000_000, "1.0 GB"),
                                    (54_700_000_000, "54.7 GB"), (1_250_000_000_000, "1250.0 GB")])
def test_bytes_are_written_in_decimal_units_like_the_preflight_does(n, text):
    assert dp.human_bytes(n) == text


@pytest.mark.parametrize("s,text", [(None, "?"), (0, "0s"), (45, "45s"), (59.6, "1m 00s"), (60, "1m 00s"), (280, "4m 40s"),
                                    (3599, "59m 59s"), (3600, "1h 00m"), (7500, "2h 05m"), (-5, "0s")])
def test_durations_are_short_and_never_negative(s, text):
    assert dp.human_duration(s) == text


def test_the_bar_is_filled_in_proportion_and_clamped():
    assert dp.bar(0.0, 10) == "░" * 10 and dp.bar(1.0, 10) == "█" * 10
    assert dp.bar(0.5, 10) == "█" * 5 + "░" * 5
    assert dp.bar(-1, 10) == "░" * 10 and dp.bar(7, 10) == "█" * 10
    assert len(dp.bar(0.333, 20)) == 20


# ---- what is measured -----------------------------------------------------------------------------------

def test_dir_bytes_adds_the_regular_files_in_the_directory(tmp_path):
    grow(tmp_path, "x", 300)
    grow(tmp_path, "y.incomplete", 200)
    (tmp_path / "sub").mkdir()
    grow(tmp_path / "sub", "z", 9999)                                  # not a blob: the directory is flat
    (tmp_path / "link").symlink_to(tmp_path / "x")                     # a link is not counted twice
    assert dp.dir_bytes(tmp_path) == 500


def share(blobs, shared, name):
    """What a newer huggingface_hub does when a blob is complete: the file moves to the shared store
    (`hub/blobs/<2 hex>/<hash>`) and the repo's own blobs/ keeps a link to it."""
    dest = shared / name[:2] / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    (blobs / name).rename(dest)
    (blobs / name).symlink_to(dest)
    return dest


def test_a_blob_moved_to_the_shared_store_still_counts(tmp_path):
    shared = tmp_path / "shared"
    grow(tmp_path, "ab" + "0" * 62, 700)
    dest = share(tmp_path, shared, "ab" + "0" * 62)
    assert (tmp_path / ("ab" + "0" * 62)).is_symlink() and dest.is_file()      # the layout of this machine's cache
    assert dp.dir_bytes(tmp_path) == 700


def test_two_names_for_one_blob_count_once(tmp_path):
    shared = tmp_path / "shared"
    grow(tmp_path, "cd" + "1" * 62, 700)
    dest = share(tmp_path, shared, "cd" + "1" * 62)
    (tmp_path / "second").symlink_to(dest)
    (tmp_path / "third").hardlink_to(dest)
    assert dp.dir_bytes(tmp_path) == 700


def test_a_dangling_link_and_a_link_to_a_directory_add_nothing(tmp_path):
    (tmp_path / "gone").symlink_to(tmp_path / "nowhere")
    (tmp_path / "sub").mkdir()
    (tmp_path / "dir-link").symlink_to(tmp_path / "sub")
    grow(tmp_path, "real", 5)
    assert dp.dir_bytes(tmp_path) == 5


def test_progress_goes_on_after_a_finished_shard_moves_to_the_shared_store(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000)
    m.begin()
    grow(blobs, "aa" + "0" * 62 + ".incomplete", 600)
    assert m.reading().done == 600
    (blobs / ("aa" + "0" * 62 + ".incomplete")).rename(blobs / ("aa" + "0" * 62))
    share(blobs, tmp_path / "shared", "aa" + "0" * 62)                  # shard one is complete and now a link
    assert m.reading().done == 600
    grow(blobs, "bb" + "0" * 62 + ".incomplete", 300)                   # shard two arrives
    assert m.reading().done == 900
    (blobs / ("bb" + "0" * 62 + ".incomplete")).rename(blobs / ("bb" + "0" * 62))
    share(blobs, tmp_path / "shared", "bb" + "0" * 62)
    grow(blobs, "cc" + "0" * 62, 100)
    assert m.reading().done == 1000


def test_a_blob_already_in_the_shared_store_at_the_start_is_not_progress(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000)
    grow(blobs, "dd" + "0" * 62, 400)
    share(blobs, tmp_path / "shared", "dd" + "0" * 62)
    m.begin()
    assert m.reading().done == 0


def test_a_missing_directory_has_no_bytes(tmp_path):
    assert dp.dir_bytes(tmp_path / "nope") == 0


def test_a_file_counts_as_done_only_when_it_resolves(tmp_path):
    blob = tmp_path / "blob"
    grow(tmp_path, "blob", 70)
    snap = tmp_path / "snap"
    (snap / "sub").mkdir(parents=True)
    (snap / "a").symlink_to(blob)                                      # complete
    (snap / "sub" / "c").symlink_to(blob)
    (snap / "dangling").symlink_to(tmp_path / "gone")                  # its blob is not there
    assert dp.snapshot_done(snap, ["a", "sub/c", "dangling", "missing", "sub"]) == (2, 140)   # a directory is not a file


def test_progress_counts_only_growth_since_the_start_and_only_what_is_still_to_fetch(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000)
    grow(blobs, "old", 5000)                                           # blobs of other revisions and models' files
    grow(blobs, "a", 200)
    (snap / "a").symlink_to(blobs / "a")                               # one file is already complete: 200 bytes
    m.begin()
    r = m.reading()
    assert (r.need, r.done, r.files_done, r.files_total) == (800, 0, 1, 2)
    grow(blobs, "b.incomplete", 400)
    r = m.reading()
    assert (r.done, r.need) == (400, 800) and r.files_done == 1


def test_done_never_goes_down_and_never_passes_the_total(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000)
    m.begin()
    grow(blobs, "b.incomplete", 600)
    assert m.reading().done == 600
    (blobs / "b.incomplete").unlink()                                  # a failed file is cleaned up
    assert m.reading().done == 600
    grow(blobs, "big", 5000)
    assert m.reading().done == 1000


def test_the_speed_is_bytes_over_the_last_window_and_old_samples_are_dropped(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=10**9)
    m.begin()
    for step, size in enumerate((0, 100, 200, 300), start=1):
        clock.advance(5)
        grow(blobs, "b", size)
        r = m.reading()
    assert r.rate == pytest.approx(20.0)                               # 300 bytes in 15 s
    clock.advance(dp.WINDOW_S + 1)
    r = m.reading()
    assert r.rate == 0.0                                               # nothing came in over the window


def test_a_download_with_a_speed_is_not_waiting_however_long_it_has_run(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=10**9)
    m.begin()
    clock.advance(dp.WAITING_S + 5)
    assert m.reading().waiting is True                                 # nothing has arrived
    grow(blobs, "b", 5000)
    clock.advance(1)
    r = m.reading()
    assert r.rate > 0 and r.waiting is False


def test_the_eta_is_the_rest_divided_by_the_speed_and_missing_without_a_speed(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000)
    m.begin()
    assert m.reading().eta is None
    clock.advance(10)
    grow(blobs, "b", 200)
    r = m.reading()
    assert r.rate == pytest.approx(20.0) and r.eta == pytest.approx(40.0)


# ---- the line --------------------------------------------------------------------------------------------

def reading(**kw):
    base = dict(label="org/model", done=21_400_000_000, need=54_700_000_000, files_done=3, files_total=11,
                rate=118_000_000.0, eta=280.0, elapsed=192.0, waiting=False)
    base.update(kw)
    return dp.Reading(**base)


def test_a_wide_line_has_every_part():
    line = dp.render_line(reading(), 200)
    for part in ("model", "21.4 GB / 54.7 GB", "39%", "3 of 11 files", "118 MB/s", "ETA 4m 40s", "3m 12s"):
        assert part in line, part
    assert "org/" not in line                                          # the org is left off to save width
    assert "█" in line and "░" in line


@pytest.mark.parametrize("width", range(30, 201, 5))
def test_a_line_fits_the_terminal_and_drops_parts_from_the_end(width):
    line = dp.render_line(reading(), width)
    assert len(line) <= width - 1                                      # one column spare: a full line would wrap
    present = [p for p in ("118 MB/s", "ETA 4m 40s", "3 of 11 files", "elapsed 3m 12s") if p in line]
    order = ["118 MB/s", "ETA 4m 40s", "3 of 11 files", "elapsed 3m 12s"]
    assert present == order[:len(present)]                             # a later part never survives an earlier one


def test_a_line_with_room_for_the_basics_keeps_the_size_and_percentage():
    line = dp.render_line(reading(), 70)
    assert "21.4 GB / 54.7 GB" in line and "39%" in line


def test_a_very_narrow_terminal_gets_a_cut_line_not_a_wrapped_one():
    assert len(dp.render_line(reading(), 12)) <= 11


def test_waiting_replaces_the_speed_and_the_eta_is_left_out():
    line = dp.render_line(reading(rate=0.0, eta=None, waiting=True), 200)
    assert "waiting for data" in line and "ETA" not in line and "MB/s" not in line


def test_an_unknown_total_shows_the_bytes_and_no_bar_or_percentage():
    line = dp.render_line(reading(need=0, done=5_000_000), 200)
    assert "5 MB" in line and "%" not in line and "█" not in line and "░" not in line


def test_the_eta_waits_for_enough_measurements_to_mean_something():
    early = dp.render_line(reading(elapsed=dp.ETA_AFTER_S - 1, eta=888.0), 200)
    assert "ETA" not in early and "118 MB/s" in early                 # the speed shows at once, the guess does not
    assert "ETA 4m 40s" in dp.render_line(reading(elapsed=dp.ETA_AFTER_S, eta=280.0), 200)


def test_a_log_line_leaves_out_an_early_eta_too(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=False)
    m.begin()
    clock.advance(2)
    grow(blobs, "b", 200)
    m.tick()
    assert "ETA" not in out.getvalue().splitlines()[-1] and "B/s" in out.getvalue().splitlines()[-1]
    clock.advance(dp.ETA_AFTER_S)
    grow(blobs, "b", 500)
    m.tick()
    assert "ETA" in out.getvalue().splitlines()[-1]


def test_a_finished_download_reads_one_hundred_percent():
    line = dp.render_line(reading(done=54_700_000_000), 200)
    assert "100%" in line and "█" * 20 in line


def test_the_bar_is_painted_with_the_accent_when_a_painter_is_given():
    seen = []
    dp.render_line(reading(), 200, paint=lambda text, role: seen.append(role) or text)
    assert "accent" in seen


# ---- output ----------------------------------------------------------------------------------------------

def test_on_a_terminal_the_line_is_redrawn_in_place_and_ends_with_a_summary(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=True)
    m.begin()
    clock.advance(2)
    grow(blobs, "b", 250)
    m.tick()
    text = out.getvalue()
    assert text.count("\r\x1b[K") >= 2 and "\n" not in text            # redrawn, never a new line
    assert "25%" in text.split("\r\x1b[K")[-1]
    clock.advance(2)
    grow(blobs, "b", 1000)
    m.finish(True)
    last = out.getvalue().split("\r\x1b[K")[-1]
    assert last.startswith("downloaded 1 KB in 4s") and last.endswith("\n")
    assert out.getvalue().count("\n") == 1


def test_a_failed_download_clears_the_line_and_claims_nothing(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, tty=True)
    m.begin()
    m.finish(False)
    assert out.getvalue().endswith("\r\x1b[K") and "downloaded" not in out.getvalue()


def test_a_log_gets_a_line_when_the_percentage_passes_a_step_and_never_a_carriage_return(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=False)
    m.begin()
    assert out.getvalue() == "downloading model: 1 KB to fetch (0 of 2 files)\n"
    for size, expect_lines in ((50, 1), (100, 2), (190, 2), (200, 3), (999, 4)):
        clock.advance(1)
        grow(blobs, "b", size)
        m.tick()
        assert out.getvalue().count("\n") == expect_lines, size
    assert "\r" not in out.getvalue() and "\x1b" not in out.getvalue()
    assert "model: 100 B of 1 KB (10%)" in out.getvalue()
    assert next(l for l in out.getvalue().splitlines() if "(10%)" in l).endswith("0 of 2 files")


def test_a_log_with_no_progress_still_gets_a_line_every_minute(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=False)
    m.begin()
    clock.advance(dp.LOG_EVERY_S - 1)
    m.tick()
    assert out.getvalue().count("\n") == 1
    clock.advance(2)
    m.tick()
    assert out.getvalue().count("\n") == 2 and "waiting for data" in out.getvalue().splitlines()[-1]


def test_a_plain_style_gets_log_lines_even_on_a_terminal(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, tty=True, pretty=False)
    m.begin()
    grow(blobs, "b", 500)
    clock.advance(1)
    m.tick()
    assert "\r" not in out.getvalue() and out.getvalue().count("\n") >= 2


def test_the_summary_counts_the_bytes_that_arrived_not_the_size_of_the_files(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=False)
    grow(blobs, "b.incomplete", 900)                                    # an earlier attempt got this far
    m.begin()
    clock.advance(10)
    grow(blobs, "b.incomplete", 1000)                                   # 100 bytes arrived in this one
    m.finish(True)
    assert out.getvalue().splitlines()[-1] == "downloaded 100 B in 10s (10 B/s average)"


def test_the_summary_takes_a_last_measurement_instead_of_the_last_tick(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=False)
    m.begin()
    clock.advance(4)
    grow(blobs, "b", 400)                                               # arrived after the last reading
    m.finish(True)
    assert out.getvalue().splitlines()[-1] == "downloaded 400 B in 4s (100 B/s average)"


def test_a_download_that_fetched_nothing_says_so(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=False)
    m.begin()
    clock.advance(3)
    m.finish(True)
    assert out.getvalue().splitlines()[-1] == "downloaded nothing new in 3s"


def test_a_failed_last_measurement_does_not_stop_the_summary(tmp_path, monkeypatch):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=False)
    m.begin()
    clock.advance(10)
    grow(blobs, "b", 250)
    m.tick()
    clock.advance(10)

    def broken(path):
        raise RuntimeError("the disk went away")
    monkeypatch.setattr(dp, "dir_bytes", broken)
    m.finish(True)                                                      # must not raise
    assert out.getvalue().splitlines()[-1] == "downloaded 250 B in 20s (12 B/s average)"


def test_a_log_ends_with_the_summary_line(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=False)
    m.begin()
    clock.advance(10)
    grow(blobs, "b", 1000)
    m.finish(True)
    assert out.getvalue().splitlines()[-1] == "downloaded 1 KB in 10s (100 B/s average)"


def test_a_failed_download_in_a_log_prints_no_summary(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, tty=False)
    m.begin()
    before = out.getvalue()
    m.finish(False)
    assert out.getvalue() == before


# ---- the thread ------------------------------------------------------------------------------------------

def test_the_thread_redraws_while_the_download_runs_and_is_gone_after_stop(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=1000, tty=True, interval=0.01)
    before = threading.active_count()
    with m:
        for size in (100, 300, 1000):
            grow(blobs, "b", size)
            time.sleep(0.05)
    assert out.getvalue().count("\r\x1b[K") >= 3
    assert out.getvalue().rstrip().split("\x1b[K")[-1].startswith("downloaded 1 KB in ")
    assert threading.active_count() == before


def test_an_exception_in_the_body_is_reported_as_a_failure_and_not_swallowed(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, tty=False, interval=0.01)
    with pytest.raises(RuntimeError):
        with m:
            raise RuntimeError("the download failed")
    assert "downloaded" not in out.getvalue()


def test_stop_without_start_and_stop_twice_are_harmless(tmp_path):
    m, *_ = make(tmp_path)
    m.finish(False)
    m.begin()
    m.finish(True)
    m.finish(True)


def test_a_meter_that_breaks_says_so_once_and_the_download_goes_on(tmp_path, monkeypatch):
    m, blobs, snap, out, clock = make(tmp_path, tty=False, interval=0.01)
    calls = []

    def boom():
        calls.append(1)
        raise OSError("disk went away")

    with m:
        monkeypatch.setattr(m, "reading", boom)
        time.sleep(0.1)
        assert len(calls) == 1                                         # the thread gave up on its first error
    assert out.getvalue().count("progress display stopped: OSError") == 1


def test_a_write_to_a_closed_stream_ends_the_display_without_an_exception(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, tty=True, interval=0.01)
    with m:
        out.close()
        time.sleep(0.05)


def test_the_percentage_is_rounded_down_so_one_hundred_means_done():
    assert "99%" in dp.render_line(reading(done=54_700_000_000 * 996 // 1000), 200)


def test_finishing_twice_prints_the_summary_once(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, tty=False)
    m.begin()
    m.finish(True)
    once = out.getvalue()
    m.finish(True)
    m.finish(False)
    assert out.getvalue() == once and once.count("downloaded") == 1


def test_a_log_with_no_known_total_says_what_has_arrived(tmp_path):
    m, blobs, snap, out, clock = make(tmp_path, total=0, tty=False)
    m.begin()
    assert out.getvalue() == "downloading model\n"
    clock.advance(dp.LOG_EVERY_S)
    grow(blobs, "b", 5)
    m.tick()
    assert out.getvalue().splitlines()[-1] == "downloading model: 5 B so far"


def test_the_thread_never_draws_after_the_summary(tmp_path):
    # `finish` joins the thread before it writes. Without the join a redraw can land after the summary line.
    for n in range(40):
        m, blobs, snap, out, clock = make(tmp_path / str(n) if (tmp_path / str(n)).mkdir() is None else tmp_path,
                                          total=1000, tty=True, interval=0.0005)
        with m:
            grow(blobs, "b", 10 * n)
            time.sleep(0.003)
        assert out.getvalue().endswith("\n") and out.getvalue().rstrip().split("\x1b[K")[-1].startswith("downloaded")
