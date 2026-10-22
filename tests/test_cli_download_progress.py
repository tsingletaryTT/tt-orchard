# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard bringup` shows a progress line while it downloads weights (orchard/dlprogress.py around the fetcher)."""
import functools
import threading
import time
from pathlib import Path

import pytest

from orchard import cli, dlprogress, fetch
from test_cli import (BUNDLE, FOUND, GEMMA_IT, BaseWorld, Pipe, Recorder, Tty, conf, run, run_base,  # noqa: F401
                      signals)

MODEL = "Cloudflare/clef"
REV = "2f3de3dd85f379784083b0814d997ab627200f0c"


class Growing(Recorder):
    """A downloader that writes into the repo's blobs directory in steps, the way `hf download` does."""

    def __init__(self, steps=(1_000_000_000, 3_000_000_000, 5_000_000_000), pause=0.04, fail=None):
        super().__init__()
        self.steps, self.pause, self.fail, self.during = steps, pause, fail, []

    def fetcher(self, model_id, revision, hf_home, *, files, **kw):
        blobs = fetch.blob_dir(hf_home, model_id)
        blobs.mkdir(parents=True, exist_ok=True)
        for size in self.steps:
            with open(blobs / "big.incomplete", "wb") as f:
                f.truncate(size)
            self.during.append(threading.active_count())
            time.sleep(self.pause)
        if self.fail:
            raise fetch.FetchError(self.fail)
        return super().fetcher(model_id, revision, hf_home, files=files)


@pytest.fixture
def quick(monkeypatch):
    """The real meter, polling fast so a short fake download shows several updates."""
    monkeypatch.setattr(cli.dlprogress, "DownloadMeter", functools.partial(dlprogress.DownloadMeter, interval=0.01))


def go(conf, args, rec, out, **kw):
    code = cli.main(["--config", str(conf), *args], env=kw.pop("env", {}), stdout=out,
                    signals=kw.pop("sig", None) or signals(), supervisor_main=rec.supervisor, fetcher=rec.fetcher)
    return code, out.getvalue()


def test_the_blob_directory_is_where_hf_keeps_a_repos_blobs(tmp_path):
    assert fetch.blob_dir(tmp_path, "Cloudflare/clef") == tmp_path / "hub" / "models--Cloudflare--clef" / "blobs"


def test_a_download_in_a_log_prints_a_start_line_progress_lines_and_a_summary(conf, quick):
    rec = Growing()
    code, out = go(conf, ["bringup", MODEL], rec, Pipe())
    lines = out.splitlines()
    fetching = next(i for i, l in enumerate(lines) if "fetching Cloudflare/clef at" in l)
    start = next(i for i, l in enumerate(lines) if l.startswith("downloading clef: 5.0 GB to fetch (0 of 3 files)"))
    assert fetching < start
    assert any(l.startswith("downloading clef: ") and "%" in l for l in lines), lines     # a progress line in between
    summary = next(i for i, l in enumerate(lines) if l.startswith("downloaded 5.0 GB in"))
    assert start < summary == max(i for i, l in enumerate(lines) if l.startswith("downloaded"))
    assert "\r" not in out and "\x1b[K" not in out
    assert code == 0 and rec.argv is not None                          # and the run still starts


def test_on_a_terminal_the_line_is_redrawn_while_the_download_runs(conf, quick):
    rec = Growing()
    code, out = go(conf, ["bringup", MODEL, "--style", "pretty"], rec, Tty())
    assert out.count("\r\x1b[K") >= 3
    assert "downloaded 5.0 GB in" in out
    assert "clef" in out.split("\r\x1b[K")[2]


def test_the_progress_thread_is_gone_when_the_download_ends(conf, quick):
    before = threading.active_count()
    go(conf, ["bringup", MODEL], Growing(), Pipe())
    assert threading.active_count() == before


def test_a_failed_download_prints_no_summary_and_still_refuses_with_its_message(conf, quick, capsys):
    rec = Growing(fail="`hf download Cloudflare/clef` exited 1:\nconnection reset")
    code, out = go(conf, ["bringup", MODEL], rec, Pipe())
    assert code == cli.EXIT_REFUSED and rec.argv is None
    assert "downloaded" not in out
    assert "connection reset" in capsys.readouterr().err


def test_a_snapshot_that_is_already_local_shows_no_download_line(conf, quick, tmp_path):
    local = tmp_path / "hf" / "hub" / "models--Cloudflare--clef" / "snapshots" / "oldrev"
    rec = Growing()
    code, out = go(conf, ["bringup", MODEL], rec, Pipe(), sig=signals(local_snapshot=lambda m: local))
    assert rec.fetched == [] and "downloading" not in out and "downloaded" not in out


def test_quiet_turns_the_progress_line_off_but_not_the_download(conf, quick):
    rec = Growing()
    code, out = go(conf, ["bringup", MODEL, "--quiet"], rec, Pipe())
    assert rec.fetched and "downloading" not in out and "downloaded" not in out


def test_a_dry_run_downloads_nothing_and_shows_no_progress(conf, quick):
    rec = Growing()
    code, out = go(conf, ["bringup", MODEL, "--dry-run"], rec, Pipe())
    assert rec.fetched == [] and "downloading" not in out


def test_a_plain_style_terminal_gets_log_lines_not_a_redrawn_line(conf, quick):
    rec = Growing()
    code, out = go(conf, ["bringup", MODEL, "--style", "plain"], rec, Tty())
    assert "\r" not in out and "downloading clef: 5.0 GB to fetch" in out


def test_the_base_model_gets_its_own_progress_line_after_the_model(conf, quick):
    w = BaseWorld(found=FOUND)
    rec = Growing()
    out = Pipe()
    code = cli.main(["--config", str(conf), "bringup", "jialinyyzz/humanizer"], env={}, stdout=out,
                    signals=w.signals(), supervisor_main=rec.supervisor, fetcher=rec.fetcher,
                    chooser=lambda c, m, b: BUNDLE, puller=w.puller)
    lines = [l for l in out.getvalue().splitlines() if l.startswith("downloaded")]
    assert code == 0 and len(lines) == 2
    starts = [l for l in out.getvalue().splitlines() if l.startswith("downloading ")]
    assert starts[0].startswith("downloading humanizer:") and "gemma" in starts[-1]
