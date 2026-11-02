# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 409: with `--no-fetch` nothing is downloaded, so the disk row must not count a download and the base row must not promise one;
and a local model the Hub cannot be asked about is a text model when its snapshot has a plain `config.json`. Injected signals, no network."""
from __future__ import annotations

import pytest

from orchard import cli
from orchard import preflight as pf
from orchard.nearest import Installed
from test_cli import Pipe, Recorder, conf, run  # noqa: F401  (conf is a fixture)
from test_preflight import CLEF_PAYLOAD, cfg as pf_cfg, signals as plain_signals

GB = 1_000_000_000


def world(tmp_path, *, base_gb=500, free=100.0):
    payload = dict(CLEF_PAYLOAD, cardData={"license": "x", "base_model": "o/base"})
    base_payload = dict(CLEF_PAYLOAD, id="o/base", cardData={"license": "x"},
                        siblings=[{"rfilename": "model.safetensors", "size": base_gb * GB}])
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)

    def hub_info(m):
        return pf.parse_hub(payload if m == "Cloudflare/clef" else base_payload), None
    return plain_signals(hub_info=hub_info, free_gb=lambda p: free, installed_bundles=lambda: [Installed("b/b", "o/base", "6", 1)],
                         local_snapshot=lambda m: str(snap) if m == "Cloudflare/clef" else None)


def rows(tmp_path, **kw):
    return {c.name: c for c in pf.run_preflight(pf_cfg(tmp_path), "Cloudflare/clef", accept_credentials=True,
                                                signals=world(tmp_path), **kw)}


def test_without_no_fetch_the_base_download_still_blocks_a_small_disk(tmp_path):
    by = rows(tmp_path)
    assert by["disk"].status == "block" and "needs" in by["disk"].detail


def test_with_no_fetch_the_disk_row_counts_no_download(tmp_path):
    by = rows(tmp_path, no_fetch=True)
    assert by["disk"].status == "ok", by["disk"].detail


def test_with_no_fetch_the_base_row_does_not_say_it_will_be_downloaded(tmp_path):
    by = rows(tmp_path, no_fetch=True)
    assert "will be downloaded" not in by["base"].detail
    assert "--no-fetch" in by["base"].detail and "not on disk" in by["base"].detail
    assert by["base"].data["needs_fetch"] is True              # stage 0 still needs the snapshot; only the promise changed


def test_a_local_text_snapshot_makes_a_hubless_model_a_text_model(tmp_path):
    snap = tmp_path / "s"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_text("w")

    class Sig:
        installed_bundles = staticmethod(lambda: [Installed("b/b", "o/base", "6", 1)])
        search_bundles = None
        hub_info = staticmethod(lambda m: (None, "HTTP 401"))
        local_snapshot = staticmethod(lambda m: str(snap))
    assert cli.audio_base_decision("b/b", "local/tiny", Sig) == ("text", None)


def test_a_hubless_model_with_no_snapshot_is_still_unknown(tmp_path):
    class Sig:
        installed_bundles = staticmethod(lambda: [Installed("b/b", "o/base", "6", 1)])
        search_bundles = None
        hub_info = staticmethod(lambda m: (None, "HTTP 401"))
        local_snapshot = staticmethod(lambda m: None)
    state, why = cli.audio_base_decision("b/b", "local/tiny", Sig)
    assert state == "unknown" and why and "could not be asked" in why


def test_a_hubless_snapshot_with_the_audio_layout_is_not_called_text(tmp_path):
    snap = tmp_path / "s"
    snap.mkdir()
    (snap / "model_config.json").write_text("{}")

    class Sig:
        installed_bundles = staticmethod(lambda: [Installed("b/b", "o/base", "6", 1)])
        search_bundles = None
        hub_info = staticmethod(lambda m: (None, "HTTP 401"))
        local_snapshot = staticmethod(lambda m: str(snap))
    assert cli.audio_base_decision("b/b", "local/x", Sig)[0] == "unknown"


def test_the_bringup_command_passes_no_fetch_to_the_preflight(conf, tmp_path):
    """The wiring: `bringup --no-fetch --dry-run` shows no disk-full block for a base that would need a 500 GB download."""
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--no-fetch", "--dry-run"], sig=world(tmp_path))
    assert "disk-full" not in out and "will be downloaded" not in out, out
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], sig=world(tmp_path))
    assert "disk-full" in out, out
