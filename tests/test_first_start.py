# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The first-start check of a published package (orchard/first_start.py, issue 843).

The figures come from the first start of `raahemnabeel/hy4-flash` on a QuietBox: a 440 GB checkpoint downloaded in 107 min 43 s (about 68 MB/s),
about 165 GB of caches built on the first start, 17 GB of host RAM to start and 116 GB for full speed, a pinned-memory limit that failed at 8 MiB,
and a ready timeout of 21600 s. The card text and the manifest in tests/fixtures/first-start are excerpts of the published files."""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import pytest

from orchard import blocked, first_start as fs
from orchard.preflight import BLOCK, OK, WARN

FIX = Path(__file__).parent / "fixtures" / "first-start"
CARD = (FIX / "hy4-flash-card.md").read_text()
MANIFEST = json.loads((FIX / "hy4-flash-manifest-excerpt.json").read_text())


def rows(**kw):
    """The check rows for the Hy4-flash package; the machine is the argument. Defaults: plenty of everything on one disk."""
    base = dict(free_gb={"hf_home": 2000.0, "cache_root": 2000.0}, same_device=True, snapshot_local=False, mem_available_gb=141.0,
                link_mb_s=None, ready_timeout_s=None, min_free_gb=40.0)
    base.update(kw)
    return {c.name: c for c in fs.check_first_start(fs.parse_card(CARD), fs.manifest_figures(MANIFEST), **base)}


# ---- the card ---------------------------------------------------------------------------------------------------------------------

def test_the_figures_of_the_real_card():
    f = fs.parse_card(CARD)
    assert (f["disk_total_gb"].value, f["checkpoint_gb"].value, f["cache_gb"].value) == (610.0, 440.0, 165.0)
    assert (f["ram_full_gb"].value, f["ram_min_gb"].value) == (116.0, 17.0)
    assert f["memlock_advised"].value == 1.0
    assert f["later_start_minutes"].value == 4.0


def test_a_figure_the_card_does_not_state_says_so_and_has_no_value():
    f = fs.parse_card(CARD)
    assert f["first_start_minutes"].value is None and f["first_start_minutes"].source == fs.NOT_STATED


def test_every_figure_names_the_line_of_the_card_it_came_from():
    f = fs.parse_card(CARD)
    assert "About 610 GB of disk" in f["disk_total_gb"].source and f["disk_total_gb"].source.startswith("card line ")
    assert "116 GB of free host RAM" in f["ram_full_gb"].source
    assert "--ulimit memlock" in f["memlock_advised"].source


def test_a_card_without_any_of_it_gives_no_figure_and_no_error():
    f = fs.parse_card("# Some model\n\nIt is a model.\n")
    assert all(v.value is None and v.source == fs.NOT_STATED for v in f.values())
    assert set(f) == set(fs.parse_card(CARD))


@pytest.mark.parametrize("text,key,want", [
    ("The first start takes about 25 minutes.", "first_start_minutes", 25.0),
    ("The first start is ready in about 2 hours.", "first_start_minutes", 120.0),
    ("About 2.5 GB of free host RAM (1 GB minimum).", "ram_full_gb", 2.5),
    ("About 99999999 GB of disk", "disk_total_gb", None),                      # a figure above the bound is no figure
    ("About 99999999999999999999 GB of free host RAM", "ram_full_gb", None),
])
def test_wording_variants_and_bounds(text, key, want):
    assert fs.parse_card(text)[key].value == want


def test_a_hostile_card_costs_no_time():
    text = "about " + "9" * 200000 + " GB of disk\n" + "a" * 500000 + "\n" + ("about 1 " * 50000)
    t = time.monotonic()
    fs.parse_card(text)
    assert time.monotonic() - t < 2.0


# ---- the manifest -----------------------------------------------------------------------------------------------------------------

def test_the_ready_timeout_is_read_from_the_serve_environment():
    f = fs.manifest_figures(MANIFEST)
    assert f["ready_timeout_s"].value == 21600.0 and "VLLM_ENGINE_READY_TIMEOUT_S" in f["ready_timeout_s"].source


@pytest.mark.parametrize("manifest", [{}, {"container": {}}, {"container": {"serve": {"env": {"VLLM_ENGINE_READY_TIMEOUT_S": "soon"}}}},
                                       {"container": {"serve": {"env": {"VLLM_ENGINE_READY_TIMEOUT_S": "-5"}}}},
                                       {"container": {"serve": {"env": {"VLLM_ENGINE_READY_TIMEOUT_S": "1e999"}}}}, None, [], "x"])
def test_a_manifest_without_a_usable_timeout_gives_none(manifest):
    f = fs.manifest_figures(manifest)
    assert f["ready_timeout_s"].value is None and f["ready_timeout_s"].source == fs.NOT_STATED


def test_the_weights_repo_and_revision_come_from_the_manifest():
    assert fs.manifest_weights(MANIFEST) == ("tencent/Hy4-flash", "b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f")
    assert fs.manifest_weights({"weights": {"repo_id": 5}}) is None and fs.manifest_weights(None) is None


# ---- disk -------------------------------------------------------------------------------------------------------------------------

def test_disk_after_the_first_start_on_a_box_with_room():
    """658 GB free on one disk: the 440 GB checkpoint and 165 GB of caches take 605 GB and leave 53 GB, which is above the 40 GB margin."""
    r = rows(free_gb={"hf_home": 658.0, "cache_root": 658.0})["first-start disk"]
    assert r.status == OK and "53 GB free after the first start" in r.detail and "605 GB" in r.detail


def test_disk_that_is_short_blocks_as_disk_full_with_both_figures():
    """152 GB free (the box after the first start filled it): the first start needs 605 GB and the margin."""
    r = rows(free_gb={"hf_home": 152.0, "cache_root": 152.0})["first-start disk"]
    assert r.status == BLOCK and r.reason == "disk-full" and "needs 645 GB, 152 GB free" in r.detail


def test_disk_just_enough_for_the_figures_but_not_the_margin_blocks():
    assert rows(free_gb={"hf_home": 640.0, "cache_root": 640.0})["first-start disk"].status == BLOCK
    assert rows(free_gb={"hf_home": 645.0, "cache_root": 645.0})["first-start disk"].status == OK


def test_a_snapshot_already_on_disk_is_not_downloaded_again():
    r = rows(free_gb={"hf_home": 250.0, "cache_root": 250.0}, snapshot_local=True)["first-start disk"]
    assert r.status == OK and "205 GB" in r.detail and "checkpoint is on disk" in r.detail


def test_two_disks_are_judged_one_by_one():
    r = rows(free_gb={"hf_home": 500.0, "cache_root": 100.0}, same_device=False)["first-start disk"]
    assert r.status == BLOCK and r.reason == "disk-full" and "cache_root: needs 205 GB, 100 GB free" in r.detail and "hf_home" not in r.detail


def test_only_the_stated_total_is_used_when_the_split_is_not_stated():
    card = fs.parse_card("About 610 GB of disk.\n")
    out = fs.check_first_start(card, fs.manifest_figures(None), free_gb={"hf_home": 600.0, "cache_root": 600.0}, same_device=True,
                               snapshot_local=False, mem_available_gb=None, min_free_gb=40.0)
    disk = next(c for c in out if c.name == "first-start disk")
    assert disk.status == BLOCK and "needs 650 GB, 600 GB free" in disk.detail


def test_no_disk_figure_is_a_row_that_says_so_and_never_a_block():
    out = fs.check_first_start(fs.parse_card("nothing"), fs.manifest_figures(None), free_gb={"hf_home": 1.0, "cache_root": 1.0},
                               same_device=True, snapshot_local=False, mem_available_gb=1.0, min_free_gb=40.0)
    assert {c.name: c.status for c in out}["first-start disk"] == OK
    assert all(c.status != BLOCK for c in out) and "not stated" in next(c for c in out if c.name == "first-start disk").detail


# ---- host RAM and pinned memory ----------------------------------------------------------------------------------------------------

def test_ram_above_the_full_speed_figure_is_ok():
    r = rows(mem_available_gb=141.0)["first-start host ram"]
    assert r.status == OK and "116 GB" in r.detail


def test_ram_between_the_minimum_and_the_full_speed_figure_warns():
    r = rows(mem_available_gb=90.0)["first-start host ram"]
    assert r.status == WARN and "17 GB" in r.detail and "116 GB" in r.detail and "slower" in r.detail


def test_ram_below_the_stated_minimum_blocks():
    r = rows(mem_available_gb=10.0)["first-start host ram"]
    assert r.status == BLOCK and r.reason == "model-does-not-fit" and "needs 17 GB to start, 10 GB available" in r.detail


def test_ram_exactly_at_the_minimum_is_not_a_block():
    assert rows(mem_available_gb=17.0)["first-start host ram"].status == WARN


def test_an_unreadable_mem_available_is_a_warning():
    assert rows(mem_available_gb=None)["first-start host ram"].status == WARN


def test_pinned_memory_is_a_warning_when_the_card_advises_the_ulimit():
    r = rows()["first-start memlock"]
    assert r.status == WARN and "--ulimit memlock=-1:-1" in r.detail


def test_pinned_memory_is_not_stated_when_the_card_says_nothing():
    out = fs.check_first_start(fs.parse_card("x"), fs.manifest_figures(None), free_gb={"hf_home": 1e4, "cache_root": 1e4}, same_device=True,
                               snapshot_local=False, mem_available_gb=100.0, min_free_gb=40.0)
    r = next(c for c in out if c.name == "first-start memlock")
    assert r.status == OK and "not stated" in r.detail


# ---- download time and the ready timeout -------------------------------------------------------------------------------------------

def test_download_time_at_the_measured_rate_of_the_first_start():
    r = rows(link_mb_s=68.0)["first-start download"]
    assert r.status == OK and "about 108 minutes" in r.detail and "440 GB" in r.detail


def test_download_time_without_a_rate_says_the_rate_is_missing():
    r = rows()["first-start download"]
    assert r.status == OK and "link speed not given" in r.detail


def test_no_download_when_the_snapshot_is_on_disk():
    assert "on disk" in rows(snapshot_local=True, link_mb_s=68.0)["first-start download"].detail


@pytest.mark.parametrize("rate", [0.0, -5.0, float("nan"), float("inf")])
def test_a_rate_that_is_not_a_speed_is_refused_as_missing(rate):
    assert "link speed not given" in rows(link_mb_s=rate)["first-start download"].detail


def test_the_ready_timeout_of_the_manifest_with_no_stated_first_start_time():
    r = rows()["first-start ready timeout"]
    assert r.status == OK and "21600 s" in r.detail and "6 h" in r.detail and "states no first-start time" in r.detail


def test_a_ready_timeout_shorter_than_the_stated_first_start_blocks():
    card = fs.parse_card(CARD + "\nThe first start takes about 25 minutes.\n")
    base = dict(free_gb={"hf_home": 2000.0, "cache_root": 2000.0}, same_device=True, snapshot_local=False, mem_available_gb=141.0, min_free_gb=40.0)
    short = next(c for c in fs.check_first_start(card, fs.manifest_figures(MANIFEST), ready_timeout_s=600.0, **base) if c.name == "first-start ready timeout")
    assert short.status == BLOCK and short.reason in blocked.REASONS and "600 s" in short.detail and "25 minutes" in short.detail
    ok = next(c for c in fs.check_first_start(card, fs.manifest_figures(MANIFEST), ready_timeout_s=3600.0, **base) if c.name == "first-start ready timeout")
    assert ok.status == OK


def test_a_timeout_that_is_not_set_says_the_default_applies():
    out = fs.check_first_start(fs.parse_card(CARD), fs.manifest_figures({}), free_gb={"hf_home": 1e4, "cache_root": 1e4}, same_device=True,
                               snapshot_local=False, mem_available_gb=141.0, min_free_gb=40.0)
    assert "not set" in next(c for c in out if c.name == "first-start ready timeout").detail


def test_the_configured_timeout_overrides_the_manifest_one():
    r = rows(ready_timeout_s=7200.0)["first-start ready timeout"]
    assert "7200 s" in r.detail and "21600" not in r.detail


def test_every_block_uses_a_reason_the_blocked_end_state_knows():
    worst = rows(free_gb={"hf_home": 1.0, "cache_root": 1.0}, mem_available_gb=1.0)
    for c in worst.values():
        assert c.status != BLOCK or c.reason in blocked.REASONS


def test_the_rows_come_in_a_fixed_order():
    assert list(rows()) == ["first-start disk", "first-start host ram", "first-start memlock", "first-start download", "first-start ready timeout"]


# ---- the command --------------------------------------------------------------------------------------------------------------------

def package(tmp_path, *, card=True, manifest=True):
    d = tmp_path / "pkg"
    d.mkdir()
    if card:
        (d / "README.md").write_text(CARD)
    if manifest:
        (d / "tt_kernel_manifest.json").write_text(json.dumps(MANIFEST))
    return d


def run(args, facts):
    out = io.StringIO()
    code = fs.main(fs.parser().parse_args(args), out=out, facts=facts)
    return code, out.getvalue()


ROOMY = lambda hf, cache: {"free_gb": {"hf_home": 2000.0, "cache_root": 2000.0}, "same_device": True, "snapshot_local": False, "mem_available_gb": 141.0}
TIGHT = lambda hf, cache: {"free_gb": {"hf_home": 152.0, "cache_root": 152.0}, "same_device": True, "snapshot_local": False, "mem_available_gb": 141.0}


def test_the_command_prints_one_line_per_row_and_exits_zero_when_nothing_blocks(tmp_path):
    code, text = run([str(package(tmp_path))], ROOMY)
    assert code == 0 and text.count("first-start ") >= 5 and "[block]" not in text
    assert "[warn]" in text and "[ok]" in text


def test_the_command_exits_blocked_when_a_minimum_is_violated(tmp_path):
    code, text = run([str(package(tmp_path))], TIGHT)
    assert code == fs.EXIT_BLOCKED and "[block]" in text and "disk-full" in text


def test_the_command_reads_a_card_and_a_manifest_named_apart(tmp_path):
    d = package(tmp_path)
    code, text = run(["--card", str(d / "README.md"), "--manifest", str(d / "tt_kernel_manifest.json")], ROOMY)
    assert code == 0 and "first-start disk" in text


def test_a_missing_manifest_is_not_stated_and_not_an_error(tmp_path):
    code, text = run([str(package(tmp_path, manifest=False))], ROOMY)
    assert code == 0 and "not set" in text


def test_a_missing_card_is_a_refusal(tmp_path):
    code, text = run([str(package(tmp_path, card=False))], ROOMY)
    assert code == fs.EXIT_REFUSED and "card" in text


def test_a_manifest_that_is_not_json_is_a_refusal(tmp_path):
    d = package(tmp_path)
    (d / "tt_kernel_manifest.json").write_text("{not json")
    code, text = run([str(d)], ROOMY)
    assert code == fs.EXIT_REFUSED and "manifest" in text


def test_a_card_over_the_size_cap_is_a_refusal(tmp_path):
    d = package(tmp_path)
    (d / "README.md").write_bytes(b"x" * (fs.CARD_MAX_BYTES + 1))
    assert run([str(d)], ROOMY)[0] == fs.EXIT_REFUSED


def test_a_card_that_is_a_pipe_is_a_refusal_and_does_not_hang(tmp_path):
    d = package(tmp_path, card=False)
    os.mkfifo(d / "README.md")
    t = time.monotonic()
    assert run([str(d)], ROOMY)[0] == fs.EXIT_REFUSED
    assert time.monotonic() - t < 5.0


def test_a_card_that_is_a_link_is_a_refusal(tmp_path):
    d = package(tmp_path, card=False)
    real = tmp_path / "elsewhere.md"
    real.write_text(CARD)
    os.symlink(real, d / "README.md")
    assert run([str(d)], ROOMY)[0] == fs.EXIT_REFUSED


@pytest.mark.parametrize("flag,value", [("--link-mbps", "0"), ("--link-mbps", "nan"), ("--link-mbps", "-3"), ("--ready-timeout-s", "0"),
                                        ("--ready-timeout-s", "inf"), ("--min-free-gb", "-1")])
def test_a_flag_that_is_not_a_positive_finite_number_is_a_refusal(tmp_path, flag, value):
    assert run([str(package(tmp_path)), flag, value], ROOMY)[0] == fs.EXIT_REFUSED


def test_a_local_snapshot_is_found_in_the_hub_cache(tmp_path):
    hf = tmp_path / "hf"
    snap = hf / "hub" / "models--tencent--Hy4-flash" / "snapshots" / "b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert fs.snapshot_present(hf, "tencent/Hy4-flash", "b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f") is True
    assert fs.snapshot_present(hf, "tencent/Hy4-flash", "0" * 40) is False
    assert fs.snapshot_present(hf, "../../etc", "b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f") is False


def test_a_figure_above_the_bound_is_no_figure_even_when_it_is_a_well_formed_number():
    """500,000 GB is six digits (the pattern reads it) and above MAX_FIGURE: only the bound refuses it."""
    f = fs.parse_card("About 500000 GB of disk\nAbout 500000 GB of free host RAM (400000 GB minimum)\n")
    assert f["disk_total_gb"].value is None and f["ram_full_gb"].value is None and f["ram_min_gb"].value is None
    assert fs.parse_card("About 100000 GB of disk")["disk_total_gb"].value == 100000.0


def test_a_line_longer_than_the_bound_is_skipped_and_one_at_the_bound_is_read():
    too_long = "x" * (fs.MAX_LINE + 10) + " About 610 GB of disk"
    assert fs.parse_card(too_long)["disk_total_gb"].value is None
    at_bound = "About 610 GB of disk" + " " * (fs.MAX_LINE - len("About 610 GB of disk"))
    assert len(at_bound) == fs.MAX_LINE and fs.parse_card(at_bound)["disk_total_gb"].value == 610.0


def test_a_revision_that_climbs_out_of_the_snapshots_directory_is_no_snapshot(tmp_path):
    """`snapshots/..` is the model's own directory, which exists and is not empty: only the revision check refuses it."""
    hf = tmp_path / "hf"
    snaps = hf / "hub" / "models--org--name" / "snapshots"
    (snaps / ("a" * 40)).mkdir(parents=True)
    (snaps / ("a" * 40) / "config.json").write_text("{}")
    assert fs.snapshot_present(hf, "org/name", "a" * 40) is True
    for bad in ("..", ".", "../snapshots/" + "a" * 40, "a" * 39, "A" * 40, ""):
        assert fs.snapshot_present(hf, "org/name", bad) is False
    for bad_repo in ("org", "org/name/extra", "/name", "org/..", "org/.hidden", "", "org/na me"):
        assert fs.snapshot_present(hf, bad_repo, "a" * 40) is False


@pytest.mark.parametrize("body", ["[1, 2]", '"text"', "7", "null"])
def test_a_manifest_that_is_json_but_not_an_object_is_a_refusal(tmp_path, body):
    d = package(tmp_path)
    (d / "tt_kernel_manifest.json").write_text(body)
    code, text = run([str(d)], ROOMY)
    assert code == fs.EXIT_REFUSED and "manifest" in text
