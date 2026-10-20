# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `sfpi` row of the bringup preflight: is the box's SFPI compiler at least what the bundle pins? (autonomy, 2026-10-18)

On 2026-10-13 a system update downgraded SFPI from 7.83.0 to 7.67.0. Every host-served boot of the Qwen3.8 bundle then
failed in 13 s with `'vLut8si' is not a member of 'sfpi'`, and three boots were spent before dpkg.log showed why. The
bundle's own ttnn wheel pins the version it was built with (`ttnn/tt_metal/sfpi-version`, 7.73.0 for the Qwen3.8-27B
bundle); 7.83.0 and 7.87.0 booted it and 7.67.0 did not. The row compares the two before the run starts. It can only
block: installing a system package is the operator's call."""
from __future__ import annotations

from pathlib import Path

from orchard import preflight as pf
from orchard import sfpi
from test_preflight import CLEF_PAYLOAD, LAB_FREE, cfg, lab_cfg, signals

BUNDLE = "episod/qwen3.8-27b-dflash2-p300"


def bundle_dir(root: Path, pin: str | None, name: str = BUNDLE) -> Path:
    d = root / name
    tt = d / "venv" / "lib" / "python3.12" / "site-packages" / "ttnn" / "tt_metal"
    tt.mkdir(parents=True)
    if pin is not None:
        (tt / "sfpi-version").write_text(f"# sfpi release {pin}\n\nsfpi_repo='https://x'\nsfpi_version='{pin}'\nsfpi_build='884'\n")
    return d


def test_versions_compare_as_numbers_not_text():
    assert sfpi.parse_version("7.73.0") == (7, 73, 0)
    assert sfpi.parse_version("7.9.0") < sfpi.parse_version("7.73.0") < sfpi.parse_version("7.87.0")
    assert sfpi.parse_version("7.87.0-1ubuntu2") == (7, 87, 0)
    assert sfpi.parse_version("garbage") is None and sfpi.parse_version(None) is None


def test_the_pin_is_read_from_the_bundles_ttnn_wheel(tmp_path):
    bundle_dir(tmp_path, "7.73.0")
    assert sfpi.pinned_version(tmp_path, BUNDLE) == "7.73.0"


def test_a_bundle_with_no_pin_file_or_an_unreadable_one_gives_none(tmp_path):
    bundle_dir(tmp_path, None)
    assert sfpi.pinned_version(tmp_path, BUNDLE) is None
    assert sfpi.pinned_version(tmp_path, "nobody/nothing") is None
    d = bundle_dir(tmp_path, "7.73.0", "org/odd")
    (next(d.rglob("sfpi-version"))).write_text("no version here\n")
    assert sfpi.pinned_version(tmp_path, "org/odd") is None


def test_the_installed_version_comes_from_dpkg():
    calls = []

    def run(argv):
        calls.append(argv)
        return 0, "7.87.0\n"
    assert sfpi.installed_version(run) == "7.87.0"
    assert calls[0][0] == "dpkg-query" and "sfpi" in calls[0]
    assert sfpi.installed_version(lambda argv: (1, "")) is None
    assert sfpi.installed_version(lambda argv: (_ for _ in ()).throw(OSError("no dpkg"))) is None


def test_an_installed_version_at_or_above_the_pin_is_ok_and_names_both():
    for installed in ("7.73.0", "7.87.0"):
        c = sfpi.check_sfpi(installed, "7.73.0", BUNDLE)
        assert c.status == pf.OK and installed in c.detail and "7.73.0" in c.detail and BUNDLE in c.detail


def test_an_installed_version_below_the_pin_blocks_as_hardware_unhealthy_and_says_how_to_fix_it():
    c = sfpi.check_sfpi("7.67.0", "7.73.0", BUNDLE)
    assert (c.status, c.reason) == (pf.BLOCK, "hardware-unhealthy")
    assert "7.67.0" in c.detail and "7.73.0" in c.detail and "tenstorrent/sfpi" in c.detail
    assert "vLut8si" in c.detail                      # the symptom, so a person who saw it recognises the cause


def test_a_numeric_comparison_decides_7_9_is_below_7_73():
    assert sfpi.check_sfpi("7.9.0", "7.73.0", BUNDLE).status == pf.BLOCK


def test_an_unreadable_installed_version_is_a_warning_and_never_a_block():
    c = sfpi.check_sfpi(None, "7.73.0", BUNDLE)
    assert c.status == pf.WARN and "7.73.0" in c.detail


def test_a_bundle_that_pins_nothing_gives_no_row():
    assert sfpi.check_sfpi("7.87.0", None, BUNDLE) is None


def test_the_row_follows_the_base_row_and_asks_about_the_chosen_bundle(tmp_path):
    asked = []

    def probe(bundle):
        asked.append(bundle)
        return "7.67.0", "7.73.0"
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(sfpi=probe, installed_bundles=lambda: [BUNDLE_INSTALLED], hub_info=lambda m: (HUB, None),
                                           local_snapshot=lambda m: Path("/hf/snap")))
    names = [c.name for c in out]
    assert names.index("sfpi") == names.index("base") + 1
    assert asked == [BUNDLE]
    assert [c.reason for c in pf.blocked(out)] == ["hardware-unhealthy"]


def test_a_failing_probe_is_a_warning_and_never_an_exception(tmp_path):
    def boom(bundle):
        raise OSError("no dpkg")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(sfpi=boom, installed_bundles=lambda: [BUNDLE_INSTALLED], hub_info=lambda m: (HUB, None),
                                           local_snapshot=lambda m: Path("/hf/snap")))
    row = next(c for c in out if c.name == "sfpi")
    assert row.status == pf.WARN and "could not check" in row.detail
    assert not pf.blocked(out)


def test_no_bundle_or_no_signal_means_no_row(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert "sfpi" not in [c.name for c in out]
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(sfpi=lambda b: ("7.0.0", "7.73.0")))      # base has no bundle
    assert "sfpi" not in [c.name for c in out]


def test_the_default_signals_carry_a_real_probe(tmp_path):
    assert callable(pf.default_signals(cfg(tmp_path)).sfpi)


from orchard.nearest import Installed        # noqa: E402  (after the tests that name it)

BUNDLE_INSTALLED = Installed(BUNDLE, "org/base-27b", "6", 2)
HUB = pf.parse_hub({**CLEF_PAYLOAD, "cardData": {"license": "apache-2.0", "base_model": "org/base-27b"}})


# ---- a lab box: the kernels compile on the lab, so the lab's SFPI is the one that counts (PR 11 review, finding 2) ----

def _lab_run(tmp_path, *, brain, lab, with_lab_signal=True):
    from orchard.bringup_config import Lab
    c = lab_cfg(tmp_path)
    kw = dict(sfpi=lambda b: (brain, "7.73.0"), installed_bundles=lambda: [BUNDLE_INSTALLED],
              hub_info=lambda m: (HUB, None), local_snapshot=lambda m: Path("/hf/snap"),
              lab_status=lambda: (True, LAB_FREE, ""))
    if with_lab_signal:
        kw["lab_sfpi"] = lab
    return pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=signals(**kw))


def test_in_lab_mode_a_stale_sfpi_on_this_box_does_not_block(tmp_path):
    out = _lab_run(tmp_path, brain="7.67.0", lab=lambda: "7.87.0")
    row = next(c for c in out if c.name == "sfpi")
    assert row.status == pf.OK and "7.87.0" in row.detail and "node4" in row.detail
    assert not pf.blocked(out)


def test_in_lab_mode_a_stale_sfpi_on_the_lab_blocks_and_names_the_lab(tmp_path):
    out = _lab_run(tmp_path, brain="7.87.0", lab=lambda: "7.67.0")
    row = next(c for c in out if c.name == "sfpi")
    assert (row.status, row.reason) == (pf.BLOCK, "hardware-unhealthy")
    assert "7.67.0" in row.detail and "node4" in row.detail


def test_in_lab_mode_an_unreadable_lab_version_is_a_warning(tmp_path):
    out = _lab_run(tmp_path, brain="7.67.0", lab=lambda: None)
    row = next(c for c in out if c.name == "sfpi")
    assert row.status == pf.WARN and "node4" in row.detail and not pf.blocked(out)


def test_in_lab_mode_a_failing_lab_probe_is_a_warning_and_never_an_exception(tmp_path):
    def boom():
        raise OSError("ssh died")
    out = _lab_run(tmp_path, brain="7.67.0", lab=boom)
    row = next(c for c in out if c.name == "sfpi")
    assert row.status == pf.WARN and not pf.blocked(out)


def test_in_lab_mode_with_no_lab_probe_the_row_is_left_out_instead_of_reading_this_box(tmp_path):
    out = _lab_run(tmp_path, brain="7.67.0", lab=None, with_lab_signal=False)
    assert "sfpi" not in [c.name for c in out]


def test_the_default_signals_read_the_labs_version_over_ssh(tmp_path, monkeypatch):
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "7.87.0\n", ""
    monkeypatch.setattr(pf.subprocess, "run", lambda argv, **kw: calls.append(argv) or Done())
    c = lab_cfg(tmp_path)
    assert pf.default_signals(c).lab_sfpi() == "7.87.0"
    assert calls and "node4" in calls[0] and "dpkg-query" in calls[0][-1] and "sfpi" in calls[0][-1]
