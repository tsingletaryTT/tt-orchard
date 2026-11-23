# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tt-orchard updates, host-stack rows: SFPI, tt-kmd, tt-flash, tt-smi, the firmware bundle and the reference venv (orchard/updates_host.py).

Every test builds the host from fake command outputs and files under tmp_path. Nothing here reads the real /sys, runs a real command or
opens a chip. The tests that matter most assert what the check does NOT do: no tt-smi or tt-flash process, no command outside the fixed
forms, no wait on a driver that does not answer.
"""
import json
import threading

import pytest

from orchard import updates, updates_host, verifications
from orchard.updates import BEHIND, CURRENT, UNKNOWN, NotReadOnly, Tools
from orchard.updates_host import HostEnv

POLICY = "{pkg}:\n  Installed: {inst}\n  Candidate: {cand}\n  Version table:\n *** {inst} 500\n"


def j(x):
    return (200, json.dumps(x).encode(), "")


class Net:
    def __init__(self, routes):
        self.routes, self.seen = routes, []

    def __call__(self, url, headers, timeout, max_bytes):
        self.seen.append(url)
        for prefix, ans in self.routes:
            if url.startswith(prefix):
                return ans
        return 404, b"", "HTTP 404"


class Runs:
    def __init__(self, table):
        self.table, self.seen = table, []

    def __call__(self, argv, timeout, env=None):
        self.seen.append(list(argv))
        for k, v in self.table.items():
            if " ".join(argv) == k:
                return v
        return 1, "", "no such"


def feed(*entries):
    """An Atom feed in GitHub's shape; `entries` are (tag, updated)."""
    body = "".join(
        f'<entry><id>tag:github.com,2008:Repository/1/{t}</id><updated>{d}</updated>'
        f'<link rel="alternate" type="text/html" href="{updates_host.FIRMWARE_RELEASES}{t}"/><title>{t}</title></entry>'
        for t, d in entries)
    return (200, ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">' + body + "</feed>").encode(), "")


def pypi(**versions):
    return [(f"https://pypi.org/pypi/{n}/json", j({"info": {"version": v}})) for n, v in versions.items()]


def dist(site, name, version):
    (site / f"{name}-{version}.dist-info").mkdir(parents=True)


def make_venv(root, packages):
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python").write_text("")
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n")
    site = root / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    for name, version in packages.items():
        dist(site, name, version)
    return root


@pytest.fixture
def host(tmp_path):
    tree = tmp_path / "tt-metal"
    (tree / "tt_metal").mkdir(parents=True)
    (tree / "tt_metal" / "sfpi-version").write_text("# sfpi release 7.89.0\n\nsfpi_repo='x'\nsfpi_version='7.89.0'\nsfpi_base=''\n")
    bundles = tmp_path / "models"
    b = bundles / "org" / "aion-bundle" / "venv" / "lib" / "python3.12" / "site-packages" / "ttnn" / "tt_metal"
    b.mkdir(parents=True)
    (b / "sfpi-version").write_text("sfpi_version='7.85.0'\nsfpi_build='x'\n")
    tt = make_venv(tmp_path / "ttvenv", {"tt_smi": "6.8.0", "tt_flash": "4.1.0"})
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for n in ("tt-smi", "tt-flash"):
        (bindir / n).write_text(f"#!{tt}/bin/python\nimport sys\n")
    ref = make_venv(tmp_path / "refvenv", {"torch": "2.14.1+cpu", "transformers": "5.19.0", "tokenizers": "0.23.2",
                                           "safetensors": "0.8.0", "huggingface_hub": "1.33.0"})
    sysfs = tmp_path / "sys"
    for i in range(2):
        d = sysfs / f"tenstorrent!{i}"
        d.mkdir(parents=True)
        (d / "tt_fw_bundle_ver").write_text("19.17.1.0\n")
    return HostEnv(tree=tree, bundle_root=bundles, reference_python=ref / "bin" / "python",
                   sysfs_glob=str(sysfs / "*" / "tt_fw_bundle_ver"), sysfs_timeout_s=0.3,
                   which_path=lambda n: str(bindir / n) if (bindir / n).exists() else None)


def runs(**over):
    t = {
        "dpkg-query -W -f=${Version} sfpi": (0, "7.71.0", ""),
        "apt-cache policy sfpi": (0, POLICY.format(pkg="sfpi", inst="7.71.0", cand="7.96.0"), ""),
        "modinfo -F version tenstorrent": (0, "2.11.0\n", ""),
        "apt-cache policy tenstorrent-dkms": (0, POLICY.format(pkg="tenstorrent-dkms", inst="2.11.0", cand="2.12.0"), ""),
    }
    t.update(over)
    return t


def routes(**over):
    r = {"feed": (updates_host.FIRMWARE_FEED, feed(("v19.16.1", "2026-11-02T16:36:31Z"), ("v19.17.1", "2026-10-29T10:00:00Z"),
                                                   ("v19.17.1-rc1", "2026-10-28T10:00:00Z"))),
         "pypi": pypi(**{"tt-flash": "4.2.0", "tt-smi": "6.8.0", "torch": "2.15.0", "transformers": "5.24.0",
                         "tokenizers": "0.23.3", "safetensors": "0.8.0", "huggingface-hub": "2.4.0"})}
    r.update(over)
    out = [r["feed"]] + list(r["pypi"])
    return out


def check(host, table=None, net=None, offline=False):
    rn = Runs(table or runs())
    nt = Net(net or routes())
    tools = Tools(nt, rn, offline=offline, environ={}, which=lambda n: True)
    return updates_host.check_host(tools, host), nt, rn


def row(rows, key):
    return next(r for r in rows if r.key == key)


# ---- SFPI ---------------------------------------------------------------------------------------------------------

def test_sfpi_behind_names_the_tree_pin_the_bundle_floor_and_the_apt_candidate(host):
    rows, _, _ = check(host)
    r = row(rows, "sfpi")
    assert (r.installed, r.upstream, r.status) == ("7.71.0", "7.96.0", BEHIND)
    assert "below the 7.89.0 that the host tt-metal tree pins" in r.why
    assert "below the 7.85.0 that the ttnn in the aion-bundle bundle declares" in r.why
    assert r.proposal and r.proposal[0].startswith("sudo apt install sfpi=7.96.0")


def test_sfpi_current_when_it_meets_the_candidate_the_pin_and_every_floor(host):
    t = runs(**{"dpkg-query -W -f=${Version} sfpi": (0, "7.96.0", "")})
    rows, _, _ = check(host, t)
    assert row(rows, "sfpi").status == CURRENT


def test_sfpi_without_an_apt_candidate_is_still_behind_the_tree_pin_and_proposes_the_pin(host):
    t = runs(**{"apt-cache policy sfpi": (0, "sfpi:\n  Installed: 7.71.0\n  Candidate: (none)\n", "")})
    r = row(check(host, t)[0], "sfpi")
    assert r.status == BEHIND and r.upstream == "-" and "sfpi=7.89.0" in r.proposal[0]


def test_sfpi_not_installed_is_unknown_with_the_reason(host):
    r = row(check(host, runs(**{"dpkg-query -W -f=${Version} sfpi": (1, "", "no packages found")}))[0], "sfpi")
    assert r.status == UNKNOWN and "not installed" in r.why


def test_the_highest_bundle_floor_wins_and_a_garbage_pin_file_is_ignored(host, tmp_path):
    other = host.bundle_root / "org" / "second" / "venv" / "lib" / "python3.12" / "site-packages" / "ttnn" / "tt_metal"
    other.mkdir(parents=True)
    (other / "sfpi-version").write_text("sfpi_version='7.91.0'\n")
    (host.tree / "tt_metal" / "sfpi-version").write_text("\x00\x01 not a pin file " * 50)
    r = row(check(host)[0], "sfpi")
    assert "7.91.0 that the ttnn in the second bundle" in r.why and "tt-metal tree pins" not in r.why


# ---- tt-kmd -------------------------------------------------------------------------------------------------------

def test_kmd_behind_carries_the_release_note_only_for_the_version_it_describes(host):
    r = row(check(host)[0], "tt-kmd")
    assert (r.installed, r.upstream, r.status) == ("2.11.0", "2.12.0", BEHIND)
    assert "pinned-memory hang fix" in r.why and "IOMMU leak fix" in r.why
    assert r.proposal[0].startswith("sudo apt install tenstorrent-dkms=2.12.0")
    t = runs(**{"apt-cache policy tenstorrent-dkms": (0, POLICY.format(pkg="tenstorrent-dkms", inst="2.11.0", cand="2.13.0"), "")})
    r2 = row(check(host, t)[0], "tt-kmd")
    assert r2.status == BEHIND and r2.why == ""


def test_kmd_current_when_the_candidate_is_not_newer(host):
    t = runs(**{"modinfo -F version tenstorrent": (0, "2.12.0\n", "")})
    assert row(check(host, t)[0], "tt-kmd").status == CURRENT


def test_kmd_without_a_module_or_a_candidate_is_unknown(host):
    assert row(check(host, runs(**{"modinfo -F version tenstorrent": (1, "", "not found")}))[0], "tt-kmd").status == UNKNOWN
    t = runs(**{"apt-cache policy tenstorrent-dkms": (0, "tenstorrent-dkms:\n  Candidate: (none)\n", "")})
    assert row(check(host, t)[0], "tt-kmd").status == UNKNOWN


# ---- tt-flash and tt-smi: from the venv, no process ---------------------------------------------------------------

def test_tt_flash_and_tt_smi_versions_come_from_the_venv_and_no_process_runs_them(host):
    rows, nt, rn = check(host)
    assert (row(rows, "tt-flash").installed, row(rows, "tt-flash").status) == ("4.1.0", BEHIND)
    assert (row(rows, "tt-smi").installed, row(rows, "tt-smi").status) == ("6.8.0", CURRENT)
    assert not [a for a in rn.seen if a[0] in ("tt-smi", "tt-flash")]
    assert all(a[0] in ("dpkg-query", "apt-cache", "modinfo") for a in rn.seen)


def test_a_tool_that_is_not_installed_is_unknown(host):
    host.which_path = lambda n: None
    r = row(check(host)[0], "tt-flash")
    assert r.status == UNKNOWN and "not installed" in r.why


# ---- firmware ------------------------------------------------------------------------------------------------------

def test_firmware_current_when_it_equals_the_newest_release_whatever_the_feed_order(host):
    r = row(check(host)[0], "firmware")
    assert (r.installed, r.upstream, r.upstream_date, r.status) == ("19.17.1.0", "v19.17.1", "2026-10-29", CURRENT)


def test_a_release_candidate_tag_is_never_the_newest(host):
    net = routes(feed=(updates_host.FIRMWARE_FEED, feed(("v19.18.0-rc1", "2026-11-20T00:00:00Z"), ("v19.17.1", "2026-10-29T00:00:00Z"))))
    assert row(check(host, net=net)[0], "firmware").upstream == "v19.17.1"


def test_firmware_behind_proposes_text_that_points_at_the_notes_and_flashes_nothing(host):
    net = routes(feed=(updates_host.FIRMWARE_FEED, feed(("v19.18.0", "2026-11-20T00:00:00Z"))))
    r = row(check(host, net=net)[0], "firmware")
    assert r.status == BEHIND
    assert "migration guide" in r.proposal[0] and r.proposal[1].startswith("tt-flash flash <")


def test_chips_that_disagree_report_the_lowest_and_say_so(host, tmp_path):
    (tmp_path / "sys" / "tenstorrent!1" / "tt_fw_bundle_ver").write_text("19.16.0.0\n")
    r = row(check(host)[0], "firmware")
    assert r.installed == "19.16.0.0 (and 1 other)" and r.status == BEHIND


def test_a_driver_that_does_not_answer_gives_unknown_and_the_command_does_not_wait(host, monkeypatch):
    gate = threading.Event()
    real = updates_host.safeio.read_bytes

    def hang(path, **kw):
        if str(path).endswith("tt_fw_bundle_ver"):
            gate.wait(10)
        return real(path, **kw)
    monkeypatch.setattr(updates_host.safeio, "read_bytes", hang)
    import time
    t0 = time.monotonic()
    try:
        r = row(check(host)[0], "firmware")
    finally:
        gate.set()
    assert time.monotonic() - t0 < 3
    assert r.status == UNKNOWN and "not read: the driver did not answer" in r.why


def test_no_firmware_file_is_unknown_not_read(host):
    host.sysfs_glob = host.sysfs_glob.replace("tt_fw_bundle_ver", "absent")
    r = row(check(host)[0], "firmware")
    assert r.status == UNKNOWN and r.why.startswith("not read:")


def test_an_unreachable_feed_leaves_the_installed_version_and_an_unknown_status(host):
    net = routes(feed=(updates_host.FIRMWARE_FEED, (503, b"", "HTTP 503")))
    r = row(check(host, net=net)[0], "firmware")
    assert r.installed == "19.17.1.0" and r.status == UNKNOWN and "HTTP 503" in r.why


# ---- the reference venv --------------------------------------------------------------------------------------------

def test_reference_packages_minor_behind_and_a_major_jump_is_labelled(host):
    rows, _, _ = check(host)
    t = row(rows, "ref-transformers")
    assert t.status == BEHIND and "reference numerics" in t.why and "COPY" in t.proposal[0]
    assert "reference numerics" in row(rows, "ref-torch").why and row(rows, "ref-torch").installed == "2.14.1+cpu"
    h = row(rows, "ref-huggingface-hub")
    assert h.status == BEHIND and h.why.startswith("major: do not auto-upgrade") and h.proposal[0].startswith("#")
    assert row(rows, "ref-safetensors").status == CURRENT
    assert "numerics" not in row(rows, "ref-tokenizers").why


def test_without_a_reference_venv_every_reference_row_is_unknown(host):
    host.reference_python = None
    rows = [r for r in check(host)[0] if r.key.startswith("ref-")]
    assert len(rows) == 5 and all(r.status == UNKNOWN for r in rows)


# ---- the read-only boundary ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["dpkg-query", "-W", "-f=${Version}", "bash"], ["dpkg-query", "-L", "sfpi"], ["apt-cache", "policy", "openssh-server"],
    ["apt-cache", "install", "sfpi"], ["modinfo", "-F", "filename", "tenstorrent"], ["modinfo", "tenstorrent"],
    ["tt-smi", "-s"], ["tt-flash", "verify"], ["apt", "install", "sfpi"], ["sudo", "apt-cache", "policy", "sfpi"]])
def test_only_the_fixed_host_reads_are_allowed(argv):
    with pytest.raises(NotReadOnly):
        Tools(lambda *a: (200, b"", ""), lambda *a, **k: (0, "", ""), which=lambda n: True).cmd(argv)


def test_the_firmware_feed_is_the_only_new_host_and_other_hosts_stay_refused():
    t = Tools(lambda *a: (200, b"", ""), lambda *a, **k: (0, "", ""))
    t.text(updates_host.FIRMWARE_FEED)
    for bad in ("https://github.com.evil.example/x", "http://github.com/x", "https://gist.github.com/x"):
        with pytest.raises(NotReadOnly):
            t.text(bad)


def test_hostile_answers_are_cut_and_cleaned_before_they_reach_a_row(host):
    net = routes(pypi=pypi(**{"tt-flash": "9.9.9" + "\x1b[31m" * 50, "tt-smi": "x", "torch": "2.15.0", "transformers": "5.24.0",
                              "tokenizers": "0.23.3", "safetensors": "0.8.0", "huggingface-hub": "1.40.0"}))
    rows = check(host, net=net)[0]
    f = row(rows, "tt-flash")
    assert "\x1b" not in f.upstream and len(f.upstream) <= 40
    assert row(rows, "tt-smi").status == UNKNOWN and row(rows, "ref-huggingface-hub").status == BEHIND


# ---- run_all, render, the web view's bound ------------------------------------------------------------------------

def test_run_all_adds_the_host_rows_only_when_a_host_is_given(host, tmp_path):
    def go(h):
        tools = Tools(Net(routes()), Runs(runs()), environ={}, which=lambda n: True)
        return updates.run_all(tools, updates.load_config(None), installed_path=tmp_path / "none.json", tt_metal_dir=tmp_path,
                               skills=None, ver=verifications.Verifications(), host=h)
    assert not [r for r in go(None) if r.key in updates.HOST_KEYS]
    rows = go(host)
    assert sorted({r.key for r in rows} & set(updates.HOST_KEYS)) == sorted(updates.HOST_KEYS)
    assert len(rows) <= 40       # the web view shows at most 40 rows (updates_view.MAX_ROWS)


def test_a_behind_host_row_names_its_rules_and_the_report_invents_no_pairing(host):
    rows = updates.with_rules(check(host)[0], verifications.Verifications(), {})
    by = {r.key: r for r in rows}
    assert any(m.startswith("R4: moves with tt-metal, tt-model-manager") for m in by["sfpi"].moves_with)
    r5 = [m for m in by["tt-kmd"].moves_with if m.startswith("R5:")][0]
    assert "No vendor table pairs these" in r5 and "invents no pairing" in r5 and "tt-kmd 2.8.0" in r5
    assert by["sfpi"].ready == updates.HOST_READY and "1x4 mesh" in by["sfpi"].ready
    text = "\n".join(updates.render(rows, 0))
    assert "propose (text only, orchard runs none of it): sudo apt install sfpi=7.96.0" in text
    assert "tt-smi" not in "\n".join(a for r in rows for a in r.proposal if "tt-smi " in a and "install" not in a)


def test_main_builds_the_host_from_the_arguments_and_prints_the_rows(host, tmp_path, monkeypatch, capsys):
    import io
    monkeypatch.setattr(updates_host, "HostEnv", lambda **kw: host)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    seen = {}
    real = updates.run_all

    def spy(*a, **kw):
        seen["host"] = kw.get("host")
        return real(*a, **kw)
    monkeypatch.setattr(updates, "run_all", spy)

    class A:
        json = True
        offline = False
        refresh = True
        print_timer = False
        fail_when_unknown = False
        tt_metal_dir = str(tmp_path)
        updates_config = None
        skills_dirs = None
        reference_python = str(host.reference_python)
        tt_model_root = str(host.bundle_root)
    out = io.StringIO()
    tools = Tools(Net(routes()), Runs(runs()), environ={}, which=lambda n: True)
    assert updates.main(A(), out=out, tools=tools) == 0
    doc = json.loads(out.getvalue())
    assert seen["host"] is host
    assert {r["key"] for r in doc["rows"]} >= set(updates.HOST_KEYS)


# ---- guards that the first tests did not pin ----------------------------------------------------------------------

def test_versions_of_different_length_compare_with_zero_padding():
    assert updates_host._cmp((19, 17, 1, 0), (19, 17, 1)) == 0
    assert updates_host._cmp((2, 12), (2, 12, 0)) == 0
    assert updates_host._cmp((19, 17, 1, 0), (19, 17, 2)) == -1


def test_a_symlinked_dist_info_folder_is_not_followed(host, tmp_path):
    site = next((tmp_path / "ttvenv").glob("lib/python3.*/site-packages"))
    elsewhere = tmp_path / "elsewhere" / "tt_flash-9.9.9.dist-info"
    elsewhere.mkdir(parents=True)
    for old in site.glob("tt_flash-*"):
        old.rmdir()
    (site / "tt_flash-9.9.9.dist-info").symlink_to(elsewhere)
    r = row(check(host)[0], "tt-flash")
    assert r.status == UNKNOWN and "not installed" in r.why


def test_a_script_whose_interpreter_is_not_in_a_venv_gives_no_version(host, tmp_path):
    # the interpreter's folder holds a tt_flash dist-info but no pyvenv.cfg: a system prefix, not a venv this check reads
    plain = tmp_path / "plain"
    (plain / "bin").mkdir(parents=True)
    (plain / "lib" / "python3.12" / "site-packages" / "tt_flash-9.9.9.dist-info").mkdir(parents=True)
    (tmp_path / "bin" / "tt-flash").write_text(f"#!{plain}/bin/python3\nimport sys\n")
    r = row(check(host)[0], "tt-flash")
    assert r.status == UNKNOWN and r.installed == "-"


def test_at_most_forty_bundles_are_read_for_a_floor(host):
    for i in range(41):
        d = host.bundle_root / "org" / f"zz{i:02d}" / "venv" / "lib" / "python3.12" / "site-packages" / "ttnn" / "tt_metal"
        d.mkdir(parents=True)
        (d / "sfpi-version").write_text("sfpi_version='7.99.0'\n" if i == 40 else "sfpi_version='7.80.0'\n")
    floors = updates_host.bundle_sfpi_floors(host.bundle_root)
    assert len(floors) == 40 and all(v != "7.99.0" for _, v in floors)


def test_a_firmware_file_without_a_version_is_not_a_reading(host, tmp_path):
    for i in range(2):
        (tmp_path / "sys" / f"tenstorrent!{i}" / "tt_fw_bundle_ver").write_text("not a version\n")
    r = row(check(host)[0], "firmware")
    assert r.status == UNKNOWN and r.installed == "-" and r.why.startswith("not read:")


def test_a_console_script_that_is_a_link_on_path_is_resolved_once(host, tmp_path):
    real = tmp_path / "bin" / "tt-smi"
    link_dir = tmp_path / "linkbin"
    link_dir.mkdir()
    (link_dir / "tt-smi").symlink_to(real)
    host.which_path = lambda n: str(link_dir / n) if n == "tt-smi" else str(tmp_path / "bin" / n)
    r = row(check(host)[0], "tt-smi")
    assert (r.installed, r.status) == ("6.8.0", CURRENT)


def test_a_pypi_answer_over_the_old_two_megabyte_limit_is_read_but_not_without_end(host):
    big = json.dumps({"info": {"version": "0.23.3"}, "pad": "x" * (3 * 1024 * 1024)}).encode()
    net = routes(pypi=[("https://pypi.org/pypi/tokenizers/json", (200, big, ""))] + pypi(**{
        "tt-flash": "4.1.0", "tt-smi": "6.8.0", "torch": "2.14.1", "transformers": "5.19.0", "safetensors": "0.8.0",
        "huggingface-hub": "1.33.0"}))
    assert row(check(host, net=net)[0], "ref-tokenizers").upstream == "0.23.3"
    huge = json.dumps({"info": {"version": "9.9.9"}, "pad": "x" * (9 * 1024 * 1024)}).encode()
    net2 = routes(pypi=[("https://pypi.org/pypi/tokenizers/json", (200, huge, ""))] + pypi(**{"torch": "2.14.1"}))
    r = row(check(host, net=net2)[0], "ref-tokenizers")
    assert r.status == UNKNOWN and "larger than the limit" in r.why
