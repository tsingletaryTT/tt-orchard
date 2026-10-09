# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard lab setup` (orchard/lab_setup.py): getting a lab box ready, from the brain.

Every machine call goes through a fake that answers by pattern and logs what was asked. The tests that
matter most prove what must never happen: setup running sudo, taking, resetting or releasing a lease,
copying a tensor cache to the lab, or writing on either box without being asked.
"""
from pathlib import Path

import pytest

from orchard import lab_setup as ls
from orchard.bringup_config import Lab
from orchard.setup_machine import FAIL, OK, TODO, WARN


class Fake:
    """Answers (argv, timeout) by the first matching substring rule; logs every call."""

    def __init__(self, rules=None, default=(0, "", "")):
        self.rules = list(rules or [])
        self.default = default
        self.calls: list[list[str]] = []

    def __call__(self, argv, timeout=60):
        self.calls.append(list(argv))
        text = " ".join(argv)
        for needle, answer in self.rules:
            if needle in text:
                return answer
        return self.default


GOZER = """grain: chip   (2 boards, 2 chips)
board 000004033192101E  (p150a)
  chip 0  0000:01:00.0  FREE
board 0000040331921021  (p150a)
  chip 1  0000:03:00.0  FREE
"""


@pytest.fixture
def box(tmp_path):
    root = tmp_path / "srv-orchard"
    for sub in ("runs", "cache", "hf/hub", "tt-model/models/episod/b1", "venvs/reference/bin"):
        (root / sub).mkdir(parents=True)
    (root / "tt-model/models/episod/b1/tt_kernel_manifest.json").write_text("{}")
    (root / "venvs/reference/bin/python").write_text("")
    lab = Lab(host="node4", root=root, path=["~/.local/bin"], test_python=str(root / "venvs/reference/bin/python"))
    return root, lab


def plan(box, run, **kw):
    root, lab = box
    kw.setdefault("local_sfpi", "7.78.0")
    return ls.plan(lab, run=run, reference_python=root / "venvs/reference/bin/python", local_fw="19.15.0.0", **kw)


def by_name(steps):
    return {s.name: s for s in steps}


def test_a_ready_lab_is_all_ok(box):
    run = Fake([("gozer status", (0, GOZER, "")), ("tt_fw_bundle_ver", (0, "19.15.0.0\n", "")),
                ("dpkg-query", (0, "7.78.0", "")), ("test -f", (0, "", ""))])
    steps = by_name(plan(box, run))
    assert {n: s.status for n, s in steps.items()} == {n: OK for n in steps}, \
        {n: (s.status, s.detail) for n, s in steps.items() if s.status != OK}


def test_ssh_that_does_not_answer_stops_everything_else(box):
    run = Fake([("true", (255, "", "ssh: connect to host node4 port 22: Connection refused"))])
    steps = plan(box, run)
    assert steps[0].name == "ssh" and steps[0].status == FAIL and "refused" in steps[0].detail
    assert len(steps) == 1


def test_a_missing_lab_root_says_how_to_make_it_and_never_runs_sudo(box):
    root, _ = box
    run = Fake([("test -d", (1, "", ""))])
    s = by_name(plan(box, run))["lab root"]
    assert s.status == FAIL and f"sudo mkdir -p {root}" in s.detail and not s.actions
    assert not [c for c in run.calls if "sudo" in " ".join(c)]


def test_gozer_missing_on_the_lab_is_installed_with_its_own_installer(box):
    run = Fake([("command -v gozer", (1, "", "")), ("gozer status", (0, GOZER, ""))])
    s = by_name(plan(box, run))["gozer (lab)"]
    assert s.status == TODO
    remote = s.actions[0].args[0][-1]
    assert "git clone" in remote and "tt-gozer/install.sh" in remote


def test_bundles_missing_on_the_lab_are_copied_without_their_caches(box):
    root, _ = box
    run = Fake([("test -f", (1, "", "")), ("gozer status", (0, GOZER, ""))])
    s = by_name(plan(box, run))["bundles (lab)"]
    assert s.status == TODO and "episod/b1" in s.detail
    argv = s.actions[0].args[0]
    assert argv[0] == "rsync" and "--delete" not in argv
    for skip in (".tt_cache", ".hf", ".cache"):
        assert skip in argv[argv.index("--exclude", argv.index(skip) - 1) + 1]
    assert argv[-2:] == [f"{root}/tt-model/models/episod/b1/", f"node4:{root}/tt-model/models/episod/b1/"]


def test_a_model_is_linked_into_the_lab_hf_cache_when_on_the_same_disk(box, tmp_path):
    root, _ = box
    src = tmp_path / "home-hf" / "hub" / "models--Org--M"
    src.mkdir(parents=True)
    run = Fake([("gozer status", (0, GOZER, "")), ("test -f", (0, "", ""))])
    s = by_name(plan(box, run, models=["Org/M"], model_sources=[tmp_path / "home-hf"]))["models"]
    assert s.status == TODO
    assert s.actions[0].args[0] == ["cp", "-al", str(src), f"{root}/hf/hub/models--Org--M"]


def test_a_firmware_difference_is_a_warning(box):
    run = Fake([("tt_fw_bundle_ver", (0, "19.12.0.0\n", "")), ("gozer status", (0, GOZER, "")),
                ("test -f", (0, "", ""))])
    s = by_name(plan(box, run))["firmware"]
    assert s.status == WARN and "19.12.0.0" in s.detail and "19.15.0.0" in s.detail


def test_busy_lab_chips_are_a_warning_not_something_setup_clears(box):
    busy = GOZER.replace("chip 1  0000:03:00.0  FREE", "chip 1  0000:03:00.0  HELD   someone pid 7")
    run = Fake([("gozer status", (0, busy, "")), ("test -f", (0, "", ""))])
    s = by_name(plan(box, run))["lab chips"]
    assert s.status == WARN and "1 of 2" in s.detail


def test_setup_never_takes_resets_or_releases_a_lease(box):
    run = Fake([("command -v gozer", (1, "", "")), ("test -f", (1, "", "")), ("gozer status", (0, GOZER, ""))])
    steps = plan(box, run)
    planned = [" ".join(a.args[0]) if a.kind == "run" else "" for s in steps for a in s.actions]
    asked = [" ".join(c) for c in run.calls]
    for text in planned + asked:
        for word in (" acquire", " reset", " release", " reconcile", "sudo "):
            assert word not in text, text


def _bad_venv(bundle: Path, orig: Path) -> None:
    """A bundle whose venv still names the directory tt-model installed it in (test_bundle_relink.py)."""
    real = bundle / ".python" / "cpython-3.12.14-linux-x86_64-gnu" / "bin"
    real.mkdir(parents=True)
    (real / "python3.12").write_text("")
    (bundle / ".python" / "cpython-3.12-linux-x86_64-gnu").symlink_to(orig / ".python" / "cpython-3.12.14-linux-x86_64-gnu")
    (bundle / "venv" / "bin").mkdir(parents=True)
    (bundle / "venv" / "bin" / "python").symlink_to(orig / ".python" / "cpython-3.12-linux-x86_64-gnu" / "bin" / "python3.12")
    (bundle / "venv" / "pyvenv.cfg").write_text(f"home = {orig}/.python/cpython-3.12-linux-x86_64-gnu/bin\n")


def test_a_bundle_interpreter_outside_the_bundle_is_relinked_here_first(box, tmp_path):
    import subprocess
    root, _ = box
    b = root / "tt-model/models/episod/b1"
    _bad_venv(b, tmp_path / "gone")
    run = Fake([("gozer status", (0, GOZER, "")), ("test -f", (0, "", ""))])
    s = by_name(plan(box, run))["bundle interpreters"]
    assert s.status == TODO and "1 here" in s.detail
    argv = s.actions[0].args[0]
    assert "-c" in argv and argv[-1] == str(b)
    assert subprocess.run(argv).returncode == 0                  # the planned command fixes it
    s = by_name(plan(box, Fake([("gozer status", (0, GOZER, "")), ("test -f", (0, "", ""))])))["bundle interpreters"]
    assert s.status == OK


def test_a_bundle_interpreter_broken_on_the_lab_is_relinked_there(box):
    root, _ = box
    b = root / "tt-model/models/episod/b1"
    run = Fake([("--check", (1, f"{b}: venv/bin/python links outside the bundle, to /home/x\n", "")),
                ("gozer status", (0, GOZER, "")), ("test -f", (0, "", ""))])
    s = by_name(plan(box, run))["bundle interpreters"]
    assert s.status == TODO and "1 on node4" in s.detail
    remote_cmd = s.actions[0].args[0]
    assert remote_cmd[-2] == "node4" and remote_cmd[-1].endswith(f"' {b}")     # the fix, not the --check


def test_bundles_still_to_be_copied_are_not_checked_on_the_lab(box):
    run = Fake([("test -f", (1, "", "")), ("gozer status", (0, GOZER, ""))])
    s = by_name(plan(box, run))["bundle interpreters"]
    assert s.status == OK and not [c for c in run.calls if "--check" in " ".join(c)]


def test_the_relink_step_prints_the_script_by_name_not_its_source(box, tmp_path):
    root, _ = box
    b = root / "tt-model/models/episod/b1"
    _bad_venv(b, tmp_path / "gone")
    run = Fake([("--check", (1, f"{b}: venv/bin/python links outside the bundle\n", "")),
                ("gozer status", (0, GOZER, "")), ("test -f", (0, "", ""))])
    s = by_name(plan(box, run))["bundle interpreters"]
    shown = [a.describe() for a in s.actions]
    assert shown == [f"python3 -c <orchard/bundle_relink.py> {b}", f"ssh node4 python3 -c <orchard/bundle_relink.py> {b}"]


def test_an_older_sfpi_on_the_lab_fails_and_says_how_to_install_this_boxs():
    """On the first lab run the 2-chip bundle opened its mesh on node4 and then failed to compile kernels:
    "'vLut8si' is not a member of 'sfpi'". The TT runtime compiles kernels at every boot with the system
    SFPI toolchain; node4 had 7.61.0 and the bundles were built against 7.78.0."""
    import tempfile
    root = Path(tempfile.mkdtemp())
    for sub in ("runs", "cache", "hf/hub", "tt-model/models", "venvs/reference/bin"):
        (root / sub).mkdir(parents=True)
    lab = Lab(host="node4", root=root)
    run = Fake([("dpkg-query", (0, "7.61.0", "")), ("gozer status", (0, GOZER, ""))])
    s = by_name(plan((root, lab), run))["sfpi"]
    assert s.status == FAIL and "7.61.0" in s.detail and "7.78.0" in s.detail
    assert "sudo dpkg -i" in s.detail and not s.actions                  # setup never runs sudo itself
    run = Fake([("dpkg-query", (1, "", "no packages found matching sfpi")), ("gozer status", (0, GOZER, ""))])
    assert by_name(plan((root, lab), run))["sfpi"].status == FAIL
    run = Fake([("dpkg-query", (0, "7.78.0", "")), ("gozer status", (0, GOZER, ""))])
    assert by_name(plan((root, lab), run))["sfpi"].status == OK
