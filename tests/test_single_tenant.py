"""SingleTenantAdapter: one tenant, no lease tool, whole boards, a fake /proc."""
import os

import pytest

from fakes import FakeRun
from orchard.adapters import LeaseLost, Refused, ResetFailed
from orchard.adapters.single_tenant import SingleTenantAdapter, device_holders

# Two p300c boards of two chips, as on this Quietbox 2.
CHIPS = [("0000:01:00.0", 0), ("0000:02:00.0", 1), ("0000:03:00.0", 2), ("0000:04:00.0", 3)]
BOARD0 = ("0000:01:00.0", "0000:02:00.0")
BOARD1 = ("0000:03:00.0", "0000:04:00.0")


def fake_proc(tmp_path, holders):
    """holders: {pid: [link targets]}. Returns the proc root."""
    root = tmp_path / "proc"
    root.mkdir(exist_ok=True)
    for pid, targets in holders.items():
        fd = root / str(pid) / "fd"
        fd.mkdir(parents=True, exist_ok=True)
        for n, t in enumerate(targets):
            os.symlink(t, fd / str(n + 3))
    return str(root)


def adapter(tmp_path, holders=None, **kw):
    # No lease tool on this "machine": the tests never look at the real PATH or /tmp/tt-gozer.
    kw.setdefault("which", lambda name: None)
    kw.setdefault("gozer_state_dirs", ())
    # No real command either: the default FakeRun has no script, so any call fails the test.
    # A test that needs a reset call passes its own scripted FakeRun. Without this default, a
    # regression in a guard would run the real reset command on a real board.
    kw.setdefault("run", FakeRun())
    return SingleTenantAdapter(CHIPS, proc_root=fake_proc(tmp_path, holders or {}), **kw)


def test_device_holders_reads_only_tenstorrent_nodes(tmp_path):
    root = fake_proc(tmp_path, {10: ["/dev/tenstorrent/0", "/dev/null"],
                                11: ["/dev/tenstorrent_x", "/dev/tenstorrent/1"]})
    assert device_holders(root) == {0: [10], 1: [11]}


def test_a_restarted_supervisor_can_build_it_while_its_coder_holds_a_board(tmp_path):
    # After a crash the old coder still holds board 0. Building the adapter must still work.
    a = adapter(tmp_path, {321: ["/dev/tenstorrent/0"]})
    assert {c.bdf: c.state for c in a.status()}["0000:01:00.0"] == "BUSY-UNTRACKED"


def test_acquire_refuses_a_board_whose_device_is_open(tmp_path):
    a = adapter(tmp_path, {321: ["/dev/tenstorrent/0"]})
    with pytest.raises(Refused, match="pid 321 holds /dev/tenstorrent/0"):
        a.acquire(2, "w", "r", exact="0000:01:00.0")
    assert a.acquire(2, "w", "r").chips == BOARD1       # the other board is still free


def test_one_chip_grants_the_whole_board(tmp_path):
    # UMD expands TT_VISIBLE_DEVICES to the whole board, as gozer notes, so leases are per board.
    lease = adapter(tmp_path).acquire(1, "orchard:t", "r")
    assert lease.chips == BOARD0 and lease.dev_indices == (0, 1)
    assert lease.env == {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}
    assert lease.units == ("local0",)


def test_two_leases_never_share_a_board(tmp_path):
    a = adapter(tmp_path)
    first = a.acquire(1, "w", "r")
    second = a.acquire(1, "w", "r")
    assert first.chips == BOARD0 and second.chips == BOARD1
    with pytest.raises(Refused, match="0 free"):
        a.acquire(1, "w", "r")


def test_four_chips_take_both_boards(tmp_path):
    lease = adapter(tmp_path).acquire(4, "w", "r")
    assert lease.chips == BOARD0 + BOARD1 and lease.units == ("local0", "local1")


def test_exact_picks_the_board_of_the_named_chip(tmp_path):
    assert adapter(tmp_path).acquire(1, "w", "r", exact="0000:04:00.0").chips == BOARD1


def test_a_chip_count_that_splits_a_board_is_a_config_error(tmp_path):
    with pytest.raises(ValueError, match="whole boards"):
        SingleTenantAdapter(CHIPS[:3], proc_root=fake_proc(tmp_path, {}), run=FakeRun())


@pytest.mark.parametrize("which,dirs,needle", [
    (lambda name: "/home/u/.local/bin/gozer", (), "gozer is on PATH"),
    (lambda name: None, ("STATE",), "gozer state directory"),
])
def test_acquire_refuses_when_a_lease_tool_is_present(tmp_path, which, dirs, needle):
    state = tmp_path / "tt-gozer"
    state.mkdir()
    dirs = tuple(str(state) if d == "STATE" else d for d in dirs)
    a = adapter(tmp_path, which=which, gozer_state_dirs=dirs)
    with pytest.raises(Refused, match=needle) as exc:
        a.acquire(2, "w", "r")
    assert exc.value.permanent


def test_there_is_no_queue(tmp_path):
    with pytest.raises(Refused, match="no queue"):
        adapter(tmp_path).claim("t", 1, "w", "r")


def test_release_does_not_touch_the_hardware(tmp_path):
    run = FakeRun()          # any call fails the test
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    a.release(a.acquire(2, "w", "r"))
    assert run.calls == []


def test_reset_without_a_configured_command_is_a_permanent_refusal(tmp_path):
    a = adapter(tmp_path)
    with pytest.raises(Refused, match="no reset command") as exc:
        a.reset(a.acquire(2, "w", "r"))
    assert exc.value.permanent


def test_reset_runs_the_configured_command_with_the_lease_bdfs(tmp_path):
    run = FakeRun({"tt-smi": [(0, "", "")]}, key=lambda argv: argv[0])
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    a.reset(a.acquire(2, "w", "r"))
    assert run.argvs() == [["tt-smi", "-r", "0000:01:00.0,0000:02:00.0"]]
    assert run.calls[0]["kill_on_timeout"] is False


def test_reset_refuses_while_a_device_is_open(tmp_path):
    run = FakeRun({"tt-smi": [(0, "", "")]}, key=lambda argv: argv[0])
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    lease = a.acquire(2, "w", "r")
    fake_proc(tmp_path, {555: ["/dev/tenstorrent/1"]})
    with pytest.raises(Refused, match="still open") as exc:
        a.reset(lease)
    assert run.calls == [] and not exc.value.permanent


def test_failed_or_hung_reset_raises(tmp_path):
    run = FakeRun({"tt-smi": [(1, "", "boom"), ("timeout", "", "")]}, key=lambda argv: argv[0])
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    lease = a.acquire(2, "w", "r")
    with pytest.raises(ResetFailed, match="boom"):
        a.reset(lease)
    with pytest.raises(ResetFailed) as exc:
        a.reset(lease)
    assert exc.value.left_running
    with pytest.raises(Refused, match="still running"):
        a.reset(lease)


def test_reset_of_an_unknown_lease_is_lost(tmp_path):
    run = FakeRun()          # any call fails the test
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    lease = a.acquire(2, "w", "r")
    a.release(lease)
    with pytest.raises(LeaseLost):
        a.reset(lease)
    assert run.calls == []


def test_status_reports_held_claimed_free_and_untracked(tmp_path):
    a = adapter(tmp_path)
    a.acquire(1, "w", "r")
    fake_proc(tmp_path, {600: ["/dev/tenstorrent/0"], 601: ["/dev/tenstorrent/2"]})
    states = {c.bdf: c.state for c in a.status()}
    assert states == {"0000:01:00.0": "HELD", "0000:02:00.0": "CLAIMED",
                      "0000:03:00.0": "BUSY-UNTRACKED", "0000:04:00.0": "FREE"}
    assert {c.board for c in a.status()} == {"local0", "local1"}
