"""safeio.replace_beneath: a rename inside a pinned run directory that never follows a link on either side (review of PR 489).

`os.replace(path, path2)` follows a link standing in a parent directory, so a prepare step that swaps `evidence/` for a link after the
directory check lets the move rename a file outside the run. `replace_beneath` opens both parent directories relative to the pinned
descriptor with O_NOFOLLOW and renames with `src_dir_fd` and `dst_dir_fd`."""
import os
import stat

import pytest

from orchard import safeio


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    (root / "stage" / "evidence").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    return root


def test_a_file_is_renamed_inside_the_run(run):
    (run / "stage" / "evidence" / "a.json").write_text("x")
    with safeio.pinned_root(run) as pin:
        assert safeio.replace_beneath(pin.fd, ("stage", "evidence", "a.json"), ("stage", "evidence", "a.json.rejected")) is True
    assert not (run / "stage" / "evidence" / "a.json").exists()
    assert (run / "stage" / "evidence" / "a.json.rejected").read_text() == "x"


def test_a_missing_source_moves_nothing(run):
    with safeio.pinned_root(run) as pin:
        assert safeio.replace_beneath(pin.fd, ("stage", "evidence", "nope.json"), ("stage", "evidence", "nope.json.rejected")) is False
        assert safeio.replace_beneath(pin.fd, ("stage", "missing-dir", "nope.json"), ("stage", "missing-dir", "nope.json.rejected")) is False


def test_a_directory_at_the_source_is_not_moved(run):
    (run / "stage" / "evidence" / "d").mkdir()
    with safeio.pinned_root(run) as pin:
        assert safeio.replace_beneath(pin.fd, ("stage", "evidence", "d"), ("stage", "evidence", "d.rejected")) is False
    assert (run / "stage" / "evidence" / "d").is_dir()


def test_a_link_at_the_last_name_is_moved_itself_and_its_target_stays(run, tmp_path):
    victim = tmp_path / "outside" / "victim.json"
    victim.write_text("keep")
    os.symlink(victim, run / "stage" / "evidence" / "a.json")
    with safeio.pinned_root(run) as pin:
        assert safeio.replace_beneath(pin.fd, ("stage", "evidence", "a.json"), ("stage", "evidence", "a.json.rejected")) is True
    assert victim.read_text() == "keep"
    assert stat.S_ISLNK(os.lstat(run / "stage" / "evidence" / "a.json.rejected").st_mode)


def test_a_link_standing_in_the_parent_directory_is_refused_and_nothing_outside_moves(run, tmp_path):
    (tmp_path / "outside" / "port-check.json").write_text("outside")
    os.rmdir(run / "stage" / "evidence")
    os.symlink(tmp_path / "outside", run / "stage" / "evidence")
    with safeio.pinned_root(run) as pin:
        with pytest.raises(safeio.SafeReadError):
            safeio.replace_beneath(pin.fd, ("stage", "evidence", "port-check.json"), ("stage", "evidence", "port-check.json.rejected"))
    assert (tmp_path / "outside" / "port-check.json").read_text() == "outside"
    assert not (tmp_path / "outside" / "port-check.json.rejected").exists()


def test_a_link_standing_in_the_destination_directory_is_refused(run, tmp_path):
    (run / "stage" / "evidence" / "a.json").write_text("x")
    os.symlink(tmp_path / "outside", run / "stage" / "elsewhere")
    with safeio.pinned_root(run) as pin:
        with pytest.raises(safeio.SafeReadError):
            safeio.replace_beneath(pin.fd, ("stage", "evidence", "a.json"), ("stage", "elsewhere", "a.json"))
    assert (run / "stage" / "evidence" / "a.json").read_text() == "x"
    assert not (tmp_path / "outside" / "a.json").exists()


@pytest.mark.parametrize("src,dst", [((), ("a",)), (("a",), ()), (("..", "x"), ("a",)), (("a",), ("..", "x")), (("a\0b",), ("c",))])
def test_odd_parts_are_refused(run, src, dst):
    with safeio.pinned_root(run) as pin:
        with pytest.raises(safeio.SafeReadError):
            safeio.replace_beneath(pin.fd, src, dst)
