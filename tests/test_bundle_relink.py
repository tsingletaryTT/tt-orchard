# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/bundle_relink.py: a copied tt-model bundle's interpreter must live inside the bundle.

tt-model installs a bundle with a uv venv whose `bin/python`, its `.python/cpython-3.12-*` alias and the
`home` line of `pyvenv.cfg` name the install directory by absolute path. A copy (to the lab root, or to
another box) still points at the original, which on another box does not exist: the stage 2 test on the
first lab run died with `venv/bin/python: No such file or directory`.
"""
import os
import subprocess
import sys
from pathlib import Path

from orchard import bundle_relink as br

REAL = "cpython-3.12.14-linux-x86_64-gnu"
ALIAS = "cpython-3.12-linux-x86_64-gnu"


def installed_bundle(tmp_path: Path) -> tuple[Path, Path]:
    """A bundle copied from `orig` to `copy`; `orig` is then removed, as on a box that never had it."""
    orig = tmp_path / "home" / "u" / ".cache" / "tt-model" / "models" / "org" / "b"
    copy = tmp_path / "srv" / "orchard" / "tt-model" / "models" / "org" / "b"
    py = copy / ".python" / REAL / "bin"
    py.mkdir(parents=True)
    (py / "python3.12").write_text("#!/bin/sh\necho real\n")
    (py / "python3.12").chmod(0o755)
    (py / "python3").symlink_to("python3.12")
    (copy / ".python" / ALIAS).symlink_to(orig / ".python" / REAL)
    vbin = copy / "venv" / "bin"
    vbin.mkdir(parents=True)
    (vbin / "python").symlink_to(orig / ".python" / ALIAS / "bin" / "python3.12")
    (vbin / "python3").symlink_to("python")
    (copy / "venv" / "pyvenv.cfg").write_text(
        f"home = {orig}/.python/{ALIAS}/bin\nimplementation = CPython\nrelocatable = true\n")
    return orig, copy


def test_a_copied_bundle_reports_each_link_that_leaves_it(tmp_path):
    _, b = installed_bundle(tmp_path)
    problems = br.problems(b)
    assert any("venv/bin/python" in p for p in problems)
    assert any(ALIAS in p for p in problems)
    assert any("pyvenv.cfg" in p for p in problems)


def test_relink_points_everything_inside_the_bundle_with_relative_links(tmp_path):
    _, b = installed_bundle(tmp_path)
    br.relink(b)
    assert br.problems(b) == []
    py = b / "venv" / "bin" / "python"
    assert not os.readlink(py).startswith("/")
    assert not os.readlink(b / ".python" / ALIAS).startswith("/")
    assert py.resolve() == (b / ".python" / REAL / "bin" / "python3.12").resolve()
    assert f"home = {b}/.python/{ALIAS}/bin" in (b / "venv" / "pyvenv.cfg").read_text()
    assert "relocatable = true" in (b / "venv" / "pyvenv.cfg").read_text()


def test_the_bundle_still_works_after_it_moves_again(tmp_path):
    _, b = installed_bundle(tmp_path)
    br.relink(b)
    moved = tmp_path / "elsewhere" / "b"
    moved.parent.mkdir()
    b.rename(moved)
    assert subprocess.run([str(moved / "venv" / "bin" / "python")], capture_output=True, text=True).stdout == "real\n"


def test_relink_is_idempotent_and_leaves_a_good_bundle_alone(tmp_path):
    _, b = installed_bundle(tmp_path)
    assert br.relink(b)
    cfg = (b / "venv" / "pyvenv.cfg").read_text()
    assert br.relink(b) == []
    assert (b / "venv" / "pyvenv.cfg").read_text() == cfg


def test_a_link_whose_target_is_not_in_the_bundle_is_left_and_reported(tmp_path):
    _, b = installed_bundle(tmp_path)
    (b / "venv" / "bin" / "python").unlink()
    (b / "venv" / "bin" / "python").symlink_to("/usr/bin/python3")
    br.relink(b)
    assert os.readlink(b / "venv" / "bin" / "python") == "/usr/bin/python3"
    assert any("venv/bin/python" in p for p in br.problems(b))


def test_a_bundle_without_a_venv_has_no_problems(tmp_path):
    (tmp_path / "b").mkdir()
    assert br.problems(tmp_path / "b") == [] and br.relink(tmp_path / "b") == []


def test_the_module_runs_as_python_c_source_for_the_lab(tmp_path):
    """lab setup sends the source over ssh as `python3 -c <source> ...`; the lab may not have orchard."""
    _, b = installed_bundle(tmp_path)
    src = Path(br.__file__).read_text()
    check = subprocess.run([sys.executable, "-c", src, "--check", str(b)], capture_output=True, text=True)
    assert check.returncode == 1 and "venv/bin/python" in check.stdout
    fix = subprocess.run([sys.executable, "-c", src, str(b)], capture_output=True, text=True)
    assert fix.returncode == 0, fix.stderr
    assert subprocess.run([sys.executable, "-c", src, "--check", str(b)]).returncode == 0


def test_a_broken_relative_link_inside_the_bundle_is_reported(tmp_path):
    _, b = installed_bundle(tmp_path)
    br.relink(b)
    (b / ".python" / REAL).rename(b / ".python" / "gone")
    assert any("broken link" in p for p in br.problems(b))
