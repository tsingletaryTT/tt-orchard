# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 403: `Supervisor._copy_package` reads stage 7's files through the run directory's pinned descriptor. A file or a parent
directory replaced after the file was listed must not copy an outside file into the operator bundle."""
import os
from types import SimpleNamespace

from orchard import safeio
from orchard.supervisor import Supervisor


def test_copy_package_refuses_a_file_swapped_for_a_link_after_it_was_listed(tmp_path, monkeypatch):
    run = tmp_path / "run"
    s7 = run / "stages" / "7"
    (s7 / "package" / "p1").mkdir(parents=True)
    (s7 / "package.json").write_text("{}")
    (s7 / "PUBLISH_COMMANDS.txt").write_text("commands")
    (s7 / "package" / "p1" / "README.md").write_text("card")
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "PUBLISH_COMMANDS.txt").write_text("outside the run")
    (outside / "README.md").write_text("outside the run")
    real = safeio.read_bytes

    def swapping(path, **kw):
        # Just before each read the file, or the directory above it, is swapped for a link to the outside copy.
        if path.endswith("PUBLISH_COMMANDS.txt"):
            (run / path).unlink()
            os.symlink(outside / "PUBLISH_COMMANDS.txt", run / path)
        if path.endswith("p1/README.md"):
            os.rename(s7 / "package" / "p1", s7 / "package" / "p1.real")
            os.symlink(outside, s7 / "package" / "p1")
        return real(path, **kw)
    monkeypatch.setattr(safeio, "read_bytes", swapping)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    Supervisor._copy_package(SimpleNamespace(run_dir=run), bundle)
    copied = {f.name: f.read_text() for f in (bundle / "package").iterdir()}
    assert copied == {"package.json": "{}"}, copied
