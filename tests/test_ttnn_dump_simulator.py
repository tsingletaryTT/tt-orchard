# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`inventory refresh-ttnn --simulator LIB`: list ttnn's names against the ttsim library, with no gozer lease."""
import io
import json

from orchard import cli
from orchard import ttnn_dump as td

LINE = "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "9.9", "names": ["add", "matmul"]}) + "\n"


def test_with_a_simulator_the_command_is_env_python_and_no_gozer(tmp_path):
    lib = tmp_path / "libttsim_bh.so"
    lib.write_bytes(b"")
    argv = td.dump_command("/py/bin/python", simulator=str(lib))
    assert argv[0] == "env" and "gozer" not in argv and "run" not in argv[:2]
    assert argv[1] == f"TT_METAL_SIMULATOR={lib}"
    assert "TT_METAL_ARCH_NAME=blackhole" in argv and "TT_METAL_SLOW_DISPATCH_MODE=1" in argv
    assert argv[-3:-1] == ["/py/bin/python", "-c"] and argv[-1] == td.SCRIPT


def test_without_a_simulator_the_command_is_unchanged():
    argv = td.dump_command("/py/bin/python")
    assert argv[:2] == ["gozer", "run"] and argv[argv.index("--") + 1] == "/py/bin/python"
    assert td.dump_command("/py/bin/python", simulator=None) == argv and td.dump_command("/py/bin/python", simulator="") == argv


def test_main_with_a_simulator_runs_it_says_there_is_no_lease_and_writes_the_dump(tmp_path):
    lib = tmp_path / "lib.so"
    lib.write_bytes(b"")
    said, calls = [], []

    def run(argv, timeout):
        calls.append(argv)
        return 0, LINE, ""

    assert td.main("/py", run=run, data_dir=tmp_path, say=said.append, simulator=str(lib)) == 0
    assert calls[0][0] == "env" and (tmp_path / "ttnn_ops-9.9.json").is_file()
    assert "no lease" in said[0] and "gozer" not in said[0]


def test_main_refuses_a_simulator_library_that_does_not_exist_and_runs_nothing(tmp_path):
    said = []
    code = td.main("/py", run=lambda *a: (_ for _ in ()).throw(AssertionError("must not run")), data_dir=tmp_path,
                   say=said.append, simulator=str(tmp_path / "missing.so"))
    assert code == 2 and "does not exist" in said[0] and not list(tmp_path.glob("ttnn_ops-*"))


def test_main_with_a_simulator_and_a_failed_run_writes_nothing(tmp_path):
    lib = tmp_path / "lib.so"
    lib.write_bytes(b"")
    assert td.main("/py", run=lambda a, t: (1, LINE, "boom"), data_dir=tmp_path, say=lambda s: None,
                   simulator=str(lib)) == 2
    assert not list(tmp_path.glob("ttnn_ops-*"))


def test_the_command_line_passes_the_simulator_through(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(td, "main", lambda python, **kw: seen.append((python, kw)) or 0)
    code = cli.main(["inventory", "refresh-ttnn", "--ttnn-python", "/t/py", "--simulator", "/s/lib.so"],
                    env={"HOME": str(tmp_path)}, stdout=io.StringIO())
    assert code == 0 and seen[0][0] == "/t/py" and seen[0][1]["simulator"] == "/s/lib.so"
    seen.clear()
    cli.main(["inventory", "refresh-ttnn", "--ttnn-python", "/t/py"], env={"HOME": str(tmp_path)}, stdout=io.StringIO())
    assert seen[0][1]["simulator"] is None
