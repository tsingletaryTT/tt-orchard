# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 339: the dump script walks the submodules of ttnn.experimental, dumps are keyed by version and build, and the loader picks
the dump for the installed build. Everything here runs against a fake module tree and temporary files. No device, no ttnn."""
import importlib.metadata
import json
import os
import sys
import types

import pytest

from orchard import opinventory as oi
from orchard import ttnn_dump as td


def ns(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def fake_tree(tmp_path):
    """ttnn.experimental.deepseek_prefill.update_padded_kv_cache, three levels down, with the traps a real tree has: a cycle back to
    the parent, a numpy-like module that is not ttnn's, a class, and a namespace the script does not walk."""
    deep = ns("ttnn.experimental.deepseek_prefill", update_padded_kv_cache=lambda *a: None, Config=type("Config", (), {}))
    deeper = ns("ttnn.experimental.deepseek_prefill.kernels", fused=lambda *a: None)
    deep.kernels = deeper
    exp = ns("ttnn.experimental", rotary=lambda *a: None, deepseek_prefill=deep, np=ns("numpy", zeros=lambda *a: None))
    deeper.back = exp                                  # a cycle
    ttnn = ns("ttnn", matmul=lambda *a: None, experimental=exp, transformer=ns("ttnn.transformer", sdpa=lambda *a: None),
              operations=ns("ttnn.operations", helper=lambda *a: None))
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    so = pkg / "_ttnn.so"
    so.write_bytes(b"")
    os.utime(so, (1793318400, 1793318400))            # 2026-10-30 00:00:00 UTC
    ttnn.__file__ = str(pkg / "__init__.py")
    return ttnn


def run_script(monkeypatch, capsys, ttnn):
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.80.0")
    exec(compile(td.SCRIPT, "<dump>", "exec"), {"__name__": "dump"})
    return td.parse(capsys.readouterr().out)


def test_the_script_walks_submodules_of_experimental_to_any_depth(monkeypatch, capsys, tmp_path):
    dump = run_script(monkeypatch, capsys, fake_tree(tmp_path))
    assert "experimental.deepseek_prefill.update_padded_kv_cache" in dump["names"]
    assert "experimental.deepseek_prefill.kernels.fused" in dump["names"]
    assert "experimental.rotary" in dump["names"] and "transformer.sdpa" in dump["names"] and "matmul" in dump["names"]


def test_the_walk_ends_on_a_cycle_leaves_classes_and_foreign_modules_out(monkeypatch, capsys, tmp_path):
    names = run_script(monkeypatch, capsys, fake_tree(tmp_path))["names"]
    assert not any(n.endswith(".back.rotary") or ".back." in n for n in names)
    assert "experimental.deepseek_prefill.Config" not in names and not any("np.zeros" in n for n in names)


def test_a_namespace_the_script_does_not_walk_is_listed_not_dropped(monkeypatch, capsys, tmp_path):
    dump = run_script(monkeypatch, capsys, fake_tree(tmp_path))
    assert dump["unwalked_namespaces"] == ["operations"] and "operations.helper" not in dump["names"]


def test_the_script_reports_the_build_date_of_the_compiled_library(monkeypatch, capsys, tmp_path):
    assert run_script(monkeypatch, capsys, fake_tree(tmp_path))["build"] == "20261030"


def test_the_build_id_adds_a_commit_and_survives_a_missing_library(tmp_path):
    (tmp_path / "_ttnn.so").write_bytes(b"")
    os.utime(tmp_path / "_ttnn.so", (1793318400, 1793318400))
    assert oi.build_id(tmp_path, "ABCDEF0123456789abcdef") == "20261030-abcdef012345"
    assert oi.build_id(tmp_path / "none", "abcdef0") == "abcdef0"
    assert oi.build_id(tmp_path / "none") is None
    assert oi.build_id(tmp_path, "not hex") == "20261030"


# ---- the file name and the overwrite guard ------------------------------------------------------------------------------------


def test_two_builds_with_one_version_get_two_files(tmp_path):
    a = td.write_dump({"ttnn_version": "0.80.0", "build": "20261029", "names": ["add"]}, tmp_path)
    b = td.write_dump({"ttnn_version": "0.80.0", "build": "20261030", "names": ["add", "x"]}, tmp_path)
    assert a.name == "ttnn_ops-0.80.0-build-20261029.json" and b.name == "ttnn_ops-0.80.0-build-20261030.json" and a != b
    assert json.loads(a.read_text())["names"] == ["add"] and json.loads(b.read_text())["build"] == "20261030"


def test_a_dump_with_no_build_will_not_overwrite_a_different_list_under_the_version_name(tmp_path):
    td.write_dump({"ttnn_version": "0.80.0", "names": ["add"]}, tmp_path)
    with pytest.raises(ValueError, match="no build id"):
        td.write_dump({"ttnn_version": "0.80.0", "names": ["add", "x"]}, tmp_path)
    td.write_dump({"ttnn_version": "0.80.0", "names": ["add"]}, tmp_path)           # the same list again is a plain refresh
    assert json.loads((tmp_path / "ttnn_ops-0.80.0.json").read_text())["names"] == ["add"]


@pytest.mark.parametrize("bad", ["../x", "2026-10-30", "zz", "20261030/../x", 20261030])
def test_a_build_id_that_is_not_a_date_or_commit_is_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        td.write_dump({"ttnn_version": "0.80.0", "build": bad, "names": ["add"]}, tmp_path)


def test_main_says_the_build_file_it_wrote(tmp_path):
    out = "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "0.80.0", "build": "20261030", "names": ["add"]}) + "\n"
    said = []
    assert td.main("/py", run=lambda a, t: (0, out, ""), data_dir=tmp_path, say=said.append) == 0
    assert (tmp_path / "ttnn_ops-0.80.0-build-20261030.json").is_file() and "build-20261030" in said[-1]


# ---- the loader ---------------------------------------------------------------------------------------------------------------


def put(tmp_path, build, names, legacy=False):
    name = "ttnn_ops-0.80.0.json" if legacy else f"ttnn_ops-0.80.0-build-{build}.json"
    body = {"ttnn_version": "0.80.0", "names": names, **({"build": build} if build else {})}
    (tmp_path / name).write_text(json.dumps(body))


def test_the_loader_picks_the_newest_dump_when_the_installed_build_is_unknown(tmp_path):
    put(tmp_path, None, ["old"], legacy=True)
    put(tmp_path, "20261029", ["mid"])
    put(tmp_path, "20261030", ["new"])
    assert oi.load_dump("0.80.0", data_dir=tmp_path)["names"] == ["new"]


def test_the_loader_picks_the_dump_for_the_installed_build(tmp_path):
    put(tmp_path, "20261029", ["mid"])
    put(tmp_path, "20261030", ["new"])
    assert oi.load_dump("0.80.0", data_dir=tmp_path, build="20261029")["names"] == ["mid"]


def test_a_legacy_dump_is_the_oldest(tmp_path):
    put(tmp_path, None, ["old"], legacy=True)
    put(tmp_path, "20261029", ["mid"])
    assert oi.load_dump("0.80.0", data_dir=tmp_path)["names"] == ["mid"]
    assert oi.load_dump("0.80.0", data_dir=tmp_path, build="20300101")["names"] == ["mid"]      # no exact match: the newest comes back


def test_a_dump_for_another_version_is_never_chosen(tmp_path):
    (tmp_path / "ttnn_ops-0.80.0-build-20261030.json").write_text(json.dumps({"ttnn_version": "0.81.0", "names": ["x"]}))
    assert oi.load_dump("0.80.0", data_dir=tmp_path) is None


def test_build_refuses_a_dump_from_another_build_date_and_says_why():
    raw = {"schema": 1, "model_dir": "/m", "architectures": ["X"], "traced_class": "X", "layers_in_config": 2, "layers_traced": 2,
           "seq_len": 8, "failures": [], "ops": [{"op": "aten.mm.default", "count": 1, "count_in_layers": 0, "sample": None, "modules": []}]}
    dump = {"ttnn_version": "0.80.0", "build": "20261029", "names": ["matmul"]}
    kw = dict(mapping={"aten.mm.default": {"ttnn": "matmul"}}, decomps={}, dump=dump, installed_ttnn="0.80.0")
    inv = oi.build(raw, installed_build="20261030", **kw)
    assert inv["ttnn_dump"] is None and "20261029" in inv["ttnn_dump_note"] and "20261030" in inv["ttnn_dump_note"]
    assert oi.build(raw, installed_build="20261029", **kw)["ttnn_dump"] is not None
    assert oi.build(raw, **kw)["ttnn_dump"] is not None                                  # build unknown: as before
    legacy = dict(kw, dump={"ttnn_version": "0.80.0", "names": ["matmul"]})
    assert oi.build(raw, installed_build="20261030", **legacy)["ttnn_dump"] is not None   # a legacy dump has no build to compare


def test_the_installed_build_is_read_without_importing_ttnn():
    seen = []

    def run(argv, timeout):
        seen.append(argv)
        return 0, "20261030-abcdef012345\n", ""
    assert oi.installed_ttnn_build(run, "/py") == "20261030"
    code = seen[0][-1]
    assert "find_spec" in code and "import ttnn" not in code
    assert oi.installed_ttnn_build(lambda a, t: (1, "", "x"), "/py") is None and oi.installed_ttnn_build(run, None) is None


def test_the_shipped_loader_returns_the_build_dated_dump_for_0_80_0():
    """The shipped file for the 2026-10-30 build now names its build. It still lacks the nested experimental ops: they need a real
    refresh on a device with the new walk (issue 339), which this change does not run."""
    d = oi.load_dump("0.80.0")
    assert d["build"] == "20261030" and len(d["names"]) == 829
    assert oi.load_dump("0.80.0", build="20261030") is d or oi.load_dump("0.80.0", build="20261030") == d
