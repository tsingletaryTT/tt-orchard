# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The managed tt-lang pair venv (orchard/ttlang_pair.py), the build-info record that feeds the consistency row, and the
kernel probe's choice of hardware interpreter. Nothing here opens a device or touches a real venv: every directory is made
up in tmp_path and every command is a fake."""
from __future__ import annotations

import ast
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from test_kernelprobe import (canned, fake_kenv, probe_args, skills_current, consistency_ok, ttlang_skill,  # noqa: F401
                              gozer_tripwire)
from orchard import kernelprobe as kp, ttlang_consistency as tc, ttlang_env, ttlang_pair as tp, ttlang_pin

EXPORTS = ("export TT_METAL_HOME=/m/tt-metal\nexport TT_METAL_RUNTIME_ROOT=\"$TT_METAL_HOME\"\n"
           "export PYTHONPATH=/m/tt-metal/ttnn:/m/tt-metal/tools${PYTHONPATH:+:$PYTHONPATH}\n"
           "export LD_LIBRARY_PATH=/m/tt-metal/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n")


class Fake:
    """A runner that records every argv and answers from a list of (needle, (rc, out, err)) rules."""
    def __init__(self, rules=(), on_run=None):
        self.calls, self.rules, self.on_run = [], list(rules), on_run

    def __call__(self, argv, timeout=60, env=None, cwd=None):
        self.calls.append(list(argv))
        if self.on_run:
            self.on_run(argv)
        for needle, answer in self.rules:
            if needle in " ".join(map(str, argv)):
                return answer
        return 0, "", ""


def make_pair_venv(root: Path, exports: str | None = EXPORTS) -> Path:
    venv = root / "pairvenv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("#!/bin/sh\n")
    if exports is not None:
        (venv / tp.EXPORTS_FILE).write_text(exports)
    return venv


# ---- A: the plan and the venv check ------------------------------------------------------------------------------

def test_default_venv_is_under_the_cache_root_and_can_be_moved(tmp_path):
    assert tp.default_venv(environ={}, home=tmp_path) == tmp_path / ".cache" / "tt-orchard" / "ttlang-pair" / "venv"
    assert tp.default_venv(environ={"ORCHARD_CACHE_ROOT": "/c"}, home=tmp_path) == Path("/c/ttlang-pair/venv")
    assert tp.default_venv(environ={"ORCHARD_PAIR_VENV": "/v"}, home=tmp_path) == Path("/v")


def test_plan_is_built_from_the_pin_constants(tmp_path):
    steps = tp.build_plan(tmp_path / "v", Path("/m/tt-metal"), tmp_path / "dl")
    text = "\n".join(s.shell for s in steps)
    assert ttlang_pin.WHEEL_VERSION in text and ttlang_pin.WHEEL_INDEX in text
    for name, sha in ttlang_pin.WHEEL_SHA256.items():
        assert name in text and sha in text
    assert "uv venv" in text and "tt-lang-setup-external-tt-metal --tt-metal-dir /m/tt-metal" in text
    assert [s.name for s in steps] == ["download", "hashes", "venv", "install", "exports"]


def test_plan_names_the_safetensors_note(tmp_path):
    assert "safetensors" in tp.NOTES and "PYTHONPATH" in tp.NOTES


def test_printing_the_plan_runs_nothing(tmp_path):
    fake = Fake()
    out = io.StringIO()
    args = argparse.Namespace(pair=True, apply=False, check=False, pair_venv_dir=str(tmp_path / "v"),
                              tt_metal_dir="/m/tt-metal", pair_download_dir=str(tmp_path / "dl"))
    assert tp.main(args, run=fake, out=out) == 2          # the venv is absent, so the check part says todo
    assert fake.calls == []
    assert "uv venv" in out.getvalue() and "--apply" in out.getvalue()
    assert not (tmp_path / "v").exists() and not (tmp_path / "dl").exists()


def test_apply_runs_the_steps_in_order_and_checks_the_hashes_before_installing(tmp_path, monkeypatch):
    dl = tmp_path / "dl"
    good = {"a-1.whl": b"aaa", "b-1.whl": b"bbb"}
    monkeypatch.setattr(ttlang_pin, "WHEEL_SHA256", {k: hashlib.sha256(v).hexdigest() for k, v in good.items()})

    def on_run(argv):
        if "download" in argv:
            dl.mkdir(parents=True, exist_ok=True)
            for k, v in good.items():
                (dl / k).write_bytes(v)
    fake = Fake(on_run=on_run, rules=[("tt-lang-setup-external-tt-metal", (0, EXPORTS, ""))])
    ok = tp.apply(tp.build_plan(tmp_path / "v", Path("/m/tt-metal"), dl), run=fake, say=lambda s: None, venv=tmp_path / "v")
    assert ok
    order = [c[0] for c in fake.calls]
    assert order == ["python3", "uv", "uv", str(tmp_path / "v" / "bin" / "tt-lang-setup-external-tt-metal")]
    assert (tmp_path / "v" / tp.EXPORTS_FILE).read_text() == EXPORTS


def test_apply_stops_before_the_venv_when_a_wheel_hash_differs(tmp_path, monkeypatch):
    dl = tmp_path / "dl"
    monkeypatch.setattr(ttlang_pin, "WHEEL_SHA256", {"a-1.whl": hashlib.sha256(b"aaa").hexdigest()})

    def on_run(argv):
        if "download" in argv:
            dl.mkdir(parents=True, exist_ok=True)
            (dl / "a-1.whl").write_bytes(b"TAMPERED")
    fake = Fake(on_run=on_run)
    lines = []
    ok = tp.apply(tp.build_plan(tmp_path / "v", Path("/m/tt-metal"), dl), run=fake, say=lines.append, venv=tmp_path / "v")
    assert not ok
    assert not any(c[0] == "uv" for c in fake.calls)
    assert any("sha256" in l for l in lines)


def test_apply_refuses_a_missing_wheel(tmp_path, monkeypatch):
    monkeypatch.setattr(ttlang_pin, "WHEEL_SHA256", {"a-1.whl": "0" * 64})
    fake = Fake(on_run=lambda argv: (tmp_path / "dl").mkdir(exist_ok=True))
    assert not tp.apply(tp.build_plan(tmp_path / "v", Path("/m/tt-metal"), tmp_path / "dl"), run=fake,
                        say=lambda s: None, venv=tmp_path / "v")
    assert not any(c[0] == "uv" for c in fake.calls)


def metadata_answer(light=ttlang_pin.WHEEL_VERSION, lang=ttlang_pin.WHEEL_VERSION + "+light"):
    return 0, json.dumps({"tt-lang-light": light, "tt-lang": lang}), ""


def test_verify_ok_for_the_pinned_wheels_and_an_exports_file(tmp_path):
    venv = make_pair_venv(tmp_path)
    (tmp_path / "tt-metal").mkdir()
    rows = tp.verify(venv, run=Fake([("importlib.metadata", metadata_answer())]))
    assert [r.status for r in rows] == ["ok"] * len(rows), rows


def test_verify_reads_the_venv_without_importing_ttl_or_ttnn(tmp_path):
    venv = make_pair_venv(tmp_path)
    fake = Fake([("importlib.metadata", metadata_answer())])
    tp.verify(venv, run=fake)
    argv = fake.calls[0]
    assert argv[0] == str(venv / "bin" / "python") and "-I" in argv
    tree = ast.parse(argv[argv.index("-c") + 1])
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"importlib", "json"} and "ttl" not in imported and "ttnn" not in imported


def test_verify_reports_a_wrong_version_missing_wheel_and_missing_exports(tmp_path):
    venv = make_pair_venv(tmp_path, exports=None)
    rows = {r.name: r for r in tp.verify(venv, run=Fake([("importlib.metadata", metadata_answer(light="1.1.6"))]))}
    assert rows["tt-lang-light"].status == "mismatch" and "1.1.6" in rows["tt-lang-light"].detail
    assert rows["exports"].status == "mismatch"
    rows = {r.name: r for r in tp.verify(venv, run=Fake([("importlib.metadata", (0, json.dumps({"tt-lang": None,
             "tt-lang-light": None}), ""))]))}
    assert rows["tt-lang-light"].status == "mismatch" and "not installed" in rows["tt-lang-light"].detail


def test_verify_of_an_absent_venv_is_one_todo_row(tmp_path):
    rows = tp.verify(tmp_path / "nope", run=Fake())
    assert [r.status for r in rows] == ["todo"]


def test_exports_are_parsed_and_unknown_names_are_dropped():
    got = tp.parse_exports(EXPORTS + "export EVIL=1\n")
    assert got["TT_METAL_HOME"] == "/m/tt-metal" and got["TT_METAL_RUNTIME_ROOT"] == "/m/tt-metal"
    assert got["PYTHONPATH"] == "/m/tt-metal/ttnn:/m/tt-metal/tools" and "EVIL" not in got


def test_exports_with_a_substitution_are_refused():
    with pytest.raises(ValueError):
        tp.parse_exports("export PYTHONPATH=$(rm -rf x)\n")
    with pytest.raises(ValueError):
        tp.parse_exports("export PYTHONPATH=$HOME/x\n")


# ---- B: the build-info file ---------------------------------------------------------------------------------------

def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "ttlang_build_info.py"
    spec = importlib.util.spec_from_file_location("ttlang_build_info", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_maps_the_real_build_info_keys():
    real = {"version": "1.1.10.dev20261026+light", "ttlang": "fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0",
            "tt_metal_tag": "v0.79.0-dev20260917", "tt_metal": "0e9d200db976120c129ab0deb13aa3f6d972b723"}
    got = load_script().normalize(real)
    assert got["tt_lang_commit"] == real["ttlang"] and got["tt_metal_commit"] == real["tt_metal"]
    assert got["version"] == real["version"] and got["tt_metal_tag"] == real["tt_metal_tag"]


def test_script_keeps_a_gap_as_a_gap():
    got = load_script().normalize({"version": "x"})
    assert got["tt_lang_commit"] is None and got["tt_metal_commit"] is None


def test_script_does_not_import_ttl_at_module_level():
    tree = ast.parse((Path(__file__).resolve().parent.parent / "scripts" / "ttlang_build_info.py").read_text())
    top = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    assert "ttl" not in top


def test_load_build_info_file_round_trips_and_refuses_junk(tmp_path):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"tt_lang_commit": ttlang_pin.SHA, "tt_metal_commit": ttlang_pin.TT_METAL_REF}))
    assert tc.load_build_info(f)["tt_lang_commit"] == ttlang_pin.SHA
    for bad in ("not json", "[1]", ""):
        f.write_text(bad)
        with pytest.raises(ValueError):
            tc.load_build_info(f)
    with pytest.raises(ValueError):
        tc.load_build_info(tmp_path / "missing.json")


def test_hardware_row_turns_ok_with_the_file_and_stays_todo_without(tmp_path, fake_kenv):
    ok = {"tt_lang_commit": ttlang_pin.SHA, "tt_metal_commit": ttlang_pin.TT_METAL_REF}
    row = tc._tree_row(fake_kenv, None, ok)
    assert row.status == "ok"
    assert tc._tree_row(fake_kenv, None, None).status == "todo"
    bad = dict(ok, tt_metal_commit="c49bb7625e6a495337483353a5228339fece2c11")
    assert tc._tree_row(fake_kenv, None, bad).status == "mismatch"


def test_main_passes_the_file_into_the_check(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(tc, "check", lambda env, skill, **kw: seen.update(kw) or [])
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"tt_lang_commit": "a" * 40, "tt_metal_commit": "b" * 40}))
    args = argparse.Namespace(src_dir=str(tmp_path), venv_dir=str(tmp_path), tt_lang_skill_dir=str(tmp_path),
                              tt_metal_dir=str(tmp_path), build_info_file=str(f))
    assert tc.main(args, out=io.StringIO()) == 0
    assert seen["build_info"]["tt_lang_commit"] == "a" * 40


def test_main_refuses_an_unreadable_build_info_file(tmp_path):
    args = argparse.Namespace(src_dir=str(tmp_path), venv_dir=str(tmp_path), tt_lang_skill_dir=str(tmp_path),
                              tt_metal_dir=str(tmp_path), build_info_file=str(tmp_path / "missing.json"))
    out = io.StringIO()
    assert tc.main(args, out=out) == 2 and "build-info-file" in out.getvalue()


# ---- C: the probe's hardware interpreter ----------------------------------------------------------------------------

def test_flag_wins_over_the_pair_venv(tmp_path):
    venv = make_pair_venv(tmp_path)
    got = tp.resolve_hardware_python("/x/python", environ={"ORCHARD_PAIR_VENV": str(venv)}, home=tmp_path, fallback="/old/python")
    assert got.python == "/x/python" and got.source == "flag" and got.warning is None


def test_pair_venv_is_the_default_when_it_exists(tmp_path):
    venv = make_pair_venv(tmp_path)
    got = tp.resolve_hardware_python(None, environ={"ORCHARD_PAIR_VENV": str(venv)}, home=tmp_path, fallback="/old/python")
    assert got.python == str(venv / "bin" / "python") and got.source == "pair" and got.warning is None


def test_missing_pair_venv_falls_back_with_a_warning(tmp_path):
    got = tp.resolve_hardware_python(None, environ={"ORCHARD_PAIR_VENV": str(tmp_path / "nope")}, home=tmp_path,
                                     fallback="/old/python")
    assert got.python == "/old/python" and got.source == "fallback"
    assert "kernel-env --pair" in got.warning and "1.1.6" in got.warning


def test_a_pair_venv_without_exports_falls_back_with_a_warning(tmp_path):
    venv = make_pair_venv(tmp_path, exports=None)
    got = tp.resolve_hardware_python(None, environ={"ORCHARD_PAIR_VENV": str(venv)}, home=tmp_path, fallback="/old/python")
    assert got.python == "/old/python" and "exports" in got.warning


def test_hw_command_adds_the_exports_only_for_a_pair_python(tmp_path):
    venv = make_pair_venv(tmp_path)
    cmd = tp.hw_command(str(venv / "bin" / "python"))
    assert cmd[0] == "env" and cmd[-1] == str(venv / "bin" / "python")
    assert "PYTHONPATH=/m/tt-metal/ttnn:/m/tt-metal/tools" in cmd and "TT_METAL_HOME=/m/tt-metal" in cmd
    assert tp.hw_command("/old/python") == ["/old/python"]


def test_the_probe_gozer_command_carries_the_pair_exports(tmp_path, monkeypatch):
    venv = make_pair_venv(tmp_path)
    monkeypatch.delenv("ORCHARD_PROBE_CHIP", raising=False)
    argv = kp._gozer("why", str(venv / "bin" / "python"), "h.py")
    after = argv[argv.index("--") + 1:]
    assert after[0] == "env" and after[-2:] == [str(venv / "bin" / "python"), "h.py"]
    assert kp._gozer("why", "/old/python", "h.py")[-3:] == ["--", "/old/python", "h.py"] or \
        kp._gozer("why", "/old/python", "h.py")[-2:] == ["/old/python", "h.py"]


def test_main_defaults_the_hardware_python_to_the_pair_venv(tmp_path, monkeypatch, fake_kenv):
    venv = make_pair_venv(tmp_path)
    monkeypatch.setenv("ORCHARD_PAIR_VENV", str(venv))
    seen = []
    monkeypatch.setattr(kp, "run_attempt", lambda *, op, n, **kw: seen.append(kw) or canned(op, n))
    monkeypatch.setattr(kp.ttlang_env, "ensure", lambda env, **k: [ttlang_env.Result("verify", ttlang_env.OK, "x")])
    out = io.StringIO()
    kp.main(probe_args(tmp_path, shared_python=None), env=fake_kenv, out=out, probe=lambda e, m: True)
    assert seen[0]["shared_python"] == str(venv / "bin" / "python")


def test_main_warns_and_uses_the_old_venv_when_there_is_no_pair_venv(tmp_path, monkeypatch, fake_kenv):
    monkeypatch.setenv("ORCHARD_PAIR_VENV", str(tmp_path / "nope"))
    seen = []
    monkeypatch.setattr(kp, "run_attempt", lambda *, op, n, **kw: seen.append(kw) or canned(op, n))
    monkeypatch.setattr(kp.ttlang_env, "ensure", lambda env, **k: [ttlang_env.Result("verify", ttlang_env.OK, "x")])
    out = io.StringIO()
    kp.main(probe_args(tmp_path, shared_python=None), env=fake_kenv, out=out, probe=lambda e, m: True)
    assert seen[0]["shared_python"] == kp.default_shared_python()
    assert "warning" in out.getvalue().lower() and "kernel-env --pair" in out.getvalue()
