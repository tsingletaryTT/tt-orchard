# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The size limit of the CPU run of a stage 2 port (issue 782, after issue 697).

`port_check.py --cpu N` refused any checkpoint above 4 GiB. A 2.3B-parameter bf16 model holds 4,593,276,272 bytes, so the CPU run gave no
verdict on it and a plain Python fault in the port reached the chips on every repair round. The limit now comes from the host's available
memory (`MemAvailable`), with a floor (the old 4 GiB), a cap, and an override in the environment. The line the template prints says which
limit applied and where it came from, and the supervisor writes that line into the ledger.

The limit function is tested by loading the shipped template as a module. The end-to-end tests run the template in a subprocess, as
tests/test_cpu_run_before_board.py does. No test reads checkpoint bytes: the checkpoints here are sparse files whose size is the only thing
the template looks at, and the port's `load` does not open them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cpu_run_before_board import GOOD, no_verdict_line, run_cpu, stage          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "orchard" / "skills" / "full-port-templates" / "port_check.py"
GIB = 1 << 30
AION_BYTES = 4_593_276_272         # the checkpoint of microsoft/Aion-1.0-Instruct-2.3B, which the old limit refused


@pytest.fixture(scope="module")
def tpl(tmp_path_factory):
    """The shipped template as a module. It finds the run directory from its own location (a `stages` directory above it), so the module is
    loaded from a copy placed the way the supervisor places it."""
    where = tmp_path_factory.mktemp("tpl") / "run" / "stages" / "2"
    where.mkdir(parents=True)
    copy = where / "port_check.py"
    copy.write_bytes(TEMPLATE.read_bytes())
    spec = importlib.util.spec_from_file_location("port_check_under_test", copy)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def meminfo(tmp_path: Path, avail_kb: int | None, name: str = "meminfo") -> Path:
    p = tmp_path / name
    lines = ["MemTotal:       263000000 kB", "MemFree:         1000000 kB"]
    if avail_kb is not None:
        lines.append(f"MemAvailable:   {avail_kb} kB")
    p.write_text("\n".join(lines) + "\n")
    return p


# ---- the limit function ---------------------------------------------------------------------------------------------------------------

def test_a_large_host_gets_the_cap_and_a_2_3b_checkpoint_fits(tpl, tmp_path):
    limit, source = tpl.cpu_run_limit({}, meminfo(tmp_path, 140 * GIB // 1024))
    assert limit == tpl.CPU_RUN_CAP_BYTES and limit > AION_BYTES
    assert "cap" in source and "MemAvailable" in source


def test_a_mid_size_host_gets_a_share_of_its_available_memory(tpl, tmp_path):
    avail = 30 * GIB
    limit, source = tpl.cpu_run_limit({}, meminfo(tmp_path, avail // 1024))
    assert limit == avail // tpl.CPU_RUN_MEMORY_DIVISOR
    assert tpl.CPU_RUN_FLOOR_BYTES < limit < tpl.CPU_RUN_CAP_BYTES
    assert "MemAvailable" in source and str(avail) in source


def test_a_small_host_keeps_the_old_floor(tpl, tmp_path):
    limit, source = tpl.cpu_run_limit({}, meminfo(tmp_path, 2 * GIB // 1024))
    assert limit == tpl.CPU_RUN_FLOOR_BYTES == 4 * GIB
    assert "floor" in source


@pytest.mark.parametrize("make", ["missing", "no_line", "garbage", "unit"])
def test_a_host_that_cannot_say_gets_the_floor_and_the_line_says_so(tpl, tmp_path, make):
    if make == "missing":
        path = tmp_path / "nope"
    elif make == "no_line":
        path = meminfo(tmp_path, None)
    elif make == "garbage":
        path = tmp_path / "meminfo"
        path.write_text("MemAvailable: lots kB\n")
    else:
        path = tmp_path / "meminfo"
        path.write_text("MemAvailable: 99999999999 MB\n")          # a unit the kernel does not use: the number is not trusted
    limit, source = tpl.cpu_run_limit({}, path)
    assert limit == tpl.CPU_RUN_FLOOR_BYTES and "floor" in source and "MemAvailable" in source


def test_the_meminfo_read_is_bounded(tpl, tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("x" * (tpl.CPU_RUN_MEMINFO_MAX * 4) + "\nMemAvailable: 999999999 kB\n")
    limit, _ = tpl.cpu_run_limit({}, p)
    assert limit == tpl.CPU_RUN_FLOOR_BYTES           # the line past the bound is not seen


def test_an_override_replaces_the_memory_rule_and_is_named(tpl, tmp_path):
    env = {tpl.CPU_RUN_OVERRIDE_ENV: str(20 * GIB)}
    limit, source = tpl.cpu_run_limit(env, meminfo(tmp_path, 2 * GIB // 1024))
    assert limit == 20 * GIB and tpl.CPU_RUN_OVERRIDE_ENV in source


def test_an_override_above_the_ceiling_is_cut_to_the_ceiling(tpl, tmp_path):
    env = {tpl.CPU_RUN_OVERRIDE_ENV: str(10_000 * GIB)}
    limit, source = tpl.cpu_run_limit(env, meminfo(tmp_path, 140 * GIB // 1024))
    assert limit == tpl.CPU_RUN_OVERRIDE_CEILING_BYTES and "ceiling" in source


@pytest.mark.parametrize("bad", ["", "0", "-5", "12GiB", "1e9", " 7 ", "9" * 40])
def test_a_bad_override_is_ignored_and_the_line_says_so(tpl, tmp_path, bad):
    env = {tpl.CPU_RUN_OVERRIDE_ENV: bad}
    limit, source = tpl.cpu_run_limit(env, meminfo(tmp_path, 140 * GIB // 1024))
    assert limit == tpl.CPU_RUN_CAP_BYTES            # the memory rule applied
    assert "ignored" in source and tpl.CPU_RUN_OVERRIDE_ENV in source


# ---- the template end to end ----------------------------------------------------------------------------------------------------------

def sparse(path: Path, size: int) -> None:
    with open(path, "wb") as fh:
        fh.truncate(size)


def run_with_limit(d: Path, limit_bytes: int, **kw):
    import os
    import subprocess
    tmp = d.parents[2]
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp / "decoy"), "ORCHARD_CPU_RUN_MAX_BYTES": str(limit_bytes)}
    return subprocess.run([sys.executable, "-P", "-s", str(d / "port_check.py"), "--cpu", "1"], capture_output=True, text=True, env=env,
                          cwd=str(d.parent.parent), timeout=300)


def test_an_aion_sized_checkpoint_gets_the_cpu_run_and_a_port_fault_is_found(tmp_path):
    bad = GOOD.replace("return _forward(model, len(ids))", "return _forward(model, model['absent'])")          # a KeyError at a port line
    d = stage(tmp_path, bad)
    sparse(tmp_path / "snapshot" / "model.safetensors", AION_BYTES - 2)          # config.json adds 2 bytes
    r = run_with_limit(d, 8 * GIB)
    assert r.returncode == 2, r.stdout + r.stderr          # a fault of the port, found without a board
    assert "CPU run found a fault of the port" in r.stdout


def test_a_passing_run_says_the_size_and_the_limit_it_used(tmp_path):
    d = stage(tmp_path, GOOD)
    sparse(tmp_path / "snapshot" / "model.safetensors", AION_BYTES - 2)          # config.json adds 2 bytes
    r = run_with_limit(d, 8 * GIB)
    assert r.returncode == 0, r.stdout + r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("CPU run passed"))
    assert str(AION_BYTES) in line and str(8 * GIB) in line and "ORCHARD_CPU_RUN_MAX_BYTES" in line


def test_a_checkpoint_above_the_limit_is_not_loaded_and_the_line_names_size_limit_and_source(tmp_path):
    marker = tmp_path / "LOAD-RAN"
    d = stage(tmp_path, GOOD.replace("def load(model_dir, mesh_device):\n", f"def load(model_dir, mesh_device):\n    open({str(marker)!r}, 'w').close()\n"))
    sparse(tmp_path / "snapshot" / "model.safetensors", AION_BYTES - 2)          # config.json adds 2 bytes
    r = run_with_limit(d, 2 * GIB)
    line = no_verdict_line(r)
    assert r.returncode == 3 and not marker.exists()
    assert str(AION_BYTES) in line and str(2 * GIB) in line and "ORCHARD_CPU_RUN_MAX_BYTES" in line and "not loaded" in line


def test_a_huge_checkpoint_is_refused_under_the_default_rule_on_any_host(tmp_path):
    d = stage(tmp_path, GOOD)
    sparse(tmp_path / "snapshot" / "model.safetensors", 200 * GIB)       # above the cap, the ceiling and any host's share
    r = run_cpu(d)
    assert r.returncode == 3 and "not loaded" in no_verdict_line(r) and "limit" in no_verdict_line(r)


# ---- the supervisor: the override travels through `--env`, and the ledger keeps the line that names the limit ---------------------------

@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_the_override_set_with_env_reaches_the_cpu_run_and_the_ledger_names_size_limit_and_source(tmp_path):
    from test_supervisor_full_port import HONEST, coder_writes, with_snapshot
    from test_supervisor import Rig
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        r.args.env.append("ORCHARD_CPU_RUN_MAX_BYTES=1")            # the snapshot holds a tokenizer file, so it is above one byte
        coder_writes(r, {}, port=HONEST)
        r.run()
        nv = next(e["data"] for e in r.entries() if e["event"] == "decision" and e["stage"] == 2 and e["data"].get("decision") == "CPU run gave no verdict")
        assert "above the CPU run's limit of 1 bytes" in nv["reason"] and "ORCHARD_CPU_RUN_MAX_BYTES=1" in nv["reason"], nv["reason"]
        assert "not loaded" in nv["reason"]
    finally:
        r.close()


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_passing_cpu_run_leaves_its_size_limit_line_in_the_ledger(tmp_path, monkeypatch):
    """The template's pass line says the size and the limit. The supervisor copies that line into the ledger decision, so a reader of the
    ledger sees which rule applied without the run's output file. The template is replaced here by a stub that prints a pass line, because
    the wiring under test belongs to the supervisor; the template's line is tested above."""
    from orchard.supervisor import Supervisor
    from test_supervisor_full_port import HONEST, coder_writes, with_snapshot
    from test_supervisor import Rig
    real = Supervisor._precheck_spawn
    line = "CPU run passed: stub. The checkpoint holds 4593276272 bytes; the size limit was 12884901888 bytes (the cap of 12884901888 bytes)."

    def spawn(self, stage_dir, command, deadline_s):
        if not command.rstrip().split()[-2:-1] == ["--cpu"]:
            return real(self, stage_dir, command, deadline_s)
        out = stage_dir / "evidence" / "hw-test-output.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(line + "\n")
        return 0, False, self.clock(), out

    monkeypatch.setattr(Supervisor, "_precheck_spawn", spawn)
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        coder_writes(r, {}, port=HONEST)
        r.run()
        d = next(e["data"] for e in r.entries() if e["event"] == "decision" and e["stage"] == 2 and e["data"].get("decision") == "CPU run passed before the board")
        assert "4593276272 bytes" in d["size_limit"] and "12884901888" in d["size_limit"] and "the cap" in d["size_limit"]
    finally:
        r.close()
