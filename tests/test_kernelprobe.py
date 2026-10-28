# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernel probe (orchard/kernelprobe.py): attempts, the authoritative re-check, the hardware phase and the report.

Most tests use a fake `run_watched` and a FakeModel, so they need no simulator and no chip. The `needs_sim` tests run the
shipped kernel_check.py on the real tt-lang simulator and skip, saying so, when the ttlang venv or the tt-lang checkout is
absent. No test runs gozer: an autouse tripwire puts a stub gozer first on PATH and fails the test if anything ran it.
Every hardware call goes through a fake `run_watched`, which records the argv and never executes it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path

import pytest

from fake_model import FakeModel, call, final, turn
from orchard import kernelprobe as kp, ttlang_env
from orchard.agent import NO_CHIP

REPO = Path(__file__).resolve().parent.parent
KERNELS = Path(__file__).parent / "fixtures" / "kernel-loop" / "kernels"
VENV = Path(os.environ.get("ORCHARD_TTLANG_VENV", Path.home() / "orchard-venvs" / "ttlang"))
SRC = Path(os.environ.get("ORCHARD_TTLANG_SRC", Path.home() / "orchard-src" / "tt-lang"))
needs_sim = pytest.mark.skipif(not (VENV / "bin" / "tt-lang-sim").exists() or not (SRC / "examples").is_dir(),
                               reason="SKIPPED: no tt-lang-sim venv or tt-lang checkout (run `tt-orchard kernel-env`)")


@pytest.fixture(autouse=True)
def gozer_tripwire(stub_tools):
    """Every test runs with stub tools first on PATH. A real `gozer` call from anywhere (an agent shell, a hardware
    phase that forgot its fake) lands in the stub's log, and the test fails here."""
    yield stub_tools
    assert not [c for c in stub_tools.calls() if c.startswith("gozer")], stub_tools.calls()


@pytest.fixture
def kenv(tmp_path):
    return ttlang_env.KEnv(src_dir=SRC, venv_dir=VENV, scratch_dir=tmp_path / "scratch")


SKILL_FILES = {"guide.md": "# stand-in for the tt-lang skill's guide\n", "examples.md": "# stand-in examples\n",
               "SKILL.md": "---\nname: ttm-tt-lang\n---\n"}


@pytest.fixture(autouse=True)
def ttlang_skill(tmp_path, monkeypatch):
    """Every test gets a stand-in tt-lang skill directory through ORCHARD_TTLANG_SKILL, so no test reads the one
    installed in the operator's home."""
    d = tmp_path / "skill-ttm-tt-lang"
    d.mkdir()
    for name, text in SKILL_FILES.items():
        (d / name).write_text(text)
    monkeypatch.setenv(ttlang_env.TTLANG_SKILL_ENV, str(d))
    return d


def _fresh(source, state, **kw):
    from orchard import skillsfresh as sf
    return sf.Freshness(source, f"/x/{source}", state, kw.pop("detail", f"{source} is {state}."),
                        local="1" * 40, remote="2" * 40, **kw)


@pytest.fixture(autouse=True)
def skills_current(monkeypatch):
    """The skills freshness check answers "current" for both kernel sources unless a test says otherwise. The calls
    are recorded (skill directory and whether the cache may be written)."""
    from orchard import skillsfresh as sf
    state = {"results": [_fresh(sf.SOURCE_TTLANG, sf.CURRENT), _fresh(sf.SOURCE_TTLANG_SKILL, sf.CURRENT)],
             "calls": [], "real": ttlang_env.skills_status}

    def fake(env, skill_dir, *, write_cache=True, preread=None):
        state["calls"].append({"skill_dir": Path(skill_dir), "write_cache": write_cache})
        state["preread"] = preread
        return list(state["results"])
    monkeypatch.setattr(ttlang_env, "skills_status", fake)
    return state


FACTS = {"sim_version": "1.1.10+gtest", "source_sha": "a" * 40, "diagnostics_patched": True, "patch_sha256": "b" * 64}


@pytest.fixture
def fake_kenv(tmp_path):
    """A KEnv whose venv and checkout are made up, so tests that do not run the simulator need neither."""
    src = tmp_path / "src"
    (src / "examples").mkdir(parents=True)
    (src / "examples" / "eltwise_add.py").write_text("# stand-in for upstream's example\nprint('PASSED!')\n")
    (src / "examples" / "eltwise_broadcast_reduce.py").write_text("# stand-in for upstream's reduce example\n")
    (src / "docs" / "sphinx" / "specs").mkdir(parents=True)
    (src / "docs" / "sphinx" / "specs" / "TTLangSpecification.md").write_text("# stand-in specification\n")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "tt-lang-sim").write_text("#!/bin/sh\nexit 1\n")
    lib = tmp_path / "purelib"
    lib.mkdir()

    def run(argv, timeout, env=None):
        if argv[-1] == "--version":
            return 0, "tt-lang-sim 1.1.10+gfake\n", ""
        if "rev-parse" in argv:
            return 0, "c" * 40 + "\n", ""
        if "sysconfig" in " ".join(argv):
            return 0, f"{lib}\n", ""
        return 1, "", "not faked"

    (src / ".git").mkdir()
    return ttlang_env.KEnv(src_dir=src, venv_dir=venv, scratch_dir=tmp_path / "scratch", run=run)


def scripted(*messages):
    it = iter(messages)
    return lambda request: next(it, final("done"))


def check_writer(result: dict, calls: list | None = None):
    """A fake run_watched that writes `result` where the authoritative check would, and records the argv."""
    def fake(argv, **kw):
        if calls is not None:
            calls.append(list(argv))
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(result))
        return kp.Watched(1 if not result.get("passed") else 0, None, False, 0.1)
    return fake


def run_attempt(tmp_path, env, server, **kw):
    args = dict(op="add", n=1, probe_dir=tmp_path / "probe", env=env, endpoint=server.endpoint, model="fake",
                coder_model="fake-coder", probe_id="t", max_turns=10, wall_s=120, tt_lang=FACTS)
    args.update(kw)
    return kp.run_attempt(**args)


# ---- the simulator phase, against the real simulator -----------------------------------------------------------


@needs_sim
def test_an_agent_that_writes_a_good_kernel_passes_and_the_report_records_it(tmp_path, kenv):
    good = (KERNELS / "add_good.py").read_text()
    with FakeModel(scripted(call("write_file", path="kernel.py", content=good),
                            call("shell", command="python3 kernel_check.py"), final("done"))) as server:
        attempt = kp.run_attempt(op="add", n=1, probe_dir=tmp_path, env=kenv, endpoint=server.endpoint, model="fake",
                                 coder_model="fake", probe_id="t", max_turns=10, wall_s=120)
    assert attempt["passed_sim"] is True and attempt["category"] is None and attempt["agent_check_untrusted"] is False
    assert (tmp_path / "add" / "attempt-1" / "attempt.json").is_file()
    assert attempt["pcc_min"] >= 0.999 and attempt["provenance"]["simulator_pcc_min"] == attempt["pcc_min"]
    # the agent's own check ran too (it wrote check.json), and it agreed
    assert json.loads((tmp_path / "add" / "attempt-1" / "check.json").read_text())["passed"] is True
    kp.write_report(tmp_path, [attempt], {"tt_lang": attempt["provenance"]["tt_lang"], "coder_model": "fake",
                                          "probe_id": "t", "bars": {}, "hw": None})
    md = (tmp_path / "kernel-probe-report.md").read_text()
    assert "| 1 | yes | - |" in md and "attempts to first pass in the simulator: 1" in md


@needs_sim
def test_an_agent_that_fakes_its_own_check_is_not_believed(tmp_path, kenv):
    """Review focus 3. The agent replaces kernel_check.py with a script that prints PASS and writes a passing check.json."""
    liar = ("import json,sys\n"
            "json.dump({'passed': True, 'category': None, 'sha256': {}}, open('check.json','w'))\n"
            "print('PASS')\n")
    bad = (KERNELS / "add_wrong_op.py").read_text()
    with FakeModel(scripted(call("write_file", path="kernel.py", content=bad),
                            call("write_file", path="kernel_check.py", content=liar),
                            call("shell", command="python3 kernel_check.py"), final("done"))) as server:
        attempt = kp.run_attempt(op="add", n=1, probe_dir=tmp_path, env=kenv, endpoint=server.endpoint, model="fake",
                                 coder_model="fake", probe_id="t", max_turns=10, wall_s=120)
    assert attempt["passed_sim"] is False and attempt["category"] == "numeric_mismatch"
    assert attempt["agent_check_untrusted"] is True


# ---- the authoritative check ------------------------------------------------------------------------------------


def test_the_authoritative_check_uses_the_shipped_spec_and_not_the_attempts_copy(tmp_path, fake_kenv):
    """The agent can edit kernel_spec.json in its directory (lower the bar, change the reference). The verdict must not read it."""
    seen = {}

    def fake_run_watched(argv, **kw):
        seen["argv"], seen["kw"] = list(argv), kw
        (Path(argv[argv.index("--out") + 1])).write_text(json.dumps({"passed": False, "category": "numeric_mismatch", "first_line": "x", "sets": []}))
        return kp.Watched(1, None, False, 0.1)

    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel_spec.json").write_text("{}")                       # tampered
    (adir / "kernel.py").write_text("# the agent's kernel\n")
    (adir / "ttnn.py").write_text("# a planted module\n")
    kp.authoritative_check(adir, op="add", env=fake_kenv, run_watched=fake_run_watched, tt_lang=FACTS)
    argv = seen["argv"]
    assert Path(argv[1]) == kp.TEMPLATES / "kernel_check.py"
    assert Path(argv[argv.index("--spec") + 1]) == kp.TEMPLATES / "ops" / "add.json"
    assert argv[argv.index("--sim") + 1] == str(fake_kenv.sim)
    assert json.loads(argv[argv.index("--tt-lang-json") + 1]) == FACTS
    # The check runs in a fresh directory that holds only a copy of kernel.py, so nothing else the agent wrote is near it.
    clean = Path(argv[argv.index("--attempt-dir") + 1])
    assert clean != adir and sorted(p.name for p in clean.iterdir()) == ["kernel.py"]
    assert (clean / "kernel.py").read_text() == "# the agent's kernel\n"


def test_a_check_killed_for_a_stall_is_a_timeout_or_hang_never_a_pass(tmp_path, fake_kenv):
    def stalled(argv, **kw):
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"passed": True, "category": None, "sets": []}))
        return kp.Watched(-9, "stalled: no output and no CPU use for 120 s", True, 130.0)
    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel.py").write_text((KERNELS / "add_spin.py").read_text())
    got = kp.authoritative_check(adir, op="add", env=fake_kenv, run_watched=stalled, tt_lang=FACTS)
    assert got["passed"] is False and got["category"] == "timeout" and "stalled" in got["first_line"]


def test_an_agent_check_from_a_changed_template_is_untrusted_even_when_it_failed(tmp_path, fake_kenv):
    """The agent's check.json says FAIL, as the authoritative one does, but its hashes say it came from an edited check."""
    def agent_writes_check(request):
        if turn(request) == 0:
            return call("write_file", path="check.json",
                        content=json.dumps({"passed": False, "category": "numeric_mismatch",
                                            "sha256": {"kernel_check.py": "0" * 64}}))
        return final("done")
    authoritative = {"passed": False, "category": "numeric_mismatch", "first_line": "set seed0: pcc 0.1", "sets": []}
    with FakeModel(agent_writes_check) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
    assert attempt["agent_check_untrusted"] is True


def test_no_agent_check_is_not_untrusted(tmp_path, fake_kenv):
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(lambda r: final("nothing")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
    assert attempt["agent_check_untrusted"] is False and attempt["category"] == "no_kernel"


def test_attempt_json_records_violations_and_the_range_of_each_measure(tmp_path, fake_kenv):
    sets = [{"name": "seed0", "pcc": 0.9995, "rel_l2": 0.004, "slope": 0.999, "max_abs_err": 0.01, "exp_abs_max": 3.0,
             "finite": True},
            {"name": "seed0-tail", "pcc": 0.9991, "rel_l2": 0.008, "slope": 1.004, "max_abs_err": 0.02,
             "exp_abs_max": 40.0, "finite": True},
            {"name": "seed1", "pcc": None, "rel_l2": None, "slope": None, "max_abs_err": 0.0, "exp_abs_max": 3.0,
             "finite": False}]
    violations = [{"rule": "import", "line": 1, "detail": "import os", "remedy": "import only ttl"}]
    authoritative = {"passed": False, "category": "forbidden_code", "first_line": "1 violation: import at line 1",
                     "sets": sets, "violations": violations}
    adir = tmp_path / "probe" / "add" / "attempt-1"
    kernel = (KERNELS / "forbidden_import.py").read_text()
    with FakeModel(agent_does(lambda: (adir / "kernel.py").write_text(kernel))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
    on_disk = json.loads((tmp_path / "probe" / ".verdict" / "add-1" / "attempt.json").read_text())
    assert on_disk == attempt
    assert json.loads((tmp_path / "probe" / "add" / "attempt-1" / "attempt.json").read_text()) == attempt
    assert attempt["category"] == "forbidden_code" and attempt["violations"] == violations
    assert attempt["pcc_min"] == 0.9991 and attempt["slope_range"] == [0.999, 1.004] and attempt["rel_l2_max"] == 0.008
    assert set(attempt) >= {"op", "attempt", "status", "turns", "agent_wall_s", "passed_sim", "category", "first_line",
                            "pcc_min", "sets", "hw", "agent_check_untrusted", "provenance"}
    assert attempt["hw"] is None and attempt["status"] == "done"
    prov = attempt["provenance"]
    assert prov["origin"] == "kernel" and prov["tt_lang"] == FACTS and prov["coder_model"] == "fake-coder"
    assert prov["probe_id"] == "t" and prov["hardware_pcc_min"] is None
    assert prov["kernel_sha256"] == hashlib.sha256(kernel.encode()).hexdigest()


# ---- the agent, its budget and its watchers ---------------------------------------------------------------------


def test_the_budget_ends_the_agent_through_the_control_object():
    now = [0.0]
    ctl = kp.ProbeControl(wall_s=100, clock=lambda: now[0])
    assert ctl.stop_reason() is None and ctl.take_nudges("a") == []
    now[0] = 101.0
    assert "wall" in ctl.stop_reason()


def test_run_attempt_stops_the_agent_when_its_wall_budget_is_spent(tmp_path, fake_kenv):
    """The budget is wired into the step: with no time left the agent gets no turn at all."""
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(lambda r: call("shell", command="ls")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, wall_s=0, run_watched=check_writer(authoritative))
        assert server.requests == []
    assert "wall" in attempt["status"] and attempt["turns"] == 0


def test_the_agent_environment_has_no_device_and_no_network_tools(tmp_path):
    adir = tmp_path / "add" / "attempt-1"
    adir.mkdir(parents=True)
    tools = kp.attempt_tools(adir)
    assert tools.env["TT_VISIBLE_DEVICES"] == NO_CHIP and tools.env["TT_METAL_VISIBLE_DEVICES"] == NO_CHIP
    assert tools.run_dir == adir and tools.stage_dir == adir
    assert tools.call("shell", json.dumps({"command": "pip install x"})).startswith("refused:")
    assert tools.call("shell", json.dumps({"command": "gozer run --chips 1 -- ls"})).startswith("refused:")


def test_a_model_that_repeats_the_same_call_is_stopped_by_the_watchdog(tmp_path, fake_kenv):
    def same(request):
        msg = call("shell", command="ls")
        msg["content"] = "let me look again"
        return msg
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(same) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, max_turns=30, run_watched=check_writer(authoritative))
    assert attempt["turns"] < 10 and attempt["status"] == "escalate"
    from orchard.ledger import read_entries
    entries = read_entries(tmp_path / "probe" / "add" / "attempt-1" / "ledger.jsonl")
    assert [e["event"] for e in entries if e["data"].get("watchdog")] == ["retry", "escalate"]


def test_a_model_that_never_writes_kernel_py_is_nudged(tmp_path, fake_kenv):
    def wanders(request):
        n = turn(request)
        return call("shell", command=f"echo word{n}") if n < 24 else final("gave up")
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(wanders) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, max_turns=30, run_watched=check_writer(authoritative))
        later = [m["content"] for r in server.requests[20:] for m in r["messages"] if m["role"] == "user"]
    assert any("no file was written" in t for t in later)
    from orchard.ledger import read_entries
    entries = read_entries(tmp_path / "probe" / "add" / "attempt-1" / "ledger.jsonl")
    nudges = [e for e in entries if e["data"].get("watchdog") and e["event"] == "retry"]
    assert nudges and nudges[0]["data"]["findings"][0]["detector"] == "no_file_written"
    assert attempt["status"] == "done"


# ---- preparing an attempt ---------------------------------------------------------------------------------------


def test_prepare_attempt_copies_the_reference_example_from_the_checkout_and_vendors_nothing(tmp_path, fake_kenv):
    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    assert (adir / "reference" / "eltwise_add.py").read_text() == \
        (fake_kenv.src_dir / "examples" / "eltwise_add.py").read_text()
    for name in ("kernel_check.py", "kernel_harness.py", "references.py"):
        assert (adir / name).read_bytes() == (kp.TEMPLATES / name).read_bytes()
    assert (adir / "kernel_spec.json").read_bytes() == (kp.TEMPLATES / "ops" / "add.json").read_bytes()
    assert not (adir / "kernel.py").exists()
    if (SRC / "examples" / "eltwise_add.py").is_file():
        upstream = (SRC / "examples" / "eltwise_add.py").read_bytes()
        # One test fixture is upstream's file byte for byte on purpose: the static gate's tests feed it to the gate
        # to show that the example's host code is refused (test_kernel_loop_templates.py, UPSTREAM). It is never given
        # to an agent. Nothing else may be a copy.
        allowed = {KERNELS / "upstream_eltwise_add.py"}
        for root in (REPO / "orchard", REPO / "tests"):
            for p in root.rglob("*.py"):
                if p not in allowed:
                    assert p.read_bytes() != upstream, f"{p} is a copy of upstream's eltwise_add.py"


def test_the_skill_names_the_gate_rules_and_the_thread_functions():
    text = kp.SKILL.read_text()
    flat = " ".join(text.split())
    for needle in ("kernel.py", "kernel_check.py", "reference/eltwise_add.py", "@ttl.operation",
                   "import torch", "from_torch", "to_torch", "Do not open a device", "main()", "@ttl.compute()",
                   "@ttl.datamovement()", "import only", "forbidden_code", "ttl.dfb_reset"):
        assert needle in flat, needle
    assert "gozer" not in text.lower()


# ---- the hardware phase -----------------------------------------------------------------------------------------


def _passed_attempt(tmp_path, env, kernel="add_good.py", n=1):
    """An attempt whose simulator check passed, built without the simulator: the authoritative copy is in place."""
    adir = tmp_path / "add" / f"attempt-{n}"
    kp.prepare_attempt(adir, op="add", env=env, tt_lang=FACTS)
    (adir / "kernel.py").write_text((KERNELS / kernel).read_text())
    clean = kp.verdict_dir(tmp_path, "add", n) / kp.CLEAN_DIR
    clean.mkdir(parents=True)
    (clean / "kernel.py").write_text((KERNELS / kernel).read_text())
    attempt = {"op": "add", "attempt": n, "passed_sim": True, "category": None, "hw": None,
               "provenance": kp.provenance(kernel_sha256=hashlib.sha256((clean / "kernel.py").read_bytes()).hexdigest(),
                                           tt_lang=FACTS, coder_model="c", probe_id="t", simulator_pcc_min=0.9999)}
    (adir / "attempt.json").write_text(json.dumps(attempt))
    return adir, attempt


def hw_writer(sets, calls, *, passed_log="PASSED!\n"):
    """A fake run_watched for the hardware: the smoke test logs PASSED!, a kernel run writes hw-result.json."""
    def fake(argv, **kw):
        calls.append(list(argv))
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": sets, "error": None}))
        else:
            with open(kw["log_path"], "a") as f:
                f.write(passed_log)
        return kp.Watched(0, None, False, 1.0)
    return fake


GOOD_HW_SET = {"name": "seed0", "pcc": 0.995, "rel_l2": 0.03, "slope": 1.03, "max_abs_err": 0.1, "exp_abs_max": 3.0,
               "finite": True}


def test_hardware_commands_run_only_through_gozer_and_only_after_a_passing_smoke_test(tmp_path, fake_kenv):
    calls = []

    def run_watched_fake(argv, **kw):
        calls.append(list(argv))
        return kp.Watched(0, None, False, 1.0)

    ok, _ = kp.smoke_test(env=fake_kenv, shared_python="/shared/python", run_watched=run_watched_fake)
    assert calls[0][:2] == ["gozer", "run"] and "--chips" in calls[0] and calls[0][calls[0].index("--chips") + 1] == "1"
    assert calls[0][calls[0].index("--") + 1] == "/shared/python"
    assert calls[0][calls[0].index("--exact") + 1] == "1" and kp.GOZER_WHO in calls[0]
    assert ok is False                                     # exit 0 without PASSED! in the log is not a pass

    # With a passing smoke test, every kernel run is a gozer run of the SHIPPED harness on the checked copy.
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    calls.clear()
    hw = kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/shared/python",
                         run_watched=hw_writer([GOOD_HW_SET], calls))
    assert hw["available"] is True and len(calls) == 2
    smoke, run = calls
    assert smoke[-1] == str(fake_kenv.src_dir / "examples" / "eltwise_add.py")
    assert run[:2] == ["gozer", "run"] and run[run.index("--chips") + 1] == "1" and run[run.index("--exact") + 1] == "1"
    tail = run[run.index("--") + 1:]
    assert tail[0] == "/shared/python" and Path(tail[1]) == kp.TEMPLATES / "kernel_harness.py"
    assert Path(tail[tail.index("--spec") + 1]) == kp.TEMPLATES / "ops" / "add.json"
    assert Path(tail[tail.index("--kernel") + 1]) == kp.verdict_dir(tmp_path, "add", 1) / kp.CLEAN_DIR / "kernel.py"
    assert Path(tail[tail.index("--out") + 1]).parent == kp.verdict_dir(tmp_path, "add", 1)
    assert tail[tail.index("--phase") + 1] == "hw"


def test_the_hardware_result_is_judged_with_the_hardware_bars(tmp_path, fake_kenv):
    """pcc 0.995 and slope 1.03 fail the simulator bars and pass the hardware ones."""
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], []))
    assert hw["passed"] is True and hw["category"] is None and hw["pcc_min"] == 0.995
    assert json.loads((kp.verdict_dir(tmp_path, "add", 1) / "attempt.json").read_text())["hw"]["passed"] is True
    assert attempt["provenance"]["hardware_pcc_min"] == 0.995
    adir2, attempt2 = _passed_attempt(tmp_path, fake_kenv, n=2)
    low = dict(GOOD_HW_SET, pcc=0.98)
    hw = kp.hardware_phase(adir2, attempt2, env=fake_kenv, shared_python="/p", run_watched=hw_writer([low], []))
    assert hw["passed"] is False and hw["category"] == "hw_numeric_mismatch"


def test_a_hardware_hang_blocks_the_probe_and_names_the_attempt(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv, n=3)
    adir4, attempt4 = _passed_attempt(tmp_path, fake_kenv, n=4)
    calls = []

    def hangs(argv, **kw):
        calls.append(list(argv))
        if "--out" not in argv:
            with open(kw["log_path"], "a") as f:
                f.write("PASSED!\n")
            return kp.Watched(0, None, False, 1.0)
        return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)

    with pytest.raises(kp.ProbeBlocked) as info:
        kp.run_hardware([attempt, attempt4], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=hangs)
    assert "add attempt 3" in str(info.value) and "stalled" in str(info.value)
    assert len(calls) == 2                                 # the smoke test and the first kernel run, which hung
    on_disk = json.loads((kp.verdict_dir(tmp_path, "add", 3) / "attempt.json").read_text())
    assert on_disk["hw"]["category"] == "hang" and on_disk["hw"]["passed"] is False


def test_a_failed_smoke_test_makes_the_report_say_hw_unavailable_and_no_gozer_call_for_kernels_follows(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    calls = []
    hw = kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                         run_watched=hw_writer([GOOD_HW_SET], calls, passed_log="TT_FATAL: no device\n"))
    assert len(calls) == 1 and hw["available"] is False and "TT_FATAL: no device" in hw["smoke_tail"]
    assert attempt["hw"] is None
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="t", ops=["add"], hw=hw, tt_lang=FACTS)
    kp.write_report(tmp_path, [attempt], header)
    md = (tmp_path / "kernel-probe-report.md").read_text()
    assert "hw: unavailable" in md and "TT_FATAL: no device" in md


def test_a_forbidden_kernel_never_reaches_gozer_in_the_hardware_phase(tmp_path, fake_kenv):
    """The simulator check applies the static gate, and the hardware path applies it again before any gozer call."""
    adir, attempt = _passed_attempt(tmp_path, fake_kenv, kernel="forbidden_import.py")
    calls = []
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], calls))
    assert calls == []
    assert hw["passed"] is False and hw["category"] == "forbidden_code"
    assert hw["violations"] and hw["violations"][0]["rule"] == "import"


def test_an_attempt_that_failed_the_simulator_gets_no_hardware_run(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    attempt["passed_sim"] = False
    calls = []
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], calls))
    assert len(calls) == 1 and attempt["hw"] is None     # only the smoke test ran


# ---- the report -------------------------------------------------------------------------------------------------


def test_the_report_header_lists_versions_patch_hashes_coder_and_provisional_bars(tmp_path, fake_kenv):
    adir = tmp_path / "probe" / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv)
    header = kp.report_header(env=fake_kenv, coder_model="qwen3-coder-next", probe_id="p1", ops=["add", "rmsnorm"])
    facts = header["tt_lang"]
    assert facts["sim_version"] == "1.1.10+gfake" and facts["source_sha"] == "c" * 40
    assert facts["diagnostics_patched"] is False and facts["patch_sha256"] == kp.defaults.TTLANG_DIAGNOSTICS_SHA256
    # the agent's own check sees the same facts the report names
    assert json.loads((adir / "check_config.json").read_text())["tt_lang"] == facts
    assert header["bars"]["add"]["sim"]["pcc"] == 0.999 and header["bars"]["add"]["hw"]["pcc"] == 0.99
    kp.write_report(tmp_path / "probe", [], header)
    md = (tmp_path / "probe" / "kernel-probe-report.md").read_text()
    for needle in ("1.1.10+gfake", "c" * 40, kp.defaults.TTLANG_DIAGNOSTICS_SHA256, "qwen3-coder-next", "provisional",
                   "pcc >= 0.999", "pcc >= 0.99", "greedy decoding", "synthetic", "diagnostics patch applied: no"):
        assert needle in md, needle
    data = json.loads((tmp_path / "probe" / "kernel-probe-report.json").read_text())
    assert data["header"] == header and data["attempts"] == []


def _row(op, n, passed, hw=None, category=None):
    return {"op": op, "attempt": n, "status": "done", "turns": 5, "agent_wall_s": 12.5, "passed_sim": passed,
            "category": category, "first_line": "" if passed else "set seed0: pcc 0.2", "pcc_min": 0.9999 if passed else 0.2,
            "sets": [], "hw": hw, "agent_check_untrusted": False, "provenance": {}}


def test_write_report_counts_attempts_to_first_pass_per_op(tmp_path):
    attempts = [_row("silu_mul", 1, False, category="numeric_mismatch"), _row("silu_mul", 2, False, category="tt_lang_error"),
                _row("silu_mul", 3, True, hw={"passed": True, "category": None, "pcc_min": 0.995}),
                _row("rmsnorm", 1, False, category="forbidden_code"), _row("rmsnorm", 2, False, category="timeout")]
    attempts[1]["agent_check_untrusted"] = True
    kp.write_report(tmp_path, attempts, {"tt_lang": FACTS, "coder_model": "c", "probe_id": "t", "bars": {}, "hw": None})
    md = (tmp_path / "kernel-probe-report.md").read_text()
    silu, rms = md.split("## silu_mul")[1].split("## rmsnorm")
    assert "attempts to first pass in the simulator: 3" in silu and "attempts to first pass on hardware: 3" in silu
    assert "none of 2 attempts passed in the simulator" in rms
    assert "| 3 | yes | yes |" in silu and "| 1 | no | - |" in rms
    assert "forbidden_code" in rms and "untrusted" in silu.split("| 2 |")[1].splitlines()[0]
    data = json.loads((tmp_path / "kernel-probe-report.json").read_text())
    assert data["first_pass"] == {"silu_mul": {"sim": 3, "hw": 3}, "rmsnorm": {"sim": None, "hw": None}}


def test_the_category_list_includes_forbidden_code_and_hang():
    assert {"no_kernel", "import_error", "tt_lang_error", "shape_mismatch", "numeric_mismatch", "timeout",
            "compile_error", "hang", "hw_numeric_mismatch", "harness_error", "forbidden_code", "queued",
            "sim_gap"} == set(kp.CATEGORIES)


# ---- the probe-owned verdict directory (fix round 1) ------------------------------------------------------------


NO_KERNEL = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}


def agent_does(action):
    """A scripted agent whose first turn runs `action()` in the test process (standing in for a shell command the
    agent ran), then ends."""
    def script(request):
        if turn(request) == 0:
            action()
            return call("shell", command="ls")
        return final("done")
    return script


def test_a_symlinked_attempt_json_is_replaced_and_its_target_left_alone(tmp_path, fake_kenv):
    operator_file = tmp_path / "operator.txt"
    operator_file.write_text("the operator's file\n")
    adir = tmp_path / "probe" / "add" / "attempt-1"
    with FakeModel(agent_does(lambda: (adir / "attempt.json").symlink_to(operator_file))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL))
    assert operator_file.read_text() == "the operator's file\n"
    assert json.loads((kp.verdict_dir(tmp_path / "probe", "add", 1) / "attempt.json").read_text()) == attempt
    assert not (adir / "attempt.json").is_symlink() and json.loads((adir / "attempt.json").read_text()) == attempt


def test_a_symlinked_kernel_py_is_forbidden_and_nothing_runs(tmp_path, fake_kenv):
    secret = tmp_path / "secret.txt"
    secret.write_text("import ttl\nTOKEN = 'hf_secretvalue'\n")
    adir = tmp_path / "probe" / "add" / "attempt-1"
    calls = []
    with FakeModel(agent_does(lambda: (adir / "kernel.py").symlink_to(secret))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL, calls))
    assert calls == [] and attempt["passed_sim"] is False and attempt["category"] == "forbidden_code"
    assert "regular file" in attempt["first_line"]
    assert "hf_secretvalue" not in (kp.verdict_dir(tmp_path / "probe", "add", 1) / "attempt.json").read_text()
    assert not (kp.verdict_dir(tmp_path / "probe", "add", 1) / kp.CLEAN_DIR / "kernel.py").exists()


def test_an_oversized_kernel_py_is_forbidden_and_nothing_runs(tmp_path, fake_kenv):
    adir = tmp_path / "probe" / "add" / "attempt-1"
    calls = []
    big = "# padding\n" * (kp.MAX_KERNEL_BYTES // 10 + 10)
    with FakeModel(agent_does(lambda: (adir / "kernel.py").write_text(big))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL, calls))
    assert calls == [] and attempt["category"] == "forbidden_code" and "256 KiB" in attempt["first_line"]


def test_a_missing_kernel_py_is_no_kernel_and_nothing_runs(tmp_path, fake_kenv):
    calls = []
    with FakeModel(lambda r: final("nothing")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL, calls))
    assert calls == [] and attempt["category"] == "no_kernel"


def test_a_planted_authoritative_symlink_does_not_crash_and_the_check_uses_the_probe_owned_copy(tmp_path, fake_kenv):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep")
    adir = tmp_path / "probe" / "add" / "attempt-1"

    def plant():
        (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
        (adir / "authoritative").symlink_to(elsewhere, target_is_directory=True)
    calls = []
    with FakeModel(agent_does(plant)) as server:
        run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL, calls))
    clean = Path(calls[0][calls[0].index("--attempt-dir") + 1])
    assert clean == kp.verdict_dir(tmp_path / "probe", "add", 1) / kp.CLEAN_DIR
    assert sorted(p.name for p in elsewhere.iterdir()) == ["keep.txt"]


def test_a_check_result_rewritten_in_the_attempt_directory_is_not_read(tmp_path, fake_kenv):
    """A background process the agent left behind writes a passing check-authoritative.json into its directory after
    the shipped check wrote the real one in the probe-owned directory."""
    adir = tmp_path / "probe" / "add" / "attempt-1"
    real = {"passed": False, "category": "numeric_mismatch", "first_line": "set seed0: pcc 0.1", "sets": []}

    def fake(argv, **kw):
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(real))
        (adir / "check-authoritative.json").write_text(json.dumps({"passed": True, "category": None, "sets": []}))
        return kp.Watched(1, None, False, 0.1)
    with FakeModel(agent_does(lambda: (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text()))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=fake)
    assert attempt["passed_sim"] is False and attempt["category"] == "numeric_mismatch"


def test_the_hardware_phase_refuses_a_clean_copy_that_changed(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    (kp.verdict_dir(tmp_path, "add", 1) / kp.CLEAN_DIR / "kernel.py").write_text(
        (KERNELS / "add_good.py").read_text() + "# changed\n")
    calls = []
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], calls))
    assert calls == [] and hw["passed"] is False and hw["category"] == "harness_error"
    assert "the clean kernel copy changed" in hw["first_line"]


def test_load_attempts_reads_the_probe_owned_records(tmp_path, fake_kenv):
    with FakeModel(lambda r: final("nothing")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL))
    (tmp_path / "probe" / "add" / "attempt-1" / "attempt.json").write_text(json.dumps(dict(attempt, passed_sim=True)))
    assert kp.load_attempts(tmp_path / "probe", ops=["add"], probe_id="t") == ([attempt], 0)


def test_a_smoke_test_killed_by_the_watcher_blocks_the_probe(tmp_path, fake_kenv):
    def hangs(argv, **kw):
        return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
    with pytest.raises(kp.ProbeBlocked) as info:
        kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=hangs)
    text = str(info.value)
    assert "the smoke test hung" in text and "stalled" in text and "gozer-gatekeeper" in text


def test_both_gozer_commands_give_gozer_time_to_release_its_lease_and_the_simulator_check_does_not(tmp_path,
                                                                                                     fake_kenv):
    """C1. A watcher stop sends SIGTERM to the group and SIGKILL after the grace. `gozer run` releases its lease in a
    `finally` after its child exits, and the release resets the chip (41.7 s measured here). A grace shorter than that
    kills gozer in the middle of the reset and leaves the lease under a dead pid. The simulator check holds no lease, so
    it keeps the short default."""
    seen = []

    def recorder(argv, **kw):
        seen.append((list(argv), kw))
        if "--out" in argv and argv[0] == "gozer":
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET]}))
        elif "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps(NO_KERNEL))
        else:
            with open(kw["log_path"], "a") as f:
                f.write("PASSED!\n")
        return kp.Watched(0, None, False, 1.0)

    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=recorder)
    gozer = [kw for argv, kw in seen if argv[0] == "gozer"]
    assert len(gozer) == 2
    assert all(kw.get("term_grace_s", 0) >= 100 for kw in gozer), [kw.get("term_grace_s") for kw in gozer]
    assert kp.GOZER_TERM_GRACE_S >= 100

    seen.clear()
    adir2 = tmp_path / "probe2" / "add" / "attempt-1"
    kp.prepare_attempt(adir2, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir2 / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
    kp.authoritative_check(adir2, op="add", env=fake_kenv, run_watched=recorder, tt_lang=FACTS)
    (argv, kw), = seen
    assert argv[0] != "gozer" and kw.get("term_grace_s", 5.0) < 100


def test_a_hung_smoke_test_says_the_lease_is_normally_released(tmp_path, fake_kenv):
    """C1. After a watcher stop gozer normally releases its lease. Only when gozer itself had to be killed after the
    grace can the lease be left under a dead process, and the message says how to look."""
    def hangs(argv, **kw):
        return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
    with pytest.raises(kp.ProbeBlocked) as info:
        kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=hangs)
    text = str(info.value)
    assert "normally" in text and "gozer status" in text and "gozer-gatekeeper" in text
    assert "second interrupt" in text
    assert "Its lease may be held by a dead process" not in text


def test_a_smoke_test_that_exits_non_zero_is_unavailable_without_blocking(tmp_path, fake_kenv):
    def fails(argv, **kw):
        with open(kw["log_path"], "a") as f:
            f.write("PASSED!\nthen a crash\n")
        return kp.Watched(1, None, False, 3.0)
    ok, tail = kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=fails)
    assert ok is False and "then a crash" in tail


def test_the_skill_states_every_rule_the_gate_enforces():
    """The skill's rules are compared with the gate's own tables, so a rule added to the gate fails this test until the
    skill names it. Each REMEDIES rule needs its word; every name the gate refuses or allows by name must appear."""
    gate = kp._template("kernel_check")
    words = {"import": "import only", "name": "getattr", "dunder": "double underscores",
             "readback": "to_torch", "operation": "one function decorated with `@ttl.operation"}
    assert set(words) == set(gate.REMEDIES)
    flat = " ".join(kp.SKILL.read_text().split())
    for rule, word in words.items():
        assert word in flat, rule
    named = (gate.FORBIDDEN_NAMES | gate.CALL_ONLY_NAMES | gate.READBACK_NAMES | gate.ALLOWED_IMPORTS
             | gate.HARMLESS_FROM_MODULES | gate.HARMLESS_DUNDER_NAMES)
    for name in sorted(named):
        if gate._is_dunder(name) and name not in gate.HARMLESS_DUNDER_NAMES:
            continue                                      # covered by the double-underscore rule
        assert re.search(rf"`[^`]*\b{re.escape(name)}\b[^`]*`", flat), name
    # the star-import refusal (its remedy text lives in the gate's import branch)
    assert "import *" in gate.Path(gate.__file__).read_text() and "`from X import *`" in flat
    assert "`__name__` and `__all__`" in flat and "from __future__" in flat
    # the file rule belongs to the probe, which applies it when it takes kernel.py
    assert "The probe also refuses" in flat
    # The reference-file section (Task 14) took the skill from 46 to about 80 lines and the skeleton (Task 14b) to 93; the bound keeps it short enough for a small coder to read in one turn.
    assert len(kp.SKILL.read_text().splitlines()) <= 100


def test_the_report_header_says_the_wall_budget_can_overrun_by_one_tool_timeout(tmp_path, fake_kenv):
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="t", ops=["add"], tt_lang=FACTS)
    kp.write_report(tmp_path, [], header)
    md = (tmp_path / "kernel-probe-report.md").read_text()
    assert "checked between turns" in md and f"{int(kp.AGENT_TOOL_TIMEOUT_S)} s" in md
    assert kp.AGENT_TOOL_TIMEOUT_S == 420


def test_a_stale_verdict_from_an_earlier_run_is_not_read(tmp_path, fake_kenv):
    """The verdict directory is made fresh for each attempt: a passing result left by an earlier run is gone before
    the check runs, so a check that writes nothing is a harness_error."""
    vdir = kp.verdict_dir(tmp_path / "probe", "add", 1)
    vdir.mkdir(parents=True)
    (vdir / "check-authoritative.json").write_text(json.dumps({"passed": True, "category": None, "sets": []}))
    adir = tmp_path / "probe" / "add" / "attempt-1"

    def writes_nothing(argv, **kw):
        return kp.Watched(1, None, False, 0.1)
    with FakeModel(agent_does(lambda: (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text()))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=writes_nothing)
    assert attempt["passed_sim"] is False and attempt["category"] == "harness_error"


# ---- fix round 2: records of this probe only, capped regular-file reads, fresh attempt directories ---------------

import stat as _stat
import threading


def _record(probe_dir, op, n, probe_id, **extra):
    vdir = kp.verdict_dir(probe_dir, op, n)
    vdir.mkdir(parents=True, exist_ok=True)
    rec = {"op": op, "attempt": n, "passed_sim": True, "category": None, "hw": None, "turns": 1, "agent_wall_s": 1.0,
           "pcc_min": 0.9999, "first_line": "", "agent_check_untrusted": False,
           "provenance": {"probe_id": probe_id}, **extra}
    (vdir / "attempt.json").write_text(json.dumps(rec))
    return rec


def test_a_stale_record_from_a_longer_earlier_run_is_not_in_the_new_report(tmp_path, fake_kenv):
    probe = tmp_path / "probe"
    _record(probe, "add", 5, "old-run")                    # an earlier 5-attempt run left add-5 behind
    with FakeModel(lambda r: final("nothing")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, probe_id="new-run", run_watched=check_writer(NO_KERNEL))
    got, ignored = kp.load_attempts(probe, ops=["add"], probe_id="new-run")
    assert got == [attempt] and ignored == 1
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="new-run", ops=["add"], tt_lang=FACTS,
                              ignored_records=ignored)
    data = kp.write_report(probe, got, header)
    assert data["first_pass"]["add"] == {"sim": None, "hw": None}
    md = (probe / "kernel-probe-report.md").read_text()
    assert "1 unrelated or unreadable records ignored" in md and "| 5 |" not in md


def test_planted_and_malformed_records_are_ignored(tmp_path):
    probe = tmp_path / "probe"
    good = _record(probe, "add", 1, "p")
    _record(probe, "add", 99, "someone-else")             # another probe id
    _record(probe, "add", 7, "p", attempt=8)              # the number does not match its directory
    _record(probe, "add", 3, "p", attempt="3")            # not an int: must not make the sort raise
    _record(probe, "add", 4, "p", attempt=True)           # a bool is not an attempt number
    _record(probe, "rmsnorm", 1, "p")                     # an op this probe did not run
    (probe / ".verdict" / "add-x").mkdir()
    (probe / ".verdict" / "add-0").mkdir()
    (probe / ".verdict" / "notes").mkdir()
    # links whose targets hold records that would otherwise match their directory names
    outside = tmp_path / "outside"
    _record(outside, "add", 6, "p")
    _record(outside, "add", 2, "p")
    linked = probe / ".verdict" / "add-6"
    linked.mkdir()
    (linked / "attempt.json").symlink_to(kp.verdict_dir(outside, "add", 6) / "attempt.json")
    (probe / ".verdict" / "add-2").symlink_to(kp.verdict_dir(outside, "add", 2), target_is_directory=True)
    got, ignored = kp.load_attempts(probe, ops=["add"], probe_id="p")
    assert got == [good] and ignored == 10                  # every entry above but add-1


def test_a_bool_attempt_number_is_ignored(tmp_path):
    """True == 1 in Python, so the number check alone would accept it in add-1."""
    _record(tmp_path / "probe", "add", 1, "p", attempt=True)
    assert kp.load_attempts(tmp_path / "probe", ops=["add"], probe_id="p") == ([], 1)


def test_a_reused_attempt_directory_starts_empty(tmp_path, fake_kenv):
    adir = tmp_path / "add" / "attempt-1"
    adir.mkdir(parents=True)
    (adir / "kernel.py").write_text("# left by an earlier run\n")
    (adir / "check.json").write_text(json.dumps({"passed": True}))
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    assert not (adir / "kernel.py").exists() and not (adir / "check.json").exists()
    assert (adir / "kernel_spec.json").is_file()


def test_an_attempt_directory_that_is_a_link_is_replaced_and_its_target_left_alone(tmp_path, fake_kenv):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep")
    (tmp_path / "add").mkdir()
    (tmp_path / "add" / "attempt-1").symlink_to(elsewhere, target_is_directory=True)
    kp.prepare_attempt(tmp_path / "add" / "attempt-1", op="add", env=fake_kenv, tt_lang=FACTS)
    assert sorted(p.name for p in elsewhere.iterdir()) == ["keep.txt"]
    assert not (tmp_path / "add" / "attempt-1").is_symlink()


def _in_thread(fn, timeout=20):
    """Run fn in a thread; fail (without hanging the suite) if it has not returned within `timeout` seconds."""
    box = {}

    def target():
        box["value"] = fn()
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    return th.is_alive(), box.get("value")


def _unblock(fifo):
    """Open the FIFO's write end once, so a reader blocked on it gets EOF (cleanup after a failing run)."""
    try:
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
    except OSError:
        pass


def test_read_regular_refuses_links_and_fifos_and_caps_its_read(tmp_path):
    f = tmp_path / "f.txt"
    f.write_bytes(b"x" * 10)
    assert kp._read_regular(f, 100) == b"x" * 10
    assert kp._read_regular(f, 4) == b"xxxxx"             # limit + 1 bytes: the caller sees it is too large
    link = tmp_path / "link"
    link.symlink_to(f)
    assert kp._read_regular(link, 100) is None
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    try:
        hung, value = _in_thread(lambda: kp._read_regular(fifo, 100), timeout=10)
    finally:
        _unblock(fifo)
    assert not hung and value is None


def test_agent_check_json_that_is_a_fifo_or_a_link_is_untrusted_and_never_blocks(tmp_path):
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"passed": False, "pad": "x" * (2 * 1024 * 1024)}))
    cases = {}
    fifo = tmp_path / "fifo-check.json"
    os.mkfifo(fifo)
    try:
        cases["fifo"] = _in_thread(lambda: kp._read_agent_check(fifo), timeout=10)
    finally:
        _unblock(fifo)
    for name, target in (("big", big),):
        link = tmp_path / f"{name}-check.json"
        link.symlink_to(target)
        cases[name] = _in_thread(lambda link=link: kp._read_agent_check(link), timeout=10)
    oversized = tmp_path / "oversized-check.json"
    oversized.write_bytes(big.read_bytes())
    cases["oversized"] = _in_thread(lambda: kp._read_agent_check(oversized), timeout=10)
    for name, (hung, value) in cases.items():
        assert not hung, name
        assert value == "unreadable", (name, value)
        assert kp.agent_check_untrusted(value, {"passed": False}, "add") is True
    assert kp._read_agent_check(tmp_path / "missing.json") is None


def test_agent_check_json_linked_to_dev_zero_is_refused_without_reading_it(tmp_path):
    """Run in a child process with a 2 GiB address-space limit and a timeout: a reader that follows the link and reads
    to the end would exhaust memory, and the limit turns that into a MemoryError in the child alone."""
    import subprocess
    import sys
    link = tmp_path / "check.json"
    link.symlink_to("/dev/zero")
    code = ("import resource, sys; resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30)); "
            f"sys.path.insert(0, {str(REPO)!r}); from pathlib import Path; from orchard import kernelprobe as kp; "
            f"print(kp._read_agent_check(Path({str(link)!r})))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip() == "unreadable"


def test_a_fifo_check_json_left_by_the_agent_does_not_block_the_attempt(tmp_path, fake_kenv):
    adir = tmp_path / "probe" / "add" / "attempt-1"
    with FakeModel(agent_does(lambda: os.mkfifo(adir / "check.json"))) as server:
        try:
            hung, attempt = _in_thread(lambda: run_attempt(tmp_path, fake_kenv, server,
                                                           run_watched=check_writer(NO_KERNEL)), timeout=30)
        finally:
            _unblock(adir / "check.json")
    assert not hung and attempt["agent_check_untrusted"] is True


def _lstat_says_regular(monkeypatch, path):
    """os.lstat reports a regular file for `path`, as if it changed between the lstat and the open."""
    real = os.lstat

    def fake(p, *a, **kw):
        st = real(p, *a, **kw)
        if Path(p) == path:
            return os.stat_result((_stat.S_IFREG | 0o644,) + tuple(st)[1:6] + (10,) + tuple(st)[7:])
        return st
    monkeypatch.setattr(kp.os, "lstat", fake)


def test_copy_kernel_refuses_a_link_even_when_lstat_was_fooled(tmp_path, monkeypatch):
    """The read-side layer alone: the open does not follow the link."""
    secret = tmp_path / "secret.py"
    secret.write_text("import ttl\n")
    adir = tmp_path / "a"
    adir.mkdir()
    (adir / "kernel.py").symlink_to(secret)
    clean = tmp_path / "clean"
    clean.mkdir()
    _lstat_says_regular(monkeypatch, adir / "kernel.py")
    category, sha, line = kp.copy_kernel(adir, clean)
    assert category == "forbidden_code" and sha is None and not (clean / "kernel.py").exists()


def test_copy_kernel_refuses_a_fifo_even_when_lstat_was_fooled(tmp_path, monkeypatch):
    """The read-side layer alone: fstat on the open file says it is not a regular file."""
    adir = tmp_path / "a"
    adir.mkdir()
    os.mkfifo(adir / "kernel.py")
    clean = tmp_path / "clean"
    clean.mkdir()
    _lstat_says_regular(monkeypatch, adir / "kernel.py")
    try:
        hung, got = _in_thread(lambda: kp.copy_kernel(adir, clean), timeout=10)
    finally:
        _unblock(adir / "kernel.py")
    assert not hung and got[0] == "forbidden_code" and not (clean / "kernel.py").exists()


def test_a_hung_smoke_test_names_its_last_lines_and_the_report_records_the_hang(tmp_path, fake_kenv):
    def hangs(argv, **kw):
        with open(kw["log_path"], "a") as f:
            f.write("opening device 0\nwaiting for ARC\n")
        return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    with pytest.raises(kp.ProbeBlocked) as info:
        kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=hangs)
    assert "waiting for ARC" in str(info.value)
    hw = info.value.hw
    assert hw["available"] is False and hw["hung"] is True and "waiting for ARC" in hw["smoke_tail"]
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="t", ops=["add"], hw=hw, tt_lang=FACTS)
    kp.write_report(tmp_path, [attempt], header)
    md = (tmp_path / "kernel-probe-report.md").read_text()
    assert "hw: blocked" in md and "the smoke test hung" in md and "waiting for ARC" in md


# ---- carry-over fixes (Task 10) ---------------------------------------------------------------------------------


def test_an_attempt_number_must_be_plain_ascii_digits(tmp_path):
    """str.isdigit accepts superscripts and other scripts' digits. `add-²` made int() raise and `add-٥` read as 5."""
    probe = tmp_path / "probe"
    good = _record(probe, "add", 1, "p")
    _record(probe, "add", "²", "p", attempt=2)
    _record(probe, "add", "٥", "p", attempt=5)
    _record(probe, "add", "05", "p", attempt=5)
    _record(probe, "add", "+5", "p", attempt=5)
    assert kp.load_attempts(probe, ops=["add"], probe_id="p") == ([good], 4)


def test_an_op_directory_that_is_a_link_is_replaced_and_its_target_left_alone(tmp_path, fake_kenv):
    """DIR/<op> itself is a link: the attempt is prepared in a real directory and nothing is written through it."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "attempt-1").mkdir(parents=True)
    (elsewhere / "attempt-1" / "keep.txt").write_text("keep")
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "add").symlink_to(elsewhere, target_is_directory=True)
    kp.prepare_attempt(probe / "add" / "attempt-1", op="add", env=fake_kenv, tt_lang=FACTS)
    assert not (probe / "add").is_symlink() and (probe / "add" / "attempt-1" / "kernel_spec.json").is_file()
    assert sorted(p.name for p in elsewhere.iterdir()) == ["attempt-1"]
    assert sorted(p.name for p in (elsewhere / "attempt-1").iterdir()) == ["keep.txt"]


# ---- the command: kernelprobe.main (Task 10) --------------------------------------------------------------------

import argparse
from types import SimpleNamespace


def probe_args(tmp_path, **kw):
    args = dict(op="add", attempts=1, phase="sim", out=str(tmp_path / "out"), endpoint="http://127.0.0.1:9/v1",
                model="fake", src_dir=None, venv_dir=None, shared_python="/shared/python")
    args.update(kw)
    return argparse.Namespace(**args)


@pytest.fixture
def env_ready(monkeypatch):
    """ttlang_env.ensure(check_only=True) answers all OK without running the simulator; the calls are recorded."""
    calls = []

    def ensure(env, *, check_only=False, say=print):
        calls.append(check_only)
        return [ttlang_env.Result("verify", ttlang_env.OK, "faked")]
    monkeypatch.setattr(kp.ttlang_env, "ensure", ensure)
    return calls


def no_run(argv, **kw):
    raise AssertionError(f"nothing may run: {argv}")


def passing_check(gozer_calls, *, gozer=None):
    """A fake run_watched: an authoritative check writes a passing result, a gozer command is recorded and answered
    by `gozer` (default: the smoke test logs PASSED!, a kernel run writes a passing hardware result)."""
    def fake(argv, **kw):
        if argv[0] == "gozer":
            gozer_calls.append(list(argv))
            if gozer is not None:
                return gozer(argv, **kw)
            if "--out" in argv:
                Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET]}))
            else:
                with open(kw["log_path"], "a") as f:
                    f.write("PASSED!\n")
            return kp.Watched(0, None, False, 1.0)
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(
            {"passed": True, "category": None, "first_line": "", "sets": [dict(GOOD_HW_SET, pcc=0.9999, slope=1.0)]}))
        return kp.Watched(0, None, False, 0.1)
    return fake


def writes_good_kernel_once():
    """Attempt 1 writes the known-good kernel and stops; every later attempt writes nothing."""
    return scripted(call("write_file", path="kernel.py", content=(KERNELS / "add_good.py").read_text()), final("done"))


def test_main_refuses_an_unknown_op_and_runs_nothing(tmp_path, fake_kenv, env_ready, capsys):
    code = kp.main(probe_args(tmp_path, op="conv2d"), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 2 and env_ready == [] and not (tmp_path / "out").exists()
    assert "unknown op" in capsys.readouterr().err


def test_main_refuses_when_the_endpoint_does_not_answer(tmp_path, fake_kenv, env_ready, capsys):
    asked = []
    code = kp.main(probe_args(tmp_path), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: asked.append((e, m)) or False, run_watched_fn=no_run)
    assert code == 2 and asked == [("http://127.0.0.1:9/v1", "fake")]
    assert not (tmp_path / "out").exists()                 # no attempt directory, no report
    err = capsys.readouterr().err
    assert "starts no server" in err and "tt-orchard bringup" in err


def test_main_refuses_when_the_environment_check_fails(tmp_path, fake_kenv, monkeypatch, capsys):
    for status in (ttlang_env.FAIL, ttlang_env.TODO):
        seen = []
        monkeypatch.setattr(kp.ttlang_env, "ensure", lambda env, *, check_only=False, say=print: seen.append(
            check_only) or [ttlang_env.Result("clone", ttlang_env.OK, ""), ttlang_env.Result("wheel", status, "x")])
        asked = []
        code = kp.main(probe_args(tmp_path), env=fake_kenv, out=io.StringIO(),
                       probe=lambda e, m: asked.append(e) or True, run_watched_fn=no_run)
        assert code == 2 and seen == [True] and asked == [] and not (tmp_path / "out").exists()
        assert "tt-orchard kernel-env" in capsys.readouterr().err


def test_main_refuses_with_no_coder_endpoint_known(tmp_path, fake_kenv, env_ready, capsys):
    code = kp.main(probe_args(tmp_path, endpoint=None, model=None), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 2 and not (tmp_path / "out").exists()
    err = capsys.readouterr().err
    assert "--endpoint" in err and "--model" in err


def test_main_takes_the_coder_from_the_tier_stage_2_uses_and_the_flags_win(tmp_path, fake_kenv, env_ready):
    cfg = SimpleNamespace(stages={2: {"run": "small", "diagnose": "large"}}, escalation="large",
                          tiers={"small": {"endpoint": "http://127.0.0.1:8011/v1", "model": "coder-next",
                                           "placement": "chip"},
                                 "large": {"endpoint": "http://127.0.0.1:8000/v1", "model": "big", "placement": "chip"}})
    asked = []

    def probe(e, m):
        asked.append((e, m))
        return e != "http://127.0.0.1:8011/v1"           # the tier's server is down: nothing else is contacted
    code = kp.main(probe_args(tmp_path, endpoint=None, model=None), env=fake_kenv, out=io.StringIO(), probe=probe,
                   run_watched_fn=no_run, tier_config=cfg)
    assert code == 2 and asked == [("http://127.0.0.1:8011/v1", "coder-next")]
    asked.clear()
    with FakeModel(lambda r: final("nothing")) as server:
        code = kp.main(probe_args(tmp_path, endpoint=server.endpoint, model=None), env=fake_kenv, out=io.StringIO(),
                       probe=probe, run_watched_fn=no_run, tier_config=cfg)
    assert code == 0 and asked == [(server.endpoint, "coder-next")]


def test_main_runs_n_attempts_writes_the_report_and_exits_0_even_when_every_attempt_fails(tmp_path, fake_kenv,
                                                                                          env_ready):
    out = io.StringIO()
    with FakeModel(lambda r: final("I will not write a kernel")) as server:
        code = kp.main(probe_args(tmp_path, attempts=3, endpoint=server.endpoint), env=fake_kenv, out=out,
                       probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 0 and env_ready == [True]
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert [a["attempt"] for a in data["attempts"]] == [1, 2, 3]
    assert [a["category"] for a in data["attempts"]] == ["no_kernel"] * 3
    assert data["header"]["hw"] is None and data["header"]["ignored_records"] == 0
    assert len({a["provenance"]["probe_id"] for a in data["attempts"]}) == 1
    assert (tmp_path / "out" / "kernel-probe-report.md").is_file()
    lines = [l for l in out.getvalue().splitlines() if l.startswith("add ")]
    assert len(lines) == 3 and all("no_kernel" in l and "kernel.py does not exist" in l and "turns" in l
                                   for l in lines)


def test_each_run_gets_a_fresh_probe_id_and_its_report_holds_only_its_own_attempts(tmp_path, fake_kenv, env_ready):
    with FakeModel(lambda r: final("nothing")) as server:
        kp.main(probe_args(tmp_path, attempts=3, endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                probe=lambda e, m: True, run_watched_fn=no_run)
        first = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
        kp.main(probe_args(tmp_path, attempts=1, endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                probe=lambda e, m: True, run_watched_fn=no_run)
    second = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert first["header"]["probe_id"] != second["header"]["probe_id"]
    assert [a["attempt"] for a in second["attempts"]] == [1]


def test_phase_sim_never_calls_gozer(tmp_path, fake_kenv, env_ready):
    """The first attempt passes the (faked) simulator check, so a hardware phase would have work to do."""
    gozer_calls = []

    def tripwire(argv, **kw):
        assert argv[0] != "gozer", f"the simulator phase ran gozer: {argv}"
        return passing_check(gozer_calls)(argv, **kw)
    with FakeModel(writes_good_kernel_once()) as server:
        code = kp.main(probe_args(tmp_path, attempts=2, endpoint=server.endpoint, phase="sim"), env=fake_kenv,
                       out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=tripwire)
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert code == 0 and gozer_calls == [] and data["attempts"][0]["passed_sim"] is True
    assert data["header"]["hw"] is None and all(a["hw"] is None for a in data["attempts"])


@needs_sim
def test_phase_hw_runs_the_smoke_test_once_and_hardware_only_for_a_sim_pass(tmp_path, kenv, env_ready):
    """Two attempts on the real simulator: the first writes the known-good kernel, the second writes nothing. Gozer
    calls go to a fake; everything else runs for real."""
    gozer_calls = []

    def fake(argv, **kw):
        if argv[0] == "gozer":
            return passing_check(gozer_calls)(argv, **kw)
        return kp.run_watched(argv, **kw)
    out = io.StringIO()
    with FakeModel(writes_good_kernel_once()) as server:
        code = kp.main(probe_args(tmp_path, attempts=2, endpoint=server.endpoint, phase="hw"), env=kenv, out=out,
                       probe=lambda e, m: True, run_watched_fn=fake)
    assert code == 0
    smoke = [c for c in gozer_calls if "--out" not in c]
    hw = [c for c in gozer_calls if "--out" in c]
    assert len(smoke) == 1 and len(hw) == 1
    assert smoke[0][smoke[0].index("--") + 1] == "/shared/python"
    assert Path(hw[0][hw[0].index("--kernel") + 1]) == kp.verdict_dir(tmp_path / "out", "add", 1) / kp.CLEAN_DIR / "kernel.py"
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    a1, a2 = data["attempts"]
    assert a1["passed_sim"] is True and a1["hw"]["passed"] is True
    assert a2["category"] == "no_kernel" and a2["hw"] is None
    assert data["header"]["hw"]["available"] is True and data["first_pass"]["add"] == {"sim": 1, "hw": 1}


def test_a_hardware_hang_exits_5_and_the_report_lists_what_ran(tmp_path, fake_kenv, env_ready):
    gozer_calls = []

    def kernel_run_hangs(argv, **kw):
        if "--out" in argv:
            return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
        with open(kw["log_path"], "a") as f:
            f.write("PASSED!\n")
        return kp.Watched(0, None, False, 1.0)
    with FakeModel(scripted(call("write_file", path="kernel.py", content=(KERNELS / "add_good.py").read_text()),
                            final("done"),
                            call("write_file", path="kernel.py", content=(KERNELS / "add_good.py").read_text()),
                            final("done"))) as server:
        code = kp.main(probe_args(tmp_path, attempts=3, endpoint=server.endpoint, phase="hw"), env=fake_kenv,
                       out=io.StringIO(), probe=lambda e, m: True,
                       run_watched_fn=passing_check(gozer_calls, gozer=kernel_run_hangs))
    assert code == 5
    assert len(gozer_calls) == 2                            # the smoke test and attempt 1's run, which hung
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert [a["attempt"] for a in data["attempts"]] == [1, 2, 3]
    assert data["attempts"][0]["hw"]["category"] == "hang" and data["attempts"][1]["hw"] is None
    assert "add attempt 1" in data["header"]["blocked"]
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "The probe stopped" in md and "add attempt 1" in md and "| hang |" in md


def test_a_hung_smoke_test_exits_5_and_the_report_says_hw_blocked(tmp_path, fake_kenv, env_ready):
    gozer_calls = []

    def smoke_hangs(argv, **kw):
        return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
    with FakeModel(writes_good_kernel_once()) as server:
        code = kp.main(probe_args(tmp_path, attempts=1, endpoint=server.endpoint, phase="hw"), env=fake_kenv,
                       out=io.StringIO(), probe=lambda e, m: True,
                       run_watched_fn=passing_check(gozer_calls, gozer=smoke_hangs))
    assert code == 5 and len(gozer_calls) == 1
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert data["header"]["hw"]["hung"] is True and data["attempts"][0]["hw"] is None
    assert "hw: blocked" in (tmp_path / "out" / "kernel-probe-report.md").read_text()


# ---- fix round 1: partial results survive a crash or an interrupt -----------------------------------------------

import signal
import subprocess
import sys
import time as _time


def canned(op, n, passed=False):
    """A finished attempt record, as run_attempt returns it, for tests that replace run_attempt."""
    return {"op": op, "attempt": n, "status": "done", "turns": 2, "agent_wall_s": 1.5, "passed_sim": passed,
            "category": None if passed else "no_kernel", "first_line": "" if passed else "kernel.py does not exist",
            "pcc_min": None, "sets": [], "violations": [], "hw": None, "agent_check_untrusted": False,
            "provenance": kp.provenance(kernel_sha256=None, tt_lang=FACTS, coder_model="fake", probe_id="t",
                                        simulator_pcc_min=None)}


def fake_attempts(monkeypatch, on_n):
    """Replace run_attempt: attempt n calls on_n(n) first (it may raise), then returns a canned record."""
    seen = []

    def run_attempt(*, op, n, **kw):
        seen.append(n)
        on_n(n)
        return canned(op, n)
    monkeypatch.setattr(kp, "run_attempt", run_attempt)
    return seen


def test_the_exit_codes_match_the_cli():
    from orchard import cli
    assert (kp.EXIT_OK, kp.EXIT_REFUSED, kp.EXIT_ERROR, kp.EXIT_ABORTED, kp.EXIT_BLOCKED) == \
        (cli.EXIT_OK, cli.EXIT_REFUSED, cli.EXIT_ERROR, cli.EXIT_ABORTED, cli.EXIT_BLOCKED)
    # m1: the module docstring names every exit code the command uses, each with its meaning
    doc = " ".join(kp.__doc__.split())
    for code, words in ((kp.EXIT_OK, "a report was written"), (kp.EXIT_REFUSED, "refusal"),
                        (kp.EXIT_ERROR, "crash"), (kp.EXIT_ABORTED, "interrupt"), (kp.EXIT_BLOCKED, "hung")):
        assert re.search(rf"\b{code}\b[^.;]*{words}", doc), (code, words)


def test_an_attempt_that_crashes_keeps_the_earlier_attempts_in_the_report_and_exits_3(tmp_path, fake_kenv, env_ready,
                                                                                       monkeypatch):
    def on_n(n):
        if n == 2:
            raise OSError("disk full while preparing the attempt")
    seen = fake_attempts(monkeypatch, on_n)
    code = kp.main(probe_args(tmp_path, attempts=4, phase="hw"), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 3 and seen == [1, 2]                     # the loop stopped, and no hardware phase ran (no_run)
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert [a["attempt"] for a in data["attempts"]] == [1]
    assert "attempt 2 crashed: OSError: disk full" in data["header"]["stopped"] and data["header"]["hw"] is None
    assert "The probe stopped: attempt 2 crashed" in (tmp_path / "out" / "kernel-probe-report.md").read_text()


def test_an_interrupt_during_an_attempt_writes_the_report_and_exits_4(tmp_path, fake_kenv, env_ready, monkeypatch):
    def on_n(n):
        if n == 2:
            raise KeyboardInterrupt
    fake_attempts(monkeypatch, on_n)
    code = kp.main(probe_args(tmp_path, attempts=3, phase="hw"), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 4
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert [a["attempt"] for a in data["attempts"]] == [1] and "interrupted" in data["header"]["stopped"]


def test_sigterm_during_an_attempt_is_an_interrupt_and_the_handler_is_restored(tmp_path, fake_kenv, env_ready,
                                                                                monkeypatch):
    before = signal.getsignal(signal.SIGTERM)

    def on_n(n):
        if n != 2:
            return
        # Only signal this process, and only while main's own handler is installed: with the default action a
        # SIGTERM would end pytest itself.
        handler = signal.getsignal(signal.SIGTERM)
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN, before), "main installed no SIGTERM handler"
        t = threading.Thread(target=os.kill, args=(os.getpid(), signal.SIGTERM))
        t.start()
        t.join()
        deadline = _time.monotonic() + 5
        while _time.monotonic() < deadline:                 # the handler runs on the main thread between bytecodes
            _time.sleep(0.01)
        raise AssertionError("the SIGTERM handler did not interrupt the attempt")
    fake_attempts(monkeypatch, on_n)
    code = kp.main(probe_args(tmp_path, attempts=3), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 4 and signal.getsignal(signal.SIGTERM) is before
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert [a["attempt"] for a in data["attempts"]] == [1] and "interrupted" in data["header"]["stopped"]


def test_a_report_that_cannot_be_written_prints_the_records_and_exits_3(tmp_path, fake_kenv, env_ready, monkeypatch,
                                                                        capsys):
    fake_attempts(monkeypatch, lambda n: None)

    def broken(*a, **kw):
        raise OSError("read-only file system")
    monkeypatch.setattr(kp, "write_report", broken)
    code = kp.main(probe_args(tmp_path, attempts=2), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 3
    err = capsys.readouterr().err
    assert "read-only file system" in err and '"attempt": 2' in err


def two_good_kernels():
    good = (KERNELS / "add_good.py").read_text()
    return scripted(call("write_file", path="kernel.py", content=good), final("done"),
                    call("write_file", path="kernel.py", content=good), final("done"))


def test_a_crash_in_the_hardware_phase_keeps_the_first_hardware_result(tmp_path, fake_kenv, env_ready):
    gozer_calls = []

    def second_kernel_run_crashes(argv, **kw):
        if "--out" in argv and len([c for c in gozer_calls if "--out" in c]) == 2:
            raise RuntimeError("the runner broke")
        return passing_check([])(argv, **kw)
    with FakeModel(two_good_kernels()) as server:
        code = kp.main(probe_args(tmp_path, attempts=2, endpoint=server.endpoint, phase="hw"), env=fake_kenv,
                       out=io.StringIO(), probe=lambda e, m: True,
                       run_watched_fn=passing_check(gozer_calls, gozer=second_kernel_run_crashes))
    assert code == 3
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    a1, a2 = data["attempts"]
    assert a1["hw"]["passed"] is True and a2["hw"] is None
    assert "hardware phase crashed: RuntimeError: the runner broke" in data["header"]["stopped"]
    assert data["header"]["hw"]["smoke_tail"] == ["PASSED!"]
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "1 of 1 attempts that ran on hardware passed; the probe stopped before the rest" in md


# ---- fix round 1: the minors ------------------------------------------------------------------------------------


def test_importing_the_cli_does_not_import_the_kernel_probe():
    # _parser() is what every `tt-orchard` invocation builds, so it is built here too.
    code = "import sys, orchard.cli; orchard.cli._parser(); print('orchard.kernelprobe' in sys.modules)"
    got = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert got.returncode == 0 and got.stdout.strip() == "False", got.stderr
    assert kp.OPS == kp.defaults.KERNEL_PROBE_OPS


def test_after_a_kernel_hang_the_report_counts_only_attempts_that_ran_on_hardware(tmp_path, fake_kenv, env_ready):
    def kernel_run_hangs(argv, **kw):
        if "--out" in argv:
            with open(kw["log_path"], "a") as f:
                f.write("PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\n")
            return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
        with open(kw["log_path"], "a") as f:
            f.write("PASSED!\n")
        return kp.Watched(0, None, False, 1.0)
    with FakeModel(two_good_kernels()) as server:
        code = kp.main(probe_args(tmp_path, attempts=2, endpoint=server.endpoint, phase="hw"), env=fake_kenv,
                       out=io.StringIO(), probe=lambda e, m: True,
                       run_watched_fn=passing_check([], gozer=kernel_run_hangs))
    assert code == 5
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert data["header"]["hw"]["available"] is True and data["header"]["hw"]["smoke_tail"] == ["PASSED!"]
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "0 of 1 attempts that ran on hardware passed; the probe stopped before the rest" in md
    assert "none of 2 attempts passed on hardware" not in md


def test_phase_hw_smoke_once_and_hardware_only_for_a_sim_pass_without_the_simulator(tmp_path, fake_kenv, env_ready):
    """The always-running variant of the needs_sim test: the simulator verdict is faked."""
    gozer_calls = []
    with FakeModel(writes_good_kernel_once()) as server:
        code = kp.main(probe_args(tmp_path, attempts=2, endpoint=server.endpoint, phase="hw"), env=fake_kenv,
                       out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=passing_check(gozer_calls))
    assert code == 0
    assert len([c for c in gozer_calls if "--out" not in c]) == 1 and len([c for c in gozer_calls if "--out" in c]) == 1
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert data["attempts"][0]["hw"]["passed"] is True and data["attempts"][1]["hw"] is None


def test_a_failed_smoke_test_in_main_says_hw_unavailable_runs_no_kernel_and_exits_0(tmp_path, fake_kenv, env_ready):
    gozer_calls = []

    def smoke_fails(argv, **kw):
        with open(kw["log_path"], "a") as f:
            f.write("TT_FATAL: no device\n")
        return kp.Watched(1, None, False, 2.0)
    with FakeModel(writes_good_kernel_once()) as server:
        code = kp.main(probe_args(tmp_path, attempts=1, endpoint=server.endpoint, phase="hw"), env=fake_kenv,
                       out=io.StringIO(), probe=lambda e, m: True,
                       run_watched_fn=passing_check(gozer_calls, gozer=smoke_fails))
    assert code == 0 and len(gozer_calls) == 1
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert data["header"]["hw"]["available"] is False and data["attempts"][0]["hw"] is None
    assert "hw: unavailable" in (tmp_path / "out" / "kernel-probe-report.md").read_text()


def test_credentials_in_the_endpoint_are_never_printed(tmp_path, fake_kenv, env_ready, monkeypatch, capsys):
    url = "http://user:secret@127.0.0.1:1/v1"
    out = io.StringIO()
    code = kp.main(probe_args(tmp_path, endpoint=url), env=fake_kenv, out=out, probe=lambda e, m: False,
                   run_watched_fn=no_run)
    err = capsys.readouterr().err
    assert code == 2 and "127.0.0.1:1" in err and "secret" not in err and "user:" not in err
    fake_attempts(monkeypatch, lambda n: None)
    asked = []
    code = kp.main(probe_args(tmp_path, endpoint=url), env=fake_kenv, out=out,
                   probe=lambda e, m: asked.append(e) or True, run_watched_fn=no_run)
    assert code == 0 and asked == [url]                     # the server itself is still asked with the given URL
    text = out.getvalue() + capsys.readouterr().err
    text += (tmp_path / "out" / "kernel-probe-report.md").read_text()
    text += (tmp_path / "out" / "kernel-probe-report.json").read_text()
    assert "127.0.0.1:1" in out.getvalue() and "secret" not in text


# ---- stray agent processes (I5) -----------------------------------------------------------------------------------
#
# Every process these tests signal is a child the test started itself, with start_new_session=True and an
# ORCHARD_RUN_DIR under the test's own tmp directory. The sweep's kill functions are wrapped so that a signal to any
# other pid is recorded and never sent (each test asserts there was none), and every child is stopped in a finally.

import signal as _signal
import subprocess as _subprocess
import sys as _sys
import time as _time

from orchard.agent import Outcome, agent_env


def _child(run_dir, *, extra_env=None):
    env = dict(os.environ)
    env["ORCHARD_RUN_DIR"] = str(Path(run_dir).resolve())
    env.update(extra_env or {})
    return _subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(60)"], env=env, start_new_session=True,
                             stdin=_subprocess.DEVNULL, stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL)


def _wait_for_env(pid, value, timeout=10.0):
    """Wait until the child's /proc environ is readable (an exec has happened)."""
    want = ("ORCHARD_RUN_DIR=" + value).encode()
    end = _time.monotonic() + timeout
    while _time.monotonic() < end:
        try:
            if want in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                return
        except OSError:
            pass
        _time.sleep(0.02)
    raise AssertionError(f"child {pid} never showed ORCHARD_RUN_DIR={value}")


def _guarded_kills(allowed, sent: list, violations: list):
    """kill and killpg stand-ins that pass a signal on only to a pid (or a group led by a pid) that `allowed` accepts
    (a set of pids, or a predicate). Any other call is recorded in `violations` and NOT sent. The sweep catches
    exceptions, so a raising guard would be swallowed; each test asserts `violations == []` at its end instead."""
    ok = allowed if callable(allowed) else (lambda pid: pid in allowed)

    def kill(pid, sig):
        sent.append(("kill", pid, sig))
        if not ok(pid):
            violations.append(("kill", pid, sig))
            return
        os.kill(pid, sig)

    def killpg(pgid, sig):
        sent.append(("killpg", pgid, sig))
        if not ok(pgid):
            violations.append(("killpg", pgid, sig))
            return
        os.killpg(pgid, sig)
    return kill, killpg


def test_the_guard_records_a_signal_the_sweep_would_swallow(tmp_path):
    """S3. A stray this test did not allow: the sweep swallows exceptions, so the guard must record the call (and not
    send it) for the test to see it."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    stray = _child(adir)
    sent, violations = [], []
    try:
        _wait_for_env(stray.pid, str(adir.resolve()))
        kill, killpg = _guarded_kills(set(), sent, violations)
        kp.sweep_strays(adir, kill=kill, killpg=killpg, grace_s=0.1)
        assert violations and all(pid == stray.pid for _, pid, _ in violations)
        assert stray.poll() is None                               # nothing was sent
    finally:
        _stop(stray)


def _stop(*procs):
    """Popen.kill skips a child that has already been reaped, so a freed pid is never signalled."""
    for p in procs:
        p.kill()
        try:
            p.wait(timeout=5)
        except Exception:
            pass


def test_the_sweep_stops_a_process_with_this_attempt_directory_and_leaves_a_neighbour_alone(tmp_path):
    """The neighbour's directory starts with the attempt's directory (attempt-1 and attempt-10), so a prefix match
    would stop it too. Only the exact directory counts."""
    adir = tmp_path / "probe" / "add" / "attempt-1"
    other = tmp_path / "probe" / "add" / "attempt-10"
    adir.mkdir(parents=True)
    other.mkdir(parents=True)
    stray, neighbour = _child(adir), _child(other)
    sent = []
    try:
        _wait_for_env(stray.pid, str(adir.resolve()))
        _wait_for_env(neighbour.pid, str(other.resolve()))
        violations = []
        kill, killpg = _guarded_kills({stray.pid}, sent, violations)
        swept = kp.sweep_strays(adir, kill=kill, killpg=killpg)
        assert [s["pid"] for s in swept] == [stray.pid] and "time.sleep" in swept[0]["cmdline"]
        assert stray.wait(timeout=5) == -_signal.SIGTERM
        assert neighbour.poll() is None
        assert all(pid == stray.pid for _, pid, _ in sent) and violations == []
    finally:
        _stop(stray, neighbour)


def test_the_sweep_kills_a_process_that_ignores_sigterm(tmp_path):
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    env = dict(os.environ, ORCHARD_RUN_DIR=str(adir.resolve()))
    stubborn = _subprocess.Popen(
        [_sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                                "print('ready', flush=True); time.sleep(60)"],
        env=env, start_new_session=True, stdout=_subprocess.PIPE, stderr=_subprocess.DEVNULL)
    sent = []
    try:
        assert stubborn.stdout.readline().strip() == b"ready"
        violations = []
        kill, killpg = _guarded_kills({stubborn.pid}, sent, violations)
        swept = kp.sweep_strays(adir, kill=kill, killpg=killpg, grace_s=0.5)
        assert [s["pid"] for s in swept] == [stubborn.pid]
        assert stubborn.wait(timeout=5) == -_signal.SIGKILL
        assert [sig for _, _, sig in sent] == [_signal.SIGTERM, _signal.SIGKILL] and violations == []
    finally:
        _stop(stubborn)


def test_the_sweep_never_signals_this_process_or_its_parent(tmp_path, monkeypatch):
    """The scan is patched to name our own pid and our parent's; a recorder stands in for every kill function, so no
    real signal can be sent whatever the sweep decides."""
    sent = []
    monkeypatch.setattr(kp, "scan_strays", lambda run_dir, **kw: [os.getpid(), os.getppid(), 1, 0])
    monkeypatch.setattr(kp, "_environ_has", lambda *a: True)      # as if each of them carried the attempt's value
    record = lambda *a: sent.append(a)                           # noqa: E731
    swept = kp.sweep_strays(tmp_path, kill=record, killpg=record, grace_s=0.1)
    assert sent == [] and swept == []


def test_a_proc_entry_that_vanishes_mid_scan_is_skipped(tmp_path):
    """A fake /proc: one entry has no environ (the process ended between the listing and the read), one has an
    unreadable environ, one cannot be stat'ed at all, one matches. Only the match is returned, and nothing raises."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    proc = tmp_path / "proc"
    for pid in ("4001", "4002", "4003"):
        (proc / pid).mkdir(parents=True)
    (proc / "4002" / "environ").mkdir()                           # reading a directory raises
    (proc / "4003" / "environ").write_bytes(b"A=1\0ORCHARD_RUN_DIR=" + str(adir.resolve()).encode() + b"\0")
    (proc / "self").mkdir()
    (proc / "4004").symlink_to(proc / "gone")                     # the whole entry vanished: stat fails
    assert kp.scan_strays(adir, proc_root=str(proc)) == [4003]


def test_the_sweep_never_raises(tmp_path, monkeypatch):
    def broken(run_dir, **kw):
        raise OSError("proc went away")
    monkeypatch.setattr(kp, "scan_strays", broken)
    assert kp.sweep_strays(tmp_path, kill=lambda *a: None, killpg=lambda *a: None) == []


def _carries(pid, adir) -> bool:
    """Whether /proc shows `pid` alive, not a zombie, with this test's unique ORCHARD_RUN_DIR in its environment."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(errors="replace")
        env = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    if stat[stat.rfind(")") + 2:].split()[0] == "Z":
        return False
    return ("ORCHARD_RUN_DIR=" + str(Path(adir).resolve())).encode() in env.split(b"\0")


def test_a_process_a_stray_starts_during_the_grace_is_found_by_a_second_scan_and_killed(tmp_path):
    """S1. On SIGTERM the stray starts a new process in a session of its own (outside any group the sweep signalled)
    and exits. A second scan after the grace finds it by its ORCHARD_RUN_DIR and kills it."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    pidfile = tmp_path / "fork.pid"
    code = ("import os, signal, subprocess, sys, time\n"
            "def on_term(*a):\n"
            "    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
            f"    open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
            "    os._exit(0)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n")
    env = dict(os.environ, ORCHARD_RUN_DIR=str(adir.resolve()))
    stray = _subprocess.Popen([_sys.executable, "-c", code], env=env, start_new_session=True,
                              stdout=_subprocess.PIPE, stderr=_subprocess.DEVNULL)
    sent, violations = [], []
    fork = None
    try:
        assert stray.stdout.readline().strip() == b"ready"
        kill, killpg = _guarded_kills(lambda pid: _carries(pid, adir), sent, violations)
        swept = kp.sweep_strays(adir, kill=kill, killpg=killpg, grace_s=2.0)
        fork = int(pidfile.read_text())
        assert stray.wait(timeout=5) == 0
        assert not _carries(fork, adir)
        assert {r["pid"]: r["stopped"] for r in swept} == {stray.pid: True, fork: True}
        assert violations == []
    finally:
        _stop(stray)
        if fork and _carries(fork, adir):
            os.kill(fork, 9)


def test_a_second_interrupt_during_the_grace_still_sends_the_sigkill_and_propagates(tmp_path):
    """S1. Ctrl-C during the grace (the injected sleep raises KeyboardInterrupt) must not skip the SIGKILL pass."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    env = dict(os.environ, ORCHARD_RUN_DIR=str(adir.resolve()))
    stubborn = _subprocess.Popen(
        [_sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                                "print('ready', flush=True); time.sleep(60)"],
        env=env, start_new_session=True, stdout=_subprocess.PIPE, stderr=_subprocess.DEVNULL)
    sent, violations = [], []

    def interrupted(_s):
        raise KeyboardInterrupt
    try:
        assert stubborn.stdout.readline().strip() == b"ready"
        kill, killpg = _guarded_kills({stubborn.pid}, sent, violations)
        with pytest.raises(KeyboardInterrupt):
            kp.sweep_strays(adir, kill=kill, killpg=killpg, grace_s=5.0, sleep=interrupted)
        assert stubborn.wait(timeout=5) == -_signal.SIGKILL
        assert violations == []
    finally:
        _stop(stubborn)


def test_a_process_still_there_after_the_sigkill_is_reported_as_not_stopped(tmp_path, monkeypatch):
    """S1. Fakes only: the scan names a pid, every kill function is a recorder, and /proc keeps showing the process.
    The record says it was not stopped, and the report says so with its pid."""
    monkeypatch.setattr(kp, "scan_strays", lambda run_dir, **kw: [999999999])
    monkeypatch.setattr(kp, "_environ_has", lambda *a: True)
    monkeypatch.setattr(kp, "_gone", lambda *a: False)               # /proc keeps showing it running
    sent = []
    record = lambda *a: sent.append(a)                           # noqa: E731
    now = [0.0]

    def clock():                                                  # each reading moves on 0.5 s, so no wait spins
        now[0] += 0.5
        return now[0]
    swept = kp.sweep_strays(tmp_path, kill=record, killpg=record, grace_s=0, sleep=lambda _s: None, clock=clock)
    assert swept == [{"pid": 999999999, "cmdline": "", "stopped": False}]
    md = _report_md(tmp_path / "r", [_row_attempt(swept=swept)])
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    assert "1 stray agent process could not be confirmed stopped (pid 999999999)" in row
    assert "process stopped" not in row
    md = _report_md(tmp_path / "r2", [_row_attempt(swept=[dict(swept[0], stopped=True)])])
    assert "1 stray agent process stopped" in md


def test_an_interrupt_during_the_sigterm_pass_still_sends_the_sigkill(tmp_path):
    """M3. Ctrl-C arrives while the sweep is sending SIGTERM (the first group signal raises KeyboardInterrupt and is
    not sent). The SIGKILL pass must still run before the interrupt goes on."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    stray = _child(adir)
    sent, violations = [], []
    try:
        _wait_for_env(stray.pid, str(adir.resolve()))
        kill, killpg = _guarded_kills({stray.pid}, sent, violations)
        first = []

        def interrupted_killpg(pgid, sig):
            if not first:
                first.append(sig)
                raise KeyboardInterrupt
            killpg(pgid, sig)
        with pytest.raises(KeyboardInterrupt):
            kp.sweep_strays(adir, kill=kill, killpg=interrupted_killpg)
        assert stray.wait(timeout=5) == -_signal.SIGKILL and violations == []
    finally:
        _stop(stray)


def test_a_stray_that_execs_without_the_marker_is_not_confirmed_stopped(tmp_path):
    """M3. On SIGTERM the stray execs a new program with an environment that lacks ORCHARD_RUN_DIR. /proc no longer
    shows the marker, but the process runs: the record must say it was not confirmed stopped."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    code = ("import os, signal, sys, time\n"
            "def on_term(*a):\n"
            "    env = {k: v for k, v in os.environ.items() if k != 'ORCHARD_RUN_DIR'}\n"
            "    os.execve(sys.executable, [sys.executable, '-c', 'import time; time.sleep(60)'], env)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n")
    env = dict(os.environ, ORCHARD_RUN_DIR=str(adir.resolve()))
    stray = _subprocess.Popen([_sys.executable, "-c", code], env=env, start_new_session=True,
                              stdout=_subprocess.PIPE, stderr=_subprocess.DEVNULL)
    sent, violations = [], []
    try:
        assert stray.stdout.readline().strip() == b"ready"
        kill, killpg = _guarded_kills({stray.pid}, sent, violations)
        swept = kp.sweep_strays(adir, kill=kill, killpg=killpg, grace_s=2.0)
        assert stray.poll() is None                                   # it is still running, under a new program
        assert [(r["pid"], r["stopped"]) for r in swept] == [(stray.pid, False)] and violations == []
    finally:
        _stop(stray)


def test_gone_means_no_entry_a_zombie_or_another_process_under_the_pid(tmp_path):
    """M3, on a fake /proc: a running process with the recorded start time is not gone; a zombie, a missing entry and
    a process with another start time (the pid was reused) are."""
    proc = tmp_path / "proc"

    def stat(pid, state, start):
        (proc / str(pid)).mkdir(parents=True)
        fields = [state, "1"] + ["0"] * 17 + [str(start)] + ["0"] * 10
        (proc / str(pid) / "stat").write_text(f"{pid} (py thon) " + " ".join(fields) + "\n")
    stat(500, "S", 1234)
    stat(501, "Z", 1234)
    assert kp._gone(str(proc), 500, 1234) is False
    assert kp._gone(str(proc), 500, 1233) is True
    assert kp._gone(str(proc), 501, 1234) is True
    assert kp._gone(str(proc), 502, 1234) is True


def _helper_with_marked_grandchild(adir):
    """A helper child (its own session, NO marker) that starts a grandchild carrying the attempt's ORCHARD_RUN_DIR in
    the helper's group. The grandchild does not lead a group, so the sweep must reach it by pid."""
    code = ("import os, subprocess, sys, time\n"
            f"env = dict(os.environ, ORCHARD_RUN_DIR={str(adir.resolve())!r})\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], env=env)\n"
            "print(g.pid, flush=True)\n"
            "time.sleep(60)\n")
    env = {k: v for k, v in os.environ.items() if k != "ORCHARD_RUN_DIR"}
    helper = _subprocess.Popen([_sys.executable, "-c", code], env=env, start_new_session=True,
                               stdout=_subprocess.PIPE, stderr=_subprocess.DEVNULL)
    grandchild = int(helper.stdout.readline())
    _wait_for_env(grandchild, str(adir.resolve()))
    return helper, grandchild


def test_a_stray_that_does_not_lead_its_group_is_stopped_by_pid(tmp_path):
    """M4. The plain kill(pid) path, with real processes: the grandchild is stopped by kill, nothing is group-signalled,
    and the helper (no marker) keeps running."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    helper, grandchild = _helper_with_marked_grandchild(adir)
    sent, violations = [], []
    try:
        kill, killpg = _guarded_kills({grandchild}, sent, violations)
        swept = kp.sweep_strays(adir, kill=kill, killpg=killpg)
        assert sent and all(kind == "kill" and pid == grandchild for kind, pid, _ in sent), sent
        assert [(r["pid"], r["stopped"]) for r in swept] == [(grandchild, True)]
        assert helper.poll() is None and violations == []
    finally:
        _stop(helper)
        if _carries(grandchild, adir):
            os.kill(grandchild, 9)


def test_the_guard_records_a_plain_kill_it_does_not_allow(tmp_path):
    """M4. The guard's kill branch: a grandchild the test does not allow is recorded and never signalled."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    helper, grandchild = _helper_with_marked_grandchild(adir)
    sent, violations = [], []
    try:
        kill, killpg = _guarded_kills(set(), sent, violations)
        kp.sweep_strays(adir, kill=kill, killpg=killpg, grace_s=0.2)
        assert violations and all(kind == "kill" and pid == grandchild for kind, pid, _ in violations)
        assert _carries(grandchild, adir)                            # nothing was sent
    finally:
        _stop(helper)
        if _carries(grandchild, adir):
            os.kill(grandchild, 9)


def test_an_interrupt_is_not_lost_when_the_second_scan_fails(tmp_path, monkeypatch):
    """A4. Ctrl-C during the grace, and the second scan raises. The known stray still gets its SIGKILL and the
    KeyboardInterrupt goes on (it is not replaced by the scan's error and swallowed)."""
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    env = dict(os.environ, ORCHARD_RUN_DIR=str(adir.resolve()))
    stubborn = _subprocess.Popen(
        [_sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                                "print('ready', flush=True); time.sleep(60)"],
        env=env, start_new_session=True, stdout=_subprocess.PIPE, stderr=_subprocess.DEVNULL)
    real_scan = kp.scan_strays
    scans = []

    def scan(run_dir, **kw):
        scans.append(1)
        if len(scans) > 1:
            raise RuntimeError("proc went away")
        return real_scan(run_dir, **kw)
    monkeypatch.setattr(kp, "scan_strays", scan)
    sent, violations = [], []

    def interrupted(_s):
        raise KeyboardInterrupt
    try:
        assert stubborn.stdout.readline().strip() == b"ready"
        kill, killpg = _guarded_kills({stubborn.pid}, sent, violations)
        with pytest.raises(KeyboardInterrupt):
            kp.sweep_strays(adir, kill=kill, killpg=killpg, grace_s=5.0, sleep=interrupted)
        assert stubborn.wait(timeout=5) == -_signal.SIGKILL and violations == [] and len(scans) == 2
    finally:
        _stop(stubborn)


class _BackgroundingStep:
    """Stands in for AgentStep: run() starts a process with the agent's environment that outlives the step, the way a
    shell command that backgrounds a process and exits would, then returns or raises."""
    started: list = []
    raise_after = False

    def __init__(self, **kw):
        self.run_dir = kw["run_dir"]

    def run(self, system, user):
        env = agent_env(self.run_dir)
        env["PATH"] = os.environ.get("PATH", "")
        p = _subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(60)"], env=env, start_new_session=True,
                              stdin=_subprocess.DEVNULL, stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL)
        type(self).started.append(p)
        (Path(self.run_dir) / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
        _wait_for_env(p.pid, str(Path(self.run_dir).resolve()))
        if type(self).raise_after:
            raise RuntimeError("the agent step crashed")
        return Outcome("done", 1, "done")


def test_run_attempt_stops_agent_processes_that_outlive_the_step_before_the_verdict(tmp_path, fake_kenv, monkeypatch):
    _BackgroundingStep.started, _BackgroundingStep.raise_after = [], False
    monkeypatch.setattr(kp, "AgentStep", _BackgroundingStep)
    order, violations = [], []
    real_sweep = kp.sweep_strays

    def sweep(run_dir, **kw):
        order.append("sweep")
        allowed = {p.pid for p in _BackgroundingStep.started}
        kill, killpg = _guarded_kills(allowed, [], violations)
        return real_sweep(run_dir, kill=kill, killpg=killpg)
    monkeypatch.setattr(kp, "sweep_strays", sweep)
    writer = check_writer(NO_KERNEL)

    def check(argv, **kw):
        order.append("check")
        assert all(p.poll() is not None for p in _BackgroundingStep.started), "a stray ran during the verdict"
        return writer(argv, **kw)
    try:
        attempt = kp.run_attempt(op="add", n=1, probe_dir=tmp_path / "probe", env=fake_kenv, endpoint="http://x/v1",
                                 model="m", coder_model="m", probe_id="t", run_watched=check, tt_lang=FACTS)
        (p,) = _BackgroundingStep.started
        assert order == ["sweep", "check", "sweep"]            # the second sweep is after the check (finding 5)
        assert [s["pid"] for s in attempt["swept"]] == [p.pid] and p.wait(timeout=5) is not None
        on_disk = json.loads((kp.verdict_dir(tmp_path / "probe", "add", 1) / "attempt.json").read_text())
        assert on_disk["swept"] == attempt["swept"]
        kp.write_report(tmp_path, [attempt], {"probe_id": "t", "bars": {}, "hw": None, "ops": ["add"]})
        assert "1 stray agent process stopped" in (tmp_path / "kernel-probe-report.md").read_text()
        assert violations == []
    finally:
        _stop(*_BackgroundingStep.started)


def test_a_crashed_agent_step_still_has_its_strays_stopped(tmp_path, fake_kenv, monkeypatch):
    _BackgroundingStep.started, _BackgroundingStep.raise_after = [], True
    monkeypatch.setattr(kp, "AgentStep", _BackgroundingStep)
    real_sweep = kp.sweep_strays
    violations = []

    def sweep(run_dir, **kw):
        kill, killpg = _guarded_kills({p.pid for p in _BackgroundingStep.started}, [], violations)
        return real_sweep(run_dir, kill=kill, killpg=killpg)
    monkeypatch.setattr(kp, "sweep_strays", sweep)
    try:
        with pytest.raises(RuntimeError):
            kp.run_attempt(op="add", n=1, probe_dir=tmp_path / "probe", env=fake_kenv, endpoint="http://x/v1",
                           model="m", coder_model="m", probe_id="t", run_watched=check_writer(NO_KERNEL),
                           tt_lang=FACTS)
        (p,) = _BackgroundingStep.started
        assert p.wait(timeout=5) == -_signal.SIGTERM and violations == []
    finally:
        _stop(*_BackgroundingStep.started)
        _BackgroundingStep.raise_after = False


# ---- the report says what a row means (I6) ------------------------------------------------------------------------


def _row_attempt(**kw):
    a = {"op": "silu_mul", "attempt": 1, "status": "done", "status_detail": "", "turns": 12, "agent_wall_s": 300.0,
         "passed_sim": True, "category": None, "first_line": "", "pcc_min": 0.9999, "hw": None,
         "agent_check_untrusted": False, "swept": []}
    a.update(kw)
    return a


def _report_md(tmp_path, attempts, ops=("silu_mul",)):
    kp.write_report(tmp_path, attempts, {"probe_id": "t", "bars": {}, "hw": None, "ops": list(ops)})
    return (tmp_path / "kernel-probe-report.md").read_text()


def test_the_table_has_a_status_column_with_the_reason_for_a_step_that_did_not_end_done(tmp_path):
    md = _report_md(tmp_path, [
        _row_attempt(status="escalate", status_detail="the watchdog saw the agent loop again after a nudge\nmore"),
        _row_attempt(attempt=2)])
    header = next(l for l in md.splitlines() if l.startswith("| attempt"))
    assert "| status |" in header
    row1 = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    row2 = next(l for l in md.splitlines() if l.startswith("| 2 |"))
    assert "escalate: the watchdog saw the agent loop again after a nudge |" in row1 and "more" not in row1
    assert "| done |" in row2
    # every row has as many cells as the header
    assert row1.count(" | ") == header.count(" | ") == row2.count(" | ")


def test_the_header_says_add_is_a_calibration_op(tmp_path):
    md = _report_md(tmp_path, [_row_attempt(op="add")], ops=("add",))
    flat = " ".join(md.split())
    assert "`add` is a calibration op" in flat and "reference example the agent reads" in flat
    assert "not evidence" in flat
    assert "calibration op" not in " ".join(_report_md(tmp_path / "other", [_row_attempt()]).split())


def test_the_header_says_the_simulator_computes_in_f32(tmp_path):
    flat = " ".join(_report_md(tmp_path, [_row_attempt()]).split())
    assert "promotes bf16 to f32" in flat and "max_abs_err" in flat and "PCC of 1.0" in flat
    # S4: transcendental ops round in f32 too, so rounding is not a hardware-only effect
    assert "Rounding on hardware is coarser (bf16) than the simulator's f32" in flat
    assert "seen only on hardware" not in flat


def test_an_agent_cell_with_nothing_to_say_is_a_dash(tmp_path):
    md = _report_md(tmp_path, [_row_attempt()])
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    assert row.rstrip().endswith("| - |"), row


@pytest.mark.parametrize("pcc,shown", [(0.99999994, "1 - 6.0e-08"), (0.995, "0.995"), (1.0, "1"),
                                       (0.9999996, "1 - 4.0e-07"), (1.0000001, "1 + 1.0e-07")])
def test_a_pcc_short_of_1_is_never_printed_as_1(tmp_path, pcc, shown):
    md = _report_md(tmp_path, [_row_attempt(pcc_min=pcc)])
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    header = [c.strip() for c in next(l for l in md.splitlines() if l.startswith("| attempt")).strip("|").split("|")]
    assert cells[header.index("PCC min (sim)")] == shown


# ---- what the stall watcher saw on hardware (I8) -------------------------------------------------------------------


def test_the_report_gives_the_watchers_longest_quiet_stretch_and_repeat_for_the_smoke_test_and_each_run(tmp_path,
                                                                                                         fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)

    def fake(argv, **kw):
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET]}))
            return kp.Watched(0, None, False, 30.0, max_quiet_s=12.34, max_repeat=4)
        with open(kw["log_path"], "a") as f:
            f.write("PASSED!\n")
        return kp.Watched(0, None, False, 9.0, max_quiet_s=3.21, max_repeat=2)

    hw = kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=fake)
    assert hw["watch"] == {"max_quiet_s": 3.2, "max_repeat": 2, "quiet_limit_s": kp.HW_QUIET_S,
                           "repeat_limit": kp.HW_REPEAT_LINES}
    assert attempt["hw"]["watch"]["max_quiet_s"] == 12.3 and attempt["hw"]["watch"]["max_repeat"] == 4
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="t", ops=["add"], hw=hw, tt_lang=FACTS)
    kp.write_report(tmp_path, [attempt], header)
    flat = " ".join((tmp_path / "kernel-probe-report.md").read_text().split())
    assert ("smoke test: longest quiet stretch 3.2 s, longest repeat 2 lines; watcher limits 60 s and 20 lines"
            in flat)
    assert ("attempt 1 on hardware: longest quiet stretch 12.3 s, longest repeat 4 lines; watcher limits 60 s and "
            "20 lines" in flat)


@pytest.mark.parametrize("reason,word", [("backstop: still running after 600 s", "backstop"),
                                         ("stalled: no output and no CPU use for 60 s", "stalled")])
def test_a_hang_row_says_whether_the_backstop_or_the_stall_rule_stopped_it(tmp_path, fake_kenv, reason, word):
    """Neither reason may be read as a device hang without the watcher's own words."""
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)

    def fake(argv, **kw):
        if "--out" in argv:
            return kp.Watched(-15, reason, True, 600.0, max_quiet_s=60.0, max_repeat=1)
        with open(kw["log_path"], "a") as f:
            f.write("PASSED!\n")
        return kp.Watched(0, None, False, 9.0)

    with pytest.raises(kp.ProbeBlocked):
        kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=fake)
    assert attempt["hw"]["category"] == "hang" and attempt["hw"]["first_line"].startswith(word)
    assert attempt["hw"]["watch"]["max_quiet_s"] == 60.0
    kp.write_report(tmp_path, [attempt], {"probe_id": "t", "bars": {}, "hw": None, "ops": ["add"]})
    row = next(l for l in (tmp_path / "kernel-probe-report.md").read_text().splitlines() if l.startswith("| 1 |"))
    assert f"| hang | {word}" in row


# ---- one probe per output directory (S6, L1) -----------------------------------------------------------------------
#
# The lock is an flock on DIR/probe.lock. Every holder in these tests is a child this test started, stopped with
# Popen.kill/wait in a finally.

REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _holder(out_dir, *, write_pid=True):
    """A child that holds the probe lock on `out_dir` and sleeps. write_pid=False takes the flock by hand and leaves
    the file empty, as a holder looks between its create and its write."""
    if write_pid:
        code = ("import sys, time\nfrom orchard import kernelprobe as kp\n"
                f"lock = kp.acquire_lock(__import__('pathlib').Path({str(out_dir)!r}))\n"
                "print('held', flush=True)\ntime.sleep(60)\n")
    else:
        code = ("import fcntl, os, time\n"
                f"os.makedirs({str(out_dir)!r}, exist_ok=True)\n"
                f"fd = os.open(os.path.join({str(out_dir)!r}, 'probe.lock'), os.O_RDWR | os.O_CREAT, 0o600)\n"
                "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\nos.ftruncate(fd, 0)\n"
                "print('held', flush=True)\ntime.sleep(60)\n")
    p = _subprocess.Popen([_sys.executable, "-c", code], cwd=REPO_ROOT, stdout=_subprocess.PIPE,
                          stderr=_subprocess.DEVNULL, env=dict(os.environ, PYTHONPATH=REPO_ROOT))
    assert p.stdout.readline().strip() == b"held"
    return p


def _end(p):
    p.kill()
    p.wait(timeout=10)


def test_a_second_probe_on_the_same_directory_is_refused(tmp_path, fake_kenv, env_ready, monkeypatch, capsys):
    seen = fake_attempts(monkeypatch, lambda n: None)
    out_dir = tmp_path / "out"
    holder = _holder(out_dir)
    try:
        code = kp.main(probe_args(tmp_path, attempts=1), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                       run_watched_fn=no_run)
        assert code == kp.EXIT_REFUSED and seen == []
        assert f"another probe is using {out_dir} (pid {holder.pid})" in capsys.readouterr().err
    finally:
        _end(holder)


def test_a_lock_left_by_a_sigkilled_probe_is_taken_over_at_once(tmp_path, fake_kenv, env_ready, monkeypatch):
    seen = fake_attempts(monkeypatch, lambda n: None)
    out_dir = tmp_path / "out"
    holder = _holder(out_dir)
    _end(holder)                                                  # SIGKILL: no cleanup ran, the file stays
    assert (out_dir / kp.LOCK_NAME).is_file()
    code = kp.main(probe_args(tmp_path, attempts=1), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    assert code == kp.EXIT_OK and seen == [1]
    assert not (out_dir / kp.LOCK_NAME).exists()                 # removed at a clean exit


def test_two_probes_racing_for_one_directory_never_both_get_it(tmp_path):
    """Twenty rounds. Each round starts with a lock file a dead probe left, and two children (this test's own) each
    write a ready file once imported, then wait on a barrier file that the parent writes only when both are ready, and
    take the lock at the same moment. Exactly one may get it."""
    code = ("import os, sys, time\nfrom pathlib import Path\nfrom orchard import kernelprobe as kp\n"
            "out, barrier, ready = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])\n"
            "ready.write_text('ready')\n"
            "while not barrier.exists():\n    time.sleep(0.0005)\n"
            "try:\n    lock = kp.acquire_lock(out)\n    print('got', flush=True)\n    time.sleep(0.3)\n"
            "    kp.release_lock(lock)\nexcept kp._Refused:\n    print('refused', flush=True)\n")
    dead = _subprocess.Popen([_sys.executable, "-c", "pass"])
    dead.wait(timeout=10)
    for rnd in range(20):
        out_dir = tmp_path / f"out{rnd}"
        out_dir.mkdir()
        (out_dir / kp.LOCK_NAME).write_text(json.dumps({"pid": dead.pid, "boot_id": "old", "start_time": 1}))
        barrier = tmp_path / f"go{rnd}"
        ready = [tmp_path / f"ready{rnd}-{i}" for i in range(2)]
        kids = [_subprocess.Popen([_sys.executable, "-c", code, str(out_dir), str(barrier), str(ready[i])],
                                  cwd=REPO_ROOT, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
                                  env=dict(os.environ, PYTHONPATH=REPO_ROOT)) for i in range(2)]
        try:
            end = _time.monotonic() + 60                            # a bounded wait for both children to be ready
            while _time.monotonic() < end and not all(r.exists() for r in ready):
                _time.sleep(0.005)
            assert all(r.exists() for r in ready), f"round {rnd}: a child never became ready"
            barrier.write_text("go")
            outs = [k.communicate(timeout=30) for k in kids]
            results = sorted(o.decode().strip() for o, _ in outs)
            assert results == ["got", "refused"], (rnd, results, [e.decode(errors="replace") for _, e in outs])
        finally:
            for k in kids:
                _end(k)


def test_a_symlink_at_the_lock_path_is_refused_and_left_alone(tmp_path, fake_kenv, env_ready, monkeypatch, capsys):
    seen = fake_attempts(monkeypatch, lambda n: None)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = tmp_path / "operator.txt"
    target.write_text("keep me\n")
    (out_dir / kp.LOCK_NAME).symlink_to(target)
    code = kp.main(probe_args(tmp_path, attempts=1), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    assert code == kp.EXIT_REFUSED and seen == [] and target.read_text() == "keep me\n"
    assert (out_dir / kp.LOCK_NAME).is_symlink() and kp.LOCK_NAME in capsys.readouterr().err


def test_a_read_only_output_directory_is_a_plain_refusal(tmp_path, fake_kenv, env_ready, monkeypatch, capsys):
    seen = fake_attempts(monkeypatch, lambda n: None)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_dir.chmod(0o500)
    try:
        code = kp.main(probe_args(tmp_path, attempts=1), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                       run_watched_fn=no_run)
        err = capsys.readouterr().err
        assert code == kp.EXIT_REFUSED and seen == [] and "Traceback" not in err
        assert "cannot open the lock" in err and "Permission denied" in err
    finally:
        out_dir.chmod(0o700)


@pytest.mark.parametrize("text", ["9" * 4000, "\x00\xff garbage", '{"pid": 99999999999999999999999}', '{"pid": "x"}'], ids=["digits", "garbage", "huge", "string"])
def test_a_lock_with_garbage_or_a_huge_pid_still_gives_a_plain_refusal(tmp_path, text):
    out_dir = tmp_path / "out"
    holder = _holder(out_dir)
    try:
        (out_dir / kp.LOCK_NAME).write_text(text, errors="replace")      # the flock is advisory; the text is info
        with pytest.raises(kp._Refused) as info:
            kp.acquire_lock(out_dir)
        assert str(info.value).startswith(f"another probe is using {out_dir}")
    finally:
        _end(holder)


def test_an_empty_lock_whose_flock_is_held_is_held(tmp_path):
    """A holder between its create and its write leaves an empty file. The flock decides, so it is refused as held."""
    out_dir = tmp_path / "out"
    holder = _holder(out_dir, write_pid=False)
    try:
        assert (out_dir / kp.LOCK_NAME).read_bytes() == b""
        with pytest.raises(kp._Refused) as info:
            kp.acquire_lock(out_dir)
        assert str(info.value).startswith(f"another probe is using {out_dir}")
    finally:
        _end(holder)


def test_a_lock_file_replaced_between_the_open_and_the_flock_is_taken_again(tmp_path, monkeypatch):
    """Another probe unlinked the file we opened and made a new one: our flock on the old inode reaches nobody, so
    the lock is taken again on the file at the path."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = out_dir / kp.LOCK_NAME
    path.write_text("")
    real_flock = kp.fcntl.flock
    calls = []

    def flock(fd, op):
        if not calls:
            os.unlink(path)
            path.write_text("")
        calls.append(op)
        return real_flock(fd, op)
    monkeypatch.setattr(kp.fcntl, "flock", flock)
    lock = kp.acquire_lock(out_dir)
    try:
        assert len(calls) >= 2
        assert os.fstat(lock.fd).st_ino == os.stat(path).st_ino
    finally:
        kp.release_lock(lock)


def test_the_lock_is_held_while_attempts_run_and_removed_after_a_crash(tmp_path, fake_kenv, env_ready, monkeypatch):
    out_dir = tmp_path / "out"
    held = []

    def on_n(n):
        held.append(json.loads((out_dir / kp.LOCK_NAME).read_text())["pid"])
        raise OSError("crash")
    fake_attempts(monkeypatch, on_n)
    code = kp.main(probe_args(tmp_path, attempts=2), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    assert code == kp.EXIT_ERROR and held == [os.getpid()]
    assert not (out_dir / kp.LOCK_NAME).exists()


def test_an_os_error_inside_the_lock_is_a_plain_refusal(tmp_path, fake_kenv, env_ready, monkeypatch, capsys):
    """A5. os.fstat failing on the lock file (called from kernelprobe only; every other caller gets the real one)
    must give exit 2 with a plain message. Before the fix it escaped as a raw OSError."""
    import errno as _errno
    real_fstat = os.fstat

    def fstat(fd):
        if _sys._getframe(1).f_globals.get("__name__") == "orchard.kernelprobe":
            raise OSError(_errno.EIO, "Input/output error")
        return real_fstat(fd)
    monkeypatch.setattr(os, "fstat", fstat)
    with pytest.raises(kp._Refused) as info:
        kp.acquire_lock(tmp_path / "direct")
    assert "Input/output error" in str(info.value)
    seen = fake_attempts(monkeypatch, lambda n: None)
    code = kp.main(probe_args(tmp_path, attempts=1), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    err = capsys.readouterr().err
    assert code == kp.EXIT_REFUSED and seen == [] and "Traceback" not in err and "Input/output error" in err


# ---- sampling (B2 to B4) ----------------------------------------------------------------------------------------------


def test_run_attempt_sends_and_records_its_sampling(tmp_path, fake_kenv):
    with FakeModel(scripted(final("done"))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL),
                              temperature=0.7, top_p=0.9, seed=11)
        sent = server.requests[0]
    want = {"temperature": 0.7, "top_p": 0.9, "seed": 11}
    assert (sent["temperature"], sent["top_p"], sent["seed"]) == (0.7, 0.9, 11)
    assert attempt["sampling"] == want and attempt["provenance"]["sampling"] == want
    on_disk = json.loads((kp.verdict_dir(tmp_path / "probe", "add", 1) / "attempt.json").read_text())
    assert on_disk["sampling"] == want


def test_run_attempt_is_greedy_by_default(tmp_path, fake_kenv):
    with FakeModel(scripted(final("done"))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL))
        sent = server.requests[0]
    assert sent["temperature"] == 0 and "seed" not in sent and "top_p" not in sent
    assert attempt["sampling"] == {"temperature": 0.0, "top_p": None, "seed": None}


def _recording_attempts(monkeypatch):
    """Replace run_attempt with one that records (n, temperature, top_p, seed) and returns a canned record."""
    seen = []

    def run_attempt(*, op, n, temperature=0.0, top_p=None, seed=None, **kw):
        seen.append((n, temperature, top_p, seed))
        rec = canned(op, n)
        rec["sampling"] = {"temperature": temperature, "top_p": top_p, "seed": seed}
        return rec
    monkeypatch.setattr(kp, "run_attempt", run_attempt)
    return seen


@pytest.mark.parametrize("flags,words", [
    ({"temperature": "2.5"}, "between 0 and 2"),
    ({"temperature": "-0.1"}, "between 0 and 2"),
    ({"temperature": "warm"}, "not a number"),
    ({"temperature": "nan"}, "between 0 and 2"),
    ({"temperatures": "0,,1"}, "not a number"),
    ({"temperatures": "0,3"}, "between 0 and 2"),
    ({"temperatures": "0.5,0.5"}, "twice"),
    ({"temperature": "0.5", "temperatures": "0,1"}, "--temperature or --temperatures"),
    ({"top_p": "0"}, "top-p"),
    ({"top_p": "1.5"}, "top-p"),
    ({"top_p": "x"}, "top-p"),
    ({"seed": "1.5"}, "--seed"),
    ({"seed": "-3"}, "--seed"),
], ids=lambda v: str(v) if isinstance(v, str) else "-".join(f"{k}={x}" for k, x in v.items()))
def test_bad_sampling_flags_are_refused_before_anything_runs(tmp_path, fake_kenv, env_ready, monkeypatch, capsys,
                                                             flags, words):
    seen = _recording_attempts(monkeypatch)
    code = kp.main(probe_args(tmp_path, **flags), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    err = capsys.readouterr().err
    assert code == kp.EXIT_REFUSED and seen == [] and words in err and "Traceback" not in err
    assert not (tmp_path / "out").exists() and env_ready == []        # refused before the environment check


def test_temperatures_run_n_attempts_each_with_seeds_base_plus_k_and_none_at_zero(tmp_path, fake_kenv, env_ready,
                                                                                  monkeypatch):
    seen = _recording_attempts(monkeypatch)
    code = kp.main(probe_args(tmp_path, attempts=3, temperatures="0,0.7", seed="100", top_p="0.9"), env=fake_kenv,
                   out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == kp.EXIT_OK
    assert seen == [(1, 0.0, 0.9, None), (2, 0.0, 0.9, None), (3, 0.0, 0.9, None),
                    (4, 0.7, 0.9, 104), (5, 0.7, 0.9, 105), (6, 0.7, 0.9, 106)]
    header = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())["header"]
    assert header["sampling"]["seed_base"] == 100 and header["sampling"]["seed_base_from"] == "--seed"
    assert header["sampling"]["temperatures"] == [0.0, 0.7] and header["sampling"]["top_p"] == 0.9


def test_without_a_seed_the_base_comes_from_the_probe_id_and_is_recorded(tmp_path, fake_kenv, env_ready,
                                                                        monkeypatch):
    seen = _recording_attempts(monkeypatch)
    monkeypatch.setattr(kp, "new_probe_id", lambda: "20261016T000000Z-abcdef12")
    code = kp.main(probe_args(tmp_path, attempts=2, temperature="1.0"), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == kp.EXIT_OK
    base = kp.seed_base_from_probe_id("20261016T000000Z-abcdef12")
    assert isinstance(base, int) and 0 <= base < 2 ** 31
    assert seen == [(1, 1.0, None, base + 1), (2, 1.0, None, base + 2)]
    out = tmp_path / "out"
    header = json.loads((out / "kernel-probe-report.json").read_text())["header"]
    assert header["sampling"]["seed_base"] == base and header["sampling"]["seed_base_from"] == "probe id"
    assert f"--seed {base}" in (out / "kernel-probe-report.md").read_text()


def test_the_default_is_one_greedy_group_with_no_seed(tmp_path, fake_kenv, env_ready, monkeypatch):
    seen = _recording_attempts(monkeypatch)
    code = kp.main(probe_args(tmp_path, attempts=2), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    assert code == kp.EXIT_OK and seen == [(1, 0.0, None, None), (2, 0.0, None, None)]


def test_the_cli_parser_has_the_sampling_flags():
    from orchard import cli
    args = cli._parser().parse_args(["kernel-probe", "--op", "add", "--temperatures", "0,0.5", "--top-p", "0.9",
                                     "--seed", "7"])
    assert (args.temperatures, args.top_p, args.seed, args.temperature) == ("0,0.5", "0.9", "7", None)


def _sampled(n, temperature, sha, op="silu_mul"):
    a = _row_attempt(op=op, attempt=n, sampling={"temperature": temperature, "top_p": None,
                                                  "seed": None if temperature == 0 else 100 + n})
    a["provenance"] = {"kernel_sha256": sha}
    return a


def test_the_report_groups_attempts_by_temperature_and_counts_distinct_kernels(tmp_path):
    attempts = [_sampled(1, 0.0, "a" * 64), _sampled(2, 0.0, "a" * 64), _sampled(3, 0.0, None),
                _sampled(4, 0.7, "b" * 64), _sampled(5, 0.7, "c" * 64), _sampled(6, 0.7, "b" * 64)]
    data = kp.write_report(tmp_path, attempts, {"probe_id": "t", "bars": {}, "hw": None, "ops": ["silu_mul"],
                                                "sampling": {"temperatures": [0.0, 0.7], "top_p": None,
                                                             "seed_base": 100, "seed_base_from": "--seed",
                                                             "attempts_per_temperature": 3}})
    md = (tmp_path / "kernel-probe-report.md").read_text()
    zero, warm = md.split("### temperature 0.7")
    assert "### temperature 0\n" in zero
    assert "| 1 |" in zero and "| 3 |" in zero and "| 4 |" not in zero and "| 4 |" in warm and "| 1 |" not in warm
    assert "distinct kernels: 1 of 2 attempts; 1 wrote no kernel" in zero
    assert ("All 2 attempts that wrote a kernel produced the same kernel (1 wrote none): temperature 0 asks the "
            "server for greedy decoding, so attempts at one prompt often give one sample.") in " ".join(zero.split())
    assert "Raise --temperature to explore." in zero
    assert "distinct kernels: 2 of 3 attempts" in warm and "produced the same kernel" not in warm
    groups = {g["temperature"]: g for g in data["groups"]}
    assert groups[0.0]["distinct_kernels"] == 1 and groups[0.0]["with_kernel"] == 2 and groups[0.0]["no_kernel"] == 1
    assert groups[0.7]["distinct_kernels"] == 2 and groups[0.7]["no_kernel"] == 0
    assert groups[0.0]["sampling"]["temperature"] == 0.0 and groups[0.7]["attempts"] == [4, 5, 6]


@pytest.mark.parametrize("shas,warn", [(["a" * 64], False), (["a" * 64, "b" * 64], False),
                                       (["a" * 64, "a" * 64, "a" * 64], True), ([None, None], False)],
                         ids=["one", "two-distinct", "three-same", "none-written"])
def test_the_same_kernel_warning_needs_more_than_one_attempt_with_a_kernel(tmp_path, shas, warn):
    md_text = " ".join(_report_md(tmp_path, [_sampled(i + 1, 0.0, s) for i, s in enumerate(shas)]).split())
    assert ("produced the same kernel" in md_text) is warn


def test_the_header_says_greedy_attempts_often_repeat_and_are_not_guaranteed_identical(tmp_path):
    """W1. The live batch gave 5 distinct kernels in 5 attempts of silu_mul at temperature 0, so the report may not
    call greedy attempts repeatable; it says they often repeat and are not guaranteed to be identical."""
    flat = " ".join(_report_md(tmp_path, [_sampled(1, 0.0, "a" * 64)]).split())
    assert ("Temperature 0 asks the server for greedy decoding. Attempts then often repeat each other (the report "
            "counts distinct kernels) but are not guaranteed to be identical.") in flat
    assert ("Attempts at temperature above 0 with different seeds sample independently, if the server honours the "
            "temperature.") in flat
    assert "repeatable" not in flat and "independent draws" not in flat


def test_no_text_calls_greedy_attempts_repeatable():
    """W1. The help, the README, the module docstring and parse_sampling make no claim that a kernel is reproducible."""
    from orchard import cli
    help_text = " ".join(cli._parser()._subparsers._group_actions[0].choices["kernel-probe"].format_help().split())
    readme = " ".join((REPO / "README.md").read_text().split())
    probe_part = readme[readme.index("tt-orchard kernel-probe"):][:12000]
    for name, text in (("help", help_text), ("README", probe_part), ("module", " ".join(kp.__doc__.split())),
                       ("parse_sampling", " ".join(kp.parse_sampling.__doc__.split()))):
        assert "repeatable" not in text and "reproducib" not in text, name
    assert "not guaranteed to be identical" in probe_part


def test_the_same_kernel_warning_above_temperature_0_names_the_likely_causes(tmp_path):
    flat = " ".join(_report_md(tmp_path, [_sampled(1, 0.7, "a" * 64), _sampled(2, 0.7, "a" * 64)]).split())
    assert ("All 2 attempts that wrote a kernel produced the same kernel at temperature 0.7: the temperature may be "
            "too low for this prompt, or the server may be ignoring it.") in flat


SEED_MAX = 2 ** 63 - 1


@pytest.mark.parametrize("flags,words", [
    ({"seed": str(SEED_MAX), "attempts": 1}, str(SEED_MAX)),
    ({"seed": str(SEED_MAX - 4), "attempts": 5}, "2**63 - 1"),
    ({"seed": True}, "--seed"),
    ({"temperature": "inf"}, "between 0 and 2"),
    ({"temperature": "-inf"}, "between 0 and 2"),
    ({"temperatures": ",".join(str(i / 10) for i in range(11))}, "at most 10 temperatures (got 11)"),
    ({"attempts": 201}, "at most 200 attempts in total (201)"),
    ({"attempts": 101, "temperatures": "0,1"}, "at most 200 attempts in total (101 x 2 = 202)"),
], ids=["seed-max-1", "seed-max-5", "seed-bool", "inf", "minus-inf", "eleven-temps", "201", "101x2"])
def test_seeds_out_of_range_and_oversized_plans_are_refused(tmp_path, fake_kenv, env_ready, monkeypatch, capsys,
                                                            flags, words):
    """W2. A seed whose last attempt would pass 2**63 - 1 would fail on every request at run time; the plan is capped
    at 10 temperatures and 200 attempts in total. Each is refused with exit 2 before anything runs."""
    seen = _recording_attempts(monkeypatch)
    code = kp.main(probe_args(tmp_path, **flags), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    err = capsys.readouterr().err
    assert code == kp.EXIT_REFUSED and seen == [] and words in err, err
    assert not (tmp_path / "out").exists()


def test_the_largest_seed_whose_last_attempt_fits_is_accepted(tmp_path, fake_kenv, env_ready, monkeypatch):
    seen = _recording_attempts(monkeypatch)
    code = kp.main(probe_args(tmp_path, attempts=5, temperature="0.5", seed=str(SEED_MAX - 5)), env=fake_kenv,
                   out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == kp.EXIT_OK and seen[-1][3] == SEED_MAX


def test_ten_temperatures_and_200_attempts_are_accepted(tmp_path, fake_kenv, env_ready, monkeypatch):
    seen = _recording_attempts(monkeypatch)
    code = kp.main(probe_args(tmp_path, attempts=20, temperatures=",".join(str(i / 10) for i in range(10))),
                   env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == kp.EXIT_OK and len(seen) == 200


def test_a_groups_sampling_lists_every_seed_and_the_base(tmp_path):
    """W4. The group's sampling is the group's, with each attempt's seed; it is not the first attempt's sampling."""
    attempts = [_sampled(1, 0.0, "a" * 64), _sampled(2, 0.0, "b" * 64), _sampled(3, 0.7, "c" * 64),
                _sampled(4, 0.7, "d" * 64)]
    data = kp.write_report(tmp_path, attempts, {"probe_id": "t", "bars": {}, "hw": None, "ops": ["silu_mul"],
                                                "sampling": {"temperatures": [0.0, 0.7], "top_p": None, "seed_base": 100,
                                                             "seed_base_from": "--seed",
                                                             "attempts_per_temperature": 2}})
    groups = {g["temperature"]: g for g in data["groups"]}
    assert groups[0.0]["sampling"] == {"temperature": 0.0, "top_p": None, "seed_base": 100, "seeds": [None, None]}
    assert groups[0.7]["sampling"] == {"temperature": 0.7, "top_p": None, "seed_base": 100, "seeds": [103, 104]}


def test_the_start_line_says_attempts_per_temperature_and_the_total(tmp_path, fake_kenv, env_ready, monkeypatch):
    """W5."""
    _recording_attempts(monkeypatch)
    out = io.StringIO()
    kp.main(probe_args(tmp_path, attempts=5, temperatures="0,0.7"), env=fake_kenv, out=out, probe=lambda e, m: True,
            run_watched_fn=no_run)
    assert "5 attempts per temperature (0, 0.7): 10 in total" in out.getvalue()


def test_a_greedy_only_run_gives_no_seed_formula_or_seed_advice(tmp_path):
    """W6. No attempt of a greedy-only run sends a seed, so the header says only that."""
    header = {"probe_id": "t", "bars": {}, "hw": None, "ops": ["add"],
              "sampling": {"temperatures": [0.0], "top_p": None, "seed_base": 1234, "seed_base_from": "probe id",
                           "attempts_per_temperature": 3}}
    kp.write_report(tmp_path, [_sampled(1, 0.0, "a" * 64, op="add")], header)
    flat = " ".join((tmp_path / "kernel-probe-report.md").read_text().split())
    assert "No seed is sent at temperature 0." in flat
    assert "--seed" not in flat and "1234" not in flat and "+ k" not in flat


def test_the_same_kernel_warning_names_no_missing_kernels_when_every_attempt_wrote_one(tmp_path):
    """W7."""
    flat = " ".join(_report_md(tmp_path, [_sampled(1, 0.0, "a" * 64), _sampled(2, 0.0, "a" * 64)]).split())
    assert "All 2 attempts that wrote a kernel produced the same kernel: temperature 0" in flat and "wrote none" not in flat


def test_minus_zero_is_temperature_0(tmp_path, fake_kenv, env_ready, monkeypatch, capsys):
    """W8. -0 is read as 0: it prints as 0 and is a duplicate of 0."""
    seen = _recording_attempts(monkeypatch)
    out = io.StringIO()
    code = kp.main(probe_args(tmp_path, attempts=1, temperature="-0"), env=fake_kenv, out=out,
                   probe=lambda e, m: True, run_watched_fn=no_run)
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert code == kp.EXIT_OK and "temperature -0" not in out.getvalue() and "temperature -0" not in md
    assert seen == [(1, 0.0, None, None)] and str(seen[0][1]) == "0.0"
    code = kp.main(probe_args(tmp_path / "b", attempts=1, temperatures="-0,0"), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == kp.EXIT_REFUSED and "twice" in capsys.readouterr().err


def test_phase_hw_with_two_temperatures_runs_one_smoke_test_and_every_sim_pass(tmp_path, fake_kenv, env_ready):
    """W11. Attempts 1 and 2 run at temperature 0, 3 and 4 at 0.7; attempts 1 and 3 write the good kernel. One smoke
    test runs, then a hardware run for each simulator pass, in both groups."""
    good = (KERNELS / "add_good.py").read_text()
    script = scripted(call("write_file", path="kernel.py", content=good), final("done"), final("done"),
                      call("write_file", path="kernel.py", content=good), final("done"), final("done"))
    gozer_calls = []
    with FakeModel(script) as server:
        code = kp.main(probe_args(tmp_path, attempts=2, temperatures="0,0.7", endpoint=server.endpoint, phase="hw"),
                       env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                       run_watched_fn=passing_check(gozer_calls))
    assert code == 0
    smoke = [c for c in gozer_calls if "--out" not in c]
    kernels = [c for c in gozer_calls if "--out" in c]
    assert len(smoke) == 1 and len(kernels) == 2
    assert gozer_calls.index(smoke[0]) == 0                       # the smoke test comes first
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    by_n = {a["attempt"]: a for a in data["attempts"]}
    assert by_n[1]["hw"]["passed"] is True and by_n[3]["hw"]["passed"] is True
    assert by_n[2]["hw"] is None and by_n[4]["hw"] is None
    assert by_n[3]["sampling"]["temperature"] == 0.7


# ---- the coder's reference files (Task 14: the tt-lang skill and the checkout's examples and specification) --------

REF_NAMES = ["guide.md", "examples.md", "TTLangSpecification.md", "eltwise_add.py", "eltwise_broadcast_reduce.py"]


def test_read_references_reads_the_skill_and_the_checkout_once_with_sha256s(fake_kenv, ttlang_skill):
    refs, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
    assert problems == []
    assert [r["name"] for r in refs] == REF_NAMES
    by = {r["name"]: r for r in refs}
    assert by["guide.md"]["source"] == "tt-lang skill" and by["guide.md"]["path"] == str(ttlang_skill / "guide.md")
    assert by["eltwise_broadcast_reduce.py"]["source"] == "tt-lang checkout"
    assert by["TTLangSpecification.md"]["path"] == str(fake_kenv.src_dir / "docs" / "sphinx" / "specs" /
                                                      "TTLangSpecification.md")
    for r in refs:
        assert r["data"] == Path(r["path"]).read_bytes()
        assert r["sha256"] == hashlib.sha256(r["data"]).hexdigest()


def test_prepare_attempt_writes_the_references_as_regular_files_and_returns_their_shas(tmp_path, fake_kenv, ttlang_skill):
    refs, _ = ttlang_env.read_references(fake_kenv, ttlang_skill)
    adir = tmp_path / "add" / "attempt-1"
    copied = kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS, references=refs)
    assert [c["name"] for c in copied] == [f"reference/{n}" for n in REF_NAMES]
    for c, r in zip(copied, refs):
        p = adir / c["name"]
        assert p.is_file() and not p.is_symlink()
        assert p.read_bytes() == r["data"] and c["sha256"] == r["sha256"] and c["source"] == r["source"]
        assert "data" not in c


@pytest.mark.parametrize("name", ["guide.md", "examples.md"])
def test_a_missing_skill_file_is_a_problem_that_names_it(fake_kenv, ttlang_skill, name):
    (ttlang_skill / name).unlink()
    refs, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
    assert len(problems) == 1 and str(ttlang_skill / name) in problems[0] and "does not exist" in problems[0]


def test_a_missing_skill_directory_is_a_problem(fake_kenv, tmp_path):
    _, problems = ttlang_env.read_references(fake_kenv, tmp_path / "nowhere")
    assert problems and all("does not exist" in p for p in problems[:2])


def test_a_missing_checkout_example_is_a_problem(fake_kenv, ttlang_skill):
    (fake_kenv.src_dir / "examples" / "eltwise_broadcast_reduce.py").unlink()
    _, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
    assert len(problems) == 1 and "eltwise_broadcast_reduce.py" in problems[0]


def test_a_reference_that_is_a_link_is_refused_and_its_target_never_read(fake_kenv, ttlang_skill, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("token=abc\n")
    (ttlang_skill / "guide.md").unlink()
    (ttlang_skill / "guide.md").symlink_to(secret)
    refs, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
    assert len(problems) == 1 and "link" in problems[0]
    assert all(b"token=abc" not in r["data"] for r in refs)


def test_a_linked_reference_is_named_a_link_by_the_lstat_layer(fake_kenv, ttlang_skill, tmp_path):
    (tmp_path / "other.md").write_text("x\n")
    (ttlang_skill / "guide.md").unlink()
    (ttlang_skill / "guide.md").symlink_to(tmp_path / "other.md")
    _, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
    assert problems == [f"{ttlang_skill / 'guide.md'} is a link; put the file itself there"]


def test_a_linked_reference_is_not_followed_even_when_lstat_was_fooled(fake_kenv, ttlang_skill, tmp_path, monkeypatch):
    """The open-side layer alone: the open does not follow the link, so the target's bytes are never read."""
    secret = tmp_path / "secret.txt"
    secret.write_text("token=abc\n")
    (ttlang_skill / "guide.md").unlink()
    (ttlang_skill / "guide.md").symlink_to(secret)
    real = os.lstat

    def fake(p, *a, **kw):
        st = real(p, *a, **kw)
        if Path(p) == ttlang_skill / "guide.md":
            return os.stat_result((_stat.S_IFREG | 0o644,) + tuple(st)[1:6] + (10,) + tuple(st)[7:])
        return st
    monkeypatch.setattr(ttlang_env.os, "lstat", fake)
    refs, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
    assert len(problems) == 1 and "cannot be opened" in problems[0]
    assert all(b"token=abc" not in r["data"] for r in refs)


def test_a_fifo_reference_is_refused_without_blocking(fake_kenv, ttlang_skill):
    (ttlang_skill / "examples.md").unlink()
    os.mkfifo(ttlang_skill / "examples.md")
    try:
        hung, value = _in_thread(lambda: ttlang_env.read_references(fake_kenv, ttlang_skill), timeout=10)
    finally:
        _unblock(ttlang_skill / "examples.md")
    assert not hung
    _, problems = value
    assert len(problems) == 1 and "not a regular file" in problems[0]


def test_an_oversized_reference_is_refused(fake_kenv, ttlang_skill):
    (ttlang_skill / "guide.md").write_bytes(b"x" * (ttlang_env.REFERENCE_MAX_BYTES + 1))
    _, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
    assert len(problems) == 1 and "larger than" in problems[0]


def test_a_reference_linked_to_dev_zero_is_refused_in_a_limited_child(tmp_path):
    """The hostile read runs in a child with a 2 GiB address-space limit and a timeout (hostile-input rule): a reader
    that followed the link would read without end."""
    import subprocess
    import sys
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "guide.md").symlink_to("/dev/zero")
    (skill / "examples.md").write_text("x\n")
    code = ("import resource, sys; resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30)); "
            f"sys.path.insert(0, {str(REPO)!r}); from pathlib import Path; from orchard import ttlang_env as t; "
            f"env = t.KEnv(src_dir=Path({str(tmp_path / 'src')!r}), venv_dir=Path('/nonexistent'), scratch_dir=Path('/tmp')); "
            f"refs, problems = t.read_references(env, Path({str(skill)!r})); print([p for p in problems if 'guide.md' in p])")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    assert "link" in done.stdout


def test_the_skill_directory_comes_from_the_flag_then_the_environment_then_the_home(tmp_path):
    from types import SimpleNamespace
    home = tmp_path / "home"
    assert ttlang_env.skill_dir_from(SimpleNamespace(tt_lang_skill_dir="/f"), environ={"ORCHARD_TTLANG_SKILL": "/e"},
                                     home=home) == Path("/f")
    assert ttlang_env.skill_dir_from(SimpleNamespace(tt_lang_skill_dir=None), environ={"ORCHARD_TTLANG_SKILL": "/e"},
                                     home=home) == Path("/e")
    assert ttlang_env.skill_dir_from(SimpleNamespace(), environ={}, home=home) == \
        home / ".claude" / "skills" / "ttm-tt-lang"


def test_main_refuses_when_the_tt_lang_skill_is_missing_and_starts_nothing(tmp_path, fake_kenv, env_ready, monkeypatch,
                                                                          capsys):
    monkeypatch.setenv(ttlang_env.TTLANG_SKILL_ENV, str(tmp_path / "no-skill"))
    asked = []
    code = kp.main(probe_args(tmp_path), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: asked.append(1) or True,
                   run_watched_fn=no_run)
    err = capsys.readouterr().err
    assert code == 2 and not (tmp_path / "out").exists() and asked == []
    assert str(tmp_path / "no-skill") in err and "--tt-lang-skill-dir" in err


def test_main_gives_every_attempt_the_references_and_records_them(tmp_path, fake_kenv, env_ready, ttlang_skill):
    with FakeModel(lambda r: final("nothing")) as server:
        code = kp.main(probe_args(tmp_path, attempts=2, endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                       probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 0
    want = {f"reference/{r['name']}": r["sha256"] for r in ttlang_env.read_references(fake_kenv, ttlang_skill)[0]}
    want.update({f"reference/{r['name']}": r["sha256"] for r in kp.library_references("add", "all")})
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert {r["name"]: r["sha256"] for r in data["header"]["references"]} == want
    assert data["header"]["library"] == {"mode": "all", "note": None}
    assert data["header"]["tt_lang_skill_dir"] == str(ttlang_skill)
    for a in data["attempts"]:
        adir = tmp_path / "out" / "add" / f"attempt-{a['attempt']}"
        assert {r["name"]: r["sha256"] for r in a["provenance"]["references"]} == want
        for name, sha in want.items():
            assert hashlib.sha256((adir / name).read_bytes()).hexdigest() == sha
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "## Reference files" in md and str(ttlang_skill / "guide.md") in md
    assert all(sha in md for sha in want.values())


def test_the_agent_can_read_and_grep_the_reference_files(tmp_path, fake_kenv, ttlang_skill):
    refs, _ = ttlang_env.read_references(fake_kenv, ttlang_skill)
    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS, references=refs)
    tools = kp.attempt_tools(adir)
    assert tools.read_file("reference/guide.md") == SKILL_FILES["guide.md"]
    got = tools.shell("grep -n stand-in reference/TTLangSpecification.md")
    assert got.startswith("exit 0") and "stand-in specification" in got


def test_reading_the_reference_files_is_not_forbidden_code(tmp_path):
    """A kernel that names the reference files in a comment is judged by the gate like any other: the gate does not
    refuse the word `reference` or the file names."""
    gate = kp._template("kernel_check")
    kernel = tmp_path / "kernel.py"
    kernel.write_text((KERNELS / "add_good.py").read_text()
                      + "\n# read reference/guide.md, reference/examples.md and reference/TTLangSpecification.md\n")
    assert gate.kernel_source_violations(kernel) == []


def test_the_skill_names_every_reference_file_and_what_to_read_for_each_kind_of_op():
    flat = " ".join(kp.SKILL.read_text().split())
    for name in REF_NAMES:
        assert f"reference/{name}" in flat, name
    for op in kp.OPS:
        assert f"`{op}`" in flat, op
    # the rules that answer the live probe's failures (element loops, an invented ttl.zeros, reduce_sum's arguments)
    for needle in ("`x_blk[i, j]`", "`ttl.zeros`", "`ttl.block.fill(0.0, shape=(...))`",
                   "`ttl.math.reduce_sum(expr, dims=[...], shape=(...))`", "There is no scaler argument",
                   "`@ttl.operation(grid=...)`", "block_count=2)`", "specification 0.12"):
        assert needle in flat, needle


SIM = SRC / "python" / "sim"


@pytest.mark.skipif(not (SIM / "__init__.py").is_file(), reason="SKIPPED: no tt-lang checkout with python/sim")
def test_the_skills_api_rules_match_the_simulator_at_the_pin():
    """The skill tells the coder which names exist. This reads the simulator's own source in the checkout (nothing is
    imported, so no device is touched) and checks each claim: the names the skill gives exist, the old names it warns
    about do not."""
    init = (SIM / "__init__.py").read_text()
    for present in ("self.operation = operation", "self.fill = block_module.fill",
                    "self.broadcast = block_module.broadcast", "self.reduce_sum = math_module.reduce_sum"):
        assert present in init, present
    for absent in ("self.kernel =", "self.zeros ="):
        assert absent not in init, absent
    reduce_sum = re.search(r"def reduce_sum\(([^)]*)\)", (SIM / "math.py").read_text()).group(1)
    assert "shape" in reduce_sum and "scaler" not in reduce_sum
    assert "block_count" in re.search(r"def make_dataflow_buffer_like\(([^)]*)\)",
                                      (SIM / "dfb.py").read_text()).group(1)
    assert "def broadcast(" not in (SIM / "math.py").read_text()


# ---- the latest skills (Task 14: "WE USE THE LATEST tenstorrent/skills and tt-lang skills available") -----------

def _behind_ttlang():
    from orchard import skillsfresh as sf
    return _fresh(sf.SOURCE_TTLANG, sf.BEHIND, behind=1, detail="HEAD 9b7ad24 is 1 commit behind upstream main.",
                  update="set TTLANG_PIN ... and run `tt-orchard kernel-env`")


def test_main_checks_the_skills_of_the_skill_directory_it_uses_and_may_write_the_cache(tmp_path, fake_kenv, env_ready,
                                                                                       skills_current, ttlang_skill):
    with FakeModel(lambda r: final("nothing")) as server:
        assert kp.main(probe_args(tmp_path, endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                       probe=lambda e, m: True, run_watched_fn=no_run) == 0
    assert skills_current["calls"] == [{"skill_dir": ttlang_skill, "write_cache": True}]


def test_a_stale_source_refuses_the_probe_with_the_update_command_and_starts_nothing(tmp_path, fake_kenv, env_ready,
                                                                                    skills_current, capsys):
    from orchard import skillsfresh as sf
    skills_current["results"] = [_behind_ttlang(), _fresh(sf.SOURCE_TTLANG_SKILL, sf.CURRENT)]
    asked = []
    code = kp.main(probe_args(tmp_path), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: asked.append(1) or True,
                   run_watched_fn=no_run)
    err = capsys.readouterr().err
    assert code == 2 and asked == [] and not (tmp_path / "out").exists()
    assert "1 commit behind" in err and "tt-orchard kernel-env" in err and "--allow-stale-skills" in err


def test_an_unknown_comparison_without_a_recent_current_check_refuses_the_probe(tmp_path, fake_kenv, env_ready,
                                                                               skills_current, capsys):
    from orchard import skillsfresh as sf
    skills_current["results"] = [_fresh(sf.SOURCE_TTLANG, sf.CURRENT),
                                 _fresh(sf.SOURCE_TTLANG_SKILL, sf.UNKNOWN, detail="It could not be compared.",
                                        update="WRONG HINT: re-sync the skill")]
    code = kp.main(probe_args(tmp_path), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    err = capsys.readouterr().err
    assert code == 2 and "could not be compared" in err
    # An unknown result states no update, even if one is attached (review I3).
    assert "WRONG HINT" not in err and "Compare again when GitHub answers" in err


def test_an_unknown_comparison_covered_by_a_recent_current_check_runs(tmp_path, fake_kenv, env_ready, skills_current):
    from orchard import skillsfresh as sf
    skills_current["results"] = [_fresh(sf.SOURCE_TTLANG, sf.CURRENT),
                                 _fresh(sf.SOURCE_TTLANG_SKILL, sf.UNKNOWN, cached={"state": sf.CURRENT})]
    with FakeModel(lambda r: final("nothing")) as server:
        code = kp.main(probe_args(tmp_path, endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                       probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 0


def test_allow_stale_skills_runs_and_the_report_records_it_with_the_shas(tmp_path, fake_kenv, env_ready,
                                                                         skills_current):
    from orchard import skillsfresh as sf
    skills_current["results"] = [_behind_ttlang(), _fresh(sf.SOURCE_TTLANG_SKILL, sf.CURRENT)]
    out = io.StringIO()
    with FakeModel(lambda r: final("nothing")) as server:
        code = kp.main(probe_args(tmp_path, endpoint=server.endpoint, allow_stale_skills=True), env=fake_kenv,
                       out=out, probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 0 and "stale skills allowed" in out.getvalue()
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    skills = data["header"]["skills"]
    assert skills["stale_allowed"] is True and skills["stale"] == [sf.SOURCE_TTLANG]
    (row,) = [f for f in skills["freshness"] if f["source"] == sf.SOURCE_TTLANG]
    assert row["local"] == "1" * 40 and row["remote"] == "2" * 40 and row["behind"] == 1 and row["stale"] is True
    for a in data["attempts"]:
        assert a["provenance"]["skills"]["stale_allowed"] is True
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "## Skill sources" in md and "Stale skills allowed (--allow-stale-skills)" in md
    assert "1" * 40 in md and "2" * 40 in md


def test_current_skills_are_recorded_and_not_called_stale(tmp_path, fake_kenv, env_ready, skills_current):
    with FakeModel(lambda r: final("nothing")) as server:
        kp.main(probe_args(tmp_path, endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                probe=lambda e, m: True, run_watched_fn=no_run)
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert data["header"]["skills"]["stale_allowed"] is False and data["header"]["skills"]["stale"] == []
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "## Skill sources" in md and "Stale skills allowed" not in md


def test_the_real_skills_status_checks_the_checkout_and_the_skill(fake_kenv, ttlang_skill, monkeypatch, skills_current):
    """ttlang_env.skills_status (not the fixture's fake) hands the checkout, the skill directory and the cache rule to
    skillsfresh."""
    from orchard import skillsfresh as sf
    seen = {}

    def fake(src, skill, *, cache=None, **kw):
        seen.update(src=src, skill=skill, write=cache.write)
        return []
    monkeypatch.setattr(sf, "check_kernel_sources", fake)
    skills_current["real"](fake_kenv, ttlang_skill, write_cache=False)
    assert seen == {"src": fake_kenv.src_dir, "skill": ttlang_skill, "write": False}


# kernel-env --check reports the same sources and changes nothing

def env_check_args(**kw):
    from types import SimpleNamespace
    base = dict(check=True, src_dir=None, venv_dir=None, tt_lang_skill_dir=None, allow_stale_skills=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_kernel_env_check_fails_on_a_stale_source_and_names_the_update(fake_kenv, env_ready, skills_current):
    from orchard import skillsfresh as sf
    skills_current["results"] = [_behind_ttlang(), _fresh(sf.SOURCE_TTLANG_SKILL, sf.CURRENT)]
    out = io.StringIO()
    assert ttlang_env.main(env_check_args(), env=fake_kenv, out=out) == 2
    text = out.getvalue()
    assert "1 commit behind" in text and "tt-orchard kernel-env" in text
    assert skills_current["calls"][-1]["write_cache"] is False


def test_kernel_env_check_passes_with_current_sources_or_the_override(fake_kenv, env_ready, skills_current):
    from orchard import skillsfresh as sf
    assert ttlang_env.main(env_check_args(), env=fake_kenv, out=io.StringIO()) == 0
    skills_current["results"] = [_behind_ttlang(), _fresh(sf.SOURCE_TTLANG_SKILL, sf.CURRENT)]
    out = io.StringIO()
    assert ttlang_env.main(env_check_args(allow_stale_skills=True), env=fake_kenv, out=out) == 0
    assert "stale skills allowed" in out.getvalue()


def test_kernel_env_check_fails_when_the_skill_files_are_missing(fake_kenv, env_ready, tmp_path):
    out = io.StringIO()
    assert ttlang_env.main(env_check_args(tt_lang_skill_dir=str(tmp_path / "none")), env=fake_kenv, out=out) == 2
    assert "does not exist" in out.getvalue()


def test_kernel_env_without_check_does_not_compare_with_upstream(fake_kenv, env_ready, skills_current):
    ttlang_env.main(env_check_args(check=False), env=fake_kenv, out=io.StringIO())
    assert skills_current["calls"] == []


# ---- Task 14 fix wave -------------------------------------------------------------------------------------------

def _needle(v: dict) -> str:
    """The construct a gate violation names, as the skill must spell it: `import torch` for "import torch",
    `from sim import` for "import from sim", the attribute's name for a read-back."""
    detail = v["detail"]
    if detail.startswith("import from "):
        return f"from {detail.removeprefix('import from ')} import"
    if detail.startswith("import "):
        return detail
    return detail.split()[-1]


def _shipped_reference_pys():
    out = [KERNELS / "upstream_eltwise_add.py"]
    for name in ("eltwise_add.py", "eltwise_broadcast_reduce.py"):
        if (SRC / "examples" / name).is_file():
            out.append(SRC / "examples" / name)
    return out


@pytest.mark.parametrize("path", _shipped_reference_pys(), ids=lambda p: p.name)
def test_the_skill_names_every_construct_the_gate_refuses_in_a_shipped_reference(path):
    """The coder is told to copy from these files. Every rule they break must be named in the skill, so a coder that
    copies a header line is warned before the gate refuses it (review I1: `from sim import ttl, ttnn`)."""
    gate = kp._template("kernel_check")
    flat = " ".join(kp.SKILL.read_text().split())
    violations = gate.kernel_source_violations(path)
    assert violations, "a shipped example with host code should break at least one rule"
    for v in violations:
        assert _needle(v) in flat, (path.name, v)


@pytest.mark.parametrize("path", _shipped_reference_pys(), ids=lambda p: p.name)
def test_the_skill_repeats_each_references_hardware_ci_tag(path):
    """Review I2: tt-lang's CI tags eltwise_broadcast_reduce.py `TTLANG_HARDWARE_CI: xfail-compiler` (run, but expect a
    non-zero exit on the hardware compiler). The skill must say so wherever a reference carries such a tag, and must not
    call that example a working kernel."""
    flat = " ".join(kp.SKILL.read_text().split())
    tags = re.findall(r"^#\s*(TTLANG_HARDWARE_CI:\s*\S+)", path.read_text(), flags=re.M)
    for tag in tags:
        assert " ".join(tag.split()) in flat, tag
    assert "`reference/eltwise_broadcast_reduce.py`: a working kernel" not in flat


def test_a_reference_that_is_not_a_regular_file_is_never_opened(fake_kenv, ttlang_skill, monkeypatch):
    """Review M3: the type is checked with lstat before the open, so a device node or a FIFO is never opened."""
    (ttlang_skill / "examples.md").unlink()
    os.mkfifo(ttlang_skill / "examples.md")
    opened = []
    real_open = os.open

    def recording_open(path, *a, **kw):
        opened.append(str(path))
        return real_open(path, *a, **kw)
    monkeypatch.setattr(ttlang_env.os, "open", recording_open)
    try:
        _, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
        seen = list(opened)                         # before the cleanup below opens the FIFO itself
    finally:
        _unblock(ttlang_skill / "examples.md")
    assert str(ttlang_skill / "examples.md") not in seen
    assert len(problems) == 1 and "not a regular file" in problems[0]


@pytest.mark.parametrize("linked", ["examples", "docs/sphinx"])
def test_a_reference_under_a_linked_directory_of_the_checkout_is_refused(fake_kenv, ttlang_skill, tmp_path, linked):
    """Review M3: O_NOFOLLOW covers only the last component. A directory between the checkout and the file that is a
    link is refused by name."""
    real = fake_kenv.src_dir / linked
    moved = tmp_path / "elsewhere" / linked.replace("/", "-")
    moved.parent.mkdir(parents=True, exist_ok=True)
    real.rename(moved)
    real.symlink_to(moved, target_is_directory=True)
    _, problems = ttlang_env.read_references(fake_kenv, ttlang_skill)
    assert problems and all("link" in p and str(real) in p for p in problems)


SPEC = SRC / "docs" / "sphinx" / "specs" / "TTLangSpecification.md"
INSTALLED_SKILL = Path.home() / ".claude" / "skills" / "ttm-tt-lang"


@pytest.mark.skipif(not SPEC.is_file(), reason="SKIPPED: no tt-lang checkout with docs/sphinx/specs")
def test_the_skills_spec_version_matches_the_checkouts_specification():
    """Review M7: the skill names the specification version of the checkout; when the pin moves, this fails until
    the text is updated."""
    from orchard import skillsfresh as sf
    flat = " ".join(kp.SKILL.read_text().split())
    (claimed,) = re.findall(r"`reference/TTLangSpecification\.md`: the specification of that tt-lang \(version "
                            r"(\d+\.\d+)\)", flat)
    assert claimed == sf.spec_version(SPEC.read_bytes())


@pytest.mark.skipif(not (INSTALLED_SKILL / "TTLangSpecification.md").is_file(),
                    reason="SKIPPED: the tt-lang skill is not installed here")
def test_the_skills_guide_version_matches_the_installed_skill():
    from orchard import skillsfresh as sf
    flat = " ".join(kp.SKILL.read_text().split())
    (claimed,) = re.findall(r"They were written for specification (\d+\.\d+)", flat)
    assert claimed == sf.spec_version((INSTALLED_SKILL / "TTLangSpecification.md").read_bytes())



def test_the_freshness_check_uses_the_same_bytes_the_coder_gets(tmp_path, fake_kenv, env_ready, skills_current,
                                                                ttlang_skill):
    """Review M8: the skill files are read once. The freshness check hashes the bytes that went into reference/, so a
    change between two reads cannot make the record disagree with the copy."""
    with FakeModel(lambda r: final("nothing")) as server:
        assert kp.main(probe_args(tmp_path, endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                       probe=lambda e, m: True, run_watched_fn=no_run) == 0
    adir = tmp_path / "out" / "add" / "attempt-1"
    assert skills_current["preread"] == {"guide.md": (adir / "reference" / "guide.md").read_bytes(),
                                         "examples.md": (adir / "reference" / "examples.md").read_bytes()}


def test_kernel_env_check_hands_its_reference_bytes_to_the_freshness_check(fake_kenv, env_ready, skills_current,
                                                                          ttlang_skill):
    ttlang_env.main(env_check_args(), env=fake_kenv, out=io.StringIO())
    assert skills_current["preread"] == {"guide.md": (ttlang_skill / "guide.md").read_bytes(),
                                         "examples.md": (ttlang_skill / "examples.md").read_bytes()}


def test_the_real_skills_status_passes_the_bytes_on(fake_kenv, ttlang_skill, monkeypatch, skills_current):
    from orchard import skillsfresh as sf
    seen = {}
    monkeypatch.setattr(sf, "check_kernel_sources", lambda src, skill, **kw: seen.update(kw) or [])
    skills_current["real"](fake_kenv, ttlang_skill, write_cache=False, preread={"guide.md": b"g"})
    assert seen["preread"] == {"guide.md": b"g"}


# ---- Task 14d: the operation's name is free ------------------------------------------------------------------------

def test_the_hardware_gate_lets_a_kernel_with_a_freely_named_operation_through(tmp_path, fake_kenv):
    """The hardware path re-applies the gate before any gozer call; a decorated function under another name passes it
    and the run reaches the (fake) gozer."""
    adir, attempt = _passed_attempt(tmp_path, fake_kenv, kernel="add_misnamed.py")
    calls = []
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], calls))
    assert len(calls) == 1 and calls[0][0] == "gozer" and hw["passed"] is True


@pytest.mark.parametrize("kernel,category", [("add_undecorated_named.py", "no_kernel"),
                                             ("add_two_misnamed.py", "forbidden_code")])
def test_the_hardware_gate_still_refuses_no_operation_and_two_operations(tmp_path, fake_kenv, kernel, category):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv, kernel=kernel)
    calls = []
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], calls))
    assert calls == [] and hw["passed"] is False and hw["category"] == category


def _skeleton() -> str:
    """The code block in the skill's step 3."""
    text = kp.SKILL.read_text()
    import textwrap
    blocks = re.findall(r"```python\n(.*?)\n *```", text, flags=re.S)
    assert len(blocks) == 1, "the skill has one code skeleton"
    return textwrap.dedent(blocks[0]) + "\n"


def test_the_skill_says_one_operation_function_with_any_name_and_shows_the_skeleton():
    """Task 14d: the name is free. The skill says so, has no literal-name rule left, and keeps the skeleton's shape."""
    text = kp.SKILL.read_text()
    flat = " ".join(text.split())
    assert "one function decorated with `@ttl.operation(...)`, with any name" in flat
    assert "literally" not in flat and "rename it to `OPERATION`" not in flat and "`OPERATION`" not in flat
    lines = [l for l in _skeleton().splitlines() if l.strip()]
    assert lines[0] == "import ttl" and lines[1] == "import ttnn"
    assert lines[2] == '@ttl.operation(grid="full")' and lines[3].startswith("def ")
    assert not lines[3].startswith("def OPERATION(")      # a name in the skeleton would read as a required name
    assert lines[3].rstrip().endswith("out):")


def test_the_skeleton_passes_the_gate_under_any_name(tmp_path):
    gate = kp._template("kernel_check")
    k = tmp_path / "kernel.py"
    k.write_text(_skeleton().replace("...", "pass"))
    assert gate.check_kernel_source(k) == (True, "")
    name = re.search(r"def (\w+)\(", _skeleton()).group(1)
    k.write_text(_skeleton().replace("...", "pass").replace(f"def {name}(", "def rmsnorm("))
    assert gate.check_kernel_source(k) == (True, "")


def test_the_skill_says_inputs_are_passed_in_the_order_of_the_spec_keys_then_out():
    """The harness calls the operation with the inputs in the order of kernel_spec.json's `inputs` keys, then out."""
    flat = " ".join(kp.SKILL.read_text().split())
    assert "in the order of the keys under `inputs`" in flat and "output tensor last" in flat
    harness = (kp.TEMPLATES / "kernel_harness.py").read_text()
    assert 'for name, shape in spec["inputs"].items():' in harness and "operation(*tt_ins, out)" in harness


def test_an_agent_alternating_two_commands_is_reported_by_the_watchdog(tmp_path, fake_kenv):
    """Task 14c, the wiring: through AgentStep and the probe's watchdog, two shell commands alternating (the recorded
    L4c pattern) give a turn_repeat finding with match "cycle" in the attempt's ledger."""
    a = call("shell", command="sed -n '1,70p' reference/eltwise_add.py")
    b = call("shell", command="sed -n '1,70p' kernel_spec.json")

    def script(request):
        k = turn(request)
        return (a if k % 2 == 0 else b) if k < 12 else final("done")
    with FakeModel(script) as server:
        run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL), max_turns=20)
    ledger = (tmp_path / "probe" / "add" / "attempt-1" / "ledger.jsonl").read_text()
    assert '"detector":"turn_repeat"' in ledger.replace(" ", "") and '"match":"cycle"' in ledger.replace(" ", "")


# ---- Task 14c: units, dtype and the output, as general facts -------------------------------------------------------

def test_the_skill_states_the_units_dtype_and_output_facts():
    flat = " ".join(kp.SKILL.read_text().split())
    for needle in ("`kernel_spec.json` shapes count elements", "a tile is 32 x 32 elements",
                   "bf16", "Buffer shapes and tensor slices count tiles",
                   "`IndexError: row slice 0:32 is outside the tensor, which has 2 tile(s) along it`",
                   "The result goes into `out`", "`ttl.copy(block, out[...])`", "returns nothing"):
        assert needle in flat, needle


def test_the_units_and_output_facts_hold_in_the_harness_and_the_simulator():
    """Each fact the skill states, checked where it comes from: the harness builds bf16 tile-layout inputs, makes `out`
    as zeros of the spec's output shape and reads it back after the call; the specification says a tile is 32 by 32
    scalar elements; the simulator's slice check counts tiles and words its IndexError the way the skill quotes it."""
    harness = (kp.TEMPLATES / "kernel_harness.py").read_text()
    for needle in ("ins[name] = x.to(torch.bfloat16)",
                   "ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)",
                   'out = ttnn.from_torch(torch.zeros(spec["output"], dtype=torch.bfloat16)',
                   "operation(*tt_ins, out)\n                        got = ttnn.to_torch(out).float()"):
        assert needle in harness, needle
    for op in kp.OPS:
        assert kp.load_spec(op)["dtype"] == "bfloat16", op
    if (SRC / "docs").is_dir():
        spec = (SRC / "docs" / "sphinx" / "specs" / "TTLangSpecification.md").read_text()
        assert "Shape unit can be either a tile (32 by 32 scalar elements)" in spec
        sim = (SRC / "python" / "sim" / "ttnnsim.py").read_text()
        assert 'f"{dim_name} slice {start}:{stop} is outside the tensor, which has "' in sim
        assert 'f"{extent} {unit}(s) along it"' in sim
        assert "TILE_SHAPE: Shape = (_TILE_SIZE, _TILE_SIZE)" in (SRC / "python" / "sim" / "constants.py").read_text()


def test_the_skill_keeps_one_fenced_skeleton_and_the_worked_kernels_live_in_the_library():
    """Worked solutions are allowed (the operator's decision of 2026-10-27) but they go in reference/library_*.py, not in
    the skill text, which keeps one fenced skeleton and stays short."""
    text = kp.SKILL.read_text()
    assert len(re.findall(r"```", text)) == 2
    assert len(text.splitlines()) <= 100


@needs_sim
def test_the_probes_recheck_passes_a_kernel_with_a_freely_named_operation(tmp_path, kenv):
    """The probe-owned re-check (the shipped check on a clean copy, on the real simulator) agrees with the agent's own
    check: the one @ttl.operation function is the operation, whatever its name."""
    from orchard.stallwatch import run_watched as real_run_watched
    adir = tmp_path / "probe" / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=kenv)
    (adir / "kernel.py").write_text((KERNELS / "add_misnamed.py").read_text())
    got = kp.authoritative_check(adir, op="add", env=kenv, run_watched=real_run_watched)
    assert got["passed"] is True, got


def test_the_hardware_phase_runs_the_shipped_harness_which_asks_the_gate_for_the_operation(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv, kernel="add_misnamed.py")
    calls = []
    kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], calls))
    (argv,) = calls
    assert str(kp.TEMPLATES / "kernel_harness.py") in argv
    harness = (kp.TEMPLATES / "kernel_harness.py").read_text()
    assert "operation = operation_of(module, a.kernel)" in harness


def test_the_skill_says_each_write_runs_the_check_and_shows_where_and_hint():
    """Task 14d: the live coder wrote kernel.py six times in a row without running the check once. The tool now runs
    it after each write (ProbeTools), and the skill says so, so the coder reads the output below the write."""
    flat = " ".join(kp.SKILL.read_text().split())
    assert "Each write of `kernel.py` runs `python3 kernel_check.py` and shows its output below the write's result." in flat
    assert "`where:` (the failing line)" in flat and "`hint:`" in flat and "`numbers:`" in flat


# ---- the check runs after every write of kernel.py (pilot 4, 2026-10-17) -------------------------------------------
# Pilot 4: the coder wrote kernel.py three times and never ran kernel_check.py, although the skill says to, so it never
# saw the failing line or the hint. The tool now runs the shipped check after each write and appends its output.

def _probe_tools(tmp_path, check_body):
    adir = tmp_path / "add" / "attempt-1"
    adir.mkdir(parents=True)
    (adir / "kernel_check.py").write_text(check_body)
    return kp.attempt_tools(adir), adir


def test_attempt_tools_run_the_check_after_a_write_of_kernel_py(tmp_path):
    tools, adir = _probe_tools(tmp_path, "print('FAIL: tt_lang_error: error: boom')\nprint('where: kernel.py line 3: x')\n")
    out = tools.call("write_file", json.dumps({"path": "kernel.py", "content": "x = 1\n"}))
    assert (adir / "kernel.py").read_text() == "x = 1\n"
    assert out.startswith("wrote 6 characters to kernel.py")
    assert "kernel_check.py ran after this write" in out and "FAIL: tt_lang_error: error: boom" in out
    assert "where: kernel.py line 3: x" in out


def test_other_files_and_unchanged_rewrites_do_not_run_the_check(tmp_path):
    tools, adir = _probe_tools(tmp_path, "open('ran', 'a').write('x')\nprint('FAIL: x: y')\n")
    assert "ran after" not in tools.call("write_file", json.dumps({"path": "notes.txt", "content": "a"}))
    assert not (adir / "ran").exists()
    tools.call("write_file", json.dumps({"path": "kernel.py", "content": "x = 1\n"}))
    again = tools.call("write_file", json.dumps({"path": "kernel.py", "content": "x = 1\n"}))
    assert again.startswith("unchanged:") and "ran after" not in again
    assert (adir / "ran").read_text() == "x"                     # once, for the one real write


def test_a_refused_write_does_not_run_the_check(tmp_path):
    tools, adir = _probe_tools(tmp_path, "open('ran', 'a').write('x')\n")
    out = tools.call("write_file", json.dumps({"path": "../kernel.py", "content": "x"}))
    assert out.startswith("refused:") and not (adir / "ran").exists()


def test_the_appended_check_output_is_bounded(tmp_path):
    tools, _ = _probe_tools(tmp_path, "print('FAIL: x: ' + 'a' * 20000)\n")
    out = tools.call("write_file", json.dumps({"path": "kernel.py", "content": "x"}))
    assert len(out) < 3000


def test_a_write_in_a_real_attempt_shows_the_check_to_the_model(tmp_path, fake_kenv):
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(scripted(call("write_file", path="kernel.py", content="x = 1\n"), final("done"))) as server:
        run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
        tool_msgs = [m["content"] for m in server.requests[1]["messages"] if m["role"] == "tool"]
    assert any("wrote 6 characters" in t and "kernel_check.py ran after this write" in t for t in tool_msgs), tool_msgs


# ---- read_file counts lines in the probe (pilot 1 and pilot 5, 2026-10-17) -----------------------------------------
# Twice the coder passed grep -n line numbers (1490, 1495, 1415) as read_file offsets, which count characters, got the
# top of the file, and looped until the watchdog ended the attempt. The skill's advice to use sed was ignored in one of
# the two. In the probe, offset and limit now count lines, the way the coder keeps using them.

def _numbered_file(tmp_path, n=2000):
    tools, adir = _probe_tools(tmp_path, "pass\n")
    (adir / "reference").mkdir()
    (adir / "reference" / "spec.md").write_text("".join(f"line {i}\n" for i in range(1, n + 1)))
    return tools, adir


def _read(tools, **args):
    return tools.call("read_file", json.dumps({"path": "reference/spec.md", **args}))


def test_probe_read_file_offset_and_limit_count_lines(tmp_path):
    tools, _ = _numbered_file(tmp_path)
    out = _read(tools, offset=1495, limit=3)
    assert out.startswith("line 1495\nline 1496\nline 1497\n") and "line 1498" not in out.split("[")[0]
    assert "[lines 1495 to 1497 of 2000; read on with offset=1498]" in out


def test_probe_read_file_accepts_a_limit_the_model_sends_as_a_string(tmp_path):
    tools, _ = _numbered_file(tmp_path)
    assert _read(tools, offset=10, limit="2").startswith("line 10\nline 11\n[lines 10 to 11 of 2000")


def test_probe_read_file_without_limit_pages_by_whole_lines_within_the_output_limit(tmp_path):
    tools, _ = _numbered_file(tmp_path)
    out = _read(tools, offset=1)
    body, footer = out.rsplit("\n[", 1)
    assert len(body) <= tools.limit and body.endswith(tuple("0123456789")) and body.startswith("line 1\n")
    nxt = int(re.search(r"offset=(\d+)\]", footer)[1])
    assert body.splitlines()[-1] == f"line {nxt - 1}"            # the page ends on a whole line, and the footer's next line follows it
    assert _read(tools, offset=nxt).startswith(f"line {nxt}\n")


def test_probe_read_file_of_a_small_file_returns_all_of_it(tmp_path):
    tools, adir = _probe_tools(tmp_path, "pass\n")
    (adir / "kernel_spec.json").write_text('{"op": "add"}\n')
    assert tools.call("read_file", json.dumps({"path": "kernel_spec.json"})) == '{"op": "add"}\n'


def test_probe_read_file_past_the_end_says_how_many_lines_there_are(tmp_path):
    tools, _ = _numbered_file(tmp_path, n=5)
    out = _read(tools, offset=99)
    assert out.startswith("error:") and "5 lines" in out


@pytest.mark.parametrize("args", [{"offset": -1}, {"offset": "x"}, {"offset": 1, "limit": 0}, {"offset": 1, "limit": "y"},
                                  {"offset": True}])
def test_probe_read_file_refuses_a_bad_offset_or_limit(tmp_path, args):
    tools, _ = _numbered_file(tmp_path)
    assert _read(tools, **args).startswith("error:")


def test_probe_read_file_keeps_the_directory_and_link_rules(tmp_path):
    tools, adir = _numbered_file(tmp_path)
    assert tools.call("read_file", json.dumps({"path": "../../secret", "offset": 1})).startswith("refused:")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    (adir / "link.txt").symlink_to(outside)
    assert tools.call("read_file", json.dumps({"path": "link.txt", "offset": 1})).startswith("refused:")
    assert tools.call("read_file", json.dumps({"path": "reference", "offset": 1})).startswith("error:")


def test_probe_tool_schema_says_lines_and_the_step_sends_it(tmp_path, fake_kenv):
    tools, _ = _probe_tools(tmp_path / "x", "pass\n")
    props = {s["function"]["name"]: s["function"]["parameters"]["properties"] for s in tools.schemas}["read_file"]
    assert "line" in props["offset"]["description"] and "limit" in props
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(scripted(final("done"))) as server:
        run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
        sent = {s["function"]["name"]: s for s in server.requests[0]["tools"]}["read_file"]
    assert "limit" in sent["function"]["parameters"]["properties"]


def test_the_generic_read_file_still_counts_characters():
    from orchard import agent
    props = {s["function"]["name"]: s["function"]["parameters"]["properties"] for s in agent.TOOL_SCHEMAS}["read_file"]
    assert "character" in props["offset"]["description"] and "limit" not in props


def test_the_skill_says_read_file_offset_and_limit_count_lines():
    flat = " ".join(kp.SKILL.read_text().split())
    assert "`read_file` with offset N and limit 60 reads from there (offset and limit count lines)" in flat
    assert "count characters" not in flat


# ---- an unchanged rewrite after a failing check (pilot 6, 2026-10-17) ----------------------------------------------
# The generic reply to an unchanged write says "end the step now". After a failing check that is the wrong advice: the
# coder rewrote the same text three times, the watchdog saw the loop and ended the attempt at 18 of 40 turns.

def _write(tools, content="x = 1\n"):
    return tools.call("write_file", json.dumps({"path": "kernel.py", "content": content}))


def test_an_unchanged_write_after_a_failing_check_says_the_check_failed_and_to_change_the_code(tmp_path):
    tools, _ = _probe_tools(tmp_path, "print('FAIL: shape_mismatch: error: ValueError: tiles 2 vs 1')\nprint('where: kernel.py line 4: y')\n")
    _write(tools)
    again = _write(tools)
    assert again.startswith("unchanged: kernel.py already holds exactly this content")
    assert "its last check failed" in again and "FAIL: shape_mismatch: error: ValueError: tiles 2 vs 1" in again
    assert "change the code" in again.lower() and "end the step" not in again


def test_an_unchanged_write_after_a_passing_check_still_tells_the_model_to_end_the_step(tmp_path):
    tools, _ = _probe_tools(tmp_path, "print('PASS')\n")
    _write(tools)
    again = _write(tools)
    assert again.startswith("unchanged:") and "end the step now" in again and "failed" not in again


def test_an_unchanged_write_before_any_check_keeps_the_generic_reply(tmp_path):
    tools, adir = _probe_tools(tmp_path, "print('FAIL: x: y')\n")
    (adir / "kernel.py").write_text("x = 1\n")                   # written outside the tool, so no check has run
    assert "end the step now" in _write(tools)


def test_a_changed_write_after_a_failing_check_runs_the_check_again_and_the_latest_result_counts(tmp_path):
    tools, adir = _probe_tools(tmp_path, "import pathlib\nprint('PASS' if 'good' in pathlib.Path('kernel.py').read_text() else 'FAIL: c: bad')\n")
    _write(tools, "bad\n")
    assert "FAIL: c: bad" in _write(tools, "bad\n")
    assert "kernel_check.py ran after this write" in _write(tools, "good\n")
    assert "end the step now" in _write(tools, "good\n")          # the latest check passed


def test_a_check_that_printed_neither_pass_nor_fail_counts_as_failed(tmp_path):
    tools, _ = _probe_tools(tmp_path, "import sys\nsys.exit(3)\n")
    _write(tools)
    again = _write(tools)
    assert "its last check failed" in again and "exit 3" in again and "end the step" not in again


# ---- the coder's decode speed is measured before each attempt (2026-10-17) ---------------------------------------------
# A probe batch ran for hours at a fraction of the coder's speed, and the numbers it produced (turns, wall time, request
# timeouts) were not comparable with the earlier runs. The probe now measures the coder before each attempt, records it,
# and stops when the coder is confirmed slow, unless --allow-slow-coder.

OK_SPEED = {"state": "ok", "tok_s": 38.7, "baseline": 38.7, "ratio": 1.0, "message": "38.7 tok/s, 100% of the best seen"}
SLOW_SPEED = {"state": "slow", "tok_s": 9.0, "baseline": 38.7, "ratio": 0.233, "message": "SLOW: 9.0 tok/s, 23% of the best seen"}


def speeds(*dicts):
    """A speed_check that answers the given dicts in order (the last repeats) and records what it was asked."""
    asked, it = [], list(dicts)

    def check(endpoint, model):
        asked.append((endpoint, model))
        return it.pop(0) if len(it) > 1 else it[0]
    check.asked = asked
    return check


def main_with(tmp_path, fake_kenv, speed_check, *, attempts=2, **kw):
    out = io.StringIO()
    with FakeModel(lambda r: final("nothing")) as server:
        code = kp.main(probe_args(tmp_path, attempts=attempts, endpoint=server.endpoint, **kw), env=fake_kenv, out=out,
                       probe=lambda e, m: True, run_watched_fn=no_run, speed_check=speed_check)
    return code, out.getvalue(), server


def test_the_speed_is_measured_before_each_attempt_and_recorded_in_its_attempt_json(tmp_path, fake_kenv, env_ready):
    check = speeds(OK_SPEED)
    code, text, server = main_with(tmp_path, fake_kenv, check)
    assert code == 0 and check.asked == [(server.endpoint, "fake")] * 2
    for n in (1, 2):
        rec = json.loads((tmp_path / "out" / "add" / f"attempt-{n}" / "attempt.json").read_text())
        assert rec["coder_speed"] == OK_SPEED
    assert "coder decode speed before attempt 1: 38.7 tok/s" in text


def test_a_confirmed_slow_coder_stops_the_probe_before_the_next_attempt_with_exit_six(tmp_path, fake_kenv, env_ready):
    code, text, _ = main_with(tmp_path, fake_kenv, speeds(OK_SPEED, SLOW_SPEED), attempts=3)
    assert code == kp.EXIT_SLOW == 6
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert [a["attempt"] for a in data["attempts"]] == [1]                       # attempt 2 never started
    assert not (tmp_path / "out" / "add" / "attempt-2").exists()
    assert "coder slow before attempt 2" in data["header"]["stopped"] and "23%" in data["header"]["stopped"]
    assert "stopped: coder slow before attempt 2" in text and "--allow-slow-coder" in text


def test_a_slow_coder_does_not_start_the_hardware_phase(tmp_path, fake_kenv, env_ready):
    code, _, _ = main_with(tmp_path, fake_kenv, speeds(OK_SPEED, SLOW_SPEED), attempts=2, phase="hw")
    assert code == 6                                                              # run_watched_fn is no_run: gozer would raise


def test_allow_slow_coder_keeps_going_and_records_the_slow_state(tmp_path, fake_kenv, env_ready):
    code, text, _ = main_with(tmp_path, fake_kenv, speeds(SLOW_SPEED), allow_slow_coder=True)
    assert code == 0
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert [a["coder_speed"]["state"] for a in data["attempts"]] == ["slow", "slow"]
    assert "coder is slow, continuing (--allow-slow-coder)" in text and "stopped" not in data["header"]


def test_a_failed_speed_measurement_does_not_stop_the_probe(tmp_path, fake_kenv, env_ready):
    err = {"state": "error", "tok_s": None, "baseline": None, "ratio": None, "message": "connection refused"}
    code, text, _ = main_with(tmp_path, fake_kenv, speeds(err))
    assert code == 0 and "could not measure the coder's speed: connection refused" in text


def test_a_suspect_coder_runs_the_attempt(tmp_path, fake_kenv, env_ready):
    suspect = dict(OK_SPEED, state="suspect", tok_s=15.0, ratio=0.39)
    code, _, _ = main_with(tmp_path, fake_kenv, speeds(suspect))
    assert code == 0


def test_without_a_speed_check_nothing_is_measured_and_the_record_says_so(tmp_path, fake_kenv, env_ready):
    code, text, _ = main_with(tmp_path, fake_kenv, None, attempts=1)
    rec = json.loads((tmp_path / "out" / "add" / "attempt-1" / "attempt.json").read_text())
    assert code == 0 and rec["coder_speed"] is None and "decode speed" not in text


def test_the_report_lists_the_speed_before_each_attempt_and_warns_about_the_unlike_ones(tmp_path, fake_kenv, env_ready):
    main_with(tmp_path, fake_kenv, speeds(OK_SPEED, dict(SLOW_SPEED, state="suspect")), attempts=2)
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "Coder decode speed, measured just before each attempt" in md
    assert "attempt 1: 38.7 tok/s (ok)" in md and "attempt 2: 9.0 tok/s (suspect)" in md
    assert "not comparable" in md


def test_the_report_has_no_speed_section_when_nothing_was_measured(tmp_path, fake_kenv, env_ready):
    main_with(tmp_path, fake_kenv, None, attempts=1)
    assert "decode speed" not in (tmp_path / "out" / "kernel-probe-report.md").read_text()


def test_the_command_line_gives_the_probe_the_real_speed_check_and_the_override_flag(monkeypatch, tmp_path):
    from orchard import cli
    seen = {}

    def fake_main(args, **kw):
        seen["args"], seen["kw"] = args, kw
        return 0
    monkeypatch.setattr(kp, "main", fake_main)
    cli.main(["kernel-probe", "--op", "add", "--endpoint", "http://127.0.0.1:9/v1", "--model", "m",
              "--allow-slow-coder"], env={"HOME": str(tmp_path)}, stdout=io.StringIO())
    assert seen["args"].allow_slow_coder is True and callable(seen["kw"]["speed_check"])
    cli.main(["kernel-probe", "--op", "add", "--endpoint", "http://127.0.0.1:9/v1", "--model", "m"],
             env={"HOME": str(tmp_path)}, stdout=io.StringIO())
    assert seen["args"].allow_slow_coder is False


def test_a_busy_coder_runs_the_attempt_and_the_report_says_the_timings_are_not_comparable(tmp_path, fake_kenv, env_ready):
    busy = {"state": "busy", "tok_s": 7.0, "baseline": 38.7, "ratio": 0.18,
            "message": "1 other request was running on the server, so this 7.0 tok/s sample does not say whether the "
                       "server is slow"}
    code, text, _ = main_with(tmp_path, fake_kenv, speeds(busy), attempts=2)
    assert code == 0 and "(busy)" in text
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "attempt 1: 7.0 tok/s (busy)" in md and "not comparable" in md and "another client" in md


# ---- a cap on the length of the coder's replies (2026-10-17) -----------------------------------------------------------
# 6 of 14 attempts at temperature 0.7 ended on two 900 s request timeouts, against 1 of 14 at temperature 0. The largest
# reply any attempt ever produced is 2706 tokens; a timed-out request is a reply of about 8K tokens decoding at about
# 9 tok/s. The probe's replies are capped well above the real ones and far below the generic 16384, so a degenerate reply
# ends as `finish_reason: length` (handled, with a nudge) in minutes and not as a 900 s timeout.

def test_the_probe_budget_names_a_reply_cap_well_below_the_generic_one_and_above_the_longest_real_reply():
    from orchard.defaults import AGENT_MAX_TOKENS
    assert kp.BUDGET["max_tokens"] == 4096 and kp.BUDGET["max_tokens"] < AGENT_MAX_TOKENS
    assert kp.BUDGET["max_tokens"] > 2706                      # the longest completion in 27 recorded attempts


def test_every_request_of_an_attempt_carries_the_probe_reply_cap(tmp_path, fake_kenv):
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(scripted(call("shell", command="ls"), final("done"))) as server:
        run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
        sent = [r["max_tokens"] for r in server.requests]
    assert sent == [kp.BUDGET["max_tokens"]] * 2


def test_a_reply_cut_at_the_cap_is_nudged_not_waited_for(tmp_path, fake_kenv):
    from fake_model import truncated
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(scripted(truncated(kp.BUDGET["max_tokens"]), final("done"))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
        later = [m["content"] for m in server.requests[1]["messages"] if m["role"] == "user"]
    from orchard.agent import TRUNCATION_NUDGE
    assert attempt["turns"] >= 1 and TRUNCATION_NUDGE in later


def test_the_report_header_records_the_reply_cap_with_the_other_budgets():
    header = kp.report_header(env=None, coder_model="m", probe_id="p", ops=["add"], hw=None,
                              tt_lang={"sim_version": "x"}, ignored_records=0)
    assert header["budget"]["max_tokens"] == 4096


# ---- the environment of the probe's children (review finding 5) -------------------------------------------------

SECRET_ENV = {"HF_TOKEN": "hf_planted", "GH_TOKEN": "gh_planted", "SSH_AUTH_SOCK": "/tmp/planted-agent.sock",
              "AWS_SECRET_ACCESS_KEY": "planted", "OPENAI_API_KEY": "sk-planted"}


@pytest.fixture
def secret_env(monkeypatch):
    for k, v in SECRET_ENV.items():
        monkeypatch.setenv(k, v)
    return SECRET_ENV


def _assert_no_operator_secrets(env: dict, *, real_home: str):
    assert isinstance(env, dict), "the child must get an explicit environment, not the probe's own"
    for k in SECRET_ENV:
        assert k not in env, k
    assert env["HOME"] != real_home and not env["HOME"].startswith(real_home + "/.")
    assert env["PATH"]
    assert env["HF_TOKEN_PATH"].startswith(env["HOME"])        # a token path that does not exist


def test_the_authoritative_check_child_gets_a_clean_environment(tmp_path, fake_kenv, secret_env):
    seen = {}

    def fake(argv, **kw):
        seen.update(kw)
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"passed": False, "category": "numeric_mismatch",
                                                                  "first_line": "x", "sets": []}))
        return kp.Watched(1, None, False, 0.1)

    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel.py").write_text("# kernel\n")
    kp.authoritative_check(adir, op="add", env=fake_kenv, run_watched=fake, tt_lang=FACTS)
    env = seen["env"]
    _assert_no_operator_secrets(env, real_home=str(Path.home()))
    # The simulator child is not leased, so it keeps the no-chip mask. Its HOME is in the probe's own verdict
    # directory, not in the attempt directory the agent can write.
    assert env["TT_VISIBLE_DEVICES"] == NO_CHIP and env["TT_METAL_VISIBLE_DEVICES"] == NO_CHIP
    assert Path(env["HOME"]).resolve().is_relative_to(kp.verdict_dir(tmp_path, "add", 1).resolve())
    assert env["ORCHARD_RUN_DIR"] == str(adir.resolve())        # so sweep_strays can find what the child leaves


def test_the_smoke_test_and_the_hardware_run_get_a_clean_environment_and_no_device_mask(tmp_path, fake_kenv,
                                                                                         secret_env):
    seen = []

    def fake(argv, **kw):
        seen.append(kw["env"])
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET],
                                                                      "error": None}))
        else:
            with open(kw["log_path"], "a") as f:
                f.write("PASSED!\n")
        return kp.Watched(0, None, False, 1.0)

    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=fake)
    assert len(seen) == 2
    for env in seen:
        _assert_no_operator_secrets(env, real_home=str(Path.home()))
        # gozer sets TT_VISIBLE_DEVICES for its child; a no-chip mask left in TT_METAL_VISIBLE_DEVICES would hide the
        # leased chip from it.
        assert "TT_VISIBLE_DEVICES" not in env and "TT_METAL_VISIBLE_DEVICES" not in env
    assert seen[1]["ORCHARD_RUN_DIR"] == str(adir.resolve())


def test_the_gozer_variables_reach_the_hardware_child(tmp_path, fake_kenv, monkeypatch):
    """The lease store is found through GOZER_ROOT; a clean environment must not lose it."""
    monkeypatch.setenv("GOZER_ROOT", "/some/gozer/root")
    seen = []

    def fake(argv, **kw):
        seen.append(kw["env"])
        return kp.Watched(0, None, False, 1.0)

    kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=fake)
    assert seen[0]["GOZER_ROOT"] == "/some/gozer/root"


def test_the_agents_background_process_is_swept_after_the_check_as_well(tmp_path, fake_kenv, monkeypatch):
    """A kernel the check imports can start a background process. The sweep before the verdict cannot see it."""
    order = []
    monkeypatch.setattr(kp, "sweep_strays", lambda d, **kw: order.append("sweep") or [])

    def fake(argv, **kw):
        order.append("check")
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"passed": False, "category": "numeric_mismatch",
                                                                  "first_line": "x", "sets": []}))
        return kp.Watched(1, None, False, 0.1)

    with FakeModel(scripted(call("write_file", path="kernel.py", content="# a kernel\n"), final("done"))) as server:
        run_attempt(tmp_path, fake_kenv, server, run_watched=fake)
    assert order == ["sweep", "check", "sweep"]


def test_the_hardware_run_is_swept_after_gozer_returns(tmp_path, fake_kenv, monkeypatch):
    """The kernel runs on a leased chip; a process it leaves running must be stopped when the lease is released."""
    order = []
    monkeypatch.setattr(kp, "sweep_strays", lambda d, **kw: order.append(("sweep", Path(d).name)) or
                        [{"pid": 1234, "cmdline": "sleep", "stopped": True}])
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)

    def fake(argv, **kw):
        order.append("gozer")
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET],
                                                                  "error": None}))
        return kp.Watched(0, None, False, 1.0)

    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=fake)
    assert order == ["gozer", ("sweep", "attempt-1")]
    assert hw["swept"] == [{"pid": 1234, "cmdline": "sleep", "stopped": True}]


def test_a_credential_named_gozer_variable_is_dropped_not_passed_and_not_fatal(fake_kenv, monkeypatch):
    monkeypatch.setenv("GOZER_AUTH_TOKEN", "planted")
    env = kp.child_env(fake_kenv.scratch_dir, hardware=True)
    assert "GOZER_AUTH_TOKEN" not in env


# ---- a queued gozer request is "queued, not run" (review finding 9) ---------------------------------------------


def queued_fake(calls, *, text="queued: ticket 3fa9c1\n"):
    """A fake run_watched for a busy box: `gozer run` prints its ticket and exits 10 without running the command."""
    def fake(argv, **kw):
        calls.append(list(argv))
        with open(kw["log_path"], "a") as f:
            f.write(text)
        return kp.Watched(10, None, False, 0.5)
    return fake


class Canceller:
    def __init__(self, error=None):
        self.tickets, self.error = [], error

    def __call__(self, ticket):
        self.tickets.append(ticket)
        return self.error


def test_a_queued_smoke_test_is_not_a_failed_one_and_its_ticket_is_cancelled(tmp_path, fake_kenv):
    calls, cancel, seen = [], Canceller(), {}
    ok, tail = kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=queued_fake(calls), record=seen,
                             cancel=cancel)
    assert ok is False and cancel.tickets == ["3fa9c1"]
    assert seen["queued"] == {"ticket": "3fa9c1", "cancel_error": None}
    # run_hardware stops there, runs no kernel, and says "queued", which the report must not call a failed test.
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    calls.clear()
    state = {}
    hw = kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                         run_watched=queued_fake(calls), state=state, cancel=Canceller())
    assert len(calls) == 1 and hw["available"] is False and hw["queued"]["ticket"] == "3fa9c1"
    assert state["smoke"]["queued"]["ticket"] == "3fa9c1"
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="t", ops=["add"], hw=hw, tt_lang=FACTS)
    kp.write_report(tmp_path, [attempt], header)
    text = (tmp_path / "kernel-probe-report.md").read_text()
    assert "queued, not run" in text and "smoke test (upstream's eltwise_add.py on one chip) failed" not in text


def test_a_queued_smoke_test_with_no_ticket_in_its_output_or_a_failed_cancel_is_reported(tmp_path, fake_kenv):
    cancel, seen = Canceller(), {}
    kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=queued_fake([], text="something else\n"),
                  record=seen, cancel=cancel)
    assert cancel.tickets == [] and "no ticket" in seen["queued"]["cancel_error"]
    bad, seen = Canceller(error="no such ticket: 3fa9c1"), {}
    kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=queued_fake([]), record=seen, cancel=bad)
    assert seen["queued"] == {"ticket": "3fa9c1", "cancel_error": "no such ticket: 3fa9c1"}


def test_a_queued_kernel_run_is_recorded_as_queued_cancelled_and_blocks_the_probe(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    cancel = Canceller()
    with pytest.raises(kp.ProbeBlocked, match="queued, not run"):
        kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=queued_fake([]),
                          cancel=cancel)
    assert cancel.tickets == ["3fa9c1"]
    hw = json.loads((adir / "attempt.json").read_text())["hw"]
    assert hw["passed"] is False and hw["category"] == "queued" and hw["queued"]["ticket"] == "3fa9c1"


def test_cancel_ticket_runs_gozer_cancel_and_reports_a_refusal():
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "cancelled 3fa9c1\n", ""

    assert kp.cancel_ticket("3fa9c1", run=lambda argv, **kw: calls.append(list(argv)) or Done()) is None
    assert calls == [["gozer", "cancel", "3fa9c1"]]

    class Refused:
        returncode, stdout, stderr = 1, "no such ticket: 3fa9c1\n", ""

    assert "no such ticket" in kp.cancel_ticket("3fa9c1", run=lambda argv, **kw: Refused())

    def boom(argv, **kw):
        raise OSError("no gozer")
    assert "no gozer" in kp.cancel_ticket("3fa9c1", run=boom)
    # a ticket that could pass as an option is not handed to gozer
    assert kp.cancel_ticket("--help", run=lambda argv, **kw: pytest.fail("ran gozer")) is not None


def test_a_run_the_watcher_killed_is_a_hang_even_if_its_exit_code_is_10(fake_kenv):
    def killed(argv, **kw):
        return kp.Watched(10, "stalled: no output and no CPU use for 60 s", True, 61.0)
    cancel = Canceller()
    with pytest.raises(kp.ProbeBlocked, match="hung"):
        kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=killed, cancel=cancel)
    assert cancel.tickets == []


def test_the_skill_states_the_two_use_row_rule_without_a_solution():
    """Pilot rms1 (2026-10-25): 2 of 3 rmsnorm attempts deadlocked (`Deadlock detected: all generators blocked`) because the
    reader sent each tile once while the compute thread waited for the same tiles a second time. The general rule is in the
    skill; the worked kernel is not (test_the_skill_holds_no_worked_solution_for_a_measured_op)."""
    flat = " ".join(kp.SKILL.read_text().split())
    assert "Deadlock detected: all generators blocked" in flat
    assert "used twice" in flat


def test_the_skill_states_the_general_facts_the_claude_rmsnorm_run_needed():
    """2026-10-27: a Claude agent passed rmsnorm in 2 attempts with the coder's materials; its one failure was a Python number
    combined with a block, and it listed six facts the skill left out (docs/run-logs/2026-10-27-rmsnorm-by-claude.md). The
    skill states them as general facts, not as a solution."""
    flat = " ".join(kp.SKILL.read_text().split())
    assert "A Python number cannot be an operand of a block expression" in flat and "`ttl.block.fill(value, shape=(1, 1))`" in flat
    assert "column 0 of each tile" in flat and "`ttl.block.broadcast(..., dims=[-1], shape=...)`" in flat
    assert "an element count (`x.shape[1]`) and a tile count" in flat
    assert "A block can be used in several statements inside one `with`" in flat
    assert "A buffer's shape may cover a whole row of tiles" in flat
    assert "Python arithmetic on shapes, such as `1.0 / x.shape[1]`" in flat and "captured by the thread functions" in flat


def test_main_records_the_library_mode_and_leaves_out_the_op_under_test_for_others(tmp_path, fake_kenv, env_ready,
                                                                                 ttlang_skill):
    with FakeModel(lambda r: final("nothing")) as server:
        code = kp.main(probe_args(tmp_path, op="rmsnorm", library="others", endpoint=server.endpoint), env=fake_kenv,
                       out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 0
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    names = {r["name"] for r in data["header"]["references"]}
    assert "reference/library_softmax.py" in names and "reference/library_rmsnorm.py" not in names
    assert data["header"]["library"] == {"mode": "others", "note": None}
    assert (tmp_path / "out" / "rmsnorm" / "attempt-1" / "reference" / "library_center.py").is_file()
    assert not (tmp_path / "out" / "rmsnorm" / "attempt-1" / "reference" / "library_rmsnorm.py").exists()


def test_main_says_in_the_report_when_the_library_held_the_op(tmp_path, fake_kenv, env_ready, ttlang_skill):
    with FakeModel(lambda r: final("nothing")) as server:
        kp.main(probe_args(tmp_path, op="rmsnorm", endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                probe=lambda e, m: True, run_watched_fn=no_run)
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "Worked-kernel library mode: all." in md and "can adapt a worked kernel" in md


def test_main_refuses_an_unknown_library_mode_and_starts_nothing(tmp_path, fake_kenv, env_ready, ttlang_skill, capsys):
    code = kp.main(probe_args(tmp_path, library="some"), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    assert code == 2 and not (tmp_path / "out").exists() and "unknown --library" in capsys.readouterr().err


# ---- the probe's read_file answers an argument it does not know (0.9.36) -------------------------------------------

def test_the_probe_read_file_takes_limit_and_names_its_arguments_for_an_unknown_one(tmp_path):
    tools, adir = _probe_tools(tmp_path, "pass\n")
    assert isinstance(tools, kp.ProbeTools)       # the probe's agent gets the line-based reader (attempt_tools)
    (adir / "notes.txt").write_text("".join(f"line {i}\n" for i in range(1, 31)))
    ok = tools.call("read_file", json.dumps({"path": "notes.txt", "offset": 5, "limit": 3}))
    assert ok.startswith("line 5\nline 6\nline 7") and "[lines 5 to 7 of 30" in ok
    bad = tools.call("read_file", json.dumps({"path": "notes.txt", "offset": 5, "count": 3}))
    assert bad.startswith("error:") and "'count'" in bad and "path" in bad and "offset" in bad and "limit" in bad


# ---- the capability ladder and --wall-s (issue 175) -------------------------------------------------------------

def test_wall_s_reaches_every_attempt_and_the_report_header(tmp_path, fake_kenv, env_ready, monkeypatch):
    walls = []

    def run_attempt(*, op, n, wall_s, **kw):
        walls.append(wall_s)
        return canned(op, n)
    monkeypatch.setattr(kp, "run_attempt", run_attempt)
    code = kp.main(probe_args(tmp_path, attempts=2, wall_s=420.0), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 0 and walls == [420.0, 420.0]
    header = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())["header"]
    assert header["budget"]["wall_s"] == 420.0 and header["budget"]["max_turns"] == 40
    assert "wall budget (420 s)" in header["budget_note"]


def test_without_wall_s_the_default_budget_is_used(tmp_path, fake_kenv, env_ready, monkeypatch):
    walls = []
    monkeypatch.setattr(kp, "run_attempt", lambda *, op, n, wall_s, **kw: walls.append(wall_s) or canned(op, n))
    kp.main(probe_args(tmp_path), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=no_run)
    assert walls == [kp.BUDGET["wall_s"]]


@pytest.mark.parametrize("bad", [0, -5, float("nan"), 7201, "soon"])
def test_main_refuses_a_bad_wall_s_before_anything_runs(tmp_path, fake_kenv, env_ready, capsys, bad):
    code = kp.main(probe_args(tmp_path, wall_s=bad), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 2 and env_ready == [] and not (tmp_path / "out").exists()
    assert "--wall-s" in capsys.readouterr().err


def test_main_with_the_ladder_runs_each_rung_in_its_own_directory_with_one_sampling(tmp_path, fake_kenv, env_ready,
                                                                                  monkeypatch):
    seen = []

    def run_attempt(*, op, n, probe_dir, wall_s, temperature, seed, **kw):
        seen.append((op, n, Path(probe_dir), wall_s, temperature, seed))
        return canned(op, n, passed=op == "add")
    monkeypatch.setattr(kp, "run_attempt", run_attempt)
    out = io.StringIO()
    code = kp.main(probe_args(tmp_path, op=None, ladder=True, rungs="add,mul", attempts=2, temperature="0.7", seed="500",
                              rung_wall="add=300"), env=fake_kenv, out=out, probe=lambda e, m: True,
                   run_watched_fn=no_run)
    assert code == 0
    assert [(s[0], s[1], s[3], s[4], s[5]) for s in seen] == [("add", 1, 300.0, 0.7, 501), ("add", 2, 300.0, 0.7, 502),
                                                              ("mul", 1, kp.BUDGET["wall_s"], 0.7, 501),
                                                              ("mul", 2, kp.BUDGET["wall_s"], 0.7, 502)]
    assert [s[2] for s in seen][0] == tmp_path / "out" / "add" and [s[2] for s in seen][2] == tmp_path / "out" / "mul"
    data = json.loads((tmp_path / "out" / "ladder-report.json").read_text())
    assert [(r["op"], r["passed"], r["attempts"]) for r in data["rungs"]] == [("add", 2, 2), ("mul", 0, 2)]
    assert data["first_zero_rung"] == "mul"
    assert "ladder rung 1 of 2: add" in out.getvalue()


def test_main_without_an_op_or_the_ladder_refuses(tmp_path, fake_kenv, env_ready, capsys):
    code = kp.main(probe_args(tmp_path, op=None), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    assert code == 2 and "--ladder" in capsys.readouterr().err




@pytest.mark.parametrize("flag, value", [("rungs", "add"), ("rung_wall", "add=300"), ("rung_expect_s", "add=60"),
                                         ("max_hours", 6.0), ("report_only", True)])
def test_main_refuses_a_ladder_only_option_without_the_ladder_and_starts_nothing(tmp_path, fake_kenv, env_ready, capsys,
                                                                               monkeypatch, flag, value):
    started = []
    monkeypatch.setattr(kp, "run_attempt", lambda **kw: started.append(kw) or canned(kw["op"], kw["n"]))
    code = kp.main(probe_args(tmp_path, **{flag: value}), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True,
                   run_watched_fn=no_run)
    assert code == 2 and started == [] and env_ready == [] and not (tmp_path / "out").exists()
    err = capsys.readouterr().err
    assert "--" + flag.replace("_", "-") in err and "only applies with --ladder" in err


def test_the_ladders_derived_seed_base_is_recorded_with_its_origin_in_every_rung_header(tmp_path, fake_kenv, env_ready,
                                                                                      monkeypatch):
    monkeypatch.setattr(kp, "run_attempt", lambda *, op, n, **kw: canned(op, n))
    kp.main(probe_args(tmp_path, op=None, ladder=True, rungs="add,mul", attempts=1, temperature="0.7", seed=None),
            env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=no_run)
    heads = [json.loads((tmp_path / "out" / r / "kernel-probe-report.json").read_text())["header"]["sampling"]
             for r in ("add", "mul")]
    assert heads[0]["seed_base"] == heads[1]["seed_base"] and isinstance(heads[0]["seed_base"], int)
    assert {h["seed_base_from"] for h in heads} == {"the ladder's id"}
    data = json.loads((tmp_path / "out" / "ladder-report.json").read_text())
    assert data["settings"]["seed_base"] == heads[0]["seed_base"] and data["settings"]["seed_base_from"] == "the ladder's id"
