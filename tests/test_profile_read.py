# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard profile-read` (orchard/profile_read.py; issues 551 and 113): a bounded summary of a report folder, read through the report
viewer's MCP server over stdio.

Every test runs the real command against a fake server: a small script, installed as `bin/ttnn-visualizer-mcp` in a fake virtual environment,
that speaks the same newline-delimited JSON-RPC and misbehaves on request (`mode.txt` in the fake environment lists what it does). The tests
that need the real server are at the end and skip when no environment for it exists.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orchard import cli, profile_read as pr

SERVER = r'''#!{python}
import json, os, sys, time
VENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(VENV, "record")
os.makedirs(REC, exist_ok=True)
MODE = open(os.path.join(VENV, "mode.txt")).read().split()
with open(os.path.join(REC, "pid.txt"), "w") as f:
    f.write(str(os.getpid()))
with open(os.path.join(REC, "env.json"), "w") as f:
    json.dump({{"env": dict(os.environ), "cwd": os.getcwd()}}, f)
UNANSWERABLE = [m.split("=", 1)[1] for m in MODE if m.startswith("unanswerable=")]
ONLY = [m.split("=", 1)[1].split(",") for m in MODE if m.startswith("answerable=")]
ALL = ["top_ops", "op_breakdown", "memory_profile", "errors", "op_configurations", "program_cache", "find_operations", "operation_detail"]
handles = 0
CURRENT_ID = None


def reply(i, payload, error=False):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "id": i, "result": {{"content": [{{"type": "text", "text": text}}], "isError": error}}}}) + "\n")
    sys.stdout.flush()


def tool(i, name, args):
    global handles, CURRENT_ID
    CURRENT_ID = i
    if "hang_on=" + name in MODE:
        time.sleep(600)
    if "slow" in MODE:
        time.sleep(0.4)
    if "chatty_on=" + name in MODE:
        for n in range(60):
            sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "method": "notifications/message", "params": {{"n": n}}}}) + "\n")
        sys.stdout.flush()
    if "some_noise_on=" + name in MODE:
        for n in range(5):
            sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "method": "notifications/message", "params": {{"n": n}}}}) + "\n")
        sys.stdout.flush()
    if "rpcerror_on=" + name in MODE:
        sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "id": CURRENT_ID, "error": {{"code": -32000, "message": "boom " + "b" * 3000}}}}) + "\n"); sys.stdout.flush(); return
    if "edge_on=" + name in MODE:
        n = int(next(m.split("=")[1] for m in MODE if m.startswith("edge_len=")))
        sys.stdout.write("x" * n + "\n"); sys.stdout.flush(); return
    if "oversize_on=" + name in MODE:
        sys.stdout.write("x" * (6 * 1024 * 1024) + "\n"); sys.stdout.flush(); return
    if "notjson_on=" + name in MODE:
        sys.stdout.write("this is not json\n"); sys.stdout.flush(); return
    if "error_on=" + name in MODE:
        return reply(i, "the report has no rank 3 " + "e" * 50000, error=True)
    if "stderr_flood" in MODE:
        sys.stderr.write("noise " * 1_000_000); sys.stderr.flush()
    if name == "load_report":
        handles += 1
        with open(os.path.join(REC, "load%d.json" % handles), "w") as f:
            json.dump(args, f)
        return reply(i, {{"handle": "../../etc/passwd" if "badhandle" in MODE else "report-%d" % handles, "arch": {{"cluster_descriptor": "blackhole"}}, "device_count": {{"profiler": 1}},
                        "answerable": ONLY[0] if ONLY else [t for t in ALL if t not in UNANSWERABLE], "unanswerable": UNANSWERABLE, "performance_csv": "ops.csv",
                        "signpost_count": 0, "app_link": "http://localhost:8000/open/{{}}".format(handles)}})
    with open(os.path.join(REC, "call_" + name + ".json"), "a") as f:
        f.write(json.dumps(args) + "\n")
    if "wide_rows_on=" + name in MODE:
        return reply(i, {{"handle": args.get("handle"), "op_count": 616, "ops": [dict({{"id": str(n), "op_code": "Matmul"}}, **{{"field%d" % k: "v" * 100 for k in range(20)}}) for n in range(10)]}})
    if "wide_rows" in MODE:
        return reply(i, {{"handle": args.get("handle"), "op_count": 616, "ops": [dict({{"id": str(n), "op_code": "Matmul"}}, **{{"field%d" % k: "v" * 100 for k in range(60)}}) for n in range(10)]}})
    if "big_result" in MODE:
        return reply(i, {{"handle": args.get("handle"), "rows": [{{"op": "Op%d" % n, "note": "n" * 200, "app_link": "http://localhost:8000/x"}} for n in range(8000)],
                        "returned": 10 ** 12}})
    if name == "diff_reports":
        return reply(i, {{"total_change_percent": -0.18, "changes": [], "handle_a": args["handle_a"], "handle_b": args["handle_b"]}})
    return reply(i, {{"handle": args.get("handle"), "tool": name, "rows": [{{"op": "MatmulDeviceOperation", "device_time_us": 540.14}}],
                    "app_link": "http://localhost:8000/ops/1", "note": "see " + "http://localhost:8000/ops/1"}})


for line in sys.stdin:
    msg = json.loads(line)
    i, method = msg.get("id"), msg.get("method")
    if method == "initialize":
        sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "id": i, "result": {{"protocolVersion": "2025-06-18", "capabilities": {{"tools": {{}}}},
                         "serverInfo": {{"name": "fake", "version": "0"}}}}}}) + "\n"); sys.stdout.flush()
    elif method == "tools/list":
        names = ALL + ["load_report", "diff_reports"]
        if "no_diff_tool" in MODE:
            names.remove("diff_reports")
        sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "id": i, "result": {{"tools": [{{"name": n, "description": "d" * 500, "inputSchema": {{"type": "object"}}}} for n in names]}}}}) + "\n"); sys.stdout.flush()
    elif method == "tools/call":
        tool(i, msg["params"]["name"], msg["params"].get("arguments", {{}}))
    if "die_after_init" in MODE and method == "initialize":
        os._exit(3)
'''


def make_venv(root: Path, *, version: str = "0.112.0", mode: str = "ok", agent_source: str = "# the agent\n") -> Path:
    venv = root / "venv"
    (venv / "bin").mkdir(parents=True)
    exe = venv / "bin" / "ttnn-visualizer-mcp"
    exe.write_text(SERVER.format(python=sys.executable), encoding="utf-8")
    exe.chmod(0o755)
    sp = venv / "lib" / "python3.13" / "site-packages"
    dist = sp / f"ttnn_visualizer-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: ttnn-visualizer\nVersion: {version}\n", encoding="utf-8")
    agent = sp / "ttnn_visualizer" / "agent"
    agent.mkdir(parents=True)
    (agent / "server.py").write_text(agent_source, encoding="utf-8")
    (agent / "__pycache__").mkdir()
    (agent / "__pycache__" / "server.cpython-313.pyc").write_bytes(b"cache that must not count")
    (venv / "mode.txt").write_text(mode, encoding="utf-8")
    return venv


def tree_digest_by_hand(agent_dir: Path) -> str:
    rows = sorted((p.relative_to(agent_dir).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in agent_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for rel, d in rows:
        h.update(f"{rel}\0{d}\n".encode())
    return h.hexdigest()


def make_report(root: Path, name: str = "report") -> Path:
    d = root / name
    d.mkdir()
    (d / "db.sqlite").write_bytes(b"not a real database")
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "ops_perf_results_x.csv").write_text("OP CODE\n", encoding="utf-8")
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake environment whose package files count as the tested ones, and a report folder."""
    venv = make_venv(tmp_path)
    digest = tree_digest_by_hand(venv / "lib" / "python3.13" / "site-packages" / "ttnn_visualizer" / "agent")
    monkeypatch.setitem(pr.TESTED, "0.112.0", {"wheel_sha256": "0" * 64, "agent_tree_sha256": digest})
    report = make_report(tmp_path)

    class E:
        pass
    e = E()
    e.tmp, e.venv, e.report, e.digest = tmp_path, venv, report, digest
    e.out = tmp_path / "summary.json"
    return e


def run(e, *extra, report=None, check_exit=None, call_timeout=20, total_timeout=120):
    args = pr.parser().parse_args([str(report or e.report), "--venv", str(e.venv), "--out", str(e.out), "--call-timeout", str(call_timeout),
                                   "--total-timeout", str(total_timeout), *extra])
    buf = io.StringIO()
    code = pr.main(args, out=buf)
    summary = json.loads(e.out.read_text(encoding="utf-8")) if e.out.is_file() else None
    if check_exit is not None:
        assert code == check_exit, (code, buf.getvalue(), summary)
    return code, buf.getvalue(), summary


def server_pid(e) -> int | None:
    p = e.venv / "record" / "pid.txt"
    return int(p.read_text()) if p.exists() else None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return "Z" not in Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------------- the happy path

def test_every_question_is_asked_and_answered(env):
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    assert s["schema"] == 1
    assert [k for k, v in s["questions"].items() if v["status"] != "answered"] == []
    assert set(s["questions"]) == {q.key for q in pr.QUESTIONS if not q.needs_baseline}
    assert s["server"]["version"] == "0.112.0" and s["server"]["tested"] is True
    assert s["server"]["agent_tree_sha256"] == env.digest
    assert s["load"]["handle"] == "report-1" and "top_ops" in s["load"]["answerable"]
    assert s["could_not_answer"] == []
    assert s["coder_tools"] == list(pr.CODER_TOOLS) and len(pr.CODER_TOOLS) == 7
    assert (env.venv / "record" / "call_top_ops.json").exists()
    args = json.loads((env.venv / "record" / "call_top_ops.json").read_text().splitlines()[0])
    assert args["handle"] == "report-1" and "kernels" in args["include"]
    assert stat.S_IMODE(env.out.stat().st_mode) == 0o600


def test_the_server_runs_with_usage_recording_off_a_private_home_and_no_secrets(env, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_not_for_children")
    monkeypatch.setenv("SOME_API_KEY", "secret")
    run(env, check_exit=pr.EXIT_OK)
    rec = json.loads((env.venv / "record" / "env.json").read_text())
    child_env = rec["env"]
    assert child_env["USAGE_RECORDING_DISABLED"] == "true"
    assert "HF_TOKEN" not in child_env and "SOME_API_KEY" not in child_env
    assert child_env["HOME"] != os.path.expanduser("~") and not os.path.exists(child_env["HOME"])      # removed afterwards
    assert child_env["PATH"].startswith(str(env.venv / "bin"))
    assert child_env["PYTHONDONTWRITEBYTECODE"] == "1" and child_env["PYTHONNOUSERSITE"] == "1"
    assert rec["cwd"] != os.getcwd()


def test_the_server_is_gone_when_the_command_ends(env):
    run(env, check_exit=pr.EXIT_OK)
    assert not alive(server_pid(env))


def test_the_report_is_loaded_by_the_path_it_was_given_with_both_files_named(env):
    run(env, check_exit=pr.EXIT_OK)
    load = json.loads((env.venv / "record" / "load1.json").read_text())
    assert load["memory_report_path"] == str(env.report.resolve()) and load["performance_path"] == str(env.report.resolve())


def test_a_report_with_only_a_performance_csv_is_loaded_without_a_memory_path(env, tmp_path):
    d = tmp_path / "csv-only"
    d.mkdir()
    (d / "ops_perf_results_y.csv").write_text("OP CODE\n", encoding="utf-8")
    run(env, report=d, check_exit=pr.EXIT_OK)
    load = json.loads((env.venv / "record" / "load1.json").read_text())
    assert "memory_report_path" not in load and load["performance_path"] == str(d.resolve())


def test_the_app_links_are_dropped_and_never_followed(env):
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    text = env.out.read_text()
    assert "\"app_link\"" not in text and "\"app_links\"" not in text
    assert s["limits"]["app_links_dropped"] >= 1
    assert s["load"].get("app_link") is None


# ---------------------------------------------------------------------------------------------------- the answerable list

def test_a_tool_the_server_calls_unanswerable_is_not_called_and_the_reason_is_recorded(env):
    (env.venv / "mode.txt").write_text("ok unanswerable=memory_profile", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_PARTIAL)
    q = s["questions"]["memory_profile"]
    assert q["status"] == "could_not_answer" and "unanswerable" in q["reason"]
    assert not (env.venv / "record" / "call_memory_profile.json").exists()
    assert "memory_profile" in s["could_not_answer"]
    assert s["questions"]["top_ops"]["status"] == "answered"


def test_a_tool_missing_from_the_server_tool_list_is_not_called(env):
    (env.venv / "mode.txt").write_text("ok no_diff_tool", encoding="utf-8")
    base = make_report(env.tmp, "base")
    code, out, s = run(env, "--baseline", str(base), check_exit=pr.EXIT_PARTIAL)
    assert s["questions"]["baseline_comparison"]["status"] == "could_not_answer"
    assert "no tool" in s["questions"]["baseline_comparison"]["reason"]


def test_a_baseline_adds_one_comparison_between_the_two_handles(env):
    base = make_report(env.tmp, "base")
    code, out, s = run(env, "--baseline", str(base), check_exit=pr.EXIT_OK)
    assert s["questions"]["baseline_comparison"]["status"] == "answered"
    call = json.loads((env.venv / "record" / "call_diff_reports.json").read_text().splitlines()[0])
    assert call["handle_a"] == "report-2" and call["handle_b"] == "report-1"       # the baseline is the first argument
    assert s["baseline_dir"].endswith("base")


def test_without_a_baseline_there_is_no_comparison(env):
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    assert "baseline_comparison" not in s["questions"]
    assert not (env.venv / "record" / "call_diff_reports.json").exists()


# ---------------------------------------------------------------------------------------------------- the version and the files

def test_an_untested_version_is_refused_before_anything_runs(env, tmp_path):
    (tmp_path / "other-root").mkdir()
    env.venv = make_venv(tmp_path / "other-root", version="9.9.9")
    code, out, s = run(env, check_exit=pr.EXIT_REFUSED)
    assert "9.9.9" in out and "--accept-untested-version" in out
    assert server_pid(env) is None


def test_an_untested_version_runs_when_the_flag_says_so_and_the_summary_says_it_is_untested(env, tmp_path):
    (tmp_path / "o").mkdir()
    env.venv = make_venv(tmp_path / "o", version="9.9.9")
    code, out, s = run(env, "--accept-untested-version", check_exit=pr.EXIT_OK)
    assert s["server"]["tested"] is False and s["server"]["version"] == "9.9.9"


def test_a_tested_version_whose_files_differ_is_refused(env):
    (env.venv / "lib/python3.13/site-packages/ttnn_visualizer/agent/server.py").write_text("# changed\n", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_REFUSED)
    assert "differ" in out and server_pid(env) is None
    code, out, s = run(env, "--accept-untested-version", check_exit=pr.EXIT_OK)
    assert s["server"]["tested"] is False and s["server"]["agent_tree_matches_tested"] is False


def test_the_tree_digest_ignores_caches_and_matches_the_documented_recipe(env):
    sp = env.venv / "lib/python3.13/site-packages"
    assert pr.agent_tree_sha256(sp) == tree_digest_by_hand(sp / "ttnn_visualizer/agent")
    (sp / "ttnn_visualizer/agent/__pycache__/other.pyc").write_bytes(b"changed cache")
    assert pr.agent_tree_sha256(sp) == tree_digest_by_hand(sp / "ttnn_visualizer/agent")
    (sp / "ttnn_visualizer/agent/extra.py").write_text("x = 1\n", encoding="utf-8")
    assert pr.agent_tree_sha256(sp) != env.digest


def test_the_tree_digest_refuses_a_link_inside_the_agent_directory(env):
    sp = env.venv / "lib/python3.13/site-packages"
    os.symlink("/etc/hostname", sp / "ttnn_visualizer/agent/link.py")
    with pytest.raises(pr.ProfileReadError):
        pr.agent_tree_sha256(sp)
    code, out, s = run(env, check_exit=pr.EXIT_REFUSED)
    assert server_pid(env) is None


def test_a_wheel_file_is_hashed_and_checked_against_the_tested_digest(env, tmp_path, monkeypatch):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"pretend wheel")
    good = hashlib.sha256(b"pretend wheel").hexdigest()
    monkeypatch.setitem(pr.TESTED, "0.112.0", {"wheel_sha256": good, "agent_tree_sha256": env.digest})
    code, out, s = run(env, "--wheel", str(wheel), check_exit=pr.EXIT_OK)
    assert s["server"]["wheel_sha256"] == good and s["server"]["wheel_matches_tested"] is True
    wheel.write_bytes(b"another wheel")
    import shutil
    shutil.rmtree(env.venv / "record")
    code, out, s = run(env, "--wheel", str(wheel), check_exit=pr.EXIT_REFUSED)
    assert "wheel" in out and server_pid(env) is None


def test_without_a_wheel_the_summary_says_the_wheel_digest_was_not_measured(env):
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    assert s["server"]["wheel_sha256"] is None
    assert any("wheel" in x for x in s["could_not_measure"])
    assert s["server"]["tested_wheel_sha256"] == "0" * 64


def test_the_pinned_digests_are_the_ones_audited_for_0_112_0():
    assert pr.TESTED["0.112.0"]["wheel_sha256"] == "c47e67e64478a769b6556b77be0881706a05b0f4a0bdbb407178a48ad93a398e"
    assert len(pr.TESTED["0.112.0"]["agent_tree_sha256"]) == 64


# ---------------------------------------------------------------------------------------------------- refusals before a server starts

def test_a_missing_report_directory_is_refused(env, tmp_path):
    code, out, s = run(env, report=tmp_path / "nope", check_exit=pr.EXIT_REFUSED)
    assert server_pid(env) is None


def test_a_report_directory_that_is_a_link_is_refused(env, tmp_path):
    os.symlink(env.report, tmp_path / "linked")
    run(env, report=tmp_path / "linked", check_exit=pr.EXIT_REFUSED)
    assert server_pid(env) is None


def test_a_directory_with_no_report_files_is_refused(env, tmp_path):
    (tmp_path / "empty").mkdir()
    code, out, s = run(env, report=tmp_path / "empty", check_exit=pr.EXIT_REFUSED)
    assert "db.sqlite" in out


def test_a_db_that_is_a_fifo_or_a_link_does_not_count_and_does_not_hang(env, tmp_path):
    d = tmp_path / "fifo"
    d.mkdir()
    os.mkfifo(d / "db.sqlite")
    os.symlink("/etc/hostname", d / "ops_perf_results_z.csv")
    start = time.monotonic()
    run(env, report=d, check_exit=pr.EXIT_REFUSED)
    assert time.monotonic() - start < 5 and server_pid(env) is None


def test_a_missing_or_unrunnable_server_is_refused(env, tmp_path):
    (env.venv / "bin" / "ttnn-visualizer-mcp").chmod(0o644)
    run(env, check_exit=pr.EXIT_REFUSED)
    (env.venv / "bin" / "ttnn-visualizer-mcp").unlink()
    code, out, s = run(env, check_exit=pr.EXIT_REFUSED)
    assert "ttnn-visualizer-mcp" in out


def test_a_server_that_is_a_link_is_refused(env, tmp_path):
    real = tmp_path / "real-server"
    real.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    real.chmod(0o755)
    (env.venv / "bin" / "ttnn-visualizer-mcp").unlink()
    os.symlink(real, env.venv / "bin" / "ttnn-visualizer-mcp")
    run(env, check_exit=pr.EXIT_REFUSED)


def test_a_baseline_that_is_not_a_report_is_refused(env, tmp_path):
    (tmp_path / "nobase").mkdir()
    code, out, s = run(env, "--baseline", str(tmp_path / "nobase"), check_exit=pr.EXIT_REFUSED)
    assert server_pid(env) is None


@pytest.mark.parametrize("flag,value", [("--call-timeout", "0"), ("--call-timeout", "nan"), ("--total-timeout", "-1"), ("--total-timeout", "inf")])
def test_a_timeout_that_is_not_a_finite_positive_number_is_refused(env, flag, value):
    args = pr.parser().parse_args([str(env.report), "--venv", str(env.venv), "--out", str(env.out), flag, value])
    assert pr.main(args, out=io.StringIO()) == pr.EXIT_REFUSED
    assert server_pid(env) is None


def test_an_output_path_that_is_a_directory_is_refused(env, tmp_path):
    env.out = tmp_path
    code, out, s = run(env, check_exit=pr.EXIT_REFUSED)
    assert server_pid(env) is None


# ---------------------------------------------------------------------------------------------------- a server that misbehaves

def test_a_reply_line_over_the_cap_ends_the_session_and_the_server_is_killed(env):
    (env.venv / "mode.txt").write_text("ok oversize_on=top_ops", encoding="utf-8")
    start = time.monotonic()
    code, out, s = run(env, check_exit=pr.EXIT_SERVER)
    assert time.monotonic() - start < 30
    assert s["questions"]["top_ops"]["status"] == "failed" and "longer than" in s["questions"]["top_ops"]["reason"]
    assert all(v["status"] in ("not_asked", "failed") for k, v in s["questions"].items() if k != "top_ops" and k != "load_report")
    assert not alive(server_pid(env))


def test_notifications_before_a_reply_are_skipped_but_a_flood_of_them_ends_the_session(env):
    (env.venv / "mode.txt").write_text("ok some_noise_on=top_ops", encoding="utf-8")
    run(env, check_exit=pr.EXIT_OK)
    shutil.rmtree(env.venv / "record")
    (env.venv / "mode.txt").write_text("ok chatty_on=top_ops", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_SERVER)
    assert s["questions"]["top_ops"]["status"] == "failed" and "messages arrived" in s["questions"]["top_ops"]["reason"]


def test_a_protocol_error_reply_ends_the_session_with_its_message_cut(env):
    (env.venv / "mode.txt").write_text("ok rpcerror_on=top_ops", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_SERVER)
    reason = s["questions"]["top_ops"]["reason"]
    assert "protocol error" in reason and "boom" in reason and len(reason) < 800


def test_a_handle_that_is_not_the_server_style_is_refused_and_every_question_says_why(env):
    (env.venv / "mode.txt").write_text("ok badhandle", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_PARTIAL)
    assert all(v["status"] == "could_not_answer" and "no usable handle" in v["reason"] for v in s["questions"].values())
    assert not (env.venv / "record" / "call_top_ops.json").exists()


def test_a_report_the_server_cannot_load_makes_every_question_unanswerable(env):
    (env.venv / "mode.txt").write_text("ok error_on=load_report", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_PARTIAL)
    assert all(v["status"] == "could_not_answer" and "load_report failed" in v["reason"] for v in s["questions"].values())
    assert s["load"] is None


def test_a_server_file_that_changed_after_the_check_is_not_started(env, monkeypatch):
    real = pr.check_server

    def stale(*a, **k):
        d = real(*a, **k)
        d["_identity"] = (0, 0, 0)
        return d
    monkeypatch.setattr(pr, "check_server", stale)
    code, out, s = run(env, check_exit=pr.EXIT_REFUSED)
    assert "changed" in out and server_pid(env) is None


def test_the_error_pipe_is_drained_to_its_cap_while_the_server_runs():
    from types import SimpleNamespace
    import threading
    r, w = os.pipe()
    s = object.__new__(pr._Session)
    s.proc = SimpleNamespace(stderr=SimpleNamespace(fileno=lambda: r))
    s._err = bytearray()

    def writer():
        for _ in range(80):
            os.write(w, b"x" * 65536)
        os.close(w)
    t = threading.Thread(target=writer)
    t.start()
    s._drain_stderr()
    t.join()
    os.close(r)
    assert len(s._err) <= pr.STDERR_MAX_BYTES


def test_a_reply_line_one_byte_over_the_cap_is_refused_as_too_long(env):
    (env.venv / "mode.txt").write_text(f"ok edge_on=top_ops edge_len={pr.REPLY_MAX_BYTES + 1}", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_SERVER)
    assert "longer than" in s["questions"]["top_ops"]["reason"]


def test_the_line_reader_stops_buffering_at_the_cap_when_no_newline_ever_comes():
    import threading
    r, w = os.pipe()
    s = object.__new__(pr._Session)
    s._buf, s._out_fd, s._call_timeout, s._total_deadline, s._now = bytearray(), r, 30.0, time.monotonic() + 30, time.monotonic

    def writer():
        try:
            for _ in range(200):
                os.write(w, b"y" * 65536)
        except OSError:
            pass
        finally:
            os.close(w)
    t = threading.Thread(target=writer)
    t.start()
    with pytest.raises(pr.ServerFailed, match="longer than"):
        s._line(time.monotonic() + 30, "a test line")
    os.close(r)
    t.join()
    assert len(s._buf) <= pr.REPLY_MAX_BYTES + 65536


def test_a_tool_missing_from_the_answerable_list_is_not_called_even_when_the_unanswerable_list_does_not_name_it(env):
    (env.venv / "mode.txt").write_text("ok answerable=top_ops,op_breakdown", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_PARTIAL)
    assert s["questions"]["top_ops"]["status"] == "answered"
    q = s["questions"]["errors"]
    assert q["status"] == "could_not_answer" and "not in the server's answerable list" in q["reason"]
    assert not (env.venv / "record" / "call_errors.json").exists()


def test_one_answer_over_its_budget_is_cut_harder_while_the_total_is_still_small(env):
    (env.venv / "mode.txt").write_text("ok wide_rows_on=top_ops", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    reply = s["questions"]["top_ops"]["reply"]
    assert len(json.dumps(reply)) <= pr.QUESTION_MAX_BYTES and "cut" not in reply and reply["ops"][0]["op_code"] == "Matmul"


def test_a_reply_that_is_not_json_ends_the_session(env):
    (env.venv / "mode.txt").write_text("ok notjson_on=top_ops", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_SERVER)
    assert s["questions"]["top_ops"]["status"] == "failed" and "JSON" in s["questions"]["top_ops"]["reason"]
    assert not alive(server_pid(env))


def test_a_server_that_hangs_is_given_the_call_timeout_and_then_killed_by_pid(env):
    (env.venv / "mode.txt").write_text("ok hang_on=top_ops", encoding="utf-8")
    start = time.monotonic()
    code, out, s = run(env, check_exit=pr.EXIT_SERVER, call_timeout=1.5)
    assert 1.4 <= time.monotonic() - start < 20
    assert s["questions"]["top_ops"]["status"] == "failed" and "no reply within" in s["questions"]["top_ops"]["reason"]
    assert not alive(server_pid(env))


def test_the_total_timeout_stops_the_questions_and_the_rest_are_not_asked(env):
    (env.venv / "mode.txt").write_text("ok slow", encoding="utf-8")
    start = time.monotonic()
    code, out, s = run(env, check_exit=pr.EXIT_SERVER, total_timeout=1.5)
    assert time.monotonic() - start < 20
    statuses = [v["status"] for v in s["questions"].values()]
    assert "not_asked" in statuses and "answered" in statuses
    assert any("total" in v.get("reason", "") for v in s["questions"].values() if v["status"] in ("failed", "not_asked"))
    assert not alive(server_pid(env))


def test_a_server_that_dies_after_the_handshake_is_reported(env):
    (env.venv / "mode.txt").write_text("ok die_after_init", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_SERVER)
    assert s is not None and s["server"]["version"] == "0.112.0"
    assert s["failure"]["what"]


def test_an_error_result_is_a_question_that_could_not_be_answered_and_the_reason_is_cut(env):
    (env.venv / "mode.txt").write_text("ok error_on=errors", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_PARTIAL)
    q = s["questions"]["errors"]
    assert q["status"] == "could_not_answer" and q["reason"].startswith("the server reported an error: the report has no rank 3")
    assert len(q["reason"]) <= 700
    assert s["questions"]["top_ops"]["status"] == "answered"       # the other questions still ran


def test_a_huge_result_is_cut_to_the_question_budget_and_the_cut_is_recorded(env):
    (env.venv / "mode.txt").write_text("ok big_result", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    size = len(env.out.read_bytes())
    assert size <= pr.SUMMARY_MAX_BYTES
    for k, v in s["questions"].items():
        assert len(json.dumps(v)) <= pr.QUESTION_MAX_BYTES + 2000, k
    assert s["limits"]["cut"] and s["limits"]["list_items_cut"] > 0
    assert "\"app_link\"" not in env.out.read_text()


def test_an_answer_that_is_too_large_at_first_is_cut_harder_and_kept_not_replaced_by_a_stub(env):
    (env.venv / "mode.txt").write_text("ok wide_rows", encoding="utf-8")
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    reply = s["questions"]["top_ops"]["reply"]
    assert "cut" not in reply and reply["handle"] == "report-1" and reply["op_count"] == 616
    assert reply["ops"][0]["id"] == "0" and reply["ops"][0]["op_code"] == "Matmul"
    assert len(json.dumps(reply)) <= pr.QUESTION_MAX_BYTES


def test_the_summary_stays_under_its_cap_when_every_question_returns_a_huge_result(env):
    (env.venv / "mode.txt").write_text("ok big_result", encoding="utf-8")
    base = make_report(env.tmp, "base")
    run(env, "--baseline", str(base), check_exit=pr.EXIT_OK)
    assert len(env.out.read_bytes()) <= pr.SUMMARY_MAX_BYTES


def test_stderr_noise_does_not_block_the_server_and_is_kept_only_up_to_its_cap(env):
    (env.venv / "mode.txt").write_text("ok stderr_flood", encoding="utf-8")
    start = time.monotonic()
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    assert time.monotonic() - start < 60
    assert len(s["server"]["stderr_tail"]) <= pr.STDERR_MAX_BYTES


# ---------------------------------------------------------------------------------------------------- what the module must never do

def test_the_module_never_imports_ttnn_or_a_network_client(tmp_path):
    src = (Path(pr.__file__)).read_text(encoding="utf-8")
    for banned in ("import ttnn", "from ttnn", "import socket", "import urllib", "import http", "import requests", "import ttml"):
        assert banned not in src, banned
    code = "import sys; import orchard.profile_read; sys.exit(1 if any(m == 'ttnn' or m.startswith('ttnn.') for m in sys.modules) else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=str(Path(pr.__file__).resolve().parent.parent)).returncode == 0


def test_the_command_is_registered_in_the_cli(env, capsys):
    args = [ "profile-read", str(env.report), "--venv", str(env.venv), "--out", str(env.out)]
    code = cli.main(args)
    assert code == pr.EXIT_OK and env.out.exists()


def test_the_coder_tool_list_and_the_first_message_note_are_documented_constants():
    assert pr.CODER_TOOLS == ("top_ops", "op_breakdown", "diff_reports", "find_operations", "operation_detail", "memory_profile", "errors")
    assert "answerable" in pr.FIRST_MESSAGE_NOTE and "load_report" in pr.FIRST_MESSAGE_NOTE


def test_the_summary_names_what_it_could_not_measure(env):
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    assert any("zone" in x or "per-core" in x for x in s["could_not_measure"]) or s["could_not_measure"]


def test_the_report_folder_identity_is_recorded_and_the_files_are_listed(env):
    code, out, s = run(env, check_exit=pr.EXIT_OK)
    assert set(s["report"]["files"]) >= {"db.sqlite", "config.json", "ops_perf_results_x.csv"}
    assert isinstance(s["report"]["identity"]["inode"], int)


# ---------------------------------------------------------------------------------------------------- the bounding function

def test_bound_value_cuts_strings_lists_keys_and_depth_and_counts_each_cut():
    cuts = pr.Cuts()
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": 1}}}}}}}}}}
    v = {"s": "x" * 5000, "l": list(range(500)), "k": {str(i): i for i in range(500)}, "deep": deep}
    out = pr.bound_value(v, max_str=100, max_items=10, max_keys=10, max_depth=5, cuts=cuts)
    assert len(out["s"]) < 160 and "cut" in out["s"]
    assert len(out["l"]) == 11 and "more" in str(out["l"][-1])
    assert len(out["k"]) == 11
    assert cuts.strings == 1 and cuts.list_items == 490 and cuts.keys >= 490 and cuts.depth >= 1


def test_bound_value_drops_app_link_keys_at_every_depth():
    cuts = pr.Cuts()
    out = pr.bound_value({"app_link": "u", "x": [{"app_link": "u", "y": 1}], "z": {"app_links": {"a": 1}}}, max_str=50, max_items=5,
                         max_keys=5, max_depth=6, cuts=cuts)
    assert "app_link" not in json.dumps(out) and cuts.app_links == 3


def test_bound_value_keeps_numbers_booleans_and_nulls_exactly():
    cuts = pr.Cuts()
    v = {"a": 1, "b": 2.5, "c": True, "d": None, "e": "ok"}
    assert pr.bound_value(v, max_str=50, max_items=5, max_keys=5, max_depth=4, cuts=cuts) == v
    assert cuts.total() == 0


# ---------------------------------------------------------------------------------------------------- the real server (skipped unless one exists)

_REAL_VENV = os.environ.get("ORCHARD_TTNN_VISUALIZER_VENV", "")
_REAL_REPORT = os.environ.get("ORCHARD_TTNN_VISUALIZER_SAMPLE", "")


@pytest.mark.skipif(not (_REAL_VENV and _REAL_REPORT and Path(_REAL_VENV, "bin", "ttnn-visualizer-mcp").exists() and Path(_REAL_REPORT, "db.sqlite").exists()),
                    reason="set ORCHARD_TTNN_VISUALIZER_VENV (a venv with the tested wheel) and ORCHARD_TTNN_VISUALIZER_SAMPLE (a report folder)")
def test_the_real_server_answers_the_fixed_questions(tmp_path):
    out = tmp_path / "s.json"
    args = pr.parser().parse_args([_REAL_REPORT, "--venv", _REAL_VENV, "--out", str(out)])
    code = pr.main(args, out=io.StringIO())
    s = json.loads(out.read_text())
    assert code in (pr.EXIT_OK, pr.EXIT_PARTIAL), s
    assert s["server"]["tested"] is True
    assert s["questions"]["top_ops"]["status"] == "answered"
    assert len(out.read_bytes()) <= pr.SUMMARY_MAX_BYTES
