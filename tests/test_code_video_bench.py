# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The code-video benchmark harness (scripts/code_video_bench): run_bench.sh, collect.py and poll_metrics.py, driven through their
command lines with stand-ins for docker, curl, systemd-run, ffprobe and ffmpeg on PATH. No container, no server and no model is started.

Each test is one review finding of PR 450 (Copilot round 1). A stand-in `docker` logs every call as one JSON line, writes the id a real
`--cidfile` would hold, and refuses what the real docker refuses (a relative mount destination or working directory, a name that is in use).
"""
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "scripts" / "code_video_bench"

# ---- the stand-ins ---------------------------------------------------------------------------------------------------------------

DOCKER_STUB = r'''#!/usr/bin/env python3
import json, os, subprocess, sys, time
args = sys.argv[1:]
log = os.environ["STUB_LOG"]
mode = os.environ.get("DOCKER_STUB_MODE", "run")
n = int(os.environ.get("STUB_COUNTER_FILE_N", "0"))
def note(rec):
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
if args[:1] == ["rm"]:
    note({"cmd": "rm", "args": args[1:]})
    sys.exit(0)
if args[:1] != ["run"]:
    note({"cmd": args[:1], "args": args[1:]})
    sys.exit(0)
vols, workdir, cidfile, name = [], None, None, None
i = 1
while i < len(args):
    a = args[i]
    if a == "-v":
        vols.append(args[i + 1]); i += 2
    elif a == "-w":
        workdir = args[i + 1]; i += 2
    elif a == "--cidfile":
        cidfile = args[i + 1]; i += 2
    elif a == "--name":
        name = args[i + 1]; i += 2
    else:
        i += 1
note({"cmd": "run", "vols": vols, "workdir": workdir, "name": name, "cidfile": cidfile, "args": args})
for v in vols:
    dst = v.split(":")[1]
    if not dst.startswith("/"):
        print("docker: Error response from daemon: invalid mount path: '%s' mount path must be absolute" % dst, file=sys.stderr)
        sys.exit(125)
if workdir is not None and not workdir.startswith("/"):
    print("docker: Error response from daemon: the working directory '%s' is invalid, it needs to be an absolute path" % workdir, file=sys.stderr)
    sys.exit(125)
if os.environ.get("DOCKER_STUB_CONFLICT") == "1":
    if cidfile:
        open(cidfile, "w").close()          # the real client creates the file before it asks the daemon, and leaves it empty on failure
    print('docker: Error response from daemon: Conflict. The container name "/%s" is already in use' % name, file=sys.stderr)
    sys.exit(125)
if cidfile:
    with open(cidfile, "w") as f:
        f.write("STUBID-%s" % os.getpid())
if mode == "sleep":
    time.sleep(60)
    sys.exit(0)
if "sh" in args and args[args.index("sh") + 1:args.index("sh") + 2] == ["-c"]:
    cmd = args[args.index("sh") + 2]
    for v in vols:                              # the host path stands in for the container path when they differ
        parts = v.split(":")
        if parts[0] != parts[1]:
            cmd = cmd.replace(parts[1], parts[0])
    r = subprocess.run(["sh", "-c", cmd], cwd=workdir or ".", capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    sys.exit(r.returncode)
sys.exit(0)
'''

CURL_STUB = '#!/bin/sh\necho \'{"data":[{"id":"/models/under-test"}]}\'\n'

SYSTEMD_RUN_STUB = r'''#!/bin/sh
echo "$@" >> "$STUB_LOG.systemd"
while [ "$1" != "--" ]; do shift; done
shift
exec "$@"
'''

# ffprobe and ffmpeg refuse anything but one existing file argument: an unquoted name with a space reaches them as two words.
FFPROBE_STUB = r'''#!/bin/sh
for last; do :; done
if [ ! -f "$last" ]; then echo "ffprobe: no such file: $last" >&2; exit 1; fi
frames=',"nb_read_frames":"240"'; [ -n "$FFPROBE_NO_FRAMES" ] && frames=
echo '{"streams":[{"codec_type":"video","codec_name":"h264","width":1280,"height":720,"pix_fmt":"yuv420p","avg_frame_rate":"24/1"'"$frames"'}],"format":{"duration":"10.000","size":"123456"}}'
'''

FFMPEG_STUB = r'''#!/bin/sh
for a; do [ "$a" = "-i" ] && want=1 && continue; if [ -n "$want" ]; then f="$a"; want=; fi; done
if [ ! -f "$f" ]; then echo "ffmpeg: no such file: $f" >&2; exit 1; fi
printf '#format: frame checksums\n0,0,0,1,1,0xaaaa\n0,1,1,1,1,0xaaaa\n0,2,2,1,1,0xbbbb\n'
'''


def write_exec(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture()
def stubs(tmp_path):
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    write_exec(bin_dir / "docker", DOCKER_STUB)
    write_exec(bin_dir / "curl", CURL_STUB)
    write_exec(bin_dir / "systemd-run", SYSTEMD_RUN_STUB)
    write_exec(bin_dir / "ffprobe", FFPROBE_STUB)
    write_exec(bin_dir / "ffmpeg", FFMPEG_STUB)
    log = tmp_path / "stub.log"
    log.write_text("")
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", STUB_LOG=str(log))
    for k in ("DOCKER_STUB_MODE", "DOCKER_STUB_CONFLICT", "CVBENCH_DOCKER_TIMEOUT_S", "SKILL_SRC", "WORK_ROOT", "FFPROBE_NO_FRAMES"):
        env.pop(k, None)
    return env, log


def run_cmd(argv, env, cwd, timeout=60):
    """Run a command in its own process group; on a timeout kill the whole group (a stand-in may be asleep) and raise."""
    p = subprocess.Popen(argv, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise
    return p.returncode, out, err


def calls(log: Path):
    return [json.loads(x) for x in log.read_text().splitlines() if x.strip()]


# ---- run_bench.sh ----------------------------------------------------------------------------------------------------------------

def bench_setup(tmp_path):
    skill = tmp_path / "skill_src"
    (skill / "scripts").mkdir(parents=True)
    (skill / "scripts" / "check_video.sh").write_text("echo trusted-check-ok\n")
    (skill / "SKILL.md").write_text("# skill\n")
    brief = tmp_path / "brief.md"
    brief.write_text("a brief\n")
    return skill, brief


def run_bench(tmp_path, env, work_root, name="r1", cwd=None, extra=None):
    skill, brief = bench_setup(tmp_path)
    e = dict(env, WORK_ROOT=str(work_root), SKILL_SRC=str(skill), ENDPOINT="http://127.0.0.1:9", **(extra or {}))
    return run_cmd(["bash", str(BENCH / "run_bench.sh"), name, str(brief), "1", "10"], e, cwd or tmp_path, timeout=90)


def test_the_shell_script_parses():
    assert subprocess.run(["bash", "-n", str(BENCH / "run_bench.sh")]).returncode == 0


def test_cleanup_removes_the_container_this_run_created_by_its_id_and_not_by_name(tmp_path, stubs):
    env, log = stubs
    rc, out, err = run_bench(tmp_path, env, tmp_path / "work")
    assert rc == 0, err
    removed = [c for c in calls(log) if c["cmd"] == "rm"]
    runs = [c for c in calls(log) if c["cmd"] == "run"]
    assert runs and runs[0]["cidfile"], "docker run must record its container id (--cidfile)"
    assert removed, "the container this run created is removed at the end"
    assert all(a.startswith("STUBID-") for c in removed for a in c["args"] if not a.startswith("-")), removed


def test_a_name_conflict_with_another_run_leaves_that_runs_container_alone(tmp_path, stubs):
    env, log = stubs
    env["DOCKER_STUB_CONFLICT"] = "1"
    rc, out, err = run_bench(tmp_path, env, tmp_path / "work")
    assert rc == 0, err
    assert (tmp_path / "work" / "r1" / "logs" / "exit_code").read_text().strip() == "125"
    assert [c for c in calls(log) if c["cmd"] == "rm"] == [], "this run created nothing, so it removes nothing"


def test_a_relative_work_root_is_made_absolute_before_the_docker_paths(tmp_path, stubs):
    env, log = stubs
    rc, out, err = run_bench(tmp_path, env, "runs", cwd=tmp_path)
    assert rc == 0, err
    run = [c for c in calls(log) if c["cmd"] == "run"][0]
    assert all(v.split(":")[1].startswith("/") for v in run["vols"]) and run["workdir"].startswith("/")
    assert (tmp_path / "runs" / "r1" / "logs" / "exit_code").read_text().strip() == "0"


@pytest.mark.parametrize("bad", ["with:colon", "with\nnewline"])
def test_a_work_root_that_cannot_be_a_docker_mount_is_refused_before_anything_starts(tmp_path, stubs, bad):
    env, log = stubs
    rc, out, err = run_bench(tmp_path, env, tmp_path / bad)
    assert rc != 0
    assert [c for c in calls(log) if c["cmd"] == "run"] == []


@pytest.mark.parametrize("bad", ["a b", "a;b", "$(x)", "a/b", ""])
def test_a_run_name_that_could_not_be_a_container_name_or_a_path_part_is_refused(tmp_path, stubs, bad):
    env, log = stubs
    skill, brief = bench_setup(tmp_path)
    e = dict(env, WORK_ROOT=str(tmp_path / "work"), SKILL_SRC=str(skill), ENDPOINT="http://127.0.0.1:9")
    rc, out, err = run_cmd(["bash", str(BENCH / "run_bench.sh"), bad, str(brief), "1", "10"], e, tmp_path, 60)
    assert rc != 0 and [c for c in calls(log) if c["cmd"] == "run"] == [] and not (tmp_path / "work" / "a").exists()


def test_the_run_records_the_trusted_skill_source_and_its_file_hashes_outside_the_task_directory(tmp_path, stubs):
    env, log = stubs
    rc, out, err = run_bench(tmp_path, env, tmp_path / "work")
    assert rc == 0, err
    logs = tmp_path / "work" / "r1" / "logs"
    assert Path((logs / "skill_src").read_text().strip()) == (tmp_path / "skill_src").resolve()
    import hashlib
    want = hashlib.sha256((tmp_path / "skill_src" / "scripts" / "check_video.sh").read_bytes()).hexdigest()
    listed = {ln.split("  ", 1)[1].removeprefix("./"): ln.split("  ", 1)[0] for ln in (logs / "skill.sha256").read_text().splitlines()}
    assert listed["scripts/check_video.sh"] == want and "SKILL.md" in listed, listed
    assert not (tmp_path / "work" / "r1" / "task" / "skill.sha256").exists()


def test_the_outer_deadline_has_a_kill_after_so_a_workload_that_ignores_sigterm_cannot_hang_the_cleanup(tmp_path, stubs):
    env, log = stubs
    rc, out, err = run_bench(tmp_path, env, tmp_path / "work")
    assert rc == 0, err
    line = Path(str(log) + ".systemd").read_text()
    assert " timeout -k " in line + " " or line.split("--", 1)[1].split()[0:2] == ["timeout", "-k"], line


# ---- poll_metrics.py -------------------------------------------------------------------------------------------------------------

class _Metrics(http.server.BaseHTTPRequestHandler):
    body = ""

    def do_GET(self):
        data = self.body.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def serve_metrics(text):
    handler = type("H", (_Metrics,), {"body": text})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def poll_once(tmp_path, text):
    srv = serve_metrics(text)
    try:
        out = tmp_path / "once.json"
        rc, so, se = run_cmd([sys.executable, str(BENCH / "poll_metrics.py"), f"http://127.0.0.1:{srv.server_port}", str(out), "once"], dict(os.environ), tmp_path, 30)
        assert rc == 0, se
        return json.loads(out.read_text())
    finally:
        srv.shutdown()


def test_series_that_differ_only_in_engine_or_model_labels_are_added_not_overwritten(tmp_path):
    text = ('vllm:num_requests_running{engine="0",model_name="m"} 1.0\n'
            'vllm:num_requests_running{engine="1",model_name="m"} 1.0\n'
            'vllm:generation_tokens_total{engine="0",model_name="m"} 100.0\n'
            'vllm:generation_tokens_total{engine="1",model_name="m"} 50.0\n')
    s = poll_once(tmp_path, text)
    assert s["vllm:num_requests_running"] == 2.0
    assert s["vllm:generation_tokens_total"] == 150.0


def test_a_single_series_and_the_labels_that_matter_are_kept(tmp_path):
    s = poll_once(tmp_path, 'vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 7.0\n'
                            'vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0\n')
    assert s['vllm:request_success_total{finished_reason="stop"}'] == 7.0
    assert s['vllm:request_success_total{finished_reason="length"}'] == 2.0


def test_a_metrics_body_over_the_read_cap_is_cut_not_read_whole(tmp_path):
    filler = "# " + "x" * 1000 + "\n"
    text = filler * 9500 + "vllm:num_requests_running 3.0\n"          # the gauge sits after about 9.5 MB
    assert len(text) > 9 * 1024 * 1024
    s = poll_once(tmp_path, text)
    assert "vllm:num_requests_running" not in s and "error" not in s, list(s)[:5]


# ---- collect.py ------------------------------------------------------------------------------------------------------------------

def make_run(tmp_path, video_name="final.mp4", check_text="echo tampered-check-ok\n", metrics=None):
    run = tmp_path / "work" / "p1"
    (run / "logs").mkdir(parents=True)
    task = run / "task"
    (task / "out").mkdir(parents=True)
    (task / "out" / video_name).write_bytes(b"x")
    chk = task / ".qwen" / "skills" / "code-video" / "scripts"
    chk.mkdir(parents=True)
    (chk / "check_video.sh").write_text(check_text)
    (run / "logs" / "started_at").write_text("2026-11-03T10:00:00Z\n")
    (run / "logs" / "ended_at").write_text("2026-11-03T10:10:00Z\n")
    (run / "logs" / "exit_code").write_text("0\n")
    (run / "logs" / "stream.jsonl").write_text("")
    if metrics is not None:
        (run / "logs" / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    return run


def trusted_skill(tmp_path):
    src = tmp_path / "trusted_skill"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "check_video.sh").write_text("echo trusted-check-ok\n")
    return src


def collect(run, env, cwd, rel=False, timeout=60):
    arg = os.path.relpath(run, cwd) if rel else str(run)
    rc, out, err = run_cmd([sys.executable, str(BENCH / "collect.py"), arg, "1280", "720", "24", "9", "11"], env, cwd, timeout)
    assert rc == 0, err
    return json.loads((Path(cwd) / arg / "logs" / "summary.json").read_text())


def test_a_relative_run_dir_gives_docker_absolute_paths(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path, rel=True)
    assert s["video"].get("frames") == 240, s["video"]
    for c in calls(log):
        if c["cmd"] == "run":
            assert c["workdir"].startswith("/") and all(v.split(":")[1].startswith("/") for v in c["vols"]), c


def test_a_render_named_with_a_space_is_probed_checked_and_hashed(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path, video_name="render v1.mp4")
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path)
    assert s["final_mp4_delivered"] is False and s["measured_file"].endswith("render v1.mp4")
    assert s["video"].get("frames") == 240, s["video"]
    assert "trusted-check-ok" in s["check_video"], s["check_video"]
    assert s["frame_hashes"]["frames"] == 3 and s["frame_hashes"]["distinct"] == 2, s["frame_hashes"]


def test_a_frame_count_ffprobe_did_not_report_is_null_not_zero(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), FFPROBE_NO_FRAMES="1"), tmp_path)
    assert s["video"]["frames"] is None, s["video"]


def test_a_name_with_shell_metacharacters_is_not_run_as_a_command(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path, video_name="a;touch pwned;.mp4")
    collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path)
    assert not (run / "task" / "pwned").exists() and not (tmp_path / "pwned").exists()


def test_the_checker_comes_from_the_trusted_copy_not_from_the_task_directory(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)                                    # the task's own copy of check_video.sh says tampered-check-ok
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path)
    assert "trusted-check-ok" in s["check_video"] and "tampered" not in s["check_video"], s["check_video"]
    mounts = [v for c in calls(log) if c["cmd"] == "run" for v in c["vols"]]
    assert any(v.endswith(":ro") and "trusted_skill" in v for v in mounts), mounts


def test_the_trusted_copy_is_found_from_the_path_the_run_recorded(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    (run / "logs" / "skill_src").write_text(str(trusted_skill(tmp_path)) + "\n")
    s = collect(run, env, tmp_path)
    assert "trusted-check-ok" in s["check_video"], s["check_video"]


def test_without_a_trusted_copy_the_check_is_not_run_and_says_so(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, env, tmp_path)
    assert "tampered" not in s["check_video"] and "not run" in s["check_video"], s["check_video"]


def test_files_of_the_skill_that_the_run_changed_are_listed(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    src = trusted_skill(tmp_path)
    (run / "logs" / "skill.sha256").write_text(
        subprocess.run("sha256sum ./scripts/check_video.sh", shell=True, cwd=src, capture_output=True, text=True).stdout)
    s = collect(run, dict(env, SKILL_SRC=str(src)), tmp_path)          # the task's copy differs from the recorded hash
    assert s["skill_files_changed_by_run"] == ["scripts/check_video.sh"], s.get("skill_files_changed_by_run")


def test_a_fifo_left_where_a_skill_file_belongs_does_not_hang_the_collection(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    target = run / "task" / ".qwen" / "skills" / "code-video" / "scripts" / "check_video.sh"
    target.unlink()
    os.mkfifo(target)
    src = trusted_skill(tmp_path)
    (run / "logs" / "skill.sha256").write_text(
        subprocess.run("sha256sum ./scripts/check_video.sh", shell=True, cwd=src, capture_output=True, text=True).stdout)
    s = collect(run, dict(env, SKILL_SRC=str(src)), tmp_path, timeout=20)
    assert s["skill_files_changed_by_run"] == ["scripts/check_video.sh"]


def test_a_timed_out_container_is_removed_by_its_id_and_collection_goes_on(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), DOCKER_STUB_MODE="sleep", CVBENCH_DOCKER_TIMEOUT_S="2"), tmp_path, timeout=45)
    removed = [c for c in calls(log) if c["cmd"] == "rm"]
    assert removed and all(a.startswith("STUBID-") for c in removed for a in c["args"] if not a.startswith("-")), removed
    assert "timed out" in json.dumps(s["video"]), s["video"]


def test_a_metrics_file_without_the_gauges_is_unavailable_not_idle(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path, metrics=[{"t": 1, "error": "refused"}, {"t": 2}, {"t": 3, "vllm:prompt_tokens_total": 5.0}])
    srv = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path)["server"]
    assert srv["max_requests_running"] is None and srv["max_requests_waiting"] is None, srv
    assert srv["samples_with_running_gauge"] == 0 and srv["samples_with_waiting_gauge"] == 0, srv


def test_only_samples_that_carry_a_gauge_count_toward_it(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path, metrics=[{"t": 1}, {"t": 2, "vllm:num_requests_running": 2.0}, {"t": 3, "vllm:num_requests_running": 1.0, "vllm:num_requests_waiting": 0.0}])
    srv = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path)["server"]
    assert srv["max_requests_running"] == 2.0 and srv["samples_with_running_gauge"] == 2 and srv["samples_with_more_than_one_running"] == 1, srv
    assert srv["max_requests_waiting"] == 0.0 and srv["samples_with_waiting_gauge"] == 1, srv


def test_the_stream_summary_says_how_many_messages_carried_usage(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    msgs = [{"type": "assistant", "message": {"content": [], "usage": {"input_tokens": 10, "output_tokens": 2}}},
            {"type": "assistant", "message": {"content": []}}]
    (run / "logs" / "stream.jsonl").write_text("".join(json.dumps(m) + "\n" for m in msgs))
    st = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path)["stream"]
    assert st["assistant_messages"] == 2 and st["assistant_messages_with_usage"] == 1, st
