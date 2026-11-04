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
    time.sleep(float(os.environ.get("DOCKER_STUB_RM_SLEEP", "0")))
    sys.exit(0)
if args[:1] != ["run"]:
    note({"cmd": args[:1], "args": args[1:]})
    sys.exit(0)
vols, workdir, cidfile, name, envs, tmpfs = [], None, None, None, {}, []
i = 1
while i < len(args):
    a = args[i]
    if a == "-v":
        vols.append(args[i + 1]); i += 2
    elif a == "-w":
        workdir = args[i + 1]; i += 2
    elif a == "-e":
        k, _, v = args[i + 1].partition("="); envs[k] = v; i += 2
    elif a == "--tmpfs":
        tmpfs.append(args[i + 1]); i += 2
    elif a == "--cidfile":
        cidfile = args[i + 1]; i += 2
    elif a == "--name":
        name = args[i + 1]; i += 2
    else:
        i += 1
note({"cmd": "run", "vols": vols, "workdir": workdir, "name": name, "cidfile": cidfile, "env": envs, "tmpfs": tmpfs, "args": args})
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
if mode in ("flood", "floodE"):                 # endless output until the client is killed: the collector must stop reading at its cap
    stream = sys.stdout.buffer if mode == "flood" else sys.stderr.buffer
    try:
        while True:
            stream.write(b"x" * 65536); stream.flush()
    except (BrokenPipeError, OSError):
        sys.exit(1)
if "sh" in args and args[args.index("sh") + 1:args.index("sh") + 2] == ["-c"]:
    cmd = args[args.index("sh") + 2]
    for v in vols:                              # the host path stands in for the container path when they differ
        parts = v.split(":")
        if parts[0] != parts[1]:
            cmd = cmd.replace(parts[1], parts[0])
    host_cwd = workdir or "."                   # the container's working directory, as a host path: a mount destination maps to its source
    for v in vols:
        parts = v.split(":")
        if workdir == parts[1]:
            host_cwd = parts[0]
    if not os.path.isdir(host_cwd):             # a tmpfs or an image directory: an empty directory stands in for it
        import tempfile
        host_cwd = tempfile.mkdtemp(prefix="stub-empty-")
    env = {"PATH": os.environ["PATH"], "HOME": "/nonexistent", **envs}      # a container does not inherit the host's environment
    for k, v in os.environ.items():             # except the knobs of the ffprobe and ffmpeg stand-ins, which stand for what the image holds
        if k.startswith(("FFPROBE_", "FFMPEG_STUB_")) or k == "CVBENCH_REAL_PYTHON":
            env[k] = v
    r = subprocess.run(["sh", "-c", cmd], cwd=host_cwd, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    sys.exit(r.returncode)
sys.exit(0)
'''

CURL_STUB = '#!/bin/sh\n[ -n "$CURL_STUB_SLEEP" ] && sleep "$CURL_STUB_SLEEP"\n[ -n "$CURL_STUB_FAIL" ] && exit 22\necho \'{"data":[{"id":"/models/under-test"}]}\'\n'

SYSTEMD_RUN_STUB = r'''#!/bin/sh
echo "$@" >> "$STUB_LOG.systemd"
while [ "$1" != "--" ]; do shift; done
shift
exec "$@"
'''

# ffprobe and ffmpeg refuse anything but one existing file argument: an unquoted name with a space reaches them as two words.
PYTHON_STUB = r'''#!/bin/sh
if [ -n "$STUB_LOG" ]; then
  echo "ARGS $*" >> "$STUB_LOG.python"
  env | grep '^PYTHON' | sed 's/^/ENV /' >> "$STUB_LOG.python"
fi
exec "$CVBENCH_REAL_PYTHON" "$@"
'''

FFPROBE_STUB = r'''#!/bin/sh
for last; do :; done
if [ ! -f "$last" ]; then echo "ffprobe: no such file: $last" >&2; exit 1; fi
[ -n "$FFPROBE_GARBAGE" ] && { echo "this is not json"; exit 0; }
frames=',"nb_read_frames":"'"${FFPROBE_FRAMES:-240}"'"'; [ -n "$FFPROBE_NO_FRAMES" ] && frames=
echo '{"streams":[{"codec_type":"video","codec_name":"h264","width":1280,"height":720,"pix_fmt":"yuv420p","avg_frame_rate":"24/1"'"$frames"'}],"format":{"duration":"10.000","size":"123456"}}'
'''

FFMPEG_STUB = r'''#!/bin/sh
for a; do [ "$a" = "-i" ] && want=1 && continue; if [ -n "$want" ]; then f="$a"; want=; fi; done
if [ ! -f "$f" ]; then echo "ffmpeg: no such file: $f" >&2; exit 1; fi
if [ "$FFMPEG_STUB_FAIL" = silent ]; then exit 1; fi
if [ -n "$FFMPEG_STUB_FAIL" ]; then echo "ffmpeg: Invalid data found when processing input" >&2; exit 1; fi
echo '#format: frame checksums'
# real framemd5 records: stream, dts, pts, duration, size, a 32-digit md5; one new picture every 24 frames
awk -v n="${FFMPEG_STUB_FRAMES:-240}" 'BEGIN { for (i = 0; i < n; i++) printf "0, %10d, %10d, %8d, %9d, %032x\n", i, i, 1, 2764800, int(i / 24) + 1 }'
if [ -n "$FFMPEG_STUB_GARBAGE" ]; then echo "[warn] something odd happened"; echo "0, 999, 999, 1, 1, 0xnothex"; fi
if [ -n "$FFMPEG_STUB_WARN" ]; then echo "Error while decoding stream #0:0: Invalid data found" >&2; fi
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
    write_exec(bin_dir / "python3", PYTHON_STUB)
    log = tmp_path / "stub.log"
    log.write_text("")
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", STUB_LOG=str(log), CVBENCH_REAL_PYTHON=sys.executable)
    for k in ("DOCKER_STUB_MODE", "DOCKER_STUB_CONFLICT", "CVBENCH_DOCKER_TIMEOUT_S", "SKILL_SRC", "WORK_ROOT", "FFPROBE_NO_FRAMES", "FFPROBE_FRAMES",
          "FFMPEG_STUB_FAIL", "FFMPEG_STUB_FRAMES", "FFMPEG_STUB_GARBAGE", "FFMPEG_STUB_WARN", "FFPROBE_GARBAGE", "CURL_STUB_SLEEP", "CURL_STUB_FAIL",
          "DOCKER_STUB_RM_SLEEP", "CVBENCH_CLEANUP_TIMEOUT_S", "CVBENCH_STREAM_CAP_BYTES", "CVBENCH_LINE_CAP_BYTES", "CVBENCH_OUTPUT_CAP_BYTES", "CVBENCH_PYTHON"):
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
        rc, so, se = run_cmd([sys.executable, "-I", "-S", str(BENCH / "poll_metrics.py"), f"http://127.0.0.1:{srv.server_port}", str(out), "once"], dict(os.environ), tmp_path, 30)
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
    rc, out, err = run_cmd([sys.executable, "-I", str(BENCH / "collect.py"), arg, "1280", "720", "24", "9", "11"], env, cwd, timeout)
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
    assert s["frame_hashes"]["frames"] == 240 and s["frame_hashes"]["distinct"] == 10, s["frame_hashes"]
    assert s["frame_hashes"]["identical_to_previous"] == 230 and s["frame_hashes"]["longest_identical_run_frames"] == 24, s["frame_hashes"]
    assert s["frame_hashes"]["error"] is None, s["frame_hashes"]


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
    # the recorded hash is that of EMPTY content, which is also what a read of an idle FIFO returns: only the regular-file check tells them apart
    (run / "logs" / "skill.sha256").write_text("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ./scripts/check_video.sh\n")
    s = collect(run, dict(env, SKILL_SRC=str(src)), tmp_path, timeout=20)
    assert s["skill_files_changed_by_run"] == ["scripts/check_video.sh"]
    assert any(r["path"].endswith("check_video.sh") and "not a regular file" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]


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


# ---- round 2 of the Copilot review of PR 450 -------------------------------------------------------------------------------------
# 1. a file the model wrote must not shadow a module the checker imports; 2. a run directory is reserved atomically; 3. a measurement that
# failed is reported as failed, never as a figure.

def shadowing_skill(tmp_path):
    """A trusted skill whose checker reads its script from stdin and imports json, as the public check_video.sh does."""
    src = tmp_path / "trusted_skill"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "check_video.sh").write_text(
        'python3 - "$1" <<\'EOF\'\nimport json, sys\nprint("checker-imported-json-from", json.__file__)\nEOF\n')
    return src


def test_a_module_the_model_wrote_cannot_shadow_the_checker_imports(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    (run / "task" / "json.py").write_text('print("SHADOWED-OK")\nraise SystemExit(0)\n')
    s = collect(run, dict(env, SKILL_SRC=str(shadowing_skill(tmp_path))), tmp_path)
    assert "SHADOWED" not in s["check_video"], s["check_video"]
    assert "checker-imported-json-from" in s["check_video"] and str(run / "task") not in s["check_video"], s["check_video"]


def test_the_checker_runs_from_the_trusted_directory_with_an_absolute_video_path_and_a_scrubbed_python_environment(tmp_path, stubs):
    import shlex
    env, log = stubs
    run = make_run(tmp_path)
    collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path)
    runs = [c for c in calls(log) if c["cmd"] == "run"]
    check = [c for c in runs if any("check_video.sh" in a for a in c["args"])]
    assert len(check) == 1, runs
    c = check[0]
    assert c["workdir"] == "/trusted-skill", c["workdir"]
    words = shlex.split(c["args"][c["args"].index("-c") + 1])
    assert words[:2] == ["sh", "/trusted-skill/scripts/check_video.sh"] and words[2].startswith("/"), words
    assert words[2] == str(run / "task" / "out" / "final.mp4"), words
    for k, v in {"PYTHONSAFEPATH": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""}.items():
        assert c["env"].get(k) == v, (k, c["env"])
    assert all(r["workdir"] != str(run / "task") for r in runs), "no call runs from the directory the model can write"
    assert all(r["workdir"] == "/trusted-skill" or (r["workdir"] == "/scratch" and any(x.startswith("/scratch:") for x in r["tmpfs"])) for r in runs), runs


def test_the_shell_script_runs_python_isolated_so_a_file_in_the_current_directory_cannot_shadow_it(tmp_path, stubs):
    env, log = stubs
    cwd = tmp_path / "caller"
    cwd.mkdir()
    (cwd / "json.py").write_text('import sys\nsys.stdout.write("SHADOWED")\nsys.exit(7)\n')
    rc, out, err = run_bench(tmp_path, env, tmp_path / "work", cwd=cwd)
    assert rc == 0, (out, err)
    assert "/models/under-test" in (tmp_path / "work" / "r1" / "logs" / "qwen-settings.json").read_text()


def test_two_runs_with_one_name_cannot_both_start(tmp_path, stubs):
    env, log = stubs
    skill, brief = bench_setup(tmp_path)
    e = dict(env, WORK_ROOT=str(tmp_path / "work"), SKILL_SRC=str(skill), ENDPOINT="http://127.0.0.1:9", CURL_STUB_SLEEP="2")
    argv = ["bash", str(BENCH / "run_bench.sh"), "r1", str(brief), "1", "10"]
    procs = [subprocess.Popen(argv, env=e, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
             for _ in range(2)]
    results = []
    try:
        for p in procs:
            o, er = p.communicate(timeout=90)
            results.append((o, er, p.returncode))
    finally:
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    assert sorted(r[2] for r in results)[0] == 0 and sorted(r[2] for r in results)[1] != 0, results
    refused = [r for r in results if r[2] != 0][0]
    assert "refusing to reuse" in refused[0] + refused[1], refused
    assert len([c for c in calls(log) if c["cmd"] == "run"]) == 1, "only the run that reserved the directory starts a container"
    logs = tmp_path / "work" / "r1" / "logs"
    cid = (logs / "container.id").read_text().strip()
    assert cid.startswith("STUBID-") and (logs / "exit_code").read_text().strip() == "0"
    removed = [a for c in calls(log) if c["cmd"] == "rm" for a in c["args"] if not a.startswith("-")]
    assert removed == [cid], "the refused run removes nothing, the winner removes its own container"


def test_the_container_name_comes_from_the_reserved_run_directory_so_two_roots_with_one_run_name_do_not_collide(tmp_path, stubs):
    env, log = stubs
    for sub in ("a", "b"):
        base = tmp_path / sub
        base.mkdir()
        rc, out, err = run_bench(base, env, base / "work")
        assert rc == 0, err
    names = [c["name"] for c in calls(log) if c["cmd"] == "run"]
    assert len(names) == 2 and names[0] != names[1] and all(n.startswith("cvbench-r1-") for n in names), names


def failed(s, name):
    assert name in s["measurement_failures"]["names"], s["measurement_failures"]


def test_a_container_that_timed_out_during_hashing_gives_no_frame_count(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), DOCKER_STUB_MODE="sleep", CVBENCH_DOCKER_TIMEOUT_S="2"), tmp_path, timeout=90)
    fh = s["frame_hashes"]
    assert fh["frames"] is None and fh["distinct"] is None and fh["identical_to_previous"] is None and fh["longest_identical_run_frames"] is None, fh
    assert "timed out" in fh["error"], fh
    failed(s, "frame_hashes")
    assert s["measurement_failures"]["count"] == len(s["measurement_failures"]["names"]) >= 2


def test_a_docker_error_is_a_failed_measurement_and_not_a_frame(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), DOCKER_STUB_CONFLICT="1"), tmp_path)
    fh = s["frame_hashes"]
    assert fh["frames"] is None and "docker" in fh["error"].lower(), fh
    failed(s, "frame_hashes")
    failed(s, "video")
    failed(s, "check_video")
    assert s["check_video"].startswith("not measured: docker error"), s["check_video"]


def test_an_ffmpeg_failure_is_reported_with_its_reason_and_not_as_zero_frames(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), FFMPEG_STUB_FAIL="1"), tmp_path)
    fh = s["frame_hashes"]
    assert fh["frames"] is None and fh["longest_identical_run_frames"] is None and "Invalid data" in fh["error"], fh
    failed(s, "frame_hashes")


def test_lines_that_are_not_framemd5_records_make_the_measurement_fail_and_are_counted(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), FFMPEG_STUB_GARBAGE="1"), tmp_path)
    fh = s["frame_hashes"]
    assert fh["frames"] is None and fh["invalid_lines"] == 2 and "invalid" in fh["error"], fh
    failed(s, "frame_hashes")


def test_a_hash_count_that_differs_from_the_probe_frame_count_is_a_failure(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), FFMPEG_STUB_FRAMES="200"), tmp_path)
    fh = s["frame_hashes"]
    assert fh["frames"] is None and "240" in fh["error"] and "200" in fh["error"], fh
    failed(s, "frame_hashes")


def test_a_good_run_reports_no_measurement_failures(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path)
    assert s["measurement_failures"] == {"count": 0, "names": []}, s["measurement_failures"]
    assert s["frame_hashes"]["error"] is None and s["frame_hashes"]["invalid_lines"] == 0, s["frame_hashes"]


def test_a_failed_start_gives_the_run_name_back(tmp_path, stubs):
    env, log = stubs
    skill, brief = bench_setup(tmp_path)
    base = dict(env, WORK_ROOT=str(tmp_path / "work"), SKILL_SRC=str(skill), ENDPOINT="http://127.0.0.1:9")
    argv = ["bash", str(BENCH / "run_bench.sh"), "r1", str(brief), "1", "10"]
    rc, out, err = run_cmd(argv, dict(base, CURL_STUB_FAIL="1"), tmp_path, 60)
    assert rc != 0 and not (tmp_path / "work" / "r1").exists(), "the empty reserved directory is removed when the start fails"
    rc, out, err = run_cmd(argv, base, tmp_path, 60)
    assert rc == 0, err


def test_an_ffmpeg_failure_with_no_message_is_still_a_failure_with_its_exit_status(tmp_path, stubs):
    env, log = stubs
    s = collect(make_run(tmp_path), dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), FFMPEG_STUB_FAIL="silent"), tmp_path)
    assert s["frame_hashes"]["frames"] is None and "exited 1" in s["frame_hashes"]["error"], s["frame_hashes"]
    failed(s, "frame_hashes")


def test_ffmpeg_error_output_with_exit_status_zero_is_a_failure(tmp_path, stubs):
    env, log = stubs
    s = collect(make_run(tmp_path), dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), FFMPEG_STUB_WARN="1"), tmp_path)
    assert s["frame_hashes"]["frames"] is None and "ffmpeg reported errors" in s["frame_hashes"]["error"], s["frame_hashes"]
    failed(s, "frame_hashes")


def test_no_frame_records_at_all_is_a_failure_and_not_zero_frames(tmp_path, stubs):
    env, log = stubs
    s = collect(make_run(tmp_path), dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), FFPROBE_NO_FRAMES="1", FFMPEG_STUB_FRAMES="0"), tmp_path)
    assert s["frame_hashes"]["frames"] is None and "no frame records" in s["frame_hashes"]["error"], s["frame_hashes"]
    failed(s, "frame_hashes")


def test_probe_output_that_is_not_json_is_a_failed_measurement(tmp_path, stubs):
    env, log = stubs
    s = collect(make_run(tmp_path), dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), FFPROBE_GARBAGE="1"), tmp_path)
    assert "probe_error" in s["video"] and "this is not json" in s["video"]["raw"], s["video"]
    failed(s, "video")


def test_more_frame_records_than_the_cap_allows_is_a_failure(tmp_path, stubs):
    env, log = stubs
    e = dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), CVBENCH_OUTPUT_CAP_BYTES="1000")
    s = collect(make_run(tmp_path), e, tmp_path)
    assert s["frame_hashes"]["frames"] is None and "stdout exceeded 1000 bytes" in s["frame_hashes"]["error"], s["frame_hashes"]
    failed(s, "frame_hashes")


# ---- PR 450, Copilot round 3 (the limit): isolation findings ---------------------------------------------------------------------

def evil_startup(tmp_path, marker):
    """Directories and files that would run model-controlled (or planted) code if the host interpreter honoured them at startup."""
    evil = tmp_path / "evil"
    evil.mkdir()
    code = f'open({str(marker)!r}, "a").write(__file__ + "\\n")\n'
    for name in ("sitecustomize.py", "usercustomize.py", "startup.py"):
        (evil / name).write_text(code)
    return evil


def test_the_control_proves_that_startup_code_on_pythonpath_really_runs_in_a_plain_interpreter(tmp_path):
    marker = tmp_path / "MARKER"
    evil = evil_startup(tmp_path, marker)
    subprocess.run([sys.executable, "-c", "pass"], env=dict(os.environ, PYTHONPATH=str(evil)), check=True)
    assert marker.exists(), "the sitecustomize hook does not fire here, so the isolation test below would prove nothing"


def test_the_collector_wrapper_starts_python_isolated_so_startup_code_cannot_run(tmp_path, stubs):
    env, log = stubs
    marker = tmp_path / "MARKER"
    evil = evil_startup(tmp_path, marker)
    cwd = tmp_path / "caller"
    cwd.mkdir()
    for name in ("sitecustomize.py", "json.py"):
        (cwd / name).write_text(f'open({str(marker)!r}, "a").write("cwd " + __file__ + "\\n")\nraise SystemExit(9)\n')
    # the interpreter the wrapper starts is a script that writes down how it was started, then runs the real one
    spy = write_exec(tmp_path / "spy-python", '#!/bin/sh\n{ echo "PWD $(pwd)"; echo "ARGS $*"; env | grep \'^PYTHON\' | sed \'s/^/ENV /\'; } >> "$STUB_LOG.wrapper"\n'
                                              'exec "$CVBENCH_REAL_PYTHON" "$@"\n')
    run = make_run(tmp_path)
    e = dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), CVBENCH_PYTHON=str(spy), PYTHONPATH=str(evil), PYTHONSTARTUP=str(evil / "startup.py"),
             PYTHONUSERBASE=str(evil), PYTHONINSPECT="1")
    rc, out, err = run_cmd(["sh", str(BENCH / "collect.sh"), os.path.relpath(run, cwd), "1280", "720", "24", "9", "11"], e, cwd, 60)
    assert rc == 0, (rc, err)
    assert not marker.exists(), marker.read_text()
    assert json.loads((run / "logs" / "summary.json").read_text())["video"]["frames"] == 240
    rows = Path(str(log) + ".wrapper").read_text().splitlines()
    assert "PWD /" in rows, rows                                                      # the caller's directory (and its json.py) is left behind
    assert [r for r in rows if r.startswith("ARGS ")][0].startswith("ARGS -I -S "), rows
    assert not [r for r in rows if r.startswith("ENV ")], rows                         # no PYTHON* variable reaches the interpreter


def test_the_collector_refuses_to_start_without_isolation_and_says_how_to_start_it(tmp_path, stubs):
    """A script cannot undo startup code that ran before its first line, so it refuses and does not pretend to re-exec."""
    env, log = stubs
    run = make_run(tmp_path)
    rc, out, err = run_cmd([sys.executable, str(BENCH / "collect.py"), str(run), "1280", "720", "24", "9", "11"],
                           dict(env, SKILL_SRC=str(trusted_skill(tmp_path))), tmp_path, 30)
    assert rc == 2 and "collect.sh" in err and "-I" in err, (rc, err)
    assert not (run / "logs" / "summary.json").exists()


def test_poll_metrics_refuses_to_start_without_isolation(tmp_path):
    rc, out, err = run_cmd([sys.executable, str(BENCH / "poll_metrics.py"), "http://127.0.0.1:9", str(tmp_path / "o.json"), "once"], dict(os.environ), tmp_path, 30)
    assert rc == 2 and "-I" in err, (rc, err)
    assert not (tmp_path / "o.json").exists()


def test_every_host_python_in_the_shell_script_is_isolated_and_gets_no_python_variables(tmp_path, stubs):
    env, log = stubs
    marker = tmp_path / "MARKER"
    evil = evil_startup(tmp_path, marker)
    rc, out, err = run_bench(tmp_path, dict(env, PYTHONPATH=str(evil), PYTHONSTARTUP=str(evil / "startup.py"), PYTHONHOME=""), tmp_path / "work")
    assert rc == 0, (out, err)
    rows = (Path(str(log) + ".python")).read_text().splitlines()
    starts = [r for r in rows if r.startswith("ARGS ") and "/stubbin/" not in r]       # the docker stand-in is itself a python script: its start is not the shell script's
    assert len(starts) >= 3, rows
    assert all(r.startswith("ARGS -I -S ") for r in starts), starts
    assert not [r for r in rows if r.startswith("ENV ")], [r for r in rows if r.startswith("ENV ")]
    assert not marker.exists()


def beneath_setup(tmp_path, env, **kw):
    run = make_run(tmp_path)
    return run, dict(env, SKILL_SRC=str(trusted_skill(tmp_path)), **kw)


def test_a_symlinked_spec_file_is_refused_and_its_target_size_is_not_reported(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret " * 1000)
    (run / "task" / "SPEC.md").symlink_to(outside)
    s = collect(run, e, tmp_path)
    assert s["files"]["SPEC.md"] is None, s["files"]
    assert any(r["path"] == "SPEC.md" and "link" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]


def test_an_out_directory_that_is_a_symlink_is_not_followed(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "final.mp4").write_bytes(b"x")
    import shutil
    shutil.rmtree(run / "task" / "out")
    (run / "task" / "out").symlink_to(outside)
    s = collect(run, e, tmp_path)
    assert s["final_mp4_delivered"] is False and s["measured_file"] is None, s
    assert not any("elsewhere" in json.dumps(c) for c in calls(log)), "no container call may name the outside path"
    assert any(r["path"] == "out" and "link" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]


def test_a_final_mp4_that_is_a_symlink_to_an_outside_file_is_not_measured(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    outside = tmp_path / "other.mp4"
    outside.write_bytes(b"x")
    (run / "task" / "out" / "final.mp4").unlink()
    (run / "task" / "out" / "final.mp4").symlink_to(outside)
    s = collect(run, e, tmp_path)
    assert s["final_mp4_delivered"] is False and s["measured_file"] is None, s
    assert not [c for c in calls(log) if c["cmd"] == "run"], "nothing should have been sent to a container"
    assert any(r["path"] == "out/final.mp4" for r in s["refused_paths"]), s["refused_paths"]


def test_a_linked_skill_directory_is_not_hashed_through_the_link_and_the_files_count_as_changed(tmp_path, stubs):
    env, log = stubs
    src = trusted_skill(tmp_path)
    run = make_run(tmp_path)
    (run / "logs" / "skill.sha256").write_text(
        subprocess.run("sha256sum ./scripts/check_video.sh", shell=True, cwd=src, capture_output=True, text=True).stdout)
    import shutil
    shutil.rmtree(run / "task" / ".qwen")
    (run / "task" / ".qwen").symlink_to(tmp_path)                 # a link whose target holds a pristine copy under skills/code-video
    pristine = tmp_path / "skills" / "code-video"
    shutil.copytree(src, pristine)
    s = collect(run, dict(env, SKILL_SRC=str(src)), tmp_path)
    assert s["skill_files_changed_by_run"] == ["scripts/check_video.sh"], s["skill_files_changed_by_run"]
    assert any(r["path"].startswith(".qwen") and "link" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]


def test_a_symlinked_file_inside_the_skill_copy_is_listed_as_added_and_not_followed(tmp_path, stubs):
    env, log = stubs
    src = trusted_skill(tmp_path)
    run = make_run(tmp_path)
    (run / "logs" / "skill.sha256").write_text(
        subprocess.run("sha256sum ./scripts/check_video.sh", shell=True, cwd=src, capture_output=True, text=True).stdout)
    (run / "task" / ".qwen" / "skills" / "code-video" / "scripts" / "check_video.sh").write_text("echo trusted-check-ok\n")
    outside = tmp_path / "o.txt"
    outside.write_text("x")
    (run / "task" / ".qwen" / "skills" / "code-video" / "loot.txt").symlink_to(outside)
    s = collect(run, dict(env, SKILL_SRC=str(src)), tmp_path)
    assert s["skill_files_changed_by_run"] == [], s["skill_files_changed_by_run"]
    assert any(r["path"].endswith("loot.txt") for r in s["refused_paths"]), s["refused_paths"]


def test_a_dangling_symlink_among_the_candidates_does_not_stop_the_collection(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    out = run / "task" / "out"
    (out / "final.mp4").unlink()
    (out / "old.mp4").write_bytes(b"x")
    (out / "dangling.mp4").symlink_to("/nonexistent/target.mp4")
    os.mkfifo(out / "pipe.mp4")
    s = collect(run, e, tmp_path, timeout=30)
    assert s["final_mp4_delivered"] is False and s["measured_file"] == "out/old.mp4", s
    assert {r["path"] for r in s["refused_paths"]} >= {"out/dangling.mp4", "out/pipe.mp4"}, s["refused_paths"]


def test_only_dangling_candidates_means_nothing_is_measured_and_no_crash(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    out = run / "task" / "out"
    (out / "final.mp4").unlink()
    (out / "dangling.mp4").symlink_to("/nonexistent/target.mp4")
    s = collect(run, e, tmp_path, timeout=30)
    assert s["measured_file"] is None and "video" not in s, s


def test_a_final_mp4_that_is_a_fifo_is_not_opened_and_does_not_hang(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    (run / "task" / "out" / "final.mp4").unlink()
    os.mkfifo(run / "task" / "out" / "final.mp4")
    s = collect(run, e, tmp_path, timeout=30)
    assert s["final_mp4_delivered"] is False and s["measured_file"] is None, s


def test_a_container_that_floods_stdout_is_stopped_at_the_cap_and_removed_by_id(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env, DOCKER_STUB_MODE="flood", CVBENCH_OUTPUT_CAP_BYTES="200000", CVBENCH_DOCKER_TIMEOUT_S="40")
    s = collect(run, e, tmp_path, timeout=60)
    assert "exceeded" in json.dumps(s["video"]) and "stdout" in json.dumps(s["video"]), s["video"]
    assert any(c["cmd"] == "rm" for c in calls(log)), "the container of a call that overran its cap must be removed by id"
    assert s["frame_hashes"]["frames"] is None and s["frame_hashes"]["error"], s["frame_hashes"]


def test_a_container_that_floods_stderr_is_stopped_at_the_cap(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env, DOCKER_STUB_MODE="floodE", CVBENCH_OUTPUT_CAP_BYTES="200000", CVBENCH_DOCKER_TIMEOUT_S="40")
    s = collect(run, e, tmp_path, timeout=60)
    assert "stderr" in json.dumps(s["video"]) and "exceeded" in json.dumps(s["video"]), s["video"]


def test_a_huge_agent_stream_is_read_to_a_cap_and_says_so(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env, CVBENCH_STREAM_CAP_BYTES="100000")
    line = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "x" * 500}], "usage": {"input_tokens": 5, "output_tokens": 1}}}) + "\n"
    (run / "logs" / "stream.jsonl").write_text(line * 2000)               # about 1 MB
    s = collect(run, e, tmp_path)
    info = s["stream"]["read"]
    assert info["truncated"] is True and info["bytes_read"] <= 100000 + len(line), info
    assert 0 < s["stream"]["assistant_messages"] < 2000, s["stream"]["assistant_messages"]


def test_one_enormous_stream_line_is_skipped_and_counted_and_the_rest_is_read(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env, CVBENCH_LINE_CAP_BYTES="10000")
    good = json.dumps({"type": "assistant", "message": {"content": [], "usage": {}}}) + "\n"
    (run / "logs" / "stream.jsonl").write_text(good + "y" * 3_000_000 + "\n" + good)
    s = collect(run, e, tmp_path)
    assert s["stream"]["assistant_messages"] == 2, s["stream"]
    assert s["stream"]["read"]["long_lines_skipped"] == 1 and s["stream"]["read"]["truncated"] is False, s["stream"]["read"]


def test_a_container_removal_that_times_out_is_recorded_and_collection_goes_on(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env, DOCKER_STUB_MODE="sleep", CVBENCH_DOCKER_TIMEOUT_S="2", DOCKER_STUB_RM_SLEEP="30", CVBENCH_CLEANUP_TIMEOUT_S="1")
    s = collect(run, e, tmp_path, timeout=90)
    assert s["cleanup_failures"], s.get("cleanup_failures")
    f = s["cleanup_failures"][0]
    assert f["container"].startswith("STUBID-") and "timed out" in f["reason"], f
    assert "measurement_failures" in s and s["measurement_failures"]["count"] >= 1


def test_a_removal_that_docker_refuses_is_recorded_too(tmp_path, stubs):
    env, log = stubs
    (tmp_path / "stubbin").joinpath("docker").write_text(DOCKER_STUB.replace('time.sleep(float(os.environ.get("DOCKER_STUB_RM_SLEEP", "0")))\n    sys.exit(0)',
                                                                           'print("Error response from daemon: removal in progress", file=sys.stderr)\n    sys.exit(1)'))
    run, e = beneath_setup(tmp_path, env, DOCKER_STUB_MODE="sleep", CVBENCH_DOCKER_TIMEOUT_S="2")
    s = collect(run, e, tmp_path, timeout=60)
    assert s["cleanup_failures"] and "exit" in s["cleanup_failures"][0]["reason"], s["cleanup_failures"]


def test_the_shell_script_cleanup_is_bounded_and_records_a_removal_that_hangs(tmp_path, stubs):
    env, log = stubs
    import time
    t0 = time.time()
    rc, out, err = run_bench(tmp_path, env, tmp_path / "work", extra={"DOCKER_STUB_RM_SLEEP": "40", "CVBENCH_CLEANUP_TIMEOUT_S": "1"})
    assert rc == 0, (out, err)
    assert time.time() - t0 < 30, "the exit trap waited for the hanging removal"
    text = (tmp_path / "work" / "r1" / "logs" / "cleanup.log").read_text()
    assert "STUBID-" in text and "timed out" in text, text


def test_a_recorded_skill_file_replaced_by_a_link_to_a_pristine_copy_still_counts_as_changed(tmp_path, stubs):
    env, log = stubs
    src = trusted_skill(tmp_path)
    run = make_run(tmp_path)
    (run / "logs" / "skill.sha256").write_text(
        subprocess.run("sha256sum ./scripts/check_video.sh", shell=True, cwd=src, capture_output=True, text=True).stdout)
    target = run / "task" / ".qwen" / "skills" / "code-video" / "scripts" / "check_video.sh"
    target.unlink()
    target.symlink_to(src / "scripts" / "check_video.sh")                      # the link's target has exactly the recorded content
    s = collect(run, dict(env, SKILL_SRC=str(src)), tmp_path)
    assert s["skill_files_changed_by_run"] == ["scripts/check_video.sh"], s["skill_files_changed_by_run"]
    assert any(r["path"].endswith("check_video.sh") and "link" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]


def test_a_directory_with_more_entries_than_the_limit_is_cut_and_says_so(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    out = run / "task" / "out"
    (out / "final.mp4").unlink()
    for i in range(5100):
        (out / f"r{i:05d}.mp4").write_bytes(b"")
    s = collect(run, e, tmp_path, timeout=60)
    assert s["measured_file"] and s["measured_file"].startswith("out/r"), s["measured_file"]
    assert any("more than 5000 entries" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]


def test_a_skill_copy_nested_deeper_than_the_limit_is_cut_and_says_so(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    deep = run / "task" / ".qwen" / "skills" / "code-video" / "d1" / "d2" / "d3" / "d4" / "d5" / "d6" / "d7" / "d8" / "d9"
    deep.mkdir(parents=True)
    (deep / "x.txt").write_text("x")
    (run / "logs" / "skill.sha256").write_text("")
    s = collect(run, e, tmp_path)
    assert any("deeper than 8 levels" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]
    assert not any(a.endswith("x.txt") for a in s["skill_files_added_by_run"]), s["skill_files_added_by_run"]


def test_a_file_the_run_added_inside_a_skill_subdirectory_is_listed(tmp_path, stubs):
    env, log = stubs
    src = trusted_skill(tmp_path)
    run = make_run(tmp_path)
    (run / "logs" / "skill.sha256").write_text(
        subprocess.run("sha256sum ./scripts/check_video.sh", shell=True, cwd=src, capture_output=True, text=True).stdout)
    (run / "task" / ".qwen" / "skills" / "code-video" / "scripts" / "check_video.sh").write_text("echo trusted-check-ok\n")
    (run / "task" / ".qwen" / "skills" / "code-video" / "scripts" / "helper.py").write_text("print(1)\n")
    (run / "task" / ".qwen" / "skills" / "code-video" / "crafts").mkdir()
    (run / "task" / ".qwen" / "skills" / "code-video" / "crafts" / "notes.md").write_text("n\n")
    s = collect(run, dict(env, SKILL_SRC=str(src)), tmp_path)
    assert s["skill_files_added_by_run"] == ["crafts/notes.md", "scripts/helper.py"], s["skill_files_added_by_run"]
    assert s["skill_files_changed_by_run"] == [], s["skill_files_changed_by_run"]


def test_a_recorded_skill_file_over_the_read_cap_is_not_hashed_and_counts_as_changed(tmp_path, stubs):
    env, log = stubs
    src = trusted_skill(tmp_path)
    run = make_run(tmp_path)
    (run / "logs" / "skill.sha256").write_text(
        subprocess.run("sha256sum ./scripts/check_video.sh", shell=True, cwd=src, capture_output=True, text=True).stdout)
    big = run / "task" / ".qwen" / "skills" / "code-video" / "scripts" / "check_video.sh"
    with open(big, "wb") as f:
        f.truncate(17 << 20)                                                       # a sparse file over the 16 MiB cap
    s = collect(run, dict(env, SKILL_SRC=str(src)), tmp_path, timeout=30)
    assert s["skill_files_changed_by_run"] == ["scripts/check_video.sh"], s["skill_files_changed_by_run"]
    assert any(r["path"].endswith("check_video.sh") and "read cap" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]


def test_a_skill_copy_with_more_files_than_the_limit_is_cut_and_says_so(tmp_path, stubs):
    env, log = stubs
    run, e = beneath_setup(tmp_path, env)
    root = run / "task" / ".qwen" / "skills" / "code-video" / "many"
    root.mkdir()
    for i in range(5100):
        (root / f"f{i:05d}.txt").write_text("")
    (run / "logs" / "skill.sha256").write_text("")
    s = collect(run, e, tmp_path, timeout=60)
    assert any("more than 5000 entries" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]
    assert len(s["skill_files_added_by_run"]) <= 5000, len(s["skill_files_added_by_run"])


# ---- PR 450, Copilot round 4 -------------------------------------------------------------------------------------------------------

def peak_kib_of_collect(run, env, cwd):
    """The peak resident size (KiB) of the largest child of a fresh interpreter that runs collect.sh once. A fresh parent keeps earlier tests'
    children out of RUSAGE_CHILDREN, whose maximum covers every child the parent ran."""
    code = ("import resource, subprocess, sys\n"
            "r = subprocess.run(sys.argv[1:], capture_output=True, text=True)\n"
            "print(r.returncode, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)\n")
    rc, out, err = run_cmd([sys.executable, "-c", code, "sh", str(BENCH / "collect.sh"), str(run), "1280", "720", "24", "9", "11"], env, cwd, 120)
    assert rc == 0, err
    code_, kib = out.split()
    assert code_ == "0", out
    return int(kib)


HUGE_DIR_FILES = 120_000


def test_a_skill_copy_directory_with_a_huge_number_of_names_is_not_held_whole_in_memory(tmp_path, stubs):
    """The entry limit (5000) was checked after `sorted()` had listed and statted every name, so a model-made directory of millions of names
    used the host's memory before the limit applied. Reading stops at the budget plus one name."""
    env, log = stubs
    run = make_run(tmp_path)
    many = run / "task" / ".qwen" / "skills" / "code-video" / "many"
    many.mkdir()
    for i in range(HUGE_DIR_FILES):
        with open(many / f"f{i:06d}.txt", "w"):
            pass
    (run / "logs" / "skill.sha256").write_text("")
    e = dict(env, SKILL_SRC=str(trusted_skill(tmp_path)))
    kib = peak_kib_of_collect(run, e, tmp_path)
    s = json.loads((run / "logs" / "summary.json").read_text())
    assert any("more than 5000 entries" in r["reason"] for r in s["refused_paths"]), s["refused_paths"]
    assert len(s["skill_files_added_by_run"]) <= 5000
    assert kib < 60_000, f"collect.py peaked at {kib} KiB on a directory of {HUGE_DIR_FILES} names"


def test_a_relative_skill_src_still_names_the_trusted_skill_after_the_wrapper_changes_directory(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    trusted_skill(tmp_path)
    cwd = tmp_path
    rc, out, err = run_cmd(["sh", str(BENCH / "collect.sh"), os.path.relpath(run, cwd), "1280", "720", "24", "9", "11"],
                           dict(env, SKILL_SRC="trusted_skill"), cwd, 60)
    assert rc == 0, (rc, err)
    s = json.loads((run / "logs" / "summary.json").read_text())
    assert "trusted-check-ok" in s["check_video"] and s["check_video_source"], s["check_video"]


def test_a_skill_src_that_names_no_directory_is_passed_on_and_the_checker_is_not_run(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    rc, out, err = run_cmd(["sh", str(BENCH / "collect.sh"), str(run), "1280", "720", "24", "9", "11"], dict(env, SKILL_SRC="no/such/dir"), tmp_path, 60)
    assert rc == 0, (rc, err)
    s = json.loads((run / "logs" / "summary.json").read_text())
    assert s["check_video"].startswith("not run"), s["check_video"]


def skill_with_checker(tmp_path, script):
    src = tmp_path / "trusted_skill"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "check_video.sh").write_text(script)
    return src


@pytest.mark.parametrize("script,code", [("exit 3\n", 3), ("echo partial; exit 1\n", 1), ("kill -9 $$\n", 137)])
def test_a_checker_that_exits_nonzero_is_a_recorded_failure_with_its_exit_status(tmp_path, stubs, script, code):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(skill_with_checker(tmp_path, script))), tmp_path)
    failed(s, "check_video_exit")
    assert s["check_video_exit"] == code, s["check_video_exit"]
    assert s["check_video"].startswith(f"exited {code}"), s["check_video"]
    if "partial" in script:
        assert "partial" in s["check_video"]                                  # what the checker printed is kept


def test_a_checker_that_exits_zero_is_not_a_failure_and_records_its_exit_status(tmp_path, stubs):
    env, log = stubs
    run = make_run(tmp_path)
    s = collect(run, dict(env, SKILL_SRC=str(skill_with_checker(tmp_path, "echo trusted-check-ok\n"))), tmp_path)
    assert "check_video_exit" not in s["measurement_failures"]["names"], s["measurement_failures"]
    assert s["check_video_exit"] == 0 and "trusted-check-ok" in s["check_video"] and not s["check_video"].startswith("exited"), s["check_video"]
