# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/hub_can_check.py: the parts that run without a board.

The script serves a published can from the Hub under a lease. These tests cover what a mistake would hide:
the isolation of every cache under one directory, the stop that targets only the process group the script
started, the port table against the cards, and the audio check on a real file."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hub_can_check.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("hub_can_check", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_cache_variable_points_inside_the_root(mod, tmp_path):
    env = mod.env_for(tmp_path)
    for name in ("XDG_CACHE_HOME", "TT_MODEL_MODELS_DIR", "HF_HOME", "TT_METAL_CACHE"):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), name


def test_the_environment_keeps_the_callers_token_without_printing_it(mod, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_for_the_test")
    env = mod.env_for(tmp_path)
    assert env["HF_TOKEN"] == "hf_secret_for_the_test"      # the server needs it
    # the script never writes the environment: its printable result holds no env and no token
    text = SCRIPT.read_text()
    assert "print(env" not in text and "environ[\"HF_TOKEN\"]" not in text


def test_each_can_has_the_port_its_card_gives(mod):
    assert {k: v["port"] for k, v in mod.CANS.items()} == {
        "granite-4.0-h-1b-blackhole": 20110, "minicpm3-4b-blackhole": 20130, "ace-step-1.5-turbo-blackhole": 20140,
        "stable-audio-3-small-music-blackhole": 20170, "minimax-music3-fiona-crapple-blackhole": 20150,
        "qwen3.5-35b-a3b-blackhole-p300": 20160}


def test_stop_group_kills_the_group_the_script_started_and_nothing_else(mod, tmp_path):
    # a shell that starts a child which would outlive it; the group stop must take both down
    marker = tmp_path / "child.pid"
    proc = subprocess.Popen(["bash", "-c", f"sleep 300 & echo $! > {marker}; wait"], start_new_session=True)
    for _ in range(100):
        if marker.exists() and marker.read_text().strip():
            break
        subprocess.run(["sleep", "0.05"])
    child = int(marker.read_text())
    bystander = subprocess.Popen(["sleep", "300"])          # another process of ours, outside the group
    try:
        assert mod.stop_group(proc, wait_s=10) == "stopped on SIGTERM"
        gone = False
        for _ in range(100):                                 # init collects the orphaned child shortly after the group dies
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                gone = True
                break
            subprocess.run(["sleep", "0.05"])
        assert gone, "the child of the started process survived the group stop"
        assert bystander.poll() is None                      # untouched
    finally:
        bystander.send_signal(signal.SIGKILL)
        bystander.wait()


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0, reason="needs ffmpeg and ffprobe")
def test_audio_facts_reads_duration_rate_and_silence_from_a_real_wav(mod, tmp_path):
    np = pytest.importorskip("numpy")
    rate = 8000
    t = np.arange(rate) / rate
    loud = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    silence = np.zeros(rate, dtype="float32")
    x = np.concatenate([loud, silence])
    pcm = (x * 32767).astype("<i2")
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    facts = mod.audio_facts(path)
    assert facts["sample_rate"] == rate and facts["channels"] == 1
    assert facts["duration_s"] == pytest.approx(2.0, abs=0.01)
    assert facts["windows_above_-40dBFS"] == pytest.approx(0.5, abs=0.06)     # one loud second, one silent second
    assert facts["peak"] == pytest.approx(0.5, abs=0.01)


def test_the_script_refuses_an_unknown_can_name(tmp_path):
    cp = subprocess.run([sys.executable, str(SCRIPT), "not-a-can", "--root", str(tmp_path), "--print-env"],
                        capture_output=True, text=True)
    assert cp.returncode == 2 and "unknown can" in cp.stderr and "Traceback" not in cp.stderr


# ---- review round 1 (PR 411): hangs, a held board, false success, writes outside the root ------------------------------------------

def test_inherited_cache_variables_are_overridden_inside_the_root(mod, tmp_path, monkeypatch):
    for name in ("HF_HUB_CACHE", "TT_DIT_CACHE_DIR", "TT_CACHE_PATH", "TT_CACHE_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(name, "/var/tmp/elsewhere")
    env = mod.env_for(tmp_path)
    for name in ("HF_HUB_CACHE", "TT_DIT_CACHE_DIR", "TT_CACHE_PATH", "TT_CACHE_HOME"):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), name
    assert "HUGGINGFACE_HUB_CACHE" not in env and "TRANSFORMERS_CACHE" not in env


def test_a_relative_root_becomes_absolute(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = mod.env_for(Path("rel-root"))
    assert all(os.path.isabs(env[n]) for n in ("XDG_CACHE_HOME", "TT_MODEL_MODELS_DIR", "HF_HOME", "TT_METAL_CACHE", "HF_HUB_CACHE"))


def test_port_busy_sees_a_listener_and_a_free_port(mod):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert mod.port_busy(port) is True
    finally:
        s.close()
    assert mod.port_busy(port) is False


def test_poll_job_gives_up_at_its_deadline(mod):
    now = [0.0]
    calls = []

    def get(timeout):
        calls.append(timeout)
        return {"status": "running"}

    def sleep(s):
        now[0] += s
        assert len(calls) < 100, "the poll loop never reached its deadline"
    with pytest.raises(TimeoutError):
        mod.poll_job(get, ("done",), deadline_s=10, interval=3, clock=lambda: now[0], sleep=sleep)
    assert max(calls) <= 10 and now[0] <= 13           # each request is capped by the time left


def test_poll_job_returns_a_terminal_state(mod):
    states = iter([{"status": "queued"}, {"status": "running"}, {"status": "error"}])
    st = mod.poll_job(lambda timeout: next(states), ("done", "error"), deadline_s=100, interval=0, clock=lambda: 0.0, sleep=lambda s: None)
    assert st["status"] == "error"


def test_a_failed_job_makes_the_run_fail(mod):
    ok = {"ready": True, "requests": [{"status": "done"}, {"status": "completed"}]}
    assert mod.run_ok(ok) is True
    for bad in ("error", "failed", "cancelled", None):
        assert mod.run_ok({"ready": True, "requests": [{"status": "done"}, {"status": bad}]}) is False
    assert mod.run_ok({"ready": True, "requests": [{"latency_s": 1.0}]}) is True      # a synchronous request has no job status
    assert mod.run_ok({"ready": False}) is False
    assert mod.run_ok({"ready": True, "error": "x"}) is False


def test_stop_group_kills_a_child_that_ignores_sigterm_after_the_launcher_exits(mod, tmp_path):
    marker = tmp_path / "child.pid"
    # the launcher exits at once; its child ignores SIGTERM and stays in the group
    # the child ignores SIGTERM before it writes its pid, so the test never signals it too early (a race under load); the shell
    # that starts it exits at once, leaving the child alone in the group
    child_code = (f"import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  f"open({str(marker)!r}, 'w').write(str(__import__('os').getpid())); time.sleep(300)")
    proc = subprocess.Popen(["bash", "-c", f"{sys.executable} -c \"$CHILD\" &"], start_new_session=True,
                            env={**os.environ, "CHILD": child_code})
    for _ in range(100):
        if marker.exists() and marker.read_text().strip():
            break
        subprocess.run(["sleep", "0.05"])
    child = int(marker.read_text())
    proc.wait(10)
    verdict = mod.stop_group(proc, wait_s=1)
    assert "SIGKILL" in verdict
    gone = False
    for _ in range(100):
        try:
            os.kill(child, 0)
            with open(f"/proc/{child}/stat") as fh:
                if fh.read().split()[2] == "Z":
                    gone = True
                    break
        except (ProcessLookupError, FileNotFoundError):
            gone = True
            break
        subprocess.run(["sleep", "0.05"])
    assert gone


def test_the_script_refuses_to_start_when_the_cards_port_is_taken(mod, tmp_path):
    import socket
    port = mod.CANS["granite-4.0-h-1b-blackhole"]["port"]
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        pytest.skip(f"port {port} is in use on this machine")
    s.listen(1)
    # a stand-in `tt-model` first on PATH: if the guard fails, the script starts this instead of a real server (which would open a board)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "tt-model").write_text(f"#!/bin/sh\necho launched > {tmp_path}/launched\nexit 0\n")
    (fakebin / "tt-model").chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
    try:
        cp = subprocess.run([sys.executable, str(SCRIPT), "granite-4.0-h-1b-blackhole", "--root", str(tmp_path / "root")],
                            capture_output=True, text=True, timeout=60, env=env)
    finally:
        s.close()
    assert not (tmp_path / "launched").exists(), "a server was launched on an occupied port"
    assert cp.returncode == 1 and "already in use" in cp.stdout


# ---- review round 2 (PR 411): signals must run the cleanup; --print-env must match env_for -----------------------------------------

@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_to_the_driver_stops_the_server_it_started(mod, tmp_path, sig):
    import socket
    port = mod.CANS["granite-4.0-h-1b-blackhole"]["port"]
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        pytest.skip(f"port {port} is in use on this machine")
    probe.close()
    # a stand-in `tt-model` that records its pid and stays up like a server; a real one must never run here
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    pidfile = tmp_path / "server.pid"
    (fakebin / "tt-model").write_text(f"#!/bin/sh\necho $$ > {pidfile}\nexec sleep 300\n")
    (fakebin / "tt-model").chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
    driver = subprocess.Popen([sys.executable, str(SCRIPT), "granite-4.0-h-1b-blackhole", "--root", str(tmp_path / "root"),
                               "--ready-timeout", "600"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(200):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            subprocess.run(["sleep", "0.05"])
        server = int(pidfile.read_text())
        subprocess.run(["sleep", "0.3"])                    # the driver is in its polling loop
        driver.send_signal(sig)
        driver.communicate(timeout=120)
        assert driver.returncode == 128 + sig
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()
    gone = False
    for _ in range(100):
        try:
            os.kill(server, 0)
            with open(f"/proc/{server}/stat") as fh:
                if fh.read().split()[2] == "Z":
                    gone = True
                    break
        except (ProcessLookupError, FileNotFoundError):
            gone = True
            break
        subprocess.run(["sleep", "0.05"])
    if not gone:
        os.kill(server, signal.SIGKILL)
    assert gone, "the server outlived the driver that a signal ended"
    record = json.loads((tmp_path / "root" / "granite-4.0-h-1b-blackhole" / "result.json").read_text())
    assert record["stop"].startswith(("stopped", "needed")) and record["terminated_by"] == sig.name


def test_print_env_shows_every_change_env_for_makes(mod, tmp_path):
    base = {**os.environ, "HUGGINGFACE_HUB_CACHE": "/var/tmp/x", "TRANSFORMERS_CACHE": "/var/tmp/y", "HF_HUB_CACHE": "/var/tmp/z"}
    cp = subprocess.run([sys.executable, str(SCRIPT), "granite-4.0-h-1b-blackhole", "--root", str(tmp_path), "--print-env"],
                        capture_output=True, text=True, env=base)
    assert cp.returncode == 0
    exports = dict(line[len("export "):].split("=", 1) for line in cp.stdout.splitlines() if line.startswith("export "))
    unsets = {line.split()[1] for line in cp.stdout.splitlines() if line.startswith("unset ")}
    # apply the printed lines to the caller's environment and compare with what the driver passes to tt-model
    applied = {k: v for k, v in base.items() if k not in unsets}
    applied.update(exports)
    want = mod.env_for(tmp_path / "granite-4.0-h-1b-blackhole")
    # the shell lines apply to the same variables, so a variable the printed lines forget differs here
    assert {k: v for k, v in want.items() if applied.get(k) != v} == {}
    assert {k for k in base if k not in want} == unsets


# ---- review round 3 (PR 411): server cache variables, shell quoting, a stream that ends badly -----------------------------------

def test_the_servers_own_cache_variables_are_set_under_the_root(mod, tmp_path, monkeypatch):
    for name in ("MINICPM3_CACHE", "ACESTEP_WORK", "AUDIO_PKG_CACHE"):
        monkeypatch.setenv(name, "/var/tmp/elsewhere")
    env = mod.env_for(tmp_path)
    for name in ("MINICPM3_CACHE", "ACESTEP_WORK", "AUDIO_PKG_CACHE"):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), name


def test_print_env_lines_survive_a_shell_with_a_space_and_a_dollar_in_the_root(mod, tmp_path):
    root = tmp_path / "a b$HOME`x`'q"
    base = {**os.environ, "HUGGINGFACE_HUB_CACHE": "/var/tmp/x", "TRANSFORMERS_CACHE": "/var/tmp/y", "MINICPM3_CACHE": "/var/tmp/z"}
    cp = subprocess.run([sys.executable, str(SCRIPT), "granite-4.0-h-1b-blackhole", "--root", str(root), "--print-env"],
                        capture_output=True, text=True, env=base)
    assert cp.returncode == 0
    # run the printed lines in a real shell and read back every variable env_for sets
    want = mod.env_for(root / "granite-4.0-h-1b-blackhole")
    names = sorted(set(mod.env_changes(root / "granite-4.0-h-1b-blackhole")[0]) | set(mod.ENV_UNSET))
    probe = cp.stdout + "\n" + "\n".join(f'printf "%s\\0" "${{{n}-__unset__}}"' for n in names)
    out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, env=base).stdout
    got = dict(zip(names, out.split("\0")[-len(names) - 1:-1]))
    for n in names:
        assert got[n] == want.get(n, "__unset__"), n


class _Stream:
    """A fake chat server that streams a fixed body, so the parser meets the endings a real server can produce."""
    def __init__(self, lines):
        import http.server
        import threading
        outer = lines

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for ln in outer:
                    self.wfile.write(ln.encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.01)

            def log_message(self, *a):
                pass
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _chunk(text):
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})


GOOD = [_chunk("one "), _chunk("two "), _chunk("three"), "data: " + json.dumps({"usage": {"completion_tokens": 3}}), "data: [DONE]"]


@pytest.mark.parametrize("name,lines,problem", [
    ("error event", [_chunk("one "), "data: " + json.dumps({"error": {"message": "boom"}}), "data: [DONE]"], "error"),
    ("eof before DONE", [_chunk("one "), _chunk("two ")], "DONE"),
    ("empty answer", ["data: " + json.dumps({"choices": [{"delta": {}}]}), "data: [DONE]"], "empty"),
])
def test_a_stream_that_ends_badly_is_rejected(mod, name, lines, problem):
    s = _Stream(lines)
    try:
        r = mod.chat_stream(s.base, None, {}, 8)
    finally:
        s.close()
    assert r["stream_ok"] is False and problem in r["stream_problem"], r


def test_a_complete_stream_is_accepted(mod):
    s = _Stream(GOOD)
    try:
        r = mod.chat_stream(s.base, None, {}, 8)
    finally:
        s.close()
    assert r["stream_ok"] is True and r["stream_problem"] is None and r["text"] == "one two three"


def test_rejected_streams_stay_out_of_the_median_and_fail_the_run(mod):
    runs = [{"stream_ok": True, "decode_tps": 30.0}, {"stream_ok": False, "decode_tps": 999.0}, {"stream_ok": True, "decode_tps": 32.0}]
    assert mod.median_tps(runs) == 32.0                      # of [30, 32], the upper middle value; the 999 is excluded
    assert mod.median_tps([{"stream_ok": False, "decode_tps": 5.0}]) is None
    assert mod.run_ok({"ready": True, "stream_runs": runs}) is False
    assert mod.run_ok({"ready": True, "stream_runs": [runs[0], runs[2]]}) is True
    assert mod.run_ok({"ready": True, "stream_runs": []}) is False
