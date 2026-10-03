"""ServerControl with fake commands: start, stop, the stop checks, readiness."""
import signal

import pytest

from fakes import LEASE, FakeClock, FakeProc, FakeRun
from orchard.server import (NotReady, ServerControl, ServerError, ServerSpec, ServerStandIn,
                            ServerStarting)

SS_HEADER = "State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process\n"
SS_LISTEN = SS_HEADER + "LISTEN 0      4096   127.0.0.1:20000      0.0.0.0:*\n"
CLEAR = {"docker ps": [(0, "", "")], "docker ps-q": [(0, "", "")], "ss -ltn": [(0, SS_HEADER, "")],
         "curl -sS": [(7, "", "curl: (7) Failed to connect to 127.0.0.1 port 20000")],
         "ps -p": [(1, "", "")], "pgrep -g": [(1, "", "")],
         "tt-model stop": [(0, "stopped 1 container(s)", "")], "tt-model serve": [(0, "", "")]}
# A real `docker ps` line from the H5 check (2026-10-02), with the container name tt-model gives it.
H5_LINE = "3c1d2e4f5a6b f0ed6056d85f tt-model-audio8-asr-infinite-p150-default 0.0.0.0:20000->20000/tcp\n"
CONTAINER = ServerSpec("episod/audio8-asr-infinite-p150", "container", 20000, "audio8",
                       image_id="f0ed6056d85f")
BUNDLE = ServerSpec("org/qwen-bundle", "bundle", 20000, "qwen")


def key(argv):
    if argv[:3] == ["docker", "ps", "-q"]:
        return "docker ps-q"
    return " ".join(argv[:2])


def control(spec=CONTAINER, **over):
    run = FakeRun({**CLEAR, **over}, key=key)
    clock = FakeClock()
    spawned, killed = [], []
    proc = FakeProc()

    def spawn(argv, env, log_path=None):
        spawned.append((argv, env))
        return proc

    def killpg(pgid, sig):
        killed.append((pgid, sig))
        if sig == signal.SIGTERM and getattr(proc, "obey_term", True):
            proc.returncode = 0

    ctl = ServerControl(spec, run=run, spawn=spawn, killpg=killpg, clock=clock, sleep=clock.sleep)
    return ctl, run, spawned, killed, proc, clock


def test_spec_names_the_container_as_tt_model_does():
    assert CONTAINER.container_name == "tt-model-audio8-asr-infinite-p150-default"
    assert CONTAINER.endpoint == "http://127.0.0.1:20000"
    with pytest.raises(ValueError):
        ServerSpec("x", "vm", 1, "m")
    with pytest.raises(ValueError):
        ServerSpec("x", "process", 1, "m")      # a process server needs argv


def test_container_start_pins_the_leased_chips():
    ctl, run, *_ = control()
    ctl.start(LEASE)
    argv = run.argvs()[0]
    assert argv[:3] == ["tt-model", "serve", "episod/audio8-asr-infinite-p150"]
    assert argv[argv.index("--device-id") + 1] == "0,1"
    assert argv[argv.index("--port") + 1] == "20000" and "--detach" in argv
    assert run.calls[0]["env"]["TT_VISIBLE_DEVICES"] == "0000:01:00.0,0000:02:00.0"
    assert ctl.dev_indices == (0, 1)


def test_container_start_failure_raises():
    ctl, *_ = control(**{"tt-model serve": [(1, "", "no such package")]})
    with pytest.raises(ServerError, match="no such package"):
        ctl.start(LEASE)


def test_container_start_has_its_own_budget_and_is_never_killed():
    ctl, run, *_ = control()
    ctl.start(LEASE)
    assert run.calls[0]["timeout"] == ctl.start_timeout > ctl.timeout
    assert run.calls[0]["kill_on_timeout"] is False


def test_a_start_that_times_out_while_the_server_comes_up_says_so():
    ctl, *_ = control(**{"tt-model serve": [("timeout", "", "")], "docker ps": [(0, H5_LINE, "")]})
    with pytest.raises(ServerStarting, match="not starting another"):
        ctl.start(LEASE)


def test_a_start_that_times_out_with_nothing_up_is_a_plain_failure():
    ctl, *_ = control(**{"tt-model serve": [("timeout", "", "")]})
    with pytest.raises(ServerError, match="nothing came up") as exc:
        ctl.start(LEASE)
    assert not isinstance(exc.value, ServerStarting)


def test_bundle_start_spawns_in_its_own_session():
    ctl, _, spawned, *_ = control(BUNDLE)
    ctl.start(LEASE)
    argv, env = spawned[0]
    assert argv[:3] == ["tt-model", "serve", "org/qwen-bundle"]
    assert env["TT_VISIBLE_DEVICES"] == "0000:01:00.0,0000:02:00.0"
    assert ctl.pid == ctl.pgid == 4321


def test_container_stop_runs_tt_model_stop_with_its_profile():
    ctl, run, *_ = control()
    assert ctl.stop()["mesh_reset"] is False
    assert run.argvs()[0] == ["tt-model", "stop", "episod/audio8-asr-infinite-p150",
                              "--profile", "default"]


def test_a_stop_that_reset_the_mesh_itself_is_flagged():
    out = "grace period expired; docker sent SIGKILL\nreset the mesh with tt-smi (throwaway container)"
    ctl, *_ = control(**{"tt-model stop": [(0, out, "")]})
    assert ctl.stop()["mesh_reset"] is True


def test_container_is_stopped_when_docker_and_the_port_are_clear():
    ctl, *_ = control()
    ctl.adopt({"dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert check.stopped and set(check.checks) == {"docker_ps", "docker_devices", "port_closed",
                                                   "health_refused"}


@pytest.mark.parametrize("line", [
    H5_LINE,                                                            # by name
    "aaa f0ed6056d85f some-other-name \n",                              # by image id
    "bbb other:latest renamed 0.0.0.0:20000->20000/tcp\n",              # by published port
])
def test_a_container_still_listed_by_docker_is_not_stopped(line):
    ctl, *_ = control(**{"docker ps": [(0, line, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert not check.stopped and check.checks["docker_ps"] is False


@pytest.mark.parametrize("devices,stopped", [
    ("abc123 /other /dev/tenstorrent/0 \n", False),
    ("abc123 /other /dev/tenstorrent/3 \n", True),       # a chip this lease does not hold
])
def test_a_container_that_maps_our_device_is_not_stopped(devices, stopped):
    ctl, *_ = control(**{"docker ps-q": [(0, "abc\n", "")], "docker inspect": [(0, devices, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().stopped is stopped


def test_another_agents_container_mapping_every_device_is_named_and_does_not_block():
    ctl, *_ = control(**{"docker ps-q": [(0, "abc\n", "")],
                         "docker inspect": [(0, "abc123 /tti-server /dev/tenstorrent \n", "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert check.stopped
    assert check.evidence["docker_inspect"]["others_with_all_devices"] == ["tti-server"]


def test_our_own_container_mapping_every_device_blocks():
    ctl, *_ = control(**{"docker ps": [(0, H5_LINE, "")], "docker ps-q": [(0, "3c1d2e4f5a6b\n", "")],
                         "docker inspect": [(0, "3c1d2e4f5a6b99 /tt-model-audio8-asr-infinite-p150-default "
                                               "/dev/tenstorrent \n", "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().checks["docker_devices"] is False


def test_unknown_chips_make_any_mapped_device_count():
    ctl, *_ = control(**{"docker ps-q": [(0, "abc\n", "")],
                         "docker inspect": [(0, "abc123 /other /dev/tenstorrent/3 \n", "")]})
    assert ctl.dev_indices is None
    assert not ctl.confirm_stopped().stopped


def test_a_listening_port_is_not_stopped():
    ctl, *_ = control(**{"ss -ltn": [(0, SS_LISTEN, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert not check.stopped and check.checks["port_closed"] is False


@pytest.mark.parametrize("rc", [0, 28])
def test_only_a_refused_connection_confirms_the_port_is_closed(rc):
    ctl, *_ = control(**{"curl -sS": [(rc, "", "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().checks["health_refused"] is False


def test_bundle_worker_left_in_the_group_is_not_stopped():
    # The parent is gone; a vLLM worker it started still runs in its process group.
    ctl, *_ = control(BUNDLE, **{"pgrep -g": [(0, "4400\n", "")]})
    ctl.adopt({"pid": 4321, "pgid": 4321, "dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert not check.stopped
    assert check.checks["process"] is True and check.checks["process_group"] is False


def test_a_child_server_is_reaped_before_ps_looks_for_it():
    ctl, run, _, _, proc, _ = control(BUNDLE)
    ctl.start(LEASE)
    proc.returncode = 0
    polls_at_ps = []
    inner = ctl.run

    def watching(argv, timeout, **kw):
        if argv[:2] == ["ps", "-p"]:
            polls_at_ps.append(proc.polls)
        return inner(argv, timeout, **kw)

    ctl.run = watching
    ctl.confirm_stopped()
    assert polls_at_ps and polls_at_ps[0] >= 1    # an unreaped child shows in ps as a zombie


def test_a_server_with_no_recorded_pid_is_never_confirmed_stopped():
    ctl, *_ = control(BUNDLE)
    check = ctl.confirm_stopped()
    assert not check.stopped and check.checks["process"] is False


def test_process_stop_sends_sigterm_to_the_group():
    spec = ServerSpec("t/fake", "process", 20990, "fake", argv=("python3", "fake.py"))
    ctl, _, _, killed, _, _ = control(spec)
    ctl.start(None)
    assert ctl.stop()["how"] == "SIGTERM"
    assert killed == [(4321, signal.SIGTERM)]


def test_process_stop_escalates_to_sigkill():
    spec = ServerSpec("t/fake", "process", 20990, "fake", argv=("python3", "fake.py"))
    ctl, _, _, killed, proc, _ = control(spec)
    proc.obey_term = False
    ctl.start(None)
    assert ctl.stop()["how"] == "SIGKILL"
    assert killed == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def test_the_standin_spawns_first_and_waits_for_readiness_separately():
    # The handoff records the pid between the two calls, so spawn must not wait.
    ctl, run, spawned, *_ = control(ServerSpec("p/standin", "process", 20000, "m", argv=("serve",)))
    standin = ServerStandIn(ctl, ready_budget_s=5)
    standin.spawn()
    assert len(spawned) == 1 and run.calls == [] and standin.record()["pid"] == 4321
    run.script["curl -sS"] = [(0, "200", "")]
    standin.wait_ready()
    assert run.argvs()[0][:2] == ["curl", "-sS"]


def test_wait_ready_returns_after_health_answers_200():
    ctl, _, _, _, _, clock = control(**{"curl -sS": [(7, "000", ""), (0, "200", "")]})
    assert ctl.wait_ready(60) == 5.0
    assert clock.sleeps == [5.0]


def test_wait_ready_gives_up_at_the_budget():
    ctl, *_ = control(**{"curl -sS": [(7, "000", "")]})
    with pytest.raises(NotReady) as exc:
        ctl.wait_ready(20)
    assert exc.value.waited_s >= 20


def test_wait_ready_stops_when_the_server_exits():
    ctl, _, _, _, proc, _ = control(BUNDLE, **{"curl -sS": [(7, "000", "")]})
    ctl.start(LEASE)
    proc.returncode = 1
    with pytest.raises(ServerError, match="exited"):
        ctl.wait_ready(600)
