# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 417, second finding: after a coder retry `tt-model serve` refused with "chip(s) 0, 1 already in use by another container" because
the previous coder container was left as `Exited (1)`, and the block was labelled `hardware-unhealthy` although the chips were free.

(a) The supervisor removes its own coder container after it confirms the exit, whether `tt-model serve` failed (owned unknown) or the
    boot failed later, and a retry whose start is refused for a leftover of this run's own earlier launch removes it and starts again.
(b) A refusal that says a container holds the chips is classed `coder-unusable`, names the leftover container and the command that
    removes it, and is not retried by itself."""
import pytest

from fakes import LEASE, FakeClock, FakeRun
from orchard.commands import CommandResult
from orchard import blocked
from orchard.ledger import Ledger
from orchard.server import ServerChipsHeld, ServerControl, ServerError, ServerSpec, StopCheck
from run_fakes import MachineCoder
from test_supervisor import Rig, never_wait

SPEC = ServerSpec("episod/qwen-coder", "container", 8000, "qwen", profile="default")
NAME = "tt-model-qwen-coder-default"
REFUSAL = "chip(s) 0, 1 already in use by another container"
SS_HEADER = "State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process\n"


def key(argv):
    if argv[:3] == ["docker", "ps", "-q"]:
        return "docker ps-q"
    if argv[:2] == ["docker", "ps"] and "-a" in argv:
        return "ps-a"
    return " ".join(argv[:2])


def control(serve, listing=(0, "", ""), rm=(0, NAME, "")):
    """A real ServerControl over scripted commands: `serve` is a list of answers to `tt-model serve` in order."""
    script = {"tt-model serve": list(serve), "tt-model stop": [(0, "stopped 1 container(s)", "")], "docker ps": [(0, "", "")],
              "docker ps-q": [(0, "", "")], "ss -ltn": [(0, SS_HEADER, "")], "curl -sS": [(7, "", "refused")],
              "ps-a": listing if isinstance(listing, list) else [listing], "docker rm": [rm]}
    run = FakeRun(script, key=key)
    clock = FakeClock()
    return ServerControl(SPEC, run=run, clock=clock, sleep=clock.sleep), run


def rms(run):
    return [a for a in run.argvs() if a[:2] == ["docker", "rm"]]


def serves(run):
    return [a for a in run.argvs() if a[:2] == ["tt-model", "serve"]]


# ---- the server control ---------------------------------------------------------------------------------------------------------


def test_the_in_use_refusal_is_its_own_error_and_marks_the_container_as_not_ours():
    ctl, run = control([(1, "", REFUSAL)])
    with pytest.raises(ServerChipsHeld, match="already in use by another container"):
        ctl.start(LEASE)
    assert isinstance(ServerChipsHeld("x"), ServerError) and ctl.owned is False


@pytest.mark.parametrize("text", ["no such package", "image pull failed", ""])
def test_another_failure_is_not_the_in_use_refusal(text):
    ctl, _ = control([(1, "", text)])
    with pytest.raises(ServerError) as info:
        ctl.start(LEASE)
    assert not isinstance(info.value, ServerChipsHeld) and ctl.owned is None


def test_a_container_after_a_failed_start_with_unknown_ownership_is_removed():
    ctl, run = control([(1, "", "container exited with code 1")], listing=(0, f"{NAME} exited c1\n", ""))
    with pytest.raises(ServerError):
        ctl.start(LEASE)
    assert ctl.owned is None
    assert ctl.remove_container()["removed"] is True and rms(run) == [["docker", "rm", "c1"]]       # by the id it listed


def test_a_known_refusal_blocks_the_removal_unless_the_caller_says_the_container_is_this_runs():
    ctl, run = control([(1, "", REFUSAL)], listing=(0, f"{NAME} exited c1\n", ""))
    with pytest.raises(ServerChipsHeld):
        ctl.start(LEASE)
    assert ctl.remove_container() is None and rms(run) == []
    assert ctl.remove_container(even_if_unowned=True)["removed"] is True and len(rms(run)) == 1


def test_even_if_unowned_still_never_removes_a_running_container_or_another_name():
    ctl, run = control([(1, "", REFUSAL)], listing=(0, f"{NAME} running\n{NAME}-x exited\n", ""))
    with pytest.raises(ServerChipsHeld):
        ctl.start(LEASE)
    assert ctl.remove_container(even_if_unowned=True)["removed"] is False and rms(run) == []


def test_find_container_lists_the_exact_name_with_its_state():
    ctl, _ = control([(0, "", "")], listing=(0, f"{NAME} exited\nother exited\n{NAME}-x running\n", ""))
    assert ctl.find_container() == [(NAME, "exited")]
    ctl, _ = control([(0, "", "")], listing=(1, "", "daemon down"))
    assert ctl.find_container() is None


# ---- the block's code ------------------------------------------------------------------------------------------------------------

LEFT = (f"blocked: the coder did not start and answer; nothing is retried: tt-model serve exited 1: {REFUSAL}. A stopped container named "
        f"{NAME} is left over (exited); remove it with `docker rm {NAME}`, then resume")


def test_a_container_holds_the_chips_block_is_coder_unusable_and_not_hardware_unhealthy():
    assert blocked.block_code({"decision": "pause", "reason": LEFT}) == "coder-unusable"
    assert blocked.block_code({"decision": "pause", "reason": "blocked: " + REFUSAL}) == "coder-unusable"


def test_other_chip_text_stays_a_hardware_block():
    assert blocked.block_code({"decision": "pause", "reason": "blocked: gozer reset failed on board 0"}) == "hardware-unhealthy"
    assert blocked.block_code({"decision": "pause", "reason": "blocked: the lease could not be released: chip hung"}) == "hardware-unhealthy"


def test_a_block_that_names_a_container_is_not_retried_by_itself():
    assert blocked.retryable("coder-unusable", LEFT) is False
    assert blocked.retryable("coder-unusable", "the coder did not start") is True


# ---- the supervisor --------------------------------------------------------------------------------------------------------------


class StatefulRun:
    """Scripted commands over a docker that remembers. `serve` entries are (returncode, stdout, stderr[, rows the call creates]); `rows` are the
    containers (name, state, id) that exist; `docker rm` removes a row by name or by id and fails when none matches; `list_script` gives the
    answers of the first `docker ps -a` calls (a container that changes between two looks), then the live rows."""

    def __init__(self, serve, rows, list_script=None):
        self.serve, self.rows, self.list_script, self.calls = list(serve), [tuple(r) for r in rows], list(list_script or []), []

    def argvs(self):
        return [c["argv"] for c in self.calls]

    def __call__(self, argv, timeout, *, env=None, kill_on_timeout=True):
        argv = [str(a) for a in argv]
        self.calls.append({"argv": argv, "timeout": timeout, "env": env, "kill_on_timeout": kill_on_timeout})

        def out(rc, text="", err=""):
            return CommandResult(tuple(argv), rc, text, err)
        if argv[:2] == ["tt-model", "serve"]:
            entry = self.serve.pop(0) if len(self.serve) > 1 else self.serve[0]
            self.rows += [tuple(r) for r in (entry[3] if len(entry) > 3 else [])]
            return out(*entry[:3])
        if argv[:2] == ["docker", "ps"] and "-a" in argv:
            rows = self.list_script.pop(0) if self.list_script else self.rows
            return out(0, "".join(f"{n} {st} {i}\n" for n, st, i in rows))
        if argv[:3] == ["docker", "ps", "-q"]:
            return out(0)
        if argv[:2] == ["docker", "ps"]:
            return out(0, "".join(f"{i} img {n} \n" for n, st, i in self.rows if st == "running"))
        if argv[:2] == ["docker", "rm"]:
            kept = [r for r in self.rows if argv[2] not in (r[0], r[2])]
            gone, self.rows = len(kept) != len(self.rows), kept
            return out(0 if gone else 1, argv[2], "" if gone else "No such container")
        if argv[:2] == ["tt-model", "stop"]:
            self.rows = [(n, "exited" if st == "running" else st, i) for n, st, i in self.rows]
            return out(0, "stopped 1 container(s)")
        if argv[:2] == ["ss", "-ltn"]:
            return out(0, SS_HEADER)
        if argv[:2] == ["curl", "-sS"]:
            return out(7, "", "refused")
        raise AssertionError(f"unexpected call: {argv}")


EXITED = [(NAME, "exited", "c1")]


class ScriptedCoder(MachineCoder):
    """The fake machine for readiness and chips; a real ServerControl over a stateful docker for the start, the stop checks and the removal."""

    def __init__(self, machine, serve, rows=EXITED, list_script=None):
        super().__init__(machine)
        self.run = StatefulRun(serve, rows, list_script)
        clock = FakeClock()
        self.ctl = ServerControl(SPEC, run=self.run, clock=clock, sleep=clock.sleep)

    owned = property(lambda self: self.ctl.owned, lambda self, v: setattr(self.ctl, "owned", v))

    def start(self, lease):
        self.ctl.start(lease)
        super().start(lease)

    def stop(self):
        out = self.ctl.stop()
        if not out.get("skipped"):
            super().stop()
        return out

    def confirm_stopped(self):
        mine, theirs = self.ctl.confirm_stopped(), super().confirm_stopped()
        return StopCheck(mine.stopped and theirs.stopped, {**mine.checks, **theirs.checks}, {})

    def record(self):
        return self.ctl.record()

    def adopt(self, record):
        return self.ctl.adopt(record)

    def remove_container(self, **kw):
        return self.ctl.remove_container(**kw)

    def find_container(self):
        return self.ctl.find_container()

    def find_container_rows(self):
        return self.ctl.find_container_rows()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def blocks(rig):
    return [e["data"].get("reason", "") for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]


def launch(rig, coder):
    made = []
    rig.coder_cls = lambda machine: made.append(coder(machine)) or made[0]
    rig.args.unattended = True
    never_wait(rig)
    return made


def earlier_launch(rig, created="c1"):
    """The ledger of a run that already launched a coder once (the first attempt of a retry). `created` is the id of the container that
    launch made, recorded as the supervisor records it; None is a launch that was refused and made none."""
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("decision", None, decision="coder starting", server={"kind": "container", "target": SPEC.target, "owned": None})
        if created:
            led.append("decision", None, decision="coder container created", container_id=created, name=NAME)


def test_a_leftover_of_this_runs_own_launch_is_removed_and_the_start_is_tried_again(rig):
    earlier_launch(rig)
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), (0, "", "", [(NAME, "running", "c2")])]))
    rig.run()
    c = made[0]
    order = [a[1] for a in c.run.argvs() if a[:2] in (["tt-model", "serve"], ["docker", "rm"])]
    assert rms(c.run)[0] == ["docker", "rm", "c1"] and order[:3] == ["serve", "rm", "serve"]     # refused, removed by id, started again
    removed = [d for d in rig.decisions() if d["decision"] == "coder container removed"]
    assert removed and removed[0]["result"]["removed"] is True and removed[0].get("before_start") is True
    assert "coder container started" in [d["decision"] for d in rig.decisions()]


def test_a_leftover_from_before_this_run_is_not_removed_and_the_block_names_it(rig):
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)]))
    assert rig.run() == 5
    c = made[0]
    assert rms(c.run) == [] and len(serves(c.run)) == 1
    reason = blocks(rig)[-1]
    assert NAME in reason and f"docker rm {NAME}" in reason and "in use by another container" in reason
    assert blocked.block_code({"decision": "pause", "reason": "blocked: " + reason}) == "coder-unusable"


def test_the_retry_gives_up_when_the_second_start_is_refused_too(rig):
    earlier_launch(rig)
    # the second refusal: another stopped container (c2) now holds the chips
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), (1, "", REFUSAL, [(NAME, "exited", "c2")])]))
    assert rig.run() == 5
    assert len(serves(made[0].run)) == 2 and len(rms(made[0].run)) == 1
    reason = blocks(rig)[-1]
    assert "was removed once and the start was refused again" in reason and f"docker rm {NAME}" in reason
    assert blocked.block_code({"decision": "pause", "reason": "blocked: " + reason}) == "coder-unusable"


def test_a_refusal_with_no_stopped_container_of_that_name_says_to_list_containers(rig):
    earlier_launch(rig)
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)], rows=[]))
    assert rig.run() == 5
    reason = blocks(rig)[-1]
    assert "docker ps -a" in reason and rms(made[0].run) == []


def test_a_running_container_of_that_name_is_never_removed_by_the_retry(rig):
    earlier_launch(rig)
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)], rows=[(NAME, "running", "c1")]))
    assert rig.run() == 5
    assert rms(made[0].run) == [] and len(serves(made[0].run)) == 1


def test_a_start_that_fails_after_creating_its_container_leaves_none_behind(rig):
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", "container exited with code 1", [(NAME, "exited", "c7")])], rows=[]))
    assert rig.run() == 5
    assert rms(made[0].run) == [["docker", "rm", "c7"]]
    assert any(d["decision"] == "coder container removed" for d in rig.decisions())


def test_a_foreign_refusal_still_removes_nothing(rig):
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", f"{NAME} is already running")]))
    assert rig.run() == 5
    assert rms(made[0].run) == []
