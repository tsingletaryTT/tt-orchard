# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The release watcher (orchard/drop_watch.py).

One `tick` sees whether the release exists and has finished uploading, runs the release-day check, waits until gozer shows every
chip free, starts `tt-orchard bringup` once, and follows the run to ready or blocked. The Hub, gozer, the launcher, the run status
and the clock are injected, so no test touches the network, the chips or a real process."""
import json
import os
from pathlib import Path

import pytest

from orchard import drop_check as dc
from orchard import drop_watch as dw

FIX = Path(__file__).parent / "fixtures" / "model-configs"
SHA = "0123456789abcdef0123456789abcdef01234567"
SHA2 = "fedcba9876543210fedcba9876543210fedcba98"
T0 = 1_800_000_000.0        # a fixed clock origin
HOUR = 3600.0


def cfg_bytes(name="qwen4-dense-provisional-buildable.json"):
    c = json.loads((FIX / name).read_text())
    c.pop("_source", None)
    return json.dumps(c).encode()


INDEX = json.dumps({"weight_map": {"a.weight": "model-00001-of-00002.safetensors",
                                   "b.weight": "model-00002-of-00002.safetensors"}}).encode()


class FakeHub:
    """The release as the Hub shows it. `modified` is the last commit time (epoch seconds)."""

    def __init__(self, exists=True, sha=SHA, modified=T0, config=None, shards=("model-00001-of-00002.safetensors",
                                                                              "model-00002-of-00002.safetensors"),
                 others=()):
        self.exists, self.sha, self.modified, self.shards, self.others = exists, sha, modified, list(shards), list(others)
        self.files = {"config.json": config or cfg_bytes(), "model.safetensors.index.json": INDEX}

    def info(self, repo, revision=None):
        if not self.exists:
            return None
        sib = [{"rfilename": n, "size": len(b)} for n, b in self.files.items()]
        sib += [{"rfilename": s, "size": 1000} for s in self.shards]
        return {"id": repo, "sha": self.sha, "lastModified": dw.iso(self.modified), "siblings": sib}

    def get(self, repo, revision, name, *, max_bytes):
        return self.files.get(name)

    def list_models(self, org, search):
        return [f"{org}/{search}"] * self.exists + self.others


class World:
    """Fakes for everything outside the watcher, and a clock."""

    def __init__(self, tmp_path, hub=None, free=True, caps=None):
        self.hub = hub or FakeHub()
        self.free, self.holders = free, ["claude:qwen38-serve"]
        self.now = T0 + HOUR
        self.launched, self.status, self.alive = [], {"state": "running"}, True
        self.caps_path = None
        if caps is not None:
            self.caps_path = tmp_path / "caps.json"
            self.caps_path.write_text(json.dumps(caps))
        self.watch_dir = tmp_path / "watch"
        dw.write_config(self.watch_dir, dw.Config(target="Qwen/Qwen4-27B", run_dir=str(tmp_path / "run"),
                                                  capabilities=str(self.caps_path) if self.caps_path else None))

    def chips(self):
        return {"free": self.free, "holders": [] if self.free else self.holders}

    def launch(self, argv, log_path):
        self.launched.append(argv)
        return {"pid": 4242, "start_time": "777"}

    def run_status(self, run_dir):
        return dict(self.status)

    def tick(self, **kw):
        return dw.tick(self.watch_dir, hub=self.hub, chips=self.chips, launch=self.launch, run_status=self.run_status,
                       alive=lambda pid, st: self.alive, now=lambda: self.now, **kw)


def kinds(res):
    return [e["kind"] for e in res["events"]]


def test_no_release_keeps_waiting_and_starts_nothing(tmp_path):
    w = World(tmp_path, hub=FakeHub(exists=False))
    for _ in range(3):
        res = w.tick()
    assert res["state"] == "waiting" and w.launched == [] and res["events"] == []


def test_an_upload_younger_than_the_quiet_time_is_seen_not_started(tmp_path):
    w = World(tmp_path, hub=FakeHub(modified=T0 + HOUR - 60))
    res = w.tick()
    assert res["state"] == "seen" and "release-seen" in kinds(res) and w.launched == []
    assert w.tick()["events"] == []                                # seen once, reported once


def test_a_missing_shard_is_an_unfinished_upload(tmp_path):
    hub = FakeHub(shards=["model-00001-of-00002.safetensors"])
    w = World(tmp_path, hub=hub)
    res = w.tick()
    assert res["state"] == "seen" and w.launched == []
    hub.shards.append("model-00002-of-00002.safetensors")
    assert w.tick()["state"] != "seen"


def test_a_new_commit_restarts_the_quiet_time(tmp_path):
    hub = FakeHub(modified=T0 + HOUR - 60)
    w = World(tmp_path, hub=hub)
    w.tick()
    hub.sha, hub.modified = SHA2, T0 + HOUR + 1000
    w.now = T0 + HOUR + 1100                                       # older than the first commit's quiet time, not the second's
    assert w.tick()["state"] == "seen" and w.launched == []


def test_not_covered_goes_to_the_full_port_fallback_once_chips_are_free(tmp_path):
    w = World(tmp_path, free=False)                                # no capabilities file yet: not covered
    res = w.tick()
    assert res["state"] == "waiting-for-chips" and w.launched == []
    assert {"check", "waiting-for-chips"} <= set(kinds(res))
    assert w.tick()["events"] == []                                # still busy: no repeat notification
    w.free = True
    res = w.tick()
    assert res["state"] == "running" and "run-started" in kinds(res)
    argv = w.launched[0]
    assert argv[:3] == ["tt-orchard", "bringup", "Qwen/Qwen4-27B"] and "--full-port" in argv
    assert argv[argv.index("--run-dir") + 1] == str(tmp_path / "run")


def test_covered_release_takes_the_package_route(tmp_path):
    w = World(tmp_path, caps=dc.EXAMPLE_CAPABILITIES)
    w.tick()
    argv = w.launched[0]
    assert "--full-port" not in argv and argv[argv.index("--base") + 1] == dc.QWEN4_BUNDLE


def test_a_second_tick_never_starts_a_second_run(tmp_path):
    w = World(tmp_path)
    for _ in range(5):
        w.tick()
    assert len(w.launched) == 1


def test_run_end_states_are_reported_once(tmp_path):
    w = World(tmp_path)
    w.tick()
    w.status = {"state": "ready-for-operator-review"}
    res = w.tick()
    assert res["state"] == "ready" and "run-ready" in kinds(res)
    assert w.tick()["events"] == [] and len(w.launched) == 1


def test_a_blocked_run_reports_its_reason(tmp_path):
    w = World(tmp_path)
    w.tick()
    w.status = {"state": "blocked", "blocked": {"code": "stage-failed", "reason": "stage 2 failed twice"}}
    res = w.tick()
    assert res["state"] == "blocked"
    ev = [e for e in res["events"] if e["kind"] == "run-blocked"][0]
    assert "stage-failed" in ev["text"] and "stage 2 failed twice" in ev["text"]


def test_a_dead_bringup_without_a_ledger_is_failed(tmp_path):
    w = World(tmp_path)
    w.tick()
    w.alive, w.status = False, {"state": "not-started"}
    res = w.tick()
    assert res["state"] == "failed" and "run-failed" in kinds(res)


def test_stage_results_are_reported_as_they_land(tmp_path):
    w = World(tmp_path)
    w.tick()
    w.status = {"state": "running", "stages": [{"stage": 0, "name": "delta triage", "status": "pass", "attempts": 1},
                                               {"stage": 1, "name": "reference", "status": "running", "attempts": 1}]}
    res = w.tick()
    assert [e["text"] for e in res["events"] if e["kind"] == "stage"] == ["stage 0 (delta triage): pass"]
    w.status["stages"][1]["status"] = "fail"
    assert [e["text"] for e in w.tick()["events"]] == ["stage 1 (reference): fail"]
    w.status["stages"][1].update(status="pass", attempts=2)                # a second attempt is reported too
    assert [e["text"] for e in w.tick()["events"]] == ["stage 1 (reference): pass"]


def test_needs_operator_waits_then_continues(tmp_path):
    cfg = json.loads(cfg_bytes())
    cfg["model_type"], cfg["architectures"] = "qwen4", ["Qwen4ForConditionalGeneration"]
    caps = json.loads(json.dumps(dc.EXAMPLE_CAPABILITIES))
    caps["model_types"].append("qwen4")
    caps["architectures"].append("Qwen4ForConditionalGeneration")
    w = World(tmp_path, hub=FakeHub(config=json.dumps(cfg).encode()), caps=caps)
    res = w.tick(class_probe=lambda classes: {c: False for c in classes})
    assert res["state"] == "needs-operator" and w.launched == []
    res = w.tick(class_probe=lambda classes: {c: True for c in classes})
    assert res["state"] == "running" and len(w.launched) == 1


def test_state_survives_a_new_process(tmp_path):
    w = World(tmp_path, free=False)
    w.tick()
    state = json.loads((w.watch_dir / "state.json").read_text())
    assert state["state"] == "waiting-for-chips" and state["route"] == "full-port" and state["revision"] == SHA


def test_a_kill_between_starting_and_the_pid_never_starts_twice(tmp_path):
    w = World(tmp_path)

    def boom(argv, log_path):
        w.launched.append(argv)
        raise KeyboardInterrupt                                    # the tick dies after the process may have started
    with pytest.raises(KeyboardInterrupt):
        dw.tick(w.watch_dir, hub=w.hub, chips=w.chips, launch=boom, run_status=w.run_status,
                alive=lambda p, s: True, now=lambda: w.now)
    res = w.tick()
    assert res["state"] == "needs-operator" and len(w.launched) == 1
    assert any("may have started" in e["text"] for e in res["events"])


def test_a_corrupt_state_file_starts_nothing(tmp_path):
    w = World(tmp_path)
    (w.watch_dir / "state.json").write_text("{not json")
    res = w.tick()
    assert res["state"] == "needs-operator" and w.launched == []


def test_a_held_lock_skips_the_tick(tmp_path):
    w = World(tmp_path)
    with dw.locked(w.watch_dir) as got:
        assert got
        res = w.tick()
    assert res.get("skipped") and w.launched == []


def test_other_qwen4_27b_names_are_reported_once_and_not_run(tmp_path):
    w = World(tmp_path, hub=FakeHub(exists=False, others=["Qwen/Qwen4-27B-FP8"]))
    res = w.tick()
    assert [e["text"] for e in res["events"] if e["kind"] == "other-release"] == ["Qwen/Qwen4-27B-FP8"]
    assert w.tick()["events"] == [] and w.launched == []


@pytest.mark.parametrize("target", ["Esteban143/Qwen4-32B", "qwen/Qwen4-27B", "Qwen/../x", ""])
def test_config_refuses_other_organisations(tmp_path, target):
    with pytest.raises(ValueError):
        dw.write_config(tmp_path / "w", dw.Config(target=target, run_dir=str(tmp_path / "run")))


@pytest.mark.parametrize("bad", ['target = 5', 'target = "Qwen/Qwen4-27B"\nquiet_minutes = -1',
                                 'target = "Qwen/Qwen4-27B"\nbringup_args = "x"', 'target = "Qwen/Qwen4-27B"\nextra = 1'])
def test_malformed_config_is_refused(tmp_path, bad):
    d = tmp_path / "w"
    d.mkdir()
    (d / "watch.toml").write_text(bad + '\nrun_dir = "/tmp/x"\n' if "run_dir" not in bad else bad)
    with pytest.raises(ValueError):
        dw.load_config(d)


def test_events_are_appended_to_a_log(tmp_path):
    w = World(tmp_path, free=False)
    w.tick()
    lines = (w.watch_dir / "events.jsonl").read_text().splitlines()
    assert lines and all(json.loads(x)["kind"] for x in lines)


def test_bringup_args_from_the_config_are_passed(tmp_path):
    w = World(tmp_path)
    c = dw.load_config(w.watch_dir)
    c.bringup_args = ["--accept-credentials-visible"]
    dw.write_config(w.watch_dir, c)
    w.tick()
    assert "--accept-credentials-visible" in w.launched[0]


def test_cli_tick_with_a_simulated_release(tmp_path, capsys, monkeypatch):
    """`drop-watch tick --hub-dir` reads a release laid out in a directory, for the M0 and M5 dry runs."""
    rel = tmp_path / "release"
    rel.mkdir()
    (rel / "config.json").write_bytes(cfg_bytes())
    (rel / "model.safetensors.index.json").write_bytes(INDEX)
    sib = [{"rfilename": n, "size": 10} for n in ("config.json", "model.safetensors.index.json",
                                                   "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")]
    (rel / "info.json").write_text(json.dumps({"sha": SHA, "lastModified": "2020-01-01T00:00:00.000Z", "siblings": sib}))
    wd = tmp_path / "watch"
    dw.write_config(wd, dw.Config(target="Qwen/Qwen4-27B", run_dir=str(tmp_path / "run")))
    monkeypatch.setattr(dw, "gozer_chips", lambda: {"free": False, "holders": ["someone"]})
    code = dw.main(["tick", "--watch-dir", str(wd), "--hub-dir", str(rel), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["state"] == "waiting-for-chips"


# ---- the real-world helpers -----------------------------------------------------------------------------------------

class Done:
    def __init__(self, stdout):
        self.stdout, self.returncode = stdout, 0


def chip(state, who=None):
    return {"bdf": "0000:01:00.0", "state": state, "who": who}


@pytest.mark.parametrize("out,free", [
    (json.dumps({"chips": [chip("FREE")] * 4}), True),
    (json.dumps({"chips": [chip("FREE")] * 3 + [chip("HELD", "claude:qwen38-serve")]}), False),
    (json.dumps({"chips": []}), False),
    (json.dumps({"chips": None}), False),
    ("not json", False),
    (json.dumps([1, 2]), False),
])
def test_gozer_chips_counts_only_all_free_as_free(out, free):
    assert dw.gozer_chips(run=lambda *a, **k: Done(out))["free"] is free


def test_gozer_failure_is_not_free():
    def boom(*a, **k):
        raise OSError("no gozer")
    res = dw.gozer_chips(run=boom)
    assert res["free"] is False and "gozer status failed" in res["holders"][0]


def test_process_alive_checks_the_start_time(tmp_path):
    started = dw.launch_process(["sleep", "5"], str(tmp_path / "log"))
    try:
        assert dw.process_alive(started["pid"], started["start_time"])
        assert not dw.process_alive(started["pid"], "1")              # a recycled pid has another start time
        assert not dw.process_alive(started["pid"], None)
    finally:
        os.kill(started["pid"], 9)
    assert oct((tmp_path / "log").stat().st_mode & 0o777) == "0o600"


def test_an_unknown_config_key_is_named(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "watch.toml").write_text('target = "Qwen/Qwen4-27B"\nrun_dir = "/tmp/x"\nextra = 1\n')
    with pytest.raises(ValueError, match="unknown keys: extra"):
        dw.load_config(d)


@pytest.mark.parametrize("index", [b"{not json", b"{}", json.dumps({"weight_map": {"a": "../x.safetensors"}}).encode()])
def test_an_unreadable_weight_index_is_an_unfinished_upload(tmp_path, index):
    hub = FakeHub()
    hub.files["model.safetensors.index.json"] = index
    w = World(tmp_path, hub=hub)
    res = w.tick()
    assert res["state"] == "seen" and w.launched == []
    assert "weight index cannot be read" in json.loads((w.watch_dir / "state.json").read_text())["upload"]


def test_tt_orchard_cli_drop_watch_show(tmp_path):
    import io

    from orchard import cli
    wd = tmp_path / "watch"
    buf = io.StringIO()
    assert cli.main(["drop-watch", "init", "--watch-dir", str(wd), "--target", "Qwen/Qwen4-27B", "--run-dir",
                     str(tmp_path / "run")], env={"HOME": str(tmp_path / "none")}, stdout=buf) == 0
    buf = io.StringIO()
    assert cli.main(["drop-watch", "show", "--watch-dir", str(wd)], env={"HOME": str(tmp_path / "none")}, stdout=buf) == 0
    assert json.loads(buf.getvalue())["state"] == "waiting"
