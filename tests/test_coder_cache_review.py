# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Copilot's first review of PR 424 (issue 417): nine findings, one test group each."""
import json
from pathlib import Path

import pytest

from orchard import blocked, cli
from orchard import coder_cache as cc
from orchard import preflight as pf
from orchard.ledger import Ledger
from orchard.server import ServerChipsHeld, ServerNotOwned
from test_coder_cache import CODER_MODEL, TIERS, coder_cfg, signals
from test_coder_chips_held import NAME, REFUSAL, ScriptedCoder, blocks, earlier_launch, launch, rig, rms, serves  # noqa: F401
from test_preflight import CLEF_PAYLOAD, FakeTierCfg


# ---- 1. the coder download shares a disk with the target and base downloads --------------------------------------------------------


def row(**over):
    base = dict(cache=Path("/hf"), model_id=CODER_MODEL, present=False, free_gb=200.0, need_gb=149.0, need_from="the hub's size for it",
                no_fetch=False)
    base.update(over)
    return pf.check_coder_cache(**base)


def test_downloads_planned_on_the_same_disk_count_against_the_coder_download():
    assert row().status == "warn"                                    # 200 GB free holds 149 + 10 alone
    c = row(planned_gb=80.0)
    assert (c.status, c.reason) == ("block", "disk-full") and "80 GB" in c.detail and "120 GB" in c.detail


def test_the_planned_downloads_leave_a_present_snapshot_alone():
    assert row(present=True, planned_gb=500.0).status == "ok"


def target_hub(gb):
    return pf.parse_hub({**CLEF_PAYLOAD, "siblings": [{"rfilename": "w.safetensors", "size": int(gb * 1e9)}]})


def preflight(tmp_path, *, model_id="Cloudflare/clef", same_device=True, no_fetch=False, coder_hub_gb=149.0, target_gb=80.0, free=200.0,
              present=False, local=False):
    def hub(model):
        return (target_hub(coder_hub_gb if model == CODER_MODEL else target_gb), None)
    sig = signals(hub_info=hub, same_device=lambda a, b: same_device, free_gb=lambda p: free,
                  local_snapshot=lambda m: Path("/snap") if local else None, coder_snapshot=lambda c, m: present)
    c = coder_cfg(tmp_path)
    c.coder.port = 8000
    return {r.name: r for r in pf.run_preflight(c, model_id, accept_credentials=False, signals=sig, no_fetch=no_fetch)}


def test_the_target_download_on_the_same_disk_is_counted_for_the_coder(tmp_path):
    rows = preflight(tmp_path, target_gb=80.0, free=200.0)
    assert (rows["coder-cache"].status, rows["coder-cache"].reason) == ("block", "disk-full")


def test_a_target_on_another_disk_is_not_counted(tmp_path):
    assert preflight(tmp_path, target_gb=80.0, free=200.0, same_device=False)["coder-cache"].status == "warn"


def test_a_target_already_on_disk_downloads_nothing(tmp_path):
    assert preflight(tmp_path, target_gb=80.0, free=200.0, local=True)["coder-cache"].status == "warn"


def test_a_coder_that_is_the_target_model_is_counted_once_by_the_disk_row(tmp_path):
    rows = preflight(tmp_path, model_id=CODER_MODEL, target_gb=149.0, free=200.0)
    assert rows["coder-cache"].status != "block" and "disk row" in rows["coder-cache"].detail


# ---- 2. the default cache is the process home's, as tt-model serve sees it -----------------------------------------------------


def test_the_default_coder_cache_ignores_the_configured_operator_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "process-home"))
    c = coder_cfg(tmp_path)
    c.operator_home = tmp_path / "other-operator"
    assert pf.default_signals(c).coder_cache() == tmp_path / "process-home" / ".cache" / "huggingface"


# ---- 3. a cache with some shards and no index is not a complete snapshot -----------------------------------------------------------


def snap(tmp_path, files):
    d = tmp_path / "snap"
    d.mkdir()
    for name, text in files.items():
        (d / name).write_text(text)
    return d


INDEX = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})


def test_one_shard_and_no_index_is_a_partial_download(tmp_path):
    d = snap(tmp_path, {"config.json": "{}", "model-00001-of-00002.safetensors": "x"})
    assert cc.shard_problem(d) and "index" in cc.shard_problem(d)


def test_all_shards_without_an_index_are_not_complete(tmp_path):
    d = snap(tmp_path, {"config.json": "{}", "model-00001-of-00002.safetensors": "x", "model-00002-of-00002.safetensors": "x"})
    assert cc.shard_problem(d)


def test_an_index_with_a_shard_missing_is_not_complete(tmp_path):
    d = snap(tmp_path, {"config.json": "{}", "model-00001-of-00002.safetensors": "x", "model.safetensors.index.json": INDEX})
    assert cc.shard_problem(d) and "2" in cc.shard_problem(d)


def test_every_shard_and_the_index_are_complete(tmp_path):
    d = snap(tmp_path, {"config.json": "{}", "model-00001-of-00002.safetensors": "x", "model-00002-of-00002.safetensors": "x",
                        "model.safetensors.index.json": INDEX})
    assert cc.shard_problem(d) is None


def test_a_single_file_model_needs_no_index(tmp_path):
    assert cc.shard_problem(snap(tmp_path, {"config.json": "{}", "model.safetensors": "x"})) is None


def test_the_default_signal_refuses_the_partial_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "scratch"))
    s = pf.default_signals(coder_cfg(tmp_path))
    d = tmp_path / "scratch" / "hub" / "models--Qwen--Qwen3-Coder-Next" / "snapshots" / "abc"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_text("x")
    assert s.coder_snapshot(tmp_path / "scratch", CODER_MODEL) is False


# ---- 4. a refused restore persists the refusal ---------------------------------------------------------------------------------


def test_the_in_use_refusal_is_a_not_owned_refusal_for_every_handler():
    assert issubclass(ServerChipsHeld, ServerNotOwned)


def test_a_restore_refused_for_held_chips_is_recorded_so_recovery_respects_it(tmp_path):
    from fakes import make_handoff
    from orchard.handoff import progress
    h, world, ledger = make_handoff(tmp_path)
    h.park()

    def refused(lease):
        h.server.owned = False
        raise ServerChipsHeld(f"tt-model serve exited 1: {REFUSAL}")

    world.coder_running = False
    h.server.start = refused
    h.server.record = lambda: {"kind": "container", "owned": h.server.owned}
    with pytest.raises(Exception, match="in use by another container"):
        h.restore()
    assert progress(ledger.read()).server["owned"] is False


# ---- 5. the ledger-free release stays ledger-free ------------------------------------------------------------------------------


def bare(rig, coder_cls):
    from orchard import supervisor
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    led = Ledger(rig.run_dir / "ledger.jsonl")
    led.__enter__()
    sup = supervisor.build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=coder_cls(rig.m),
                           versions={"tt_model": "test"}, clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p),
                           home=rig.home, containers=rig.containers)
    return sup, led


def test_the_ledger_free_release_removes_the_container_without_writing_and_still_releases(rig):
    sup, led = bare(rig, lambda m: ScriptedCoder(m, serve=[(0, "", "")]))
    try:
        sup.coder_lease = sup.adapter.acquire(2, "orchard:supervisor", "coder")

        writes = []
        sup.ledger.append = lambda *a, **k: writes.append(a)
        assert sup._release_from_memory() is True
        assert writes == []                                  # nothing is written: the ledger is what failed
        assert rig.m.leases == {} and len(rms(sup.coder.run)) == 1
    finally:
        led.__exit__(None, None, None)


def test_a_removal_that_raises_cannot_stop_the_ledger_free_release(rig):
    class Boom(ScriptedCoder):
        def remove_container(self, **kw):
            raise RuntimeError("docker exploded")
    sup, led = bare(rig, lambda m: Boom(m, serve=[(0, "", "")]))
    try:
        sup.coder_lease = sup.adapter.acquire(2, "orchard:supervisor", "coder")
        assert sup._release_from_memory() is True and rig.m.leases == {}
    finally:
        led.__exit__(None, None, None)


def test_a_failing_ledger_write_after_a_removal_does_not_raise(rig):
    sup, led = bare(rig, lambda m: ScriptedCoder(m, serve=[(0, "", "")]))
    try:
        def boom(*a, **k):
            raise OSError("no space left on device")
        sup.ledger.append = boom
        sup._remove_coder_container()                       # must return, not raise
        assert len(rms(sup.coder.run)) == 1
    finally:
        led.__exit__(None, None, None)


# ---- 6. the container this run created, by identity, not by a launch count ------------------------------------------------------


def test_a_container_that_predates_the_run_is_not_removed_after_an_earlier_refused_attempt(rig):
    earlier_launch(rig, created=None)                       # the earlier attempt was refused and made nothing
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)]))
    assert rig.run() == 5
    assert rms(made[0].run) == [] and len(serves(made[0].run)) == 1


def test_a_container_with_another_id_than_the_one_this_run_created_is_not_removed(rig):
    earlier_launch(rig, created="c1")
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)], listing=(0, f"{NAME} exited zzz\n", "")))
    assert rig.run() == 5
    assert rms(made[0].run) == []


def test_a_started_coder_records_the_id_of_its_container(rig):
    launch(rig, lambda m: ScriptedCoder(m, serve=[(0, "", "")], listing=(0, f"{NAME} running c9\n", "")))
    rig.run()
    made = [d for d in rig.decisions() if d["decision"] == "coder container created"]
    assert made and made[0]["container_id"] == "c9" and made[0]["name"] == NAME


def test_a_start_that_fails_after_creating_its_container_records_the_id_too(rig):
    launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", "container exited with code 1")], listing=(0, f"{NAME} exited c7\n", "")))
    assert rig.run() == 5
    assert [d["container_id"] for d in rig.decisions() if d["decision"] == "coder container created"] == ["c7"]


def test_a_refused_start_records_no_created_container(rig):
    launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)]))
    assert rig.run() == 5
    assert not [d for d in rig.decisions() if d["decision"] == "coder container created"]


def test_the_removal_checks_the_id_it_was_given():
    from test_coder_chips_held import control
    ctl, run = control([(1, "", REFUSAL)], listing=(0, f"{NAME} exited c1\n", ""))
    with pytest.raises(ServerChipsHeld):
        ctl.start(__import__("fakes").LEASE)
    assert ctl.remove_container(even_if_unowned=True, only_id="other")["removed"] is False and rms(run) == []
    assert ctl.remove_container(even_if_unowned=True, only_id="c1")["removed"] is True


# ---- 7. a refusal is persisted before the block --------------------------------------------------------------------------------


def test_the_in_use_refusal_is_recorded_as_not_started_with_the_owner_false(rig):
    launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)]))
    assert rig.run() == 5
    from orchard.stages import coder_state
    entries = rig.entries()
    refused = [e["data"] for e in entries if e["event"] == "decision" and e["data"].get("decision") == "coder not started"]
    assert refused and refused[-1]["server"]["owned"] is False and REFUSAL in refused[-1]["reason"]
    assert coder_state(entries)[1]["owned"] is False


# ---- 8. an off-bus chip keeps its hardware code ----------------------------------------------------------------------------------


def test_an_off_bus_finding_wins_over_the_container_text():
    reason = (f"blocked: a chip this run uses is {blocked.OFF_BUS_MARK}: chip 1 is off. It was found after this failure: the coder did not "
              f"start and answer; nothing is retried: tt-model serve exited 1: {REFUSAL}")
    assert blocked.block_code({"decision": "pause", "reason": reason}) == "hardware-unhealthy"
    assert blocked.retryable("hardware-unhealthy", reason) is False


# ---- 9. --prune-caches with a coder-cache-only disk block ---------------------------------------------------------------------------


from test_prune_to_fit import box, real_caches, make  # noqa: E402,F401  (fixtures)
from test_cli import Pipe, Recorder, conf  # noqa: E402,F401  (conf is a fixture)


class CoderTiers:
    tiers = {"large": {"placement": "chips", "endpoint": "http://127.0.0.1:8000/v1", "model": CODER_MODEL}}


def prune_run(conf, *extra):
    from test_cli import PAYLOAD
    out, rec = Pipe(), Recorder()
    sig = pf.Signals(hub_info=lambda m: (pf.parse_hub(PAYLOAD), None) if m == "Cloudflare/clef" else (None, "HTTP 401"),
                     local_snapshot=lambda m: None, free_gb=lambda p: 10.0 if Path(p) == Path("/hf") else 1000.0,
                     same_device=lambda a, b: False, credentials=lambda: [], port_in_use=lambda port: False,
                     gozer_status=lambda: __import__("test_cli").FREE, load_tiers=lambda p: CoderTiers(),
                     coder_cache=lambda: Path("/hf"), coder_snapshot=lambda c, m: False, coder_containers=lambda n: [])
    code = cli.main(["--config", str(conf), "bringup", "Cloudflare/clef", *extra], env={}, stdout=out, signals=sig,
                    supervisor_main=rec.supervisor, fetcher=rec.fetcher, sleep=lambda s: None)
    return code, out.getvalue(), rec


def test_prune_caches_with_only_the_coder_cache_short_refuses_and_does_not_crash(real_caches, box):
    old = make(box, "old", 130, 90)
    code, out, rec = prune_run(real_caches, "--prune-caches")
    assert code == 2 and rec.argv is None and old.exists()


def test_the_same_on_a_dry_run(real_caches, box):
    old = make(box, "old", 130, 90)
    code, out, rec = prune_run(real_caches, "--prune-caches", "--dry-run")
    assert code == 2 and old.exists()


def test_a_container_that_took_the_name_between_the_check_and_the_removal_is_left_alone(rig):
    earlier_launch(rig, created="c1")
    first, swapped = (0, f"{NAME} exited c1\n", ""), (0, f"{NAME} exited zzz\n", "")
    # the supervisor's look matches the recorded id; the removal's own look finds another container under the name
    made = launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)], listing=[first, swapped]))
    assert rig.run() == 5
    assert rms(made[0].run) == []


def test_the_listing_asks_docker_for_the_container_id():
    from test_coder_chips_held import control
    ctl, run = control([(0, "", "")], listing=(0, f"{NAME} exited c1\n", ""))
    assert ctl.find_container_rows() == [(NAME, "exited", "c1")]
    listing = [a for a in run.argvs() if a[:3] == ["docker", "ps", "-a"]][0]
    assert "{{.ID}}" in listing[listing.index("--format") + 1]
