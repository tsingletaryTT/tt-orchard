# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 417: the coder container mounts a Hugging Face cache and downloads its base model into it when the model is missing, and a
stopped coder container left on disk refuses the next `tt-model serve`. Everything here runs against fake signals, temporary
directories and a scripted docker. No network, no docker, no device."""
import io
import os
from pathlib import Path

import pytest

from fakes import LEASE, FakeClock, FakeRun
from orchard import coder_cache as cc
from orchard import narrate
from orchard import preflight as pf
from orchard.server import ServerControl, ServerSpec, container_name_for
from test_preflight import CLEF_PAYLOAD, FREE, FakeTierCfg, cfg

CODER_MODEL = "Qwen/Qwen3-Coder-Next"
TIERS = {"small": {"placement": "chips", "endpoint": "http://127.0.0.1:8000/v1", "model": CODER_MODEL},
         "cpu": {"placement": "cpu", "endpoint": "http://127.0.0.1:11434/v1", "model": "q"}}


# ---- which cache the coder mounts and what a download leaves in it -------------------------------------------------------------


def test_the_mounted_cache_is_hf_home_from_the_environment_else_the_default_under_home(tmp_path):
    assert cc.effective_cache({"HF_HOME": str(tmp_path / "scratch")}, tmp_path) == tmp_path / "scratch"
    assert cc.effective_cache({}, tmp_path) == tmp_path / ".cache" / "huggingface"
    assert cc.effective_cache({"HF_HOME": ""}, tmp_path) == tmp_path / ".cache" / "huggingface"


def make_blob(cache, repo, name, size):
    blobs = cache / "hub" / repo / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / name).write_bytes(b"x" * size)


def test_partial_downloads_are_the_incomplete_files_of_every_repository(tmp_path):
    make_blob(tmp_path, "models--a--b", "aaa.incomplete", 1000)
    make_blob(tmp_path, "models--c--d", "bbb.incomplete", 500)
    make_blob(tmp_path, "models--c--d", "done", 9999)                  # a finished blob is not partial
    assert cc.partial_download_bytes(tmp_path) == 1500


def test_a_missing_cache_and_a_repository_without_blobs_count_as_zero(tmp_path):
    assert cc.partial_download_bytes(tmp_path / "nowhere") == 0
    (tmp_path / "hub" / "models--a--b").mkdir(parents=True)
    assert cc.partial_download_bytes(tmp_path) == 0


def test_a_linked_repository_directory_is_not_followed(tmp_path):
    other = tmp_path / "elsewhere"
    make_blob(other, "x", "big.incomplete", 4000)
    hub = tmp_path / "cache" / "hub"
    hub.mkdir(parents=True)
    os.symlink(other / "hub" / "x", hub / "models--a--b")
    assert cc.partial_download_bytes(tmp_path / "cache") == 0


def test_a_linked_incomplete_file_is_not_counted(tmp_path):
    big = tmp_path / "big"
    big.write_bytes(b"x" * 4000)
    blobs = tmp_path / "hub" / "models--a--b" / "blobs"
    blobs.mkdir(parents=True)
    os.symlink(big, blobs / "l.incomplete")
    assert cc.partial_download_bytes(tmp_path) == 0


# ---- the boot watch ------------------------------------------------------------------------------------------------------------


def watch(partial_gb, free_start_gb, free_now_gb):
    free = iter([int(free_start_gb * 1e9), int(free_now_gb * 1e9)])
    return cc.BootWatch(Path("/c"), free=lambda p: next(free), partial=lambda p: int(partial_gb * 1e9))


def test_a_quiet_boot_has_nothing_to_report():
    assert watch(0.0, 150, 149.5).report() is None


def test_partial_downloads_are_reported_with_the_cache_path():
    text = watch(12.3, 150, 149.9).report()
    assert "/c" in text and "12.3 GB of partial downloads" in text and "downloading a model" in text


def test_a_fall_in_free_space_is_reported_when_no_partial_file_shows():
    text = watch(0.0, 150, 120).report()
    assert "30.0 GB" in text and "other writers" in text


def test_free_space_that_rose_is_not_a_fall():
    assert watch(0.0, 100, 150).report() is None


def test_an_unreadable_free_space_does_not_raise():
    w = cc.BootWatch(Path("/c"), free=lambda p: None, partial=lambda p: 0)
    assert w.report() is None


# ---- the preflight row for the coder's cache -----------------------------------------------------------------------------------


def row(**over):
    base = dict(cache=Path("/hf"), model_id=CODER_MODEL, present=False, free_gb=1000.0, need_gb=149.0, need_from="the hub's size for it",
                no_fetch=False)
    base.update(over)
    return pf.check_coder_cache(**base)


def test_a_cache_that_has_the_snapshot_passes_and_names_the_cache():
    c = row(present=True)
    assert c.status == "ok" and "/hf" in c.detail and CODER_MODEL in c.detail


def test_a_missing_snapshot_on_a_disk_that_cannot_hold_it_is_refused_for_disk():
    c = row(free_gb=120.0)
    assert (c.status, c.reason) == ("block", "disk-full")
    assert "/hf" in c.detail and "149 GB" in c.detail and "120 GB free" in c.detail


def test_the_headroom_counts_in_the_refusal():
    assert row(free_gb=155.0).status == "block"              # 149 + 10 headroom needs 159
    assert row(free_gb=160.0).status == "warn"


def test_a_missing_snapshot_is_refused_under_no_fetch_even_with_room():
    c = row(no_fetch=True)
    assert (c.status, c.reason) == ("block", "coder-unusable") and "--no-fetch" in c.detail


def test_a_missing_snapshot_with_room_is_a_warning_that_says_where_and_how_much():
    c = row()
    assert c.status == "warn" and "/hf" in c.detail and "149 GB" in c.detail and "10 to 25 minutes" in c.detail


def test_an_unreadable_free_space_warns_and_says_so():
    c = row(free_gb=None)
    assert c.status == "warn" and "free space could not be read" in c.detail


def test_the_size_estimate_names_its_source():
    assert "fixed estimate" in row(need_from="a fixed estimate, because the hub could not be asked").detail


def test_an_hf_home_flag_that_differs_from_the_mounted_cache_is_said():
    c = row(present=True, hf_home_flag=Path("/other"))
    assert c.status == "ok" and "--hf-home is /other" in c.detail and "mounts /hf" in c.detail
    assert "--hf-home" not in row(present=True, hf_home_flag=Path("/hf")).detail


# ---- the preflight row for a leftover container --------------------------------------------------------------------------------

NAME = "tt-model-qwen3-coder-next-p300"


def test_no_container_of_that_name_passes():
    assert pf.check_coder_container(name=NAME, found=[("other", "running")]).status == "ok"


def test_a_stopped_container_of_that_name_is_refused_and_the_fix_is_named():
    c = pf.check_coder_container(name=NAME, found=[(NAME, "exited")])
    assert (c.status, c.reason) == ("block", "coder-unusable") and f"docker rm {NAME}" in c.detail


def test_a_running_container_of_that_name_is_refused_unless_the_run_is_resuming():
    assert pf.check_coder_container(name=NAME, found=[(NAME, "running")]).status == "block"
    assert pf.check_coder_container(name=NAME, found=[(NAME, "running")], resuming=True).status == "warn"


def test_a_stopped_leftover_on_a_resume_is_a_warning_that_keeps_the_fix():
    c = pf.check_coder_container(name=NAME, found=[(NAME, "exited")], resuming=True)
    assert c.status == "warn" and f"docker rm {NAME}" in c.detail


def test_a_docker_that_does_not_answer_is_a_warning():
    assert pf.check_coder_container(name=NAME, found=None).status == "warn"


def test_a_name_that_only_contains_ours_does_not_match():
    assert pf.check_coder_container(name=NAME, found=[(NAME + "-v2", "exited")]).status == "ok"


def test_the_container_name_follows_tt_models_rule():
    assert container_name_for("raahemnabeel/qwen3-coder-next", "p300") == NAME
    assert ServerSpec("raahemnabeel/qwen3-coder-next", "container", 8011, "m", profile="p300").container_name == NAME


# ---- the whole preflight -------------------------------------------------------------------------------------------------------


def coder_cfg(tmp_path, **over):
    c = cfg(tmp_path)
    c.coder.target, c.coder.profile = "raahemnabeel/qwen3-coder-next", "p300"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def signals(**over):
    base = dict(hub_info=lambda m: (pf.parse_hub(CLEF_PAYLOAD), None), local_snapshot=lambda m: None, free_gb=lambda p: 1000.0,
                same_device=lambda a, b: False, credentials=lambda: [], port_in_use=lambda port: False, gozer_status=lambda: FREE,
                load_tiers=lambda p: FakeTierCfg(TIERS), reference_problem=lambda p: None,
                coder_cache=lambda: Path("/hf"), coder_snapshot=lambda cache, model: True, coder_containers=lambda name: [])
    base.update(over)
    return pf.Signals(**base)


def names(out):
    return [c.name for c in out]


def run(tmp_path, sig, **kw):
    c = coder_cfg(tmp_path)
    c.coder.port = 8000
    return pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=sig, **kw)


def test_the_two_rows_follow_the_port_row_and_pass_on_a_clean_machine(tmp_path):
    out = run(tmp_path, signals())
    assert names(out)[:6] == ["hub", "disk", "credentials", "tiers", "port", "coder-cache"] and "container" in names(out)
    by = {c.name: c for c in out}
    assert by["coder-cache"].status == "ok" and by["container"].status == "ok"


def test_a_cache_without_the_coder_snapshot_on_a_small_disk_blocks_the_run(tmp_path):
    asked = []
    sig = signals(coder_snapshot=lambda cache, model: asked.append((cache, model)) and False,
                  hub_info=lambda m: (None, "HTTP 401") if m == CODER_MODEL else (pf.parse_hub(CLEF_PAYLOAD), None),
                  free_gb=lambda p: 100.0 if Path(p) == Path("/hf") else 1000.0)
    out = run(tmp_path, sig)
    row_ = {c.name: c for c in out}["coder-cache"]
    assert (row_.status, row_.reason) == ("block", "disk-full")
    assert asked == [(Path("/hf"), CODER_MODEL)]                  # the row asks about the coder's model, in the mounted cache


def test_the_size_comes_from_the_hub_when_it_answers_and_from_the_fallback_when_it_does_not(tmp_path):
    big = pf.parse_hub({**CLEF_PAYLOAD, "siblings": [{"rfilename": "w.safetensors", "size": 300_000_000_000}]})
    seen = {}

    def hub(model):
        return (big, None) if model == CODER_MODEL else (pf.parse_hub(CLEF_PAYLOAD), None)
    sig = signals(coder_snapshot=lambda c, m: False, hub_info=hub, free_gb=lambda p: 1000.0)
    detail = {c.name: c for c in run(tmp_path, sig)}["coder-cache"].detail
    assert "300 GB" in detail and "hub's size" in detail
    sig = signals(coder_snapshot=lambda c, m: False, hub_info=lambda m: (None, "HTTP 401") if m == CODER_MODEL else (pf.parse_hub(CLEF_PAYLOAD), None))
    detail = {c.name: c for c in run(tmp_path, sig)}["coder-cache"].detail
    assert f"{cc.FALLBACK_BASE_GB:.0f} GB" in detail and "fixed estimate" in detail


def test_no_fetch_reaches_the_cache_row(tmp_path):
    sig = signals(coder_snapshot=lambda c, m: False)
    out = run(tmp_path, sig, no_fetch=True)
    assert {c.name: c for c in out}["coder-cache"].reason == "coder-unusable"


def test_a_leftover_container_blocks_the_run_through_the_container_row(tmp_path):
    asked = []

    def containers(name):
        asked.append(name)
        return [(name, "exited")]
    out = run(tmp_path, signals(coder_containers=containers))
    row_ = {c.name: c for c in out}["container"]
    assert (row_.status, row_.reason) == ("block", "coder-unusable") and asked == [NAME]


def test_a_resumed_run_passes_its_resume_flag_to_the_container_row(tmp_path):
    out = run(tmp_path, signals(coder_containers=lambda name: [(name, "exited")]), resuming=True)
    assert {c.name: c for c in out}["container"].status == "warn"


def test_a_signal_that_raises_is_a_block_row_and_not_a_crash(tmp_path):
    def boom(cache, model):
        raise OSError("disk went away")
    out = run(tmp_path, signals(coder_snapshot=boom))
    row_ = {c.name: c for c in out}["coder-cache"]
    assert row_.status == "block" and "OSError" in row_.detail


def test_without_the_new_signals_the_rows_are_left_out(tmp_path):
    out = run(tmp_path, signals(coder_cache=None, coder_snapshot=None, coder_containers=None))
    assert "coder-cache" not in names(out) and "container" not in names(out)


def test_a_tier_model_that_is_not_an_org_slash_name_leaves_the_cache_row_out(tmp_path):
    tiers = {"small": {"placement": "chips", "endpoint": "http://127.0.0.1:8000/v1", "model": "m"}}
    out = run(tmp_path, signals(load_tiers=lambda p: FakeTierCfg(tiers)))
    assert "coder-cache" not in names(out)


def test_a_bundle_coder_has_no_container_row(tmp_path):
    c = coder_cfg(tmp_path)
    c.coder.port, c.coder.kind = 8000, "bundle"
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert "container" not in names(out)


def test_a_kept_coder_needs_neither_row(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_kept_on_port", lambda c: True)
    out = run(tmp_path, signals())
    assert "coder-cache" not in names(out) and "container" not in names(out)


def test_the_default_signals_read_the_mounted_cache_for_a_complete_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "scratch"))
    s = pf.default_signals(coder_cfg(tmp_path))
    assert s.coder_cache() == tmp_path / "scratch"
    assert s.coder_snapshot(tmp_path / "scratch", CODER_MODEL) is False
    snap = tmp_path / "scratch" / "hub" / "models--Qwen--Qwen3-Coder-Next" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert s.coder_snapshot(tmp_path / "scratch", CODER_MODEL) is False          # a config alone is not a model
    (snap / "model.safetensors").write_bytes(b"x")
    assert s.coder_snapshot(tmp_path / "scratch", CODER_MODEL) is True


# ---- removing the stopped container after a confirmed stop ---------------------------------------------------------------------


def docker_key(argv):
    if argv[:2] == ["docker", "ps"] and "-a" in argv:
        return "ps-a"
    if argv[:2] == ["docker", "rm"]:
        return "rm"
    return " ".join(argv[:2])


def owned_control(listing, rm=(0, NAME, ""), owned=True, kind="container"):
    run_ = FakeRun({"ps-a": [listing], "rm": [rm]}, key=docker_key)
    spec = ServerSpec("raahemnabeel/qwen3-coder-next", kind, 8011, "m", profile="p300", argv=("x",) if kind == "process" else ())
    ctl = ServerControl(spec, run=run_, clock=FakeClock(), sleep=FakeClock().sleep)
    ctl.owned = owned
    return ctl, run_


def test_a_stopped_container_of_ours_is_removed_by_exact_name():
    ctl, run_ = owned_control((0, f"{NAME} exited\n", ""))
    res = ctl.remove_container()
    assert res["removed"] is True and res["state"] == "exited"
    assert ["docker", "rm", NAME] in run_.argvs() and "-f" not in sum(run_.argvs(), [])


def test_a_running_container_is_never_removed():
    ctl, run_ = owned_control((0, f"{NAME} running\n", ""))
    res = ctl.remove_container()
    assert res["removed"] is False and not any(a[:2] == ["docker", "rm"] for a in run_.argvs())


def test_a_container_that_only_contains_our_name_is_left_alone():
    ctl, run_ = owned_control((0, f"{NAME}-v2 exited\nx{NAME} exited\n", ""))
    assert ctl.remove_container() is None and not any(a[:2] == ["docker", "rm"] for a in run_.argvs())


def test_a_container_that_a_refusal_said_another_process_owns_is_not_removed():
    ctl, run_ = owned_control((0, f"{NAME} exited\n", ""), owned=False)
    assert ctl.remove_container() is None and run_.calls == []


def test_a_container_of_unknown_ownership_is_removed_when_it_is_stopped_and_ours_by_name():
    """Unknown (None) is a start that failed for another reason or a ledger record without the field: tt-model made the container under our
    name. Issue 417: after a boot failure the container was left as `Exited (1)` and refused the retry."""
    ctl, run_ = owned_control((0, f"{NAME} exited\n", ""), owned=None)
    assert ctl.remove_container()["removed"] is True and ["docker", "rm", NAME] in run_.argvs()


def test_a_bundle_server_has_no_container_to_remove():
    ctl, run_ = owned_control((0, f"{NAME} exited\n", ""), kind="bundle")
    assert ctl.remove_container() is None and run_.calls == []


def test_a_listing_that_fails_is_reported_and_nothing_is_removed():
    ctl, run_ = owned_control((1, "", "Cannot connect to the Docker daemon"))
    res = ctl.remove_container()
    assert res["listed"] is False and not any(a[:2] == ["docker", "rm"] for a in run_.argvs())


def test_a_failed_rm_is_reported_as_not_removed():
    ctl, _ = owned_control((0, f"{NAME} exited\n", ""), rm=(1, "", "No such container"))
    assert ctl.remove_container()["removed"] is False


# ---- the supervisor's stop path calls it after a confirmed stop ----------------------------------------------------------------


class RecordingLedger:
    def __init__(self):
        self.entries = []

    def append(self, event, stage, **data):
        self.entries.append((event, data))


def bare_supervisor(coder):
    from orchard.supervisor import Supervisor
    sup = Supervisor.__new__(Supervisor)
    sup.coder, sup.ledger = coder, RecordingLedger()
    return sup


def test_the_supervisor_records_a_removal_and_a_leftover():
    class Coder:
        def remove_container(self):
            return {"container": NAME, "state": "exited", "removed": True}
    sup = bare_supervisor(Coder())
    sup._remove_coder_container()
    assert sup.ledger.entries == [("decision", {"decision": "coder container removed",
                                                "result": {"container": NAME, "state": "exited", "removed": True}})]

    class Left:
        def remove_container(self):
            return {"container": NAME, "removed": False, "why": "it is running"}
    sup = bare_supervisor(Left())
    sup._remove_coder_container()
    assert sup.ledger.entries[0][1]["decision"] == "coder container left"


def test_a_coder_without_the_method_and_a_failing_docker_do_not_stop_the_run():
    sup = bare_supervisor(object())
    sup._remove_coder_container()
    assert sup.ledger.entries == []

    class Boom:
        def remove_container(self):
            raise OSError("docker is gone")
    sup = bare_supervisor(Boom())
    sup._remove_coder_container()
    assert sup.ledger.entries[0][0] == "notice" and "docker is gone" in sup.ledger.entries[0][1]["error"]


def test_a_none_result_writes_nothing():
    class Quiet:
        def remove_container(self):
            return None
    sup = bare_supervisor(Quiet())
    sup._remove_coder_container()
    assert sup.ledger.entries == []


@pytest.mark.parametrize("where", ["release loop", "end-of-run release", "relaunch"])
def test_each_confirmed_stop_site_calls_the_removal(where):
    """The three places the supervisor confirms a coder stop (`wait_stopped` ok) each call `_remove_coder_container`: read from the source,
    because those paths need a whole run to reach."""
    import inspect
    from orchard import supervisor
    text = inspect.getsource(supervisor)
    marker = {"release loop": 'check = wait_stopped(self.coder, self.adapter, lease,\n                                         quiet_wait_s',
              "end-of-run release": 'reason="the coder is not confirmed stopped; the lease was not "',
              "relaunch": 'self._block(None, "the coder could not be confirmed stopped for the relaunch", **check)'}[where]
    after = text.split(marker, 1)[1][:700]
    assert "self._remove_coder_container()" in after


# ---- the narrator --------------------------------------------------------------------------------------------------------------


def ledger_with_boot(tmp_path):
    from orchard.ledger import Ledger
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    led = Ledger(run_dir / "ledger.jsonl")
    led.append("decision", None, decision="coder starting", server={"target": "raahemnabeel/qwen3-coder-next"})
    return run_dir, led


def narrator(run_dir, clock, factory):
    out = io.StringIO()
    n = narrate.Narrator(run_dir, narrate.Style(False, False), out, replay=True, clock=clock, heartbeat_s=60.0, boot_watch=factory)
    return n, out


def test_the_heartbeat_of_a_coder_boot_adds_the_cache_growth(tmp_path):
    run_dir, led = ledger_with_boot(tmp_path)
    clock = FakeClock()
    n, out = narrator(run_dir, clock, lambda: type("W", (), {"report": lambda self: "; the cache /c holds 9.0 GB of partial downloads"})())
    n.poll()
    clock.sleep(61)
    n.poll()
    text = " ".join(out.getvalue().split())                 # the narrator wraps its lines
    assert "to boot (1m01s); the cache /c holds 9.0 GB of partial downloads" in text


def test_a_quiet_boot_heartbeat_is_the_old_line(tmp_path):
    run_dir, led = ledger_with_boot(tmp_path)
    clock = FakeClock()
    n, out = narrator(run_dir, clock, lambda: type("W", (), {"report": lambda self: None})())
    n.poll()
    clock.sleep(61)
    n.poll()
    assert "to boot (" in out.getvalue() and "partial" not in out.getvalue()


def test_the_growth_is_only_asked_during_a_coder_boot(tmp_path):
    from orchard.ledger import Ledger
    run_dir, led = ledger_with_boot(tmp_path)
    led.append("decision", None, decision="coder started")
    led.append("decision", None, decision="agent step", tier="small")
    asked = []
    clock = FakeClock()
    n, out = narrator(run_dir, clock, lambda: type("W", (), {"report": lambda self: asked.append(1) or "; GROWTH"})())
    n.poll()
    clock.sleep(61)
    n.poll()
    assert asked == [] and "GROWTH" not in out.getvalue()


def test_a_probe_that_raises_does_not_stop_the_narration(tmp_path):
    run_dir, led = ledger_with_boot(tmp_path)
    clock = FakeClock()

    def boom(self):
        raise OSError("gone")
    n, out = narrator(run_dir, clock, lambda: type("W", (), {"report": boom})())
    n.poll()
    clock.sleep(61)
    n.poll()
    assert "to boot (" in out.getvalue()


def test_the_removal_decisions_are_narrated():
    d = lambda what, result: {"event": "decision", "stage": None, "seq": 1, "ts": None, "data": {"decision": what, "result": result}}
    assert "removed the stopped coder container" in narrate.lines_for(d("coder container removed", {"container": NAME}))[0].text
    left = narrate.lines_for(d("coder container left", {"container": NAME, "why": "it is running"}))[0]
    assert left.bad and "it is running" in left.text


def test_the_bringup_command_gives_its_narrator_the_boot_watch():
    import inspect
    from orchard import cli
    src = inspect.getsource(cli)
    assert "boot_watch=lambda: coder_cache.BootWatch(coder_cache.effective_cache())" in src
