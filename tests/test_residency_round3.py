# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 327, review of PR 435 round 2 (Copilot round 3): eight findings, each pinned by a test that failed on the round 2 code.

1. host-resident parameters: the totals only block when placement is established from the tensor names of the weight index as well as the
   config; 2. a finish-only resume (round 1, kept); 3. FP8 vectors; 4. a deeply nested config or answer; 5. the Hub's current revision against
   the snapshot the run will use; 6. a cache root that does not exist yet; 7. a MemAvailable of zero; 8. a resume marker the supervisor
   would not accept; 9. scoring ports, which convert nothing. Nothing here touches a network and no weight is read."""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import decode_floor, preflight as pf, residency as rs, sizing
from test_residency_options import FOUR_CHIPS, FakeResp, HYDROGEN, REAL_MACHINE_FACTS, _cfg, hub_payload, option

REPO = Path(__file__).resolve().parent.parent
GIB = 1 << 30


# ---- a child with a hard memory limit and a clock, for input that is built to be deep ----------------------------------------------------

def in_limited_child(code: str, timeout: int = 60) -> subprocess.CompletedProcess:
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (2 * GIB, 2 * GIB))
    return subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(REPO)!r})\n{code}"], capture_output=True, text=True,
                          timeout=timeout, preexec_fn=limit, cwd=str(REPO), start_new_session=True)


# ---- 3. FP8 vectors: the bound is the smaller of the stored width and the precision's width --------------------------------------------------

@pytest.mark.parametrize("dtype,width", [("F8_E4M3", 1), ("F8_E5M2", 1), ("F8_E8M0", 1), ("BF16", 2), ("F16", 2), ("F32", 4), ("F64", 8)])
def test_the_bound_counts_each_float_dtype_at_the_smaller_of_its_stored_width_and_the_precisions(dtype, width):
    n = 1_000_000_000
    rec = rs.totals_bound({dtype: n}, box_chips=4)
    for p in rs.PRECISIONS:
        expect = int(min(Fraction(width), Fraction(decode_floor.DTYPE_BYTES[p])) * n)
        assert option(rec, p)["device_bytes"] == expect, (dtype, p)


@pytest.mark.parametrize("dtype", ["F8_E4M3", "F8_E5M2", "F8_E8M0", "BF16", "F16", "F32", "F64"])
@pytest.mark.parametrize("shape", [(4096,), (64, 64), (8, 8, 64)])
def test_the_bound_never_exceeds_what_the_header_estimate_stores_for_the_same_tensor(dtype, shape):
    elements = 1
    for d in shape:
        elements *= d
    t = rs.Tensor("model.layers.0.x.weight", dtype, shape, elements * rs.DTYPE_WIDTH[dtype], elements, "f")
    rec = rs.totals_bound({dtype: elements}, box_chips=4)
    for p in rs.PRECISIONS:
        assert option(rec, p)["device_bytes"] <= rs._stored_bytes(t, p), (dtype, shape, p)


def test_the_hydrogen_figures_do_not_change_with_the_new_bound():
    rec = rs.totals_bound(HYDROGEN, box_chips=4)
    assert [option(rec, p)["chips_needed"] for p in rs.PRECISIONS] == [49, 26, 14]       # a BF16 and F32 checkpoint: the same as before


def test_the_bound_says_what_it_assumes_about_the_stored_width():
    rec = rs.totals_bound({"F8_E4M3": 10 ** 9}, box_chips=4)
    note = rec["estimated"]["precision_options"]["note"]
    assert "smaller" in note and "stored" in note


# ---- 6 and 7. the machine readings -----------------------------------------------------------------------------------------------------------

def test_a_cache_root_that_does_not_exist_yet_is_measured_at_its_nearest_existing_parent(tmp_path):
    here = REAL_MACHINE_FACTS(tmp_path)[0]
    free = REAL_MACHINE_FACTS(tmp_path / "fresh" / "root" / "kernels")[0]
    assert isinstance(free, int) and free > 0 and abs(free - here) < GIB
    (tmp_path / "afile").write_text("x")
    under_a_file = REAL_MACHINE_FACTS(tmp_path / "afile" / "below")[0]
    assert isinstance(under_a_file, int) and under_a_file > 0
    assert REAL_MACHINE_FACTS(None)[0] is None and REAL_MACHINE_FACTS(12)[0] is None


def test_a_cache_root_that_does_not_exist_yet_can_still_be_short_of_disk(tmp_path, monkeypatch):
    seen = []
    real = rs.shutil.disk_usage

    def fake(p):
        seen.append(Path(p))
        if Path(p) == tmp_path:
            return SimpleNamespace(free=7, total=100, used=93)
        raise FileNotFoundError(p)
    monkeypatch.setattr(rs.shutil, "disk_usage", fake)
    assert REAL_MACHINE_FACTS(tmp_path / "a" / "b")[0] == 7 and seen[-1] == tmp_path and len(seen) == 3
    monkeypatch.setattr(rs.shutil, "disk_usage", real)


@pytest.mark.parametrize("reading,expect", [(0, 0), (5, 5), (-1, None), (None, None), (True, None), ("7", None)])
def test_a_mem_available_of_zero_is_a_reading_and_a_negative_or_odd_one_is_not(tmp_path, monkeypatch, reading, expect):
    monkeypatch.setattr(sizing, "mem_available_bytes", lambda: reading)
    assert REAL_MACHINE_FACTS(tmp_path)[1] == expect


def test_no_available_ram_at_all_is_a_shortfall_not_a_skipped_check(tmp_path, monkeypatch):
    monkeypatch.setattr(sizing, "mem_available_bytes", lambda: 0)
    free, mem = REAL_MACHINE_FACTS(tmp_path)
    rec = {"label": "estimate", "scan": "ran", "assumptions": [], "estimated": {"chips_needed": 1, "device_bytes": 1, "first_boot": {
        "snapshot_local": True, "checkpoint_bytes_to_download": 0, "cache_bytes": 1, "disk_bytes": 1, "conversion_in_flight_bytes": 0,
        "host_ram_bytes": 10 * GIB}}}
    why = rs.machine_shortfall(rec, free_disk_bytes=free, mem_available_bytes=mem)
    assert why and "host RAM" in why and "0.0 GiB is available" in why


# ---- 4. nesting that is deeper than the parser can read ----------------------------------------------------------------------------------------

DEEP = "body = ('[' * 400000 + ']' * 400000).encode()"
# The response gives the body once and then ends, as a real one does. An earlier version returned the same bytes for every read, so the reader
# stopped at its byte cap and the parser was never reached (review of PR 435, round 4).
OPENER = ("import io\nclass R:\n    def __init__(s): s.f = io.BytesIO(body)\n    def __enter__(s): return s\n    def __exit__(s, *a): return False\n"
          "    def read(s, n=-1): return s.f.read(n)\n")


def test_a_config_nested_past_the_recursion_limit_is_unreadable_and_never_raises():
    done = in_limited_child(f"{DEEP}\n{OPENER}from orchard import preflight as pf\n"
                            "print(pf.read_hub_config('owner/name', 'a' * 40, opener=lambda *a, **k: R()))")
    assert done.returncode == 0 and done.stdout.strip() == "None", done.stderr[-300:]


def test_a_weight_index_nested_past_the_recursion_limit_is_unreadable_and_never_raises():
    done = in_limited_child(f"{DEEP}\n{OPENER}from orchard import preflight as pf\n"
                            "print(pf.read_hub_weight_index('owner/name', 'a' * 40, opener=lambda *a, **k: R()))")
    assert done.returncode == 0 and done.stdout.strip() == "None", done.stderr[-300:]


def test_a_hub_answer_nested_past_the_recursion_limit_is_an_error_text_and_never_raises(tmp_path):
    done = in_limited_child(
        f"{DEEP}\n{OPENER}import urllib.request\nfrom orchard import preflight as pf\nfrom orchard.bringup_config import BringupConfig, Coder\n"
        f"from pathlib import Path\ncfg = BringupConfig(runs_root=Path({str(tmp_path)!r}) / 'runs', coder=Coder(target='t', port=8000, chips=4),"
        f" tiers=Path({str(tmp_path)!r}) / 't.toml', hf_home=Path({str(tmp_path)!r}) / 'hf', cache_root=Path({str(tmp_path)!r}) / 'cache')\n"
        "urllib.request.urlopen = lambda *a, **k: R()\n"
        "info, err = pf.default_signals(cfg).hub_info('owner/name')\n"
        "print(info, err is not None and 'RecursionError' in err)")
    assert done.returncode == 0 and done.stdout.strip() == "None True", done.stderr[-300:]


# ---- 1. host-resident parameters: placement is established from the tensor names too -------------------------------------------------------

def test_the_weight_index_is_read_without_a_token_pinned_to_the_revision_and_bounded():
    seen = []

    def opener(req, timeout=None):
        seen.append(req)
        return FakeResp(json.dumps({"weight_map": {"model.layers.0.q.weight": "a.safetensors", "lm_head.weight": "b.safetensors"}}).encode())
    names = pf.read_hub_weight_index("owner/name", "a" * 40, opener=opener)
    assert names == ["model.layers.0.q.weight", "lm_head.weight"]
    assert seen[0].full_url == f"https://huggingface.co/owner/name/resolve/{'a' * 40}/model.safetensors.index.json"
    assert not any(k.lower() == "authorization" for k in seen[0].headers)
    big = FakeResp(b" " * (pf.INDEX_FETCH_BYTES + 1))
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, **k: big) is None
    assert pf.read_hub_weight_index("owner/name/../x", "main", opener=lambda *a, **k: pytest.fail("requested")) is None


def test_the_index_size_cap_is_what_refuses_a_valid_but_oversized_answer():
    body = json.dumps({"weight_map": {"model.layers.0.q.weight": "a.safetensors"}}).encode()
    at_the_cap = FakeResp(body + b" " * (pf.INDEX_FETCH_BYTES - len(body)))
    over = FakeResp(body + b" " * (pf.INDEX_FETCH_BYTES + 1 - len(body)))
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, **k: at_the_cap) == ["model.layers.0.q.weight"]
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, **k: over) is None          # valid JSON, so only the cap refuses it


@pytest.mark.parametrize("body", [b"not json", b"[]", b"{}", b'{"weight_map": []}', b'{"weight_map": {}}', b'{"weight_map": {"a": 1}, "x": [',
                                  json.dumps({"weight_map": {f"t{i}": "f" for i in range(3)}, "extra": {"1": 2}}).encode()[:-3]])
def test_a_weight_index_that_is_not_a_name_map_is_unreadable(body):
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, body=body, **k: FakeResp(body)) is None


def test_a_missing_or_failing_index_is_unreadable():
    def gone(*a, **k):
        raise OSError("404")
    assert pf.read_hub_weight_index("owner/name", "main", opener=gone) is None
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, **k: (_ for _ in ()).throw(TimeoutError("slow"))) is None


def test_the_names_of_a_weight_index_say_whether_a_host_table_is_among_the_counts():
    plain = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "lm_head.weight"]
    assert rs.index_has_host_table(plain) is False
    assert rs.index_has_host_table(plain + ["model.layers.3.per_layer_embedding.weight"]) is True
    assert rs.index_has_host_table(["model.ngram_embed.0.weight"]) is True
    assert rs.index_has_host_table(["model.layers.0.ple.embed.weight"]) is True
    assert rs.index_has_host_table([]) is None and rs.index_has_host_table(None) is None and rs.index_has_host_table("x") is None
    assert rs.index_has_host_table([3, None]) is False or rs.index_has_host_table([3, None]) is None


@pytest.mark.parametrize("config,names,expect", [
    ({"hidden_size": 8}, ["a.weight"], False),                       # both read, neither names a table: placement is established
    ({"hidden_size": 8}, None, None),                                 # the index could not be read (a single file, a gated repository)
    (None, ["a.weight"], None),                                       # the config could not be read
    ({"hidden_size": 8}, [], None),
    ({"ngram_size": 3}, ["a.weight"], True),                          # the config names one
    ({"hidden_size": 8}, ["x.per_layer.weight"], True),               # a tensor name names one although the config is silent
    (None, ["x.per_layer.weight"], True),
])
def test_placement_is_established_only_when_the_config_and_the_tensor_names_were_both_read_and_name_no_table(config, names, expect):
    assert rs.host_table_from(config, names) is expect


def test_a_totals_block_needs_established_placement_and_a_silent_config_alone_is_advisory():
    hub = pf.parse_hub(hub_payload())
    hub.host_table = rs.host_table_from({"hidden_size": 8}, None)
    assert pf.check_fit(hub, 4).status == "warn"
    hub.host_table = rs.host_table_from({"hidden_size": 8}, ["model.layers.0.q.weight"])
    assert pf.check_fit(hub, 4).status == "block"


def test_the_hub_reader_reads_the_config_and_the_index_for_a_model_with_counts(monkeypatch, tmp_path):
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        if "/api/models/" in req.full_url:
            return FakeResp(json.dumps(hub_payload()).encode())
        if req.full_url.endswith("/config.json"):
            return FakeResp(json.dumps({"hidden_size": 8}).encode())
        return FakeResp(json.dumps({"weight_map": {"model.layers.0.q.weight": "a"}}).encode())
    monkeypatch.setattr(pf.urllib.request, "urlopen", fake_urlopen)
    info, err = pf.default_signals(_cfg(tmp_path)).hub_info("deep-principle/Hydrogen-1.0")
    assert err is None and info.host_table is False
    assert any(u.endswith("/config.json") for u in seen) and any(u.endswith("/model.safetensors.index.json") for u in seen)


# ---- 5. the Hub's current revision against the snapshot the run will use ------------------------------------------------------------------------

def test_the_totals_block_only_when_the_snapshot_the_run_uses_is_the_revision_they_describe():
    hub = pf.parse_hub(hub_payload())
    hub.host_table = False
    assert pf.check_fit(hub, 4).status == "block"                                  # nothing local: the download is the Hub's current revision
    assert pf.check_fit(hub, 4, local_sha=hub.sha).status == "block"               # the local snapshot is that revision
    c = pf.check_fit(hub, 4, local_sha="0123456789abcdef")
    assert c.status == "warn" and c.reason is None and "0123456789ab" in c.detail and hub.sha in c.detail and "Stage 0" in c.detail
    assert pf.check_fit(hub, 4, local_sha="").status == "block"                    # an empty name is no local snapshot


def test_a_hub_answer_with_no_revision_cannot_vouch_for_a_local_snapshot():
    payload = hub_payload()
    del payload["sha"]
    hub = pf.parse_hub(payload)
    hub.host_table = False
    assert hub.sha == ""
    assert pf.check_fit(hub, 4, local_sha="0123456789abcdef").status == "warn"             # nothing to compare the local revision with
    assert pf.check_fit(hub, 4).status == "block"                                          # no local snapshot: the download is what the Hub has


def test_run_preflight_does_not_block_a_local_snapshot_of_another_revision_on_the_hubs_current_counts(tmp_path):
    import test_preflight as tp
    info = pf.parse_hub(hub_payload())
    info.host_table = False
    snap = tmp_path / "hf" / "hub" / "models--deep-principle--Hydrogen-1.0" / "snapshots" / "0123456789abcdef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    for local, status in ((snap, "warn"), (None, "block")):
        s = tp.signals(hub_info=lambda m: (info, None), local_snapshot=lambda m, local=local: local, gozer_status=lambda: FOUR_CHIPS,
                       free_gb=lambda p: 5000.0)
        out = pf.run_preflight(_cfg(tmp_path), "deep-principle/Hydrogen-1.0", accept_credentials=False, signals=s)
        fit = next(c for c in out if c.name == "fit")
        assert fit.status == status, (local, fit.detail)
        assert ("fit" in [c.name for c in pf.blocked(out)]) is (status == "block")


# ---- 8. a resume marker the supervisor would not accept ------------------------------------------------------------------------------------------

class FakeLedger:
    def __init__(self, entries=()):
        self.entries = list(entries)

    def read(self):
        return self.entries

    def append(self, *a, **k):
        self.entries.append({"event": a[0], "stage": a[1], "data": k})


def marker_stub(tmp_path, *, vouched, body="{}", failed=False):
    from orchard import stages
    d = tmp_path / "stages" / "2"
    d.mkdir(parents=True)
    (d / "test-result.json").write_text(body)
    ledger = FakeLedger()
    if vouched:
        rec = stages.evidence_record(tmp_path, d / "test-result.json")
        ledger.entries.append({"event": "evidence", "stage": 2, "data": {"path": rec["path"], "sha256": rec["sha256"], "returncode": 0,
                                                                          "timed_out": False}})
    return SimpleNamespace(run_dir=tmp_path, ledger=ledger, _test_failure=lambda spec, stage_dir: ({"problem": "failed"} if failed else None))


def pending(stub, **kw):
    from orchard.supervisor import Supervisor
    return Supervisor._hardware_pending(stub, SimpleNamespace(number=2, boards=1), resuming=kw.get("resuming", True), escalated=kw.get("escalated", False))


def test_a_marker_the_supervisor_did_not_record_does_not_make_a_finish_only_resume(tmp_path):
    assert pending(marker_stub(tmp_path, vouched=False)) is True          # open_stage_dir moves it aside and the hardware phase reruns


def test_a_marker_the_supervisor_recorded_is_a_finish_only_resume(tmp_path):
    assert pending(marker_stub(tmp_path, vouched=True)) is False


def test_a_marker_edited_after_its_record_does_not_make_a_finish_only_resume(tmp_path):
    stub = marker_stub(tmp_path, vouched=True)
    (tmp_path / "stages" / "2" / "test-result.json").write_text('{"edited": true}')
    assert pending(stub) is True


def test_a_marker_the_bounded_reader_refuses_does_not_make_a_finish_only_resume(tmp_path):
    stub = marker_stub(tmp_path, vouched=False)
    marker = tmp_path / "stages" / "2" / "test-result.json"
    marker.unlink()
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_text("{}")
    marker.symlink_to(outside)
    assert pending(stub) is True


def test_an_escalated_resume_still_reruns_a_failed_test_only_with_a_vouched_marker(tmp_path):
    assert pending(marker_stub(tmp_path, vouched=True, failed=True), escalated=True) is True
    assert pending(marker_stub(tmp_path / "b", vouched=False, failed=True), escalated=True) is True


def test_the_pending_check_writes_no_ledger_entry(tmp_path):
    stub = marker_stub(tmp_path, vouched=False)
    before = len(stub.ledger.entries)
    pending(stub)
    assert len(stub.ledger.entries) == before


def test_open_stage_dir_and_the_pending_check_agree_on_which_markers_resume(tmp_path):
    """Both read `stages.marker_state`, so a marker that resumes in one resumes in the other, for each way a marker can be refused."""
    from orchard import stages
    spec = stages.FULL_PORT_STAGE_2
    assert spec.marker == "test-result.json"
    for name, vouched, edit in (("recorded", True, False), ("unrecorded", False, False), ("edited", True, True)):
        run = tmp_path / name
        stub = marker_stub(run, vouched=vouched)
        if edit:
            (run / "stages" / "2" / "test-result.json").write_text('{"edited": true}')
        _, resumed = stages.open_stage_dir(run, spec, resuming=True, ledger=stub.ledger)
        assert resumed is (not pending(stub)), name
        assert resumed is (name == "recorded"), name


def test_marker_state_names_each_way_a_marker_is_refused_and_vouches_the_plan_marker_by_its_own_entry(tmp_path):
    from orchard import stages
    run = tmp_path
    plan = run / "stages" / "2" / "tests" / "plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}")
    ledger = FakeLedger()
    assert stages.marker_state(run, 2, "tests/plan.json", ledger)[0] == "not vouched"         # no "hardware test list" entry
    assert stages.marker_state(run, 2, "nothing.json", ledger) == ("absent", None, None)
    assert stages.marker_state(run, 2, None, ledger) == ("absent", None, None)
    rec = stages.evidence_record(run, plan)
    ledger.entries.append({"event": "evidence", "stage": 2, "data": {"path": rec["path"], "sha256": rec["sha256"], "what": "hardware test list"}})
    state, got, reason = stages.marker_state(run, 2, "tests/plan.json", ledger)
    assert state == "ok" and got == rec and reason is None
    ledger.entries[-1]["data"]["what"] = "something else"
    assert stages.marker_state(run, 2, "tests/plan.json", ledger)[0] == "not vouched"


def test_open_stage_dir_records_why_a_marker_is_not_a_resume_point(tmp_path):
    from orchard import stages
    spec = stages.FULL_PORT_STAGE_2
    unrecorded = marker_stub(tmp_path / "a", vouched=False)
    _, resumed = stages.open_stage_dir(tmp_path / "a", spec, resuming=True, ledger=unrecorded.ledger)
    assert resumed is False
    assert [e["data"].get("decision") for e in unrecorded.ledger.entries if e["event"] == "decision"][:1] == ["marker not written by the supervisor"]
    linked = marker_stub(tmp_path / "b", vouched=False)
    marker = tmp_path / "b" / "stages" / "2" / "test-result.json"
    marker.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    marker.symlink_to(outside)
    _, resumed = stages.open_stage_dir(tmp_path / "b", spec, resuming=True, ledger=linked.ledger)
    decisions = [e["data"] for e in linked.ledger.entries if e["event"] == "decision"]
    assert resumed is False and decisions[0]["decision"] == "marker cannot be read and is not a resume point" and decisions[0]["reason"]


# ---- 9. scoring ports convert nothing ---------------------------------------------------------------------------------------------------------

def _rec_needing_the_machine():
    return {"label": "estimate", "scan": "ran", "assumptions": [], "estimated": {"chips_needed": 1, "device_bytes": 1,
            "first_boot": {"snapshot_local": True, "checkpoint_bytes_to_download": 0, "cache_bytes": 1, "disk_bytes": 1 << 40,
                           "conversion_in_flight_bytes": 0, "host_ram_bytes": 1 << 40}}}


@pytest.mark.parametrize("draft,reads,blocks", [("full_port", 1, 1), ("scoring_check", 0, 0)])
def test_only_a_stage_whose_port_converts_weights_here_is_held_to_the_first_boots_disk_and_ram(tmp_path, monkeypatch, draft, reads, blocks):
    from orchard.supervisor import Supervisor
    monkeypatch.setattr(rs, "read_record", lambda run_dir: _rec_needing_the_machine())
    seen, blocked_with = [], []
    monkeypatch.setattr(rs, "machine_facts", lambda path: seen.append(path) or (1, 1))
    stub = SimpleNamespace(run_dir=tmp_path, lab=None, paths=SimpleNamespace(cache_root=tmp_path / "cache"), _box_chips=lambda spec: 4,
                           _block=lambda stage, why, **ev: blocked_with.append(why))
    Supervisor._check_residency(stub, SimpleNamespace(boards=1, number=2, draft=draft))
    assert len(seen) == reads and len(blocked_with) == blocks


def test_a_scoring_stage_still_gets_the_chip_count_check(tmp_path, monkeypatch):
    from orchard.supervisor import Supervisor
    rec = {"label": "estimate", "scan": "ran", "assumptions": [], "estimated": {"chips_needed": 9, "device_bytes": 9 << 30}}
    monkeypatch.setattr(rs, "read_record", lambda run_dir: rec)
    blocked_with = []
    stub = SimpleNamespace(run_dir=tmp_path, lab=None, paths=SimpleNamespace(cache_root=tmp_path), _box_chips=lambda spec: 4,
                           _block=lambda stage, why, **ev: blocked_with.append(why))
    Supervisor._check_residency(stub, SimpleNamespace(boards=1, number=2, draft="scoring_check"))
    assert blocked_with and "chips" in blocked_with[0]


# ---- review of PR 435, round 4: a real response is read in single receives, so the deadline is checked after each one -------------------------

def test_the_deep_nesting_fixtures_reach_the_parser_and_are_not_stopped_by_the_byte_cap():
    done = in_limited_child(f"{DEEP}\n{OPENER}from orchard import preflight as pf\n"
                            "r = R(); got = pf._read_capped(r, pf.CONFIG_FETCH_BYTES, lambda: 0.0)\n"
                            "print(len(got) if got is not None else None, len(body))")
    assert done.returncode == 0 and done.stdout.split()[0] == done.stdout.split()[1] != "None", done.stdout + done.stderr[-300:]


def _drip_http_response(tick_state):
    """A real http.client.HTTPResponse over a raw stream that gives its header at once and then one byte per receive, moving a fake clock one
    second per receive. `read(n)` on it makes one receive after another until n bytes have come."""
    import http.client
    import io

    class Raw(io.RawIOBase):
        def __init__(self):
            self.head = b"HTTP/1.1 200 OK\r\nContent-Length: 100000000\r\n\r\n"

        def readable(self):
            return True

        def readinto(self, b):
            if self.head:
                n = min(len(b), len(self.head))
                b[:n] = self.head[:n]
                self.head = self.head[n:]
                return n
            tick_state["receives"] += 1
            tick_state["now"] += 1.0
            b[0:1] = b"x"
            return 1

    class Sock:
        def makefile(self, *a, **k):
            return io.BufferedReader(Raw(), buffer_size=1 << 16)

    resp = http.client.HTTPResponse(Sock())
    resp.begin()
    return resp


def test_a_real_response_that_drips_a_byte_per_receive_is_cut_off_near_the_total_deadline():
    st = {"receives": 0, "now": 0.0}
    resp = _drip_http_response(st)
    assert pf._read_capped(resp, pf.CONFIG_FETCH_BYTES, lambda: st["now"]) is None
    assert st["receives"] <= int(pf.HUB_TOTAL_S) + 3, st
