# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The helper-driven repair: the findings of the second Copilot round of PR 497. Each test names the finding."""
import pytest

from orchard import port_repair
from orchard import repair_help as rh
from test_helper_drives_port import _stage_with_port
from test_helper_drives_review import _passed_check_stub


def _bests(s):
    return [e["data"] for e in s.ledger.read() if e["data"].get("decision") == rh.BEST]


def test_a_passing_port_with_a_supporting_data_file_is_kept_at_the_passing_rank(tmp_path):
    """Finding: the record lists only Python sources, the snapshot holds every file, so a port with converted_cache.json was always refused."""
    s, stage = _passed_check_stub(tmp_path, vouched=False)
    (stage / "port" / "converted_cache.json").write_text("{}")
    # the supervisor's own pre-test manifest and evidence entry, with the cache file already in place
    import hashlib
    marker = (stage / port_repair.MARKER).read_bytes()
    s.ledger.append("decision", 2, decision=port_repair.PORT_AT_TEST, sources=port_repair.fingerprint(stage / "port"))
    s.ledger.append("evidence", 2, path="stages/2/" + port_repair.MARKER, sha256=hashlib.sha256(marker).hexdigest(), returncode=0, timed_out=False)
    s._record_best_passed(2, stage)
    (best,) = _bests(s)
    assert best["rank"] == rh.PASSED_RANK and "converted_cache.json" in [f["path"] for f in best["files"]]


def test_a_supporting_file_edited_after_the_test_is_still_refused(tmp_path):
    s, stage = _passed_check_stub(tmp_path, vouched=True)
    (stage / "port" / "converted_cache.json").write_text("{}")              # added after the manifest was written
    s._record_best_passed(2, stage)
    assert not _bests(s)


def test_a_port_edited_while_the_supervisor_was_stopped_is_not_saved_at_the_failed_checks_rank(tmp_path):
    """Finding: after an edit made between the failed check and the resume, the untested files were saved at the old failure's rank."""
    s, stage = _passed_check_stub(tmp_path, vouched=True, fails_at="prefill")
    s._record_best(2, stage, {"op": "prefill"})
    assert len(_bests(s)) == 1                                              # the control: the files the check ran on are kept
    s2, stage2 = _passed_check_stub(tmp_path / "other", vouched=True, fails_at="prefill")
    (stage2 / "port" / "helper.py").write_text("y = 3  # edited while the supervisor was stopped\n")
    s2._record_best(2, stage2, {"op": "prefill"})
    assert not _bests(s2)
    assert any("not the one the failed check ran on" in e["data"].get("reason", "") for e in s2.ledger.read())


def test_a_restore_interrupted_after_a_supporting_file_can_be_retried(tmp_path, monkeypatch):
    """Finding: the supporting file written before the interruption made the retry refuse with 'already exists'."""
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"import helper\n", **{"helper.py": b"X = 1\n"})
    entry = rh.save_best(stage)
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    (stage / "port" / "helper.py").unlink()
    real = rh.safeio.write_beneath_if
    state = {"fail": True}

    def interrupted(fd, parts, data, **kw):
        if parts == ("port_model.py",) and state["fail"]:
            state["fail"] = False
            raise rh.safeio.ReadFailed("interrupted")
        return real(fd, parts, data, **kw)
    monkeypatch.setattr(rh.safeio, "write_beneath_if", interrupted)
    placed, _ = rh.restore_best(stage, entry, skeleton)
    assert placed is False and (stage / "port" / "helper.py").read_bytes() == b"X = 1\n"
    assert rh.restore_best(stage, entry, skeleton) == (True, "")
    assert (stage / "port" / "port_model.py").read_bytes() == b"import helper\n"


def test_a_supporting_file_with_other_bytes_still_stops_the_restore(tmp_path):
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"import helper\n", **{"helper.py": b"X = 1\n"})
    entry = rh.save_best(stage)
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    (stage / "port" / "helper.py").write_bytes(b"X = 99  # the coder's own\n")
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "already exists" in why
    assert (stage / "port" / "helper.py").read_bytes() == b"X = 99  # the coder's own\n"


def test_a_call_reported_not_helped_and_then_helped_is_served_from_the_cache_again(tmp_path):
    """Finding: cache eligibility counted every historical negative report, while the summary and the model choice use each call's latest one."""
    from test_telegraph import make
    tel, wire = make(tmp_path)
    first = tel.ask("p", "t", "f1")
    tel.report(first.call_id, helped=False)
    tel.report(first.call_id, helped=True)
    again = tel.ask("p", "t", "f1")
    assert again.cached is True and len(wire.requests) == 1


def test_a_call_whose_latest_report_is_not_helped_is_still_asked_again(tmp_path):
    from test_telegraph import make
    tel, wire = make(tmp_path)
    first = tel.ask("p", "t", "f1")
    tel.report(first.call_id, helped=True)
    tel.report(first.call_id, helped=False)
    tel.ask("p", "t", "f1")
    assert len(wire.requests) == 2


# ---- review of PR 497, round 5 ---------------------------------------------------------------------------------------------------

def test_the_rank_of_a_best_port_comes_from_the_authenticated_failure_and_not_the_one_passed_in(tmp_path):
    """Finding: an earlier test's marker and output replayed after a rerun gave the current port that earlier failure's rank, because the
    failure handed to `_record_best` was read without matching the record to the ledger's newest test."""
    s, stage = _passed_check_stub(tmp_path, vouched=True, fails_at="load")      # the newest test, vouched by the ledger, failed at load
    s._record_best(2, stage, {"op": "decode"})                                   # a replayed older record says the check got as far as decode
    (best,) = _bests(s)
    assert best["op"] == "load" and best["rank"] == rh.best_rank("load") < rh.best_rank("decode")


def test_a_failed_check_the_ledger_never_vouched_for_records_no_best_port(tmp_path):
    s, stage = _passed_check_stub(tmp_path, vouched=False, fails_at="load")
    # The ledger holds a test for these files, but its evidence entry names another record: the marker on disk is a replay of an older one.
    s.ledger.append("decision", 2, decision=port_repair.PORT_AT_TEST, sources=port_repair.fingerprint(stage / "port"))
    s.ledger.append("evidence", 2, path="stages/2/" + port_repair.MARKER, sha256="0" * 64, returncode=2, timed_out=False)
    s._record_best(2, stage, {"op": "decode"})
    assert not _bests(s)
