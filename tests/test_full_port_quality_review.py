# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 6 of a full port: the findings of the Copilot review of PR 491. Each test names the finding."""
import importlib.util
import os
import threading
from pathlib import Path

import pytest

from orchard import full_port_quality

README = Path(__file__).resolve().parent.parent / "README.md"


def load_template():
    spec = importlib.util.spec_from_file_location("port_quality_under_test", full_port_quality.TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bare_worker(mod):
    """A Worker with two private pipes in place of a worker process: `replies` is where the 'worker' writes, `requests` is where it reads."""
    w = mod.Worker.__new__(mod.Worker)
    r_reply, w_reply = os.pipe()
    r_req, w_req = os.pipe()
    w.rd_fd, w._buf = r_reply, b""
    w.rd = os.fdopen(os.dup(r_reply), "r", encoding="utf-8")          # the earlier reader, so a test of the old way fails on its behaviour
    w.wr = os.fdopen(w_req, "w", encoding="utf-8")
    return w, w_reply, r_req


def test_worker_ask_waits_for_a_reply_without_starting_a_thread_for_each_request():
    """Finding: `Worker.ask` started a thread per request for the timeout, and a timed-out request left its thread blocked on the pipe."""
    mod = load_template()
    w, w_reply, r_req = bare_worker(mod)
    before = threading.active_count()
    seen = []
    for i in range(5):
        os.write(w_reply, b'{"ok": true, "id": %d}\n' % i)
        seen.append(w.ask(op="decode", n=i)["id"])
        seen.append(threading.active_count())
    assert seen[0::2] == [0, 1, 2, 3, 4]
    assert max(seen[1::2]) == before, "a request started a thread"


def test_worker_ask_that_times_out_raises_and_leaves_no_thread_behind(monkeypatch):
    mod = load_template()
    monkeypatch.setattr(mod, "REPLY_TIMEOUT_S", 0.2)
    w, w_reply, r_req = bare_worker(mod)
    before = threading.active_count()
    with pytest.raises(mod.QualityFailed, match="did not answer"):
        w.ask(op="prefill")
    assert threading.active_count() == before, "the timed-out request left a blocked thread"


def test_worker_ask_reads_a_reply_that_arrives_in_pieces(monkeypatch):
    mod = load_template()
    w, w_reply, r_req = bare_worker(mod)
    os.write(w_reply, b'{"ok": tr')
    threading.Timer(0.1, lambda: os.write(w_reply, b'ue, "id": 7}\n')).start()
    assert w.ask(op="decode")["id"] == 7


def test_worker_ask_treats_a_closed_pipe_as_an_exited_worker():
    mod = load_template()
    w, w_reply, r_req = bare_worker(mod)
    os.close(w_reply)
    with pytest.raises(mod.QualityFailed, match="did not answer"):
        w.ask(op="decode")


def test_the_readme_has_one_full_port_class_row_and_it_does_not_say_stages_5_or_6_are_unbuilt():
    """Finding: the table had two `full-port` rows; one said stages 4 and 5 are not built."""
    rows = [l for l in README.read_text().splitlines() if l.startswith("| `full-port` | The model needs new model code")]
    assert len(rows) == 1, f"{len(rows)} rows"
    assert "are not built" not in rows[0] and "Stage 5" in rows[0] and "Stage 6" in rows[0] and "Stage 4" in rows[0]


# ---- round 2: the six open findings of the review of PR 491 ----------------------------------------------------------------------------

from quality_fakes import check as quality_check, derive_and_write  # noqa: E402
from test_full_port_quality import good  # noqa: E402
from orchard import full_port, full_port_bundle  # noqa: E402


def _has(reasons, text):
    return any(text in r for r in reasons)


def test_a_source_record_replaced_with_its_port_after_the_source_stage_passed_is_refused(tmp_path):
    """Finding: the source stage's port-check.json was trusted without checking its hash against the passing ledger entry."""
    run, stage, seeded = good(tmp_path)
    derive_and_write(run, stage)
    accepted = full_port._sha(run / "stages" / "2" / full_port.RECORD_FILE)
    assert quality_check(run, stage, seeded, accepted_record_sha=accepted)[0] == []
    # the record is replaced after the source stage passed (the port file would be replaced in step); the ledger's hash no longer matches
    rec = run / "stages" / "2" / full_port.RECORD_FILE
    rec.write_text(rec.read_text() + "\n")
    reasons = quality_check(run, stage, seeded, accepted_record_sha=accepted)[0]
    assert _has(reasons, "is not the file its gate accepted")


def test_no_ledger_hash_for_the_source_record_is_a_refusal_not_a_skip(tmp_path):
    run, stage, seeded = good(tmp_path)
    derive_and_write(run, stage)
    assert _has(quality_check(run, stage, seeded, accepted_record_sha=None)[0], "holds no sha256 for stage 2")


def test_a_record_with_only_warm_up_runs_is_refused_before_it_is_summarised(tmp_path):
    """Finding: a record of warm-up runs only passed `record_problems`, and `summarise` then indexed an empty list (IndexError)."""
    run, stage, seeded = good(tmp_path)
    import json
    rec_path = stage / full_quality_record_name()
    rec = json.loads(rec_path.read_text())
    for r in rec["bench"]["runs"]:
        r["warmup"] = True
    rec_path.write_text(json.dumps(rec))
    result, why = full_port_quality.derive(stage, run, source_stage=2)          # must return a reason, never raise
    assert result is None and "no measured run" in why


def full_quality_record_name():
    return full_port_quality.RECORD_FILE


def test_the_host_load_average_of_each_measured_run_reaches_the_result(tmp_path):
    """Finding: the projection dropped host_loadavg_1m, so the result held a list of nulls."""
    run, stage, seeded = good(tmp_path)
    import json
    rec_path = stage / full_port_quality.RECORD_FILE
    rec = json.loads(rec_path.read_text())
    measured = [r for r in rec["bench"]["runs"] if not r["warmup"]]
    for i, r in enumerate(measured):
        r["host_loadavg_1m"] = 1.5 + i
    rec_path.write_text(json.dumps(rec))
    result, why = full_port_quality.derive(stage, run, source_stage=2)
    assert result is not None, why
    assert result["benchmark"]["host_loadavg_1m"] == [1.5 + i for i in range(len(measured))]


def test_the_bundle_labels_the_bandwidth_floor_input_as_a_datasheet_value(tmp_path):
    """Finding: the bandwidth input was printed with the label (measured), though this stage measures no bandwidth."""
    run, stage, seeded = good(tmp_path)
    result, why = derive_and_write(run, stage)
    assert result is not None, why
    text = "\n".join(full_port_bundle.section(result, lambda s: s))
    bw = result["benchmark"]["floor"]["inputs"]["per_die_bandwidth_gb_s"]
    assert f"{bw:.4g} (datasheet value) GB/s per die" in text
    assert f"{bw:.4g} (measured) GB/s" not in text


def test_the_assumed_bf16_floor_warning_says_the_share_is_overstated():
    """Finding: smaller weight formats raise the true floor, so dividing by the lower assumed floor overstates the share."""
    res = {"schema": full_port_quality.RESULT_SCHEMA, "coherent": True, "benchmark": {"dtype_policy_source": "assumed bf16: the port declares none"}}
    text = "\n".join(full_port_bundle.risks(res, lambda s: s))
    assert "overstated" in text and "understated" not in text
