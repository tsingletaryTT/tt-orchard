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


# ---- round 3 -------------------------------------------------------------------------------------------------------------------------

def test_the_mutation_script_finds_the_text_of_every_row_it_mutates():
    """Finding: the `seeded-hashes` row searched for an old call, matched nothing, and the default run always ended with status 1."""
    import importlib.util
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("mut6", root / "scripts" / "mutations" / "full_port_stage_6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = []
    for name, row in mod.MUTATIONS.items() if hasattr(mod, "MUTATIONS") else []:
        path, old = row[0], row[1]
        if (root / path).read_text(encoding="utf-8").count(old) != 1:
            missing.append(name)
    assert hasattr(mod, "MUTATIONS") and not missing, missing


def _child():
    import subprocess
    import sys
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_a_descendant_whose_start_time_differs_is_not_signalled():
    """Finding: only pids were kept, so a pid freed and reused between the scan and the kill could take SIGKILL. The start time recorded in the
    scan is checked against the live process first."""
    mod = load_template()
    child = _child()
    try:
        started = mod.proc_stat(child.pid)[1]
        assert mod.kill_if_same(child.pid, started + 1) is False
        assert child.poll() is None, "a process with another start time was killed"
        assert mod.kill_if_same(child.pid, started) is True
        child.wait(timeout=10)
        assert child.returncode == -9
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_kill_descendants_reads_each_childs_start_time_in_the_scan(monkeypatch):
    mod = load_template()
    child = _child()
    seen = []
    try:
        w = mod.Worker.__new__(mod.Worker)
        w.proc = type("P", (), {"pid": os.getpid()})()
        monkeypatch.setattr(mod, "kill_if_same", lambda pid, started: seen.append((pid, started)))
        w.kill_descendants()
        assert (child.pid, mod.proc_stat(child.pid)[1]) in seen
    finally:
        child.kill()
        child.wait()


def test_a_port_that_rewrites_the_model_config_while_it_runs_fails_the_test(tmp_path):
    """Finding: the config hash was first taken after the worker finished, so a changed config.json was accepted as the model's."""
    from port_check_fakes import HONEST
    from quality_fakes import make_run, run_template
    port = HONEST.replace("def load(model_dir, mesh_device):\n",
                          "def load(model_dir, mesh_device):\n    import pathlib\n"
                          "    c = pathlib.Path(model_dir) / 'config.json'\n    c.write_text(c.read_text().replace('64', '65'))\n", 1)
    run, stage, snap, seeded = make_run(tmp_path, port)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "config.json changed while the port ran" in done.stdout, done.stdout + done.stderr
    assert not (stage / "evidence" / "port-quality.json").exists()
