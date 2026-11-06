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
