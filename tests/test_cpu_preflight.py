# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `cpu` row of the bringup preflight: does the CPU stand-in answer? (autonomy, 2026-10-18)

Stage 4's 4-chip test parks the coder and a CPU-tier model takes its place. On the RAG-Gate-27B run ollama was not
running, the park's canary failed with `coder-unusable` three hours in, and a person had to start it. The row moves
that failure to the start of the run. The signal is injected here; one test runs the real probe against a local fake."""
from __future__ import annotations

from orchard import preflight as pf
from fake_model import FakeModel
from test_preflight import TIERS, cfg, signals

ENDPOINT, MODEL = "http://127.0.0.1:11434/v1", "qwen3-coder:30b"


def test_a_stand_in_that_answers_is_ok_and_names_the_model():
    c = pf.check_cpu_standin(None, ENDPOINT, MODEL)
    assert c.status == pf.OK and MODEL in c.detail and ENDPOINT in c.detail


def test_a_stand_in_that_does_not_answer_blocks_as_coder_unusable_and_says_how_to_start_it():
    c = pf.check_cpu_standin("connection refused", ENDPOINT, MODEL)
    assert (c.status, c.reason) == (pf.BLOCK, "coder-unusable")
    assert "connection refused" in c.detail and "ollama serve" in c.detail and "OLLAMA_MODELS" in c.detail


def test_the_row_comes_after_the_port_row_and_asks_for_the_cpu_tiers_endpoint_and_model(tmp_path):
    asked = []
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(cpu_standin=lambda e, m: asked.append((e, m)) or None))
    names = [c.name for c in out]
    assert names.index("cpu") == names.index("port") + 1
    cpu_tier = next(t for t in TIERS.values() if t["placement"] == "cpu")
    assert asked == [(cpu_tier["endpoint"], cpu_tier["model"])]
    assert not pf.blocked(out)


def test_a_dead_stand_in_blocks_the_run_before_it_starts(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(cpu_standin=lambda e, m: "connection refused"))
    assert [c.reason for c in pf.blocked(out)] == ["coder-unusable"]


def test_a_probe_that_raises_is_a_block_and_never_an_exception(tmp_path):
    def boom(e, m):
        raise OSError("no route")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(cpu_standin=boom))
    row = next(c for c in out if c.name == "cpu")
    assert row.status == pf.BLOCK and "could not check" in row.detail


def test_without_the_signal_there_is_no_row(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert "cpu" not in [c.name for c in out]


def test_the_real_probe_reads_the_servers_model_list():
    with FakeModel(lambda r: {}, models=[MODEL, "other"]) as srv:
        assert pf.cpu_standin_problem(srv.endpoint, MODEL) is None
        missing = pf.cpu_standin_problem(srv.endpoint, "not-pulled:1b")
        assert missing and "not-pulled:1b" in missing and "ollama pull" in missing


def test_the_real_probe_reports_a_closed_port():
    problem = pf.cpu_standin_problem("http://127.0.0.1:9/v1", MODEL)
    assert problem and "refused" in problem.lower()


def test_the_default_signals_carry_the_real_probe(tmp_path):
    assert pf.default_signals(cfg(tmp_path)).cpu_standin is pf.cpu_standin_problem


# ---- a lab box: the coder is never parked, so the stand-in is never used (PR 11 review, finding 1) ----

def test_in_lab_mode_the_stand_in_is_not_asked_for_because_a_lab_run_never_parks(tmp_path):
    from pathlib import Path
    from orchard.bringup_config import Lab
    c = cfg(tmp_path)
    c.lab = Lab(host="node4", root=Path("/srv/orchard"))
    c.mode = "lab"
    asked = []
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False,
                           signals=signals(cpu_standin=lambda e, m: asked.append((e, m)) or "connection refused"))
    assert not asked and "cpu" not in [x.name for x in out]
    assert "coder-unusable" not in [x.reason for x in pf.blocked(out)]


def test_a_local_mode_config_with_a_lab_table_still_checks_the_stand_in(tmp_path):
    from pathlib import Path
    from orchard.bringup_config import Lab
    c = cfg(tmp_path)
    c.lab = Lab(host="node4", root=Path("/srv/orchard"))
    c.mode = "local"
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False,
                           signals=signals(cpu_standin=lambda e, m: "connection refused"))
    assert [x.reason for x in pf.blocked(out)] == ["coder-unusable"]
