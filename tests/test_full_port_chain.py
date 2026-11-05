# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The whole full-port route in one run, stages 0 to 8, with the chip, the coder and the model as fakes (the integration of stages 2 to 6).

Each stage's own supervisor tests keep the neighbouring stages generic, so they cannot show that the stages fit together: a ledger decision one
stage writes and the next reads, a seed that one stage changes and another relies on, a pause left behind by a stage that has since been built.
This run has none of the generic stand-ins: stage 5 serves the port and replays the corpus, stage 6 measures quality and the benchmark, both for
real on the CPU against the fake `ttnn`, and the run ends ready, with no pause for an unbuilt stage."""
import json

import pytest

from orchard import full_port_serve
from orchard.supervisor import EXIT_READY, REHEARSAL_WAIVED
import port_serve_fakes as fk
from quality_fakes import TINY_CONFIG
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, STAGE_1, make, with_snapshot

pytestmark = pytest.mark.usefixtures("stub_tools")

KEYS = [{"length": 256, "key": "48213"}, {"length": 1984, "key": "90417"}]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    (tmp_path / "snapshot" / "config.json").write_text(json.dumps(TINY_CONFIG))        # stage 6 reads the model's size for the decode floor
    r.args.full_port = True
    monkeypatch.setattr(full_port_serve, "plan_passkeys", lambda corpus_sha, max_len: [dict(k) for k in KEYS])
    fk.tokenizer_with(tmp_path / "snapshot" / "tokenizer.json", [k["key"] for k in KEYS])
    yield r
    r.close()


def go(rig):
    """The scripted coder passes stage 2 with a port that answers the passkey; stage 1 carries the top-five file stage 6 compares with. Stages 5 and
    6 ask no coder. A pause is a failure: the run stops there."""
    src = fk.port_source(fk.ANSWERING, rig.tmp / "snapshot")
    make(rig, over={(1, "run"): STAGE_1, (2, "prepare"): {**PREPARE, "port/port_model.py": src}})
    from orchard.supervisor import Control
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


def results(rig):
    return {n: [d["result"] for d in rig.ends(n)] for n in range(9)}


def test_a_full_port_run_goes_through_every_stage_and_ends_ready(rig):
    go(rig)
    assert rig.run() == EXIT_READY, [(d.get("decision"), d.get("reason") or d.get("code")) for d in rig.decisions() if d.get("decision") in ("pause", "blocked")]
    res = results(rig)
    assert res[0] == ["pass"] and res[1] == ["pass"] and res[2] == ["pass"]
    assert res[3] == ["skipped"] and res[4] == ["skipped"]          # no `--required-chips`: stage 2 already ran the port on the configured board
    assert res[5] == ["pass"] and res[6] == ["pass"]                 # the real stage 5 (chat server and corpus) and the real stage 6 (quality and benchmark)
    assert res[7] and res[8], res
    pauses = [d for d in rig.decisions() if d.get("decision") == "pause"]
    assert not pauses, pauses                                       # no stage is left unbuilt
    assert not [d for d in rig.decisions() if d.get("code") == "full-port-stages-unbuilt"]


def test_each_stage_reads_what_the_stage_before_it_wrote(rig):
    go(rig)
    rig.run()
    run = rig.run_dir
    assert (run / "stages" / "5" / "result.json").is_file() and (run / "stages" / "6" / "result.json").is_file()
    six = json.loads((run / "stages" / "6" / "result.json").read_text())
    five = json.loads((run / "stages" / "5" / "result.json").read_text())
    assert six["derived_by"] == five["derived_by"] == "supervisor" and six["source_stage"] == 2
    copied = [d for d in rig.decisions() if d.get("decision") == "stage 2 port copied"]
    assert copied, "stage 6's gate reads the newest 'stage N port copied' decision of its own attempt"


def test_the_chain_with_chips_configured_runs_stage_3_on_four_and_stage_4_covers_both_counts(rig):
    from test_supervisor_full_port_stage3 import PREPARE_3
    rig.args.required_chips = (2, 4)
    src = fk.port_source(fk.ANSWERING, rig.tmp / "snapshot")
    make(rig, over={(1, "run"): STAGE_1, (2, "prepare"): {**PREPARE, "port/port_model.py": src}, (3, "prepare"): {**PREPARE_3, "port/port_model.py": src}})
    from orchard.supervisor import Control
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    assert rig.run() == EXIT_READY, [(d.get("decision"), d.get("reason") or d.get("code")) for d in rig.decisions() if d.get("decision") in ("pause", "blocked")]
    res = results(rig)
    assert res[3] == ["pass"] and res[4] == ["pass"] and res[5] == ["pass"] and res[6] == ["pass"], res
    five = json.loads((rig.run_dir / "stages" / "5" / "result.json").read_text())
    six = json.loads((rig.run_dir / "stages" / "6" / "result.json").read_text())
    assert six["source_stage"] == 3 and len(json.loads((rig.run_dir / "stages" / "6" / "evidence" / "port-quality.json").read_text())["chips"]) == 4
    assert five["derived_by"] == "supervisor"
