# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fakes for the scoring check template (orchard/skills/scoring-check-templates/scoring_check.py): a fixed pair set whose reference score is
written into the last token id, and port modules (one honest, several that miss a bound or try to forge the record). The fake `ttnn` is
the one in port_check_fakes. Nothing here touches a chip."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from orchard import scoring_check
from port_check_fakes import CHIPS, install_fake_ttnn, python_path

BASE = [3.0, 0.5, -1.0, -2.5]                       # the reference scores of one group's four documents, before the per-group shift
N_GROUPS = 8
KINDS = ["short", "short", "multilingual", "multilingual", "code", "code", "long", "long"]


def score_of(group: int, doc: int) -> float:
    return (round(10 * (BASE[doc] + 0.1 * group))) / 10


def pair_rows() -> list[dict]:
    rows = []
    for g in range(N_GROUPS):
        for d in range(4):
            rows.append({"group": f"g{g}", "kind": KINDS[g], "query": "q", "document": "d",
                         "input_ids": [g + 1, d + 1, 100 + round(10 * score_of(g, d))]})
    return rows


REFERENCE_SCORES = [score_of(g, d) for g in range(N_GROUPS) for d in range(4)]
CHIP_NOT_WRITTEN = object()

# An honest port: the score is encoded in the last id, so it reproduces the reference exactly without reading any reference file.
HONEST = '''
import ttnn
def load(model_dir, mesh_device):
    return {"dir": model_dir, "device": mesh_device}
def score_pairs(model, pairs):
    return [(ids[-1] - 100) / 10 for ids in pairs]
def close(model):
    pass
'''
# Reverses the order of every group: its scores are the negatives of the reference.
REVERSED = HONEST.replace("(ids[-1] - 100) / 10", "-(ids[-1] - 100) / 10")
# Right order and right correlation, but every score is off by 0.4 in alternating directions: mean difference 0.4, above the bound.
NOISY = HONEST.replace("(ids[-1] - 100) / 10 for ids in pairs", "(ids[-1] - 100) / 10 + (0.4 if k % 2 else -0.4) for k, ids in enumerate(pairs)")
# One constant score for every pair.
CONSTANT = HONEST.replace("(ids[-1] - 100) / 10 for ids in pairs", "1.0 for ids in pairs")


def make_run(tmp_path: Path, port_src: str = HONEST, *, extra_files: dict | None = None, template_text: str | None = None,
             config: dict | None = None, rows=None, scores=None):
    """A run directory with the stage 1 pair file and reference, stages/2/port/port_model.py, and the template and its config copied the way
    the supervisor copies them. Returns (run_dir, stage_dir, snapshot_dir)."""
    run = tmp_path / "run"
    ev = run / "stages" / "1" / "evidence" / "reference"
    ev.mkdir(parents=True)
    rows = pair_rows() if rows is None else rows
    pairs_bytes = (json.dumps({"schema": "orchard-scoring-pairs/1", "model_kind": "reranker", "pairs": rows}) + "\n").encode()
    (ev / "scoring-pairs.json").write_bytes(pairs_bytes)
    ref = {"schema": "orchard-scoring-reference/1", "model_kind": "reranker", "precision": "fp32",
           "pairs": {"file": "scoring-pairs.json", "sha256": hashlib.sha256(pairs_bytes).hexdigest(), "count": len(rows)},
           "scores": REFERENCE_SCORES if scores is None else scores}
    (ev / "scoring-reference.json").write_text(json.dumps(ref))
    snap = tmp_path / "snapshot"
    snap.mkdir()
    stage = run / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text(port_src)
    for rel, text in (extra_files or {}).items():
        p = stage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    if template_text is None:
        shutil.copy2(scoring_check.TEMPLATE, stage / scoring_check.TEMPLATE_NAME)
    else:
        (stage / scoring_check.TEMPLATE_NAME).write_text(template_text)
    (stage / scoring_check.CONFIG_NAME).write_text(json.dumps(config if config is not None else scoring_check.draft_config(str(snap))))
    install_fake_ttnn(tmp_path)
    return run, stage, snap


def run_template(tmp_path: Path, run: Path, stage: Path, *, chips=CHIPS, command: str = scoring_check.COMMAND):
    """Run the supervisor's command in the run directory the way the supervisor does, and leave what it leaves: the output file and
    test-result.json."""
    cmd = command.split()
    cmd[0] = sys.executable
    env = {"PATH": os.environ.get("PATH", ""), "TT_VISIBLE_DEVICES": ",".join(chips), "PYTHONPATH": python_path(tmp_path)}
    done = subprocess.run(cmd, cwd=run, env=env, capture_output=True, text=True, timeout=120)
    write_marker(run, stage, done, chips, command)
    return done


def write_marker(run: Path, stage: Path, done, chips, command: str = scoring_check.COMMAND) -> None:
    from orchard import full_port
    out = stage / "evidence" / "hw-test-output.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(done.stdout + done.stderr)
    marker = {"command": command, "returncode": done.returncode, "timed_out": False, "seconds": 1.0, "chips": list(chips),
              "output": full_port.file_record(run, out)}
    (stage / "test-result.json").write_text(json.dumps(marker))
