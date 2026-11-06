# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The supervisor's side of stage 6 of a full port (issue 329): which stage's port is measured, the template and config copied into the stage
directory, `result.json` written from the record, and the gate with the arguments only the supervisor knows. The checks themselves are in
orchard/full_port_quality.py. These are functions that take the supervisor as their first argument, so the supervisor file gains only a few
call lines.

Stage 6 has no agent step. The port is the one that passed the earlier stage, copied by the supervisor (`Supervisor._seed_stage_3_port`, which
stage 3 also uses); the check is the shipped template; the result is derived from the template's record.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

from orchard import full_port, full_port_quality, full_port_serve, safeio
from orchard.defaults import CHIPS_PER_BOARD
from orchard.stages import full_port_rehearsal

DERIVED_BY = "supervisor"
RUN_JSON_MAX = full_port.JSON_MAX


def source_stage(sup) -> int:
    """The stage whose port stage 6 measures: stage 3 when it ran on the configured chips, else stage 2 (its one board is the configured count)."""
    return 3 if sup._stage_3_chips() else 2


def expected_chips(sup) -> int:
    """The chip count the stage 6 test must run on: the one stage 3 ran on, or one board's chips."""
    return sup._stage_3_chips() or CHIPS_PER_BOARD


def seeded(sup) -> list:
    """The `files` list ({path, sha256}) of the newest `stage N port copied` decision of the current attempt of stage 6, from the ledger. An
    attempt starts at a stage_start that is not a resume. [] when there is none."""
    want = f"stage {source_stage(sup)} port copied"
    files: list = []
    for e in sup.ledger.read():
        if e.get("stage") != full_port_quality.STAGE:
            continue
        data = e.get("data") or {}
        if e["event"] == "stage_start" and not data.get("resumed"):
            files = []
        elif e["event"] == "decision" and data.get("decision") == want and isinstance(data.get("files"), list):
            files = data["files"]
    return files


def top5_accepted(sup) -> str | None:
    """The sha256 the ledger holds for stage 1's top-five file among the evidence stage 1's gate accepted, or None."""
    return full_port_serve.accepted_sha(sup.ledger.read(), 1, full_port_quality.TOP5_REL)


def gate(sup, spec):
    """The stage's gate with the arguments that come from the supervisor and the ledger, never from a file the stage directory holds."""
    source = source_stage(sup)
    accepted = full_port_serve.accepted_sha(sup.ledger.read(), source, f"stages/{source}/{full_port.RECORD_FILE}")
    kwargs = {"expected_chips": expected_chips(sup), "seeded": seeded(sup), "source_stage": source, "accepted_record_sha": accepted,
              "top5_accepted_sha": top5_accepted(sup)}
    if full_port_rehearsal(sup.ledger.read()):
        kwargs["rehearsal"] = True
    return functools.partial(spec.gate, **kwargs)


def pin(sup, spec, stage_dir: Path) -> list[str]:
    """Put the shipped template and its config in the stage directory (replacing any copy), and write the fixed hw_test.json. Returns the problems
    that stop the stage (no model snapshot in the run). The ledger holds the sha256 of each file as written."""
    snapshot = sup.inputs.get("model")
    if not snapshot:
        return ["the run has no model snapshot (inputs `model=`), so the quality check has nothing to load"]
    # Before the test runs: the top-five file on disk is the one stage 1's gate accepted (the gate checks again after the test).
    top5 = full_port._inside(sup.run_dir, full_port_quality.TOP5_REL)
    want = top5_accepted(sup)
    if not want:
        return [f"the ledger holds no sha256 for {full_port_quality.TOP5_REL} among stage 1's accepted evidence, so the gap bar has no reference"]
    if top5 is None or full_port._sha(top5) != want:
        return [f"{full_port_quality.TOP5_REL} is not the file stage 1's gate accepted"]
    shipped = Path(sup.paths.orchard_dir) / "orchard" / "skills" / "full-port-templates" / full_port_quality.TEMPLATE_NAME
    template = safeio.read_bytes(shipped, max_bytes=RUN_JSON_MAX)
    config = (json.dumps({"model_snapshot": str(Path(snapshot).absolute())}, indent=2) + "\n").encode("utf-8")
    hw = (json.dumps({"command": full_port_quality.command(), "deadline_s": full_port_quality.DEADLINE_S}) + "\n").encode("utf-8")
    safeio.atomic_write(stage_dir / full_port_quality.TEMPLATE_NAME, template, mode=os.stat(shipped).st_mode & 0o777)
    safeio.atomic_write(stage_dir / full_port_quality.CONFIG_NAME, config, mode=0o644)
    safeio.atomic_write(stage_dir / "hw_test.json", hw, mode=0o644)
    note = {"goal": f"measure the quality and speed of the port that passed stage {source_stage(sup)}", "stage": spec.number,
            "evidence": [f"stages/{spec.number}/{full_port_quality.RECORD_FILE}"],
            "next_action": "none: the supervisor writes result.json from the record",
            "check_on_return": "the test exit code and the record's PORT_QUALITY_SHA256 line"}
    safeio.atomic_write(stage_dir / "handoff.json", (json.dumps(note) + "\n").encode("utf-8"), mode=0o644)         # the hardware phase's own note (`_read_test`)
    sup.ledger.append("decision", spec.number, decision="port quality template copied",
                      copied={full_port_quality.TEMPLATE_NAME: hashlib.sha256(template).hexdigest(),
                              full_port_quality.CONFIG_NAME: hashlib.sha256(config).hexdigest()})
    return []


def derive(sup, spec, stage_dir: Path) -> tuple[bool, str]:
    """Write result.json from the record when the test passed and the record is complete, run the stage's own gate on it, and say so in the
    ledger. (True, "") when the file stands. (False, why) otherwise: the file is deleted and the reason says what was missing or refused."""
    n = spec.number
    rehearsal = full_port_rehearsal(sup.ledger.read())
    result, why = full_port_quality.derive(stage_dir, sup.run_dir, source_stage=source_stage(sup), rehearsal=rehearsal)
    if result is None:
        sup.ledger.append("decision", n, decision="result not derived by the supervisor", why=why[:300])
        return False, why
    path = stage_dir / "result.json"
    try:
        safeio.atomic_write(path, (json.dumps(result, indent=2) + "\n").encode("utf-8"), mode=0o644)
    except OSError as exc:
        why = f"result.json could not be written ({exc.__class__.__name__})"
        sup.ledger.append("decision", n, decision="result not derived by the supervisor", why=why)
        return False, why
    checked = gate(sup, spec)(stage_dir, sup.run_dir)       # the pure gate; `_check_gate` adds the ledger checks and the waiver decision
    if not checked.ok:
        path.unlink(missing_ok=True)
        why = "the derived result did not pass the gate's check: " + "; ".join(list(checked.reasons)[:3])
        sup.ledger.append("decision", n, decision="result not derived by the supervisor", why=why[:300])
        return False, why
    found = full_port.file_record(sup.run_dir, path) or {}
    sup.ledger.append("evidence", n, what="result.json derived by the supervisor", **found)
    bench, ppl = result["benchmark"], result["perplexity"]
    sup.ledger.append("decision", n, decision="result derived by the supervisor", file=f"stages/{n}/result.json",
                      source_stage=result["source_stage"], perplexity=ppl["perplexity"], top5_gap_diff=ppl["top5_gap_diff"],
                      leading_match=result["free_run"]["leading_match"], coherent=result["coherent"], benchmark_label=bench["label"],
                      repeats=bench["repeats"], share_of_floor_median=bench["share_of_floor_median"])
    return True, ""
