# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`full_port._output_reasons` reads the captured output once (review of PR 498, round 5). The hash that the supervisor's test record must match and
the mark line that must match the check's record come from the same bytes, so a file replaced between two reads cannot pass with altered text."""
import hashlib
import json

from orchard import full_port, safeio

GOOD_SHA = "a" * 64
OTHER_SHA = "b" * 64


def _setup(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "evidence").mkdir(parents=True)
    out = stage / full_port.OUTPUT_FILE
    out.write_bytes(f"loaded\n{full_port.OUTPUT_MARK} {GOOD_SHA}\n".encode())
    marker = {"output": {"path": str(out.relative_to(run)), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}}
    return run, stage, out, marker


def test_an_output_that_is_valid_for_its_record_passes(tmp_path):
    run, stage, out, marker = _setup(tmp_path)
    reasons = []
    full_port._output_reasons(stage, run, marker, reasons, record_sha=GOOD_SHA)
    assert reasons == []


def test_an_output_replaced_after_its_first_read_does_not_authenticate_altered_text(tmp_path, monkeypatch):
    """Finding: the file was hashed and then opened again for its mark line, so a replacement between the two reads could carry a mark that
    matches an altered record while the hash still matched the supervisor's test record."""
    run, stage, out, marker = _setup(tmp_path)
    real = safeio.read_bytes
    state = {"reads": 0}

    def swapping(path, **kw):
        data = real(path, **kw)
        if str(path) == str(out):
            state["reads"] += 1
            if state["reads"] == 1:       # whatever the first reader saw, the next reader gets text that names the altered record
                out.write_bytes(f"loaded\n{full_port.OUTPUT_MARK} {OTHER_SHA}\n".encode())
        return data
    monkeypatch.setattr(safeio, "read_bytes", swapping)
    reasons = []
    full_port._output_reasons(stage, run, marker, reasons, record_sha=OTHER_SHA)       # the altered record's digest
    assert reasons, "the replaced output authenticated an altered record"
    assert state["reads"] == 1                                                         # the output is read once
