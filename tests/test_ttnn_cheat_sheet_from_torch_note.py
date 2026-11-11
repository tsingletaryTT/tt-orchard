# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 563: the TTNN cheat sheet's note for `ttnn.from_torch` said that on a mesh without a mesh_mapper the tensor lands on one device. Run 20
of the tiny Qwen4 rehearsal (stage 2, attempt 2) called from_torch with no mapper on a 1x2 mesh and its prefill failed with "Can't convert a
tensor distributed on MeshShape([1, 2]) mesh to row-major logical tensor. Supply a mesh composer": the tensor was on the mesh. What omitting
the mapper does (replicate, shard or something else) has not been measured; the issue stays open for a chip run. The note must not claim
the one-device behaviour, must name the composer needed to read a mesh tensor back, and must say what is unmeasured."""
import json
from pathlib import Path

SHEET = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "full-port-templates" / "ttnn_cheat_sheet.json"


def from_torch_entry():
    sheet = json.loads(SHEET.read_text(encoding="utf-8"))
    for section in sheet["sections"]:
        for e in section.get("entries", []):
            if e.get("signature", "").startswith("ttnn.from_torch("):
                return e
    raise AssertionError("the sheet has no ttnn.from_torch entry")


def test_the_note_does_not_claim_a_tensor_without_a_mapper_lands_on_one_device():
    note = from_torch_entry()["notes"]
    assert "lands on one device" not in note and "one device" not in note


def test_the_note_names_the_composer_a_mesh_tensor_needs_when_read_back():
    note = from_torch_entry()["notes"]
    assert "mesh_composer" in note and "distributed" in note


def test_the_note_says_the_replication_was_measured_and_points_at_the_issue():
    # The first version of the note said replication was "not measured". The
    # lease measurement settled it, so the note must now say "measured" and must
    # not keep the old hedge.
    note = from_torch_entry()["notes"]
    assert "measured on 1 x 2" in note and "563" in note
    assert "not measured" not in note


def test_the_usage_line_still_passes_a_mapper_for_a_mesh():
    assert "mesh_mapper=ttnn.ReplicateTensorToMesh(" in from_torch_entry()["usage"]
