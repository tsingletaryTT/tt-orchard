# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7 for a vocabulary trim (issue 56): the package serves the model with the drafter off and the card says why."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchard.package import finish, read_run  # noqa: E402
from orchard.stages import gate_package  # noqa: E402
from package_fakes import fake_boot_result, write  # noqa: E402
from test_package_graft import stage, world  # noqa: E402,F401


def make_trim(w, vocab=None):
    """Stage 2 recorded a vocabulary trim; the model ships its own mtp.* tensors (the CJK-Trim case), so the drafter is off for
    the vocabulary and not for the missing head."""
    delta = json.loads((w["run"] / "stages/0/delta.json").read_text())
    write(w["run"] / "stages/0/delta.json", {**delta, "mtp": {"nearest_tensors": 3, "new_tensors": 15}})
    write(w["run"] / "stages/2/model-dir-sources.json", {"schema": 1, "vocab": vocab or {"model": 181760, "base": 248320, "trimmed": True}})


def finished(w):
    stage_dir, package = stage(w)
    fake_boot_result(stage_dir)
    finish(w["run"], stage_dir)
    return stage_dir, w["run"] / package["profiles"][0]["dir"]


def flat(path: Path) -> str:
    return " ".join(path.read_text().split())


def test_the_run_facts_carry_the_vocabulary_of_a_trim(world):
    make_trim(world)
    assert read_run(world["run"]).vocab == (181760, 248320)


@pytest.mark.parametrize("record", [None, {"model": 5, "base": 9, "trimmed": False}, {"model": 5, "base": 9}, {"model": True, "base": 9, "trimmed": True},
                                    {"model": 9, "base": 5, "trimmed": True}, "x", {"model": 5, "base": 9, "trimmed": "yes"}])
def test_anything_but_a_recorded_trim_gives_no_vocabulary(world, record):
    write(world["run"] / "stages/2/model-dir-sources.json", {"schema": 1, **({"vocab": record} if record is not None else {})})
    assert read_run(world["run"]).vocab is None


def test_the_package_card_of_a_trim_gives_the_vocabulary_as_the_reason_and_the_gate_passes(world):
    make_trim(world)
    stage_dir, pkg = finished(world)
    c = flat(pkg / "README.md")
    assert "181,760" in c and "248,320" in c and "token ids are renumbered" in c
    assert "needs the MTP tensors of the model" not in c
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_the_gate_refuses_the_old_sentence_on_a_trim_package(world):
    make_trim(world)
    stage_dir, pkg = finished(world)
    card = (pkg / "README.md").read_text()
    lines = card.splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.startswith("- Speculative decoding is off."))
    lines[at] = ("- Speculative decoding is off. The drafter of Qwen/Qwen3.8-27B needs the MTP tensors of the model, and "
                 "ProprietaryLegal/Qwen3.8-27B-CJK-Trim has none.")
    (pkg / "README.md").write_text("\n".join(lines) + "\n")
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any("says the model has no mtp.* tensors" in r for r in reasons), reasons


def test_the_gate_refuses_a_trim_card_that_leaves_the_sizes_out(world):
    make_trim(world)
    stage_dir, pkg = finished(world)
    (pkg / "README.md").write_text((pkg / "README.md").read_text().replace("181,760", "REDACTED"))
    assert any("181,760" in r for r in gate_package(stage_dir, world["run"]).reasons)


def test_a_package_for_a_model_that_is_not_a_trim_has_no_vocabulary_text(world):
    stage_dir, pkg = finished(world)
    assert "token ids are renumbered" not in flat(pkg / "README.md")
    assert gate_package(stage_dir, world["run"]).ok
