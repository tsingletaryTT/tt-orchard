# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 records the op inventory in delta.json when it can: record_in_delta, the module entry point, the gate and the skill."""
import json
import os
from pathlib import Path

from orchard import opinventory as oi
from orchard.stages import gate_delta
from run_fakes import TEMPLATE_SHA, TEMPLATE_TEXT

SKILL = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage.md"

DELTA = {
    "model": "Altworld/Hemmingway-1", "nearest_model": "Qwen/Qwen3.8-27B", "path": "weights-only",
    "differences": [{"area": "config", "finding": "equal", "evidence": ["stages/0/evidence/diff.txt"]}],
    "hazards": [{"area": "disk", "finding": "big"}],
    "extra": {"nested": [1, 2, {"a": None}]},
}


def inv_json(ops, **kw):
    distinct = {s: 0 for s in oi.STATUSES}
    for o in ops:
        distinct[o["status"]] += 1
    d = {"traced_class": "Qwen3ForCausalLM", "ops": ops, "summary": {"distinct": distinct},
         "ttnn_dump": {"ttnn_version": "0.1", "sha256": "ab"}}
    d.update(kw)
    return d


def stage_dir(tmp_path, delta=DELTA, inventory=None):
    sd = tmp_path / "stages" / "0"
    sd.mkdir(parents=True)
    if delta is not None:
        (sd / "delta.json").write_text(json.dumps(delta))
    if inventory is not None:
        (sd / "ops-inventory.json").write_text(json.dumps(inventory))
    return sd


def delta_of(sd):
    return json.loads((sd / "delta.json").read_text())


def test_record_merges_the_summary_and_keeps_every_other_key(tmp_path):
    ops = [{"op": "aten.mm", "scaled_count": 5, "status": "mapped"},
           {"op": "aten.weird", "scaled_count": 2, "status": "unmapped"}]
    sd = stage_dir(tmp_path, inventory=inv_json(ops))
    assert oi.record_in_delta(sd) is True
    got = delta_of(sd)
    inv = got.pop("inventory")
    assert got == DELTA
    assert inv == {"traced_class": "Qwen3ForCausalLM", "skipped": None,
                   "distinct": {"mapped": 1, "composable": 0, "verified": 0, "candidate": 0, "unmapped": 1,
                                "unchecked": 0, "ignored": 0},
                   "unmapped_ops": ["aten.weird"], "candidate_ops": [], "patterns": [], "ttnn_dump": {"ttnn_version": "0.1", "sha256": "ab"}}
    assert [p.name for p in sd.iterdir() if p.name.startswith(("tmp", ".")) or p.suffix == ".tmp"] == []


def test_unmapped_ops_are_at_most_ten_most_frequent_first_and_only_unmapped(tmp_path):
    ops = [{"op": f"aten.u{i}", "scaled_count": i, "status": "unmapped"} for i in range(15)]
    ops.append({"op": "aten.big", "scaled_count": 999, "status": "mapped"})
    sd = stage_dir(tmp_path, inventory=inv_json(ops))
    oi.record_in_delta(sd)
    names = delta_of(sd)["inventory"]["unmapped_ops"]
    assert names == [f"aten.u{i}" for i in range(14, 4, -1)]


def test_a_missing_file_returns_false_and_changes_nothing(tmp_path):
    sd = stage_dir(tmp_path)                                   # no inventory
    before = (sd / "delta.json").read_bytes()
    assert oi.record_in_delta(sd) is False
    assert (sd / "delta.json").read_bytes() == before
    sd2 = tmp_path / "other"
    sd2.mkdir()
    (sd2 / "ops-inventory.json").write_text(json.dumps(inv_json([])))
    assert oi.record_in_delta(sd2) is False and not (sd2 / "delta.json").exists()


def test_a_file_that_is_not_json_returns_false_and_changes_nothing(tmp_path):
    sd = stage_dir(tmp_path, inventory=inv_json([]))
    (sd / "ops-inventory.json").write_text("{nope")
    before = (sd / "delta.json").read_bytes()
    assert oi.record_in_delta(sd) is False and (sd / "delta.json").read_bytes() == before
    (sd / "ops-inventory.json").write_text(json.dumps(inv_json([])))
    (sd / "delta.json").write_text("{nope")
    assert oi.record_in_delta(sd) is False and (sd / "delta.json").read_text() == "{nope"


def test_a_skipped_inventory_records_only_the_reason(tmp_path):
    sd = stage_dir(tmp_path, inventory={"traced_class": "X", "skipped": "needs remote code", "ops": [],
                                        "summary": {}, "ttnn_dump": None})
    assert oi.record_in_delta(sd) is True
    inv = delta_of(sd)["inventory"]
    assert inv["skipped"] == "needs remote code"
    assert "distinct" not in inv and "unmapped_ops" not in inv


def test_the_gate_accepts_an_inventory_key_and_still_needs_its_old_fields(tmp_path):
    sd = stage_dir(tmp_path, delta={**DELTA, "template_sha256": TEMPLATE_SHA["delta_triage.py"]},
                   inventory=inv_json([{"op": "a", "scaled_count": 1, "status": "unmapped"}]))
    (sd / "delta_triage.py").write_text(TEMPLATE_TEXT["delta_triage.py"])        # the gate wants an unedited copy and its stamp
    run = tmp_path
    (run / "stages/0/evidence").mkdir()
    (run / "stages/0/evidence/diff.txt").write_text("e")
    oi.record_in_delta(sd)
    assert "inventory" in delta_of(sd)
    g = gate_delta(sd, run)
    assert g.ok, g.reasons
    broken = delta_of(sd)
    del broken["nearest_model"]
    (sd / "delta.json").write_text(json.dumps(broken))
    g = gate_delta(sd, run)
    assert not g.ok and any("nearest_model" in r for r in g.reasons)


def test_the_entry_point_runs_the_inventory_then_records_it(tmp_path, monkeypatch):
    sd = stage_dir(tmp_path)
    seen = {}

    def fake_run_inventory(model_dir, **kw):
        seen.update(model_dir=model_dir, **kw)
        (Path(kw["out_dir"]) / "ops-inventory.json").write_text(json.dumps(inv_json([])))
        return 0

    monkeypatch.setattr(oi, "run_inventory", fake_run_inventory)
    rc = oi.main(["--model-dir", "/m", "--out", str(sd), "--python", "py", "--ttnn-python", "tpy",
                  "--layers", "3", "--seq", "16", "--class", "C"])
    assert rc == 0 and "inventory" in delta_of(sd)
    assert (seen["model_dir"], seen["python"], seen["ttnn_python"], seen["layers"], seen["seq"], seen["cls"]) == (
        "/m", "py", "tpy", 3, 16, "C")
    assert str(seen["out_dir"]) == str(sd)


def test_the_entry_point_returns_the_inventory_exit_code_and_records_nothing_on_failure(tmp_path, monkeypatch):
    sd = stage_dir(tmp_path)
    monkeypatch.setattr(oi, "run_inventory", lambda *a, **k: 2)
    assert oi.main(["--model-dir", "/m", "--out", str(sd), "--python", "py"]) == 2
    assert "inventory" not in delta_of(sd)


def test_the_entry_point_defaults_ttnn_python_to_the_shared_venv_under_the_real_home(tmp_path, monkeypatch):
    """I2. With no --ttnn-python every op was `unchecked`. The default is the shared venv, as `tt-orchard inventory`
    uses. It is under the home from the passwd entry: in an agent shell HOME points inside the run directory."""
    import pwd
    sd = stage_dir(tmp_path)
    seen = {}
    monkeypatch.setattr(oi, "run_inventory", lambda m, **k: seen.update(k) or 2)
    monkeypatch.setenv("HOME", str(tmp_path / "agent-home"))
    oi.main(["--model-dir", "/m", "--out", str(sd), "--python", "py"])
    real = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert seen["ttnn_python"] == str(real / ".tenstorrent-venv" / "bin" / "python")
    assert seen["layers"] == 2 and seen["seq"] == 8 and seen["cls"] is None


def test_the_skill_runs_the_inventory_when_reference_python_is_an_input():
    flat = " ".join(SKILL.read_text().split())
    assert "python3 -m orchard.opinventory" in flat
    assert "ops_inventory.py" in flat and "reference_python" in flat
    assert "{{ORCHARD_DIR}}" in flat
    assert "/home/" not in flat and "/mnt/" not in flat
    # I2: without the shared venv's interpreter the inventory cannot read the installed ttnn's version
    assert "--ttnn-python {{OPERATOR_HOME}}/.tenstorrent-venv/bin/python" in flat
    assert "ops-inventory.md" in flat
    assert "skip this step" in flat and "Keep the `inventory` key" in flat
    assert "measured class still decides" in flat
    assert "`unmapped`" in flat and "`weights-only`" in flat
