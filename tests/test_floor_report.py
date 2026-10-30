# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Decode speed as a share of the DRAM-bandwidth floor in stages 0, 2 and 4 and the operator bundle (orchard/floor_report.py, issue 181).

The floor arithmetic is tested in tests/test_decode_floor.py. These tests check which model config, dtype policy, die count and measured
speed are put together, and that each report says what it is an estimate of."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from orchard import decode_floor as df
from orchard import floor_report as fr
from orchard import model_config as mc

FIX = Path(__file__).parent / "fixtures" / "model-configs"
DENSE = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 320, "torch_dtype": "bfloat16"}


def snap(tmp, cfg, name="m"):
    d = tmp / name
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


# ---- the dtype policy ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("torch_dtype,want", [("bfloat16", "bf16"), ("float16", "fp16"), ("float32", "fp32")])
def test_the_policy_is_the_dtype_the_config_stores_the_weights_in(torch_dtype, want):
    policy, source = fr.policy_for(dict(DENSE, torch_dtype=torch_dtype))
    assert policy == {"default": want} and "torch_dtype" in source


@pytest.mark.parametrize("cfg", [{}, {"torch_dtype": "float8_e4m3fn"}, {"torch_dtype": 5}, {"torch_dtype": ["bfloat16"]}])
def test_an_unstated_or_unknown_dtype_is_bf16_and_the_source_says_it_was_assumed(cfg):
    policy, source = fr.policy_for(cfg)
    assert policy == {"default": "bf16"} and "assumed" in source


def test_a_multimodal_config_reads_the_dtype_from_its_text_config():
    policy, _ = fr.policy_for({"text_config": {"torch_dtype": "float16"}})
    assert policy == {"default": "fp16"}


# ---- stage 0 ---------------------------------------------------------------------------------------------------------------

def test_stage_0_records_the_floor_for_one_two_and_four_dies_with_every_input(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    rec = fr.write_floor(stage, model_dir=snap(tmp_path, DENSE))
    assert json.loads((stage / fr.FLOOR_FILE).read_text()) == rec
    assert rec["scan"] == "ran" and rec["label"] == "estimate" and rec["board"] == "p300"
    for dies in (1, 2, 4):
        f = df.decode_floor(DENSE, {"default": "bf16"}, dies, df.BOARDS["p300"].per_die_gb_s)
        got = rec["floors"][str(dies)]
        assert got["floor_tok_s"] == pytest.approx(f.floor_tok_s) and got["inputs"]["dies"] == dies
        assert got["inputs"]["per_die_bandwidth_gb_s"] == 512.0 and got["inputs"]["dtype_policy"] == {"default": "bf16"}
    assert rec["floors"]["4"]["floor_tok_s"] == pytest.approx(2 * rec["floors"]["2"]["floor_tok_s"])
    assert "datasheet" in rec["bandwidth_source"] and rec["assumptions"]


def test_stage_0_floor_for_a_moe_counts_only_active_experts():
    cfg = mc.read_model_config(FIX / "qwen3-coder-next.json")
    rec = fr.build_record(model_config=cfg)
    dense_equiv = rec["floors"]["2"]["bytes_per_token"]
    every_expert = rec["floors"]["2"]["stored_params"] * 2.0
    assert dense_equiv < every_expert / 5                   # a 512-expert model reads a small part of what it stores


@pytest.mark.parametrize("make", ["missing", "hostile", "zero"])
def test_a_model_the_floor_cannot_count_is_recorded_as_skipped_and_never_raises(tmp_path, make):
    stage = tmp_path / "stage"
    stage.mkdir()
    d = tmp_path / "m"
    d.mkdir()
    if make == "hostile":
        (d / "config.json").write_text("{nope")
    elif make == "zero":
        (d / "config.json").write_text(json.dumps({"hidden_size": 0}))
    rec = fr.write_floor(stage, model_dir=d)
    assert rec["scan"] == "skipped" and rec["floors"] == {} and rec["reason"]


def test_a_layer_type_the_floor_cannot_count_is_skipped_with_the_reason(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    cfg = dict(DENSE, layer_types=["full_attention", "warp_drive"])
    rec = fr.write_floor(stage, model_dir=snap(tmp_path, cfg))
    assert rec["scan"] == "skipped" and "warp_drive" in rec["reason"]


def test_the_model_directory_comes_from_the_triage_config_when_the_stage_has_one(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "triage_config.json").write_text(json.dumps({"model_snapshot": str(snap(tmp_path, DENSE))}))
    assert fr.write_floor(stage)["scan"] == "ran"


# ---- stages 2 and 4: the measured speeds, from the raw records --------------------------------------------------------------

def probe(seconds=2.0, tokens=96):
    from test_sampling_probe import record, entry
    return record(greedy=entry(seconds=seconds, tokens=tokens))


def run_dir(tmp_path, *, floor=True, chips2=None, stage4=(), graft=None, cfg=DENSE):
    run = tmp_path / "run"
    if floor:
        (run / "stages/0").mkdir(parents=True)
        fr.write_floor(run / "stages/0", model_dir=snap(tmp_path, cfg))
    write(run / "stages/2/evidence/swap-check.json", {"sampling_probe": probe()})
    if chips2:
        write(run / "stages/2/swap_config.json", {"chips": chips2})
    for n in stage4:
        write(run / f"stages/4/configs/{n}/evidence/swap-check.json", {"sampling_probe": probe(seconds=1.0)})
        write(run / f"stages/4/configs/{n}/swap_config.json", {"chips": n})
    if graft:
        write(run / "stages/2/graft-decision.json", {"choice": "graft", "reason": "x", "numbers": graft})
    return run


def test_the_stage_2_probe_speed_is_a_share_of_the_floor_for_the_boards_dies(tmp_path):
    run = run_dir(tmp_path)
    rows = fr.share_rows(run)
    assert len(rows) == 1
    r = rows[0]
    floor = fr.read_floor(run)["floors"]["2"]["floor_tok_s"]
    assert r["tok_s"] == pytest.approx(48.0) and r["dies"] == 2 and r["floor_tok_s"] == pytest.approx(floor)
    assert r["share"] == pytest.approx(48.0 / floor) and r["evidence"] == "stages/2/evidence/swap-check.json"
    assert "includes" in r["basis"] and r["stage"] == 2


def test_stage_4_rows_use_their_own_chip_count_as_the_die_count(tmp_path):
    run = run_dir(tmp_path, stage4=(4,))
    by_stage = {(r["stage"], r["dies"]): r for r in fr.share_rows(run)}
    assert by_stage[(4, 4)]["tok_s"] == pytest.approx(96.0)
    assert by_stage[(4, 4)]["floor_tok_s"] == pytest.approx(2 * by_stage[(2, 2)]["floor_tok_s"])


def test_a_stage_2_config_that_names_its_chips_sets_the_die_count(tmp_path):
    assert fr.share_rows(run_dir(tmp_path, chips2=1))[0]["dies"] == 1


def test_the_graft_trial_speeds_are_rows_and_the_variant_row_says_a_drafter_can_pass_the_floor(tmp_path):
    run = run_dir(tmp_path, graft={"safe_decode_tok_s": 20.0, "variant_decode_tok_s": 60.0})
    rows = {r["source"]: r for r in fr.share_rows(run)}
    safe = next(v for k, v in rows.items() if "safe" in k)
    var = next(v for k, v in rows.items() if "graft" in k and "safe" not in k)
    assert safe["tok_s"] == 20.0 and var["tok_s"] == 60.0 and "drafter" in var["basis"]


def test_a_share_above_the_floor_says_an_input_is_wrong_or_the_run_decodes_several_tokens_per_read(tmp_path):
    run = run_dir(tmp_path, graft={"safe_decode_tok_s": 1e9})
    row = [r for r in fr.share_rows(run) if r["tok_s"] == 1e9][0]
    assert row["share"] > 1 and "above the floor" in row["note"]


@pytest.mark.parametrize("bad", [None, "x", True, -1, 0, float("nan"), float("inf")])
def test_a_speed_that_is_not_a_positive_finite_number_makes_no_row(tmp_path, bad):
    run = run_dir(tmp_path, graft={"safe_decode_tok_s": bad, "variant_decode_tok_s": bad})
    assert [r["stage"] for r in fr.share_rows(run)] == [2]          # only the probe row


def test_a_run_with_no_floor_record_or_a_skipped_one_has_no_rows(tmp_path):
    assert fr.share_rows(run_dir(tmp_path, floor=False)) == []
    run = run_dir(tmp_path)
    write(run / "stages/0" / fr.FLOOR_FILE, {"scan": "skipped", "reason": "x", "floors": {}})
    assert fr.share_rows(run) == []


def test_a_die_count_the_stage_0_record_has_no_floor_for_makes_no_row_and_the_report_says_so(tmp_path):
    run = run_dir(tmp_path, stage4=(3,))
    assert [r["dies"] for r in fr.share_rows(run) if r["stage"] == 4] == []
    assert any("3 dies" in n for n in fr.share_notes(run))


def test_a_hostile_floor_record_is_refused_without_raising(tmp_path):
    run = run_dir(tmp_path)
    for bad in ('{"scan": "ran", "floors": {"2": {"floor_tok_s": "fast"}}}', '{"scan": "ran", "floors": {"2": {"floor_tok_s": 0}}}',
                '{"scan": "ran", "floors": []}', "[]", "{nope"):
        write(run / "stages/0" / fr.FLOOR_FILE, bad)
        assert fr.share_rows(run) == [], bad


def test_a_hostile_swap_check_makes_no_row_and_never_raises(tmp_path):
    run = run_dir(tmp_path)
    for bad in ("{nope", "[]", '{"sampling_probe": 5}', '{"sampling_probe": {"ran": true, "requests": "x"}}'):
        write(run / "stages/2/evidence/swap-check.json", bad)
        assert fr.share_rows(run) == [], bad


# ---- the hint ---------------------------------------------------------------------------------------------------------------

def test_a_share_below_the_stated_threshold_gets_the_hint_with_the_causes_in_order(tmp_path):
    run = run_dir(tmp_path)
    low = fr.share_rows(run, threshold=1.0)[0]                      # any real share is below 100 percent
    assert low["hint"] and low["hint"].index("op count") < low["hint"].index("weight streaming") < low["hint"].index("host round trips")
    assert fr.share_rows(run, threshold=1e-9)[0]["hint"] is None


def test_the_default_threshold_is_stated_and_a_choice(tmp_path):
    assert 0 < fr.LOW_SHARE_THRESHOLD < 1 and "choice" in fr.LOW_SHARE_SOURCE


# ---- the supervisor writes the stage records --------------------------------------------------------------------------------

def test_write_share_writes_the_file_with_the_inputs_and_the_estimate_label(tmp_path):
    run = run_dir(tmp_path, stage4=(2,))
    rec2 = fr.write_share(run, run / "stages/2", 2)
    assert json.loads((run / "stages/2" / fr.SHARE_FILE).read_text()) == rec2
    assert rec2["label"] == "estimate" and [r["stage"] for r in rec2["rows"]] == [2] and rec2["assumptions"]
    rec4 = fr.write_share(run, run / "stages/4", 4)
    assert [r["stage"] for r in rec4["rows"]] == [4]


def test_write_share_for_a_stage_with_nothing_to_report_writes_nothing(tmp_path):
    run = run_dir(tmp_path, floor=False)
    assert fr.write_share(run, run / "stages/2", 2) is None
    assert not (run / "stages/2" / fr.SHARE_FILE).exists()


def test_stage_0_of_a_supervised_run_writes_the_floor_and_the_ledger_records_it(tmp_path):
    from test_supervisor_stock import Rig, ends
    r = Rig(tmp_path)
    try:
        r.run()
        es = r.entries()
        rec = json.loads((r.run_dir / "stages/0" / fr.FLOOR_FILE).read_text())
        assert rec["scan"] == "ran" and set(rec["floors"]) == {"1", "2", "4"}
        ev = [x for x in es if x["event"] == "evidence" and x["stage"] == 0 and x["data"].get("what") == "decode floor"]
        assert len(ev) == 1 and ev[0]["data"]["path"] == f"stages/0/{fr.FLOOR_FILE}"
        assert ends(es)[0]["result"] == "pass"
    finally:
        r.close()


# ---- the bundle ------------------------------------------------------------------------------------------------------------

def bundle(tmp_path, *, floor=True, threshold_low=False):
    from test_operator_bundle_template import build, bundle_of, make_run, section
    run = make_run(tmp_path)
    if floor:
        fr.write_floor(run / "stages/0", model_dir=snap(tmp_path, DENSE))
        write(run / "stages/2/evidence/swap-check.json", {"sampling_probe": probe()})
    r = build(run)
    assert r.returncode == 0, r.stdout + r.stderr
    b = bundle_of(run)
    results = (b / "RESULTS.md").read_text()
    return (section(results, "## Decode against the DRAM floor") if "## Decode against the DRAM floor\n" in results else None), results


def test_the_bundle_shows_each_speed_with_its_floor_share_inputs_and_the_estimate_label(tmp_path):
    sec, _ = bundle(tmp_path)
    assert sec is not None
    assert "estimate" in sec and "48" in sec and "stages/2/evidence/swap-check.json" in sec
    assert "GB/s" in sec and "bf16" in sec and "stages/0/decode-floor.json" in sec
    assert "datasheet" in sec and "includes" in sec          # the bandwidth source and the speed's basis are stated


def test_a_low_share_in_the_bundle_gets_the_hint_text(tmp_path):
    sec, _ = bundle(tmp_path)
    assert "op count" in sec and "host round trips" in sec    # the dense test model's floor is far above 48 tok/s


def test_a_run_without_a_floor_record_gets_the_same_results_as_before(tmp_path):
    sec, results = bundle(tmp_path, floor=False)
    assert sec is None and "DRAM floor" not in results


def test_the_supervisor_writes_decode_share_when_stages_2_and_4_pass_and_records_it(tmp_path):
    from test_supervisor_stock import Rig, _supervisor
    r = Rig(tmp_path)
    try:
        r.run()                                          # stage 0 wrote the floor record for the snapshot
        write(r.run_dir / "stages/2/evidence/swap-check.json", {"sampling_probe": probe()})
        write(r.run_dir / "stages/4/configs/4/evidence/swap-check.json", {"sampling_probe": probe(seconds=1.0)})
        sup, led = _supervisor(r)
        try:
            sup._decode_floor_record(2, r.run_dir / "stages/2")
            sup._decode_floor_record(4, r.run_dir / "stages/4")
            ev = [e for e in led.read() if e["event"] == "evidence" and e["data"].get("what") == "decode share"]
        finally:
            led.__exit__(None, None, None)
        assert [e["stage"] for e in ev] == [2, 4] and ev[0]["data"]["path"] == "stages/2/decode-share.json"
        assert json.loads((r.run_dir / "stages/4" / fr.SHARE_FILE).read_text())["rows"][0]["dies"] == 4
    finally:
        r.close()


def test_a_record_that_says_it_was_skipped_gives_no_rows_even_when_it_holds_floors(tmp_path):
    run = run_dir(tmp_path)
    rec = fr.read_floor(run)
    write(run / "stages/0" / fr.FLOOR_FILE, {**rec, "scan": "skipped"})
    assert fr.read_floor(run) is None and fr.share_rows(run) == []


def test_a_speed_just_above_the_floor_is_flagged_and_one_below_is_not(tmp_path):
    run = run_dir(tmp_path)
    floor = fr.read_floor(run)["floors"]["2"]["floor_tok_s"]
    write(run / "stages/2/graft-decision.json", {"choice": "graft", "reason": "x",
                                                  "numbers": {"safe_decode_tok_s": floor * 1.5, "variant_decode_tok_s": floor * 0.5}})
    rows = {r["source"]: r for r in fr.share_rows(run)}
    assert "above the floor" in rows["stage 2 graft trial, safe configuration"]["note"]
    assert rows["stage 2 graft trial, grafted variant"]["note"] is None


def test_stage_2_and_stage_4_passing_each_write_their_share_through_the_real_end_of_stage(tmp_path):
    from test_supervisor_stock import Rig, _supervisor
    from orchard.stages import GateResult, spec_for
    r = Rig(tmp_path)
    try:
        r.run()
        write(r.run_dir / "stages/2/evidence/swap-check.json", {"sampling_probe": probe()})
        write(r.run_dir / "stages/4/configs/4/evidence/swap-check.json", {"sampling_probe": probe(seconds=1.0)})
        for n in (2, 4):
            write(r.run_dir / f"stages/{n}/gate.json", {})
        sup, led = _supervisor(r)
        try:
            for n in (2, 4):
                spec = spec_for(n, "weights-only")
                stage = r.run_dir / "stages" / str(n)
                (stage / spec.gate_file).write_text("{}") if spec.gate_file and not (stage / spec.gate_file).exists() else None
                sup._end_stage(spec, stage, "pass", [], GateResult(True, (), ()), False)
        finally:
            led.__exit__(None, None, None)
        for n in (2, 4):
            assert (r.run_dir / "stages" / str(n) / fr.SHARE_FILE).is_file(), n
    finally:
        r.close()
