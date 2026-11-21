# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 check's position cap is a recorded setting (issue 731, from Compression-Bench).

The fidelity score needs about 4,000 scored tokens to separate bfp8 from bfp4 (9.991 against 9.821 on the Olmo-Hybrid-7B port), and the check
used to stop at a fixed 64 positions. `port_check_config.json` now carries an optional `max_positions` (the supervisor writes it from
`--port-check-positions`), the template reads and validates it, the record carries a `positions` block (the cap, the positions scored and
the stage 1 reference's length), and the gate recomputes the block. The default stays 64, so a run that does not name the option is what it
was. The fidelity block stays report-only."""
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from orchard import full_port
from orchard.stages import gate_full_port
from port_check_fakes import HONEST_LP, WRONG, make_run, run_template, write_result


def _load_template():
    """The template imported from a copy under a `stages/` directory, where it finds its run directory the way it does under the supervisor."""
    stage = Path(tempfile.mkdtemp(prefix="pc-pos-")) / "run" / "stages" / "2"
    stage.mkdir(parents=True)
    shutil.copy2(full_port.TEMPLATE, stage / full_port.TEMPLATE_NAME)
    spec = importlib.util.spec_from_file_location("port_check_positions_under_test", stage / full_port.TEMPLATE_NAME)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pc = _load_template()


def record(stage):
    return json.loads((stage / "evidence" / "port-check.json").read_text())


def write_record(stage, rec):
    (stage / "evidence" / "port-check.json").write_text(json.dumps(rec, indent=2) + "\n")


def set_config(stage, **extra):
    """The stage's config with the snapshot it already names and the keys in `extra`, before the template runs (so the record hashes it)."""
    path = stage / full_port.CONFIG_NAME
    path.write_text(json.dumps({**json.loads(path.read_text()), **extra}))


def longer_reference(run, n):
    """A stage 1 reference of `n` ids (the HONEST fake port predicts each id as the last one plus 1, up to its vocabulary of 200)."""
    ev = run / "stages" / "1" / "evidence" / "reference"
    ids = list(range(100, 100 + n))
    (ev / "generated-ids.json").write_text(json.dumps({"generated_ids": ids, "generated_text": "x"}))
    (ev / "top5-logits.json").write_text(json.dumps({"steps": [{"step": k, "chosen": t, "logprob": HONEST_LP, "top5": []}
                                                               for k, t in enumerate(ids)]}))


def ran(tmp_path, *, cap=None, reference=None):
    """Run the template for real (CPU, fake ttnn) with `max_positions` set to `cap` when given and a stage 1 reference of `reference` ids."""
    run, stage, _ = make_run(tmp_path)
    if reference is not None:
        longer_reference(run, reference)
    if cap is not None:
        set_config(stage, max_positions=cap)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    return run, stage


def positions_reasons(run, stage):
    write_result(stage)
    return [r for r in gate_full_port(stage, run).reasons if "positions" in r or "max_positions" in r]


# ---- the range ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [16, 64, 4000, 8192, "128"])
def test_check_positions_accepts_a_whole_number_in_range(value):
    assert full_port.check_positions(value) == int(value)


@pytest.mark.parametrize("value", [15, 8193, 0, -1, True, 64.0, "abc", "", None, "9" * 400])
def test_check_positions_refuses_everything_else_with_a_message_that_names_the_range(value):
    with pytest.raises(ValueError, match=r"from 16 to 8192"):
        full_port.check_positions(value)


def test_the_template_and_the_gate_use_the_same_range_and_default():
    assert pc.MAX_POSITIONS == full_port.DEFAULT_POSITIONS == 64
    assert pc.MIN_POSITIONS == full_port.MIN_POSITIONS and pc.POSITIONS_LIMIT == full_port.POSITIONS_LIMIT


def test_the_lowest_cap_is_the_fewest_positions_the_gate_accepts():
    assert full_port.MIN_POSITIONS == full_port.MIN_TOKENS


def test_a_record_at_the_highest_cap_stays_far_below_the_gates_json_limit():
    """The reason POSITIONS_LIMIT gives for its size: ids and log-probabilities of 8,192 positions, as the template writes them."""
    n = full_port.POSITIONS_LIMIT
    rec = {"reference_ids": list(range(100_000, 100_000 + n)), "chip_ids": list(range(100_000, 100_000 + n)),
           "fidelity": pc.fidelity_block([-1.2345678901234567] * n, [-1.2345678901234568] * n, cap=n, reference_length=n)}
    assert len(json.dumps(rec, indent=2)) < full_port.JSON_MAX // 8


# ---- the template reads the setting ---------------------------------------------------------------------------------------

def test_no_key_means_the_default_cap():
    assert pc.positions_cap({"model_snapshot": "x"}) == 64


def test_a_key_in_range_is_the_cap():
    assert pc.positions_cap({"max_positions": 200}) == 200


@pytest.mark.parametrize("value", [True, 15, 8193, 64.5, "64", None, 10 ** 400])
def test_a_key_out_of_range_or_of_the_wrong_type_is_a_setup_failure_that_names_the_key(value):
    with pytest.raises(pc.PortFailed) as caught:
        pc.positions_cap({"max_positions": value})
    assert caught.value.kind == "setup" and "max_positions" in str(caught.value) and len(str(caught.value)) < 300


def test_the_template_run_refuses_a_cap_below_the_range_before_it_loads_the_port(tmp_path):
    run, stage, _ = make_run(tmp_path)
    set_config(stage, max_positions=8)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "max_positions" in done.stdout and not (stage / "evidence" / "port-check.json").exists()


# ---- the record -----------------------------------------------------------------------------------------------------------

def test_a_run_with_no_option_records_the_default_cap_and_scores_the_whole_short_reference(tmp_path):
    run, stage = ran(tmp_path)
    assert record(stage)["positions"] == {"cap": 64, "scored": 32, "reference_length": 32}
    assert positions_reasons(run, stage) == [] and gate_full_port(stage, run).ok


def test_a_cap_below_the_reference_length_limits_the_positions_scored(tmp_path):
    run, stage = ran(tmp_path, cap=16)
    rec = record(stage)
    assert rec["positions"] == {"cap": 16, "scored": 16, "reference_length": 32}
    assert len(rec["chip_ids"]) == len(rec["reference_ids"]) == len(rec["fidelity"]["chip_logprobs"]) == 16
    assert rec["fidelity"]["tokens"] == 16 and positions_reasons(run, stage) == []
    assert gate_full_port(stage, run).ok


def test_a_cap_above_the_reference_length_scores_only_what_the_reference_holds(tmp_path):
    run, stage = ran(tmp_path, cap=128)
    assert record(stage)["positions"] == {"cap": 128, "scored": 32, "reference_length": 32}
    assert positions_reasons(run, stage) == []


def test_a_longer_reference_is_scored_up_to_the_default_cap_and_the_record_says_the_cap_stopped_it(tmp_path):
    run, stage = ran(tmp_path, reference=80)
    assert record(stage)["positions"] == {"cap": 64, "scored": 64, "reference_length": 80}
    assert positions_reasons(run, stage) == [] and gate_full_port(stage, run).ok


def test_a_raised_cap_scores_the_whole_longer_reference(tmp_path):
    run, stage = ran(tmp_path, cap=100, reference=80)
    rec = record(stage)
    assert rec["positions"] == {"cap": 100, "scored": 80, "reference_length": 80} and len(rec["chip_ids"]) == 80
    assert positions_reasons(run, stage) == [] and gate_full_port(stage, run).ok


# ---- the fidelity note names what limited the count --------------------------------------------------------------------------

def test_the_note_says_the_reference_was_the_limit_when_the_reference_is_shorter_than_the_cap(tmp_path):
    _, stage = ran(tmp_path)
    note = record(stage)["fidelity"]["note"]
    assert "only 32 scored tokens (< 4000)" in note and "reported, not gated" in note          # the earlier text is kept
    assert "stage 1 reference holds 32" in note and "--port-check-positions" not in note


def test_the_note_names_the_option_when_the_cap_was_the_limit(tmp_path):
    _, stage = ran(tmp_path, reference=80)
    note = record(stage)["fidelity"]["note"]
    assert "only 64 scored tokens (< 4000)" in note and "--port-check-positions" in note and "reported, not gated" in note


def test_a_score_of_4000_tokens_carries_no_note_and_still_does_not_gate():
    block = pc.fidelity_block([-1.0] * 4000, [-1.5] * 4000, cap=4000, reference_length=4000)
    assert "note" not in block and block["tokens"] == 4000 and block["value"] < 10


# ---- the gate recomputes the block -----------------------------------------------------------------------------------------

def tampered(tmp_path, change, **kw):
    run, stage = ran(tmp_path, **kw)
    rec = record(stage)
    change(rec)
    write_record(stage, rec)
    return positions_reasons(run, stage)


def test_the_gate_refuses_positions_scored_above_the_recorded_cap(tmp_path):
    reasons = tampered(tmp_path, lambda r: r["positions"].update(cap=16))                # 32 scored, cap 16
    assert any("more than its cap" in r for r in reasons), reasons


@pytest.mark.parametrize("cap", [4, 15, 8193, 10 ** 20, True, 64.0, "64", None])
def test_the_gate_refuses_a_cap_outside_the_range_or_not_a_whole_number(tmp_path, cap):
    reasons = tampered(tmp_path, lambda r: r["positions"].update(cap=cap))
    assert any("cap" in r and ("outside" in r or "whole number" in r) for r in reasons), reasons


def test_a_cap_too_large_to_read_as_an_integer_is_refused_with_the_whole_record(tmp_path):
    run, stage = ran(tmp_path)
    rec = record(stage)
    rec["positions"]["cap"] = 10 ** 400
    write_record(stage, rec)
    write_result(stage)
    got = gate_full_port(stage, run)
    assert not got.ok and any("not readable JSON" in r for r in got.reasons), got.reasons


def test_the_gate_refuses_a_scored_count_that_is_not_the_number_of_positions_the_record_compares(tmp_path):
    reasons = tampered(tmp_path, lambda r: r["positions"].update(scored=31))
    assert any("scored 31 differs from the 32 positions" in r for r in reasons), reasons


def test_the_gate_refuses_a_reference_length_that_is_not_the_stage_1_reference(tmp_path):
    reasons = tampered(tmp_path, lambda r: r["positions"].update(reference_length=500))
    assert any("reference_length 500 differs from the 32 ids in the stage 1 reference" in r for r in reasons), reasons


def test_the_gate_refuses_a_scored_count_that_is_not_the_smaller_of_the_cap_and_the_reference(tmp_path):
    # 80 reference ids, cap 100, 80 scored; the record claims a cap of 100 but only 64 scored and 64 ids (a quietly shortened record).
    def shorten(r):
        for key in ("reference_ids", "chip_ids"):
            r[key] = r[key][:64]
        r["fidelity"]["reference_logprobs"] = r["fidelity"]["reference_logprobs"][:64]
        r["fidelity"]["chip_logprobs"] = r["fidelity"]["chip_logprobs"][:64]
        r["fidelity"]["tokens"] = 64
        r["positions"]["scored"] = 64
    reasons = tampered(tmp_path, shorten, cap=100, reference=80)
    assert any("not the smaller of the cap 100 and the reference length 80" in r for r in reasons), reasons


def test_the_gate_refuses_a_cap_that_is_not_the_one_the_config_sets(tmp_path):
    reasons = tampered(tmp_path, lambda r: r["positions"].update(cap=64), cap=16)           # the run ran with 16; the record says 64
    assert any("max_positions 16" in r and "cap 64" in r for r in reasons), reasons


def test_the_gate_refuses_a_record_whose_block_was_removed_when_the_config_sets_a_cap(tmp_path):
    reasons = tampered(tmp_path, lambda r: r.pop("positions"), cap=16)
    assert any("no positions block" in r and "max_positions 16" in r for r in reasons), reasons


def test_the_gate_refuses_a_record_with_no_block_that_compares_more_than_the_default_cap(tmp_path):
    run, stage = ran(tmp_path, cap=100, reference=80)               # 80 positions scored under a cap of 100
    rec = record(stage)
    rec.pop("positions")
    write_record(stage, rec)
    cfg = json.loads((stage / full_port.CONFIG_NAME).read_text())
    cfg.pop("max_positions")                                         # and the config no longer names a cap: only the default can apply
    (stage / full_port.CONFIG_NAME).write_text(json.dumps(cfg))
    reasons = positions_reasons(run, stage)
    assert any("no positions block, so the default cap 64 applies, and it compares 80 positions" in r for r in reasons), reasons


def test_a_record_written_before_the_setting_existed_still_passes(tmp_path):
    reasons = tampered(tmp_path, lambda r: r.pop("positions"))                  # 32 positions, no key in the config
    assert reasons == [], reasons


def test_the_gate_refuses_a_block_that_is_not_an_object(tmp_path):
    reasons = tampered(tmp_path, lambda r: r.update(positions=[64, 32, 32]))
    assert any("positions must be an object" in r for r in reasons), reasons


def test_the_gate_refuses_a_block_with_other_keys_or_a_missing_key(tmp_path):
    reasons = tampered(tmp_path, lambda r: r["positions"].update(extra=1))
    assert any("positions must hold exactly" in r for r in reasons), reasons


def test_the_gate_refuses_a_block_with_a_missing_key(tmp_path):
    reasons = tampered(tmp_path, lambda r: r["positions"].pop("scored"))
    assert any("positions must hold exactly" in r for r in reasons), reasons


def test_a_low_fidelity_with_a_valid_block_adds_no_reason_so_the_score_stays_report_only(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG)                  # its probabilities for the reference tokens are low: the score is far below 10
    set_config(stage, max_positions=16)
    assert run_template(tmp_path, run, stage).returncode == 0
    rec = record(stage)
    assert rec["fidelity"]["value"] < 9 and rec["positions"] == {"cap": 16, "scored": 16, "reference_length": 32}
    write_result(stage)
    assert not [r for r in gate_full_port(stage, run).reasons if "fidelity" in r or "positions" in r]
