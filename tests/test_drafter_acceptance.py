# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Drafter acceptance as a measured field (issue 14). The DFlash2 serving code logs one `[dflash2-serve] slot N session: ...` line when a
request ends. `orchard/drafter_log.py` reads those lines; measure_serving.py records what the measured prompts produced; the graft trial
carries the number to the decision, and the package card and the stage 8 bundle show it or say it was not measured. The log lines in the
fixtures have the exact shape of a real AREX-2 server log (2026-10-23)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchard import drafter_log as dl  # noqa: E402
from orchard import measure_serving as ms  # noqa: E402
from orchard.graft_trial import decide, write_decision  # noqa: E402
from orchard.package import ACCEPTANCE_ROW, graft_trial_numbers  # noqa: E402

REAL = ("(EngineCore pid=1802370) 2026-10-23 07:37:28.910 | INFO     | models.demos.blackhole.qwen36.tt.dflash2_serving:end:907 - "
        "[dflash2-serve] slot 0 session: 4 iters, 18 committed tokens, accept 3.75/7 -> 4.50 tok/iter, 21.3 tok/s in-model (incl. seed); "
        "histogram [0, 2, 0, 0, 0, 0, 1, 1]\n"
        "(EngineCore pid=1802370) 2026-10-23 07:37:28.910 | INFO     | models.demos.blackhole.qwen36.tt.dflash2_serving:end:907 - "
        "[dflash2-serve] slot 1 session: 4 iters, 4 committed tokens, accept 0.25/7 -> 1.00 tok/iter, 5.8 tok/s in-model (incl. seed); "
        "histogram [3, 1, 0, 0, 0, 0, 0, 0]\n")
ACCEPTED = 2 * 1 + 6 + 7 + 1            # histogram index times count, both slots


# ---- the module ---------------------------------------------------------------------------------------------------------------------

def test_the_real_log_lines_give_accepted_over_drafted():
    s = dl.drafter_stats(REAL)
    assert s["iters"] == 8 and s["draft_length"] == 7 and s["sessions"] == 2
    assert s["accepted_per_iter"] == pytest.approx(ACCEPTED / 8) and s["accept_rate"] == pytest.approx(ACCEPTED / (8 * 7))


def test_bench_session_still_offers_the_same_function():
    from orchard import bench_session
    assert bench_session.drafter_stats is dl.drafter_stats and bench_session.SESSION_LINE is dl.SESSION_LINE


def test_a_log_slice_starts_at_the_byte_offset(tmp_path):
    p = tmp_path / "server.log"
    p.write_text("before\n" + REAL)
    assert dl.read_since(p, 0).startswith("before")
    assert dl.read_since(p, len("before\n")) == REAL
    assert dl.read_since(p, p.stat().st_size) == ""
    assert dl.read_since(p, p.stat().st_size + 50) == ""


def test_a_missing_log_has_size_zero_and_no_text(tmp_path):
    assert dl.size(tmp_path / "nope.log") == 0 and dl.read_since(tmp_path / "nope.log", 0) == ""


def test_the_size_is_in_bytes(tmp_path):
    p = tmp_path / "s.log"
    p.write_bytes("é\n".encode())
    assert dl.size(p) == 3
    assert dl.read_since(p, 2) == "\n"


def test_a_summary_keeps_the_fields_the_trial_records():
    s = dl.summary(dl.drafter_stats(REAL))
    assert set(s) == {"accept_rate", "accepted_per_iter", "draft_length", "iters", "sessions"}
    assert s["accept_rate"] == pytest.approx(ACCEPTED / 56) and s["sessions"] == 2 and s["iters"] == 8 and s["draft_length"] == 7
    assert dl.summary(None) is None and dl.summary({}) is None


# ---- measure_serving ----------------------------------------------------------------------------------------------------------------

def test_the_metrics_rate_wins_when_the_server_exports_counters():
    rate, note, stats = ms.drafter_record(metrics_rate=0.5, log_text=REAL)
    assert rate == 0.5 and "/metrics" in note and stats["iters"] == 8


def test_with_no_counters_the_rate_comes_from_the_log_and_the_note_says_so():
    rate, note, stats = ms.drafter_record(metrics_rate=None, log_text=REAL)
    assert rate == pytest.approx(ACCEPTED / 56) and "[dflash2-serve]" in note and "log" in note
    assert stats["draft_length"] == 7


def test_with_neither_the_rate_is_none_and_the_note_says_not_measured():
    rate, note, stats = ms.drafter_record(metrics_rate=None, log_text="INFO nothing\n")
    assert rate is None and stats is None and note.startswith("not measured")
    assert "no `[dflash2-serve]` session line" in note and "drafter is off" in note


def test_the_old_claim_that_the_log_prints_no_counters_is_gone():
    assert "prints none in its log" not in Path(ms.__file__).read_text()


def test_the_record_fields_hold_the_rate_the_note_and_the_compact_summary():
    f = ms.acceptance_fields(metrics_rate=None, log_text=REAL)
    assert set(f) == {"drafter_acceptance", "drafter_acceptance_note", "drafter"}
    assert f["drafter_acceptance"] == pytest.approx(ACCEPTED / 56) and set(f["drafter"]) == set(dl.SUMMARY_KEYS)
    none = ms.acceptance_fields(metrics_rate=None, log_text="")
    assert none["drafter_acceptance"] is None and none["drafter"] is None and none["drafter_acceptance_note"].startswith("not measured")


def test_the_script_import_falls_back_to_the_checkout_when_there_is_no_package():
    import runpy
    ns = runpy.run_path(ms.__file__, run_name="measure_serving_as_script")
    assert ns["drafter_log"].drafter_stats is dl.drafter_stats


def test_the_measured_prompts_are_bracketed_by_log_offsets():
    """main() reads the log from the offset taken after the warm-up, so the warm-up's session line is not counted."""
    text = Path(ms.__file__).read_text()
    assert "drafter_log.size(log_path)" in text and "drafter_log.read_since(log_path" in text
    assert "acceptance_fields(metrics_rate=acceptance(before=metrics0, after=metrics1), log_text=log_text)" in text and "**fields," in text


# ---- the trial script ---------------------------------------------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates" / "graft_trial.py"
_spec = importlib.util.spec_from_file_location("graft_trial_script_for_acceptance", SCRIPT)
gt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gt)

DRAFTER = {"accept_rate": 0.31, "accepted_per_iter": 2.17, "draft_length": 7, "iters": 800, "sessions": 12}


def record(drafter):
    return {"server_ready_s": 100.0, "prompt_set_sha256": "x", "reps": 3, "drafter": drafter,
            "prompts": {"a": {"summary": {"decode_tok_s": {"median": 50.0}, "ttft_s": {"median": 0.2}}}}}


def test_the_speed_summary_carries_the_drafter_numbers():
    assert gt.speed_summary(record(DRAFTER))["drafter"] == DRAFTER


def test_a_record_without_a_drafter_gives_none_and_a_malformed_one_too():
    assert gt.speed_summary(record(None))["drafter"] is None
    assert gt.speed_summary({k: v for k, v in record(None).items() if k != "drafter"})["drafter"] is None
    assert gt.speed_summary(record("lots"))["drafter"] is None


def test_only_the_five_known_fields_are_carried():
    assert gt.speed_summary(record({**DRAFTER, "histogram": [1, 2], "junk": 1}))["drafter"] == DRAFTER


def test_the_variant_record_gets_the_drafter_numbers():
    assert '"drafter": speed.get("drafter")' in SCRIPT.read_text()


# ---- the decision -------------------------------------------------------------------------------------------------------------------

def _trial(var_drafter=DRAFTER, safe_drafter=None):
    return {"safe": {"decode_tok_s": 20.0, "ttft_s": 0.3, "drafter": safe_drafter},
            "variant": {"served": True, "coherent": True, "top1_agreement": 0.96875, "decode_tok_s": 60.0, "ttft_s": 0.2,
                        "server_ready_s": 240.0, "drafter": var_drafter}}


def test_the_decision_numbers_carry_the_acceptance_rate_of_each_side():
    d = decide(_trial(safe_drafter={**DRAFTER, "accept_rate": 0.5}), safe_top1=0.96875)
    assert d["numbers"]["variant_drafter_accept_rate"] == 0.31 and d["numbers"]["safe_drafter_accept_rate"] == 0.5


def test_a_rate_of_exactly_one_is_kept():
    d = decide(_trial(var_drafter={**DRAFTER, "accept_rate": 1.0}), safe_top1=0.96875)
    assert d["numbers"]["variant_drafter_accept_rate"] == 1.0


@pytest.mark.parametrize("bad", [None, {}, {"accept_rate": True}, {"accept_rate": 1.5}, {"accept_rate": -0.1},
                                 {"accept_rate": float("nan")}, "x"])
def test_a_missing_or_malformed_rate_is_none_in_the_numbers(bad):
    d = decide(_trial(var_drafter=bad), safe_top1=0.96875)
    assert d["numbers"]["variant_drafter_accept_rate"] is None


def test_acceptance_never_changes_the_choice():
    low = decide(_trial(var_drafter={**DRAFTER, "accept_rate": 0.0}), safe_top1=0.96875)
    none = decide(_trial(var_drafter=None), safe_top1=0.96875)
    assert low["choice"] == "graft" and none["choice"] == "graft" and low["reason"] == none["reason"]


# ---- the card rows ------------------------------------------------------------------------------------------------------------------

def _run_with_trial(tmp_path, var_drafter):
    s2 = tmp_path / "stages" / "2"
    (s2 / "graft-trial").mkdir(parents=True)
    t = _trial(var_drafter=var_drafter)
    (s2 / "graft-trial" / "trial-result.json").write_text(json.dumps(t))
    write_decision(s2, decide(t, safe_top1=0.96875), candidate={"x": 1})
    return tmp_path


def test_the_card_gets_an_acceptance_row_when_the_trial_measured_one(tmp_path):
    rows = {n.name: n for n in graft_trial_numbers(_run_with_trial(tmp_path, DRAFTER))}
    n = rows[ACCEPTANCE_ROW]
    assert n.value == 0.31 and n.unit == "fraction" and n.label == "measured"
    assert "accepted" in ACCEPTANCE_ROW and "drafted" in ACCEPTANCE_ROW and "graft trial" in ACCEPTANCE_ROW


def test_an_acceptance_of_zero_is_a_measurement_and_gets_a_row(tmp_path):
    rows = {n.name: n for n in graft_trial_numbers(_run_with_trial(tmp_path, {**DRAFTER, "accept_rate": 0.0}))}
    assert rows[ACCEPTANCE_ROW].value == 0.0


@pytest.mark.parametrize("bad", [None, {}, {"accept_rate": 2}, {"accept_rate": True}, {"accept_rate": "0.3"}])
def test_no_row_when_the_trial_has_no_usable_rate(tmp_path, bad):
    assert ACCEPTANCE_ROW not in {n.name for n in graft_trial_numbers(_run_with_trial(tmp_path, bad))}


def test_measure_serving_runs_as_a_script_the_way_the_graft_trial_starts_it(tmp_path):
    """The trial runs `python orchard/measure_serving.py`, so the import of drafter_log must work with no package context."""
    import subprocess
    p = subprocess.run([sys.executable, str(Path(ms.__file__)), "--help"], capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
    assert p.returncode == 0 and "--run-sh" in p.stdout, p.stderr


def test_the_scripts_copy_of_the_key_list_equals_the_modules():
    assert gt.DRAFTER_KEYS == dl.SUMMARY_KEYS
