# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 461: a number from JSON that a float cannot hold is refused, and the check never raises OverflowError.

`math.isfinite(10**400)` and `float(10**400)` raise OverflowError, because an integer of more than about 308 digits does not fit a float.
The safe reader (`safeio`, MAX_INT_DIGITS = 100) refuses such an integer before any module sees it. A value parsed by `json.loads`, built
in memory, or read with a higher digit cap reaches the module directly, so each test below hands the value straight to the entry point that
reads it. That is the same as reading it with the reader's cap raised.

Each converted module has a test that feeds 10**400, -10**400, inf and NaN and checks that the module refuses the value (an error of its
own type, a problem in its list, a None, or the safe choice) and does not crash.
"""
from __future__ import annotations

import math

import pytest

from orchard import safeio

BAD = [10 ** 400, -10 ** 400, float("inf"), float("nan")]
BAD_IDS = ["huge", "minus-huge", "inf", "nan"]
bad = pytest.mark.parametrize("v", BAD, ids=BAD_IDS)


# ---- the helpers in safeio ---------------------------------------------------------------------------------------------------------------

def test_fits_float_is_true_for_a_number_a_float_can_hold():
    for v in [0, 1, -5, 1.5, 0.0, -2.25, 10 ** 300, -10 ** 300, 1e308]:
        assert safeio.fits_float(v) is True


def test_fits_float_is_false_for_a_huge_int_a_non_finite_float_and_a_non_number():
    for v in BAD + [float("-inf"), None, "1", [1], {"a": 1}, True, False, 10 ** 309]:
        assert safeio.fits_float(v) is False


def test_an_int_just_past_the_float_range_does_not_fit():
    # the largest float is about 1.798e308; 2**1024 is the first power of two past it
    assert safeio.fits_float(2 ** 1023) is True
    assert safeio.fits_float(2 ** 1024) is False


# ---- spec_break_even -----------------------------------------------------------------------------------------------------------------

from orchard import spec_break_even as sbe


@bad
def test_spec_break_even_refuses_a_time_or_an_acceptance_a_float_cannot_hold(v):
    for args in [(v, 2.0, 10.0), (5.0, v, 10.0), (5.0, 2.0, v)]:
        with pytest.raises(sbe.BreakEvenInputError):
            sbe.speculative_break_even(*args)
    with pytest.raises(sbe.BreakEvenInputError):
        sbe.speculative_break_even(5.0, 2.0, 10.0, margin=v)
    with pytest.raises(sbe.BreakEvenInputError):
        sbe.verdict_by_prompt_kind(5.0, 10.0, {"code": [1.0, v]})


# ---- decode_floor --------------------------------------------------------------------------------------------------------------------

from orchard import decode_floor as df

DENSE = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 320}


@bad
def test_decode_floor_refuses_a_bandwidth_a_speed_or_a_share_a_float_cannot_hold(v):
    with pytest.raises(df.FloorInputError):
        df.decode_floor(DENSE, {"default": "bf16"}, 1, v)
    floor = df.decode_floor(DENSE, {"default": "bf16"}, 1, 100.0)
    with pytest.raises(df.FloorInputError):
        df.share_of_floor(v, floor)
    with pytest.raises(df.FloorInputError):
        df.low_share_hint(v, threshold=0.5)
    with pytest.raises(df.FloorInputError):
        df.low_share_hint(0.2, threshold=v)


# ---- graft_trial ---------------------------------------------------------------------------------------------------------------------

from orchard import graft_trial as gt


def _trial(**side):
    safe = {"server_ready_s": 100.0, "decode_tok_s": 20.0, "ttft_s": 0.3}
    var = {"served": True, "coherent": True, "top1_agreement": 0.96875, "n_tokens": 32, "server_ready_s": 240.0, "decode_tok_s": 60.0,
           "ttft_s": 0.2, "failure": None}
    for key, val in side.items():
        who, field = key.split("__")
        (safe if who == "safe" else var)[field] = val
    return {"safe": safe, "variant": var, "steps": []}


def test_graft_trial_baseline_chooses_the_graft():
    assert gt.decide(_trial(), safe_top1=0.9)["choice"] == "graft"


@bad
def test_graft_trial_keeps_the_safe_configuration_for_a_speed_or_agreement_a_float_cannot_hold(v):
    for kw in [{"variant__decode_tok_s": v}, {"safe__decode_tok_s": v}, {"variant__top1_agreement": v}]:
        d = gt.decide(_trial(**kw), safe_top1=0.9)
        assert d["choice"] == "safe"
    d = gt.decide(_trial(), safe_top1=v)
    assert d["numbers"]["safe_top1_agreement"] is None
    assert gt.accept_rate({"accept_rate": v}) is None


# ---- sampling_probe ------------------------------------------------------------------------------------------------------------------

from orchard import sampling_probe as sp


def _entry(text, seconds=2.0, tokens=96, params=None):
    return {"params": params or {}, "accepted": True, "error": None, "seconds": seconds, "completion_tokens": tokens, "text": text,
            "finish_reason": "length", "drafter_log": ""}


def _record(**greedy):
    g = _entry("The lighthouse keeper climbed the stairs one more time that night and found the lamp already burning.",
               params={"temperature": 0}, **greedy)
    a = _entry("The lighthouse keeper climbed the long spiral stairs every evening, counting the steps and listening to the sea.",
               params={"temperature": 1.0, "top_p": 0.9, "seed": 1234})
    b = _entry("On the island there was a bell that rang by itself whenever the fog came in, and nobody remembered who hung it.",
               params={"temperature": 1.0, "top_p": 0.9, "seed": 4321})
    return {"ran": True, "prompt": "p", "max_tokens": 96, "requests": {"greedy": g, "sampled_a": a, "sampled_b": b}}


def test_sampling_probe_baseline_record_is_well_formed_and_has_a_greedy_speed():
    assert sp.problems(_record()) == []
    assert sp.summarize(_record())["greedy"]["tokens_per_s"] == pytest.approx(48.0)


@bad
def test_sampling_probe_names_a_problem_for_seconds_a_float_cannot_hold(v):
    assert any("seconds" in p for p in sp.problems(_record(seconds=v)))


def test_sampling_probe_gives_no_speed_for_a_token_count_a_float_cannot_hold():
    assert sp.summarize(_record(tokens=10 ** 400))["greedy"]["tokens_per_s"] is None


# ---- resident ------------------------------------------------------------------------------------------------------------------------

from orchard import resident


@bad
def test_resident_spread_refuses_a_timing_a_float_cannot_hold(v):
    assert resident.spread([1.0, 2.0])["median"] == 1.5
    with pytest.raises(resident.ResidentError):
        resident.spread([1.0, v])


# ---- noise_floor ---------------------------------------------------------------------------------------------------------------------

from orchard import noise_floor as nf

FLOOR = {"groups": {"layer": {"worst": {"rel_l2": 0.01, "slope_dev": 0.01, "pcc": 0.99}, "arrays": 3}}}


@bad
def test_noise_floor_refuses_a_margin_a_float_cannot_hold(v):
    assert "layer" in nf.derive_bars(FLOOR, margin=3.0)
    with pytest.raises(nf.NoiseFloorError):
        nf.derive_bars(FLOOR, margin=v)


# ---- first_divergence ----------------------------------------------------------------------------------------------------------------

from orchard import first_divergence as fd


@bad
def test_first_divergence_tolerance_refuses_a_bound_a_float_cannot_hold(v):
    fd.Tolerance()
    for kw in [{"rel_l2": v}, {"margin": v}, {"fallback_rel_l2": v}, {"min_pcc": v}]:
        with pytest.raises(fd.DivergenceError):
            fd.Tolerance(**kw)


# ---- queue_probe ---------------------------------------------------------------------------------------------------------------------

from orchard import queue_probe as qp


@bad
def test_queue_probe_refuses_a_setting_or_a_time_a_float_cannot_hold(v):
    for key in ("duration", "cfg_scale"):
        with pytest.raises(qp.ProbeError):
            qp.check_settings({"prompt": "rain on a tin roof", key: v})
    with pytest.raises(qp.ProbeError):
        qp.check_times(v, 10.0)
    with pytest.raises(qp.ProbeError):
        qp.check_times(1.0, v)
    assert qp._num(v) is None                     # the server's own timing numbers in a job reply


@bad
def test_queue_probe_request_refuses_a_time_limit_a_float_cannot_hold_before_any_connection(v, monkeypatch):
    def no_connection(*a, **kw):
        raise AssertionError("a refused time limit must not open a connection")
    monkeypatch.setattr(qp.http.client, "HTTPConnection", no_connection)
    status, obj, problem = qp.request("GET", "http://127.0.0.1:9/jobs", timeout=v)
    assert status is None and obj is None and "time limit" in problem


# ---- virtualspend --------------------------------------------------------------------------------------------------------------------

from orchard import virtualspend as vs


@bad
def test_virtualspend_refuses_a_price_a_float_cannot_hold(v):
    vs.load({"price_in": 1.0})
    for table in [{"price_in": v}, {"price_out": v}]:
        with pytest.raises(vs.ConfigError):
            vs.load(table)


# ---- telegraph -----------------------------------------------------------------------------------------------------------------------

from orchard import telegraph as tg


def _tg_table(**kw):
    base = {"enabled": True, "endpoint": "https://llm.example.org/v1", "models": ["small", "big"], "key_env": "TG_KEY"}
    base.update(kw)
    return base


@bad
def test_telegraph_refuses_a_price_or_a_cap_a_float_cannot_hold(v, tmp_path):
    prices = {"small": [1.0, 3.0], "big": [5.0, 15.0]}
    tg.load(_tg_table(prices=prices, max_cost_usd=1.0, max_cost_per_problem_usd=0.5), base=tmp_path)
    for kw in [{"prices": {"small": [v, 3.0]}}, {"prices": prices, "max_cost_usd": v}, {"prices": prices, "max_cost_per_problem_usd": v}]:
        with pytest.raises(tg.ConfigError):
            tg.load(_tg_table(**kw), base=tmp_path)


# ---- telegraph_store -----------------------------------------------------------------------------------------------------------------

from orchard import telegraph_store as tgs


@bad
def test_telegraph_store_gives_no_time_for_a_stamp_a_float_cannot_hold(v):
    assert tgs._ts_of({"ts": 12.5}) == 12.5
    assert tgs._ts_of({"ts": v}) is None


# ---- hazards -------------------------------------------------------------------------------------------------------------------------

from orchard import hazards as hz


@bad
def test_hazards_refuses_a_limit_a_float_cannot_hold(v):
    hz.scan_model_hazards(DENSE, None, limits={"kv_heads_max": 8})
    with pytest.raises(hz.HazardInputError):
        hz.scan_model_hazards(DENSE, None, limits={"kv_heads_max": v})


# ---- residency -----------------------------------------------------------------------------------------------------------------------

from orchard import residency as rs


@bad
def test_residency_refuses_a_memory_size_a_float_cannot_hold(v):
    rs.totals_bound({"BF16": 27_000_000_000}, dram_gib=32.0)
    for kw in [{"dram_gib": v}, {"kv_gib": v}, {"reserve_gib": v}]:
        with pytest.raises(rs.ResidencyError):
            rs.totals_bound({"BF16": 27_000_000_000}, **kw)


# ---- profile_summary -----------------------------------------------------------------------------------------------------------------

import io

from orchard import profile_summary as ps


@bad
def test_profile_refuses_a_time_a_float_cannot_hold(v, tmp_path):
    for flag in ("timeout", "report_timeout", "wait_s"):
        args = ps.parser().parse_args(["--out", str(tmp_path / "p"), "--", "python3", "-c", "pass"])
        setattr(args, flag, v)
        out = io.StringIO()
        assert ps.main(args, out=out) == ps.EXIT_REFUSED
        assert "must be a finite number of seconds" in out.getvalue()
        assert not (tmp_path / "p").exists()        # refused before the output directory is made


# ---- agreement_core ------------------------------------------------------------------------------------------------------------------

from orchard import agreement_core as ac


@bad
def test_agreement_core_refuses_an_agreement_or_a_speed_a_float_cannot_hold(v):
    assert ac.control_verdict({"pooled_top1": 0.9}, {"pooled_top1": 0.5})["separates"] is True
    assert ac.control_verdict({"pooled_top1": v}, {"pooled_top1": 0.5})["separates"] is False
    assert ac.control_verdict({"pooled_top1": 0.9}, {"pooled_top1": v})["separates"] is False
    cells = [{"input_tokens": 128, "concurrency": 1, "decode_tok_s": 20.0, "prefill_tok_s": 500.0},
             {"input_tokens": 1024, "concurrency": 1, "decode_tok_s": v, "prefill_tok_s": 500.0}]
    s = ac.sweep_summary(cells)
    assert s["missing_cells"] == 1


# ---- can -----------------------------------------------------------------------------------------------------------------------------

from orchard import can


@bad
def test_can_boot_record_drops_a_rate_a_float_cannot_hold(v):
    assert can.boot_record({"passed": True, "timings": {"decode_tok_s": 20.0}, "ready_s": 90.0}, None)["decode_tok_s"] == 20.0
    rec = can.boot_record({"passed": True, "timings": {"decode_tok_s": v}, "ready_s": v}, None)
    assert rec["decode_tok_s"] is None and rec["ready_s"] is None


# ---- package -------------------------------------------------------------------------------------------------------------------------

import json

from orchard import package as pk


def _graft_run(tmp_path, variant_speed):
    st = tmp_path / "stages" / "2"
    (st / gt.TRIAL_DIR).mkdir(parents=True)
    (st / gt.DECISION_FILE).write_text(json.dumps({"choice": "graft", "trial": f"{gt.TRIAL_DIR}/{gt.TRIAL_RESULT}"}))
    # json.dumps writes 10**400 as digits and inf/nan as Infinity/NaN; graft_trial_numbers reads the file with json.loads, which takes all three
    (st / gt.TRIAL_DIR / gt.TRIAL_RESULT).write_text(json.dumps({"safe": {"decode_tok_s": 20.0}, "variant": {"decode_tok_s": variant_speed}}))
    return tmp_path


def test_package_graft_trial_numbers_baseline_lists_both_speeds(tmp_path):
    names = [n.name for n in pk.graft_trial_numbers(_graft_run(tmp_path, 60.0))]
    assert any("grafted drafter" in n for n in names) and any("safe configuration" in n for n in names)


@bad
def test_package_graft_trial_numbers_leaves_out_a_speed_a_float_cannot_hold(v, tmp_path):
    rows = pk.graft_trial_numbers(_graft_run(tmp_path, v))
    assert [r.name for r in rows] == ["decode speed, safe configuration (graft trial, median of four prompts)"]


@bad
def test_package_rounding_gives_none_for_a_value_a_float_cannot_hold(v):
    assert pk._rounded(1.23456, 2) == 1.23
    assert pk._rounded(v, 2) is None


# ---- package_cmd ---------------------------------------------------------------------------------------------------------------------

from orchard import package_cmd as pc


@bad
def test_package_cmd_refuses_a_benchmark_score_a_float_cannot_hold(v):
    pc.benchmark_rows({"tasks": {"math": {"mine": {"n": 10, "score": 0.5}}}}, "mine", None)
    with pytest.raises(pc.PackageCommandError):
        pc.benchmark_rows({"tasks": {"math": {"mine": {"n": 10, "score": v}}}}, "mine", None)
    rows = pc.benchmark_rows({"tasks": {"math": {"mine": {"n": 10, "score": 0.5, "cut_off": v}}}}, "mine", None)
    assert len(rows) == 1                          # the cut-off row is left out


# ---- text_bars -----------------------------------------------------------------------------------------------------------------------

from orchard import text_bars as tb


@bad
def test_text_bars_refuses_a_perplexity_a_float_cannot_hold(v):
    for ref, chip in [({"perplexity": v}, {"perplexity": 5.0}), ({"perplexity": 5.0}, {"perplexity": v})]:
        r = tb.perplexity_report(ref, chip)
        assert any("perplexity is not a positive finite number" in x for x in r["reasons"])


# ---- audio_reference -----------------------------------------------------------------------------------------------------------------

from orchard import audio_reference as ar


def _floor_report(**result):
    r = {"kind": "layer", "name": "0", "comparable": True, "pcc": 0.99, "rel_l2": 0.01, "slope": 1.0}
    r.update(result)
    return {"comparable": True, "cross_precision": True, "results": [r]}


@bad
def test_audio_reference_refuses_a_margin_or_a_floor_number_a_float_cannot_hold(v):
    assert "layer" in ar.bars_from_floor(_floor_report(), margin=3.0)
    with pytest.raises(ar.ReferenceError):
        ar.bars_from_floor(_floor_report(), margin=v)
    for key in ("pcc", "rel_l2", "slope"):
        with pytest.raises(ar.ReferenceError):
            ar.bars_from_floor(_floor_report(**{key: v}), margin=3.0)


# ---- full_port_quality ---------------------------------------------------------------------------------------------------------------

from orchard import full_port_quality as fpq


@bad
def test_full_port_quality_names_a_problem_for_a_log_probability_a_float_cannot_hold(v):
    n = fpq.MIN_TOKENS
    good = {"reference_ids": [1] * n, "chip_ids": [1] * n, "logprobs": [-0.1] * n}
    assert not any(p.startswith("logprobs") for p in fpq.record_problems(good))
    worse = dict(good, logprobs=[-0.1] * (n - 1) + [v])
    assert any(p.startswith("logprobs") for p in fpq.record_problems(worse))


# ---- floor_report --------------------------------------------------------------------------------------------------------------------

from orchard import floor_report as fr


def _floor_record(per_die_bw=100.0, floor_tok_s=50.0):
    return {"scan": "ran", "dtype_policy": {"default": "bf16"}, "board": "p300", "dtype_policy_source": "config", "bandwidth_source": "spec",
            "assumptions": "a", "floors": {"1": {"floor_tok_s": floor_tok_s, "floor_ms": 20.0, "bytes_per_token": 1000,
                                                 "inputs": {"per_die_bandwidth_gb_s": per_die_bw}}}}


@pytest.fixture
def unbounded_reader(monkeypatch):
    """safeio.read_json with no digit cap and with NaN and Infinity allowed, so that the value reaches the module as it would from a reader
    with a higher cap or from json.loads."""
    def read_json(path, **kw):
        with open(path, encoding="utf-8") as fh:
            return json.loads(fh.read())
    monkeypatch.setattr(safeio, "read_json", read_json)


def _write_floor(tmp_path, rec):
    d = tmp_path / "stages" / "0"
    d.mkdir(parents=True)
    (d / fr.FLOOR_FILE).write_text(json.dumps(rec))
    return tmp_path


def test_floor_report_baseline_record_is_read(tmp_path, unbounded_reader):
    assert fr.read_floor(_write_floor(tmp_path, _floor_record())) is not None


@bad
def test_floor_report_refuses_a_floor_record_with_a_number_a_float_cannot_hold(v, tmp_path, unbounded_reader):
    assert fr.read_floor(_write_floor(tmp_path / "a", _floor_record(per_die_bw=v))) is None
    assert fr.read_floor(_write_floor(tmp_path / "b", _floor_record(floor_tok_s=v))) is None
