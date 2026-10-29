# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The record of a set of measured runs (orchard/resident.py `summarise_runs`): the label, the spread of each part, the load and the load averages."""
from orchard import resident as r


def test_the_warmup_decides_the_label_with_the_count():
    runs = [{"a": 1.0}] * 3
    assert r.summarise_runs(runs, 1)["label"] == "benchmark" and r.summarise_runs(runs, 0)["label"] == "unwarmed"
    assert r.summarise_runs(runs)["label"] == "unwarmed" and r.summarise_runs(runs[:2], 1)["label"] == "single sample"


def test_summarise_runs_gives_the_label_the_spread_the_load_and_the_load_averages():
    runs = [{"generate_and_decode_s": 1.8, "model_load_s": 6.0, "chip_weights_upload_s": 1.2, "host_loadavg_1m": 3.0},
            {"generate_and_decode_s": 1.6, "host_loadavg_1m": 4.0}, {"generate_and_decode_s": 1.7}]
    rec = r.summarise_runs(runs, 1)
    assert rec["label"] == "benchmark" and rec["repeats"] == 3
    assert rec["parts"] == {"generate_and_decode_s": {"n": 3, "min": 1.6, "median": 1.7, "max": 1.8}}
    assert rec["load"] == {"model_load_s": 6.0, "chip_weights_upload_s": 1.2} and rec["host_loadavg_1m"] == [3.0, 4.0, None]
    assert rec["incomplete_parts"] == []


def test_summarise_runs_names_a_part_some_runs_lack_and_does_not_spread_it():
    rec = r.summarise_runs([{"a": 1.0, "b": 2.0}, {"a": 1.5}, {"a": 2.0, "b": 3.0}])
    assert set(rec["parts"]) == {"a"} and rec["incomplete_parts"] == ["b"]


def test_summarise_runs_of_two_is_a_single_sample_and_of_none_is_no_measurement():
    assert r.summarise_runs([{"a": 1.0}, {"a": 2.0}])["label"] == "single sample"
    empty = r.summarise_runs([])
    assert empty["label"] == "no measurement" and empty["parts"] == {} and empty["repeats"] == 0


def test_summarise_runs_ignores_what_is_not_a_finite_number():
    rec = r.summarise_runs([{"a": 1.0, "n": float("nan"), "t": True}] * 3)
    assert set(rec["parts"]) == {"a"} and rec["incomplete_parts"] == []


def test_a_run_that_is_not_an_object_is_not_counted():
    rec = r.summarise_runs([None, {"a": 1.0}, "x", {"a": 2.0}, {"a": 3.0}])
    assert rec["repeats"] == 3 and rec["parts"]["a"]["median"] == 2.0


def test_the_load_and_the_load_average_are_not_parts():
    rec = r.summarise_runs([{"a": 1.0, "model_load_s": 6.0, "chip_weights_upload_s": 1.0, "host_loadavg_1m": 2.0}] * 3)
    assert set(rec["parts"]) == {"a"}


def test_the_first_run_supplies_the_load_even_when_later_runs_differ():
    rec = r.summarise_runs([{"a": 1.0, "model_load_s": 6.0}, {"a": 1.0, "model_load_s": 9.0}, {"a": 1.0}])
    assert rec["load"] == {"model_load_s": 6.0}
