# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The resident-runtime contract (orchard/resident.py): `load()` once, `generate(settings)` many times, and the benchmark that uses it.

A fake runtime stands in for the chip. The tests state what the contract promises: one load for many generations, the same settings and seed giving
the same output on the first call and on a later call, a warm-up that is discarded, spread for each timed part, and a label that says
"single sample" when fewer than three runs were measured."""
import pytest

from orchard import resident as r


def settings(**over):
    return {"prompt": "p", "duration": 10.0, "steps": 8, "cfg_scale": 1.0, "seed": 42, "sampler": "pingpong", "sample_size": 5292032, **over}


NOLOAD = lambda: (0.0, 0.0, 0.0)          # noqa: E731


# ---- the contract -------------------------------------------------------------------------------------------------------------------

def test_the_fake_loads_once_and_generates_many_times():
    rt = r.FakeRuntime()
    assert rt.loads == 0
    load = rt.load()
    outs = [rt.generate(settings()) for _ in range(4)]
    assert rt.loads == 1 and load["model_load_s"] > 0 and len(outs) == 4


def test_generate_before_load_is_refused():
    with pytest.raises(r.ResidentError, match="load"):
        r.FakeRuntime().generate(settings())


def test_a_second_load_is_refused():
    rt = r.FakeRuntime()
    rt.load()
    with pytest.raises(r.ResidentError, match="already"):
        rt.load()


def test_generate_after_close_is_refused():
    rt = r.FakeRuntime()
    rt.load()
    rt.close()
    with pytest.raises(r.ResidentError, match="closed"):
        rt.generate(settings())


def test_the_same_seed_gives_the_same_output_on_a_fresh_runtime_and_on_a_late_call():
    """The one-shot path is a fresh runtime that loads and generates once; the resident path generates again on a loaded one. Same arrays."""
    one_shot = r.FakeRuntime()
    one_shot.load()
    first = one_shot.generate(settings(seed=7))["output_sha256"]
    resident = r.FakeRuntime()
    resident.load()
    resident.generate(settings(seed=1))                  # other work in between must not change what seed 7 gives
    resident.generate(settings(seed=2, duration=30.0))
    assert resident.generate(settings(seed=7))["output_sha256"] == first
    assert resident.generate(settings(seed=8))["output_sha256"] != first


def test_a_setting_that_changes_the_audio_changes_the_output():
    rt = r.FakeRuntime()
    rt.load()
    base = rt.generate(settings())["output_sha256"]
    for change in ({"duration": 30.0}, {"steps": 9}, {"cfg_scale": 2.0}, {"prompt": "q"}):
        assert rt.generate(settings(**change))["output_sha256"] != base, change


def test_same_output_notices_a_runtime_whose_later_calls_differ():
    rt = r.FakeRuntime(drift=True)
    rt.load()
    a, b = rt.generate(settings()), rt.generate(settings())
    assert a["output_sha256"] != b["output_sha256"]
    assert r.same_output(rt, settings(), calls=3)["same"] is False


def test_same_output_says_the_calls_agree_for_a_runtime_with_no_drift():
    rt = r.FakeRuntime()
    rt.load()
    res = r.same_output(rt, settings(), calls=3)
    assert res["same"] is True and res["calls"] == 3 and len(set(res["sha256"])) == 1


# ---- spread --------------------------------------------------------------------------------------------------------------------------

def test_spread_is_min_median_max_and_the_count():
    assert r.spread([3.0, 1.0, 2.0]) == {"n": 3, "min": 1.0, "median": 2.0, "max": 3.0}
    assert r.spread([4.0, 1.0, 2.0, 3.0])["median"] == 2.5


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")], [True], ["1"], [None]])
def test_spread_refuses_what_is_not_a_finite_number(bad):
    with pytest.raises(r.ResidentError):
        r.spread(bad)


# ---- the rules ----------------------------------------------------------------------------------------------------------------------

def test_the_rules_are_written_down_as_numbers():
    assert r.WARMUP_RUNS == 1 and r.MIN_REPEATS == 3
    assert r.SWEEP_SECONDS == (10.0, 30.0, 60.0, 120.0) and r.MAX_SECONDS == 120.0


@pytest.mark.parametrize("n, label", [(0, "no measurement"), (1, "single sample"), (2, "single sample"), (3, "benchmark"), (5, "benchmark")])
def test_the_label_follows_the_number_of_measured_repeats(n, label):
    assert r.label_for(n, 1) == label


@pytest.mark.parametrize("warmup", [None, 0, -1, True, 0.5, "1"])
def test_three_repeats_without_a_stated_warmup_are_unwarmed_not_a_benchmark(warmup):
    assert r.label_for(3, warmup) == "unwarmed"


def test_a_warmup_does_not_rescue_fewer_than_three_repeats():
    assert r.label_for(2, 5) == "single sample" and r.label_for(0, 5) == "no measurement"


def test_more_warmup_than_the_rule_asks_is_still_a_benchmark():
    assert r.label_for(3, 2) == "benchmark"


# ---- the benchmark -------------------------------------------------------------------------------------------------------------------

def test_the_benchmark_discards_the_warmup_and_measures_the_repeats():
    rt = r.FakeRuntime(times=[9.0, 3.0, 1.0, 2.0])        # the first call is the slow, discarded one
    res = r.run_benchmark(rt, settings(), warmup=1, repeats=3, loadavg=lambda: (0.5, 0.4, 0.3))
    assert rt.loads == 1 and rt.generate_calls == 4
    assert res["label"] == "benchmark" and res["warmup_runs"] == 1 and res["repeats"] == 3
    assert res["parts"]["generate_and_decode_s"] == {"n": 3, "min": 1.0, "median": 2.0, "max": 3.0}
    assert res["load"]["model_load_s"] == rt.load_timings["model_load_s"]
    assert len(res["runs"]) == 3 and all(run["host_loadavg_1m"] == 0.5 for run in res["runs"])


def test_each_timed_part_has_its_own_spread():
    rt = r.FakeRuntime(times=[1.0, 2.0, 3.0, 4.0], parts={"chip_transformer_call_s": [0.1, 0.2, 0.3, 0.4]})
    res = r.run_benchmark(rt, settings(), warmup=1, repeats=3, loadavg=lambda: (1.0, 1.0, 1.0))
    assert set(res["parts"]) == {"generate_and_decode_s", "chip_transformer_call_s"}
    assert res["parts"]["chip_transformer_call_s"] == {"n": 3, "min": 0.2, "median": 0.3, "max": 0.4}


def test_three_repeats_with_no_warmup_are_unwarmed():
    res = r.run_benchmark(r.FakeRuntime(), settings(), warmup=0, repeats=3, loadavg=NOLOAD)
    assert res["label"] == "unwarmed" and res["warmup_runs"] == 0


def test_fewer_than_three_repeats_is_a_single_sample():
    res = r.run_benchmark(r.FakeRuntime(), settings(), warmup=1, repeats=2, loadavg=lambda: (1.0, 1.0, 1.0))
    assert res["label"] == "single sample" and res["repeats"] == 2


def test_zero_repeats_and_negative_warmup_are_refused():
    with pytest.raises(r.ResidentError):
        r.run_benchmark(r.FakeRuntime(), settings(), warmup=1, repeats=0)
    with pytest.raises(r.ResidentError):
        r.run_benchmark(r.FakeRuntime(), settings(), warmup=-1, repeats=3)


def test_a_part_missing_from_one_run_is_refused():
    rt = r.FakeRuntime(times=[1.0, 2.0, 3.0, 4.0], drop_part_on_call=3)
    with pytest.raises(r.ResidentError, match="part"):
        r.run_benchmark(rt, settings(), warmup=1, repeats=3, loadavg=NOLOAD, required_parts=("generate_and_decode_s",))


def test_a_failed_loadavg_read_is_recorded_as_none():
    def broken():
        raise OSError("no")
    res = r.run_benchmark(r.FakeRuntime(), settings(), warmup=0, repeats=3, loadavg=broken)
    assert [run["host_loadavg_1m"] for run in res["runs"]] == [None, None, None]


def test_a_benchmark_leaves_the_runtime_loaded_for_the_caller_to_close():
    rt = r.FakeRuntime()
    r.run_benchmark(rt, settings(), warmup=1, repeats=3, loadavg=NOLOAD)
    assert rt.closed is False


# ---- the length sweep ---------------------------------------------------------------------------------------------------------------

def test_the_sweep_runs_each_stated_size_on_one_loaded_runtime():
    rt = r.FakeRuntime()
    res = r.run_sweep(rt, settings(), warmup=1, repeats=3, loadavg=NOLOAD)
    assert rt.loads == 1
    assert [row["seconds"] for row in res["sizes"]] == [10.0, 30.0, 60.0, 120.0]
    assert all(row["label"] == "benchmark" for row in res["sizes"])
    assert rt.generate_calls == 4 * (1 + 3)
    assert rt.seen_durations[:4] == [10.0] * 4


@pytest.mark.parametrize("bad", [(10.0, 121.0), (0.0,), (float("nan"),), (-5.0,), (True,), ()])
def test_the_sweep_refuses_a_size_over_the_maximum_or_not_a_positive_number(bad):
    with pytest.raises(r.ResidentError):
        r.run_sweep(r.FakeRuntime(), settings(), sizes=bad, warmup=0, repeats=3, loadavg=NOLOAD)


def test_the_sweep_keeps_the_seed_and_the_other_settings_for_every_size():
    rt = r.FakeRuntime()
    r.run_sweep(rt, settings(seed=5), sizes=(10.0, 30.0), warmup=0, repeats=3, loadavg=NOLOAD)
    assert set(rt.seen_seeds) == {5}


# ---- naming the repeats' directories --------------------------------------------------------------------------------------------------

def test_repeat_dir_matches_the_stage_plans_names(tmp_path):
    from orchard.audio_plan import repeat_name
    out = tmp_path / "small-music"
    for k in range(4):
        assert r.repeat_dir(out, k).name == repeat_name("small-music", k)
        assert r.repeat_dir(out, k).parent == out.parent


def test_measured_runs_record_into_the_directories_asked_for_and_the_warmup_does_not():
    rt = r.FakeRuntime()
    r.run_benchmark(rt, settings(), warmup=2, repeats=3, loadavg=NOLOAD, out_dirs=lambda k: f"d{k}")
    assert rt.out_dirs == [None, None, "d0", "d1", "d2"]


def test_a_runtime_the_caller_already_loaded_is_not_loaded_again():
    rt = r.FakeRuntime()
    load = rt.load()
    res = r.run_benchmark(rt, settings(), warmup=0, repeats=3, loadavg=NOLOAD, load_timings=load)
    assert rt.loads == 1 and res["load"] == load


def test_extra_timings_reach_the_measured_runs_they_are_asked_for_and_stay_out_of_the_spread():
    rt = r.FakeRuntime()
    res = r.run_benchmark(rt, settings(), warmup=1, repeats=3, loadavg=NOLOAD, extra_timings=lambda k: {"model_load_s": 6.0} if k == 0 else None)
    assert rt.extras == [None, {"model_load_s": 6.0}, None, None]
    assert "model_load_s" not in res["parts"]
