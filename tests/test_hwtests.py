"""The list of hardware tests a weights-only stage 4 runs, and the records it leaves."""
import json
from pathlib import Path

import pytest

from orchard.hwtests import (HwTest, load_plan, move_aside, pending, read_plan, suspect_caches,
                             write_plan, write_record, write_summary)

NOTE = {"goal": "g", "stage": 4, "evidence": ["e"], "next_action": "n", "check_on_return": "c"}


def config(sd, n, script, cache):
    d = sd / "configs" / str(n)
    d.mkdir(parents=True, exist_ok=True)
    (d / script).write_text("# template copy\n")
    (d / "swap_config.json").write_text(json.dumps({"tt_cache": str(cache)}))


@pytest.fixture
def plan(tmp_path):
    """A stage directory with three configurations, each with its own cache, and the list."""
    sd = tmp_path / "run" / "stages" / "4"
    caches = {n: tmp_path / "orchard-cache" / "hemmingway-1" / f"{n}chip" / "tt_cache" for n in (1, 2, 4)}
    config(sd, 1, "serve_and_compare.py", caches[1])
    config(sd, 2, "serve_and_compare.py", caches[2])
    config(sd, 4, "serve_and_compare_container.py", caches[4])
    tests = [{"chips": 4, "script": "serve_and_compare_container.py", "deadline_s": 3600},
             {"chips": 2, "script": "serve_and_compare.py", "deadline_s": 2400},
             {"chips": 1, "script": "serve_and_compare.py", "deadline_s": 2400}]
    (sd / "hw_tests.json").write_text(json.dumps({"tests": tests}))
    (sd / "handoff.json").write_text(json.dumps(NOTE))
    home = tmp_path / "operator-home"

    def read(required=(2, 4), budget=28800.0, edit=None):
        if edit:
            data = json.loads((sd / "hw_tests.json").read_text())
            edit(data["tests"])
            (sd / "hw_tests.json").write_text(json.dumps(data))
        return read_plan(sd, required=required, max_chips=4, budget_s=budget, home=home)
    return {"sd": sd, "caches": caches, "home": home, "read": read, "tmp": tmp_path}


def test_a_good_list_is_read_in_order_of_chips_with_boards_and_commands(plan):
    tests, problems = plan["read"]()
    assert problems == []
    assert [(t.chips, t.boards, t.deadline_s) for t in tests] == [(1, 1, 2400), (2, 1, 2400), (4, 2, 3600)]
    assert tests[2].command(4) == "python3 stages/4/configs/4/serve_and_compare_container.py"
    assert tests[0].cache == str(plan["caches"][1])


def set_key(i, key, value):
    def edit(tests):
        if value is ...:
            tests[i].pop(key)
        else:
            tests[i][key] = value
    return edit


@pytest.mark.parametrize("edit,words", [
    (set_key(0, "chips", 0), "chips from 1 to 4"),
    (set_key(0, "chips", 8), "chips from 1 to 4"),
    (set_key(0, "chips", True), "chips from 1 to 4"),
    (set_key(1, "chips", 4), "listed more than once"),
    (set_key(0, "script", "run.sh"), "script must be one of"),
    (set_key(1, "deadline_s", 0), "positive deadline_s"),
    (set_key(1, "deadline_s", ...), "positive deadline_s"),
])
def test_a_bad_entry_is_refused_with_a_reason_that_names_it(plan, edit, words):
    _, problems = plan["read"](required=(), edit=edit)
    assert len(problems) == 1 and words in problems[0], problems


def test_a_required_configuration_without_a_test_is_refused(plan):
    _, problems = plan["read"](edit=lambda t: t.pop(0))
    assert problems == ["the 4-chip configuration is required and hw_tests.json has no test for it"]


def test_deadlines_longer_than_the_stage_budget_are_refused(plan):
    _, problems = plan["read"](budget=8000.0)
    assert problems == ["the tests' deadlines add up to 8400 s, more than the stage budget of 8000 s"]


def test_a_missing_script_or_config_is_refused(plan):
    (plan["sd"] / "configs" / "2" / "serve_and_compare.py").unlink()
    (plan["sd"] / "configs" / "1" / "swap_config.json").unlink()
    _, problems = plan["read"](required=())
    assert len(problems) == 2
    assert "configs/2/serve_and_compare.py does not exist" in problems[0]
    assert "configs/1/swap_config.json must exist" in problems[1]


def test_a_cache_inside_a_package_cache_is_refused(plan):
    config(plan["sd"], 4, "serve_and_compare_container.py",
           plan["home"] / ".cache" / "tt-model" / "qwen3.8-27b-p300x2" / "tensors")
    _, problems = plan["read"](required=(2,))
    assert len(problems) == 1 and "a cache another model uses" in problems[0], problems


def test_two_configurations_with_one_cache_are_refused(plan):
    config(plan["sd"], 1, "serve_and_compare.py", plan["caches"][2])
    _, problems = plan["read"]()
    assert problems == [f"tests[2]: tt_cache {plan['caches'][2]} is also the 2-chip configuration's; "
                        "each configuration needs its own"]


def test_a_relative_cache_and_a_bad_handoff_note_are_refused(plan):
    config(plan["sd"], 1, "serve_and_compare.py", "cache/tt_cache")
    (plan["sd"] / "handoff.json").write_text(json.dumps({**NOTE, "goal": ""}))
    _, problems = plan["read"]()
    assert problems[0].startswith("handoff.json needs goal")
    assert "needs an absolute tt_cache" in problems[1]


def test_the_plan_round_trips_and_pending_follows_the_records(plan):
    tests, _ = plan["read"]()
    write_plan(plan["sd"], tests)
    assert load_plan(plan["sd"]) == tests
    assert pending(plan["sd"], tests) == tests
    write_record(plan["sd"], 1, {"returncode": 0, "timed_out": False, "chips": ["a"]})
    assert [t.chips for t in pending(plan["sd"], tests)] == [2, 4]
    summary = json.loads(write_summary(plan["sd"], tests).read_text())
    assert summary["tests"] == [{"returncode": 0, "timed_out": False, "chips": ["a"]},
                                {"chips": 2, "missing": True}, {"chips": 4, "missing": True}]


def test_an_unreadable_plan_loads_as_none(plan):
    assert load_plan(plan["sd"]) is None
    (plan["sd"] / "tests").mkdir()
    (plan["sd"] / "tests" / "plan.json").write_text('{"tests": [{"chips": 2}]}')
    assert load_plan(plan["sd"]) is None


def entry(event, **data):
    return {"event": event, "stage": 4, "data": data}


def test_a_cache_is_suspect_while_its_latest_test_has_not_exited_0():
    started = lambda c: entry("decision", decision="hardware test started", cache=c)      # noqa: E731
    ended = lambda c, rc, to=False: entry("evidence", what="hardware test", cache=c,      # noqa: E731
                                           returncode=rc, timed_out=to)
    assert suspect_caches([started("/a")], 4) == {"/a"}                      # killed mid-test
    assert suspect_caches([started("/a"), ended("/a", 0)], 4) == set()
    assert suspect_caches([started("/a"), ended("/a", 4)], 4) == {"/a"}
    assert suspect_caches([started("/a"), ended("/a", None, True)], 4) == {"/a"}
    assert suspect_caches([started("/a"), ended("/a", 4), started("/a"), ended("/a", 0)], 4) == set()
    other = dict(started("/b"), stage=2)
    assert suspect_caches([other], 4) == set()


def test_move_aside_keeps_the_cache_under_a_new_name_and_leaves_an_empty_one(tmp_path):
    cache = tmp_path / "tt_cache"
    assert move_aside(cache) is None
    cache.mkdir()
    assert move_aside(cache) is None and cache.is_dir()
    (cache / "layer0.bin").write_text("half written")
    aside = move_aside(cache)
    assert aside == tmp_path / "tt_cache.interrupted-1" and (aside / "layer0.bin").is_file()
    assert not cache.exists()
    (cache).mkdir()
    (cache / "x").write_text("again")
    assert move_aside(cache) == tmp_path / "tt_cache.interrupted-2"


def test_a_test_needs_whole_boards():
    assert [HwTest(n, "s", 1.0, "/c").boards for n in (1, 2, 3, 4)] == [1, 1, 2, 2]


def test_a_record_counts_only_with_the_sha256_the_ledger_recorded(tmp_path):
    from orchard.hwtests import unrecorded
    from orchard.stages import evidence_record
    run = tmp_path / "run"
    sd = run / "stages" / "4"
    path = write_record(sd, 2, {"returncode": 0, "timed_out": False, "chips": ["a", "b"]})
    good = [{"event": "evidence", "stage": 4, "data": {"what": "hardware test", "config": 2,
                                                       **evidence_record(run, path)}}]
    assert unrecorded(good, sd, run, 4) == []
    assert unrecorded([], sd, run, 4) == ["stages/4/tests/2/test-result.json was not written by the "
                                          "supervisor: its sha256 is not the one the ledger recorded"]
    path.write_text(path.read_text().replace('"returncode": 0', '"returncode": 0 '))
    assert len(unrecorded(good, sd, run, 4)) == 1


def test_failed_tests_names_each_test_that_must_pass_and_did_not(plan):
    from orchard.hwtests import failed_tests
    tests, _ = plan["read"]()
    assert failed_tests(plan["sd"], (2, 4)) == {"configs": [], "problem": "tests/plan.json could not be read"}
    write_plan(plan["sd"], tests)
    ok = {"returncode": 0, "timed_out": False}
    write_record(plan["sd"], 1, {**ok, "returncode": 4, "chips": ["a"]})
    write_record(plan["sd"], 2, {**ok, "chips": ["a", "b"]})
    write_record(plan["sd"], 4, {**ok, "chips": ["a", "b", "c", "d"]})
    assert failed_tests(plan["sd"], (2, 4)) is None                  # only the optional test failed
    assert failed_tests(plan["sd"], None) == {"configs": [1], "problem": "the 1-chip configuration: its test exited 4"}
    write_record(plan["sd"], 4, {"returncode": None, "timed_out": True, "chips": ["a", "b", "c", "d"]})
    got = failed_tests(plan["sd"], (2, 4))
    assert got["configs"] == [4] and "did not finish before its deadline" in got["problem"]
