# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The virtual local spend (orchard/virtualspend.py): reading the agent logs and pricing the tokens.

The logs here have the shape `AgentStep._log` writes (one object per turn with `usage`) and the benchmark proxy's
`requests.jsonl`. Hostile inputs (a malformed line, bytes that are not UTF-8, a huge line, a turn with no usage, a mix of
local and helper turns) each have a test that names what is counted and what is reported as left out."""
import json
import math
import os
import shutil

import pytest

from orchard import virtualspend as vs


def turn(n=1, p=1000, c=100, **extra):
    rec = {"turn": n, "sent": [{"role": "user", "content": "x"}], "received": {"role": "assistant", "content": "y"},
           "finish_reason": "stop", "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}}
    rec.update(extra)
    return json.dumps(rec)


def write_log(root, rel, lines, *, raw=False):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(l if isinstance(l, bytes) else l.encode() + b"\n" for l in lines) if raw else "\n".join(lines) + "\n"
    if raw:
        path.write_bytes(data)
    else:
        path.write_text(data)
    return path


@pytest.fixture
def root(tmp_path):
    return tmp_path / "probe"


# ---- counting ----------------------------------------------------------------------------------------------------

def test_agent_turns_are_counted_from_their_usage(root):
    write_log(root, "op/attempt-1/log/probe-00001.jsonl", [turn(1, 1000, 100), turn(2, 2000, 50)])
    write_log(root, "op/attempt-2/log/probe-00001.jsonl", [turn(1, 500, 10)])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"], s["files"]) == (3, 3500, 160, 2)
    assert s["sources"] == {"agent": 3, "benchmark": 0} and s["complete"] is True
    assert not any(s["skipped"].values())


def test_a_log_outside_a_log_folder_is_not_an_agent_log(root):
    write_log(root, "op/attempt-1/notes/probe.jsonl", [turn()])
    write_log(root, "op/attempt-1/ledger.jsonl", [turn()])
    assert vs.scan(root)["turns"] == 0


def test_the_helpers_own_ledger_is_never_read(root):
    write_log(root, "telegraph.jsonl", [turn()])
    write_log(root, "log/telegraph.jsonl", [json.dumps({"call_id": "c1", "outcome": "ok", "tokens_in": 9, "tokens_out": 9})])
    s = vs.scan(root)
    assert s["turns"] == 0 and s["prompt_tokens"] == 0
    assert s["skipped"]["helper_turns"] == 1                      # the second file is under log/, and its record is a helper call


def test_mixed_local_and_helper_turns_count_only_the_local_ones(root):
    write_log(root, "log/run-1.jsonl", [
        turn(1, 100, 10),
        turn(2, 9000, 900, call_id="c-1"),                        # a helper call record
        turn(3, 9000, 900, origin="telegraph"),
        turn(4, 9000, 900, source="Helper"),
        turn(5, 9000, 900, helper=True),
        turn(6, 200, 20, source="local"),                         # another source name stays local
        turn(7, 300, 30, helper=False)])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (3, 600, 60)
    assert s["skipped"]["helper_turns"] == 4


def test_a_malformed_line_is_skipped_and_counted(root):
    write_log(root, "log/run-1.jsonl", [turn(1, 10, 1), "{not json", "[1, 2]", "\"text\"", turn(2, 20, 2)])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"]) == (2, 30)
    assert s["skipped"]["malformed_lines"] == 3


def test_an_empty_file_and_blank_lines_give_nothing_and_no_error(root):
    write_log(root, "log/empty.jsonl", [])
    (root / "log" / "blank.jsonl").write_text("\n\n   \n")
    s = vs.scan(root)
    assert s["turns"] == 0 and s["files"] == 2 and not any(s["skipped"].values())


def test_a_turn_with_no_usable_usage_is_counted_as_skipped(root):
    bad = [json.dumps({"turn": 1}), json.dumps({"turn": 2, "usage": {}}), json.dumps({"turn": 3, "usage": None}),
           json.dumps({"turn": 4, "usage": {"prompt_tokens": None, "completion_tokens": 5}}),
           json.dumps({"turn": 5, "usage": {"prompt_tokens": "7", "completion_tokens": 5}}),
           json.dumps({"turn": 6, "usage": {"prompt_tokens": True, "completion_tokens": 5}}),
           json.dumps({"turn": 7, "usage": {"prompt_tokens": -1, "completion_tokens": 5}}),
           json.dumps({"turn": 8, "usage": {"prompt_tokens": 10 ** 12, "completion_tokens": 5}}),
           json.dumps({"turn": 9, "usage": {"prompt_tokens": 1.5, "completion_tokens": 5}}),
           json.dumps({"turn": 10, "usage": [1, 2]})]
    write_log(root, "log/run-1.jsonl", [turn(11, 10, 1)] + bad)
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (1, 10, 1)
    assert s["skipped"]["no_usage"] == 10


def test_bytes_that_are_not_utf8_do_not_stop_the_scan(root):
    good = turn(1, 10, 1).encode() + b"\n"
    broken_but_parseable = (turn(2, 20, 2)[:-1] + ', "note": "caf\xe9"}').encode("latin-1") + b"\n"
    garbage = b"\xff\xfe\x00\x80 not a record\n"
    write_log(root, "log/run-1.jsonl", [good, broken_but_parseable, garbage, good], raw=True)
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"]) == (3, 40)           # the line with one bad byte is still a turn
    assert s["skipped"]["not_utf8_lines"] == 2 and s["skipped"]["malformed_lines"] == 1


def test_a_huge_line_is_not_parsed_but_its_usage_is_read_from_its_end(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 5000)
    big = turn(1, 777, 7, sent=["x" * 40000])                    # `usage` is the last key, as the agent writes it
    write_log(root, "log/run-1.jsonl", [turn(2, 10, 1), big, turn(3, 20, 2)])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (3, 807, 10)
    assert s["skipped"]["long_lines"] == 1 and s["skipped"]["no_usage"] == 0


def test_a_huge_line_with_no_usage_at_its_end_is_skipped(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 5000)
    write_log(root, "log/run-1.jsonl", ['{"turn": 1, "sent": "' + "x" * 40000 + '"}', turn(2, 20, 2)])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"]) == (1, 20)
    assert s["skipped"]["long_lines"] == 1 and s["skipped"]["no_usage"] == 1


def test_a_line_exactly_at_the_cap_is_still_parsed(root, monkeypatch):
    line = turn(1, 5, 1)
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", len(line) + 1)     # the newline is part of the line read
    write_log(root, "log/run-1.jsonl", [line])
    s = vs.scan(root)
    assert s["turns"] == 1 and s["skipped"]["long_lines"] == 0


def test_a_file_over_the_read_cap_is_read_up_to_it_and_the_scan_says_so(root, monkeypatch):
    one = turn(1, 10, 1)
    monkeypatch.setattr(vs, "MAX_FILE_BYTES", (len(one) + 1) * 3)
    write_log(root, "log/run-1.jsonl", [one] * 10)
    s = vs.scan(root)
    assert s["turns"] == 3 and s["skipped"]["oversize_files"] == 1 and s["complete"] is False


def test_the_total_read_budget_stops_a_scan_of_many_files(root, monkeypatch):
    one = turn(1, 10, 1)
    monkeypatch.setattr(vs, "MAX_TOTAL_BYTES", (len(one) + 1) * 4)
    for i in range(5):
        write_log(root, f"a{i}/log/run.jsonl", [one, one])
    s = vs.scan(root)
    assert 0 < s["turns"] <= 4 and s["skipped"]["oversize_files"] >= 1 and s["complete"] is False


def test_too_many_files_are_counted_not_read(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_FILES", 3)
    for i in range(5):
        write_log(root, f"a{i}/log/run.jsonl", [turn(1, 10, 1)])
    s = vs.scan(root)
    assert s["files"] == 3 and s["turns"] == 3 and s["skipped"]["files_not_read"] >= 1 and s["complete"] is False


def test_a_deep_folder_is_not_walked_past_the_depth_limit(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_DEPTH", 2)
    write_log(root, "a/b/log/run.jsonl", [turn()])               # log folder at depth 3: not reached
    write_log(root, "a/log/run.jsonl", [turn()])                 # depth 2: reached
    assert vs.scan(root)["turns"] == 1


def test_links_are_never_followed(root, tmp_path):
    outside = tmp_path / "outside"
    write_log(outside, "log/run.jsonl", [turn(1, 999, 9)])
    root.mkdir(parents=True)
    (root / "linked").symlink_to(outside)                       # a folder link
    (root / "log").mkdir()
    (root / "log" / "run.jsonl").symlink_to(outside / "log" / "run.jsonl")     # a file link
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["links"] == 2


def test_a_scan_of_a_link_or_a_missing_folder_is_empty_and_incomplete(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "ln").symlink_to(tmp_path / "real")
    for p in (tmp_path / "ln", tmp_path / "missing", tmp_path / "file"):
        if p.name == "file":
            p.write_text("x")
        s = vs.scan(p)
        assert s["turns"] == 0 and s["complete"] is False


def test_an_unreadable_file_is_counted_and_the_rest_are_read(root):
    ok = write_log(root, "a/log/run.jsonl", [turn(1, 10, 1)])
    dead = write_log(root, "b/log/run.jsonl", [turn(1, 50, 5)])
    os.chmod(dead, 0)
    try:
        if os.access(dead, os.R_OK):
            pytest.skip("running as a user that can read mode 0 files")
        s = vs.scan(root)
        assert s["turns"] == 1 and s["skipped"]["unreadable_files"] == 1
    finally:
        os.chmod(dead, 0o644)
    assert ok.exists()


def test_a_deeply_nested_json_line_is_malformed_not_a_crash(root):
    write_log(root, "log/run.jsonl", ["[" * 100000 + "]" * 100000, turn(1, 5, 1)])
    s = vs.scan(root)
    assert s["turns"] == 1 and s["skipped"]["malformed_lines"] == 1


# ---- times ---------------------------------------------------------------------------------------------------------

def test_a_kernel_probe_attempt_gives_its_agent_wall_seconds_once_per_attempt(root):
    write_log(root, "op/attempt-1/log/probe-00001.jsonl", [turn()])
    write_log(root, "op/attempt-1/log/probe-00002.jsonl", [turn()])
    (root / "op" / "attempt-1" / "attempt.json").write_text(json.dumps({"agent_wall_s": 120.5}))
    write_log(root, "op/attempt-2/log/probe-00001.jsonl", [turn()])
    (root / "op" / "attempt-2" / "attempt.json").write_text(json.dumps({"agent_wall_s": 30}))
    s = vs.scan(root)
    assert s["agent_wall_s"] == 150.5 and s["request_s"] is None


@pytest.mark.parametrize("content", ["not json", json.dumps({"agent_wall_s": "9"}), json.dumps({"agent_wall_s": True}),
                                     json.dumps({"agent_wall_s": -5}), json.dumps({"agent_wall_s": float("nan")}),
                                     json.dumps([1]), ""])
def test_a_bad_attempt_record_gives_no_time(root, content):
    write_log(root, "op/attempt-1/log/probe-00001.jsonl", [turn()])
    (root / "op" / "attempt-1" / "attempt.json").write_text(content)
    s = vs.scan(root)
    assert s["agent_wall_s"] is None and s["turns"] == 1


def test_an_attempt_record_that_is_a_link_or_huge_is_ignored(root, tmp_path):
    write_log(root, "op/attempt-1/log/p.jsonl", [turn()])
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"agent_wall_s": 99}))
    (root / "op" / "attempt-1" / "attempt.json").symlink_to(real)
    assert vs.scan(root)["agent_wall_s"] is None
    (root / "op" / "attempt-1" / "attempt.json").unlink()
    (root / "op" / "attempt-1" / "attempt.json").write_text(json.dumps({"agent_wall_s": 99, "pad": "x" * (2 * 1024 * 1024)}))
    assert vs.scan(root)["agent_wall_s"] is None


def req(p=100, c=10, lat=1.5, **extra):
    rec = {"t": 1.0, "path": "/v1/chat/completions", "status": 200, "error": False, "latency_s": lat,
           "prompt_tokens": p, "completion_tokens": c, "finish_reason": "stop"}
    rec.update(extra)
    return json.dumps(rec)


def test_benchmark_requests_are_counted_with_their_summed_request_seconds(root):
    write_log(root, "bench/gsm8k/requests.jsonl", [req(100, 10, 1.5), req(200, 20, 2.5), req(None, 5), "oops"])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (2, 300, 30)
    assert s["sources"] == {"agent": 0, "benchmark": 2} and s["request_s"] == 4.0 and s["agent_wall_s"] is None
    assert s["skipped"]["no_usage"] == 1 and s["skipped"]["malformed_lines"] == 1


@pytest.mark.parametrize("lat", [None, "3", True, -1, float("nan"), float("inf")])
def test_a_bad_latency_counts_the_request_with_no_time(root, lat):
    write_log(root, "bench/requests.jsonl", [req(10, 1, lat)])
    s = vs.scan(root)
    assert s["turns"] == 1 and s["request_s"] == 0.0


# ---- price and config ----------------------------------------------------------------------------------------------

def test_the_cost_is_tokens_times_price_per_million():
    assert vs.cost(3_373_905, 78_468, 0.10, 0.50) == pytest.approx(0.3766245)
    assert vs.cost(3_373_905, 78_468, 2.0, 10.0) == pytest.approx(7.53249)
    assert vs.cost(0, 0, 2.0, 10.0) == 0


def test_priced_lists_every_reference_and_labels_the_figure_virtual(root):
    write_log(root, "log/run.jsonl", [turn(1, 1_000_000, 100_000)])
    cfg = vs.Config(extra=(("claude-sonnet-5-5", 2.0, 10.0),))
    out = vs.priced(vs.scan(root), cfg)
    assert [r["name"] for r in out["virtual"]] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert out["virtual"][0]["cost_usd"] == pytest.approx(0.15) and out["virtual"][1]["cost_usd"] == pytest.approx(3.0)
    assert "virtual" in out["label"] and "never added" in out["label"]
    assert out["turns"] == 1 and out["prompt_tokens"] == 1_000_000          # the scan's own fields come through


def test_no_table_gives_the_defaults():
    c = vs.load(None)
    assert (c.reference, c.price_in, c.price_out, c.extra) == ("claude-haiku-5-5", 0.10, 0.50, ())
    assert vs.load({}) == c


def test_a_table_sets_the_reference_and_extra_prices():
    c = vs.load({"reference": "my-model", "price_in": 1, "price_out": 4.5,
                 "extra": [{"name": "big", "price_in": 2.0, "price_out": 10}]})
    assert (c.reference, c.price_in, c.price_out) == ("my-model", 1.0, 4.5)
    assert c.extra == (("big", 2.0, 10.0),)
    assert c.references() == [("my-model", 1.0, 4.5), ("big", 2.0, 10.0)]


@pytest.mark.parametrize("table,needle", [
    ({"referance": "x"}, "did you mean 'reference'"),
    ({"bogus": 1}, "unknown key 'bogus'"),
    ({"reference": ""}, "virtual_spend.reference"),
    ({"reference": 5}, "virtual_spend.reference"),
    ({"reference": "x" * 61}, "virtual_spend.reference"),
    ({"price_in": -1}, "virtual_spend.price_in"),
    ({"price_in": True}, "virtual_spend.price_in"),
    ({"price_in": "1"}, "virtual_spend.price_in"),
    ({"price_out": float("nan")}, "virtual_spend.price_out"),
    ({"price_out": float("inf")}, "virtual_spend.price_out"),
    ({"extra": "x"}, "virtual_spend.extra must be a list"),
    ({"extra": ["x"]}, "virtual_spend.extra must be a list"),
    ({"extra": [{"name": "a", "price_in": 1}]}, "needs price_out"),
    ({"extra": [{"name": "a", "price_in": 1, "price_out": 2, "color": 1}]}, "unknown key 'color'"),
    ({"extra": [{"name": "a", "price_in": -1, "price_out": 2}]}, "extra.price_in"),
    ({"extra": [{"name": "", "price_in": 1, "price_out": 2}]}, "extra.name"),
    ({"extra": [{"name": "claude-haiku-5-5", "price_in": 1, "price_out": 2}]}, "twice"),
    ({"extra": [{"name": "a", "price_in": 1, "price_out": 2}, {"name": "a", "price_in": 1, "price_out": 2}]}, "twice"),
    ({"extra": [{"name": f"n{i}", "price_in": 1, "price_out": 2} for i in range(9)]}, "more than 8"),
])
def test_a_bad_table_is_refused_with_the_key_named(table, needle):
    with pytest.raises(vs.ConfigError) as exc:
        vs.load(table)
    assert needle in str(exc.value)


def test_a_table_that_is_not_a_table_is_refused():
    with pytest.raises(vs.ConfigError, match="must be a table"):
        vs.load([1])


def test_the_reference_name_is_stripped_and_zero_prices_are_allowed():
    c = vs.load({"reference": "  local-free  ", "price_in": 0, "price_out": 0})
    assert c.reference == "local-free" and c.price_in == 0.0


def test_the_bringup_config_reads_and_validates_the_table(tmp_path):
    from orchard import bringup_config as bc
    base = ('runs_root = "runs"\n[coder]\ntarget = "org/coder"\nkind = "bundle"\nprofile = "p300"\nport = 8001\nchips = 2\n')
    f = tmp_path / "bringup.toml"
    f.write_text(base)
    try:
        default = bc.load(f).virtual_spend
    except bc.BringupConfigError as exc:
        pytest.skip(f"this minimal config is not valid here: {exc}")
    assert default.reference == "claude-haiku-5-5"
    f.write_text(base + '[virtual_spend]\nreference = "x"\nprice_in = 3\nprice_out = 9\n')
    assert bc.load(f).virtual_spend.reference == "x"
    f.write_text(base + '[virtual_spend]\nprice_in = -3\n')
    with pytest.raises(bc.BringupConfigError, match="virtual_spend.price_in"):
        bc.load(f)


def test_load_file_reads_the_table_and_reports_a_bad_file(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[virtual_spend]\nreference = "r"\n')
    assert vs.load_file(f).reference == "r"
    f.write_text("[other]\nx = 1\n")
    assert vs.load_file(f) == vs.Config()
    f.write_text("[[[")
    with pytest.raises(vs.ConfigError, match="cannot parse"):
        vs.load_file(f)
    f.write_bytes(b"\xff\xfe")
    with pytest.raises(vs.ConfigError, match="cannot parse"):
        vs.load_file(f)
    with pytest.raises(vs.ConfigError, match="cannot read"):
        vs.load_file(tmp_path / "missing.toml")


def test_skipped_text_names_each_kind_and_is_empty_when_nothing_was_left_out():
    assert vs.skipped_text({k: 0 for k in vs._skipped()}) == ""
    s = vs._skipped()
    s.update(malformed_lines=2, helper_turns=1)
    text = vs.skipped_text(s)
    assert "2 lines that are not JSON" in text and "1 turns from an outside helper" in text and text.startswith("Left out:")
    for key in vs._skipped():
        one = vs._skipped()
        one[key] = 3
        assert "3 " in vs.skipped_text(one), key


def test_a_non_finite_total_is_not_possible_from_the_bounds():
    assert math.isfinite(vs.cost(vs.MAX_TOKENS_PER_TURN, vs.MAX_TOKENS_PER_TURN, 1e6, 1e6))



# ---- cases added after the first mutation table ----------------------------------------------------------------------

def test_a_huge_benchmark_line_is_skipped_even_when_it_holds_a_usage_object(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 5000)
    big = json.dumps({"prompt_tokens": 5, "note": "x" * 40000, "usage": {"prompt_tokens": 999, "completion_tokens": 9}})
    write_log(root, "bench/requests.jsonl", [req(10, 1), big])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"]) == (1, 10) and s["skipped"]["long_lines"] == 1


def test_usage_that_straddles_the_chunks_of_a_huge_line_is_still_found(root, monkeypatch):
    cap = 1000
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", cap)
    head, tail = turn(1, 424242, 4242, sent=["@@"]).split("@@")
    # The first read takes cap+1 bytes and each later read takes cap, so a chunk boundary falls at 2*cap+1.
    # Pad so that boundary lands inside the usage object, which is the end of the line.
    pad = (2 * cap + 1) - 10 - len(head) - tail.index('"usage"')
    line = head + "x" * pad + tail
    start = line.index('"usage"')
    assert start < 2 * cap + 1 < len(line) - 5, "the test line must put a chunk boundary inside usage"
    write_log(root, "log/run.jsonl", [line])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (1, 424242, 4242)


def test_only_a_file_named_requests_jsonl_is_a_benchmark_log(root):
    write_log(root, "bench/old-requests.jsonl", [req(10, 1)])
    write_log(root, "bench/requests.jsonl.bak", [req(10, 1)])
    assert vs.scan(root)["turns"] == 0


def test_a_benchmark_file_with_no_usable_request_has_no_request_seconds(root):
    write_log(root, "bench/requests.jsonl", [req(None, 5), req(5, None)])
    s = vs.scan(root)
    assert s["turns"] == 0 and s["request_s"] is None and s["skipped"]["no_usage"] == 2



# ---- review fix 2: a huge integer latency keeps the request's tokens -------------------------------------------------

def test_a_huge_integer_latency_is_no_time_and_the_tokens_are_kept(root):
    huge = '{"prompt_tokens": 100, "completion_tokens": 10, "latency_s": 1' + "0" * 400 + "}"
    write_log(root, "bench/requests.jsonl", [huge, req(5, 1, 2.0)])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (2, 105, 11)
    assert s["request_s"] == 2.0



# ---- review fix 5: a hostile attempt.json gives no time and the log counts stay -----------------------------------------

@pytest.mark.parametrize("content", ["[" * 100000 + "]" * 100000, '{"agent_wall_s": 1' + "0" * 400 + "}",
                                     '{"a": ' * 50000 + "1" + "}" * 50000], ids=["deep-list", "big-int", "deep-object"])
def test_a_hostile_attempt_record_gives_no_time_and_the_turn_counts_stay(root, content):
    write_log(root, "op/attempt-1/log/probe-00001.jsonl", [turn(1, 70, 7)])
    (root / "op" / "attempt-1" / "attempt.json").write_text(content)
    s = vs.scan(root)
    assert s["agent_wall_s"] is None and (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (1, 70, 7)



# ---- review fix 3: every read stays inside the remaining file and scan budgets -----------------------------------------

class _Spy:
    """Counts the bytes a scan takes from the files it opens."""

    def __init__(self):
        self.bytes = 0

    def install(self, monkeypatch):
        real_open = vs._open_log
        spy = self

        class Wrapped:
            def __init__(self, fh):
                self._fh = fh

            def readline(self, size=-1):
                data = self._fh.readline(size)
                spy.bytes += len(data)
                return data

            def peek(self, n=1):
                return self._fh.peek(n)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._fh.close()

        monkeypatch.setattr(vs, "_open_log", lambda *a, **k: (lambda fh: Wrapped(fh) if fh is not None else None)(real_open(*a, **k)))


def test_one_long_line_cannot_make_a_scan_read_past_the_file_budget(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 1000)
    monkeypatch.setattr(vs, "MAX_FILE_BYTES", 3000)
    write_log(root, "log/run.jsonl", [turn(1, 5, 1, sent=["x" * 200000])])
    spy = _Spy()
    spy.install(monkeypatch)
    s = vs.scan(root)
    assert spy.bytes <= 3000, spy.bytes
    assert s["turns"] == 0 and s["skipped"]["oversize_files"] == 1, "a record cut off by the budget is not priced"


def test_one_long_line_cannot_make_a_scan_read_past_the_total_budget(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 1000)
    monkeypatch.setattr(vs, "MAX_TOTAL_BYTES", 4000)
    write_log(root, "a/log/run.jsonl", [turn(1, 5, 1, sent=["x" * 3000])])         # fits, as a long line
    write_log(root, "b/log/run.jsonl", [turn(1, 7, 1, sent=["x" * 200000])])       # cut off by the budget
    spy = _Spy()
    spy.install(monkeypatch)
    s = vs.scan(root)
    assert spy.bytes <= 4000, spy.bytes
    assert (s["turns"], s["prompt_tokens"]) == (1, 5) and s["skipped"]["oversize_files"] == 1 and s["complete"] is False


def test_a_last_line_without_a_newline_that_ends_exactly_at_the_budget_is_kept(root, monkeypatch):
    line = turn(1, 11, 1)
    monkeypatch.setattr(vs, "MAX_FILE_BYTES", len(line))
    (root / "log").mkdir(parents=True)
    (root / "log" / "run.jsonl").write_text(line)                                    # no trailing newline
    s = vs.scan(root)
    assert s["turns"] == 1 and s["skipped"]["oversize_files"] == 0


def test_a_short_line_cut_by_the_budget_is_not_priced(root, monkeypatch):
    one = turn(1, 10, 1)
    monkeypatch.setattr(vs, "MAX_FILE_BYTES", len(one) + 1 + 20)                      # the second line is cut after 20 bytes
    write_log(root, "log/run.jsonl", [one, one])
    s = vs.scan(root)
    assert s["turns"] == 1 and s["skipped"]["oversize_files"] == 1 and s["skipped"]["malformed_lines"] == 0



# ---- review fix 4: the bounded suffix is parsed with a JSON decoder, so nested usage details stay included -------------

def _long(usage, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    return turn(1, 0, 0, sent=["x" * 30000], usage=usage)


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 900, "completion_tokens": 90, "completion_tokens_details": {"reasoning_tokens": 40}},
    {"completion_tokens_details": {"reasoning_tokens": 40, "audio": {"n": 1}}, "prompt_tokens": 900, "completion_tokens": 90},
    {"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": {"a": [1, 2, {"b": 3}]}}, "completion_tokens": 90,
     "completion_tokens_details": None},
], ids=["details-after", "details-before", "nested-both-sides"])
def test_a_huge_turn_with_nested_usage_details_is_still_priced(root, monkeypatch, usage):
    write_log(root, "log/run.jsonl", [_long(usage, monkeypatch)])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (1, 900, 90)
    assert s["skipped"]["long_lines"] == 1 and s["skipped"]["no_usage"] == 0


def test_a_huge_turn_whose_usage_is_cut_off_is_not_priced(root, monkeypatch):
    line = _long({"prompt_tokens": 900, "completion_tokens": 90, "completion_tokens_details": {"reasoning_tokens": 40}}, monkeypatch)
    cut = line[:line.index('"reasoning_tokens"') + 5]                    # ends inside the nested object
    write_log(root, "log/run.jsonl", [cut])
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["no_usage"] == 1


def test_a_usage_key_inside_an_earlier_value_does_not_count_for_the_record(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    inner = '\\"usage\\": {\\"prompt_tokens\\": 1, \\"completion_tokens\\": 1}'      # inside a string, with more text after it
    line = '{"turn": 1, "sent": ["' + "x" * 30000 + inner + '"], "received": {"role": "assistant"}, "finish_reason": "stop"}'
    write_log(root, "log/run.jsonl", [line])
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["no_usage"] == 1


def test_a_deeply_nested_usage_suffix_is_not_a_crash(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    line = '{"turn": 1, "sent": "' + "x" * 30000 + '", "usage": ' + "[" * 3000 + "]" * 3000 + "}"
    write_log(root, "log/run.jsonl", [line])
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["no_usage"] == 1



# ---- review fix 6: an over-long record gets the helper filter from its first and last bytes ---------------------------------
# Choice: the first 4 KiB and the last 8 KiB of an over-long line are searched for helper markers, and the line must start the way
# the agent writes a turn (`{"turn"`). A line that does not is skipped and counted as `long_unverified`. A marker in the middle of
# a record over the line cap cannot be seen.

def _big(before=None, after=None, lead="turn"):
    """An over-long turn: `lead` key first, optional extra keys before the big value, and optional keys after `usage`."""
    rec = {lead: 1}
    rec.update(before or {})
    rec["sent"] = ["x" * 30000]
    rec["received"] = {"role": "assistant"}
    rec["finish_reason"] = "stop"
    rec["usage"] = {"prompt_tokens": 900, "completion_tokens": 90}
    rec.update(after or {})
    return json.dumps(rec)


@pytest.mark.parametrize("before,after", [
    ({"call_id": "c-1"}, None),
    ({"origin": "telegraph"}, None),
    ({"source": "Helper"}, None),
    ({"helper": True}, None),
    (None, {"call_id": "c-1"}),
    (None, {"origin": "telegraph"}),
    (None, {"helper": True}),
], ids=["call_id-head", "origin-head", "source-head", "helper-head", "call_id-tail", "origin-tail", "helper-tail"])
def test_an_over_long_helper_record_is_not_priced(root, monkeypatch, before, after):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    write_log(root, "log/run.jsonl", [_big(before, after), _big()])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"]) == (1, 900)                  # only the plain record
    assert s["skipped"]["helper_turns"] == 1 and s["skipped"]["long_lines"] == 2


def test_false_or_local_markers_never_make_an_over_long_record_a_helper_turn(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    write_log(root, "log/run.jsonl", [_big({"helper": False}), _big({"source": "local"}), _big({"helper": None}),
                                      _big({"origin": "agent"})])
    s = vs.scan(root)
    assert s["skipped"]["helper_turns"] == 0 and (s["turns"], s["prompt_tokens"]) == (4, 3600)


def test_an_over_long_line_that_does_not_start_like_an_agent_turn_is_unverified_and_not_priced(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    write_log(root, "log/run.jsonl", [_big(lead="note")])
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["long_unverified"] == 1 and s["skipped"]["long_lines"] == 1


def test_the_same_record_is_treated_alike_below_and_above_the_line_cap(root, monkeypatch):
    short = json.dumps({"turn": 1, "call_id": "c-1", "usage": {"prompt_tokens": 900, "completion_tokens": 90}})
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    write_log(root, "a/log/run.jsonl", [short])
    write_log(root, "b/log/run.jsonl", [_big({"call_id": "c-1"})])
    a, b = vs.scan(root / "a"), vs.scan(root / "b")
    assert a["turns"] == b["turns"] == 0 and a["skipped"]["helper_turns"] == b["skipped"]["helper_turns"] == 1


def test_a_marker_in_the_middle_of_an_over_long_record_is_not_seen(root, monkeypatch):
    """The documented limit of the choice: only the first 4 KiB and the last 8 KiB are searched."""
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    rec = {"turn": 1, "sent": ["x" * 20000], "call_id": "mid", "more": ["y" * 20000],
           "usage": {"prompt_tokens": 900, "completion_tokens": 90}}
    write_log(root, "log/run.jsonl", [json.dumps(rec)])
    assert vs.scan(root)["turns"] == 1


def test_left_out_text_names_unverified_long_records():
    s = vs._skipped()
    s["long_unverified"] = 2
    assert "2 lines over the size cap that do not start like an agent turn" in vs.skipped_text(s)



def test_usage_must_be_the_last_key_of_an_over_long_record(root, monkeypatch):
    """Only the closing brace may follow the usage object. A record with another key after it is not priced from its end."""
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    line = json.dumps({"turn": 1, "sent": ["x" * 30000], "usage": {"prompt_tokens": 900, "completion_tokens": 90}, "note": "late"})
    write_log(root, "log/run.jsonl", [line])
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["no_usage"] == 1


def test_the_last_usage_key_wins_over_an_earlier_nested_one(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    line = json.dumps({"turn": 1, "sent": ["x" * 30000], "received": {"role": "assistant", "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                       "finish_reason": "stop", "usage": {"prompt_tokens": 900, "completion_tokens": 90}})
    write_log(root, "log/run.jsonl", [line])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["completion_tokens"]) == (1, 900, 90)



# ---- review fix 2 (second round): the source counts add up to the turn count ------------------------------------------------

def test_sources_add_up_to_the_turns_for_short_recovered_long_and_benchmark_records(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    write_log(root, "a/log/run.jsonl", [turn(1, 10, 1), turn(2, 20, 2)])                        # short agent turns
    write_log(root, "b/log/run.jsonl", [_big()])                                                # a recovered long turn
    write_log(root, "c/requests.jsonl", [req(5, 1), req(6, 1)])                                 # benchmark requests
    s = vs.scan(root)
    assert s["turns"] == 5
    assert s["sources"] == {"agent": 3, "benchmark": 2}
    assert sum(s["sources"].values()) == s["turns"]


def test_a_recovered_long_turn_alone_counts_as_one_agent_source(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_LINE_BYTES", 2000)
    write_log(root, "log/run.jsonl", [_big()])
    s = vs.scan(root)
    assert s["turns"] == 1 and s["sources"] == {"agent": 1, "benchmark": 0}
from types import SimpleNamespace  # noqa: E402



# ---- review fix 1 (third round): logs are opened without following links, and the open is the check -------------------------

def _lie_about_links(monkeypatch):
    """Make the earlier path checks (`is_symlink`, `is_file`) answer as they would before a file was swapped for a link."""
    monkeypatch.setattr(vs.Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(vs.Path, "is_file", lambda self: True)


def test_a_log_replaced_by_a_link_after_discovery_is_refused_when_it_is_opened(root, tmp_path, monkeypatch):
    outside = tmp_path / "outside.jsonl"
    outside.write_text(turn(1, 999, 9) + "\n")
    write_log(root, "a/log/run.jsonl", [turn(1, 5, 1)])
    real_find = vs._find_files

    def find_then_swap(r, skipped, *a):
        found = real_find(r, skipped, *a)
        for p, _ in found:                                                 # the swap happens after discovery
            p.unlink()
            p.symlink_to(outside)
        return found
    monkeypatch.setattr(vs, "_find_files", find_then_swap)
    s = vs.scan(root)
    assert s["turns"] == 0 and s["prompt_tokens"] == 0
    assert s["skipped"]["links"] == 1


def test_a_link_is_refused_even_when_the_path_checks_are_fooled(root, tmp_path, monkeypatch):
    outside = tmp_path / "outside.jsonl"
    outside.write_text(turn(1, 999, 9) + "\n")
    (root / "log").mkdir(parents=True)
    (root / "log" / "run.jsonl").symlink_to(outside)
    _lie_about_links(monkeypatch)
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["links"] >= 1


def test_a_log_that_is_not_a_regular_file_is_refused_when_it_is_opened(root, monkeypatch):
    (root / "log").mkdir(parents=True)
    fifo = root / "log" / "run.jsonl"
    os.mkfifo(fifo)                                                         # opening a fifo for reading would block
    _lie_about_links(monkeypatch)
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["links"] == 1


def test_a_missing_log_is_unreadable_not_a_link(root, monkeypatch):
    write_log(root, "log/run.jsonl", [turn()])
    real_find = vs._find_files

    def find_then_delete(r, skipped, *a):
        found = real_find(r, skipped, *a)
        for p, _ in found:
            p.unlink()
        return found
    monkeypatch.setattr(vs, "_find_files", find_then_delete)
    s = vs.scan(root)
    assert s["skipped"]["unreadable_files"] == 1 and s["skipped"]["links"] == 0


# ---- review fix 2 (third round): attempt.json is opened without following links and read with a limit -------------------------

def test_an_attempt_record_that_is_a_link_is_not_followed_even_when_the_path_checks_are_fooled(root, tmp_path, monkeypatch):
    write_log(root, "op/attempt-1/log/p.jsonl", [turn(1, 5, 1)])
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"agent_wall_s": 99}))
    (root / "op" / "attempt-1" / "attempt.json").symlink_to(real)
    _lie_about_links(monkeypatch)
    s = vs.scan(root)
    assert s["agent_wall_s"] is None and s["turns"] == 1


def test_an_attempt_record_over_the_cap_is_refused_by_the_bytes_read(root):
    write_log(root, "op/attempt-1/log/p.jsonl", [turn(1, 5, 1)])
    (root / "op" / "attempt-1" / "attempt.json").write_text(json.dumps({"agent_wall_s": 99}) + " " * (vs.MAX_ATTEMPT_BYTES + 10))
    s = vs.scan(root)
    assert s["agent_wall_s"] is None and s["turns"] == 1, "the log was scanned and the oversize record gave no time"


def test_an_attempt_record_exactly_at_the_cap_is_read(root):
    write_log(root, "op/attempt-1/log/p.jsonl", [turn(1, 5, 1)])
    body = json.dumps({"agent_wall_s": 99})
    (root / "op" / "attempt-1" / "attempt.json").write_text(body + " " * (vs.MAX_ATTEMPT_BYTES - len(body)))
    assert vs.scan(root)["agent_wall_s"] == 99.0


def test_the_attempt_read_asks_for_at_most_the_cap_plus_one_byte(root, monkeypatch):
    write_log(root, "op/attempt-1/log/p.jsonl", [turn(1, 5, 1)])
    (root / "op" / "attempt-1" / "attempt.json").write_text(json.dumps({"agent_wall_s": 1}))
    asked = []
    real_read = os.read
    monkeypatch.setattr(vs.os, "read", lambda fd, n: (asked.append(n), real_read(fd, n))[1])
    vs.scan(root)
    assert asked and max(asked) <= vs.MAX_ATTEMPT_BYTES + 1


def test_a_log_is_read_through_the_descriptor_that_was_checked(root, tmp_path, monkeypatch):
    """Swap the file for a link at the moment of the descriptor check: the handle must still read the file that was opened."""
    outside = tmp_path / "outside.jsonl"
    outside.write_text(turn(1, 999, 9) + "\n")
    path = write_log(root, "log/run.jsonl", [turn(1, 5, 1)])
    real_fstat = os.fstat

    def swapping_fstat(fd):
        result = real_fstat(fd)
        if path.exists() and not path.is_symlink():
            path.unlink()
            path.symlink_to(outside)
        return result
    monkeypatch.setattr(vs.os, "fstat", swapping_fstat)
    skipped = vs._skipped()
    fh = vs._open_log(path, skipped)
    assert fh is not None
    with fh:
        assert json.loads(fh.readline())["usage"]["prompt_tokens"] == 5


def test_a_large_attempt_record_is_never_read_past_the_cap_plus_one_byte(root, monkeypatch):
    write_log(root, "op/attempt-1/log/p.jsonl", [turn(1, 5, 1)])
    (root / "op" / "attempt-1" / "attempt.json").write_text(json.dumps({"agent_wall_s": 5}) + " " * (3 * vs.MAX_ATTEMPT_BYTES))
    returned = []
    real_read = os.read

    def counting_read(fd, n):
        data = real_read(fd, n)
        returned.append(len(data))
        return data
    monkeypatch.setattr(vs.os, "read", counting_read)
    s = vs.scan(root)
    assert sum(returned) == vs.MAX_ATTEMPT_BYTES + 1, "exactly the cap plus one byte is taken from a larger file"
    assert s["agent_wall_s"] is None


# ---- review fix 1 (fourth round): every directory on the way to a file is opened without following links ------------------

def _swap_dir_for_link_after_discovery(monkeypatch, victim, target):
    real_find = vs._find_files

    def find_then_swap(r, skipped, *a):
        found = real_find(r, skipped, *a)
        shutil.rmtree(victim)                                              # the swap happens after discovery
        victim.symlink_to(target, target_is_directory=True)
        return found
    monkeypatch.setattr(vs, "_find_files", find_then_swap)


def _outside_run(tmp_path):
    outside = tmp_path / "outside"
    write_log(outside, "log/run.jsonl", [turn(1, 999, 9)])
    (outside / "attempt.json").write_text(json.dumps({"agent_wall_s": 777}))
    return outside


def test_a_run_folder_replaced_by_a_link_after_discovery_is_not_read(root, tmp_path, monkeypatch):
    outside = _outside_run(tmp_path)
    write_log(root, "a/log/run.jsonl", [turn(1, 5, 1)])
    (root / "a" / "attempt.json").write_text(json.dumps({"agent_wall_s": 1}))
    _swap_dir_for_link_after_discovery(monkeypatch, root / "a", outside)
    s = vs.scan(root)
    assert s["turns"] == 0 and s["prompt_tokens"] == 0 and s["agent_wall_s"] is None
    assert s["skipped"]["links"] >= 1


def test_the_log_folder_replaced_by_a_link_after_discovery_is_not_read(root, tmp_path, monkeypatch):
    outside = tmp_path / "outside-log"
    write_log(outside, "run.jsonl", [turn(1, 999, 9)])
    write_log(root, "a/log/run.jsonl", [turn(1, 5, 1)])
    _swap_dir_for_link_after_discovery(monkeypatch, root / "a" / "log", outside)
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["links"] >= 1


def test_a_nested_folder_swapped_two_levels_down_is_not_read(root, tmp_path, monkeypatch):
    outside = tmp_path / "outside-deep"
    write_log(outside, "log/run.jsonl", [turn(1, 999, 9)])
    write_log(root, "a/b/c/log/run.jsonl", [turn(1, 5, 1)])
    _swap_dir_for_link_after_discovery(monkeypatch, root / "a" / "b", outside)
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["links"] >= 1


def test_the_scan_root_is_pinned_when_the_scan_starts(root, tmp_path, monkeypatch):
    """The root itself is replaced by a link after the scan has opened it: the pinned descriptor still reads the original."""
    outside = tmp_path / "outside-root"
    write_log(outside, "a/log/run.jsonl", [turn(1, 999, 9)])
    write_log(root, "a/log/run.jsonl", [turn(1, 5, 1)])
    real_find = vs._find_files

    def find_then_swap_root(r, skipped, *a):
        found = real_find(r, skipped, *a)
        moved = r.parent / "moved-away"
        r.rename(moved)
        r.symlink_to(outside, target_is_directory=True)
        return found
    monkeypatch.setattr(vs, "_find_files", find_then_swap_root)
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"]) == (1, 5), "the pinned descriptor reads the original tree, not what the path now names"


def test_an_untouched_tree_is_read_through_the_pinned_descriptors(root):
    write_log(root, "a/b/log/run.jsonl", [turn(1, 5, 1), turn(2, 6, 2)])
    (root / "a" / "b" / "attempt.json").write_text(json.dumps({"agent_wall_s": 2.5}))
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"], s["agent_wall_s"]) == (2, 11, 2.5) and not s["skipped"]["links"]


def test_no_descriptor_is_left_open_after_a_scan(root):
    write_log(root, "a/b/log/run.jsonl", [turn()])
    before = set(os.listdir("/proc/self/fd"))
    vs.scan(root)
    assert set(os.listdir("/proc/self/fd")) == before


def test_a_scan_root_swapped_for_a_link_just_before_it_is_pinned_is_refused(root, tmp_path, monkeypatch):
    outside = tmp_path / "outside-root"
    write_log(outside, "log/run.jsonl", [turn(1, 999, 9)])
    link = tmp_path / "root-link"
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(vs.Path, "is_symlink", lambda self: False)            # the earlier check was passed before the swap
    s = vs.scan(link)
    assert s["turns"] == 0 and s["complete"] is False


def test_a_path_outside_the_pinned_root_cannot_be_opened(root, tmp_path):
    (root).mkdir(parents=True, exist_ok=True)
    other = tmp_path / "elsewhere.jsonl"
    other.write_text("x")
    pin = vs._Pin(root)
    try:
        with pytest.raises(OSError):
            pin.open(other, vs._FILE_FLAGS)
        with pytest.raises(OSError):
            pin.open(root, vs._FILE_FLAGS)                                      # the root itself is not a file
    finally:
        pin.close()


# ---- review (fifth round): discovery of log files streams under an entry budget and stops at the file cap ---------------------

class _WalkSpy:
    """Counts the entries taken from, and the folders opened by, os.scandir inside the reader."""

    def __init__(self, monkeypatch):
        self.entries = 0
        self.opened = 0
        real = os.scandir
        spy = self

        def scandir(path):
            spy.opened += 1
            inner = real(path)

            class Wrapped:
                def __enter__(self_inner):
                    inner.__enter__()
                    return self_inner

                def __exit__(self_inner, *a):
                    return inner.__exit__(*a)

                def __iter__(self_inner):
                    return self_inner

                def __next__(self_inner):
                    entry = next(inner)
                    spy.entries += 1
                    return entry
            return Wrapped()
        monkeypatch.setattr(vs.os, "scandir", scandir)


def test_the_walk_takes_no_more_entries_than_its_budget_and_says_the_scan_is_incomplete(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_WALK_ENTRIES", 10)
    for i in range(30):
        (root / f"d{i:02d}").mkdir(parents=True, exist_ok=True)
        (root / f"f{i:02d}.txt").write_text("x")
    write_log(root, "zz/log/run.jsonl", [turn()])
    spy = _WalkSpy(monkeypatch)
    s = vs.scan(root)
    assert spy.entries <= 10 + 1, spy.entries
    assert s["complete"] is False and s["skipped"]["walk_cut"] >= 1
    assert "directory entries past the walk budget" in vs.skipped_text(s["skipped"])


def test_a_tree_that_fits_the_walk_budget_is_complete(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_WALK_ENTRIES", 50)
    write_log(root, "a/log/run.jsonl", [turn()])
    s = vs.scan(root)
    assert s["complete"] is True and s["skipped"]["walk_cut"] == 0 and s["turns"] == 1


def test_a_tree_exactly_at_the_walk_budget_is_complete(root, monkeypatch):
    write_log(root, "a/log/run.jsonl", [turn()])                              # entries: a, log, run.jsonl = 3
    monkeypatch.setattr(vs, "MAX_WALK_ENTRIES", 3)
    s = vs.scan(root)
    assert s["complete"] is True and s["turns"] == 1


def test_the_walk_stops_opening_folders_after_the_file_cap(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_FILES", 2)
    for i in range(20):
        write_log(root, f"d{i:02d}/log/run.jsonl", [turn()])
    spy = _WalkSpy(monkeypatch)
    s = vs.scan(root)
    assert s["files"] == 2 and s["skipped"]["files_not_read"] >= 1 and s["complete"] is False
    assert spy.opened <= 1 + 3 * 4, f"{spy.opened} folders opened after the cap was reached"


def test_the_walk_still_skips_links_and_respects_the_depth_limit(root, tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "MAX_DEPTH", 2)
    outside = tmp_path / "outside"
    write_log(outside, "log/run.jsonl", [turn(1, 999, 9)])
    (root).mkdir(parents=True)
    (root / "linked").symlink_to(outside, target_is_directory=True)
    write_log(root, "a/log/run.jsonl", [turn(1, 5, 1)])
    write_log(root, "a/b/c/log/run.jsonl", [turn(1, 7, 1)])
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"]) == (1, 5) and s["skipped"]["links"] == 1


def test_the_walk_reads_folders_in_name_order(root):
    for name in ("c", "a", "b"):
        write_log(root, f"{name}/log/run.jsonl", [turn(1, ord(name), 0)])
    assert [p.parts[-3] for p, _ in vs._find_files(root, vs._skipped())] == ["a", "b", "c"]


def test_scanning_the_pinned_folder_itself_leaves_the_pin_usable(root):
    write_log(root, "log/run.jsonl", [turn(1, 5, 1)])
    pin = vs.open_pin(root)
    try:
        assert vs.scan(root, beneath=pin)["turns"] == 1
        assert vs.scan(root, beneath=pin)["turns"] == 1, "the first scan must not have closed the pin's own descriptor"
    finally:
        pin.close()


def test_a_component_refused_after_earlier_ones_were_opened_leaks_no_descriptor(root, tmp_path):
    write_log(root, "a/b/log/run.jsonl", [turn(1, 5, 1)])
    outside = tmp_path / "out"
    outside.mkdir()
    shutil.rmtree(root / "a" / "b")
    (root / "a" / "b").symlink_to(outside, target_is_directory=True)           # the second component is the link
    pin = vs.open_pin(root)
    try:
        before = set(os.listdir("/proc/self/fd"))
        s = vs.scan(root / "a" / "b", beneath=pin)
        assert s["skipped"]["links"] == 1 and set(os.listdir("/proc/self/fd")) == before
    finally:
        pin.close()


def test_the_default_walk_budget_is_large_but_bounded():
    assert 10_000 <= vs.MAX_WALK_ENTRIES <= 1_000_000


# ---- review (sixth round): folders are listed through the pinned descriptor too ----------------------------------------------

def test_a_root_replaced_by_a_tree_with_other_names_after_pinning_is_still_listed_from_the_original(root, tmp_path, monkeypatch):
    write_log(root, "a/log/run.jsonl", [turn(1, 5, 1)])
    other = tmp_path / "other-tree"
    write_log(other, "z/log/different.jsonl", [turn(1, 999, 9)])
    real = vs._scan_tree

    def swap_then_scan(r, pin, out, skipped):
        moved = r.parent / "moved-away"
        r.rename(moved)
        r.symlink_to(other, target_is_directory=True)               # after the pin, before any listing
        return real(r, pin, out, skipped)
    monkeypatch.setattr(vs, "_scan_tree", swap_then_scan)
    s = vs.scan(root)
    assert (s["turns"], s["prompt_tokens"]) == (1, 5) and s["complete"] is True
    assert s["files"] == 1


def test_find_files_lists_through_the_pin_and_names_files_by_their_paths(root):
    write_log(root, "a/log/run.jsonl", [turn()])
    write_log(root, "b/requests.jsonl", [req(1, 1)])
    pin = vs.open_pin(root)
    try:
        found = vs._find_files(root, vs._skipped(), pin)
    finally:
        pin.close()
    assert sorted((p.relative_to(root).as_posix(), k) for p, k in found) == [("a/log/run.jsonl", "agent"), ("b/requests.jsonl", "bench")]


def test_a_folder_swapped_for_a_link_between_listing_and_descending_is_refused(root, tmp_path, monkeypatch):
    write_log(root, "a/b/log/run.jsonl", [turn(1, 5, 1)])
    outside = tmp_path / "out"
    write_log(outside, "log/run.jsonl", [turn(1, 999, 9)])
    real_list = vs._list_dir

    def list_then_swap(source, budget):
        result = real_list(source, budget)
        victim = root / "a" / "b"
        if victim.exists() and not victim.is_symlink() and any(n == "b" for n, *_ in result[0]):
            shutil.rmtree(victim)
            victim.symlink_to(outside, target_is_directory=True)
        return result
    monkeypatch.setattr(vs, "_list_dir", list_then_swap)
    s = vs.scan(root)
    assert s["turns"] == 0 and s["skipped"]["links"] >= 1


def test_find_files_without_a_pin_pins_the_root_itself(root):
    write_log(root, "a/log/run.jsonl", [turn()])
    before = set(os.listdir("/proc/self/fd"))
    assert len(vs._find_files(root, vs._skipped())) == 1
    assert set(os.listdir("/proc/self/fd")) == before
