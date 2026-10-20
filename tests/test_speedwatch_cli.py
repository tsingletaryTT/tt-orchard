# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard coder-speed` (orchard/speedwatch.py `run`, wired in orchard/cli.py).

The logic is tested with a fake post function, a fake clock and a fake sleep, through the real argument parser so a
flag the tests use is a flag the command has. Two tests run the whole command against a local fake server.
"""
import io
import json
from pathlib import Path

import pytest

from orchard import cli
from orchard import speedwatch as sw
from orchard.tiers import TierConfig

URL, MODEL = "http://127.0.0.1:8011/v1", "coder"


@pytest.fixture(autouse=True)
def no_real_metrics_request(monkeypatch):
    """127.0.0.1:8011 is a live coder on the machine these tests were written on. Nothing here may ask it anything:
    unless a test says otherwise, no other request is running."""
    monkeypatch.setattr(sw, "running_requests", lambda endpoint, **kw: 0)


class Server:
    """A fake post and clock. `tok_s` is a list consumed one value per request; the last one repeats."""

    def __init__(self, tok_s, tokens=300, fail=None):
        self.tok_s, self.tokens, self.now, self.calls, self.slept = list(tok_s), tokens, [0.0], [], []
        self.fail = fail

    def post(self, url, body, timeout):
        self.calls.append((url, body["model"]))
        if self.fail and self.fail(len(self.calls)):
            raise OSError("connection refused")
        v = self.tok_s.pop(0) if len(self.tok_s) > 1 else self.tok_s[0]
        self.now[0] += self.tokens / v
        return {"usage": {"completion_tokens": self.tokens}}

    def clock(self):
        return self.now[0]

    def sleep(self, s):
        self.slept.append(s)


def args(*argv):
    return cli._parser().parse_args(["coder-speed", *argv])


def run(tmp_path, server, *argv, tier_config=None, running=None):
    out = io.StringIO()
    code = sw.run(args(*argv), out=out, home=tmp_path, tier_config=tier_config, post=server.post, clock=server.clock,
                  sleep=server.sleep, running=running)
    return code, out.getvalue()


def store(tmp_path):
    return sw.Baselines(tmp_path / ".cache" / "tt-orchard" / "speed-baseline.json")


def target(*extra):
    return ("--endpoint", URL, "--model", MODEL, *extra)


# ---- one-shot --------------------------------------------------------------------------------------------------

def test_the_first_run_records_a_baseline_and_exits_zero(tmp_path):
    code, text = run(tmp_path, Server([40.0]), *target())
    assert code == 0 and "40.0 tok/s" in text and "becomes the baseline" in text
    assert store(tmp_path).get(URL, MODEL) == pytest.approx(40.0)


def test_a_healthy_second_run_says_ok(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    code, text = run(tmp_path, Server([38.0]), *target())
    assert code == 0 and "[ok]" in text and "95%" in text


def test_three_samples_below_half_the_best_exit_three_and_say_slow(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    s = Server([10.0])
    code, text = run(tmp_path, s, *target("--samples", "3", "--interval", "7"))
    assert code == 3 and "[slow]" in text and "restart" in text
    assert s.slept == [7, 7] and len(s.calls) == 3                       # sleeps between samples, not after the last
    assert len(text.splitlines()) == 1                                    # one line per server for the whole run, not per sample


def test_one_low_sample_is_suspect_and_exits_zero(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    code, text = run(tmp_path, Server([10.0]), *target())
    assert code == 0 and "[suspect]" in text


def test_the_stored_baseline_is_not_lowered_by_a_slow_run(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    run(tmp_path, Server([10.0]), *target("--samples", "3"))
    assert store(tmp_path).get(URL, MODEL) == 40.0


def test_an_unreachable_server_exits_two_and_writes_nothing(tmp_path):
    code, text = run(tmp_path, Server([40.0], fail=lambda n: True), *target())
    assert code == 2 and "connection refused" in text and not (tmp_path / ".cache").exists()


def test_reset_baseline_forgets_it_so_the_run_starts_fresh(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    code, text = run(tmp_path, Server([10.0]), *target("--reset-baseline"))
    assert code == 0 and "becomes the baseline" in text and store(tmp_path).get(URL, MODEL) == pytest.approx(10.0)


def test_a_baseline_flag_is_used_and_not_stored(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    code, text = run(tmp_path, Server([10.0]), *target("--baseline", "12", "--samples", "3"))
    assert code == 0 and "[ok]" in text and store(tmp_path).get(URL, MODEL) == 40.0


def test_a_baseline_file_flag_moves_the_store(tmp_path):
    other = tmp_path / "elsewhere.json"
    run(tmp_path, Server([40.0]), *target("--baseline-file", str(other)))
    assert sw.Baselines(other).get(URL, MODEL) == pytest.approx(40.0) and not (tmp_path / ".cache").exists()


def test_json_output_has_one_object_per_target_with_the_numbers(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    code, text = run(tmp_path, Server([38.0]), *target("--json"))
    (row,) = json.loads(text)
    assert code == 0 and row["state"] == "ok" and row["endpoint"] == URL and row["model"] == MODEL
    assert row["tok_s"] == pytest.approx(38.0) and row["baseline"] == pytest.approx(40.0) and 0.9 < row["ratio"] < 1.0
    assert "message" in row


# ---- watch -----------------------------------------------------------------------------------------------------

def test_watch_samples_until_the_verdict_is_slow_then_exits_three(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    s = Server([38.0, 38.0, 10.0])
    code, text = run(tmp_path, s, *target("--watch", "60"))
    assert code == 3 and len(s.calls) == 5 and s.slept == [60, 60, 60, 60]
    assert text.count("tok/s") >= 5 and "[slow]" in text.splitlines()[-1]


def test_watch_gives_up_after_three_errors_in_a_row_with_exit_two(tmp_path):
    s = Server([40.0], fail=lambda n: n >= 2)
    code, text = run(tmp_path, s, *target("--watch", "30"))
    assert code == 2 and len(s.calls) == 4 and "3 errors in a row" in text


def test_watch_survives_one_error_between_good_samples(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    s = Server([10.0], fail=lambda n: n == 2)
    code, _ = run(tmp_path, s, *target("--watch", "30"))
    assert code == 3 and len(s.calls) == 4                                # samples 1, 3 and 4: the error did not reset them


def test_watch_ends_quietly_on_ctrl_c(tmp_path):
    s = Server([40.0])

    def stop(seconds):
        raise KeyboardInterrupt
    s.sleep = stop
    code, text = run(tmp_path, s, *target("--watch", "30"))
    assert code == 0 and "stopped" in text


# ---- which servers ---------------------------------------------------------------------------------------------

TIERS = TierConfig(tiers={"large": {"role": "r", "endpoint": "http://127.0.0.1:8001", "model": "big", "placement": "p"},
                          "small": {"role": "r", "endpoint": "http://127.0.0.1:8011/v1", "model": "coder", "placement": "p"},
                          "twin": {"role": "r", "endpoint": "http://127.0.0.1:8011/v1/", "model": "coder", "placement": "p"}},
                   stages={}, escalation="large")


def test_tiers_checks_each_distinct_server_once_and_labels_it(tmp_path):
    s = Server([40.0])
    code, text = run(tmp_path, s, "--tiers", tier_config=TIERS)
    assert code == 0 and len(s.calls) == 2                                # large, and small (twin is the same server)
    assert "large" in text and "small" in text and "big" in text and "coder" in text


def test_tiers_exit_three_when_any_server_is_slow(tmp_path):
    store(tmp_path).update("http://127.0.0.1:8001", "big", 40.0)
    store(tmp_path).update(URL, "coder", 40.0)
    s = Server([10.0])
    code, text = run(tmp_path, s, "--tiers", "--samples", "3", tier_config=TIERS)
    assert code == 3 and text.count("[slow]") == 2


def test_with_no_endpoint_and_no_tier_config_the_command_says_what_to_pass(tmp_path):
    code, text = run(tmp_path, Server([40.0]))
    assert code == 2 and "--endpoint" in text and "--model" in text


def test_an_endpoint_without_a_model_is_refused(tmp_path):
    code, text = run(tmp_path, Server([40.0]), "--endpoint", URL)
    assert code == 2 and "go together" in text


# ---- the real command line -------------------------------------------------------------------------------------

def test_the_command_runs_end_to_end_against_a_local_server(tmp_path):
    from fake_model import FakeModel, final

    def answer(request):
        msg = final("x")
        msg["completion_tokens"] = 300
        return msg
    out = io.StringIO()
    with FakeModel(answer) as server:
        code = cli.main(["coder-speed", "--endpoint", server.endpoint, "--model", "fake", "--baseline-file",
                         str(tmp_path / "b.json"), "--json"], env={"HOME": str(tmp_path)}, stdout=out)
    (row,) = json.loads(out.getvalue())
    assert code == 0 and row["model"] == "fake" and row["tok_s"] > 0
    assert sw.Baselines(tmp_path / "b.json").get(server.endpoint, "fake") == pytest.approx(row["tok_s"])


def test_the_command_exits_two_for_a_server_that_is_not_there(tmp_path):
    out = io.StringIO()
    code = cli.main(["coder-speed", "--endpoint", "http://127.0.0.1:1/v1", "--model", "x", "--baseline-file",
                     str(tmp_path / "b.json")], env={"HOME": str(tmp_path)}, stdout=out)
    assert code == 2 and "failed" in out.getvalue()


def test_with_no_flags_the_coder_tier_of_the_config_is_measured(tmp_path):
    cfg = TierConfig(tiers=TIERS.tiers, stages={2: {"run": "small"}}, escalation="large")
    s = Server([40.0])
    code, text = run(tmp_path, s, tier_config=cfg)
    assert code == 0 and s.calls == [(("http://127.0.0.1:8011/v1/chat/completions"), "coder")] and "small" in text


def test_errors_that_are_not_in_a_row_never_end_the_watch(tmp_path):
    """Errors on calls 2, 4 and 6 with good samples between them: the count of errors in a row restarts after each good
    sample, so only the operator's Ctrl-C (the fake sleep raises it on the 8th sleep) ends this watch."""
    s = Server([40.0], fail=lambda n: n in (2, 4, 6))
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 8:
            raise KeyboardInterrupt
    s.sleep = sleep
    code, text = run(tmp_path, s, *target("--watch", "30"))
    assert code == 0 and "stopped" in text and len(s.calls) == 8 and "errors in a row" not in text


# ---- another client is using the server ------------------------------------------------------------------------------

def test_a_busy_server_is_reported_busy_exits_zero_and_learns_nothing(tmp_path):
    code, text = run(tmp_path, Server([7.0]), *target(), running=lambda e: 1)
    assert code == 0 and "[busy]" in text and "1 other request" in text
    assert store(tmp_path).get(URL, MODEL) is None                        # a contended sample is not a baseline


def test_busy_samples_never_end_a_run_as_slow(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    code, text = run(tmp_path, Server([7.0]), *target("--samples", "3"), running=lambda e: 2)
    assert code == 0 and "[busy]" in text and "[slow]" not in text


def test_json_says_busy(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    code, text = run(tmp_path, Server([7.0]), *target("--json"), running=lambda e: 1)
    (row,) = json.loads(text)
    assert code == 0 and row["state"] == "busy" and row["tok_s"] == pytest.approx(7.0)


def test_a_watch_over_a_busy_server_keeps_going_and_a_later_quiet_slow_run_ends_it(tmp_path):
    store(tmp_path).update(URL, MODEL, 40.0)
    s = Server([7.0])
    load = iter([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])                           # busy for two samples (two readings each), then quiet
    code, text = run(tmp_path, s, *target("--watch", "30"), running=lambda e: next(load))
    assert code == 3 and text.count("[busy]") == 2 and "[slow]" in text.splitlines()[-1] and len(s.calls) == 5


def test_a_cut_off_response_ends_the_command_with_an_error_line_and_exit_2(tmp_path):
    import http.client
    server = Server([40.0])

    def cut(url, body, timeout):
        raise http.client.IncompleteRead(b"x", 10)
    server.post = cut
    code, text = run(tmp_path, server, "--endpoint", URL, "--model", MODEL)
    assert code == 2 and "error:" in text and "Traceback" not in text


def test_watch_survives_cut_off_responses_and_ends_after_three_in_a_row(tmp_path):
    import http.client
    server = Server([40.0])

    def cut(url, body, timeout):
        raise http.client.BadStatusLine("x")
    server.post = cut
    code, text = run(tmp_path, server, "--endpoint", URL, "--model", MODEL, "--watch", "1")
    assert code == 2 and "3 errors in a row" in text


@pytest.mark.parametrize("value", ["0", "-3", "nan"])
def test_a_baseline_that_is_not_positive_is_refused_with_a_message(tmp_path, value):
    server = Server([40.0])
    code, text = run(tmp_path, server, "--endpoint", URL, "--model", MODEL, "--baseline", value)
    assert code == 2 and "--baseline" in text and "greater than 0" in text
    assert server.calls == []                                       # refused before any request
