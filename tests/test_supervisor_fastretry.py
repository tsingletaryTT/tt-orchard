# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A multi-board test that dies at startup is run once more on freshly reset chips.

Evidence: the 4-chip test failed with `Fabric Router Sync: Timeout` after 57 s (bytkim/Qwen3.8-27B-pi, 2026-10-18), twice in
the RAG-Gate-27B run, and passed on the same command after the chips were reset together. The cause was not isolated, so the
supervisor does not guess at it. A failure that early is a link or startup failure, not a wrong answer, and a second try costs
about a minute. A late failure, a timeout, or a smaller test is never retried: those cost long or say something about the model."""
import json

import pytest

from orchard import supervisor
from orchard.supervisor import EXIT_READY
from run_fakes import FAKE_SWAP_TEST, bringup, swap_entry, swap_prepare
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")

FLAKY = ("import pathlib, sys\n"
         "m = pathlib.Path('stages/4/configs/{n}/first-try-done')\n"
         "if not m.exists():\n    m.write_text('x')\n    print('Fabric Router Sync: Timeout')\n    sys.exit(4)\n")
NEVER = "import sys\nprint('server never healthy')\nsys.exit(4)\n"


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def with_config(n, body, ok=True):
    files = swap_prepare()
    files[f"configs/{n}/serve_and_compare{'_container' if n == 4 else ''}.py"] = body
    return {(4, "prepare"): files,
            (4, "finish"): {"result.json": {"configs": [swap_entry(1), swap_entry(2), swap_entry(4, ok=ok)]}}}


def starts(rig, config):
    return [e["data"] for e in rig.entries() if e["stage"] == 4 and e["data"].get("decision") == "hardware test started"
            and e["data"].get("config") == config]


def retries(rig, config):
    return [e["data"] for e in rig.entries() if e["stage"] == 4 and e["data"].get("config") == config
            and e["data"].get("decision") == "test failed at startup; resetting the test chips and trying once more"]


def record(rig, config):
    return json.loads((rig.run_dir / "stages" / "4" / "tests" / str(config) / "test-result.json").read_text())


def test_a_4_chip_test_that_fails_at_startup_is_run_again_after_every_leased_chip_is_reset(rig):
    rig.script = lambda r: bringup(r, overrides=with_config(4, FLAKY.format(n=4) + FAKE_SWAP_TEST))
    assert rig.run() == EXIT_READY
    assert len(starts(rig, 4)) == 2 and len(retries(rig, 4)) == 1
    assert record(rig, 4)["returncode"] == 0                         # the record is the second try's
    es = [e for e in rig.entries() if e["stage"] == 4]
    first, second = [i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started"
                     and e["data"].get("config") == 4]
    resets_between = [e for e in es[first:second] if e["event"] == "decision" and e["data"].get("decision") == "test chips reset"
                      and e["data"].get("config") == 4]
    # the park lease (the coder's board) and the further board's lease are both reset between the two tries
    assert len(resets_between) == 2
    assert len({e["data"]["lease_id"] for e in resets_between}) == 2


def test_the_first_tries_output_is_kept_beside_the_second_tries(rig):
    """The second spawn used to open the same output.txt for writing, so the text that said why the first try failed
    was gone."""
    rig.script = lambda r: bringup(r, overrides=with_config(4, FLAKY.format(n=4) + FAKE_SWAP_TEST))
    assert rig.run() == EXIT_READY
    out = rig.run_dir / "stages" / "4" / "tests" / "4"
    assert "Fabric Router Sync: Timeout" in (out / "output-first-try.txt").read_text()
    assert "Fabric Router Sync: Timeout" not in (out / "output.txt").read_text()
    kept = retries(rig, 4)[0]["first_output"]
    assert kept["path"].endswith("tests/4/output-first-try.txt") and len(kept["sha256"]) == 64


def test_the_second_try_is_the_last(rig):
    rig.args.required_chips = (2,)
    rig.script = lambda r: bringup(r, overrides=with_config(4, NEVER, ok=False))
    assert rig.run() == EXIT_READY
    assert len(starts(rig, 4)) == 2 and len(retries(rig, 4)) == 1
    assert record(rig, 4)["returncode"] == 4


def test_a_failure_after_a_long_run_is_not_retried(rig, monkeypatch):
    real = supervisor.spawn_checked

    def slow(command, run_dir, env, timeout, out):
        if command.endswith("configs/4/serve_and_compare_container.py"):
            rig.clock.sleep(supervisor.FAST_FAIL_S + 1)
        return real(command, run_dir, env, timeout, out)
    monkeypatch.setattr(supervisor, "spawn_checked", slow)
    rig.args.required_chips = (2,)
    rig.script = lambda r: bringup(r, overrides=with_config(4, NEVER, ok=False))
    assert rig.run() == EXIT_READY
    assert len(starts(rig, 4)) == 1 and not retries(rig, 4)


def test_a_timeout_is_not_retried(rig, monkeypatch):
    real = supervisor.spawn_checked

    def timed_out(command, run_dir, env, timeout, out):
        if command.endswith("configs/4/serve_and_compare_container.py"):
            return None, True
        return real(command, run_dir, env, timeout, out)
    monkeypatch.setattr(supervisor, "spawn_checked", timed_out)
    rig.args.required_chips = (2,)
    rig.script = lambda r: bringup(r, overrides=with_config(4, NEVER, ok=False))
    assert rig.run() == EXIT_READY
    assert len(starts(rig, 4)) == 1 and not retries(rig, 4)


def test_a_one_chip_test_is_not_retried(rig):
    files = swap_prepare()
    files["configs/1/serve_and_compare.py"] = "import sys\nsys.exit(4)\n"
    rig.args.required_chips = (2, 4)
    finish = {"result.json": {"configs": [swap_entry(1, ok=False), swap_entry(2), swap_entry(4)]}}
    rig.script = lambda r: bringup(r, overrides={(4, "prepare"): files, (4, "finish"): finish})
    assert rig.run() == EXIT_READY
    assert len(starts(rig, 1)) == 1 and not retries(rig, 1)
