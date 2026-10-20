# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard bringup` retries a run that ended blocked for a transient reason (autonomy, 2026-10-18).

Before this a person ran the same command again after every block. Retrying costs a resume from the ledger and a
fresh coder context. Only blocks that a retry can plausibly clear are retried, only a few times, and only after a
pause that lets the released chips settle (`gozer status` shows the board BUSY-UNTRACKED for about a minute after a
release). A block that names a missing model, a full disk or a rule is never retried."""
import json

import pytest

from orchard import blocked, cli
from test_cli import Pipe, Recorder, conf, signals  # noqa: F401  (conf is a fixture)


class Blocks(Recorder):
    """A supervisor that ends blocked with each code in turn, then with `then`."""

    def __init__(self, codes, then=0):
        super().__init__(0)
        self.codes, self.then, self.calls = list(codes), then, 0

    def supervisor(self, argv):
        from pathlib import Path
        self.argv = argv
        self.calls += 1
        run_dir = Path(argv[argv.index("--run-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.codes:
            (run_dir / "blocked.json").write_text(json.dumps({"code": self.codes.pop(0), "reason": "x"}))
            return 5
        (run_dir / "blocked.json").unlink(missing_ok=True)
        return self.then


def go(conf, rec, *extra):
    out, naps = Pipe(), []
    code = cli.main(["--config", str(conf), "bringup", "Cloudflare/clef", *extra], env={}, stdout=out,
                    signals=signals(), supervisor_main=rec.supervisor, fetcher=rec.fetcher, sleep=naps.append)
    return code, out.getvalue(), naps


def test_a_transient_block_is_retried_after_a_pause_and_the_run_can_finish(conf):
    rec = Blocks(["coder-unusable"])
    code, out, naps = go(conf, rec)
    assert code == 0 and rec.calls == 2
    assert naps == [cli.AUTO_RETRY_WAIT_S]
    assert "retrying" in out.lower() and "coder-unusable" in out and "1 of 2" in out


@pytest.mark.parametrize("reason", ["needs-new-model-code", "disk-full", "credentials-needed", "stage-failed",
                                    "retry-budget-spent", "unclassified", "blocked"])
def test_a_block_that_a_retry_cannot_clear_is_never_retried(conf, reason):
    rec = Blocks([reason])
    code, out, naps = go(conf, rec)
    assert code == 5 and rec.calls == 1 and naps == []
    assert "retrying" not in out.lower()


def test_the_retries_are_bounded_and_the_final_exit_is_still_blocked(conf):
    rec = Blocks(["hardware-unhealthy"] * 5)
    code, out, naps = go(conf, rec)
    assert code == 5 and rec.calls == 3 and len(naps) == 2          # the first run and two retries
    assert "after 2 automatic retries" in out


def test_zero_turns_the_retries_off(conf):
    rec = Blocks(["coder-unusable"])
    code, out, naps = go(conf, rec, "--auto-retries", "0")
    assert code == 5 and rec.calls == 1 and naps == []


def test_the_reason_is_read_again_after_each_run(conf):
    rec = Blocks(["agent-stuck", "needs-new-model-code"])
    code, out, naps = go(conf, rec)
    assert code == 5 and rec.calls == 2 and len(naps) == 1


def test_every_retryable_reason_is_a_real_block_reason():
    assert set(blocked.AUTO_RETRY) <= set(blocked.REASONS)
    assert "needs-new-model-code" not in blocked.AUTO_RETRY and "disk-full" not in blocked.AUTO_RETRY


def test_a_run_that_ends_ready_or_aborted_is_not_retried(conf):
    for code_out in (0, 4):
        rec = Blocks([], then=code_out)
        code, out, naps = go(conf, rec)
        assert code == code_out and rec.calls == 1 and naps == []
