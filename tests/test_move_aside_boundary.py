# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The move-aside of a stage directory writes its boundary entry before the rename (review of PR 403).

Two kills in a row used to leave no boundary: one after the rename, one after the fresh directory was made and before stage_start. An agent could
then copy the earlier attempt's marker bytes into the fresh directory and the earlier attempt's record vouched for them."""
import os

import pytest

from orchard.ledger import Ledger
from orchard.stages import AUDIO_STAGE_1, TEXT_STAGE_1, WEIGHTS_ONLY_STAGE_4, STAGES, evidence_record, open_stage_dir

CASES = [
    (WEIGHTS_ONLY_STAGE_4, "tests/plan.json", 4, "hardware test list", None),
    (TEXT_STAGE_1, "text_reference.json", 1, "text stage record", None),
    (AUDIO_STAGE_1, "audio_reference.json", 1, "audio stage record", None),
    (STAGES[2], "test-result.json", 2, "hardware test", {"returncode": 0, "timed_out": False}),
]
IDS = [c[1] for c in CASES]


class Killed(BaseException):
    pass


class KillAfterRename(Ledger):
    """Raises when asked to append the entry that follows the rename: the process died between the rename and that entry."""
    armed = False

    def append(self, event, stage, **data):
        if self.armed and data.get("decision") == "moved the partial stage directory aside":
            raise Killed()
        return super().append(event, stage, **data)


@pytest.mark.parametrize("spec,rel,n,what,extra", CASES, ids=IDS)
def test_two_kills_in_a_row_leave_a_boundary_so_an_old_record_cannot_vouch(tmp_path, spec, rel, n, what, extra):
    run = tmp_path / "run"
    with KillAfterRename(run / "ledger.jsonl") as ledger:
        d, _ = open_stage_dir(run, spec, resuming=False, ledger=ledger)
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("{}")
        ledger.append("evidence", n, what=what, **(extra or {}), **evidence_record(run, d / rel))
        old_bytes = (d / rel).read_bytes()
        ledger.armed = True
        with pytest.raises(Killed):                                          # kill 1: after the rename
            open_stage_dir(run, spec, resuming=False, ledger=ledger)
        ledger.armed = False
        assert not d.exists()
        # The next process: no directory, so a fresh one is made and the process is killed again before stage_start.
        fresh, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
        assert not resumed
        (fresh / rel).parent.mkdir(parents=True, exist_ok=True)
        (fresh / rel).write_bytes(old_bytes)                                  # an agent copies the old marker's bytes in
        # kill 2 happened before stage_start; a third process resumes
        _, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
        assert not resumed
