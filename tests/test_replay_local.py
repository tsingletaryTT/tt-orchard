# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Opt-in replay of every qwen-code transcript on this machine (spec section 12).

Normal test runs read nothing outside the repo, so this test runs only with ORCHARD_REPLAY=1.
It reads ~/.qwen/projects (or ORCHARD_QWEN_PROJECTS). A skipped replay is not evidence that the
detectors fire on the loop or stay quiet elsewhere.
"""
import os
from pathlib import Path

import pytest

from orchard.transcripts import iso, qwen_events
from orchard.watchdog import replay, transcript_detectors
from test_transcripts import LOOP_FINDINGS

QWEN = Path(os.environ.get("ORCHARD_QWEN_PROJECTS", str(Path.home() / ".qwen" / "projects")))
LOOP = "197354ac-2d30-4b0f-9d2a-a1518f7b33de.jsonl"

pytestmark = [
    pytest.mark.skipif(os.environ.get("ORCHARD_REPLAY") != "1",
                       reason="set ORCHARD_REPLAY=1 to replay the local qwen transcripts; "
                              "a skipped replay is not evidence"),
]


def test_the_detectors_fire_on_the_recorded_loop_and_nowhere_else():
    # The directory check is here and not in a module-level skipif: collection must not touch
    # ~/.qwen in a normal run. The ORCHARD_REPLAY skip above runs first.
    if not QWEN.is_dir():
        pytest.skip(f"no qwen transcripts at {QWEN}; this replay was not run, and a skipped "
                    "replay is not evidence")
    paths = sorted(QWEN.glob("*/chats/*.jsonl"))
    assert any(p.name == LOOP for p in paths), f"{LOOP} is no longer under {QWEN}"
    hits = {}
    for p in paths:
        with p.open(encoding="utf-8", errors="replace") as f:
            found = replay(qwen_events(f), transcript_detectors())
        if found:
            hits[p.name] = [(x.detector, iso(x.ts)) for x in found]
    assert hits == {LOOP: LOOP_FINDINGS}
