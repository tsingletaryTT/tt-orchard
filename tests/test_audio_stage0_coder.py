# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 143: stage 0 of an audio run needs no model. Since the command records the run_start input `audio_model=1`, the supervisor runs stage 0 itself and
boots no coder (tests/test_audio_stage0_harness.py, which holds the wanted behaviour). The test below pins what stays the same: a run started without that input
(every run from before the change) keeps the agent stage 0, so the coder boots once for it, before the pause for the missing environment."""
import pytest

from test_supervisor import Stop
from test_supervisor_audio_runtime import arig, decisions  # noqa: F401


def test_an_audio_run_started_without_the_audio_input_starts_the_coder_once_for_stage_0(arig):
    """A run with no `audio_model` input: one coder start, for the agent stage 0, before the pause for the missing environment."""
    with pytest.raises(Stop):
        arig.run()
    assert decisions(arig, "pause") and arig.m.coder_starts == 1
