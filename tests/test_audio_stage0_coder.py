# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 143: a fresh audio run boots the local coder for stage 0, which needs no model for an audio model. The audio stages 1, 2 and 8 already
skip the coder (`needs_coder` False, test_supervisor_audio_runtime.py). The test below states the wanted behaviour for stage 0 and is expected to
fail until stage 0 has a harness body and the supervisor knows the model is audio before stage 0 runs (the issue's "what it would take"). It is
strict: when someone fixes the issue it passes, pytest reports that as a failure, and the marker is to be removed."""
import pytest

from test_supervisor import Stop
from test_supervisor_audio_runtime import arig, decisions  # noqa: F401


def test_the_audio_run_starts_the_coder_once_for_stage_0_as_things_stand(arig):
    """The present behaviour, pinned so a change to it is noticed: one coder start, for stage 0, before the pause for the missing environment."""
    with pytest.raises(Stop):
        arig.run()
    assert decisions(arig, "pause") and arig.m.coder_starts == 1


@pytest.mark.xfail(strict=True, reason="issue 143: stage 0 is a template stage run by the coder agent, so the coder boots for an audio model that never uses it")
def test_stage_0_of_an_audio_run_boots_no_coder(arig):
    with pytest.raises(Stop):
        arig.run()
    assert arig.m.coder_starts == 0
