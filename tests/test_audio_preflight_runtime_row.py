# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The preflight base row for an audio model the harness's own audio runtime serves (issue 166).

After the audio runtime landed, a Stable Audio 3 run whose model the runtime implements goes on to stages 1, 2 and 8. The base row has to
give the answer the supervisor will give (`_select_audio_runtime`): the same library-name rule, `audio_runtime.runtime_problems` on the model's
snapshot, and `audio_runtime.environment_problems` for the interpreters and the library checkout."""
from __future__ import annotations

import json
from pathlib import Path

from orchard import preflight as pf
from orchard.audio_runtime import AudioPaths
from test_audio_chip_config import REAL_CONFIG
from test_audio_preflight import (AUDIO_FILES, BASE, EVIDENCE, MODEL, REV, audio_base, audio_payload, base_hub, full_snaps, model_config,
                                  rows, snap, world)


def served_snapshot(root: Path) -> Path:
    """A model snapshot whose model_config.json is one the audio runtime implements (the real Stable Audio 3 small music settings)."""
    s = snap(root, MODEL, REV, names=AUDIO_FILES)
    cfg = model_config(MODEL)
    cfg["model"] = {**cfg["model"], **REAL_CONFIG["model"]}
    (s / "model_config.json").write_text(json.dumps(cfg))
    return s


def audio_paths(tmp_path, *, python=True, src=True) -> AudioPaths:
    ref, chip, lib = tmp_path / "ref-python", tmp_path / "chip-python", tmp_path / "src"
    for p in (ref, chip) if python else ():
        p.write_text("#!/bin/sh\n")
        p.chmod(0o755)
    if src:
        (lib / "stable_audio_3").mkdir(parents=True)
        (lib / "stable_audio_3" / "model.py").write_text("")
    return AudioPaths(str(ref), str(chip), str(lib))


def served_row(tmp_path, **kw):
    s = served_snapshot(tmp_path / "hf")
    return audio_base(base_info=(base_hub(gated=False), None), local={MODEL: s}, audio_paths=audio_paths(tmp_path, **kw))


def test_a_model_the_audio_runtime_serves_goes_on_and_the_row_names_what_it_needs(tmp_path):
    c = served_row(tmp_path)
    assert c.status == "warn" and c.reason is None
    assert "audio-path-unbuilt" not in c.detail and "stop after stage 0" not in c.detail
    for part in ("audio runtime", "stages 1, 2 and 8", str(tmp_path / "ref-python"), str(tmp_path / "chip-python"), str(tmp_path / "src")):
        assert part in c.detail, part


def test_a_served_model_with_a_missing_interpreter_or_checkout_says_what_is_missing(tmp_path):
    c = served_row(tmp_path, python=False, src=False)
    assert c.status == "warn" and "audio-path-unbuilt" not in c.detail
    assert "is missing" in c.detail and "reference interpreter" in c.detail and "library checkout" in c.detail


def test_a_model_whose_configuration_the_runtime_does_not_implement_still_stops_after_stage_0(tmp_path):
    s = snap(tmp_path / "hf", MODEL, REV, names=AUDIO_FILES)          # the plain test config has no diffusion settings
    c = audio_base(base_info=(base_hub(gated=False), None), local={MODEL: s}, audio_paths=audio_paths(tmp_path))
    assert "audio-path-unbuilt" in c.detail and "stage 0" in c.detail
    assert "does not implement" in c.detail


def test_a_model_that_is_not_downloaded_yet_does_not_claim_the_stop_as_certain(tmp_path):
    c = audio_base(base_info=(base_hub(gated=False), None), local={}, audio_paths=audio_paths(tmp_path))
    assert "decided after its snapshot is downloaded" in c.detail and "stages 1, 2 and 8" in c.detail
    assert "The run will stop after stage 0" not in c.detail


def test_a_repository_the_planners_do_not_know_is_not_served(tmp_path):
    """The supervisor selects the runtime only for a repository audio_plan knows; the row must give the same answer."""
    s = served_snapshot(tmp_path / "hf")
    hub = pf.parse_hub(audio_payload("other-org/other-audio"))
    c = pf.check_base(model_id="other-org/other-audio", hub=hub, base_override=BASE, installed=[], search=None, audio=EVIDENCE,
                      local_snapshot=lambda m: s, base_hub_info=lambda m: (base_hub(gated=False), None), audio_paths=audio_paths(tmp_path))
    assert "audio-path-unbuilt" in c.detail


def test_run_preflight_passes_the_runtime_locations_to_the_base_row(tmp_path):
    snaps = full_snaps(tmp_path)
    snaps[MODEL] = served_snapshot(tmp_path / "other")
    import dataclasses
    from test_preflight import cfg
    c = dataclasses.replace(cfg(tmp_path), operator_home=tmp_path / "empty-home")     # none of the three locations exists under it
    by = {x.name: x for x in pf.run_preflight(c, MODEL, accept_credentials=True, signals=world(local=snaps))}
    assert "audio runtime" in by["base"].detail and "audio-path-unbuilt" not in by["base"].detail
    assert str(tmp_path / "empty-home" / "orchard-venvs") in by["base"].detail and "Missing now" in by["base"].detail
