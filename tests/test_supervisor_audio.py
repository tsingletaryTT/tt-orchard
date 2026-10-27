# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The supervisor and the audio facts: stage 0's passing end records form, serving and modality, and a run whose model is audio
stops before stage 1 until the audio stages exist (plan docs/superpowers/plans/2026-10-28-audio-bringup.md, tasks A1 and A3)."""
import json

import pytest

from orchard import supervisor
from orchard.supervisor import EXIT_READY
from run_fakes import DELTA, FILES, bringup
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop("the run waited for the operator"))
    yield r
    r.close()

AUDIO_FACTS = {"form": "full", "modality": "audio"}
AUDIO_FILES = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "facts": AUDIO_FACTS}}}
TEXT_FILES = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "facts": {"form": "full", "modality": "text"}}}}


def install_bundle(rig, repo, schema="6", name="org/some-bundle"):
    """A manifest in the run's tt-model models root, the way tt-model leaves it."""
    d = rig.home / ".cache" / "tt-model" / "models" / name
    d.mkdir(parents=True)
    (d / "tt_kernel_manifest.json").write_text(json.dumps({"schema_version": schema, "device_count": 2,
                                                          "weights": {"repo_id": repo}}))


def test_a_text_run_records_no_facts_so_the_default_applies(rig):
    assert rig.run() == EXIT_READY
    assert "facts" not in rig.ends(0)[0]


def test_stage_0_records_the_facts_with_serving_from_the_installed_bundles(rig):
    install_bundle(rig, "Qwen/Qwen3.8-27B")
    rig.script = lambda r: bringup(r, overrides=TEXT_FILES)
    assert rig.run() == EXIT_READY
    assert rig.ends(0)[0]["facts"] == {"form": "full", "serving": "thin-bundle", "modality": "text"}


def test_a_base_nothing_serves_records_serving_none(rig):
    rig.script = lambda r: bringup(r, overrides=TEXT_FILES)
    assert rig.run() == EXIT_READY
    assert rig.ends(0)[0]["facts"]["serving"] == "none"


def test_a_container_package_records_serving_container(rig):
    install_bundle(rig, "Qwen/Qwen3.8-27B", schema="5.1")
    rig.script = lambda r: bringup(r, overrides=TEXT_FILES)
    assert rig.run() == EXIT_READY
    assert rig.ends(0)[0]["facts"]["serving"] == "container"


def test_the_form_and_the_modality_are_the_templates_values(rig):
    files = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "class": "weights-only",
                                                              "facts": {"form": "adapter", "modality": "audio"}}}}
    rig.script = lambda r: bringup(r, overrides=files)
    try:
        rig.run()
    except Stop:
        pass
    assert rig.ends(0)[0]["facts"] == {"form": "adapter", "serving": "none", "modality": "audio"}


def test_a_package_the_operators_registry_lists_counts_as_a_container(rig):
    manifest = rig.home / "pulled" / "m.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema_version": "5.1", "device_count": 4, "weights": {"repo_id": "Qwen/Qwen3.8-27B"}}))
    reg = rig.home / ".cache" / "tt-model" / "installed.json"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps({"org/pkg": {"container": True, "manifest": str(manifest)}}))
    rig.script = lambda r: bringup(r, overrides=TEXT_FILES)
    rig.run()
    assert rig.ends(0)[0]["facts"]["serving"] == "container"
