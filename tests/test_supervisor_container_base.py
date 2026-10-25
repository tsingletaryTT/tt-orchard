# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 on a container (v5.1) package (issue 60): the supervisor drafts the container config, copies the container script, and stops
what the test leaves behind. A v6 bundle that fits is used as before."""
import hashlib
import json
from pathlib import Path

import pytest

from orchard.supervisor import EXIT_READY, parse
from run_fakes import FakeContainers, argv
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")

NEAREST = "Qwen/Qwen3.8-27B"


def container_rig(tmp_path, *, bundle=False, listings=None, chips=2):
    r = Rig(tmp_path, chips=chips)
    op, root = tmp_path / "op", tmp_path / "tt-models"
    a = argv(tmp_path, r.tiers, r.chip_server.endpoint, chips=chips)
    a[a.index("--input") + 1] = f"base={tmp_path}/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/abc"
    a += ["--input", f"model={tmp_path}/hf/hub/models--Altworld--Hemmingway-1/snapshots/def", "--hf-home", f"{tmp_path}/hf",
          "--cache-root", f"{tmp_path}/cache", "--tt-model-root", str(root), "--operator-home", str(op)]
    r.args = parse(a)
    for key in ("base", "model"):
        Path(next(i for i in r.args.input if i.startswith(key + "=")).split("=", 1)[1]).mkdir(parents=True)
    mf = op / ".cache" / "tt-model" / "pulled" / "someone__qwen3.8-27b-p300" / "tt_kernel_manifest.json"
    mf.parent.mkdir(parents=True)
    mf.write_text(json.dumps({"schema_version": "5.1", "device_count": 2, "weights": {"repo_id": NEAREST}}))
    (op / ".cache" / "tt-model" / "installed.json").write_text(json.dumps(
        {"someone/qwen3.8-27b-p300": {"container": True, "image": "tt-model/qwen:abc", "profile": "default",
                                      "profiles": ["default"], "manifest": str(mf)}}))
    if bundle:
        b = root / "episod" / "qwen3.8-27b-dflash2-p300"
        b.mkdir(parents=True)
        (b / "tt_kernel_manifest.json").write_text(json.dumps(
            {"schema_version": "6", "device_count": 2, "weights": {"repo_id": NEAREST}}))
        (b / "run.sh").write_text("#!/bin/bash\n")
    r.containers = FakeContainers(listings)
    return r


def drafted(r):
    return next(e for e in r.entries() if e["data"].get("decision") == "swap config drafted")


def test_stage_2_is_drafted_from_the_container_package_when_no_bundle_fits(tmp_path):
    r = container_rig(tmp_path)
    try:
        assert r.run() == EXIT_READY
        e = drafted(r)
        assert e["stage"] == 2 and e["data"]["package"] == "someone/qwen3.8-27b-p300" and e["data"]["bundle_dir"] is None
        assert e["data"]["copied"] == ["prepare_swap.py", "serve_and_compare.py", "prompt_suite.json", "serve_and_compare_container.py"]
        sd = r.run_dir / "stages" / "2"
        cfg = json.loads((sd / "swap_config.json").read_text())
        assert cfg["package"] == "someone/qwen3.8-27b-p300" and cfg["chips"] == 2 and "bundle_dir" not in cfg
        assert (sd / "serve_and_compare_container.py").is_file()
    finally:
        r.close()


def test_a_bundle_that_fits_is_used_and_no_container_script_is_copied(tmp_path):
    r = container_rig(tmp_path, bundle=True)
    try:
        assert r.run() == EXIT_READY
        d = drafted(r)["data"]
        assert d["bundle_dir"].endswith("qwen3.8-27b-dflash2-p300") and "package" not in d
        assert d["copied"] == ["prepare_swap.py", "serve_and_compare.py", "prompt_suite.json"]
        assert not (r.run_dir / "stages" / "2" / "serve_and_compare_container.py").exists()
    finally:
        r.close()


def stage2_sweeps(r):
    return [e for e in r.entries() if e["stage"] == 2 and e["event"] == "notice"
            and e["data"].get("what") == "a hardware test left containers running; stopping them"]


def test_a_container_left_by_stage_2s_test_is_stopped_by_label(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"], []])
    try:
        assert r.run() == EXIT_READY
        [note] = stage2_sweeps(r)
        assert note["data"]["containers"] == ["c1"]
        assert note["data"]["label"] == "orchard.test=" + hashlib.sha256(str(r.run_dir).encode()).hexdigest()[:12]
        assert ("remove", "c1") in r.containers.calls
    finally:
        r.close()


def test_a_container_that_survives_the_stop_blocks_stage_2_before_the_lease_is_given_back(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"]])
    r.args.unattended = True                # a block ends the run instead of waiting for an operator
    try:
        assert r.run() != EXIT_READY
        blocked = [e for e in r.entries() if e["event"] == "notice" and e["data"].get("blocked") and e["stage"] == 2]
        assert blocked and "still running" in blocked[0]["data"]["reason"]
        assert not [e for e in r.entries() if e["data"].get("decision") == "test lease released" and e["stage"] == 2]
    finally:
        r.close()


def test_a_bundle_test_is_not_swept_at_stage_2(tmp_path):
    r = container_rig(tmp_path, bundle=True, listings=[["c1"], []])
    try:
        assert r.run() == EXIT_READY
        assert stage2_sweeps(r) == []
    finally:
        r.close()


@pytest.mark.parametrize("text,expected", [('{"package": "a/b", "profile": "p"}', True), ('{"bundle_dir": "/b"}', False),
                                           ('["package"]', False), ('"package"', False), ("not json", False), ("", False)])
def test_only_a_config_object_that_names_a_package_is_a_container_config(tmp_path, text, expected):
    from orchard.supervisor import Supervisor
    (tmp_path / "swap_config.json").write_text(text)
    assert Supervisor._container_config(tmp_path) is expected


def test_a_stage_with_no_swap_config_is_not_a_container_config(tmp_path):
    from orchard.supervisor import Supervisor
    assert Supervisor._container_config(tmp_path) is False
