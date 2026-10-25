# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A container that fetches the base model from the Hub did not serve the model directory (found 2026-10-25, issue 60 validation).

`raahemnabeel/qwen3-coder-30b-a3b` calls `snapshot_download(HF_MODEL_ID)` with a constant in its own generator, so it ignores MODEL_WEIGHTS_DIR and
HF_MODEL and loads the base model. The test container's /hf was empty on purpose ("a runtime that ignores MODEL_WEIGHTS_DIR fails with an error"), but with a
network the fetch simply worked: 57 GB downloaded during a 24-minute boot, stage 2 passed (32 of 32 tokens, top-1 1.0: the fine-tune's CPU reference and the
base agree on an easy prompt), and the pass measured the base model. The template now runs the container with the Hub switched off, starts from an empty /hf,
and fails a test whose /hf or log shows a fetch. The gates refuse a report that records one."""
import json

import pytest

from orchard import stages
from test_container_template import crig, docker_runs, edit, load_template, pairs, paths  # noqa: F401

sac = load_template()


def test_the_container_runs_with_the_hub_switched_off(paths):
    out = edit(paths)
    from test_container_template import IMAGE
    envs = pairs(out[:out.index(IMAGE)], "--env")
    assert envs.count("HF_HUB_OFFLINE=1") == 1 and envs.count("TRANSFORMERS_OFFLINE=1") == 1


def test_an_offline_setting_the_package_prints_is_replaced_not_duplicated(paths):
    from test_container_template import printed_argv, IMAGE
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3], extra_env={"HF_HUB_OFFLINE": "0"})
    out = edit(paths, printed=printed)
    k = out.index(IMAGE)
    envs = pairs(out[:k], "--env")
    assert [e for e in envs if e.startswith("HF_HUB_OFFLINE=")] == ["HF_HUB_OFFLINE=1"]


def hub(tmp_path, repo="Qwen--Qwen3-Coder-30B-A3B-Instruct", size=10):
    blob = tmp_path / "hf-isolated" / "hub" / f"models--{repo}" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_text("x" * size)
    return tmp_path / "hf-isolated"


def test_a_blob_in_the_isolated_hub_cache_is_a_fetch(tmp_path):
    hf = hub(tmp_path)
    assert sac.hub_fetches(hf, tmp_path / "none.log") == ["Qwen/Qwen3-Coder-30B-A3B-Instruct"]


def test_an_empty_isolated_directory_and_empty_blobs_are_no_fetch(tmp_path):
    (tmp_path / "hf-isolated").mkdir()
    assert sac.hub_fetches(tmp_path / "hf-isolated", tmp_path / "none.log") == []
    hub(tmp_path, size=0)
    assert sac.hub_fetches(tmp_path / "hf-isolated", tmp_path / "none.log") == []


def test_a_fetching_line_in_the_server_log_is_a_fetch(tmp_path):
    (tmp_path / "hf-isolated").mkdir()
    log = tmp_path / "server.log"
    log.write_text("starting\nFetching 28 files:   4%|▎         | 1/28 [00:00<00:03,  7.07it/s]\n")
    assert sac.hub_fetches(tmp_path / "hf-isolated", log)


def test_ordinary_log_lines_are_no_fetch(tmp_path):
    (tmp_path / "hf-isolated").mkdir()
    log = tmp_path / "server.log"
    log.write_text("Fetching state for layer 3\nloaded 28 files from the model directory\n")
    assert sac.hub_fetches(tmp_path / "hf-isolated", log) == []


def test_a_clean_run_records_no_fetch(crig):
    r = crig["start"]()
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())
    assert rep["hub_fetch"] == []


def test_a_container_that_fetches_from_the_hub_fails_the_test_and_says_so(crig):
    leftover = str(crig["cdir"] / "hf-isolated" / "hub" / "models--Qwen--Qwen3.8-27B" / "blobs" / "abc")
    r = crig["start"](server_extra={"write_files": {leftover: "weights"}})
    assert r.returncode == 6, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "Qwen/Qwen3.8-27B" in out and "MODEL_WEIGHTS_DIR" in out
    rep = json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())
    assert rep["hub_fetch"] == ["Qwen/Qwen3.8-27B"]


def test_the_isolated_directory_starts_empty_each_time(crig):
    junk = crig["cdir"] / "hf-isolated" / "hub" / "models--Old--Leftover" / "blobs" / "x"
    junk.parent.mkdir(parents=True)
    junk.write_text("left by an earlier attempt")
    r = crig["start"]()
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())
    assert rep["hub_fetch"] == [] and not junk.exists()


def test_the_gate_refuses_a_report_that_records_a_fetch():
    report = {"hub_fetch": ["Qwen/Qwen3-Coder-30B-A3B-Instruct"]}
    reasons = stages._hub_fetch_reasons(report, "the 4-chip configuration: ")
    assert reasons and "Qwen/Qwen3-Coder-30B-A3B-Instruct" in reasons[0] and reasons[0].startswith("the 4-chip configuration: ")
    assert stages._hub_fetch_reasons({"hub_fetch": []}, "") == [] and stages._hub_fetch_reasons({}, "") == []
    assert "must be a list" in stages._hub_fetch_reasons({"hub_fetch": "yes"}, "")[0]   # a malformed record does not count as clean


def test_the_gates_apply_it_through_the_suite_reasons_helper(tmp_path):
    run = tmp_path
    out = stages._suite_reasons({"hub_fetch": ["a/b"]}, run, "x: ")
    assert any("a/b" in r for r in out)
