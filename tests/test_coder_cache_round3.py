# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 424, Copilot round 3 (issue 417). Four findings were real and one was already fixed:

1. A snapshot with PyTorch `.bin` shards and no index passed `shard_problem`, so the disk refusal and the `--no-fetch` block were skipped.
2. The coder's download was treated as already counted by the disk row whenever its model was the target or base and the cache was the
   run's, even when the disk row budgeted no download of it (the target is local, or the base needs none).
3. `blocked.retryable` compared the two refusal marks case-sensitively against the raw reason.
4. The refusal text in the exception was cut from stderr alone, so a warning on stderr hid the refusal that arrived on stdout.
5. (already fixed in round 1) `--prune-caches` with only the coder cache short: tests/test_coder_cache_review.py section 9."""
import json
from pathlib import Path

import pytest

from fakes import LEASE
from orchard import blocked
from orchard import coder_cache as cc
from orchard import preflight as pf
from orchard.server import ServerChipsHeld
from test_coder_cache import CODER_MODEL, coder_cfg, signals
from test_coder_cache_review import preflight, snap, target_hub
from test_coder_chips_held import REFUSAL, control
from test_preflight import CLEF_PAYLOAD

# ---- 1. `.bin` shard groups ------------------------------------------------------------------------------------------------------

BIN_INDEX = json.dumps({"weight_map": {"a": "pytorch_model-00001-of-00002.bin", "b": "pytorch_model-00002-of-00002.bin"}})
ST_INDEX = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
BIN = {"pytorch_model-00001-of-00002.bin": "x", "pytorch_model-00002-of-00002.bin": "x"}
ST = {"model-00001-of-00002.safetensors": "x", "model-00002-of-00002.safetensors": "x"}


def test_one_bin_shard_and_no_index_is_a_partial_download(tmp_path):
    d = snap(tmp_path, {"config.json": "{}", "pytorch_model-00001-of-00002.bin": "x"})
    assert cc.shard_problem(d) and "pytorch_model.bin.index.json" in cc.shard_problem(d)


def test_all_bin_shards_without_an_index_are_not_complete(tmp_path):
    assert cc.shard_problem(snap(tmp_path, {"config.json": "{}", **BIN}))


def test_a_bin_index_with_a_shard_missing_is_not_complete(tmp_path):
    d = snap(tmp_path, {"config.json": "{}", "pytorch_model-00001-of-00002.bin": "x", "pytorch_model.bin.index.json": BIN_INDEX})
    assert cc.shard_problem(d) and "1 of 2" in cc.shard_problem(d)


def test_every_bin_shard_and_the_bin_index_are_complete(tmp_path):
    assert cc.shard_problem(snap(tmp_path, {"config.json": "{}", **BIN, "pytorch_model.bin.index.json": BIN_INDEX})) is None


def test_a_safetensors_index_does_not_stand_in_for_a_bin_index(tmp_path):
    assert cc.shard_problem(snap(tmp_path, {"config.json": "{}", **BIN, "model.safetensors.index.json": ST_INDEX}))


def test_a_complete_format_passes_when_the_other_format_is_partial(tmp_path):
    # a repository that carries both formats loads from either one
    both = {"config.json": "{}", **ST, "model.safetensors.index.json": ST_INDEX, "pytorch_model-00001-of-00002.bin": "x"}
    assert cc.shard_problem(snap(tmp_path, both)) is None


def test_a_complete_bin_format_passes_when_the_safetensors_are_partial(tmp_path):
    both = {"config.json": "{}", **BIN, "pytorch_model.bin.index.json": BIN_INDEX, "model-00001-of-00002.safetensors": "x"}
    assert cc.shard_problem(snap(tmp_path, both)) is None


def test_two_partial_formats_are_still_a_problem(tmp_path):
    both = {"config.json": "{}", "model-00001-of-00002.safetensors": "x", "pytorch_model-00001-of-00002.bin": "x"}
    assert cc.shard_problem(snap(tmp_path, both))


def test_a_single_bin_file_needs_no_index(tmp_path):
    assert cc.shard_problem(snap(tmp_path, {"config.json": "{}", "pytorch_model.bin": "x"})) is None


def _bin_snapshot(root: Path):
    d = root / "hub" / "models--Qwen--Qwen3-Coder-Next" / "snapshots" / "abc"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "pytorch_model-00001-of-00002.bin").write_text("x")


def test_the_default_signal_refuses_a_partial_bin_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "scratch"))
    _bin_snapshot(tmp_path / "scratch")
    assert pf.default_signals(coder_cfg(tmp_path)).coder_snapshot(tmp_path / "scratch", CODER_MODEL) is False


def _bin_preflight(tmp_path, monkeypatch, *, free, no_fetch):
    """The real `coder_snapshot` signal over a scratch cache that holds one `.bin` shard and no index."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "scratch"))
    _bin_snapshot(tmp_path / "scratch")
    real = pf.default_signals(coder_cfg(tmp_path)).coder_snapshot
    sig = signals(hub_info=lambda m: (target_hub(149.0 if m == CODER_MODEL else 80.0), None), same_device=lambda a, b: True,
                  free_gb=lambda p: free, local_snapshot=lambda m: None, coder_snapshot=real, coder_cache=lambda: tmp_path / "scratch")
    c = coder_cfg(tmp_path)
    c.coder.port = 8000
    return {r.name: r for r in pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=sig, no_fetch=no_fetch)}


def test_a_partial_bin_cache_gets_the_disk_refusal(tmp_path, monkeypatch):
    row = _bin_preflight(tmp_path, monkeypatch, free=100.0, no_fetch=False)["coder-cache"]
    assert (row.status, row.reason) == ("block", "disk-full")


def test_a_partial_bin_cache_gets_the_no_fetch_block(tmp_path, monkeypatch):
    row = _bin_preflight(tmp_path, monkeypatch, free=900.0, no_fetch=True)["coder-cache"]
    assert (row.status, row.reason) == ("block", "coder-unusable")


# ---- 2. the coder's download is counted by the disk row only when the disk row budgets it ---------------------------------------------


def counted(model=CODER_MODEL, *, model_id=CODER_MODEL, base_id=None, target_local=False, base_gb=0.0, same=True):
    return pf.coder_download_counted(model, model_id=model_id, base_id=base_id, target_local=target_local, base_gb=base_gb,
                                     cache=Path("/hf"), hf_home=Path("/hf") if same else Path("/other"))


def test_a_target_download_the_disk_row_budgets_counts_for_the_coder():
    assert counted() is True


def test_a_local_target_has_no_download_to_count():
    assert counted(target_local=True) is False


def test_a_base_download_the_disk_row_budgets_counts_for_the_coder():
    assert counted(model_id="Org/target", base_id=CODER_MODEL, base_gb=149.0) is True


def test_a_base_the_disk_row_does_not_download_has_nothing_to_count():
    assert counted(model_id="Org/target", base_id=CODER_MODEL, base_gb=0.0) is False


def test_a_coder_model_that_is_neither_is_never_counted():
    assert counted(model_id="Org/target", base_id="Org/base", base_gb=149.0) is False


def test_another_cache_directory_is_never_counted():
    assert counted(same=False) is False


def test_a_local_target_that_the_coder_cannot_use_still_needs_its_own_download_check(tmp_path):
    # `local_snapshot` accepts a config and one shard; `coder_snapshot` rejects it. The disk row budgets no download of the target, so the
    # coder row has to do the disk check itself.
    rows = preflight(tmp_path, model_id=CODER_MODEL, target_gb=149.0, free=100.0, local=True, present=False, cache=tmp_path / "hf")
    assert (rows["coder-cache"].status, rows["coder-cache"].reason) == ("block", "disk-full")


# ---- 3. the refusal marks are matched without regard to case ------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["chip(s) 0, 1 already in use by another container", "CHIP(S) 0, 1 ALREADY IN USE BY ANOTHER CONTAINER",
                                  "Already In Use By Another Container"])
def test_a_container_refusal_is_not_retried_in_any_case(text):
    assert blocked.retryable("coder-unusable", f"the coder did not start and answer; nothing is retried: {text}") is False


@pytest.mark.parametrize("text", ["off the PCIe bus or unreadable", "OFF THE PCIE BUS OR UNREADABLE"])
def test_an_off_bus_finding_is_not_retried_in_any_case(text):
    assert blocked.retryable("hardware-unhealthy", f"chip 2 is {text}") is False


def test_an_ordinary_coder_block_is_still_retried():
    assert blocked.retryable("coder-unusable", "the coder did not start and answer") is True
    assert blocked.retryable("hardware-unhealthy", "a lease settled late") is True


@pytest.mark.parametrize("reason", [None, 5, ["x"], b"x"])
def test_a_reason_that_is_not_text_does_not_crash_the_retry_decision(reason):
    assert blocked.retryable("coder-unusable", reason) in (True, False)


# ---- 4. the refusal survives a long warning on the other stream ------------------------------------------------------------------


def _refused(stdout, stderr):
    ctl, _run = control([(1, stdout, stderr)])
    with pytest.raises(ServerChipsHeld) as caught:
        ctl.start(LEASE)
    return str(caught.value)


@pytest.mark.parametrize("stdout,stderr", [(REFUSAL, "WARNING: " + "x" * 900), (REFUSAL, "x" * 600 + "\n" + "y" * 600),
                                           ("progress " * 200 + REFUSAL, ""), ("", "y" * 800 + REFUSAL),
                                           (REFUSAL.upper(), "WARNING: " + "x" * 900)],
                         ids=["long-warning-on-stderr", "two-long-lines-on-stderr", "long-progress-before-it-on-stdout",
                              "long-line-before-it-on-stderr", "upper-case-with-a-warning"])
def test_the_exception_keeps_the_marker_whatever_the_streams_hold(stdout, stderr):
    message = _refused(stdout, stderr)
    assert blocked.CONTAINER_HOLDS_MARK in message.lower()
    assert blocked.retryable("coder-unusable", message) is False
    assert blocked.block_code({"decision": "pause", "reason": f"blocked: the coder did not start and answer; nothing is retried: {message}"}) \
        == "coder-unusable"


def test_the_exception_text_stays_bounded():
    assert len(_refused(REFUSAL, "x" * 100000)) < 2000


def test_the_two_streams_are_not_joined_into_a_refusal():
    # the search runs over both streams, so a refusal cannot be made of the end of one and the start of the other
    ctl, _run = control([(1, " by another container", "chip(s) 0, 1 already in use")])
    with pytest.raises(Exception) as caught:
        ctl.start(LEASE)
    assert not isinstance(caught.value, ServerChipsHeld)
