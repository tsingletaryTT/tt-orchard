# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Follow-ups from the reviews of PRs 506 and 564 on the stage tools (issues 516, 518, 519, 565, 566, 567).

516: the element cap is checked before a torch tensor is converted to float32 (port_check.py `layer_arrays`).
518: layer-diff hashes the bytes it hands to NumPy: one bounded read, the digest of that snapshot, and the arrays from the same bytes.
519: layer-diff says how many layers it compared out of how many, and that a layer under the threshold is not thereby clean.
565: an incomplete or failed scan of the port directory is "not known", not "unchanged": no edit command, and a sentence that says why.
566: the edit command is offered only when its old text occurs once in the file.
567: a runtime device-ops failure (op load, prefill or decode, message starting `device-ops:`) gets the cheat sheet command too."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from test_gate_lists_host_weights import PC
from test_stage_tools_last_check import FRAME_OUT, PORT_SRC, load_tool, make_stage, next_lines, text, with_port
from test_stage_tools_layer_diff import PROMPT, TOOL, build, reference_layers, run_tool, sha


# ---- 516 --------------------------------------------------------------------------------------------------------------------------------

class HugeTorchLike:
    """Looks like a torch tensor to layer_arrays: `detach` and `numel`. Converting it would allocate gigabytes, so `.float()` fails the test."""

    def __init__(self, numel):
        self._numel = numel

    def numel(self):
        return self._numel

    def detach(self):
        return self

    def float(self):
        raise AssertionError("the tensor was converted to float32 before its size was checked")


def test_a_torch_tensor_over_the_element_cap_is_refused_before_it_is_converted():
    with pytest.raises(ValueError, match="more than"):
        PC.layer_arrays([HugeTorchLike(PC.MAX_NUMEL + 1)], [1, 2, 3])


def test_the_cap_counts_what_earlier_layers_already_used():
    small = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="more than"):
        PC.layer_arrays([small, HugeTorchLike(PC.MAX_NUMEL)], [1])


# ---- 518 --------------------------------------------------------------------------------------------------------------------------------

def test_a_cached_reference_swapped_after_the_digest_is_not_the_one_that_is_loaded(tmp_path, monkeypatch):
    """The old code hashed the file through one descriptor and let NumPy open it again by path. If the digest function reports the sealed digest
    while the file holds other arrays, the arrays must still not be used: the digest is now taken from the same bytes NumPy gets."""
    tool = load_tool()
    stage = build(tmp_path)
    side = json.loads((stage / "evidence" / "layer-reference.json").read_text())
    key = {k: side[k] for k in ("prompt_sha256", "snapshot", "dtype")}
    assert tool.cached_reference(stage, key) is not None                     # the honest cache is used
    other = {f"layer_{k:03d}": a + 1 for k, a in enumerate(reference_layers())}
    np.savez(stage / "evidence" / "layer-reference.npz", **other)           # swapped for other arrays
    honest = side["npz_sha256"]
    monkeypatch.setattr(tool, "file_sha256", lambda *a, **k: (honest, None), raising=False)
    assert tool.cached_reference(stage, key) is None


def test_the_inner_part_refuses_a_port_file_that_is_not_the_one_the_outer_part_hashed(tmp_path):
    stage = build(tmp_path)
    import shutil
    shutil.copy2(TOOL, stage / "stage_tools.py")
    wrong = "0" * 64
    done = subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), "--layer-diff-inner", "0.1", "float32", wrong, "6"],
                          cwd=stage, env={"PATH": os.environ.get("PATH", "")}, capture_output=True, text=True, timeout=120)
    assert done.returncode == 2 and "refused" in done.stdout and "sha256" in done.stdout, done.stdout + done.stderr


def test_the_inner_part_still_compares_when_the_digest_matches(tmp_path):
    stage = build(tmp_path)
    import shutil
    shutil.copy2(TOOL, stage / "stage_tools.py")
    digest = sha(stage / "evidence" / "layer-port.npz")
    done = subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), "--layer-diff-inner", "0.1", "float32", digest, "6"],
                          cwd=stage, env={"PATH": os.environ.get("PATH", "")}, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "all 6 compared layers" in done.stdout, done.stdout + done.stderr


# ---- 519 --------------------------------------------------------------------------------------------------------------------------------

def test_the_report_says_how_many_layers_were_compared_out_of_how_many(tmp_path):
    ref = reference_layers()
    done = run_tool(build(tmp_path, port_layers=ref[:4], ref_layers=ref))
    assert "compared 4 of 6 reference layers" in done.stdout
    assert "layers 4 to 5 were not compared" in done.stdout


def test_a_clean_report_does_not_call_the_layers_identical(tmp_path):
    done = run_tool(build(tmp_path))
    assert "compared 6 of 6 reference layers" in done.stdout
    assert "can still change" in done.stdout and "threshold" in done.stdout


def test_a_report_with_a_missing_tail_does_not_say_all_layers_are_within_the_threshold(tmp_path):
    ref = reference_layers()
    done = run_tool(build(tmp_path, port_layers=ref[:4], ref_layers=ref))
    assert "all 4 compared layers are within" not in done.stdout and "all 6 compared layers are within" not in done.stdout


# ---- 567 --------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("op", ["load", "prefill", "decode"])
def test_a_runtime_device_ops_failure_points_at_the_cheat_sheet(tmp_path, op):
    out_text = (f"FAILED: device-ops: after load() the port holds 5 weight elements as host torch or numpy tensors\n"
                f"PORT_CHECK_FAILURE {{\"kind\": \"port\", \"op\": \"{op}\"}}\n")
    nxt = next_lines(text(load_tool(), make_stage(tmp_path, output=out_text, returncode=2)))
    assert nxt == ["next: python3 stages/2/stage_tools.py cheat ttnn.linear"]


def test_a_device_ops_message_with_the_bare_prefix_is_matched_whatever_the_op(tmp_path):
    out_text = "FAILED: device-ops: no matmul\nPORT_CHECK_FAILURE {\"kind\": \"port\", \"op\": \"load\"}\n"
    assert next_lines(text(load_tool(), make_stage(tmp_path, output=out_text, returncode=2))) == ["next: python3 stages/2/stage_tools.py cheat ttnn.linear"]


def test_another_load_failure_still_gets_no_cheat_command(tmp_path):
    out_text = "FAILED: the port failed at 'load': 'KeyError: x'\nPORT_CHECK_FAILURE {\"kind\": \"port\", \"op\": \"load\"}\n"
    assert not any("cheat" in l for l in next_lines(text(load_tool(), make_stage(tmp_path, output=out_text, returncode=2))))


# ---- 566 --------------------------------------------------------------------------------------------------------------------------------

def test_a_line_that_occurs_twice_gets_no_edit_command_but_keeps_the_location_sentence(tmp_path):
    src = "import ttnn\n\ndef prefill(model, ids):\n    logits = ttnn.to_torch(logits)\n    return logits\n\ndef decode(model, ids):\n    logits = ttnn.to_torch(logits)\n    return logits\n"
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2), src)
    out = text(load_tool(), stage)
    assert "port_model.py line 4" in out
    assert not next_lines(out)
    assert "more than one place" in out


def test_a_line_that_occurs_once_still_gets_the_edit_command(tmp_path):
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    assert len(next_lines(text(load_tool(), stage))) == 1


# ---- 565 --------------------------------------------------------------------------------------------------------------------------------

def test_a_scan_that_hits_its_limit_says_unknown_and_offers_no_edit_command(tmp_path, monkeypatch):
    tool = load_tool()
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    for k in range(5):
        extra = stage / "port" / f"helper_{k}.py"
        extra.write_text("x = 1\n")
        old = (stage / "test-result.json").stat().st_mtime_ns - 10**9
        os.utime(extra, ns=(old, old))
    monkeypatch.setattr(tool, "PORT_SCAN_MAX", 3)
    out = "\n".join(tool.last_check(stage))
    assert not next_lines(out)
    assert "could not be established" in out or "not known" in out


def test_a_directory_that_cannot_be_read_says_unknown_and_offers_no_edit_command(tmp_path):
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    locked = stage / "port" / "locked"
    locked.mkdir()
    os.chmod(locked, 0)
    try:
        out = text(load_tool(), stage)
    finally:
        os.chmod(locked, 0o755)
    assert not next_lines(out)
    assert "could not be established" in out or "not known" in out


def test_directories_and_bytecode_caches_do_not_use_up_the_scan_limit(tmp_path, monkeypatch):
    tool = load_tool()
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    for k in range(6):
        (stage / "port" / f"empty_{k}").mkdir()
    cache = stage / "port" / "__pycache__"
    cache.mkdir()
    for k in range(6):
        (cache / f"m{k}.pyc").write_bytes(b"x")
    monkeypatch.setattr(tool, "PORT_SCAN_MAX", 2)           # one source file and some slack
    assert len(next_lines("\n".join(tool.last_check(stage)))) == 1


def test_a_complete_scan_of_an_unchanged_port_is_still_unchanged(tmp_path):
    tool = load_tool()
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    assert tool.port_changed_after_check(stage) is False


def test_an_edited_file_is_still_reported_as_changed(tmp_path):
    tool = load_tool()
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    os.utime(stage / "port" / "port_model.py", None)
    os.utime(stage / "test-result.json", ns=(1, 1))
    assert tool.port_changed_after_check(stage) is True
