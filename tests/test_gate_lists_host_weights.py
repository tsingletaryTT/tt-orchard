# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 574: the stage 2 load gate names the host-held weight tensors. Runs 22 and 24 of the tiny Qwen4 rehearsal failed the gate with only a
count (590095 host elements); the coder never found which dict held the host copies. The scan (orchard/skills/full-port-templates/port_check.py
`scan_weights`) now returns the largest host tensors with paths, the failure message lists them, and `last-check` prints the whole message.
Fakes only: classes whose module name is `torch`, `numpy` or `ttnn`. Nothing imports the real packages or opens a device."""
import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path

import pytest

from orchard import full_port
from port_check_fakes import make_run, run_template
from test_stage_tools_last_check import make_stage, load_tool

TEMPLATE = Path(full_port.TEMPLATE).parent / "port_check.py"


def load_check():
    """The template is written to sit under a `stages` directory (it finds the run directory from its own path), so a copy goes in one."""
    import shutil
    import tempfile
    where = Path(tempfile.mkdtemp(prefix="gate574-")) / "stages" / "2"
    where.mkdir(parents=True)
    shutil.copy(TEMPLATE, where / "port_check.py")
    spec = importlib.util.spec_from_file_location("port_check_under_test", where / "port_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PC = load_check()


def fake(module: str, name: str, shape, **extra):
    """A tensor-like object whose type claims `module` (the scan reads the type's module only)."""
    cls = type(name, (), {"__module__": module})
    obj = cls()
    obj.shape = tuple(shape)
    n = 1
    for d in shape:
        n *= d
    obj.numel = lambda: n
    obj.size = n
    obj.dtype = "float32"
    for k, v in extra.items():
        setattr(obj, k, v)
    return obj


def torch_t(*shape):
    return fake("torch", "Tensor", shape)


def numpy_t(*shape):
    return fake("numpy", "ndarray", shape)


def device_t(*shape):
    t = fake("ttnn", "Tensor", shape)
    del t.numel, t.dtype, t.size
    return t


class Holder:
    pass


def scan(*roots, **kw):
    return PC.scan_weights(*roots, labels=["model"], **kw)


def paths(result):
    return [r["path"] for r in result["top_host"]]


def failure(host_scan):
    with pytest.raises(PC.PortFailed) as err:
        PC.DeviceOps(1, 1).check_weights(host_scan)
    return str(err.value)


# ---- the paths --------------------------------------------------------------------------------------------------------------------------

def test_a_dict_of_torch_tensors_is_listed_with_key_path_shape_elements_and_kind():
    r = scan({"layers.0.mlp.up_proj.weight": torch_t(8, 4), "norm": torch_t(4)})
    top = r["top_host"][0]
    assert top["path"] == "model['layers.0.mlp.up_proj.weight']" and top["shape"] == [8, 4] and top["elements"] == 32 and top["kind"] == "torch"


def test_nested_objects_give_attribute_and_key_paths():
    m = Holder()
    m.params = {"layers": [Holder()]}
    m.params["layers"][0].w = torch_t(3, 3)
    assert paths(scan(m)) == ["model.params['layers'][0].w"]


def test_numpy_arrays_are_listed_as_numpy():
    r = scan({"emb": numpy_t(10, 10)})
    assert r["top_host"][0]["kind"] == "numpy" and r["top_host"][0]["elements"] == 100


def test_a_ttnn_tensor_is_device_not_host_and_is_paired_by_name_with_its_host_copy():
    m = Holder()
    m.host_weights = {"up_proj.weight": torch_t(16, 16)}
    m.weights = {"up_proj.weight": device_t(16, 16)}
    r = scan(m)
    assert r["device_elements"] == 256 and r["host_elements"] == 256
    row = r["top_host"][0]
    assert row["path"] == "model.host_weights['up_proj.weight']"
    assert row["device"]["path"] == "model.weights['up_proj.weight']"


def test_a_host_tensor_with_no_device_twin_has_none():
    assert scan({"w": torch_t(2, 2)})["top_host"][0]["device"] is None


def test_only_the_eight_largest_are_listed_in_size_order_and_the_rest_are_counted():
    sizes = [5, 40, 3, 90, 7, 60, 1, 22, 11, 80, 2]
    r = scan({f"w{i}": torch_t(n) for i, n in enumerate(sizes)})
    assert [e["elements"] for e in r["top_host"]] == sorted(sizes, reverse=True)[:8]
    assert r["host_tensors"] == 11
    msg = failure(r)
    assert "and 3 more host tensors" in msg


def test_equal_sizes_keep_their_arrival_order():
    r = scan({f"w{i}": torch_t(4) for i in range(10)})
    assert paths(r) == [f"model['w{i}']" for i in range(8)]


# ---- the caps ---------------------------------------------------------------------------------------------------------------------------

def test_a_cycle_ends_and_lists_the_tensor_once():
    a = Holder()
    a.me, a.other = a, {"w": torch_t(5)}
    a.other["back"] = a
    r = scan(a)
    assert r["host_tensors"] == 1 and r["truncated"] is False


def test_a_huge_flat_list_stops_at_the_node_cap_and_says_so():
    big = [torch_t(1) for _ in range(PC.SCAN_NODES + 500)]
    r = scan(big)
    assert r["truncated"] is True and r["visited"] <= PC.SCAN_NODES and len(r["top_host"]) == PC.SCAN_TOP
    assert "walk truncated" in failure(r)


def _deep(out):
    node = leaf = {}
    for _ in range(40):                            # a chain deeper than the depth cap (SCAN_DEPTH); the tensor at its end must not be found
        node["x"] = {}
        node = node["x"]
    node["w"] = torch_t(9)
    out.put(scan(leaf))


def test_a_chain_deeper_than_the_cap_ends_with_truncated_set():
    """Run in a child with an alarm: the mutation that removes the depth cap makes this a hang, not a failure."""
    q = multiprocessing.get_context("fork").Queue()
    child = multiprocessing.get_context("fork").Process(target=_deep, args=(q,))
    child.start()
    child.join(20)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("the walk did not end: no depth cap")
    r = q.get(timeout=5)
    assert r["truncated"] is True and r["host_tensors"] == 0


# ---- the message ------------------------------------------------------------------------------------------------------------------------

def test_the_message_lists_paths_and_ends_with_the_instruction():
    msg = failure(scan({"layers.0.mlp.up_proj.weight": torch_t(8, 4)}) | {"host_elements": 900, "device_elements": 100})
    assert "model['layers.0.mlp.up_proj.weight'] shape 8x4 = 32 elements, torch" in msg
    assert "Delete or do not store the host copy" in msg and "decoder_idioms.md" in msg
    assert "590095" not in msg and "after load() the port holds 900 weight elements" in msg


def test_the_message_is_capped_and_keeps_the_instruction_when_the_listing_is_long():
    long_key = "k" * 70
    r = scan({"host": {f"{long_key}{i}": torch_t(i + 1) for i in range(50)}, "dev": {f"{long_key}{i}": device_t(2) for i in range(50)}})
    r["host_elements"], r["device_elements"] = 10 ** 6, 1
    msg = failure(r)
    listing = msg[msg.index("Largest host copies"):msg.index(" Move the computation")]
    assert "device copy" in listing and len(listing) <= PC.HOST_MESSAGE_MAX and listing.rstrip().endswith("see stages/2/decoder_idioms.md.")


def test_values_never_appear_only_names_and_shapes():
    t = torch_t(2)
    t.secret = "hunter2-token"
    msg = failure(scan({"w": t}) | {"host_elements": 9, "device_elements": 1})
    assert "hunter2" not in msg


def test_a_worker_reply_with_bad_rows_does_not_crash_the_parent():
    bad = {"host_elements": 9, "device_elements": 1, "top_host": [None, 3, {"path": 5}, {"path": "model.w\x00", "elements": "x"},
                                                                {"path": "model.ok\x1b[31m", "elements": 4, "shape": "zz", "kind": ["x"], "device": 7}]}
    msg = failure(bad)
    assert "model.ok" in msg and "\x1b" not in msg


def test_an_incomplete_scan_cannot_pass_the_gate():
    """Review of PR 582: a walk cut short by a cap showed host=0, so a port with host copies below the cap passed. An incomplete total is a
    failure whatever the share looks like."""
    msg = failure({"host_elements": 0, "device_elements": 100, "host_tensors": 0, "device_tensors": 1, "visited": 5, "truncated": True,
                   "top_host": []})
    assert "walk truncated" in msg and "complete" in msg


def test_a_container_wider_than_the_cap_sets_truncated_even_when_the_extra_children_are_skipped_primitives():
    """Review of PR 582: one device tensor, SCAN_NODES None entries, then a host tensor: the host tensor is never reached, so the scan is
    incomplete and must say so."""
    wide = [device_t(10)] + [None] * PC.SCAN_NODES + [torch_t(900)]
    r = scan(wide)
    assert r["truncated"] is True and r["host_tensors"] == 0
    assert failure(r)


# ---- follow-ups from the PR 582 review (issues 583, 584, 592, 593) ---------------------------------------------------------------------

def test_a_device_twin_is_found_when_the_path_mixes_key_and_attribute_steps():
    """Issue 583: the leaf was cut from the whole path text, so `model['host'].weight` was named `host'].weight`."""
    m = {"host": Holder(), "dev": Holder()}
    m["host"].weight = torch_t(4, 4)
    m["dev"].weight = device_t(4, 4)
    row = scan(m)["top_host"][0]
    assert row["path"] == "model['host'].weight"
    assert row["device"] is not None and row["device"]["path"] == "model['dev'].weight"


def test_the_leaf_is_the_last_step_for_each_kind_of_step():
    assert PC.step_leaf("['layers.0.up_proj.weight']") == "layers.0.up_proj.weight"
    assert PC.step_leaf(".weight") == "weight"
    assert PC.step_leaf("[3]") == "3"


def _rows(n, path_len=110, twin_len=90):
    return [{"path": "model." + "p" * path_len + str(i), "shape": [8, 4], "elements": 1000 - i, "kind": "torch",
             "device": {"path": "model." + "d" * twin_len + str(i), "shape": [8, 4]}} for i in range(n)]


def test_eight_long_rows_keep_the_remainder_count_and_the_truncation_notice_and_every_row_is_whole():
    """Issue 584: the final slice cut the notices off the end, or cut a row in half. Rows that do not fit are left out and counted."""
    scan_result = {"host_elements": 9000, "device_elements": 10, "host_tensors": 40, "truncated": True, "top_host": _rows(8)}
    msg = PC.DeviceOps.host_listing(scan_result)
    assert len(msg) <= PC.HOST_MESSAGE_MAX
    assert "walk truncated" in msg
    shown = msg.count("shape 8x4")
    assert 0 < shown < 8, "some rows must be left out for this input to test the budget"
    assert f"and {40 - shown} more host tensors" in msg
    assert msg.endswith("see stages/2/decoder_idioms.md. ")
    for line in msg.split(" | "):
        if "shape 8x4" in line:
            assert "elements, torch" in line       # a row is never cut in the middle


def test_a_short_listing_is_unchanged():
    msg = failure({"host_elements": 9, "device_elements": 1, "host_tensors": 2, "top_host": _rows(2, 10, 10)})
    assert msg.count("shape 8x4") == 2 and "more host tensors" not in msg and "walk truncated" not in msg


def test_a_credential_in_a_dict_key_is_scrubbed_before_the_key_is_shortened():
    """Issue 592: cutting a key at 80 characters left the start of a URL's user:password; a later scrub no longer recognised it."""
    key = "https://" + "u" * 40 + ":" + "P" * 60 + "@host/model.bin"
    msg = failure(scan({key: torch_t(5)}) | {"host_elements": 9, "device_elements": 1})
    assert "PPPP" not in msg and "uuuuu" not in msg and "@host" in msg


def test_a_token_that_would_be_cut_in_half_is_scrubbed_whole():
    key = "x" * 70 + "/hf_" + "A" * 34
    row = scan({key: torch_t(5)})["top_host"][0]
    assert "AAAA" not in row["path"] and "hf_" not in row["path"]


def test_an_attribute_name_and_a_printed_row_are_scrubbed_too():
    m = Holder()
    setattr(m, "https://" + "u" * 40 + ":" + "P" * 60 + "@h", torch_t(3))
    assert "PPPP" not in paths(scan(m))[0]
    msg = failure({"host_elements": 9, "device_elements": 1,
                   "top_host": [{"path": "model.https://" + "u" * 30 + ":" + "P" * 100 + "@h", "elements": 4, "shape": [1], "kind": "torch"}]})
    assert "PPPP" not in msg


def test_the_walk_builds_paths_lazily_under_one_work_budget(monkeypatch):
    """Issue 593: every child was queued with its concatenated path before the visited check, so nested lists that share children queued
    thousands of paths while visiting 17 objects. A path is now built only for an object that is kept."""
    monkeypatch.setattr(PC, "SCAN_NODES", 1000)
    built = []
    real = PC.extend_path
    monkeypatch.setattr(PC, "extend_path", lambda path, step: built.append(step) or real(path, step))
    node = [torch_t(1)]
    for _ in range(12):
        node = [node] * 16                      # 16 references to the same child at each of 12 levels
    r = scan(node)
    assert r["visited"] <= 40 and len(built) <= r["visited"] + 1
    assert r["truncated"] is False


def test_the_work_budget_counts_entries_that_are_skipped_and_marks_the_scan_truncated(monkeypatch):
    monkeypatch.setattr(PC, "SCAN_NODES", 1000)
    flat = [None] * 5000 + [torch_t(9)]
    r = scan(flat)
    assert r["truncated"] is True and r["host_tensors"] == 0 and failure(r)


def test_a_passing_split_raises_nothing():
    PC.DeviceOps(1, 1).check_weights(scan({"w": torch_t(1)}) | {"host_elements": 1, "device_elements": 9})


# ---- the real subprocess, and last-check ------------------------------------------------------------------------------------------------

COACH_PORT = '''
import ttnn
class Tensor:
    pass
Tensor.__module__ = "torch"
def _t(n):
    t = Tensor(); t.shape = (n,); t.numel = lambda: n; t.dtype = "f"
    return t
def load(model_dir, mesh_device):
    host = {"layers.0.mlp.up_proj.weight": _t(900)}
    dev = {"layers.0.mlp.up_proj.weight": ttnn.from_torch(host["layers.0.mlp.up_proj.weight"], device=mesh_device)}
    return {"params": host, "dev": dev}
def prefill(model, ids):
    ttnn.linear(None, None)
    return [0.0]
def decode_step(model, token_id, position):
    return [0.0]
def close(model):
    pass
'''


def test_the_real_check_fails_the_load_gate_with_the_path_and_last_check_prints_it_whole(tmp_path):
    run, stage, _ = make_run(tmp_path, COACH_PORT)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2, done.stdout
    assert "model['params']['layers.0.mlp.up_proj.weight'] shape 900 = 900 elements, torch; the device copy is model['dev']['layers.0.mlp.up_proj.weight']" in done.stdout
    line = [l for l in done.stdout.splitlines() if l.startswith("FAILED: ")][-1]
    stage2 = make_stage(tmp_path / "ls", output=done.stdout, returncode=2)
    out = "\n".join(load_tool().last_check(stage2))
    assert "decoder_idioms.md" in out and "model['params']['layers.0.mlp.up_proj.weight']" in out and len(line) > 300
