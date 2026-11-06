# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The gate on ttnn usage (issue 541): a port must make ttnn matmul-family calls, one per layer in a forward pass, and keep its weights on the
device. Run 18 of the tiny Qwen4 rehearsal wrote 22 versions of a port with `import ttnn  # noqa` and no call; its check passed the source
test ("mentions ttnn") and called the host torch run a chip run.

Three layers are tested here.
  * orchard/ttnn_calls.py: the static count, on text.
  * the template (orchard/skills/full-port-templates/port_check.py): the early reject and the run-time count, run as the real subprocess with
    the fake `ttnn` of tests/port_check_fakes.py. Nothing opens a device and nothing imports the real ttnn.
  * the gate (orchard/full_port.py): the record's `device_ops`, checked again.
Every guard has a mutation in the log of PR 541 that turns one of these tests red."""
import json
import re
from pathlib import Path

import pytest

from orchard import full_port, port_repair, ttnn_calls
from orchard.stages import gate_full_port
from port_check_fakes import HONEST, make_run, run_template, write_result

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run18" / "v11_host_torch_port.py"
SHARED = re.compile(r"# --- begin shared: ttnn_calls.*?# --- end shared: ttnn_calls", re.S)


def calls(src):
    return ttnn_calls.static_calls(src)["calls"]


# ---- the static count -------------------------------------------------------------------------------------------------------------------

def test_an_import_with_a_noqa_comment_is_not_a_call():
    assert calls("import ttnn  # noqa: F401\nx = 1\n") == {}


@pytest.mark.parametrize("src", [
    'x = "ttnn.linear(a, b)"\n',
    '# ttnn.matmul(a, b)\nimport ttnn\n',
    'import ttnn\n"""ttnn.linear(a, b) in a docstring"""\n',
    'import ttnn\ndef f():\n    """call ttnn.matmul(x)"""\n',
    'import ttnn\nname = ttnn.linear\n',                       # a reference is not a call
    'import ttnn\nprint(f"ttnn.matmul")\n',
])
def test_a_mention_in_a_string_a_comment_or_a_reference_is_not_a_call(src):
    assert calls(src) == {}


@pytest.mark.parametrize("src", [
    'import ttnn\ndef f(x):\n    return x\n    ttnn.linear(x, x)\n',                 # after a return
    'import ttnn\ndef f(x):\n    raise RuntimeError("no")\n    ttnn.linear(x, x)\n',
    'import ttnn\nif False:\n    ttnn.linear(1, 2)\n',
    'import ttnn\nif 0:\n    ttnn.linear(1, 2)\n',
    'import ttnn\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    ttnn.linear(1, 2)\n',
    'import ttnn\nfor i in range(3):\n    break\n    ttnn.linear(1, 2)\n',
    'import ttnn\nwhile False:\n    ttnn.linear(1, 2)\n',
])
def test_a_call_that_cannot_run_is_not_counted(src):
    assert calls(src) == {}


@pytest.mark.parametrize("src,name", [
    ("import ttnn\nttnn.linear(a, b)\n", "ttnn.linear"),
    ("import ttnn as t\nt.matmul(a, b)\n", "ttnn.matmul"),
    ("from ttnn import linear\nlinear(a, b)\n", "ttnn.linear"),
    ("from ttnn import linear as lin\nlin(a, b)\n", "ttnn.linear"),
    ("import ttnn\nop = ttnn.matmul\nop(a, b)\n", "ttnn.matmul"),
    ("import ttnn\nttnn.experimental.minimal_matmul(a, b)\n", "ttnn.experimental.minimal_matmul"),
    ("import ttnn\ndef f(x):\n    if x:\n        return ttnn.linear(x, x)\n", "ttnn.linear"),
    ("import ttnn\nclass M:\n    def f(self, x):\n        return ttnn.linear(x, x)\n", "ttnn.linear"),
    ("import ttnn\nfor i in range(2):\n    y = ttnn.matmul(a, b)\n", "ttnn.matmul"),
    ("import ttnn\nwith open('f') as f:\n    ttnn.linear(a, b)\n", "ttnn.linear"),
    ("import ttnn\ntry:\n    pass\nexcept Exception:\n    ttnn.linear(a, b)\n", "ttnn.linear"),
    ("import ttnn\nif x:\n    pass\nelse:\n    ttnn.linear(a, b)\n", "ttnn.linear"),
    ("import ttnn\nx = [ttnn.linear(a, b) for _ in range(2)]\n", "ttnn.linear"),
])
def test_a_reachable_call_in_each_plain_form_is_counted(src, name):
    assert name in calls(src)


def test_a_call_on_something_that_is_not_ttnn_is_not_counted():
    assert calls("import torch\ntorch.linear(a, b)\nlinear(a, b)\nx.ttnn.linear(a)\n") == {}


def test_a_file_that_does_not_parse_reports_an_error_and_no_calls():
    got = ttnn_calls.static_calls("def f(:\n")
    assert got["calls"] == {} and got["error"]
    assert ttnn_calls.static_calls("x = " + "(" * 5000)["error"]        # a nesting the parser gives up on


@pytest.mark.parametrize("name,matmul,compute", [
    ("ttnn.matmul", True, True), ("ttnn.linear", True, True), ("ttnn.experimental.minimal_matmul", True, True),
    ("ttnn.transformer.scaled_dot_product_attention", True, True), ("ttnn.conv2d", True, True),
    ("ttnn.add", False, True), ("ttnn.rms_norm", False, True),
    ("ttnn.from_torch", False, False), ("ttnn.open_device", False, False), ("ttnn.experimental.convert_to_chw", False, True),
])
def test_which_names_are_matmul_family_and_which_are_computation(name, matmul, compute):
    assert ttnn_calls.is_matmul(name) is matmul and ttnn_calls.is_compute(name) is compute


@pytest.mark.parametrize("config,expected", [
    ({"num_hidden_layers": 4}, (4, 4)), ({"text_config": {"num_hidden_layers": 28}, "model_type": "x"}, (28, 28)),
    ({"layer_types": ["a", "b", "c"]}, (3, 3)), ({"n_layer": 12}, (12, 12)),
    ({}, (1, None)), ({"num_hidden_layers": 0}, (1, None)), ({"num_hidden_layers": -3}, (1, None)), ({"num_hidden_layers": True}, (1, None)),
    ({"num_hidden_layers": 10 ** 20}, (1, None)), ({"num_hidden_layers": "4"}, (1, None)), ("text", (1, None)), (None, (1, None)),
    ({"text_config": "x", "num_hidden_layers": 2}, (2, 2)),
])
def test_the_floor_is_one_matmul_per_layer_and_one_when_the_config_names_no_count(config, expected):
    assert ttnn_calls.layer_floor(config) == expected


def test_the_template_carries_the_same_static_count_and_floor_as_the_gate_byte_for_byte():
    # The template runs from the stage directory and cannot import orchard, so it holds a copy. The two must not drift.
    mine = SHARED.search(Path(ttnn_calls.__file__).read_text()).group(0)
    theirs = SHARED.search(full_port.TEMPLATE.read_text()).group(0)
    assert mine == theirs


# ---- the run-18 port ------------------------------------------------------------------------------------------------------------------------

def test_the_run_18_port_has_no_ttnn_call_by_the_static_count():
    text = FIXTURE.read_text()
    assert "import ttnn  # noqa: F401" in text and "F.silu(zf.float())" in text          # the file really is the 0.5625 port with the wrong gate
    assert calls(text) == {}


def test_the_template_rejects_the_run_18_port_before_it_touches_the_device(tmp_path):
    run, stage, _ = make_run(tmp_path, FIXTURE.read_text())
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and not (stage / "evidence" / "port-check.json").exists()
    assert "device-ops: no source file under stages/2/port/ makes a reachable ttnn matmul-family call" in done.stdout
    assert "loaded in" not in done.stdout and "port| " not in done.stdout          # the worker never started


def test_the_gate_refuses_the_run_18_port_even_with_a_record_that_claims_a_pass(tmp_path):
    # A record built by hand, as an agent that wrote the record would build it, with the run-18 source and counts that clear the floor.
    from test_full_port import build
    stage_dir, run = build(tmp_path, model_src=FIXTURE.read_text(), record={"device_ops": good_ops()})
    g = gate_full_port(stage_dir, run)
    assert not g.ok and any("no model source in port_sources makes a reachable ttnn matmul-family call" in r for r in g.reasons)


# ---- the template at run time, on the fake ttnn ---------------------------------------------------------------------------------------

def port_with(body_prefill: str, *, load_extra: str = "", decode: str = "_peak(token_id + 1)", header: str = "") -> str:
    """A port whose forward passes do `body_prefill` (extra statements) and then answer the way HONEST does."""
    return f'''
import ttnn
{header}
VOCAB = 200
def _peak(i):
    v = [0.0] * VOCAB
    v[i] = 1.0
    return v
def load(model_dir, mesh_device):
    {load_extra or "pass"}
    return {{"dir": model_dir, "device": mesh_device{', "w": W' if load_extra else ''}}}
def prefill(model, ids):
{body_prefill}
    return _peak(100)
def decode_step(model, token_id, position):
{body_prefill}
    return {decode}
def close(model):
    pass
'''


def with_layers(tmp_path, port, layers):
    run, stage, snap = make_run(tmp_path, port)
    (snap / "config.json").write_text(json.dumps({"model_type": "toy", "num_hidden_layers": layers}))
    return run, stage


LINEARS = "    for _ in range({n}):\n        ttnn.linear(None, None)"


def test_a_port_that_calls_ttnn_once_per_layer_passes_and_the_record_shows_the_counts(tmp_path):
    run, stage = with_layers(tmp_path, port_with(LINEARS.format(n=4)), layers=4)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout
    ops = json.loads((stage / "evidence" / "port-check.json").read_text())["device_ops"]
    assert ops["floor"] == 4 and ops["layers"] == 4 and ops["prefill_matmul_calls"] == 4 and ops["decode_matmul_calls_max"] == 4
    assert ops["top_calls"]["ttnn.linear"] > 4 and ops["static_matmul_calls"] >= 1
    write_result(stage)
    g = gate_full_port(stage, run)
    assert g.ok, g.reasons


def test_a_port_with_real_looking_calls_passes_with_its_weights_on_the_device(tmp_path):
    port = port_with("    ttnn.matmul(model['w'], model['w'])\n    ttnn.linear(model['w'], model['w'])", load_extra="W = ttnn.from_torch([[1.0]], device=mesh_device)",
                     header="W = None")
    port = port.replace("    W = ttnn.from_torch", "    global W\n    W = ttnn.from_torch")
    run, stage = with_layers(tmp_path, port, layers=2)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout
    ops = json.loads((stage / "evidence" / "port-check.json").read_text())["device_ops"]
    assert ops["weights"]["device_tensors"] >= 1 and ops["weights"]["host_elements"] == 0 and ops["prefill_matmul_calls"] == 2


def test_one_call_fewer_than_the_layers_fails_at_the_first_prefill_and_says_what_to_do(tmp_path):
    run, stage = with_layers(tmp_path, port_with(LINEARS.format(n=3)), layers=4)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and not (stage / "evidence" / "port-check.json").exists()
    assert "the first prefill made 3 matmul-family ttnn calls" in done.stdout and "at least 4" in done.stdout and "4 layers" in done.stdout
    assert "ttnn.from_torch" in done.stdout and "cheat_sheet.md" in done.stdout and "ttnn.linear x3" in done.stdout


def test_the_failure_is_one_the_coder_can_repair_and_names_the_gate(tmp_path):
    run, stage = with_layers(tmp_path, port_with(LINEARS.format(n=1)), layers=4)
    done = run_template(tmp_path, run, stage)
    assert port_repair.parse_failure(done.stdout) == {"kind": "port", "op": "prefill"}


def test_a_call_in_a_function_nobody_calls_does_not_count_at_run_time(tmp_path):
    # The static count sees the call (so the early reject lets the port through) and the run-time count sees none.
    port = port_with("    pass", header="def never_used(x):\n    return ttnn.linear(x, x)")
    run, stage = with_layers(tmp_path, port, layers=1)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "the first prefill made 0 matmul-family ttnn calls" in done.stdout


def test_calls_that_only_manage_the_device_or_copy_data_do_not_count_as_matmuls(tmp_path):
    run, stage = with_layers(tmp_path, port_with("    ttnn.add(None, None)\n    ttnn.from_torch(None)"), layers=1)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "no source file under stages/2/port/ makes a reachable ttnn matmul-family call" in done.stdout
    assert "ttnn.add x" in done.stdout


def test_weights_left_as_host_arrays_fail_the_run_after_load(tmp_path):
    np = pytest.importorskip("numpy", reason="the weight scan test builds a numpy array")
    port = port_with(LINEARS.format(n=1), load_extra="global W\n    import numpy\n    W = numpy.zeros((64, 64), dtype='float32')", header="W = None")
    run, stage = with_layers(tmp_path, port, layers=1)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "4096 weight elements as host torch or numpy tensors and 0 as ttnn tensors" in done.stdout
    assert "ttnn.from_torch" in done.stdout and port_repair.parse_failure(done.stdout)["op"] == "load"


def test_a_mostly_device_port_with_a_small_host_table_is_not_refused(tmp_path):
    pytest.importorskip("numpy", reason="the weight scan test builds a numpy array")
    port = port_with(LINEARS.format(n=1), header="W = None; T = None",
                     load_extra="global W, T\n    import numpy\n    W = ttnn.from_torch(None, device=mesh_device)\n    T = numpy.zeros((2, 2))")
    port = port.replace("(None, device=mesh_device)", "([[1.0]], device=mesh_device)")
    run, stage = with_layers(tmp_path, port, layers=1)
    # the fake from_torch tensor has 16 elements and the host table 4: the host share is 20 percent
    assert run_template(tmp_path, run, stage).returncode == 0


def test_a_decode_that_never_uses_the_device_fails_after_the_teacher_forced_loop(tmp_path):
    port = port_with(LINEARS.format(n=2)).replace("def decode_step(model, token_id, position):\n" + LINEARS.format(n=2), "def decode_step(model, token_id, position):\n    pass")
    run, stage = with_layers(tmp_path, port, layers=2)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "no decode step made as many as 2" in done.stdout
    assert port_repair.parse_failure(done.stdout)["op"] == "decode"


def test_a_decode_that_replays_a_trace_captured_during_load_passes(tmp_path):
    # load runs the capture pass (two linear calls), prefill makes two, every decode step only replays the trace.
    port = port_with(LINEARS.format(n=2), load_extra="ttnn.linear(None, None)\n    ttnn.linear(None, None)")
    port = port.replace("def decode_step(model, token_id, position):\n" + LINEARS.format(n=2),
                        "def decode_step(model, token_id, position):\n    ttnn.execute_trace(None, None)")
    run, stage = with_layers(tmp_path, port.replace('{"dir": model_dir, "device": mesh_device, "w": W}', '{"dir": model_dir, "device": mesh_device}'), layers=2)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout
    assert json.loads((stage / "evidence" / "port-check.json").read_text())["device_ops"]["decode_trace_replays"] >= 1


def test_trace_replays_with_no_captured_pass_anywhere_do_not_excuse_the_decode(tmp_path):
    port = port_with(LINEARS.format(n=2)).replace("def decode_step(model, token_id, position):\n" + LINEARS.format(n=2),
                                                   "def decode_step(model, token_id, position):\n    ttnn.execute_trace(None, None)")
    run, stage = with_layers(tmp_path, port, layers=2)
    assert "no decode step made as many as 2" in run_template(tmp_path, run, stage).stdout


def test_trace_replays_do_not_excuse_a_prefill_that_made_no_calls(tmp_path):
    port = port_with("    pass").replace("def decode_step(model, token_id, position):\n    pass", "def decode_step(model, token_id, position):\n    ttnn.execute_trace(None, None)")
    port += "\ndef _x():\n    ttnn.linear(None, None)\n"
    run, stage = with_layers(tmp_path, port, layers=2)
    assert "the first prefill made 0" in run_template(tmp_path, run, stage).stdout


def test_the_counting_proxy_leaves_classes_and_constants_alone(tmp_path):
    port = port_with(LINEARS.format(n=1) + "\n    assert isinstance(ttnn.Tensor((1,)), ttnn.Tensor)\n    assert ttnn.FabricConfig.FABRIC_1D == 'fabric-1d'")
    run, stage = with_layers(tmp_path, port, layers=1)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout


def test_a_floor_of_one_applies_when_the_config_names_no_layer_count(tmp_path):
    run, stage, _ = make_run(tmp_path, HONEST)         # the fake checkpoint has no config.json
    assert run_template(tmp_path, run, stage).returncode == 0
    assert json.loads((stage / "evidence" / "port-check.json").read_text())["device_ops"]["floor"] == 1


# ---- the gate on the record ----------------------------------------------------------------------------------------------------------------

def good_ops(**over):
    return {"floor": 4, "layers": 4, "prefill_matmul_calls": 4, "decode_matmul_calls_max": 4, "decode_trace_replays": 0,
            "matmul_calls_total": 200, "static_matmul_calls": 3, "weights": {"host_elements": 0, "device_elements": 5000}, "top_calls": {}, **over}


def gate_reasons(tmp_path, ops, **kw):
    from test_full_port import build
    stage_dir, run = build(tmp_path, record={"device_ops": ops}, **kw)
    return gate_full_port(stage_dir, run)


def test_a_record_with_good_counts_passes(tmp_path):
    g = gate_reasons(tmp_path, good_ops())
    assert g.ok, g.reasons


@pytest.mark.parametrize("over,word", [
    ({"prefill_matmul_calls": 3}, "first prefill made 3"),
    ({"prefill_matmul_calls": 0}, "first prefill made 0"),
    ({"decode_matmul_calls_max": 0}, "no decode step"),
    ({"decode_matmul_calls_max": 0, "decode_trace_replays": 5, "matmul_calls_total": 7}, "no decode step"),
    ({"weights": {"host_elements": 900, "device_elements": 100}}, "weights stayed on the host"),
    ({"floor": 0}, "floor must be"), ({"floor": "4"}, "floor must be"), ({"floor": 10 ** 9}, "floor must be"), ({"floor": True}, "floor must be"),
    ({"prefill_matmul_calls": "4"}, "whole-number counts"), ({"decode_matmul_calls_max": None}, "whole-number counts"),
    ({"prefill_matmul_calls": -1}, "whole-number counts"), ({"matmul_calls_total": 1.5}, "whole-number counts"),
])
def test_the_gate_refuses_each_record_that_does_not_show_device_work(tmp_path, over, word):
    g = gate_reasons(tmp_path, good_ops(**over))
    assert not g.ok and any(word in r for r in g.reasons), g.reasons


def test_trace_replays_stand_for_the_decode_calls_when_the_whole_run_made_the_captured_pass(tmp_path):
    assert gate_reasons(tmp_path, good_ops(decode_matmul_calls_max=0, decode_trace_replays=30, matmul_calls_total=8)).ok


@pytest.mark.parametrize("ops", [None, [], "x", 5])
def test_a_record_whose_device_ops_is_not_an_object_is_refused(tmp_path, ops):
    g = gate_reasons(tmp_path, ops)
    assert not g.ok and any("has no device_ops" in r for r in g.reasons)


def test_a_record_without_the_field_is_refused_not_skipped(tmp_path):
    from test_full_port import build
    stage_dir, run = build(tmp_path)
    path = stage_dir / "evidence" / "port-check.json"
    rec = json.loads(path.read_text())
    del rec["device_ops"]
    path.write_text(json.dumps(rec))
    out = stage_dir / "evidence" / "hw-test-output.txt"
    import hashlib
    out.write_text(f"PORT_CHECK_SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    marker = json.loads((stage_dir / "test-result.json").read_text())
    marker["output"]["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (stage_dir / "test-result.json").write_text(json.dumps(marker))
    g = gate_full_port(stage_dir, run)
    assert not g.ok and any("has no device_ops" in r for r in g.reasons)


def test_a_floor_below_the_one_the_checkpoints_config_gives_is_refused(tmp_path):
    from test_full_port import build
    snap = tmp_path / "snapshot_dir"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 40}))
    stage_dir, run = build(tmp_path, record={"device_ops": good_ops(floor=1, prefill_matmul_calls=1, decode_matmul_calls_max=1)})
    (stage_dir / "port_check_config.json").write_text(json.dumps({"model_snapshot": str(snap)}))
    import hashlib
    path = stage_dir / "evidence" / "port-check.json"
    rec = json.loads(path.read_text())
    rec["config"]["sha256"] = hashlib.sha256((stage_dir / "port_check_config.json").read_bytes()).hexdigest()
    path.write_text(json.dumps(rec))
    out = stage_dir / "evidence" / "hw-test-output.txt"
    out.write_text(f"PORT_CHECK_SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    marker = json.loads((stage_dir / "test-result.json").read_text())
    marker["output"]["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (stage_dir / "test-result.json").write_text(json.dumps(marker))
    g = gate_full_port(stage_dir, run)
    assert not g.ok and any("below the 40 the checkpoint's config gives" in r for r in g.reasons) and any("first prefill made 1" in r for r in g.reasons)


def test_no_name_in_the_shared_block_is_defined_twice_in_the_template():
    # The block was first pasted with a `MAX_LAYERS` that silently replaced the template's own constant of the same name (a test of the layer hook caught it).
    import ast
    tree = ast.parse(full_port.TEMPLATE.read_text())
    seen = {}
    for node in tree.body:
        names = [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else \
                [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else []
        for n in names:
            seen[n] = seen.get(n, 0) + 1
    assert not {n: c for n, c in seen.items() if c > 1}
