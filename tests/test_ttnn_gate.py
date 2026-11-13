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
    ("ttnn.create_matmul_attributes", False, True), ("ttnn.MatmulMultiCoreReuseProgramConfig", False, True),
    ("ttnn.linear_program_config", False, True),
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


def test_a_port_that_defines_its_functions_before_importing_ttnn_still_has_its_calls_counted():
    """Function bodies run after the whole module has loaded, so a later import or alias binds the name they use (Copilot, PR 543)."""
    src = "def f(x):\n    return ttnn.linear(x, x)\n\ndef g(x):\n    return op(x, x)\n\nimport ttnn\nop = ttnn.matmul\n"
    got = calls(src)
    assert got.get("ttnn.linear") == 1 and got.get("ttnn.matmul") == 1


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
    assert ops["trace_captures"] == []          # issue 547: the field is always written, empty when the port captured no trace
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


def traced_port(load: str, decode: str = "    ttnn.execute_trace(model['device'], TRACE[0])", prefill_linears: int = 4) -> str:
    """A port for the trace tests (issue 547): `load` is the body of load() (it may set TRACE[0] and TRACE[1]), prefill makes
    `prefill_linears` linear calls, and decode_step runs `decode` and then answers the way HONEST does."""
    return f'''
import ttnn
VOCAB = 200
TRACE = [None, None]
def _peak(i):
    v = [0.0] * VOCAB
    v[i] = 1.0
    return v
def _linears(n):
    for _ in range(n):
        ttnn.linear(None, None)
def load(model_dir, mesh_device):
{load}
    return {{"dir": model_dir, "device": mesh_device}}
def prefill(model, ids):
    _linears({prefill_linears})
    return _peak(100)
def decode_step(model, token_id, position):
{decode}
    return _peak(token_id + 1)
def close(model):
    pass
'''


CAPTURE_4 = ("    TRACE[0] = ttnn.begin_trace_capture(mesh_device)\n    _linears(4)\n"
             "    ttnn.end_trace_capture(mesh_device, TRACE[0])")


def trace_run(tmp_path, port, layers=4):
    run, stage = with_layers(tmp_path, port, layers=layers)
    return run_template(tmp_path, run, stage), stage


def test_a_decode_that_replays_a_trace_captured_during_load_passes(tmp_path):
    # load captures a pass of four linear calls, prefill makes four, every decode step only replays that trace.
    done, stage = trace_run(tmp_path, traced_port(CAPTURE_4))
    assert done.returncode == 0, done.stdout
    ops = json.loads((stage / "evidence" / "port-check.json").read_text())["device_ops"]
    assert ops["decode_trace_replays"] >= 1
    assert ops["trace_captures"] == [{"id": 1, "matmul_calls": 4, "decode_replays": ops["decode_trace_replays"]}]


def test_a_decode_that_replays_an_unrelated_empty_trace_is_refused_even_when_the_run_total_is_high(tmp_path):
    """Issue 547, the case from the review of PR 543: eight matmuls outside any capture (four in load, four in prefill) and a decode that
    replays a trace that holds none. The old rule (any replay plus a run total of twice the floor) passed it."""
    load = ("    _linears(4)\n    TRACE[0] = ttnn.begin_trace_capture(mesh_device)\n"
            "    ttnn.end_trace_capture(mesh_device, TRACE[0])")
    done, _ = trace_run(tmp_path, traced_port(load))
    assert done.returncode == 2 and "no decode step made as many as 4" in done.stdout, done.stdout
    assert "a replayed trace whose capture made at least 4" in done.stdout


def test_a_replay_of_another_trace_than_the_full_capture_is_refused(tmp_path):
    load = CAPTURE_4 + ("\n    TRACE[1] = ttnn.begin_trace_capture(mesh_device)\n    ttnn.end_trace_capture(mesh_device, TRACE[1])")
    done, _ = trace_run(tmp_path, traced_port(load, decode="    ttnn.execute_trace(model['device'], TRACE[1])"))
    assert done.returncode == 2 and "no decode step made as many as 4" in done.stdout, done.stdout


def test_a_capture_that_never_ended_does_not_excuse_the_decode(tmp_path):
    load = "    TRACE[0] = ttnn.begin_trace_capture(mesh_device)\n    _linears(4)"
    done, _ = trace_run(tmp_path, traced_port(load))
    assert done.returncode == 2 and "no decode step made as many as 4" in done.stdout, done.stdout


def test_a_capture_whose_end_failed_does_not_excuse_the_decode(tmp_path):
    load = ("    class Bad:\n        fail_end = True\n    TRACE[0] = ttnn.begin_trace_capture(mesh_device)\n    _linears(4)\n"
            "    try:\n        ttnn.end_trace_capture(Bad(), TRACE[0])\n    except RuntimeError:\n        pass")
    done, _ = trace_run(tmp_path, traced_port(load))
    assert done.returncode == 2 and "no decode step made as many as 4" in done.stdout, done.stdout


def test_a_capture_below_the_floor_does_not_excuse_the_decode(tmp_path):
    load = ("    TRACE[0] = ttnn.begin_trace_capture(mesh_device)\n    _linears(3)\n"
            "    ttnn.end_trace_capture(mesh_device, TRACE[0])\n    _linears(5)")
    done, _ = trace_run(tmp_path, traced_port(load))
    assert done.returncode == 2 and "no decode step made as many as 4" in done.stdout, done.stdout


def test_trace_ids_passed_by_keyword_are_matched(tmp_path):
    load = ("    TRACE[0] = ttnn.begin_trace_capture(mesh_device=mesh_device)\n    _linears(4)\n"
            "    ttnn.end_trace_capture(mesh_device=mesh_device, trace_id=TRACE[0])")
    done, _ = trace_run(tmp_path, traced_port(load, decode="    ttnn.execute_trace(mesh_device=model['device'], trace_id=TRACE[0], blocking=False)"))
    assert done.returncode == 0, done.stdout


def load_template(tmp_path):
    """The template as a module (it sits under a `stages` directory, since it finds the run directory from its own path)."""
    import importlib.util
    import shutil
    where = tmp_path / "tpl" / "stages" / "2"
    where.mkdir(parents=True)
    shutil.copy(full_port.TEMPLATE, where / "port_check.py")
    spec = importlib.util.spec_from_file_location("port_check_trace_test", where / "port_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Ids:
    """begin_trace_capture that hands out the ids it is given, in order (the runtime may reuse an id after release_trace)."""

    def __init__(self, *ids):
        self.ids = list(ids)

    def __call__(self, device, **kw):
        return self.ids.pop(0)


def test_the_worker_trace_table_starts_a_capture_again_under_a_reused_id(tmp_path):
    pc = load_template(tmp_path)
    begin, end = pc.Counted(Ids(7, 7), "ttnn.begin_trace_capture"), pc.Counted(lambda d, t, **kw: None, "ttnn.end_trace_capture")
    linear, replay = pc.Counted(lambda *a, **k: None, "ttnn.linear"), pc.Counted(lambda d, t, **kw: None, "ttnn.execute_trace")
    tid = begin(None)
    for _ in range(4):
        linear(None, None)
    end(None, tid)
    assert pc.take_traces()["captures"] == [[7, 4]]
    tid = begin(None)                     # the same id again, captured with no matmul
    end(None, tid)
    replay(None, tid)
    got = pc.take_traces()
    assert got == {"captures": [[7, 0]], "replays": [[7, 0, 1]]}      # the replay is of the empty pass
    assert pc.take_traces()["replays"] == []        # the replays are per request; the captures stay


def test_the_worker_trace_table_ignores_ids_it_cannot_read_and_is_bounded(tmp_path):
    pc = load_template(tmp_path)
    end, replay = pc.Counted(lambda d, t, **kw: None, "ttnn.end_trace_capture"), pc.Counted(lambda d, t, **kw: None, "ttnn.execute_trace")
    ids = [None, "x", object(), True, -1, 2 ** 40] + list(range(pc.TRACE_MAX + 5))
    begin = pc.Counted(Ids(*ids), "ttnn.begin_trace_capture")
    for _ in ids:
        tid = begin(None)
        end(None, tid)
        replay(None, tid)
    got = pc.take_traces()
    assert len(got["captures"]) == pc.TRACE_MAX and all(type(i) is int and 0 <= i < 2 ** 32 for i, _ in got["captures"])
    assert len(got["replays"]) <= pc.TRACE_MAX


def worker_ops(pc, *ids):
    return (pc.Counted(Ids(*ids), "ttnn.begin_trace_capture"), pc.Counted(lambda d, t, **kw: None, "ttnn.end_trace_capture"),
            pc.Counted(lambda *a, **k: None, "ttnn.linear"), pc.Counted(lambda d, t, **kw: None, "ttnn.execute_trace"))


def test_an_end_that_names_another_id_records_no_capture(tmp_path):
    pc = load_template(tmp_path)
    begin, end, linear, replay = worker_ops(pc, 7)
    begin(None)
    for _ in range(4):
        linear(None, None)
    end(None, 8)
    replay(None, 8)
    assert pc.take_traces() == {"captures": [], "replays": []}


def test_a_capture_whose_end_raised_is_not_recorded_by_a_later_end(tmp_path):
    pc = load_template(tmp_path)
    begin, _, linear, replay = worker_ops(pc, 7)
    calls = []

    def end_fn(d, t, **kw):
        calls.append(t)
        if len(calls) == 1:
            raise RuntimeError("capture failed")
    end = pc.Counted(end_fn, "ttnn.end_trace_capture")
    tid = begin(None)
    for _ in range(4):
        linear(None, None)
    with pytest.raises(RuntimeError):
        end(None, tid)
    end(None, tid)                         # the capture already failed; a second end does not revive it
    assert pc.take_traces()["captures"] == []


def test_a_new_capture_under_a_used_id_drops_the_old_pass_even_before_it_ends(tmp_path):
    pc = load_template(tmp_path)
    begin, end, linear, replay = worker_ops(pc, 7, 7)
    tid = begin(None)
    for _ in range(4):
        linear(None, None)
    end(None, tid)
    begin(None)                            # captured again and never ended
    replay(None, 7)
    assert pc.take_traces() == {"captures": [], "replays": []}


def test_a_bool_or_a_numeric_string_is_not_a_trace_id(tmp_path):
    pc = load_template(tmp_path)
    begin, end, linear, replay = worker_ops(pc, True, "5", 3.0)
    for _ in range(3):
        end(None, begin(None))
    assert pc.take_traces()["captures"] == [] and pc.TRACES == {}


def test_the_worker_table_itself_stays_within_its_bound(tmp_path):
    pc = load_template(tmp_path)
    n = pc.TRACE_MAX + 5
    begin, end, linear, replay = worker_ops(pc, *range(n))
    for _ in range(n):
        tid = begin(None)
        end(None, tid)
        replay(None, tid)
    assert len(pc.TRACES) == pc.TRACE_MAX and len(pc.TRACE_REPLAYS) <= pc.TRACE_MAX


def test_the_parent_counts_replays_only_in_decode_steps(tmp_path):
    pc = load_template(tmp_path)
    ops = pc.DeviceOps(4, 4)
    ops.add_traces({"captures": [[1, 4]], "replays": [[1, 4, 3]]})               # a load or prefill reply
    assert ops.full_replays() == 0 and ops.record()["trace_captures"] == [{"id": 1, "matmul_calls": 4, "decode_replays": 0}]
    ops.add_traces({"captures": [[1, 4]], "replays": [[1, 4, 2]]}, decode=True)
    assert ops.full_replays() == 2


@pytest.mark.parametrize("traces", [None, [], "x", {"captures": "x"}, {"captures": [[1]]}, {"captures": [["1", 4]]},
                                    {"captures": [[1, -1]]}, {"captures": [[1, 10 ** 20]]}, {"captures": [[True, 4]]},
                                    {"replays": [[1, "2"]]}, {"captures": [[1, 4]] * 1000}])
def test_the_parent_takes_only_well_formed_trace_rows(tmp_path, traces):
    pc = load_template(tmp_path)
    ops = pc.DeviceOps(4, 4)
    ops.add_traces(traces, decode=True)
    rec = ops.record()["trace_captures"]
    assert isinstance(rec, list) and len(rec) <= pc.TRACE_MAX
    assert all(type(r["id"]) is int and type(r["matmul_calls"]) is int and r["matmul_calls"] >= 0 for r in rec)
    assert not any(r["matmul_calls"] >= 4 and r["decode_replays"] for r in rec)


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
            "matmul_calls_total": 200, "static_matmul_calls": 3, "weights": {"host_elements": 0, "device_elements": 5000}, "top_calls": {},
            "trace_captures": [], **over}


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


def test_trace_replays_stand_for_the_decode_calls_when_the_replayed_capture_met_the_floor(tmp_path):
    ops = good_ops(decode_matmul_calls_max=0, decode_trace_replays=30, matmul_calls_total=8,
                   trace_captures=[{"id": 1, "matmul_calls": 4, "decode_replays": 30}])
    g = gate_reasons(tmp_path, ops)
    assert g.ok, g.reasons


@pytest.mark.parametrize("captures", [
    [],                                                                   # replays of nothing captured (the PR 543 review case)
    [{"id": 1, "matmul_calls": 0, "decode_replays": 30}],                  # the replayed trace holds no matmul
    [{"id": 1, "matmul_calls": 3, "decode_replays": 30}],                  # below the floor of 4
    [{"id": 1, "matmul_calls": 4, "decode_replays": 0}, {"id": 2, "matmul_calls": 0, "decode_replays": 30}],   # the full one is not replayed
])
def test_the_gate_refuses_replays_that_no_full_capture_backs_even_with_a_high_run_total(tmp_path, captures):
    """Issue 547: the run total no longer stands for a replayed pass; the replayed capture itself must meet the floor."""
    ops = good_ops(decode_matmul_calls_max=0, decode_trace_replays=30, matmul_calls_total=200, trace_captures=captures)
    g = gate_reasons(tmp_path, ops)
    assert not g.ok and any("no decode step" in r for r in g.reasons), g.reasons


def test_a_record_without_trace_captures_cannot_excuse_its_decode_with_replays(tmp_path):
    ops = good_ops(decode_matmul_calls_max=0, decode_trace_replays=30, matmul_calls_total=200)
    del ops["trace_captures"]
    g = gate_reasons(tmp_path, ops)
    assert not g.ok and any("trace_captures" in r for r in g.reasons), g.reasons


def test_a_record_without_trace_captures_still_passes_when_its_decode_met_the_floor(tmp_path):
    ops = good_ops()
    del ops["trace_captures"]
    g = gate_reasons(tmp_path, ops)
    assert g.ok, g.reasons


@pytest.mark.parametrize("captures", [None, "x", 5, {"id": 1}, [None], [{"id": "1", "matmul_calls": 4, "decode_replays": 1}],
                                      [{"id": 1, "matmul_calls": -1, "decode_replays": 1}], [{"id": 1, "matmul_calls": 4}],
                                      [{"id": 1, "matmul_calls": 4, "decode_replays": 10 ** 20}],
                                      [{"id": True, "matmul_calls": 4, "decode_replays": 1}],
                                      [{"id": i, "matmul_calls": 4, "decode_replays": 1} for i in range(100)]])
def test_a_malformed_trace_captures_field_is_refused(tmp_path, captures):
    g = gate_reasons(tmp_path, good_ops(trace_captures=captures))
    assert not g.ok and any("trace_captures" in r for r in g.reasons), g.reasons


def test_a_full_capture_does_not_meet_a_floor_the_checkpoints_config_raises(tmp_path):
    from test_full_port import build
    import hashlib
    snap = tmp_path / "snapshot_dir"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 40}))
    ops = good_ops(prefill_matmul_calls=40, decode_matmul_calls_max=0, decode_trace_replays=5,
                   trace_captures=[{"id": 1, "matmul_calls": 4, "decode_replays": 5}])
    stage_dir, run = build(tmp_path, record={"device_ops": ops})
    (stage_dir / "port_check_config.json").write_text(json.dumps({"model_snapshot": str(snap)}))
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
    assert not g.ok and any("no decode step" in r for r in g.reasons), g.reasons


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
