# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ops_inventory.py, the template `tt-orchard inventory` runs in the reference interpreter. Run end to end on a tiny
config; nothing is downloaded and no device is opened (the model lives on the meta device)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "ops-inventory-templates" / "ops_inventory.py"
QWEN3_TINY = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": 128, "hidden_size": 32,
              "intermediate_size": 64, "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
              "head_dim": 16, "max_position_embeddings": 64}


def stack_python():
    """An interpreter with torch and transformers: $ORCHARD_REFERENCE_PYTHON, else this one, else None."""
    for cand in (os.environ.get("ORCHARD_REFERENCE_PYTHON"), sys.executable):
        if cand and subprocess.run([cand, "-c", "import torch, transformers"], capture_output=True).returncode == 0:
            return cand
    return None


PY = stack_python()
needs_stack = pytest.mark.skipif(PY is None, reason=(
    "SKIPPED: no interpreter with torch and transformers (set ORCHARD_REFERENCE_PYTHON), so the ops-inventory "
    "template was not run end to end"))


def run_template(tmp_path, config: dict, *extra):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    out = tmp_path / "raw.json"
    done = subprocess.run([PY or sys.executable, str(TEMPLATE), "--model-dir", str(d), "--out", str(out), *extra],
                          capture_output=True, text=True)
    return done, (json.loads(out.read_text()) if out.exists() else None)


def test_a_config_with_auto_map_is_skipped_and_no_code_runs(tmp_path):
    """Review focus 1. A repo's own modeling code is never imported. This test needs no torch."""
    done, raw = run_template(tmp_path, {**QWEN3_TINY, "auto_map": {"AutoModelForCausalLM": "modeling_x.X"}})
    assert done.returncode == 0 and "remote code" in raw["skipped"] and "ops" not in raw


def test_a_missing_config_exits_2(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    done = subprocess.run([sys.executable, str(TEMPLATE), "--model-dir", str(d), "--out", str(tmp_path / "o.json")],
                          capture_output=True, text=True)
    assert done.returncode == 2 and "config.json" in done.stderr


@needs_stack
def test_a_class_the_installed_transformers_lacks_is_skipped(tmp_path):
    done, raw = run_template(tmp_path, {**QWEN3_TINY, "architectures": ["NoSuchForCausalLM"]})
    assert done.returncode == 0 and "NoSuchForCausalLM" in raw["skipped"]


@needs_stack
def test_a_tiny_qwen3_traces_and_reports_layers_and_ops(tmp_path):
    done, raw = run_template(tmp_path, QWEN3_TINY, "--layers", "2")
    assert done.returncode == 0, done.stderr
    assert raw["layers_in_config"] == 4 and raw["layers_traced"] == 2 and raw["traced_class"] == "Qwen3ForCausalLM"
    ops = {o["op"]: o for o in raw["ops"]}
    assert len(ops) >= 10 and not raw["failures"]
    assert any(o["count_in_layers"] > 0 for o in ops.values())
    assert any(o["count_in_layers"] == 0 for o in ops.values()), "embedding and the final norm are outside the layers"
    assert all(0 <= o["count_in_layers"] <= o["count"] for o in ops.values())


def test_reduce_layers_reports_none_when_the_config_has_no_layer_count():
    """Review focus 5. The template must not invent a layer count. No torch needed."""
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location("ops_inventory_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.reduce_layers(types.SimpleNamespace(hidden_size=8), 2) == (None, None)
    cfg = types.SimpleNamespace(num_hidden_layers=24)
    assert mod.reduce_layers(cfg, 2) == (24, 2) and cfg.num_hidden_layers == 2
    assert mod.reduce_layers(types.SimpleNamespace(num_hidden_layers=1), 2) == (1, 1)


@needs_stack
def test_an_export_failure_is_recorded_and_exits_3(tmp_path):
    """--seq 0 makes the export fail on a zero-length input. The template must record the failure and exit 3 without a traceback."""
    done, raw = run_template(tmp_path, QWEN3_TINY, "--seq", "0")
    if done.returncode == 0:
        pytest.skip("this transformers traced a zero-length input; pick another way to force a failure")
    assert done.returncode == 3 and raw["failures"] and raw["failures"][0]["stage"] in ("export", "build")


def test_the_template_never_trusts_remote_code():
    """Source-level guard. A trust_remote_code=True cannot be seen at run time, so look for it in the text."""
    assert "trust_remote_code=True" not in TEMPLATE.read_text()


FIXTURES = REPO / "tests" / "fixtures" / "ops-inventory"


@pytest.mark.parametrize("name", ["qwen3-tiny", "rwkv", "mamba2", "moshi-lm"])
def test_each_recorded_fixture_has_the_schema_keys(name):
    raw = json.loads((FIXTURES / f"raw-{name}.json").read_text())
    assert raw["schema"] == 1 and "traced_class" in raw and "failures" in raw
    assert "ops" in raw or "skipped" in raw
    assert raw["model_dir"] == f"<fixture {name}>"
    for op in raw.get("ops", []):
        assert {"op", "count", "count_in_layers", "sample", "modules"} <= set(op)


# ---- Task 4b: higher-order bodies and python-level nodes -------------------------------------------------------------

@needs_stack
def test_ops_inside_a_higher_order_body_are_counted_and_the_wrapper_is_not_an_op(tmp_path):
    """torch.export puts the ops under `wrap_with_set_grad_enabled` in a sub-GraphModule that `ep.graph.nodes` never
    enters. The template must walk every GraphModule, count the body ops, and leave the wrapper out of `ops`."""
    done, raw = run_template(tmp_path, QWEN3_TINY, "--layers", "2")
    assert done.returncode == 0, done.stderr
    names = {o["op"] for o in raw["ops"]}
    assert not any("wrap_with_set_grad_enabled" in n for n in names), names
    assert raw["higher_order_nodes"].get("wrap_with_set_grad_enabled", 0) >= 1
    assert "bodies" in raw["note"].lower(), "the note must say that bodies were traversed"
    # The body of the qwen3 wrapper holds the whole decoder stack: its matmul-family ops must now be counted.
    # Before the change the top-level graph alone had a handful of ops; the layers live in the body.
    # The rotary embedding (cos, sin) runs inside the wrapper's body in this transformers version, so these two ops
    # exist in the trace only if the body was entered.
    assert {"aten.cos.default", "aten.sin.default"} <= names, sorted(names)


@needs_stack
def test_python_level_nodes_are_not_ops_but_are_recorded(tmp_path):
    """`operator.getitem` is tuple indexing in the exported graph. It is not an operation and must not be in `ops`,
    and it must not vanish: it is counted under `python_nodes`."""
    done, raw = run_template(tmp_path, QWEN3_TINY, "--layers", "2")
    assert done.returncode == 0, done.stderr
    assert not any(o["op"].startswith("<built-in") for o in raw["ops"])
    assert raw["python_nodes"].get("<built-in function getitem>", 0) >= 1
    assert all(o["op"].startswith("aten.") or "." in o["op"] for o in raw["ops"])


def test_aggregate_keeps_hop_wrappers_and_python_nodes_out_of_ops():
    """Unit test on `aggregate` with fake nodes and a real HigherOrderOperator. Needs torch, no export."""
    import importlib.util
    import types
    if PY is None:
        pytest.skip("SKIPPED: no torch in this interpreter")
    code = r'''
import importlib.util, json, operator, sys, types
import torch
spec = importlib.util.spec_from_file_location("t", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def node(target, op="call_function"):
    return types.SimpleNamespace(op=op, target=target, meta={})
hop = torch.ops.higher_order.wrap_with_set_grad_enabled
nodes = [node(torch.ops.aten.add.Tensor), node(operator.getitem), node(operator.getitem), node(hop), node(None, "placeholder")]
ops, py, hops = m.aggregate(nodes)
print(json.dumps({"ops": [o["op"] for o in ops], "py": py, "hops": hops}))
'''
    done = subprocess.run([PY, "-c", code, str(TEMPLATE)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["ops"] == ["aten.add.Tensor"]
    assert got["py"] == {"<built-in function getitem>": 2}
    assert got["hops"] == {"wrap_with_set_grad_enabled": 1}


@needs_stack
@pytest.mark.parametrize("name", ["set_seed", "pipeline", "AutoTokenizer"])
def test_a_config_naming_a_callable_that_is_not_a_model_class_is_skipped(tmp_path, name):
    """I3. config.json comes from the Hub, so its `architectures` entry must not choose which public callable of
    transformers runs. A function (set_seed, pipeline) and a class that is not a model (AutoTokenizer) are skipped."""
    done, raw = run_template(tmp_path, {**QWEN3_TINY, "architectures": [name]})
    assert done.returncode == 0, done.stderr
    assert "not a transformers model class" in raw["skipped"] and name in raw["skipped"] and "ops" not in raw


@needs_stack
def test_a_callable_named_by_the_config_is_never_called(tmp_path):
    """I3. A recorder planted in the transformers namespace stands in for any public callable. It must not run, and the
    class given with --class goes through the same rule."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({**QWEN3_TINY, "architectures": ["OrchardPlantedCallable"]}))
    marker = tmp_path / "called"
    code = r'''
import importlib.util, sys, transformers
marker = sys.argv[2]
def planted(*a, **k):
    open(marker, "w").write("called")
transformers.OrchardPlantedCallable = planted
spec = importlib.util.spec_from_file_location("t", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rc1 = m.main(["--model-dir", sys.argv[3], "--out", sys.argv[4]])
rc2 = m.main(["--model-dir", sys.argv[3], "--out", sys.argv[4], "--class", "OrchardPlantedCallable"])
print(rc1, rc2)
'''
    out = tmp_path / "raw.json"
    done = subprocess.run([PY, "-c", code, str(TEMPLATE), str(marker), str(d), str(out)], capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[-2:] == ["0", "0"]
    assert not marker.exists()
    assert "not a transformers model class" in json.loads(out.read_text())["skipped"]


def _fake_stack(tmp_path):
    """A stand-in torch and transformers on PYTHONPATH. The transformers stand-in raises on any attribute lookup it
    does not define, as the real lazy module does when a backend import fails. Nothing real is imported."""
    root = tmp_path / "fake-stack"
    (root / "torch").mkdir(parents=True)
    (root / "torch" / "__init__.py").write_text("__version__ = '0.0-fake'\n")
    (root / "transformers").mkdir()
    (root / "transformers" / "__init__.py").write_text(
        "__version__ = '0.0-fake'\n"
        "class PreTrainedModel:\n    pass\n"
        "def __getattr__(name):\n"
        "    raise RuntimeError(f'Failed to import transformers.models.{name} because of a backend error')\n")
    return root


@pytest.mark.parametrize("arch,words", [("BrokenForCausalLM", "RuntimeError"), (5, "not a string"),
                                        ({"a": 1}, "not a string")])
def test_a_class_lookup_that_raises_or_a_name_that_is_not_a_string_is_skipped(tmp_path, arch, words):
    """S5. The lookup in transformers can raise (a lazy module whose backend import fails), and architectures[0] can
    be any JSON value. Either ends in the skipped form, exit 0, with a message naming the problem."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({**QWEN3_TINY, "architectures": [arch]}))
    out = tmp_path / "raw.json"
    env = dict(os.environ, PYTHONPATH=str(_fake_stack(tmp_path)))
    done = subprocess.run([sys.executable, str(TEMPLATE), "--model-dir", str(d), "--out", str(out)],
                          capture_output=True, text=True, timeout=60, env=env)
    assert done.returncode == 0, done.stderr
    raw = json.loads(out.read_text())
    assert words in raw["skipped"] and "ops" not in raw, raw["skipped"]


def _run_with_fake_stack(tmp_path, config, init=None):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    root = _fake_stack(tmp_path)
    if init is not None:
        (root / "transformers" / "__init__.py").write_text(init)
    out = tmp_path / "raw.json"
    done = subprocess.run([sys.executable, str(TEMPLATE), "--model-dir", str(d), "--out", str(out)],
                          capture_output=True, text=True, timeout=60, env=dict(os.environ, PYTHONPATH=str(root)))
    return done, (json.loads(out.read_text()) if out.exists() else None)


@pytest.mark.parametrize("archs", [5, {"a": 1}, "Qwen3ForCausalLM"], ids=["int", "dict", "string"])
def test_architectures_that_is_not_a_list_is_skipped(tmp_path, archs):
    """M5. `architectures` must be a list; anything else ends in the skipped form, exit 0, with no traceback."""
    done, raw = _run_with_fake_stack(tmp_path, {**QWEN3_TINY, "architectures": archs})
    assert done.returncode == 0 and "Traceback" not in done.stderr, done.stderr
    assert "architectures" in raw["skipped"] and "not a list" in raw["skipped"]


def test_a_failure_to_read_pretrainedmodel_is_skipped(tmp_path):
    """M5. The class lookup succeeds but reading transformers.PreTrainedModel raises (the lazy module again)."""
    init = ("__version__ = '0.0-fake'\n"
            "class GoodForCausalLM:\n    pass\n"
            "def __getattr__(name):\n"
            "    raise RuntimeError(f'Failed to import {name}')\n")
    done, raw = _run_with_fake_stack(tmp_path, {**QWEN3_TINY, "architectures": ["GoodForCausalLM"]}, init=init)
    assert done.returncode == 0 and "Traceback" not in done.stderr, done.stderr
    assert "RuntimeError" in raw["skipped"] and "PreTrainedModel" in raw["skipped"]
