# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 520 and 550: layer-diff and the effective-construction report load a dense Qwen4 checkpoint the way stage 1 does.

Stage 1 prepares a dense `qwen4_exp` config with the shim (reference-gate-templates/qwen4_dense.py): placeholder expert settings so
transformers accepts the config, and a context in which the dense layers get the model's own dense MLP. Layer-diff called
AutoModelForCausalLM.from_pretrained directly and the effective-construction report built the model from the raw config, so on a dense
checkpoint the first failed on the config and the second reported MoE blocks the reference replaces. The tests run in the reference
interpreter (ORCHARD_REFERENCE_PYTHON, transformers 5.19.0 with qwen4_exp) and skip without it. They copy stage_tools.py into
`<run>/stages/2/` and the shim into `<run>/stages/1/`, which is where a real run has them."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from test_ops_inventory_stub import needs_real_forward, needs_stack
from test_ops_inventory_template import PY
from test_qwen4_dense import SHIM, TINY_DENSE, TINY_QWEN4

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "full-port-templates"
CONSTRUCTION = TEMPLATES / "effective_construction.py"
TOOLS = TEMPLATES / "stage_tools.py"

MAKE = r'''
import importlib.util, json, sys, torch, transformers
from pathlib import Path
from safetensors.torch import save_file
spec = importlib.util.spec_from_file_location("qwen4_dense", sys.argv[1])
shim = importlib.util.module_from_spec(spec); spec.loader.exec_module(shim)
raw = json.loads(sys.argv[2]); snap = Path(sys.argv[3]); snap.mkdir(parents=True)
torch.manual_seed(0)
if shim.needs_dense_shim(raw):
    with shim.dense_layers():
        model = transformers.Qwen4ExpForCausalLM(shim.prepare_config(transformers, raw))
else:
    model = transformers.Qwen4ExpForCausalLM(transformers.AutoConfig.for_model(**raw))
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(snap / "model.safetensors"), metadata={"format": "pt"})
(snap / "config.json").write_text(json.dumps({**raw, "architectures": ["Qwen4ExpForCausalLM"]}))
'''

CONSTRUCT = r'''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("effective_construction", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
model = m.build_model(sys.argv[2])
print(json.dumps({"mlp": [type(l.mlp).__name__ for l in model.model.layers], "device": str(next(model.parameters()).device)}))
'''

LAYERS = r'''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("stage_tools", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
arrays = m.reference_layers(sys.argv[2], [1, 2, 3, 4, 5, 6], "float32")
print(json.dumps({"layers": len(arrays), "shape": list(arrays[0].shape)}))
'''

LOAD = r'''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("stage_tools", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
model = m.load_reference_model(sys.argv[2], "float32")
print(json.dumps({"cls": type(model).__name__, "mlp": [type(l.mlp).__name__ for l in model.model.layers]}))
'''


def make_snapshot(tmp_path, raw):
    snap = tmp_path / "snap"
    done = subprocess.run([PY, "-I", "-c", MAKE, str(SHIM), json.dumps(raw), str(snap)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return snap


def run_dir_with_tools(tmp_path, *, shim_in_stage_1=True):
    """<run>/stages/2/stage_tools.py, and <run>/stages/1/qwen4_dense.py as stage 1 leaves it for a Qwen4 model."""
    stage2 = tmp_path / "run" / "stages" / "2"
    stage2.mkdir(parents=True)
    shutil.copy(TOOLS, stage2 / "stage_tools.py")
    if shim_in_stage_1:
        (tmp_path / "run" / "stages" / "1").mkdir(parents=True)
        shutil.copy(SHIM, tmp_path / "run" / "stages" / "1" / "qwen4_dense.py")
    return stage2


def run(script, *args):
    return subprocess.run([PY, "-I", "-c", script, *map(str, args)], capture_output=True, text=True)


@needs_stack
def test_the_construction_report_builds_a_dense_checkpoint_with_dense_mlps_and_no_expert_blocks(tmp_path):
    snap = make_snapshot(tmp_path, TINY_DENSE)
    done = run(CONSTRUCT, CONSTRUCTION, snap)
    assert done.returncode == 0, done.stderr[-1500:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["mlp"] == ["Qwen4ExpTextMLP"] * 4 and out["device"] == "meta"


@needs_stack
def test_the_construction_report_leaves_a_moe_qwen4_config_as_it_was(tmp_path):
    snap = make_snapshot(tmp_path, TINY_QWEN4)
    done = run(CONSTRUCT, CONSTRUCTION, snap)
    assert done.returncode == 0, done.stderr[-1500:]
    assert json.loads(done.stdout.strip().splitlines()[-1])["mlp"] == ["Qwen4ExpTextSparseMoeBlock"] * 4


@needs_stack
def test_layer_diff_loads_a_dense_checkpoint_the_way_stage_1_does(tmp_path):
    snap = make_snapshot(tmp_path, TINY_DENSE)
    stage2 = run_dir_with_tools(tmp_path)
    done = run(LOAD, stage2 / "stage_tools.py", snap)
    assert done.returncode == 0, done.stderr[-1500:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["cls"] == "Qwen4ExpForCausalLM" and out["mlp"] == ["Qwen4ExpTextMLP"] * 4


@needs_stack
def test_layer_diff_finds_the_shim_in_the_repo_when_stage_1_left_none(tmp_path):
    """A resumed or hand-made run may have no stages/1/qwen4_dense.py; the shipped copy next to the reference gate template is used. The
    tools file is copied into a place whose neighbour directory is the repo's templates, as in the repo tree."""
    snap = make_snapshot(tmp_path, TINY_DENSE)
    done = run(LOAD, TOOLS, snap)                                  # HERE.parent is orchard/skills, which holds reference-gate-templates
    assert done.returncode == 0, done.stderr[-1500:]
    assert json.loads(done.stdout.strip().splitlines()[-1])["mlp"] == ["Qwen4ExpTextMLP"] * 4


@needs_stack
def test_layer_diff_loads_a_moe_qwen4_checkpoint_without_the_shim(tmp_path):
    snap = make_snapshot(tmp_path, TINY_QWEN4)
    stage2 = run_dir_with_tools(tmp_path)
    done = run(LOAD, stage2 / "stage_tools.py", snap)
    assert done.returncode == 0, done.stderr[-1500:]
    assert json.loads(done.stdout.strip().splitlines()[-1])["mlp"] == ["Qwen4ExpTextSparseMoeBlock"] * 4


@needs_real_forward
def test_layer_diff_computes_reference_layers_for_a_dense_checkpoint_end_to_end(tmp_path):
    """Needs a torch that accepts the Qwen4 indexer's int32 scatter; skipped where torch refuses it."""
    snap = make_snapshot(tmp_path, TINY_DENSE)
    stage2 = run_dir_with_tools(tmp_path)
    done = run(LAYERS, stage2 / "stage_tools.py", snap)
    assert done.returncode == 0, done.stderr[-1500:]
    assert json.loads(done.stdout.strip().splitlines()[-1])["layers"] == 4
