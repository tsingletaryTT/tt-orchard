# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The dense-layer shim for qwen4_exp configs (orchard/skills/reference-gate-templates/qwen4_dense.py). PROVISIONAL: the
configs here are synthetic, built from public readings of an unreleased model. The rule tests need no torch. The build tests
run in the reference interpreter (set ORCHARD_REFERENCE_PYTHON; transformers 5.19.0 has qwen4_exp) and skip without it."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_ops_inventory_stub import _has_qwen4, needs_stack, TINY_QWEN4, op_names
from test_ops_inventory_template import PY, run_template

REPO = Path(__file__).resolve().parent.parent
SHIM = REPO / "orchard" / "skills" / "reference-gate-templates" / "qwen4_dense.py"
spec = importlib.util.spec_from_file_location("qwen4_dense", SHIM)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)

# a tiny dense config: no num_experts key at all (the transformers default of 512 must not be used), a plain MLP of 96
DENSE_TEXT = {k: v for k, v in TINY_QWEN4.items()
              if k not in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size")}
DENSE_TEXT["intermediate_size"] = 96
TINY_DENSE = {**DENSE_TEXT}                                              # model_type qwen4_exp_text, flat
FIXTURE = REPO / "tests" / "fixtures" / "model-configs" / "qwen4-dense-provisional.json"


# ---- the layer rule (no torch) ----------------------------------------------------------------------

def moe(text, n=4):
    return [shim.layer_is_moe(text, i) for i in range(n)]


def test_every_layer_is_moe_by_default_when_experts_exist():
    assert moe({"num_experts": 8}) == [True] * 4


def test_no_experts_key_or_zero_experts_means_every_layer_is_dense():
    assert moe({}) == [False] * 4
    assert moe({"num_experts": 0}) == [False] * 4
    assert moe({"num_experts": None}) == [False] * 4


def test_mlp_only_layers_and_the_sparse_step_mark_the_dense_layers():
    assert moe({"num_experts": 8, "mlp_only_layers": [1]}) == [True, False, True, True]
    assert moe({"num_experts": 8, "decoder_sparse_step": 2}) == [False, True, False, True]
    assert moe({"num_experts": 8, "decoder_sparse_step": 2, "mlp_only_layers": [3]}) == [False, True, False, False]


def test_needs_the_shim_only_for_a_qwen4_config_with_a_dense_layer():
    flash = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 4, "num_experts": 512}}
    assert not shim.needs_dense_shim(flash)
    assert shim.needs_dense_shim({"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 4}})
    assert shim.needs_dense_shim({"model_type": "qwen4_exp_text", "num_hidden_layers": 4, "num_experts": 8, "mlp_only_layers": [0]})
    assert not shim.needs_dense_shim({"model_type": "qwen3", "num_hidden_layers": 4})        # another family: never touched


# ---- the build (reference interpreter) --------------------------------------------------------------

BUILD = r'''
import importlib.util, json, sys, torch, transformers
spec = importlib.util.spec_from_file_location("qwen4_dense", sys.argv[1])
shim = importlib.util.module_from_spec(spec); spec.loader.exec_module(shim)
raw = json.loads(sys.argv[2]); mode = sys.argv[3]
if shim.needs_dense_shim(raw):
    config = shim.prepare_config(transformers, raw)
    ctx = shim.dense_layers()
else:
    import contextlib
    config = transformers.AutoConfig.for_model(**raw)
    ctx = contextlib.nullcontext()
torch.manual_seed(0)
with ctx:
    model = transformers.Qwen4ExpForCausalLM(config).eval()
layers = model.model.layers
out = {"mlp": [type(l.mlp).__name__ for l in layers],
       "experts": [hasattr(l.mlp, "experts") for l in layers],
       "widths": [getattr(l.mlp, "intermediate_size", None) for l in layers]}
if mode == "forward":
    ids = torch.randint(0, 100, (1, 12))
    logits = model(input_ids=ids, use_cache=False).logits
    out["finite"] = bool(torch.isfinite(logits).all()); out["shape"] = list(logits.shape)
# the class is restored on exit
from transformers.models.qwen4_exp import modeling_qwen4_exp as m
out["restored"] = m.Qwen4ExpTextDecoderLayer.__init__.__qualname__
print(json.dumps(out))
'''


def build(raw, mode="forward"):
    done = subprocess.run([PY, "-I", "-c", BUILD, str(SHIM), json.dumps(raw), mode], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@needs_stack
def test_a_dense_config_builds_and_gives_finite_logits():
    out = build(TINY_DENSE)
    assert out["finite"] and out["shape"] == [1, 12, 128]
    assert out["mlp"] == ["Qwen4ExpTextMLP"] * 4 and out["widths"] == [96] * 4


@needs_stack
def test_a_config_that_writes_num_experts_0_builds_through_the_shim():
    out = build({**TINY_DENSE, "num_experts": 0, "num_experts_per_tok": 0})
    assert out["finite"] and out["mlp"] == ["Qwen4ExpTextMLP"] * 4


@needs_stack
def test_dense_layers_hold_no_experts():
    assert build(TINY_DENSE, "build")["experts"] == [False] * 4


@needs_stack
def test_a_mixed_schedule_marks_the_right_layers():
    raw = {**TINY_QWEN4, "intermediate_size": 96, "mlp_only_layers": [0], "decoder_sparse_step": 2}
    out = build(raw)
    # layer 0 is listed; layer 1 is MoE ((1+1)%2==0); layer 2 is off the stride; layer 3 is MoE
    assert out["mlp"] == ["Qwen4ExpTextMLP", "Qwen4ExpTextSparseMoeBlock", "Qwen4ExpTextMLP", "Qwen4ExpTextSparseMoeBlock"]
    assert out["experts"] == [False, True, False, True] and out["finite"]


@needs_stack
def test_the_moe_path_is_unchanged_for_a_flash_next_style_config():
    out = build(TINY_QWEN4)
    assert out["mlp"] == ["Qwen4ExpTextSparseMoeBlock"] * 4 and out["experts"] == [True] * 4 and out["finite"]


@needs_stack
def test_the_decoder_layer_class_is_restored_after_the_context():
    assert build(TINY_DENSE, "build")["restored"] == "Qwen4ExpTextDecoderLayer.__init__"


@needs_stack
def test_a_dense_config_without_the_shim_is_refused_by_transformers():
    """Why the shim exists: the installed config validation rejects num_experts 0."""
    done = subprocess.run([PY, "-I", "-c", "import sys, json, transformers; transformers.AutoConfig.for_model(**json.loads(sys.argv[1]))",
                           json.dumps({**TINY_DENSE, "num_experts": 0})], capture_output=True, text=True)
    assert done.returncode != 0 and "num_experts" in done.stderr


# ---- stage 1: the reference gate's load path --------------------------------------------------------

GATE = REPO / "orchard" / "skills" / "reference-gate-templates" / "reference_gate.py"
LOAD = r'''
import importlib.util, json, shutil, sys, tempfile, torch, transformers
from pathlib import Path
from safetensors.torch import save_file
shim_spec = importlib.util.spec_from_file_location("qwen4_dense", sys.argv[1])
shim = importlib.util.module_from_spec(shim_spec); shim_spec.loader.exec_module(shim)
raw = json.loads(sys.argv[2]); work = Path(tempfile.mkdtemp())
snap = work / "snap"; snap.mkdir()
# a dense checkpoint: weights saved from the shim-built model, a config.json with no experts at all
with shim.dense_layers():
    model = transformers.Qwen4ExpForCausalLM(shim.prepare_config(transformers, raw))
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(snap / "model.safetensors"), metadata={"format": "pt"})
(snap / "config.json").write_text(json.dumps({**raw, "architectures": ["Qwen4ExpForCausalLM"]}))
gate_dir = work / "gate"; gate_dir.mkdir()
shutil.copy(sys.argv[3], gate_dir / "reference_gate.py")
if sys.argv[4] == "with-shim":
    shutil.copy(sys.argv[1], gate_dir / "qwen4_dense.py")
spec = importlib.util.spec_from_file_location("gate", gate_dir / "reference_gate.py")
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
try:
    _, rep = gate.load_model(torch, transformers, snap)
    print(json.dumps({"pass": rep["pass"], "missing": rep["missing_keys"], "unexpected": rep["unexpected_keys"],
                      "cls": rep["class"], "dense": rep.get("dense_qwen4")}))
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
'''


def load_through_gate(raw, shim_present=True):
    done = subprocess.run([PY, "-I", "-c", LOAD, str(SHIM), json.dumps(raw), str(GATE), "with-shim" if shim_present else "no-shim"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@needs_stack
def test_the_reference_gate_loads_a_dense_checkpoint_with_no_missing_or_unexpected_keys():
    out = load_through_gate(TINY_DENSE)
    assert out["pass"] is True and out["missing"] == [] and out["unexpected"] == [], out
    assert out["cls"] == "Qwen4ExpForCausalLM" and out["dense"]["layers"] == [0, 1, 2, 3] and "provisional" in out["dense"]["note"]


@needs_stack
def test_the_reference_gate_stops_with_a_clear_message_when_the_shim_file_is_missing():
    out = load_through_gate(TINY_DENSE, shim_present=False)
    assert "qwen4_dense.py is not next to this script" in out["error"]


@needs_stack
def test_the_reference_gate_does_not_use_the_shim_for_a_moe_qwen4_config():
    out = load_through_gate(TINY_QWEN4)
    assert out["pass"] is True and out["dense"] is None, out


# ---- the op inventory on a dense config -------------------------------------------------------------

@needs_stack
def test_the_inventory_traces_a_dense_config_and_has_no_expert_ops(tmp_path):
    done, raw = run_template(tmp_path, TINY_DENSE, "--layers", "4", "--seq", "16", "--stub-data-dependent")
    assert done.returncode == 0 and raw["failures"] == [], (done.stderr, raw["failures"])
    assert raw["dense_layers"] == [0, 1, 2, 3] and "provisional" in raw["dense_note"]
    names = " ".join(str(o) for o in raw["ops"])
    assert "grouped_mm" not in names and "topk" in names          # the one topk is the QSA indexer's, not a router's
    assert not any("grouped_mm" in k for k in raw["python_nodes"])


@needs_stack
def test_the_inventory_of_a_moe_config_has_no_dense_note(tmp_path):
    done, raw = run_template(tmp_path, TINY_QWEN4, "--layers", "2", "--seq", "16")
    assert done.returncode == 0 and "dense_layers" not in raw


# ---- the synthetic fixture --------------------------------------------------------------------------

def test_the_fixture_is_labeled_provisional_and_is_dense():
    raw = json.loads(FIXTURE.read_text())
    assert "provisional" in raw["_source"].lower() and "no qwen4 weights" in raw["_source"].lower()
    t = raw["text_config"]
    assert "num_experts" not in t and t["hc_lowrank"] == 640 and t["hc_count"] == 4 and t["hidden_size"] == 5120
