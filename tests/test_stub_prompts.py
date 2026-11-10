# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The thin prompt path (orchard/stub_prompts.py): one request per stub of a generated port, and the splice of the reply back into the
port. No model runs here; the requests are checked for what they hold and the splice for what it refuses."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import op_cheats, port_template as pt, stub_prompts as sp

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "port-generator"
SHEET = op_cheats.load_sheet(op_cheats.TTNN_SHEET)
EXCERPTS = ("# Reference excerpts\n\n## Qwen4ExpTextMLP\n\nWhy: mlp.\n\n```python\nclass Qwen4ExpTextMLP(nn.Module):\n    def forward(self, x):\n"
            "        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))\n```\n\n## Qwen4ExpTextGatedDeltaNet\n\nWhy: linear attention.\n\n"
            "```python\nclass Qwen4ExpTextGatedDeltaNet(nn.Module):\n    def forward(self, hidden_states):\n        return hidden_states\n```\n\n## Qwen4ExpTextPLELayer\n\nx\n")
CONSTRUCTION_TEXT = ("# Effective construction\n\n## Resolved vs class default\n\n- Qwen4ExpTextRMSNormGated.activation: resolved 'sigmoid', class default 'silu'\n\n"
                     "## Modules\n\nQwen4ExpTextGatedDeltaNet x3 at model.layers.{i}.linear_attn: activation='silu', num_k_heads=2, num_v_heads=6\n"
                     "Qwen4ExpTextMLP x4 at model.layers.{i}.mlp: hidden_size=64\n")

def _generate_without_families(**kw):
    """The Qwen4 plan with no registered block family: the hyper-connection blocks stay stubs, as before issue 696. These tests exercise the stub
    requests for that whole list; the plan with the registry is tested in tests/test_port_template_registry.py."""
    from unittest import mock
    with mock.patch.dict(pt.BLOCK_FAMILIES, clear=True):
        return pt.generate(**kw)



def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny():
    text, plan = _generate_without_families(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"),
                             tensors=_load("tiny-qwen4-tensors.json"), inventory=_load("tiny-qwen4-ops-inventory.json"))
    return text, plan


def test_stub_names_lists_the_marked_functions_in_file_order(tiny):
    text, plan = tiny
    names = sp.stub_names(text)
    assert set(names) == {s.function for s in plan.stubs} and len(names) == len(plan.stubs)
    assert names[0] == "_model_prelude" and names.index("_A_linear_attn") < names.index("_layer_A")      # file order: parts before their layer
    assert sp.stub_names("def (\n") == []


def test_a_stub_request_holds_the_helpers_the_sheet_the_construction_the_excerpt_and_the_stub_last(tiny):
    text, plan = tiny
    stub = next(s for s in plan.stubs if s.function == "_A_linear_attn")
    req = sp.stub_request(text, stub, sheet=SHEET, excerpts=EXCERPTS, construction_text=CONSTRUCTION_TEXT, inventory=_load("tiny-qwen4-ops-inventory.json"))
    user = req["user"]
    assert req["function"] == "_A_linear_attn" and req["args"] == ["model", "i", "x", "ids"] and req["system"] == sp.RULES
    assert "- _linear(model, prefix, x):" in user and "- _to_host(model, t):" in user
    assert "`ttnn.linear(" in user and user.index("`ttnn.linear(") < user.index("## Reference class")     # the sheet before the block
    assert "class Qwen4ExpTextGatedDeltaNet(nn.Module)" in user and "class Qwen4ExpTextMLP" not in user   # this class's excerpt only
    assert "num_v_heads=6" in user and "hidden_size=64" not in user                                       # this class's construction lines only
    assert user.rstrip().endswith("Return a ttnn tensor of the shape the layer expects.")
    assert user.rindex("def _A_linear_attn(") > user.index("## Reference class")                            # the stub comes last
    assert "# AGENT:" in user                                                                               # the stub as it stands, marker included
    assert "Mistakes seen in runs:" in user
    assert len(user) < 20_000


def test_the_sheet_section_names_the_ops_stage_0_mapped_and_says_which_the_sheet_lacks(tiny):
    inventory = _load("tiny-qwen4-ops-inventory.json")
    mapped, other = sp.ops_for(inventory, "model.layers.{i}.linear_attn")
    assert "exp" in mapped and "softplus" in mapped and "matmul" not in mapped              # stage 0 labelled aten.matmul a candidate here
    assert any(o.startswith("aten.clone.default (candidate)") for o in other) and any(o.startswith("aten.matmul.default (candidate)") for o in other)
    section = sp.sheet_section(SHEET, wanted=mapped)
    assert section.startswith("`ttnn.linear(") and "`ttnn.matmul(" in section
    assert "the sheet has not verified them" in section and "softplus" in section.split("the sheet has not verified them")[1]
    assert sp.ops_for(None, "x") == ([], []) and sp.ops_for({"ops": [None, {"op": 1}]}, "x") == ([], [])


def test_a_class_excerpt_is_one_section_and_is_cut_at_the_cap():
    assert sp.class_excerpt(EXCERPTS, "Qwen4ExpTextMLP").startswith("Why: mlp.") and "GatedDeltaNet" not in sp.class_excerpt(EXCERPTS, "Qwen4ExpTextMLP")
    assert sp.class_excerpt(EXCERPTS, "Qwen4ExpTextPLELayer") == "x"
    assert sp.class_excerpt(EXCERPTS, "NoSuchClass") is None and sp.class_excerpt(None, "Qwen4ExpTextMLP") is None
    long = "## Big\n\n" + "y" * 20_000 + "\n"
    cut = sp.class_excerpt(long, "Big", max_chars=100)
    assert len(cut) < 200 and "cut at 100 characters" in cut


def test_a_request_for_a_class_the_excerpts_lack_says_so(tiny):
    text, plan = tiny
    stub = next(s for s in plan.stubs if s.function == "_B_ple")
    req = sp.stub_request(text, stub, sheet=SHEET, excerpts="# nothing\n")
    assert "the excerpts file has no section for this class" in req["user"]


def test_a_repair_request_holds_the_report_the_diff_and_the_function(tiny):
    text, _ = tiny
    req = sp.repair_request(text, "_mlp", last_check="FAILED at prefill: " + "z" * 10_000, layer_diff="layer 2 differs", sheet=SHEET)
    assert req["function"] == "_mlp" and req["args"] == ["model", "prefix", "x"]
    assert len(req["user"]) < 4_000 + 3_000 + 6_000 + 3_000 and "layer 2 differs" in req["user"] and "def _mlp(" in req["user"]
    assert "## Mistakes seen in runs" in req["user"]
    with pytest.raises(sp.SpliceError, match="no top-level function"):
        sp.repair_request(text, "_no_such", last_check="x")


# ---- splice ---------------------------------------------------------------------------------------------------------------------------------

GOOD = ("Here is the function.\n\n```python\nimport torch\n\n\ndef _A_linear_attn(model, i, x, ids):\n    y = _linear(model, f\"model.layers.{i}.linear_attn.in_proj_qkv\", x)\n"
        "    return _linear(model, f\"model.layers.{i}.linear_attn.out_proj\", y)\n```\n")


def test_a_good_reply_replaces_the_stub_and_the_check_sees_one_stub_fewer(tiny):
    text, plan = tiny
    before = pt.check_sources({"port_model.py": text})
    new = sp.splice(text, "_A_linear_attn", GOOD)
    after = pt.check_sources({"port_model.py": new})
    assert len(after["stubs"]) == len(before["stubs"]) - 1 and after["matmul_calls"] >= before["matmul_calls"]
    assert "_A_linear_attn" not in sp.stub_names(new) and 'in_proj_qkv", x)' in sp.function_source(new, "_A_linear_attn")
    assert sp.function_source(new, "_B_ple") == sp.function_source(text, "_B_ple")        # the other functions are untouched
    assert new.count("def _A_linear_attn(") == 1


def test_a_reply_without_a_fence_is_taken_whole(tiny):
    text, _ = tiny
    new = sp.splice(text, "_A_linear_attn", "def _A_linear_attn(model, i, x, ids):\n    return x\n")
    assert sp.function_source(new, "_A_linear_attn").endswith("return x")


def test_a_helper_in_the_reply_is_placed_with_the_function(tiny):
    text, _ = tiny
    reply = "```python\ndef _A_linear_attn_recurrence(q, k, v):\n    return q\n\n\ndef _A_linear_attn(model, i, x, ids):\n    return _A_linear_attn_recurrence(x, x, x)\n```"
    new = sp.splice(text, "_A_linear_attn", reply)
    assert "def _A_linear_attn_recurrence(" in new and pt.check_sources({"port_model.py": new})["parse_errors"] == []


@pytest.mark.parametrize("reply,why", [
    ("no code here", "does not parse"),
    ("```python\ndef other(model, i, x, ids):\n    return x\n```", "does not define _A_linear_attn"),
    ("```python\ndef _A_linear_attn(model, x):\n    return x\n```", "exactly these arguments: model, i, x, ids"),
    ("```python\ndef _A_linear_attn(model, i, x, ids):\n    # AGENT: still a stub\n    return x\n```", "keeps the stub marker"),
    ("```python\nprint('hi')\ndef _A_linear_attn(model, i, x, ids):\n    return x\n```", "top-level Expr"),
    ("```python\nN = 3\ndef _A_linear_attn(model, i, x, ids):\n    return x\n```", "top-level Assign"),
    ("```python\ndef _A_linear_attn(model, i, x, ids):\n    return (\n```", "does not parse"),
    ("```python\ndef _linear(model, prefix, x):\n    return x\ndef _A_linear_attn(model, i, x, ids):\n    return x\n```", "redefines _linear"),
    ("```python\nimport subprocess\ndef _A_linear_attn(model, i, x, ids):\n    return x\n```", "may not use"),
    ("```python\nfrom transformers import AutoModel\ndef _A_linear_attn(model, i, x, ids):\n    return x\n```", "may not use"),
    ("```python\ndef _A_linear_attn(model, i, x, ids):\n    return x\n```\n```python\ndef _A_linear_attn(model, i, x, ids):\n    return x\n```", "2 code blocks"),
    (None, "not a string"),
    ("x" * 200_001, "not a string of bounded size"),
])
def test_a_bad_reply_is_refused_with_a_reason(tiny, reply, why):
    text, _ = tiny
    with pytest.raises(sp.SpliceError, match=why):
        sp.splice(text, "_A_linear_attn", reply)
    assert sp.function_source(text, "_A_linear_attn") in text          # the port is unchanged


# The splice is a guard on what a coder reply may contain, not a sandbox: the coder can already write any file in the port directory.
# These replies are refused because a port function has no use for them (review of PR 685).
def _fn(body: str, *, head: str = "") -> str:
    return "```python\n" + head + "def _A_linear_attn(model, i, x, ids):\n" + "\n".join("    " + l for l in body.split("\n")) + "\n```"


@pytest.mark.parametrize("reply,why", [
    (_fn("exec('x = 1')\nreturn x"), "may not call exec"),
    (_fn("return eval('x')"), "may not call eval"),
    (_fn("return compile('x', 'f', 'eval')"), "may not call compile"),
    (_fn("return open('/etc/passwd').read()"), "may not call open"),
    (_fn("return __import__('os').getcwd()"), "dunder"),
    (_fn("return x.__class__.__subclasses__()"), "dunder"),
    (_fn("return getattr(x, '__class__')"), "dunder"),
    (_fn("return getattr(x, name)"), "getattr"),
    (_fn("import importlib\nreturn x"), "may not use"),
    (_fn("import ctypes\nreturn x"), "may not use"),
    (_fn("import pickle\nreturn pickle.loads(x)"), "may not use"),
    (_fn("from pathlib import Path\nreturn Path('.')"), "may not use"),
    (_fn("import shutil\nreturn x"), "may not use"),
    (_fn("import builtins\nreturn x"), "may not use"),
    (_fn("import torch\nreturn torch.load('f')"), "torch.load"),
    ("```python\ndef _helper(a, y=open('/etc/passwd')):\n    return a\ndef _A_linear_attn(model, i, x, ids):\n    return _helper(x)\n```", "may not call open"),
    ("```python\ndef deco(f):\n    return f\n@deco\ndef _A_linear_attn(model, i, x, ids):\n    return x\n```", "decorator"),
    ("```python\ndef _A_linear_attn(model, i, x, ids):\n    raise NotImplementedError('_A_linear_attn: Qwen4ExpTextGatedDeltaNet at model.layers.0.linear_attn is not written yet')\n```",
     "generated stub text"),
])
def test_a_reply_with_a_call_or_import_a_port_function_has_no_use_for_is_refused(tiny, reply, why):
    text, _ = tiny
    with pytest.raises(sp.SpliceError, match=why):
        sp.splice(text, "_A_linear_attn", reply)
    assert sp.function_source(text, "_A_linear_attn") in text


@pytest.mark.parametrize("imp,why", [
    ("from .helpers import _linear", "relative import"),
    ("from . import helpers", "relative import"),
    ("from port_helpers import _linear", "cannot import"),
    ("import helpers", "cannot import"),
    ("from my_utils import rms", "cannot import"),
    ("import scipy", "cannot import"),
])
def test_an_import_of_a_module_the_port_does_not_have_is_refused_with_the_reason(tiny, imp, why):
    # Arm E3: the stubs imported modules that do not exist (`from .helpers import ...`, `from port_helpers import ...`), and the port failed at init
    # on every board check. The port is one file, so an import is refused unless the module is one a port uses.
    text, _ = tiny
    with pytest.raises(sp.SpliceError, match=why):
        sp.splice(text, "_A_linear_attn", _fn("return x", head=imp + "\n"))


@pytest.mark.parametrize("imp", ["import ttnn", "import torch", "import math", "import numpy as np", "from functools import lru_cache",
                                 "from typing import Any", "import re", "from collections import defaultdict", "import itertools"])
def test_the_imports_a_port_uses_are_accepted(tiny, imp):
    text, _ = tiny
    new = sp.splice(text, "_A_linear_attn", _fn("return x", head=imp + "\n"))
    assert imp in new


def test_a_reply_that_uses_ordinary_port_idioms_is_still_accepted(tiny):
    text, _ = tiny
    reply = _fn("import torch\nimport ttnn\nscale = getattr(model['cfg'], 'scale', 1.0)\nh = ttnn.linear(x, model['dev']['w'])\n"
                "return torch.nn.functional.silu(h) * scale")
    assert "ttnn.linear" in sp.function_source(sp.splice(text, "_A_linear_attn", reply), "_A_linear_attn")


def test_splice_refuses_a_function_the_port_lacks_and_a_port_that_does_not_parse(tiny):
    text, _ = tiny
    with pytest.raises(sp.SpliceError, match="no top-level function"):
        sp.splice(text, "_nope", GOOD)
    with pytest.raises(sp.SpliceError, match="does not parse"):
        sp.function_source("def (\n", "_x")


def test_every_stub_of_the_tiny_port_can_be_requested_and_spliced_in_turn(tiny):
    """The loop the supervisor would run: one request per stub, a trivial reply spliced in, until the CPU check sees no stub."""
    text, plan = tiny
    for name in sp.stub_names(text):
        req = sp.stub_request(text, next(s for s in plan.stubs if s.function == name), sheet=SHEET, excerpts=EXCERPTS)
        args = ", ".join(req["args"])
        text = sp.splice(text, name, f"```python\ndef {name}({args}):\n    return x\n```")
    result = pt.check_sources({"port_model.py": text})
    assert result["ok"] and result["stubs"] == []
