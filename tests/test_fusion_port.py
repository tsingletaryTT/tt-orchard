# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/fusion_port.py: the candidate list for porting a fused decode bundle to a model of the same family (issue 184).

The fake bundle below holds the limits of the real fused-op modules (asserts, constants, the env gates) in a few lines each.
Expected statuses are worked out by hand from those limits."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from orchard import fusion_port as fp
from orchard import model_config as mc

FIX = Path(__file__).parent / "fixtures" / "model-configs"

FILES = {
    "tt/moe_decode/topk.py": '''
class RouterTopK:
    def __init__(self, mesh_device, num_experts: int, k: int, core=(0, 0)):
        assert num_experts % 32 == 0 and k <= 16, (num_experts, k)
''',
    "tt/moe_decode/fused.py": '''
from .experts import TILE
def supported(num_experts, hidden, inter_local, dtype, banks=8):
    m = (2 * inter_local // TILE) // banks
    return (
        dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)
        and (2 * inter_local // TILE) % banks == 0
        and m * (2 if m % 2 else 1) + 1 <= 8
        and (hidden // TILE) % banks == 0
        and num_experts % TILE == 0
        and num_experts <= 512  # router_topk.hpp route_union's expert mask
    )
''',
    "tt/gdn_decode/ops.py": '''
TILE = 32
class GdnDecode:
    def __init__(self, mesh_device, num_k_heads, num_v_heads, head_k_dim, head_v_dim, proj_dim, scale, max_batch=32):
        assert head_k_dim % TILE == 0 and head_v_dim == 128 and num_v_heads % num_k_heads == 0 and num_v_heads <= 128
        assert max_batch <= TILE, "one tile row of users"
''',
    "tt/attn_decode/ops.py": '''
TILE = 32
class AttnDecodePrep:
    def __init__(self, mesh_device, num_heads, num_kv_heads, head_dim, rope_dim, max_batch=32, eps=1e-6):
        assert head_dim % TILE == 0 and rope_dim == 2 * TILE, "rotate-half over the first two tiles of a head"
        assert num_heads <= 16 and num_kv_heads <= 16 and max_batch <= TILE
''',
    "tt/decode_sampling/ops.py": '''
GRID = (10, 10)
MAX_ROWS = 8
MAX_WIDTH = 1 << 20
class RowsTopK:
    def __init__(self, mesh_device, width, k=32):
        assert width % 32 == 0 and k == 32
        assert width <= MAX_WIDTH, width
''',
    "tt/fabric_boundary/ops.py": '''
TILE = 32
BOUNDARY_CORE = (9, 2)
ROWS_CORE_ROWS = (2, 5)
MAX_ROWS = 8
class FabricBoundary:
    def __init__(self, mesh_device, hidden, eps, ring_topology=None, max_rows=MAX_ROWS):
        assert hidden % 1024 == 0, "whole flat pages"
        self.ntiles = hidden // TILE
        assert self.ntiles % 16 == 0
''',
    "tt/addnorm/ops.py": '''
TILE = 32
class AddNorm:
    def __init__(self, mesh_device, hidden, cores=(8, 2), eps=1e-6):
        self.NC = 16
        assert hidden % (self.NC * TILE) == 0
''',
    "tt/moe.py": '''
import os
ROUTER_TOPK_ROUTE_K = int(os.environ.get("QWEN35_ROUTER_TOPK_ROUTE_K", "80") or 0)
ROUTER_TOPK_FUSED = os.environ.get("QWEN35_ROUTER_TOPK_FUSED", "1") == "1"
MOE_FUSED_DECODE = os.environ.get("QWEN35_MOE_FUSED_DECODE", "0") == "1"
''',
    "tt/model_config.py": '''
import os
fb = os.environ.get("QWEN35_FABRIC_BOUNDARY", "0") == "1"
''',
    "tt/attention/tp.py": '''
import os
a = os.environ.get("QWEN35_ATTN_DECODE_OPS", "0") == "1"
''',
    "tt/gdn/tp.py": '''
import os
g = os.environ.get("QWEN35_GDN_FUSED_DECODE", "0") == "1"
''',
    "tt/model.py": '''
import os
r = os.environ.get("QWEN35_ROWS_TOPK", "0") == "1"
''',
}

ALL_OPS = ("router_topk", "moe_decode", "gdn_decode", "attn_decode", "sampler_rows_topk", "fabric_boundary", "addnorm")


def make_bundle(root: Path, files=None, drop=()):
    for rel, text in (FILES if files is None else files).items():
        if rel in drop:
            continue
        p = root / "qwen36" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def cfg(name):
    return mc.read_model_config(FIX / name)


@pytest.fixture
def bundle(tmp_path):
    return make_bundle(tmp_path / "bundle")


def by_name(report):
    return {o["name"]: o for o in report["ops"]}


def op_of(inv, name):
    return next(o for o in inv["ops"] if o["name"] == name)


# ---- reading the inventory -------------------------------------------------------------------------------------------

def test_inventory_finds_each_fused_op_and_its_gate_default(bundle):
    inv = fp.read_bundle_inventory(bundle)
    assert set(ALL_OPS) <= {o["name"] for o in inv["ops"] if o["found"]}
    router = op_of(inv, "router_topk")
    assert router["family"] == "router"
    assert router["gate"] == {"env": "QWEN35_ROUTER_TOPK_FUSED", "code_default": "1"}
    assert [c["text"] for c in router["constraints"]] == ["num_experts % 32 == 0", "k <= 16"]
    assert op_of(inv, "moe_decode")["gate"]["code_default"] == "0"
    # the comment on the last return term is not part of the constraint text
    assert "num_experts <= 512" in [c["text"] for c in op_of(inv, "moe_decode")["constraints"]]


def test_inventory_reads_module_constants_and_tunables(bundle):
    inv = fp.read_bundle_inventory(bundle)
    samp = op_of(inv, "sampler_rows_topk")
    assert samp["constants"]["MAX_WIDTH"] == 1 << 20 and samp["constants"]["GRID"] == [10, 10]
    assert {"name": "QWEN35_ROUTER_TOPK_ROUTE_K", "value": "80"} in [
        {"name": t["name"], "value": t["value"]} for t in op_of(inv, "router_topk")["tunables"]]


def test_inventory_never_runs_the_bundle_code(tmp_path):
    marker = tmp_path / "ran"
    files = dict(FILES)
    files["tt/moe_decode/topk.py"] = f'open({str(marker)!r}, "w").write("x")\n' + FILES["tt/moe_decode/topk.py"]
    fp.read_bundle_inventory(make_bundle(tmp_path / "b", files))
    assert not marker.exists()


def test_a_missing_module_is_reported_not_raised(tmp_path):
    inv = fp.read_bundle_inventory(make_bundle(tmp_path / "b", drop=("tt/addnorm/ops.py",)))
    op = op_of(inv, "addnorm")
    assert op["found"] is False and "addnorm/ops.py" in op["note"]


def test_a_module_that_does_not_parse_is_reported(tmp_path):
    files = dict(FILES)
    files["tt/addnorm/ops.py"] = "def broken(:\n"
    op = op_of(fp.read_bundle_inventory(make_bundle(tmp_path / "b", files)), "addnorm")
    assert op["found"] is False and "parse" in op["note"]


def test_a_file_over_the_size_limit_is_refused_and_named(tmp_path):
    files = dict(FILES)
    files["tt/addnorm/ops.py"] = "x = 1\n" * (fp.MAX_SOURCE_BYTES // 6 + 10)
    op = op_of(fp.read_bundle_inventory(make_bundle(tmp_path / "b", files)), "addnorm")
    assert op["found"] is False and "larger than" in op["note"]


def test_two_files_that_match_one_op_are_ambiguous(tmp_path):
    b = make_bundle(tmp_path / "b")
    other = tmp_path / "b" / "copy" / "tt" / "moe_decode"
    other.mkdir(parents=True)
    (other / "topk.py").write_text(FILES["tt/moe_decode/topk.py"])
    op = op_of(fp.read_bundle_inventory(b), "router_topk")
    assert op["found"] is False and "more than one" in op["note"]


def test_a_symlinked_module_is_not_followed(tmp_path):
    b = make_bundle(tmp_path / "b", drop=("tt/addnorm/ops.py",))
    real = tmp_path / "real.py"
    real.write_text(FILES["tt/addnorm/ops.py"])
    d = tmp_path / "b" / "qwen36" / "tt" / "addnorm"
    d.mkdir(parents=True)
    (d / "ops.py").symlink_to(real)
    assert op_of(fp.read_bundle_inventory(b), "addnorm")["found"] is False


def test_bundle_dir_that_is_not_a_directory_raises_fusion_error(tmp_path):
    with pytest.raises(fp.FusionPortError):
        fp.read_bundle_inventory(tmp_path / "nope")


# ---- the constraint evaluator ----------------------------------------------------------------------------------------

def test_evaluator_handles_arithmetic_and_refuses_calls():
    ev = fp.eval_constraint
    assert ev("a % 32 == 0", {"a": 64}) is True
    assert ev("a <= 16", {"a": 17}) is False
    assert ev("(2 * a // 32) % 8 == 0", {"a": 256}) is True
    assert ev("a <= b", {"a": 1}) is None                # b unbound
    assert ev("__import__('os').getcwd() == ''", {}) is None   # a call is never made
    assert ev("a.b == 1", {"a": 1}) is None


def test_split_conjuncts_keeps_each_term():
    assert fp.split_conjuncts("a % 32 == 0 and k <= 16") == ["a % 32 == 0", "k <= 16"]


# ---- classification ---------------------------------------------------------------------------------------------------

def test_the_80b_sibling_has_the_same_dimensions_and_every_op_applies_as_is(bundle):
    rep = fp.candidates(cfg("qwen3-next-80b-a3b-instruct.json"), bundle, bundle_config=cfg("qwen3-coder-next.json"), dies=4)
    ops = by_name(rep)
    for name in ALL_OPS:
        assert ops[name]["status"] == fp.AS_IS, (name, ops[name])
    assert rep["counts"][fp.CHANGED] == 0 and rep["counts"][fp.NEW_CODE] == 0


def test_35b_model_changes_the_router_and_sampler_constants(bundle):
    rep = fp.candidates(cfg("qwen3.5-35b-a3b.json"), bundle, bundle_config=cfg("qwen3-coder-next.json"), dies=2)
    ops = by_name(rep)
    # 256 experts against 512 and top-k 8 against 10: the route-k constant was tuned for the bundle's expert count
    assert ops["router_topk"]["status"] == fp.CHANGED
    assert "QWEN35_ROUTER_TOPK_ROUTE_K" in " ".join(ops["router_topk"]["reasons"])
    # the vocabulary per die differs and the scan grid was sized for the bundle's
    assert ops["sampler_rows_topk"]["status"] == fp.CHANGED
    assert "GRID" in " ".join(ops["sampler_rows_topk"]["reasons"])
    # every constraint of these ops still passes
    assert ops["moe_decode"]["status"] == fp.AS_IS
    assert ops["gdn_decode"]["status"] == fp.AS_IS


def test_a_failing_kernel_constraint_is_new_code(bundle):
    c = dict(cfg("qwen3-coder-next.json"), num_experts=1024)          # past the 512-wide expert mask
    rep = fp.candidates(c, bundle, bundle_config=cfg("qwen3-coder-next.json"), dies=2)
    op = by_name(rep)["moe_decode"]
    assert op["status"] == fp.NEW_CODE
    failing = [x for x in op["constraints"] if x["value"] is False]
    assert [x["text"] for x in failing] == ["num_experts <= 512"] and failing[0]["kernel_side"] is True
    assert any(g["op"] == "moe_decode" for g in rep["gaps"])


def test_fabric_boundary_is_new_code_on_two_dies(bundle):
    rep = fp.candidates(cfg("qwen3-coder-next.json"), bundle, bundle_config=cfg("qwen3-coder-next.json"), dies=2)
    op = by_name(rep)["fabric_boundary"]
    assert op["status"] == fp.NEW_CODE
    assert any("four dies" in x["source"] for x in op["constraints"] if x["value"] is False)


def test_model_without_linear_attention_does_not_use_the_gdn_op(bundle):
    rep = fp.candidates(cfg("qwen3-coder-30b-a3b.json"), bundle, bundle_config=cfg("qwen3-coder-next.json"), dies=2)
    ops = by_name(rep)
    assert ops["gdn_decode"]["status"] == fp.NOT_USED
    # full rotary on a 128-wide head: the attention prep's rotate-half over two tiles (64) does not hold
    assert ops["attn_decode"]["status"] == fp.NEW_CODE
    assert any(c["text"] == "rope_dim == 2 * TILE" and c["value"] is False for c in ops["attn_decode"]["constraints"])


def test_a_linear_attention_model_with_no_gdn_module_in_the_bundle_needs_new_code(tmp_path):
    b = make_bundle(tmp_path / "b", drop=("tt/gdn_decode/ops.py",))
    rep = fp.candidates(cfg("qwen3-coder-next.json"), b, bundle_config=cfg("qwen3-coder-next.json"), dies=4)
    op = by_name(rep)["gdn_decode"]
    assert op["status"] == fp.NEW_CODE and "not in the bundle" in " ".join(op["reasons"])


def test_unevaluable_terms_are_listed_and_do_not_change_the_status(bundle):
    rep = fp.candidates(cfg("qwen3-coder-next.json"), bundle, bundle_config=cfg("qwen3-coder-next.json"), dies=4)
    moe = by_name(rep)["moe_decode"]
    dtype_term = next(c for c in moe["constraints"] if c["text"].startswith("dtype in"))
    assert dtype_term["value"] is None
    assert moe["status"] == fp.AS_IS


def test_without_a_bundle_config_tunables_are_not_compared_and_the_report_says_so(bundle):
    rep = fp.candidates(cfg("qwen3.5-35b-a3b.json"), bundle, bundle_config=None, dies=2)
    assert by_name(rep)["router_topk"]["status"] == fp.AS_IS
    assert any("bundle config" in n for n in rep["notes"])


def test_the_report_carries_hazards_and_the_dependency_list(bundle):
    rep = fp.candidates(cfg("qwen3.5-35b-a3b.json"), bundle, bundle_config=cfg("qwen3-coder-next.json"), dies=2)
    assert "hazards" in rep
    router = by_name(rep)["router_topk"]
    assert set(router["depends_on"]) >= {"routed_experts", "top_k"}
    assert router["values"]["routed_experts"] == {"model": 256, "bundle": 512}


# ---- the command ------------------------------------------------------------------------------------------------------

def test_main_prints_json(bundle, capsys):
    rc = fp.main([str(FIX / "qwen3.5-35b-a3b.json"), str(bundle), "--bundle-config", str(FIX / "qwen3-coder-next.json"),
                  "--dies", "2", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"][fp.CHANGED] == 2


def test_main_text_table_names_every_op(bundle, capsys):
    rc = fp.main([str(FIX / "qwen3.5-35b-a3b.json"), str(bundle), "--bundle-config", str(FIX / "qwen3-coder-next.json")])
    assert rc == 0
    text = capsys.readouterr().out
    for n in ALL_OPS:
        assert n in text
    assert "applies with a changed constant" in text


def test_main_bad_config_exits_2_with_one_line(bundle, capsys, tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text("[]")
    assert fp.main([str(bad), str(bundle)]) == 2
    assert "fusion-candidates:" in capsys.readouterr().out


def test_main_bad_bundle_exits_2(capsys, tmp_path):
    assert fp.main([str(FIX / "qwen3-coder-next.json"), str(tmp_path / "missing")]) == 2


def test_cli_routes_the_command(bundle, capsys):
    from orchard import cli
    rc = cli.main(["fusion-candidates", str(FIX / "qwen3.5-35b-a3b.json"), str(bundle), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ops"]


# ---- the real bundle, when one is on hand -----------------------------------------------------------------------------

REAL = os.environ.get("ORCHARD_FUSED_BUNDLE_DIR")


@pytest.mark.skipif(not REAL or not Path(REAL).is_dir(), reason="set ORCHARD_FUSED_BUNDLE_DIR to a copy of the bundle's code")
def test_real_bundle_inventory_finds_the_ops():
    inv = fp.read_bundle_inventory(REAL)
    assert set(ALL_OPS) <= {o["name"] for o in inv["ops"] if o["found"]}
