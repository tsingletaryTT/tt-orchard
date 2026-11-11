# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU-only checks of probes 10 to 14 in scripts/qwen4_ops_probe (issue 723): the router, the routed experts (loop and sparse matmul), the shared expert and the
block, and the memory estimate of the Qwen4 sparse MoE block. Each selftest passes without ttnn, each device mode refuses to run when ttnn cannot be imported, each
selftest can fail (a broken reference form turns it red), the nnz rule of `ttnn.sparse_matmul` is checked against a stand-in that fails like the kernel would hang, and
the shared reference code matches transformers' Qwen4Exp MoE block when an interpreter that has it exists."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe"
DEVICE_PROBES = ["p10_moe_router", "p11_moe_experts_loop", "p12_moe_sparse_matmul", "p13_moe_shared_expert"]
PROBES = DEVICE_PROBES + ["p14_moe_memory"]


def _run(name, args, no_ttnn=False):
    code = (("import sys; sys.modules['ttnn'] = None; " if no_ttnn else "")
            + f"import sys, runpy; sys.argv = ['x'] + {args!r}; runpy.run_path(r'{SCRIPTS / (name + '.py')}', run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_RESULT ")]
    return proc, lines


@pytest.mark.parametrize("name", PROBES)
def test_selftest_passes_with_one_result_line_and_without_ttnn(name):
    proc, lines = _run(name, ["--selftest"], no_ttnn=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(lines) == 1
    res = json.loads(lines[0][len("PROBE_RESULT "):])
    assert res["ok"] is True and res["mode"] == "selftest" and res["probe"] == name and res["checks"]
    assert all(res["checks"].values()), res["checks"]


@pytest.mark.parametrize("name", DEVICE_PROBES)
def test_device_mode_refuses_when_ttnn_cannot_be_imported(name):
    proc, lines = _run(name, [], no_ttnn=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    res = json.loads(lines[-1][len("PROBE_RESULT "):])
    assert res["ok"] is False and "ttnn is not importable" in res["failure"]


def test_the_memory_probe_has_no_device_mode_and_prints_the_table_without_ttnn():
    proc, lines = _run("p14_moe_memory", [], no_ttnn=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    res = json.loads(lines[-1][len("PROBE_RESULT "):])
    assert res["mode"] == "estimate" and res["table"]["bfp4"]["all_layers_gb"] == pytest.approx(67.9, abs=0.1)


def _mod(name):
    sys.path.insert(0, str(SCRIPTS))
    import importlib
    return importlib.import_module(name)


# ---- each selftest goes red when its reference form is wrong -----------------------------------------------------------------------------

def test_p10_goes_red_when_the_selected_weights_are_not_normalised(monkeypatch):
    common, p10 = _mod("common"), _mod("p10_moe_router")
    assert p10.selftest()["ok"]
    real = common.moe_router_ref
    monkeypatch.setattr(common, "moe_router_ref", lambda h, r, k, norm_topk_prob=True, stable=False: real(h, r, k, False, stable))
    res = p10.selftest()
    assert res["ok"] is False and res["checks"]["selected_weights_are_normalised_to_one"] is False


def test_p10_goes_red_when_the_tie_report_counts_the_wrong_expert(monkeypatch):
    p10 = _mod("p10_moe_router")
    real = p10.tie_report
    monkeypatch.setattr(p10, "tie_report", lambda idx, pairs: {**real(idx, pairs), "rows_where_the_lower_index_won": 0})
    assert p10.selftest()["checks"]["the_stable_reference_always_takes_the_lower_index_of_a_tied_pair"] is False


def test_p10_goes_red_when_the_stable_reference_is_not_stable(monkeypatch):
    common, p10 = _mod("common"), _mod("p10_moe_router")
    monkeypatch.setattr(common, "stable_topk_ref", lambda s, k: torch.topk(s.flip(-1), k, dim=-1))
    assert p10.selftest()["ok"] is False


def test_p11_goes_red_when_the_row_padding_is_not_a_multiple_of_32(monkeypatch):
    p11 = _mod("p11_moe_experts_loop")
    assert p11.selftest()["ok"]
    monkeypatch.setattr(p11, "pad_rows", lambda tok: (tok, int(tok.numel())))
    assert p11.selftest()["ok"] is False


def test_p11_goes_red_when_the_reference_ignores_the_routing_weights(monkeypatch):
    common, p11 = _mod("common"), _mod("p11_moe_experts_loop")
    real = common.moe_experts_ref
    monkeypatch.setattr(common, "moe_experts_ref", lambda h, i, w, g, u, d: real(h, i, torch.ones_like(w), g, u, d))
    res = p11.selftest()
    assert res["ok"] is False and res["checks"]["dense_masked_equals_the_reference_loop"] is False


def test_p11_goes_red_when_the_scatter_matrix_puts_a_token_in_the_wrong_row(monkeypatch):
    p11 = _mod("p11_moe_experts_loop")
    real = p11.scatter_matrix
    monkeypatch.setattr(p11, "scatter_matrix", lambda tok, rows, total: real((tok + 1) % total, rows, total))
    assert p11.selftest()["checks"]["gather_with_padding_then_one_hot_scatter_equals_index_add"] is False


def test_p12_goes_red_when_the_emulated_op_uses_the_wrong_expert(monkeypatch):
    p12 = _mod("p12_moe_sparse_matmul")
    assert p12.selftest()["ok"]
    real = p12.emulate_sparse_matmul

    def wrong(a, b, sparsity=None, *, a_sparse=False, b_sparse=True, indices=None):
        return real(a, b.roll(1, dims=1), sparsity, a_sparse=a_sparse, b_sparse=b_sparse, indices=indices)
    monkeypatch.setattr(p12, "emulate_sparse_matmul", wrong)
    assert p12.selftest()["ok"] is False


def test_p12_goes_red_when_a_candidate_config_does_not_divide_the_k_tiles(monkeypatch):
    p12 = _mod("p12_moe_sparse_matmul")
    real = p12.candidate_configs
    monkeypatch.setattr(p12, "candidate_configs", lambda m, k, n, grid_max=(8, 8): [dict(c, in0_block_w=7) for c in real(m, k, n, grid_max)])
    assert p12.selftest()["checks"]["candidate_configs_exist_and_are_valid_for_every_preset_shape"] is False


def test_p12_goes_red_when_the_nnz_is_not_the_exact_count(monkeypatch):
    p12 = _mod("p12_moe_sparse_matmul")
    monkeypatch.setattr(p12, "exact_nnz", lambda sp: int(torch.count_nonzero(sp)) + 1)
    assert p12.selftest()["checks"]["exact_nnz_equals_the_count_of_non_zero_entries"] is False


def test_p12_goes_red_when_the_sparsity_marks_the_wrong_pairs(monkeypatch):
    p12 = _mod("p12_moe_sparse_matmul")
    monkeypatch.setattr(p12, "group_sparsity", lambda idx, e, group_rows=32: torch.ones(idx.shape[0] // group_rows, e))
    assert p12.selftest()["checks"]["the_sparsity_of_a_hand_built_routing_marks_each_group_separately"] is False


def test_p13_goes_red_when_the_coverage_table_loses_an_op(monkeypatch):
    p13 = _mod("p13_moe_shared_expert")
    assert p13.selftest()["ok"]
    smaller = dict(p13.OP_COVERAGE)
    smaller.pop("router.topk")
    monkeypatch.setattr(p13, "OP_COVERAGE", smaller)
    assert p13.selftest()["checks"]["every_op_of_the_reference_forward_is_listed"] is False


def test_p13_goes_red_when_the_one_hot_routing_is_wrong(monkeypatch):
    p13 = _mod("p13_moe_shared_expert")
    real = p13.one_hot_routing
    monkeypatch.setattr(p13, "one_hot_routing", lambda i, w, e: real(i, w, e).roll(1, dims=1))
    assert p13.selftest()["checks"]["one_hot_routing_equals_the_scatter_form"] is False


def test_p13_goes_red_when_the_shared_gate_is_dropped(monkeypatch):
    common, p13 = _mod("common"), _mod("p13_moe_shared_expert")
    real = common.moe_shared_ref
    monkeypatch.setattr(common, "moe_shared_ref", lambda h, g, u, d, gw: real(h, g, u, d, gw) / torch.sigmoid(h.float() @ gw.float().t()))
    assert p13.selftest()["checks"]["shared_expert_is_the_gated_mlp_times_the_sigmoid"] is False


def test_p14_goes_red_when_a_tile_size_is_wrong(monkeypatch):
    p14 = _mod("p14_moe_memory")
    assert p14.selftest()["ok"]
    monkeypatch.setitem(p14.BYTES_PER_ELEM, "bfp4", 0.5)
    assert p14.selftest()["checks"]["tile_sizes_give_the_bytes_per_element"] is False


def test_p14_goes_red_when_the_chip_count_rounds_down(monkeypatch):
    p14 = _mod("p14_moe_memory")
    monkeypatch.setattr(p14, "chips_needed", lambda total, share=1.0, usable_gb=29.9: max(1, int(total / (usable_gb * 1e9 * share))))
    assert p14.selftest()["ok"] is False


# ---- the shared reference code matches transformers' Qwen4Exp MoE block ----------------------------------------------------------------------

def _reference_python():
    cand = os.environ.get("ORCHARD_REFERENCE_PYTHON") or str(Path.home() / "orchard-venvs" / "reference" / "bin" / "python")
    return cand if Path(cand).exists() else None


def test_moe_reference_copies_match_transformers_qwen4_exp():
    py = _reference_python()
    if py is None:
        pytest.skip("set ORCHARD_REFERENCE_PYTHON to an interpreter with transformers that has qwen4_exp")
    code = f"""
import sys, torch
sys.path.insert(0, r'{SCRIPTS}')
import common
from transformers.models.qwen4_exp import modeling_qwen4_exp as m
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
for experts, top_k, hidden in ((8, 2, 64), (16, 4, 128)):
    t = common.moe_inputs(tokens=32, hidden=hidden, experts=experts, top_k=top_k, inter=64, shared=64, seed=3)
    cfg = Qwen4ExpTextConfig(hidden_size=hidden, num_experts=experts, num_experts_per_tok=top_k, moe_intermediate_size=64, shared_expert_intermediate_size=64)
    assert cfg.norm_topk_prob is True
    blk = m.Qwen4ExpTextSparseMoeBlock(cfg).eval()
    with torch.no_grad():
        blk.gate.weight.copy_(t['router'])
        blk.experts.gate_up_proj.copy_(torch.cat([t['gate'], t['up']], dim=1))
        blk.experts.down_proj.copy_(t['down'])
        blk.shared_expert.gate_proj.weight.copy_(t['s_gate']); blk.shared_expert.up_proj.weight.copy_(t['s_up']); blk.shared_expert.down_proj.weight.copy_(t['s_down'])
        blk.shared_expert_gate.weight.copy_(t['s_gate_w'])
        want = blk(t['hidden'][None])[0]
        _, rw, ri = blk.gate(t['hidden'])
    logits, w, i = common.moe_router_ref(t['hidden'], t['router'], top_k)
    assert torch.equal(i.sort(-1).values, ri.sort(-1).values), 'router indices'
    assert (w.gather(1, i.argsort(-1)) - rw.gather(1, ri.argsort(-1))).abs().max() < 1e-6, 'router weights'
    assert (common.moe_block_ref(t) - want).abs().max() < 1e-5, 'block'
print('MATCH')
"""
    proc = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=600)
    assert "MATCH" in proc.stdout, proc.stdout + proc.stderr


# ---- the device paths run against a torch-backed stand-in for ttnn: the probe code itself (names, shapes, comparisons) is exercised without a board -----

class _T:
    def __init__(self, t):
        self.t = t

    @property
    def shape(self):
        return tuple(self.t.shape)


def _fake_ttnn(topk_min_width=0, bad_grid=None, fail_all_configs=False):
    """A torch-backed ttnn. `topk_min_width`: ttnn.topk raises below that many columns (to exercise the padded form); `bad_grid`: sparse_matmul raises for programs on
    that core grid (to exercise the next candidate); `fail_all_configs`: every sparse_matmul raises. `sparse_matmul` fails when `nnz` is not the exact count of non-zero
    sparsity entries: the real kernel would hang, and a hang cannot be tested."""
    import torch.nn.functional as F
    p12 = _mod("p12_moe_sparse_matmul")
    ttnn = types.ModuleType("ttnn")
    ttnn.bfloat16 = ttnn.uint32 = ttnn.int32 = ttnn.uint16 = "dt"
    ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT = "tile", "rm"
    ttnn.MathFidelity = types.SimpleNamespace(HiFi4="HiFi4")
    ttnn.CoreCoord = lambda x, y: (x, y)
    ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig = lambda **kw: types.SimpleNamespace(**kw)
    ttnn.init_device_compute_kernel_config = lambda arch, **kw: "ckc"
    dev = types.SimpleNamespace(arch=lambda: "bh", compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=8, y=8))
    ttnn.open_device = lambda device_id=0, **k: dev
    ttnn.close_device = lambda d: None
    ttnn.from_torch = lambda t, dtype=None, layout=None, device=None: _T(t.float() if t.is_floating_point() else t.clone())
    ttnn.to_torch = lambda x: x.t.clone()
    w = lambda x: x.t if isinstance(x, _T) else x
    ttnn.reshape = lambda x, s: _T(x.t.reshape(tuple(s)))
    ttnn.slice = lambda x, a, b: _T(x.t[tuple(slice(i, j) for i, j in zip(a, b))])
    ttnn.add = lambda a, b: _T(w(a) + w(b))
    ttnn.subtract = lambda a, b: _T(w(a) - w(b))
    ttnn.multiply = lambda a, b: _T(w(a) * w(b))
    ttnn.matmul = lambda a, b: _T(torch.matmul(a.t, b.t))
    ttnn.linear = lambda a, b, **k: _T(torch.matmul(a.t, b.t))
    ttnn.silu = lambda x: _T(F.silu(x.t))
    ttnn.sigmoid = lambda x: _T(torch.sigmoid(x.t))
    ttnn.exp = lambda x: _T(torch.exp(x.t))
    ttnn.reciprocal = lambda x: _T(1.0 / x.t)
    ttnn.softmax = lambda x, dim=-1, **k: _T(torch.softmax(x.t, dim=dim))
    ttnn.sum = lambda x, dim=None, keepdim=False: _T(x.t.sum(dim=dim, keepdim=keepdim))
    ttnn.pad = lambda x, padding, value: _T(F.pad(x.t, (padding[-1][0], padding[-1][1]), value=value))
    ttnn.to_layout = lambda x, layout: x
    ttnn.embedding = lambda ids, table, layout=None, **k: _T(table.t.reshape(-1, table.t.shape[-1])[ids.t.long()])
    ttnn.scatter = lambda zero, dim=-1, index=None, src=None: _T(zero.t.scatter(dim, index.t.long(), src.t.to(zero.t.dtype)))

    def topk(x, k, dim=-1, largest=True, sorted=True, stable=False):
        if x.t.shape[-1] < topk_min_width:
            raise RuntimeError("topk needs at least a tile of columns")
        order = torch.sort(x.t, dim=-1, descending=True, stable=True)
        return _T(order.values[..., :k]), _T(order.indices[..., :k])
    ttnn.topk = topk

    def sparse_matmul(a, b, sparsity=None, nnz=None, is_input_a_sparse=False, is_input_b_sparse=True, indices=None, program_config=None, **k):
        if fail_all_configs or (bad_grid is not None and program_config.compute_with_storage_grid_size == bad_grid):
            raise RuntimeError("TT_FATAL: unsupported program config")
        if nnz is not None and nnz != int(torch.count_nonzero(sparsity.t)):
            raise AssertionError("nnz differs from the count of non-zero sparsity entries: the kernel would hang")
        return _T(p12.emulate_sparse_matmul(a.t, b.t, None if sparsity is None else sparsity.t, a_sparse=is_input_a_sparse, b_sparse=is_input_b_sparse,
                                            indices=None if indices is None else indices.t.reshape(-1)))
    ttnn.sparse_matmul = sparse_matmul
    return ttnn


def _result(capsys):
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("PROBE_RESULT ")][-1]
    return json.loads(line[len("PROBE_RESULT "):])


@pytest.mark.parametrize("name,expected", [("p10_moe_router", {"reference_form", "selected_softmax", "selected_softmax_manual"}),
                                           ("p11_moe_experts_loop", {"dense_masked", "dense_masked_fused_gate_up", "gather_scatter_host", "gather_scatter_device"}),
                                           ("p12_moe_sparse_matmul", {"grouped", "grouped_nnz", "indexed"}),
                                           ("p13_moe_shared_expert", {"shared_expert", "routing_matrix_scatter", "routing_matrix_one_hot", "block_dense"})])
def test_device_path_runs_against_a_torch_stand_in_and_every_form_passes(name, expected, monkeypatch, capsys):
    mod = _mod(name)
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn())
    assert mod.main(["--preset", "tiny"]) == 0
    res = _result(capsys)
    assert res["ok"] is True and res["mode"] == "device"
    forms = res["result"]
    assert set(forms) == expected
    bad = {k: v for k, v in forms.items() if not v.get("ok")}
    assert not bad, bad


def test_p10_uses_the_padded_topk_when_the_unpadded_one_raises(monkeypatch, capsys):
    p10 = _mod("p10_moe_router")
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn(topk_min_width=32))
    assert p10.main(["--preset", "tiny"]) == 0
    res = _result(capsys)
    assert "topk_padded" in res["workarounds"], res
    assert all(f["ok"] for f in res["result"].values()), res["result"]


def test_p10_reports_that_the_lower_index_wins_among_tied_experts(monkeypatch, capsys):
    p10 = _mod("p10_moe_router")
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn())
    assert p10.main(["--preset", "tiny", "--tied-pairs", "2"]) == 0
    ties = _result(capsys)["result"]["selected_softmax"]["ties"]
    assert ties["rows_with_one_of_a_tied_pair"] > 0 and ties["rows_with_one_of_a_tied_pair"] == ties["rows_where_the_lower_index_won"]


def test_p12_tries_the_next_config_when_one_raises_and_lists_the_failures(monkeypatch, capsys):
    p12 = _mod("p12_moe_sparse_matmul")
    first_grid = p12.candidate_configs(1, 64, 64, (8, 8))[0]["grid"]
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn(bad_grid=first_grid))
    assert p12.main(["--preset", "tiny"]) == 0
    res = _result(capsys)
    assert res["configs_that_raised"], res
    assert all(r["ok"] for r in res["result"].values()), res["result"]
    assert res["result"]["grouped"]["gate_up_config"]["grid"] != list(first_grid)


def test_p12_records_a_failure_and_still_returns_when_no_config_runs(monkeypatch, capsys):
    p12 = _mod("p12_moe_sparse_matmul")
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn(fail_all_configs=True))
    assert p12.main(["--preset", "tiny"]) == 0
    res = _result(capsys)
    assert all(not r["ok"] and "no program config ran" in r["failure"] for r in res["result"].values()), res["result"]
    assert len(res["configs_that_raised"]) > 0


def test_p12_never_passes_a_nnz_that_differs_from_the_count_and_the_guard_can_fail(monkeypatch, capsys):
    p12 = _mod("p12_moe_sparse_matmul")
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn())
    assert p12.main(["--preset", "tiny"]) == 0
    assert _result(capsys)["result"]["grouped_nnz"]["ok"] is True
    monkeypatch.setattr(p12, "exact_nnz", lambda sp: int(torch.count_nonzero(sp)) + 1)           # the stand-in fails like the kernel would hang
    assert p12.main(["--preset", "tiny"]) == 0
    res = _result(capsys)["result"]
    assert res["grouped_nnz"]["ok"] is False and "hang" in res["grouped_nnz"]["failure"], res["grouped_nnz"]
    assert res["grouped"]["ok"] is True                                                           # without nnz the op counts itself


def test_p11_device_form_fails_in_the_result_when_the_stand_in_computes_a_wrong_silu(monkeypatch, capsys):
    p11 = _mod("p11_moe_experts_loop")
    fake = _fake_ttnn()
    fake.silu = lambda x: _T(x.t)                                                                  # no activation
    monkeypatch.setitem(sys.modules, "ttnn", fake)
    assert p11.main(["--preset", "tiny"]) == 0
    assert not any(f["ok"] for f in _result(capsys)["result"].values())
