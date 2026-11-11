# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU-only checks of probes 6 to 9 in scripts/qwen4_ops_probe (issue 711): the rotary broadcast, topk ties, masked SDPA and the other attention
ops of the Qwen4 indexer attention. Each selftest passes without ttnn, the device mode refuses to run when ttnn cannot be imported, each selftest
can fail (a broken reference form turns it red), and the shared reference code matches transformers' Qwen4Exp code when an interpreter that has
it exists."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe"
PROBES = ["p6_rotary_broadcast", "p7_topk_ties", "p8_masked_sdpa", "p9_attention_ops"]


def _run(name, args, no_ttnn=False):
    code = (("import sys; sys.modules['ttnn'] = None; " if no_ttnn else "")
            + f"import sys, runpy; sys.argv = ['x'] + {args!r}; runpy.run_path(r'{SCRIPTS / (name + '.py')}', run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_RESULT ")]
    return proc, lines


@pytest.mark.parametrize("name", PROBES)
def test_selftest_passes_with_one_result_line_and_without_ttnn(name):
    proc, lines = _run(name, ["--selftest"], no_ttnn=True)        # ttnn raises on import: the selftest must not touch it
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(lines) == 1
    res = json.loads(lines[0][len("PROBE_RESULT "):])
    assert res["ok"] is True and res["mode"] == "selftest" and res["probe"] == name and res["checks"]
    assert all(res["checks"].values()), res["checks"]


@pytest.mark.parametrize("name", PROBES)
def test_device_mode_refuses_when_ttnn_cannot_be_imported(name):
    proc, lines = _run(name, [], no_ttnn=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    res = json.loads(lines[-1][len("PROBE_RESULT "):])
    assert res["ok"] is False and "ttnn is not importable" in res["failure"]


def _mod(name):
    sys.path.insert(0, str(SCRIPTS))
    import importlib
    return importlib.import_module(name)


# ---- each selftest goes red when its reference form is wrong -----------------------------------------------------------------------------

def test_p6_selftest_goes_red_when_the_rotate_half_matrix_is_wrong(monkeypatch):
    common, p6 = _mod("common"), _mod("p6_rotary_broadcast")
    assert p6.selftest()["ok"]
    real = common.rotate_half_matrix
    monkeypatch.setattr(common, "rotate_half_matrix", lambda hd, rot: -real(hd, rot))
    assert p6.selftest()["ok"] is False


def test_p6_selftest_goes_red_when_the_padded_tables_do_not_pass_the_tail_through(monkeypatch):
    common, p6 = _mod("common"), _mod("p6_rotary_broadcast")
    real = common.padded_tables

    def wrong(cos, sin, hd):
        c, s = real(cos, sin, hd)
        return c * 0.5, s
    monkeypatch.setattr(common, "padded_tables", wrong)
    assert p6.selftest()["ok"] is False


def test_p7_selftest_goes_red_when_the_stable_reference_is_not_stable(monkeypatch):
    common, p7 = _mod("common"), _mod("p7_topk_ties")
    assert p7.selftest()["ok"]
    monkeypatch.setattr(common, "stable_topk_ref", lambda s, k: torch.topk(s.flip(-1), k, dim=-1))        # reversed ties: lowest index no longer wins
    assert p7.selftest()["ok"] is False


def test_p8_selftest_goes_red_when_the_block_mask_expansion_is_wrong(monkeypatch):
    p8 = _mod("p8_masked_sdpa")
    assert p8.selftest()["ok"]
    real = p8.expansion_matrix
    monkeypatch.setattr(p8, "expansion_matrix", lambda nb, ratio, seq: real(nb, ratio, seq).roll(1, dims=1))
    assert p8.selftest()["ok"] is False


def test_p9_selftest_goes_red_when_the_pooling_reference_is_wrong(monkeypatch):
    common, p9 = _mod("common"), _mod("p9_attention_ops")
    assert p9.selftest()["ok"]
    real = common.rms_norm_ref
    monkeypatch.setattr(common, "rms_norm_ref", lambda x, w, eps=1e-6: real(x, w, eps) * 1.02)
    assert p9.selftest()["ok"] is False


def test_p7_selftest_goes_red_when_the_padding_value_beats_the_valid_scores(monkeypatch):
    common, p7 = _mod("common"), _mod("p7_topk_ties")
    real = p7.padded_scores

    def wrong(seq, ratio, zero_share, seed=0):
        scores, n = real(seq, ratio, zero_share, seed)
        scores[torch.arange(scores.shape[1])[None, :] >= n[:, None]] = 9.0          # padding above every valid score
        return scores, n
    monkeypatch.setattr(p7, "padded_scores", wrong)
    res = p7.selftest()
    assert res["ok"] is False and res["checks"]["padding_is_never_chosen_before_a_valid_block"] is False


def test_p8_selftest_goes_red_when_the_reference_ignores_the_mask(monkeypatch):
    p8 = _mod("p8_masked_sdpa")
    real = p8.reference_attention
    monkeypatch.setattr(p8, "reference_attention",
                        lambda q, k, v, mask, gate, scale, n_rep: real(q, k, v, torch.ones_like(mask).tril(), gate, scale, n_rep))
    res = p8.selftest()
    assert res["ok"] is False and res["checks"]["the_selection_changes_the_output_against_full_attention"] is False


def test_p9_selftest_goes_red_when_the_coverage_table_loses_an_op(monkeypatch):
    p9 = _mod("p9_attention_ops")
    smaller = dict(p9.OP_COVERAGE)
    smaller.pop("indexer.topk")
    monkeypatch.setattr(p9, "OP_COVERAGE", smaller)
    assert p9.selftest()["checks"]["every_op_of_the_reference_forward_is_listed"] is False


# ---- the shared reference code matches transformers' Qwen4Exp -------------------------------------------------------------------------------

def _reference_python():
    cand = os.environ.get("ORCHARD_REFERENCE_PYTHON") or str(Path.home() / "orchard-venvs" / "reference" / "bin" / "python")
    return cand if Path(cand).exists() else None


def test_rotary_and_indexer_copies_match_transformers_qwen4_exp():
    py = _reference_python()
    if py is None:
        pytest.skip("set ORCHARD_REFERENCE_PYTHON to an interpreter with transformers that has qwen4_exp")
    code = f"""
import sys, torch
sys.path.insert(0, r'{SCRIPTS}')
import common
from transformers.models.qwen4_exp import modeling_qwen4_exp as m
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
S, H, D, ROT = 32, 4, 32, 8
# rotary: tables and application
cfg = Qwen4ExpTextConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=2, head_dim=32, partial_rotary_factor=0.25,
                         indexer_budget=16, indexer_compress_ratio=4, indexer_head_dim=D, indexer_kv_heads=1, indexer_n_heads=H)
rot_mod = m.Qwen4ExpTextRotaryEmbedding(cfg)
pos = torch.arange(S)[None]
cos, sin = rot_mod(torch.zeros(1, S, 64), pos)
ccos, csin = common.rope_tables_ref(S, 32, ROT, theta=cfg.rope_parameters['rope_theta'])
assert cos.shape[-1] == ROT and (cos[0] - ccos).abs().max() < 1e-5 and (sin[0] - csin).abs().max() < 1e-5
x = torch.randn(1, 8, S, 32)
want = m.apply_rotary_pos_emb(x, None, cos, sin)
got = common.apply_rotary_ref(x, ccos[None, None], csin[None, None])
assert (want - got).abs().max() < 1e-5
# the generated formula x * cos + (x @ R) * sin with padded tables
pc, ps = common.padded_tables(ccos, csin, 32)
formula = x * pc + (x @ common.rotate_half_matrix(32, ROT)) * ps
assert (want - formula).abs().max() < 1e-5
# indexer: the mask of the module equals qsa_selection_ref on the same hidden states
torch.manual_seed(0)
idx = m.Qwen4ExpTextQSAIndexer(cfg, 0).eval()
with torch.no_grad():
    idx.q_layernorm.weight.normal_(0, 0.1); idx.k_layernorm.weight.normal_(0, 0.1)
h = torch.randn(1, S, 64)
causal = torch.ones(S, S, dtype=torch.bool).tril()[None, None]
with torch.no_grad():
    mod_mask = idx(h, (cos, sin), causal, None)[0, 0]
    qk = idx.index_qk_proj(h)
    q, tk = torch.split(qk, [H * D, D], dim=-1)
    q = idx.q_layernorm(q.reshape(1, S, H, D))
    q = m.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)[0]
    raw = tk.reshape(1, S, D)[0]
    ours = common.qsa_selection_ref(q, raw, cos[0], sin[0], budget=16, ratio=4, k_weight=idx.k_layernorm.weight)
assert torch.equal(ours['mask'], mod_mask), int((ours['mask'] != mod_mask).sum())
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


def _fake_ttnn():
    import types
    import torch.nn.functional as F
    ttnn = types.ModuleType("ttnn")
    ttnn.bfloat16 = ttnn.uint32 = ttnn.int32 = "dt"
    ttnn.TILE_LAYOUT = ttnn.ROW_MAJOR_LAYOUT = "layout"
    ttnn.Shape = lambda s: tuple(s)
    ttnn.open_device = lambda device_id=0, **k: "dev"
    ttnn.close_device = lambda dev: None
    ttnn.from_torch = lambda t, dtype=None, layout=None, device=None: _T(t.float() if t.is_floating_point() else t.clone())
    ttnn.to_torch = lambda x: x.t.clone()
    w = lambda x: x.t if isinstance(x, _T) else x
    ttnn.reshape = lambda x, s: _T(x.t.reshape(tuple(s)))
    ttnn.permute = lambda x, d: _T(x.t.permute(*d))
    ttnn.slice = lambda x, a, b: _T(x.t[tuple(slice(i, j) for i, j in zip(a, b))])
    ttnn.add = lambda a, b: _T(w(a) + w(b))
    ttnn.subtract = lambda a, b: _T(w(a) - w(b))
    ttnn.multiply = lambda a, b: _T(w(a) * w(b))
    ttnn.matmul = lambda a, b: _T(torch.matmul(a.t, b.t))
    ttnn.linear = lambda a, b, dtype=None, **k: _T(torch.matmul(a.t, b.t))
    ttnn.relu = lambda x: _T(torch.relu(x.t))
    ttnn.sigmoid = lambda x: _T(torch.sigmoid(x.t))
    ttnn.sum = lambda x, dim=None, keepdim=False: _T(x.t.sum(dim=dim, keepdim=keepdim))
    ttnn.mean = lambda x, dim=None, keepdim=False: _T(x.t.mean(dim=dim, keepdim=keepdim))
    ttnn.concat = lambda xs, dim=0: _T(torch.cat([x.t for x in xs], dim=dim))
    ttnn.repeat = lambda x, s: _T(x.t.repeat(*s))
    ttnn.repeat_interleave = lambda x, n, dim=0: _T(x.t.repeat_interleave(n, dim=dim))
    ttnn.rms_norm = lambda x, epsilon=1e-6, weight=None: _T(x.t * torch.rsqrt(x.t.pow(2).mean(-1, keepdim=True) + epsilon) * (weight.t if weight is not None else 1.0))
    ttnn.embedding = lambda idx, table: _T(table.t.reshape(table.t.shape[-2], -1)[idx.t.long()])
    def topk(x, k, dim=-1, largest=True, sorted=True, stable=False):
        order = torch.sort(x.t, dim=-1, descending=True, stable=True)
        return _T(order.values[..., :k]), _T(order.indices[..., :k])
    ttnn.topk = topk

    def sdpa(q, k, v, attn_mask=None, is_causal=True, scale=None):
        kk = k.t.repeat_interleave(q.t.shape[1] // k.t.shape[1], dim=1)
        vv = v.t.repeat_interleave(q.t.shape[1] // v.t.shape[1], dim=1)
        return _T(F.scaled_dot_product_attention(q.t, kk, vv, attn_mask=None if attn_mask is None else attn_mask.t, is_causal=is_causal, scale=scale))
    ttnn.transformer = types.SimpleNamespace(scaled_dot_product_attention=sdpa)
    return ttnn


@pytest.mark.parametrize("name,preset", [("p6_rotary_broadcast", "tiny"), ("p7_topk_ties", "tiny"), ("p8_masked_sdpa", "tiny"), ("p9_attention_ops", "tiny")])
def test_device_path_runs_against_a_torch_stand_in_and_every_step_passes(name, preset, monkeypatch, capsys):
    mod, common = _mod(name), _mod("common")
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn())
    assert mod.main(["--preset", preset]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("PROBE_RESULT ")][-1]
    res = json.loads(line[len("PROBE_RESULT "):])
    assert res["ok"] is True and res["mode"] == "device"

    def walk(obj):
        if isinstance(obj, dict):
            if "ok" in obj and "failure" not in obj and not obj["ok"] and name != "p7_topk_ties":
                yield obj
            if "failure" in obj:
                yield obj
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)
    bad = list(walk(res["result"]))
    assert not bad, bad[:3]


# ---- the flash presets (issue 696: the QSA block at Flash-Next widths) --------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["p8_masked_sdpa", "p9_attention_ops"])
def test_the_flash_long_preset_holds_the_real_budget_and_a_prompt_past_it(name):
    mod = _mod(name)
    cfg = mod.PRESETS["flash-long"]
    assert cfg["hd"] == 256 and cfg["index_dim"] == 128 and cfg["budget"] == 2048 and cfg["ratio"] == 4
    assert cfg["seq"] > cfg["budget"] + cfg["ratio"] - 1 and cfg["seq"] % cfg["ratio"] == 0
    assert cfg["rot"] == cfg["hd"] // 4 and cfg["heads"] == 24 and cfg["kv_heads"] == 2


@pytest.mark.parametrize("name", ["p8_masked_sdpa", "p9_attention_ops"])
def test_device_path_runs_against_a_torch_stand_in_at_the_flash_preset(name, monkeypatch, capsys):
    mod = _mod(name)
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn())
    assert mod.main(["--preset", "flash"]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("PROBE_RESULT ")][-1]
    assert json.loads(line[len("PROBE_RESULT "):])["ok"] is True
