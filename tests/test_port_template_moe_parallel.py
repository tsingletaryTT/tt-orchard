# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The expert-parallel form of the generated sparse MoE block (issue 723, item 3).

When the routed experts of a checkpoint do not fit one chip's budget (`pt.MOE_EXPERT_BYTES_MAX`) the generator no longer leaves the block a stub. It writes the
form in which every chip holds `experts // chips` of the experts of each layer (chip c holds experts c*per .. (c+1)*per - 1; slot j of chip c is expert c*per + j),
each chip runs the experts it holds that some token picked (the gather/scatter form), and `ttnn.all_reduce` adds the chips' partial outputs. The chip count is the
mesh's, known only at load time, so `load` refuses a mesh that is too small (the routed experts must fit `MOE_EXPERT_BYTES_MAX` per chip) or one the expert count does
not divide. The comparison runs the generated block through the torch-backed fake ttnn with a mesh of 2 and 4 chips (sharded tensors and all_reduce are in the fake)
against the real transformers `Qwen4ExpTextSparseMoeBlock`. Probe 15 ran the dense-masked variant of this layout with a real all_reduce on a board; the
gather/scatter chain with sharded token lists has not run on chips."""
import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import import_port
from test_port_template_moe import SEQ, _checkpoint, _flash_like, _generate, _hf_block

PER_LAYER = 8 * 3 * 64 * 64 * 2                   # one layer's experts in the tiny fixture, bf16
TOTAL = PER_LAYER * 4                             # four layers hold experts


def parallel_text(monkeypatch, budget):
    monkeypatch.setattr(pt, "MOE_EXPERT_BYTES_MAX", budget)
    text, p = _generate()
    assert p.stubs == [], [s.reason for s in p.stubs]
    return text


_PORTS = iter(range(10 ** 6))


def run_block(monkeypatch, tmp_path, text, chips, seed=11):
    # a new module name each time: two texts of one size written within a second would otherwise share a stale .pyc
    mod = import_port(monkeypatch, text, tmp_path, name=f"generated_port_{next(_PORTS)}")
    _config, block = _hf_block(seed)
    model = mod.load(str(_checkpoint(tmp_path, block)), fake.MeshDevice(chips))
    x_host = (torch.randn(1, SEQ, 64, generator=torch.Generator().manual_seed(5)) * 0.8).to(torch.bfloat16)
    x = fake.Tensor(x_host.reshape(1, 1, SEQ, 64).float(), replicated=True, layout="TILE")
    with torch.no_grad():
        want = block(x_host.float()).float()
    out = mod._A_mlp(model, 0, x, list(range(SEQ)))
    return out, want, mod, model


def check(out, want):
    scale = want.abs().max().item()
    assert scale > 0.2
    for c in range(out.t.shape[0]):
        err = (out.t[c].float().reshape(want.shape) - want).abs().max().item()
        assert err <= 0.05 * scale, f"chip {c}: max abs error {err} against scale {scale}"


# ---- the plan -------------------------------------------------------------------------------------------------------------------------------

def test_experts_over_the_budget_are_written_in_the_parallel_form_and_the_text_says_so(monkeypatch):
    text = parallel_text(monkeypatch, TOTAL // 2)
    assert "parallel=True" in text and "expert_shard" in text and "_moe_experts_parallel" in text


def test_experts_that_fit_the_budget_keep_the_replicated_form(monkeypatch):
    monkeypatch.setattr(pt, "MOE_EXPERT_BYTES_MAX", TOTAL)
    text, p = _generate()
    assert p.stubs == [] and "parallel=True" not in text and "expert_shard" not in text.split("PLACEMENT")[1].split("\n\n")[0]


def test_the_real_512_expert_shape_is_written_in_the_parallel_form_instead_of_left_a_stub():
    mc, mt = _flash_like(512)
    text, p = _generate(mutate_construction=mc, mutate_tensors=mt)
    assert not {s.function for s in p.stubs} & {"_A_mlp", "_B_mlp", "_C_mlp"}
    assert "experts=512, top_k=10, norm_topk=True, dense=False, parallel=True" in text


# ---- the comparison with the transformers module ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chips", [2, 4])
def test_the_parallel_block_matches_the_transformers_module_on_a_mesh_of_2_and_4_chips(monkeypatch, tmp_path, chips):
    text = parallel_text(monkeypatch, TOTAL // 2)             # 2 chips hold exactly the budget each
    out, want, _mod, _model = run_block(monkeypatch, tmp_path, text, chips)
    assert out.chips == chips
    check(out, want)


def test_every_chip_ends_with_the_same_output(monkeypatch, tmp_path):
    text = parallel_text(monkeypatch, TOTAL // 2)
    out, _want, _m, _model = run_block(monkeypatch, tmp_path, text, 4)
    assert all(torch.equal(out.t[0], out.t[c]) for c in range(1, 4))


def test_each_chip_holds_only_its_share_of_the_experts(monkeypatch, tmp_path):
    text = parallel_text(monkeypatch, TOTAL // 2)
    _out, _want, _mod, model = run_block(monkeypatch, tmp_path, text, 4)
    slots = [n for n in model["dev"] if ".experts.slot" in n and n.startswith("model.layers.0.")]
    assert len(slots) == 2 * 3                                # 8 experts over 4 chips = 2 slots, three projections each
    assert not [n for n in model["dev"] if ".experts." in n and ".slot" not in n], "no replicated expert weight may be uploaded"
    w = model["dev"]["model.layers.0.mlp.experts.slot0.gate_proj.weight"]
    assert w.t.shape[0] == 4 and w.shape == (1, 1, 64, 64)


def test_slot_j_of_chip_c_holds_expert_c_times_per_plus_j(monkeypatch, tmp_path):
    text = parallel_text(monkeypatch, TOTAL // 2)
    _o, _w, _mod, model = run_block(monkeypatch, tmp_path, text, 4)
    _config, block = _hf_block(11)
    inter = block.experts.intermediate_dim
    gate_up = block.state_dict()["experts.gate_up_proj"].to(torch.bfloat16)
    w = model["dev"]["model.layers.0.mlp.experts.slot1.gate_proj.weight"].t            # [chips, 1, in, out]
    for c in range(4):
        e = c * 2 + 1
        assert torch.equal(w[c, 0].to(torch.bfloat16), gate_up[e][:inter].t().to(torch.bfloat16))


# ---- what load refuses -----------------------------------------------------------------------------------------------------------------------

def test_a_mesh_too_small_for_the_budget_is_refused_at_load_and_the_message_names_the_chips_needed(monkeypatch, tmp_path):
    text = parallel_text(monkeypatch, TOTAL // 2)
    with pytest.raises(ValueError, match=r"at least 2 chips"):
        run_block(monkeypatch, tmp_path, text, 1)


def test_a_chip_count_the_experts_do_not_divide_is_refused(monkeypatch, tmp_path):
    text = parallel_text(monkeypatch, TOTAL // 3)
    with pytest.raises(ValueError, match=r"8 experts do not split evenly over 3 chips"):
        run_block(monkeypatch, tmp_path, text, 3)


# ---- mutations of the emitted text: each must turn the comparison red -------------------------------------------------------------------------

MUTATIONS = {
    "no all_reduce": ("    return ttnn.all_reduce(out, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)", "    return out"),
    "slot j of chip c is expert j*chips + c": ("for c in range(chips)]).reshape(chips, 1, *shape)", "for c in range(chips)][::-1]).reshape(chips, 1, *shape)"),
    "the routing weights are ignored": ("weight[c, 0, :n, 0] = w", "weight[c, 0, :n, 0] = 1.0"),
    "the scatter is dropped": ("back[c, 0, tokens, torch.arange(n)] = 1.0", "back[c, 0, 0, 0] = 1.0"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_mutation_of_the_parallel_form_turns_the_comparison_red(monkeypatch, tmp_path, name):
    text = parallel_text(monkeypatch, TOTAL // 2)
    old, new = MUTATIONS[name]
    assert old in text, f"the emitted text no longer contains {old!r}: update the mutation"
    out, want, _m, _model = run_block(monkeypatch, tmp_path, text.replace(old, new, 1), 4)
    with pytest.raises(AssertionError):
        check(out, want)


# ---- the weight format of the sharded experts (issue 731: bfp4 only above a measured bar; the plan command says so) ---------------------------------------------

def dtype_text(monkeypatch, fmt, budget=TOTAL // 2):
    monkeypatch.setattr(pt, "MOE_EXPERT_DTYPE", fmt)
    return parallel_text(monkeypatch, budget)


def test_the_expert_format_is_bf16_unless_the_environment_names_bfp8_or_bfp4():
    assert pt._moe_dtype({}) == "bf16"
    assert pt._moe_dtype({"ORCHARD_MOE_EXPERT_DTYPE": "bfp4"}) == "bfp4"
    assert pt._moe_dtype({"ORCHARD_MOE_EXPERT_DTYPE": " bfp8 "}) == "bfp8"
    for bad in ("fp8", "int4", "BFP4x", "0"):
        with pytest.raises(ValueError, match="ORCHARD_MOE_EXPERT_DTYPE"):
            pt._moe_dtype({"ORCHARD_MOE_EXPERT_DTYPE": bad})


def test_the_generated_text_names_the_format_and_the_bytes_each_format_takes(monkeypatch):
    text = dtype_text(monkeypatch, "bfp4")
    assert 'MOE_EXPERT_DTYPE = "bfp4"' in text and "bfloat4_b" in text and "_BYTES_PER_ELEMENT" in text


@pytest.mark.parametrize("fmt,bound", [("bfp8", 0.12), ("bfp4", 0.40)])
def test_the_block_with_quantized_experts_stays_near_the_reference_and_is_not_exact(monkeypatch, tmp_path, fmt, bound):
    text = dtype_text(monkeypatch, fmt)
    out, want, _m, _model = run_block(monkeypatch, tmp_path, text, 2)
    scale = want.abs().max().item()
    err = (out.t[0].float().reshape(want.shape) - want).abs().max().item()
    assert 0.002 * scale < err < bound * scale, f"{fmt}: error {err / scale:.4f} of scale"


def test_the_uploaded_expert_weights_are_the_rounded_values_not_the_bf16_ones(monkeypatch, tmp_path):
    text = dtype_text(monkeypatch, "bfp4")
    _o, _w, _mod, model = run_block(monkeypatch, tmp_path, text, 2)
    w = model["dev"]["model.layers.0.mlp.experts.slot0.gate_proj.weight"]
    assert w.dtype == "bfloat4_b"
    vals = w.t[0, 0]
    assert len({round(float(v), 6) for v in vals[0, :16]}) <= 8 * 2, "a block of 16 bfp4 values has few distinct magnitudes"


def test_the_chip_count_check_uses_the_bytes_of_the_chosen_format(monkeypatch, tmp_path):
    """The tiny experts hold 786432 bytes in bf16 and 221184 in bfp4. With a budget of 393216 one chip cannot hold them in bf16 and can in bfp4."""
    with pytest.raises(ValueError, match="at least 2 chips"):
        run_block(monkeypatch, tmp_path, dtype_text(monkeypatch, "bf16"), 1)
    out, want, _m, _model = run_block(monkeypatch, tmp_path, dtype_text(monkeypatch, "bfp4"), 1)
    assert out.t.shape[0] == 1


def test_the_refusal_names_the_format_when_a_smaller_format_would_fit(monkeypatch, tmp_path):
    text = dtype_text(monkeypatch, "bf16", budget=TOTAL // 8)         # 98304 bytes: bf16 needs 8 chips, bfp4 (221184) needs 3
    with pytest.raises(ValueError, match=r"in bf16 .* at least 8 chips"):
        run_block(monkeypatch, tmp_path, text, 4)
