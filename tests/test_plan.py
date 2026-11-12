# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard plan`: what precision and placement a checkpoint needs on N chips, from safetensors headers alone (issue 739, after the Strata
fit plan). The tests write real safetensors headers over zero bytes, so the parameter counts are known exactly."""
import io
import json
import struct

import pytest

from orchard import defaults, plan

GIB = float(1 << 30)


def write_st(path, tensors: dict):
    """A safetensors file with the given {name: shape} as bf16 tensors of zeros."""
    header, off = {}, 0
    for name, shape in tensors.items():
        n = 2
        for d in shape:
            n *= d
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(off))


MOE = {"model.embed_tokens.weight": (1000, 64), "model.layers.0.self_attn.q_proj.weight": (64, 64),
       "model.layers.0.mlp.experts.0.gate_proj.weight": (128, 64), "model.layers.0.mlp.experts.1.gate_proj.weight": (128, 64),
       "model.layers.0.mlp.shared_expert.gate_proj.weight": (128, 64)}


def snap(tmp_path, tensors=MOE):
    d = tmp_path / "snap"
    d.mkdir()
    write_st(d / "model-00001.safetensors", tensors)
    return d


def test_routed_experts_are_split_from_everything_else_and_the_shared_expert_counts_as_other(tmp_path):
    inv = plan.inventory(snap(tmp_path))
    assert inv["routed_expert_params"] == 2 * 128 * 64
    assert inv["other_params"] == 1000 * 64 + 64 * 64 + 128 * 64 and inv["tensors"] == 5 and inv["unreadable"] == []


def test_each_format_is_priced_at_its_bytes_per_element(tmp_path):
    res = plan.plan(snap(tmp_path), 4)
    params = 1000 * 64 + 64 * 64 + 3 * 128 * 64
    by = {r["name"]: r for r in res["rows"]}
    assert by["all bf16"]["total_gib"] == pytest.approx(params * 2 / GIB, abs=0.01)
    assert by["all bfp8"]["total_gib"] == pytest.approx(params * 1.0625 / GIB, abs=0.01)
    exp_bfp4 = (2 * 128 * 64) * 0.5625 / GIB
    other_bfp8 = (params - 2 * 128 * 64) * 1.0625 / GIB
    assert by["bfp8 + bfp4 experts"]["total_gib"] == pytest.approx(exp_bfp4 + other_bfp8, abs=0.01)


def test_a_dense_model_has_no_mixed_row_and_fits_replicated(tmp_path):
    res = plan.plan(snap(tmp_path, {"model.layers.0.mlp.gate_proj.weight": (64, 64)}), 1)
    assert [r["name"] for r in res["rows"]] == ["all bf16", "all bfp8", "all bfp4"]
    assert res["recommendation"] == {"row": "all bf16", "placement": "replicated", "needs": None}


def test_a_model_too_big_to_replicate_needs_the_experts_split_and_the_plan_says_so_and_names_the_issue(tmp_path):
    # bf16 totals 0.0001726 GiB, the non-expert part alone 0.0001421 and the 2-way split 0.0001574; a budget of 0.0001635 lets only the split fit
    res = plan.plan(snap(tmp_path), 4, chip_gib=0.000175, share=1.0)
    row = {r["name"]: r for r in res["rows"]}["all bf16"]
    assert not row["fits_replicated"] and row["fits_sharded"] and row["min_chips_sharded"] == 2
    assert res["recommendation"] == {"row": "all bfp8", "placement": "replicated", "needs": None}      # a smaller format that fits replicated beats a split of bf16


def test_a_recommendation_that_needs_the_experts_split_says_so_and_names_the_issue(tmp_path):
    # all bfp4 totals 4.855e-5 GiB and splits to 4.21e-5 over 4 chips; a budget of 4.48e-5 lets only that split fit, and no other row fits at all
    res = plan.plan(snap(tmp_path), 4, chip_gib=4.8e-5, share=1.0)
    assert res["recommendation"]["row"] == "all bfp4" and res["recommendation"]["placement"] == "sharded"
    assert "issue 723" in res["recommendation"]["needs"]


def test_when_nothing_fits_even_split_the_plan_says_so(tmp_path):
    res = plan.plan(snap(tmp_path), 4, chip_gib=0.00001, share=1.0)
    assert res["recommendation"] is None and any("nothing fits" in n for n in res["notes"])


def test_sharding_divides_only_the_experts(tmp_path):
    res = plan.plan(snap(tmp_path), 4)
    r = {x["name"]: x for x in res["rows"]}["all bf16"]
    other = (1000 * 64 + 64 * 64 + 128 * 64) * 2 / GIB
    experts = 2 * 128 * 64 * 2 / GIB
    assert r["sharded_per_chip_gib"] == pytest.approx(other + experts / 4, abs=0.01)
    assert r["replicated_per_chip_gib"] == pytest.approx(other + experts, abs=0.01)


def test_the_weight_share_sets_the_budget():
    assert plan.budget_gib(32.0, 0.85) == pytest.approx(32.0 * plan.USABLE_FRACTION * 0.85)
    assert plan.budget_gib(32.0, 0.5) < plan.budget_gib(32.0, 0.85)


def test_a_hugging_face_snapshot_link_into_blobs_is_followed(tmp_path):
    blobs = tmp_path / "models--x" / "blobs"
    d = tmp_path / "models--x" / "snapshots" / "rev"
    blobs.mkdir(parents=True)
    d.mkdir(parents=True)
    write_st(blobs / "abc", MOE)
    (d / "model.safetensors").symlink_to("../../blobs/abc")
    assert plan.inventory(d)["tensors"] == 5


def test_a_link_that_leaves_the_snapshot_is_named_and_not_read(tmp_path):
    outside = tmp_path / "elsewhere.safetensors"
    write_st(outside, MOE)
    d = tmp_path / "snap"
    d.mkdir()
    (d / "model.safetensors").symlink_to(outside)
    inv = plan.inventory(d)
    assert inv["tensors"] == 0 and "leaves the snapshot" in inv["unreadable"][0]


def test_a_broken_header_is_named_not_counted(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    (d / "model.safetensors").write_bytes(struct.pack("<Q", 10**12) + b"{}")
    inv = plan.inventory(d)
    assert inv["tensors"] == 0 and inv["unreadable"]


def test_main_prints_json_and_refuses_bad_arguments(tmp_path, capsys):
    out = io.StringIO()
    assert plan.main([str(snap(tmp_path)), "--chips", "2", "--json"], out) == 0
    assert json.loads(out.getvalue())["chips"] == 2
    for argv in ([str(tmp_path / "nope")], [str(tmp_path / "snap"), "--chips", "0"], [str(tmp_path / "snap"), "--weight-share", "2"]):
        assert plan.main(argv, io.StringIO()) == 2


def test_a_directory_with_no_tensors_exits_two(tmp_path):
    (tmp_path / "empty").mkdir()
    assert plan.main([str(tmp_path / "empty")], io.StringIO()) == 2


def test_the_text_table_names_each_combination_and_the_best_fit(tmp_path):
    text = plan.render(plan.plan(snap(tmp_path), 4))
    for name in ("all bf16", "all bfp8", "bfp8 + bfp4 experts", "all bfp4", "best fit:"):
        assert name in text


def test_the_cli_has_the_plan_subcommand(tmp_path, capsys):
    from orchard import cli
    assert cli.main(["plan", str(snap(tmp_path)), "--chips", "1"]) == 0
    assert "best fit" in capsys.readouterr().out
