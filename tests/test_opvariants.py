# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Candidate ttnn names for aten ops with no table row (orchard/opvariants.py).

The table cases run against the committed dump of ttnn 0.80.0, so they say what the shipped tool answers for real
names. The expectations were read by a person: an entry is there because the ttnn op does the same job, or because the
simulator check (`inventory verify-candidates`) is the next step for it. The negative cases are names that share a
token or a spelling with an op that does a different job.
"""
import json
from pathlib import Path

import pytest

from orchard import opvariants as ov

DUMP = Path(ov.__file__).resolve().parent / "data" / "ttnn_ops-0.80.0.json"
NAMES = json.loads(DUMP.read_text())["names"]
IX = ov.Index(NAMES)


def got(op, index=IX):
    return [(c["ttnn"], c["why"]) for c in ov.candidates(op, index)]


EQ, SY, TS, NE, AF = "normalised-equal", "synonym", "token-subset", "near", "affix"

# (aten op, expected [(ttnn, why)]). The first block is the real unmapped lists of the 2026-10-16 inventory runs.
CASES = [
    # Qwen3 and Moshi
    ("aten.__and__.Tensor", [("bitwise_and", SY), ("logical_and", SY)]),
    ("aten.index.Tensor", [("gather", SY), ("embedding", SY)]),
    ("aten.to.dtype_layout", [("typecast", SY)]),
    ("aten.diff.default", []),
    ("aten.new_ones.default", [("ones", SY), ("full", SY)]),
    ("aten.to.device", [("to_device", SY)]),
    ("aten.type_as.default", [("typecast", SY)]),
    # RWKV
    ("aten.copy_.default", [("copy", EQ), ("assign", SY), ("clone", SY)]),
    ("aten.div_.Tensor", [("div_", EQ), ("div", EQ), ("divide_", SY)]),
    # Mamba2
    ("aten.einsum.default", []),
    ("aten.bitwise_not.default", [("bitwise_not", EQ)]),
    ("aten.masked_fill.Scalar", [("where", SY)]),
    ("aten.split_with_sizes.default", [("split", SY), ("chunk", SY)]),
    # names the table could take as they are
    ("aten.exp2.default", [("exp2", EQ)]),
    ("aten.cos.default", [("cos", EQ)]),
    ("aten.rsqrt.default", [("rsqrt", EQ)]),
    ("aten.erf.default", [("erf", EQ)]),
    ("aten.gelu.default", [("gelu", EQ)]),
    ("aten.sin.default", [("sin", EQ)]),
    ("aten.sinh.default", [("sinh", EQ)]),
    ("aten.expm1.default", [("expm1", EQ)]),
    ("aten.log1p.default", [("log1p", EQ)]),
    ("aten.round.default", [("round", EQ)]),
    ("aten.softplus.default", [("softplus", EQ)]),
    ("aten.hardtanh.default", [("hardtanh", EQ)]),
    ("aten.fmod.Scalar", [("fmod", EQ)]),
    ("aten.remainder.Scalar", [("remainder", EQ)]),
    ("aten.lerp.Scalar", [("lerp", EQ)]),
    ("aten.triu.default", [("triu", EQ)]),
    ("aten.roll.default", [("roll", EQ)]),
    ("aten.glu.default", [("glu", EQ)]),
    ("aten.nonzero.default", [("nonzero", EQ)]),
    ("aten.topk.default", [("topk", EQ)]),
    ("aten.cumprod.default", [("cumprod", EQ)]),
    ("aten.isnan.default", [("isnan", EQ)]),
    ("aten.maximum.default", [("maximum", EQ)]),
    ("aten.chunk.default", [("chunk", EQ)]),
    ("aten.arange.start", [("arange", EQ)]),
    ("aten.full.default", [("full", EQ)]),
    ("aten.ones_like.default", [("ones_like", EQ)]),
    ("aten.layer_norm.default", [("layer_norm", EQ)]),
    ("aten.rms_norm.default", [("rms_norm", EQ)]),
    ("aten.group_norm.default", [("group_norm", EQ)]),
    ("aten.scatter.src", [("scatter", EQ)]),
    ("aten.gather.default", [("gather", EQ)]),
    ("aten.bitwise_and.Tensor", [("bitwise_and", EQ)]),
    ("aten.divide.Tensor", [("divide", EQ)]),
    ("aten.scaled_dot_product_attention.default", [("transformer.scaled_dot_product_attention", EQ)]),
    ("aten.dropout.default", [("experimental.dropout", EQ)]),
    # in-place spellings: the in-place ttnn name first, then the plain one
    ("aten.add_.Tensor", [("add_", EQ), ("add", EQ)]),
    ("aten.sub_.Tensor", [("sub_", EQ), ("sub", EQ), ("subtract_", SY)]),
    ("aten.mul_.Tensor", [("mul_", EQ), ("mul", EQ), ("multiply_", SY)]),
    ("aten.sigmoid_.default", [("sigmoid", EQ)]),
    ("aten.silu_.default", [("silu", EQ)]),
    ("aten.fill_.Scalar", [("fill", EQ)]),
    ("aten.masked_fill_.Scalar", [("where", SY)]),
    ("aten.bernoulli_.float", [("bernoulli", EQ)]),
    # an in-place ttnn name is not proposed for an op that is not in place
    ("aten.logical_and.default", [("logical_and", EQ)]),
    ("aten.__or__.Tensor", [("bitwise_or", SY), ("logical_or", SY)]),
    # synonyms
    ("aten.clip.default", [("clip", EQ), ("clamp", SY)]),
    ("aten.clamp_.default", [("clamp", EQ), ("clip", SY)]),
    ("aten.floor_divide.default", [("floor_div", SY)]),
    ("aten.rsub.Tensor", [("rsub", EQ), ("subtract", SY)]),
    ("aten.mm.default", [("matmul", SY)]),
    ("aten.bmm.default", [("moreh_bmm", EQ), ("matmul", SY)]),
    ("aten.native_layer_norm.default", [("layer_norm", SY)]),
    ("aten._softmax.default", [("softmax", EQ)]),
    ("aten.log_softmax.int", [("moreh_logsoftmax", EQ)]),
    ("aten.index_select.default", [("embedding", SY), ("gather", SY)]),
    ("aten.flatten.using_ints", [("reshape", SY)]),
    ("aten.t.default", [("transpose", SY)]),
    ("aten.new_zeros.default", [("zeros", SY), ("full", SY)]),
    # qualifiers on the aten side
    ("aten.new_empty.default", [("empty", TS)]),
    ("aten.clamp_min.default", [("clamp", TS)]),
    # spelling slips are `near`
    ("aten.softmaxx", [("softmax", NE)]),
    ("aten.sigmoidd", [("sigmoid", NE)]),
    ("aten.group_normm.default", [("group_norm", NE)]),
    # nothing close
    ("aten.all.default", []),
    ("aten.any.default", []),
    ("aten.unbind.int", []),
    ("aten.sym_size.int", []),
    ("aten.index_put_.default", []),
    ("aten.nan_to_num.default", []),
    ("aten.var_mean.correction", []),
    ("aten.argsort.default", []),
]

# Names that share a token or most of a spelling with a ttnn op that does a different job.
NEVER = [
    ("aten.diff.default", "squared_difference"),          # shares a stem, is not it
    ("aten.diff.default", "experimental.fast_reduce_nc"),
    ("aten.index.Tensor", "index_fill"),                  # the token `index` alone is not enough
    ("aten.index.Tensor", "indexed_fill"),
    ("aten.flip.default", "clip"),                        # one letter apart, a different op
    ("aten.slice_scatter.default", "scatter"),            # ends with `scatter`, is not a scatter
    ("aten.select_scatter.default", "scatter"),
    ("aten.masked_scatter.default", "scatter"),
    ("aten.embedding_bag.default", "embedding"),          # starts with `embedding`, sums bags
    ("aten.all.default", "all_gather"),                   # the collective, not the reduction
    ("aten.all.default", "experimental.all_reduce_async"),
    ("aten.reduce_scatter.default", "reduce_scatter_minimal_async"),
    ("aten.to.device", "typecast"),                       # a device move casts nothing
    ("aten.to.dtype_layout", "to_device"),
    ("aten.einsum.default", "matmul"),                    # no table rule for an arbitrary equation
    ("aten.rsqrt.default", "sqrt"),                       # an exact name beats a one-letter neighbour
    ("aten.sin.default", "sinh"),
    ("aten.cos.default", "cosh"),
    ("aten.add.Tensor", "addalpha"),
    ("aten.silu.default", "silu_bw"),                     # a backward op is never a forward candidate
    ("aten.softmax.int", "softmax_in_place_bw"),
    ("aten.exp.default", "exp2"),
    ("aten.mean.default", "experimental.dit_fused_distributed_rmsnorm"),
    ("aten.empty.memory_format", "empty_like"),
    ("aten.fill.Scalar", "index_fill"),
    ("aten.fill_.Scalar", "fill_ones_rm"),
    ("aten.where.self", "scale_mask_softmax"),
    ("aten.select.int", "select"),                         # nothing in the dump is named select
    ("aten.lerp.Scalar", "lerp_bw"),
    ("aten.new_ones.default", "ones_like"),
    ("aten.new_ones.default", "fill_ones_rm"),
]


def test_the_table_has_at_least_sixty_cases():
    assert len(CASES) >= 60
    assert len({op for op, _ in CASES}) == len(CASES)


@pytest.mark.parametrize("op,expected", CASES)
def test_candidates_for_real_ops(op, expected):
    assert got(op) == expected


@pytest.mark.parametrize("op,never", NEVER)
def test_names_that_must_not_be_proposed(op, never):
    assert never not in [t for t, _ in got(op)]


def test_the_dump_in_the_tests_is_the_one_the_tool_ships_for_0_80_0():
    assert json.loads(DUMP.read_text())["ttnn_version"] == "0.80.0"
    assert "typecast" in NAMES and "to_device" in NAMES


def test_at_most_three_candidates_and_each_has_a_label():
    for op, _ in CASES:
        out = ov.candidates(op, IX)
        assert len(out) <= 3
        assert all(c["why"] in ov.WHY and isinstance(c["ttnn"], str) for c in out)
        assert len({c["ttnn"] for c in out}) == len(out)


def test_answers_do_not_depend_on_the_order_of_the_names():
    shuffled = ov.Index(list(reversed(NAMES)))
    for op, _ in CASES:
        assert got(op, shuffled) == got(op)


# --- reading names -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,base,overload,inplace,toks", [
    ("aten.__and__.Tensor", "__and__", "Tensor", False, ["and"]),
    ("aten::div_.Tensor", "div", "Tensor", True, ["div"]),
    ("aten.to.dtype_layout", "to", "dtype_layout", False, ["to"]),
    ("aten.split_with_sizes.default", "split_with_sizes", "", False, ["split", "with", "sizes"]),
    ("aten._softmax.default", "_softmax", "", False, ["softmax"]),
    ("aten.new_ones", "new_ones", "", False, ["new", "ones"]),
    ("aten.scaled_dot_product_attention.default", "scaled_dot_product_attention", "", False,
     ["scaled", "dot", "product", "attention"]),
    ("aten.sum.dim_IntList", "sum", "dim_IntList", False, ["sum"]),
    ("aten.relu_.default", "relu", "", True, ["relu"]),
])
def test_parse_aten(raw, base, overload, inplace, toks):
    a = ov.parse_aten(raw)
    assert (a["base"], a["overload"], a["inplace"], a["tokens"]) == (base, overload, inplace, toks)
    assert a["key"] == "".join(toks)


@pytest.mark.parametrize("name,toks,inplace,legacy,namespaced,blocked,skip", [
    ("rms_norm", ["rms", "norm"], False, False, False, False, False),
    ("experimental.rotary_embedding_hf", ["rotary", "embedding", "hf"], False, False, True, False, False),
    ("moreh_layer_norm", ["layer", "norm"], False, True, False, False, False),
    ("div_", ["div"], True, False, False, False, False),
    ("RMSNorm", ["rms", "norm"], False, False, False, False, True),
    ("CloseDevice", ["close", "device"], False, False, False, True, True),
    ("all_gather", ["all", "gather"], False, False, False, True, False),
    ("silu_bw", ["silu", "bw"], False, False, False, True, False),
    ("experimental.dit_rms_norm_unary_fused", ["rms", "norm", "unary", "fused"], False, True, True, False, False),
])
def test_parse_ttnn(name, toks, inplace, legacy, namespaced, blocked, skip):
    e = ov.parse_ttnn(name)
    assert e["tokens"] == toks
    assert (e["inplace"], e["legacy"], e["namespaced"], e["blocked"], e["skip"]) == \
        (inplace, legacy, namespaced, blocked, skip)


def test_rms_norm_rmsnorm_and_RMSNorm_share_one_key():
    keys = {ov.parse_ttnn(n)["key"] for n in ("rms_norm", "rmsnorm", "experimental.rms_norm")}
    assert keys == {"rmsnorm"}


def test_tokens_split_camel_case_and_keep_digits_with_letters():
    assert ov.tokens("scaledDotProduct") == ["scaled", "dot", "product"]
    assert ov.tokens("conv1d") == ["conv1d"]
    assert ov.tokens("HTTPServer") == ["http", "server"]


def test_distance():
    assert ov.distance("kitten", "sitting") == 3
    assert ov.distance("", "abc") == 3
    assert ov.distance("same", "same") == 0


# --- the tiers, on names written for the test ----------------------------------------------------------------------

def ix(*names):
    return ov.Index(list(names))


def test_an_exact_name_and_a_synonym_both_count_but_the_exact_name_is_first():
    out = ov.candidates("aten.clip.default", ix("clamp", "clip"))
    assert [(c["ttnn"], c["why"]) for c in out] == [("clip", EQ), ("clamp", SY)]


def test_a_close_name_is_not_listed_beside_an_exact_one():
    assert got("aten.cos.default", ix("cos", "cosh", "cox")) == [("cos", EQ)]


def test_token_subset_in_both_directions_with_qualifier_tokens_only():
    assert got("aten.and", ix("bitwise_and")) == [("bitwise_and", TS)]
    assert got("aten.new_thing.default", ix("thing")) == [("thing", TS)]
    # an extra token that changes the job (`fill`, `async`) is not a qualifier
    assert got("aten.index.Tensor", ix("index_fill")) == []
    assert got("aten.and", ix("and_async")) == []


def test_token_subset_needs_a_strict_subset():
    assert got("aten.qwerty.default", ix("qwerty_other_thing")) == []


def test_near_allows_one_letter_for_short_keys_and_two_for_long_keys():
    assert got("aten.sigmoidd", ix("sigmoid")) == [("sigmoid", NE)]
    assert got("aten.sigmoidxxx", ix("sigmoid")) == []
    assert got("aten.layernormalizationn", ix("layernormalization")) == [("layernormalization", NE)]
    assert got("aten.layernormalizationnn", ix("layernormalization")) == [("layernormalization", NE)]
    assert got("aten.layernormalizationnnn", ix("layernormalization")) == []


def test_near_is_not_tried_for_keys_under_five_letters():
    assert got("aten.flip.default", ix("clip")) == []
    assert got("aten.flipp.default", ix("flip")) == []        # four-letter ttnn name
    assert got("aten.flippo.default", ix("flippy")) == [("flippy", NE)]


def test_affix_is_a_start_or_an_end_with_at_most_two_more_letters():
    assert got("aten.rsqrtxy", ix("rsqrt")) == [("rsqrt", AF)]
    assert got("aten.xyrsqrt", ix("rsqrt")) == [("rsqrt", AF)]
    assert got("aten.rsqrtxyz", ix("rsqrt")) == []          # three more letters
    assert got("aten.rsqrt_xyz", ix("rsqrt")) == []
    assert got("aten.sqrtxy", ix("sqrt")) == []            # the shorter key has four letters
    assert got("aten.rsxqrtx", ix("rsqrt")) == []          # inside, not at an end, and two letters off


def test_blocked_tokens_are_left_out_of_the_weak_tiers_only():
    assert got("aten.gather.default", ix("all_gather", "gather")) == [("gather", EQ)]
    assert got("aten.all.default", ix("all_gather")) == []
    # a synonym target may carry a blocked token on purpose (`to_device`)
    assert got("aten.to.device", ix("to_device")) == [("to_device", SY)]


def test_camel_case_api_names_are_never_proposed():
    assert got("aten.close_device", ix("CloseDevice")) == []


def test_synonym_lookup_prefers_the_overload_entry_then_the_base():
    index = ix("typecast", "to_device")
    assert got("aten.to.device", index) == [("to_device", SY)]
    assert got("aten.to.dtype", index) == [("typecast", SY)]
    assert got("aten.to.something_else", index) == [("typecast", SY)]     # the base entry


def test_each_synonym_target_is_listed_before_a_second_spelling_of_the_first():
    out = ov.candidates("aten.new_ones.default", ix("ones", "ones_", "full"))
    assert [c["ttnn"] for c in out][:2] == ["ones", "full"]


def test_a_legacy_prefix_loses_to_the_plain_name_and_is_used_when_alone():
    assert got("aten.mean.default", ix("mean", "moreh_mean")) == [("mean", EQ)]
    assert got("aten.dot.default", ix("moreh_dot")) == [("moreh_dot", EQ)]


def test_a_legacy_name_ranks_below_a_namespaced_one():
    assert got("aten.mean.default", ix("experimental.mean", "moreh_mean")) == [("experimental.mean", EQ)]


def test_a_blocked_token_keeps_a_near_name_out_too():
    assert got("aten.sigmoidb", ix("sigmoid_bw")) == []
    assert got("aten.sigmoidb", ix("sigmoid_bw", "sigmoid")) == [("sigmoid", NE)]


def test_near_needs_five_letters_on_the_aten_side_as_well():
    assert got("aten.flip", ix("flipp")) == []


def test_two_synonym_targets_each_with_an_in_place_spelling_alternate_in_the_answer():
    out = got("aten.copy_.default", ix("assign", "assign_", "clone", "clone_"))
    assert out == [("assign_", SY), ("clone_", SY), ("assign", SY)]


def test_an_empty_name_proposes_nothing_even_for_names_made_of_qualifiers():
    assert got("aten.", ix("default", "scalar")) == []
    assert ov.candidates("", ix("default")) == []


def test_a_tier_three_hit_is_not_topped_up_with_near_names():
    assert got("aten.new_thing.default", ix("thing", "newthingx")) == [("thing", TS)]


def test_reverse_subset_needs_a_ttnn_key_of_three_letters():
    assert got("aten.new_ab.default", ix("ab")) == []
    assert got("aten.new_abc.default", ix("abc")) == [("abc", TS)]


def test_a_leading_underscore_is_dropped_to_find_a_synonym():
    assert got("aten._clip.default", ix("clamp")) == [("clamp", SY)]


def test_a_plain_name_beats_a_namespaced_one():
    assert got("aten.typecast", ix("experimental.typecast", "typecast")) == [("typecast", EQ)]
    assert got("aten.typecast", ix("experimental.typecast")) == [("experimental.typecast", EQ)]


def test_nothing_for_an_empty_dump_or_an_empty_name():
    assert got("aten.add.Tensor", ix()) == []
    assert ov.candidates("aten.", ix("add")) == []


# --- the data file -------------------------------------------------------------------------------------------------

def test_every_synonym_has_a_target_in_the_shipped_dump():
    d = ov.load_data()
    keys = {e["key"] for e in IX.entries}
    for src, targets in d["synonyms"].items():
        assert targets and all(isinstance(t, str) for t in targets), src
        assert any("".join(ov.tokens(t)) in keys for t in targets), f"{src}: no target in the dump"


def test_the_data_file_has_every_key_the_code_reads():
    d = ov.load_data()
    for k in ("ttnn_noise_prefixes", "ttnn_blocked_tokens", "ttnn_qualifier_tokens", "aten_qualifier_tokens",
              "near_min_length", "near_max_distance_short", "near_max_distance_long", "near_long_from_length",
              "affix_min_length", "affix_max_extra", "max_candidates", "synonyms"):
        assert k in d
    assert d["max_candidates"] == 3
