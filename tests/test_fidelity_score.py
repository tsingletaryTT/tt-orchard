# SPDX-License-Identifier: Apache-2.0
"""scripts/fidelity_score.py: the score arithmetic, the simulated formats and the noise note.

The model run itself (sampling, two forward passes) needs a downloaded checkpoint and is covered by a
recorded run in docs/run-logs; these tests cover the parts that decide what the number means.
"""
import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "fidelity_score", Path(__file__).resolve().parent.parent / "scripts" / "fidelity_score.py")
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)


def test_identical_models_score_exactly_ten():
    lp = [-1.0, -2.5, -0.1, -3.0]
    r = fs.fidelity_from_logprobs(lp, list(lp))
    assert r["fidelity"] == pytest.approx(10.0)


def test_a_candidate_that_is_worse_on_the_reference_text_scores_below_ten_by_exp_of_the_gap():
    ref = [-1.0] * 100
    cand = [-1.2] * 100
    # PPL ratio is exp(-mean ref) / exp(-mean cand) = exp(mean cand - mean ref) = exp(-0.2)
    assert fs.fidelity_from_logprobs(ref, cand)["fidelity"] == pytest.approx(10 * math.exp(-0.2))


def test_the_score_uses_the_reference_as_numerator():
    # A candidate that scores the sampled tokens higher than the reference gets more than ten.
    assert fs.fidelity_from_logprobs([-2.0] * 10, [-1.0] * 10)["fidelity"] > 10.0


@pytest.mark.parametrize("a,b", [([], []), ([-1.0], [-1.0, -2.0])])
def test_empty_or_unequal_lists_are_refused(a, b):
    with pytest.raises(ValueError):
        fs.fidelity_from_logprobs(a, b)


def test_noise_note_appears_below_the_token_floor_and_not_above():
    assert "noise" in fs.noise_note(fs.MIN_TOKENS_FOR_CLAIM - 1)
    assert fs.noise_note(fs.MIN_TOKENS_FOR_CLAIM) is None


def _weight(seed=0, shape=(64, 300)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


def test_none_returns_the_same_tensor():
    w = _weight()
    assert fs.quantize_tensor(w, "none") is w


def test_formats_keep_shape_and_dtype_including_a_width_that_is_not_a_multiple_of_the_group():
    w = _weight(shape=(8, 300)).to(torch.bfloat16)
    for kind in ("fp8", "int8", "int4", "bfp8", "bfp4"):
        q = fs.quantize_tensor(w, kind)
        assert q.shape == w.shape and q.dtype == w.dtype, kind


def test_error_grows_as_the_format_gets_coarser():
    w = _weight()
    err = {k: (fs.quantize_tensor(w, k) - w).pow(2).mean().item() for k in ("int8", "bfp8", "int4", "bfp4")}
    assert err["int8"] < err["int4"]
    assert err["bfp8"] < err["bfp4"]
    assert err["int8"] > 0 and err["bfp8"] > 0


def test_int4_has_at_most_fifteen_levels_per_group():
    w = _weight(shape=(4, 128))
    q = fs.quantize_tensor(w, "int4")
    for row in q:
        assert row.unique().numel() <= 15


def test_block_float_values_in_a_block_share_one_step_size():
    w = _weight(shape=(2, 16))
    q = fs.quantize_tensor(w, "bfp4")
    # With 3 mantissa bits there are at most 2*7+1 levels per block.
    assert q[0].unique().numel() <= 15


def test_an_unknown_format_and_a_non_2d_weight_are_refused_by_name():
    with pytest.raises(ValueError, match="unknown format"):
        fs.quantize_tensor(_weight(), "fp3")
    with pytest.raises(ValueError, match="2-D"):
        fs.quantize_tensor(torch.zeros(3), "int8")


def test_quantize_model_rounds_linear_layers_but_not_the_head_or_embeddings():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 32)
            self.proj = torch.nn.Linear(32, 32, bias=False)
            self.lm_head = torch.nn.Linear(32, 10, bias=False)

    m = M()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    n = fs.quantize_model(m, "int4")
    assert n == 1
    assert torch.equal(m.embed.weight, before["embed.weight"])
    assert torch.equal(m.lm_head.weight, before["lm_head.weight"])
    assert not torch.equal(m.proj.weight, before["proj.weight"])
