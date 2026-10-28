# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/first_divergence.py: find the first block, and the first operation inside it, where a candidate run leaves a tolerance around a
reference run. Every run here is built from small random arrays in a temporary directory, with a divergence planted at a chosen block and
operation, so the tests know the right answer. No model and no chip."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from orchard import audio_reference as ar
from orchard import first_divergence as fd

SETTINGS = {"prompt": "a piano", "duration": 10, "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong", "seed": 42,
            "dtype": "float32", "sample_size": 705600}
MODEL = {"id": "org/model", "revision": "a" * 40}
LIB = {"name": "lib", "commit": "b" * 40}
OPS = ("attn", "cross", "ff")
ROWS, CH, MEM = 40, 16, 6          # the first MEM rows are "memory" rows with a large norm


def _write(path, arrays, dtype="float32", model=None, library=None):
    rec = ar.Recorder(path, model=model or MODEL, library=library or LIB, settings={**SETTINGS, "dtype": dtype}, seeds={"torch": 42})
    for (kind, name), arr in arrays:
        rec.record(kind, name, arr)
    rec.finish()
    return path


def build(tmp_path, *, n_blocks=8, plant=None, plant_rows=None, plant_size=0.05, jitter=1e-4, with_ops=True, dip_after=None):
    """A reference run, a candidate run and a pair of environment-floor runs. `plant` is (block, op): from that operation on, the candidate's
    stream carries a fixed extra error of relative size `plant_size` on `plant_rows` (all rows when None). `jitter` is the small error that
    every candidate tensor carries. `dip_after`: the planted error disappears again after that block (a non-monotone stack)."""
    rng = np.random.default_rng(7)
    scale = np.ones((ROWS, 1))
    scale[:MEM] = 10.0
    stream = rng.normal(size=(ROWS, CH)) * scale
    ref, chip, floor_b = [], [], []
    delta = None
    for b in range(n_blocks):
        for op in OPS:
            stream = stream + 0.2 * rng.normal(size=(ROWS, CH)) * scale
            if plant == (b, op):
                delta = np.zeros((ROWS, CH))
                rows = slice(None) if plant_rows is None else plant_rows
                delta[rows] = (rng.normal(size=(ROWS, CH)) * scale * plant_size)[rows]
            if dip_after is not None and b > dip_after:
                delta = None
            cur = stream + (delta if delta is not None else 0)
            name = f"layer_{b:02d}.{op}"
            ref.append((("layer", name), stream[None]))
            chip.append((("layer", name), (cur + jitter * rng.normal(size=(ROWS, CH)) * scale)[None]))
            floor_b.append((("layer", name), (stream + jitter * rng.normal(size=(ROWS, CH)) * scale)[None]))
        # the block output is the stream after its last operation
        ref.append((("layer", f"layer_{b:02d}"), stream[None]))
        chip.append((("layer", f"layer_{b:02d}"), (cur + jitter * rng.normal(size=(ROWS, CH)) * scale)[None]))
        floor_b.append((("layer", f"layer_{b:02d}"), (stream + jitter * rng.normal(size=(ROWS, CH)) * scale)[None]))
    if not with_ops:
        ref = [x for x in ref if "." not in x[0][1]]
        chip = [x for x in chip if "." not in x[0][1]]
        floor_b = [x for x in floor_b if "." not in x[0][1]]
    final = ("latent", "final")
    ref.append((final, stream[None][:, :4]))
    chip.append((final, (cur + jitter * rng.normal(size=(ROWS, CH)) * scale)[None][:, :4]))
    floor_b.append((final, stream[None][:, :4]))
    return (_write(tmp_path / "ref", ref), _write(tmp_path / "chip", chip, dtype="bfloat16"), _write(tmp_path / "env_b", floor_b))


def run(tmp_path, **kw):
    r, c, f = build(tmp_path, **kw)
    return ar.load_run(r), ar.load_run(c), ar.load_run(f)


# ---- finding a planted divergence ------------------------------------------------------------------------------------

def test_a_planted_divergence_is_found_at_the_chosen_block_and_operation(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(5, "cross"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["located"] is True
    assert rep["first_block"]["name"] == "layer_05"
    assert rep["first_op"]["name"] == "layer_05.cross"
    assert rep["bisect_agrees_with_scan"] is True
    text = "\n".join(fd.summarise(rep))
    assert "layer_05" in text and "layer_05.cross" in text


def test_every_block_and_operation_is_tried_in_turn_not_just_the_last(tmp_path):
    for block, op in ((0, "attn"), (3, "ff"), (7, "attn")):
        ref, cand, _ = run(tmp_path / f"{block}{op}", plant=(block, op))
        rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
        assert (rep["first_block"]["name"], rep["first_op"]["name"]) == (f"layer_{block:02d}", f"layer_{block:02d}.{op}")


def test_a_clean_candidate_reports_no_divergence(tmp_path):
    ref, cand, _ = run(tmp_path)
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["located"] is False and rep["first_block"] is None and rep["first_op"] is None
    assert "within" in " ".join(fd.summarise(rep))


def test_the_bisect_reads_few_blocks_when_the_table_is_not_wanted(tmp_path):
    ref, cand, _ = run(tmp_path, n_blocks=16, plant=(11, "attn"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01), table=False)
    assert rep["first_block"]["name"] == "layer_11"
    assert len(rep["bisect"]["block_probes"]) <= 5            # log2(16) + the check of the last block
    assert rep["blocks"] is None or len(rep["blocks"]) < 16


def test_a_candidate_that_is_back_within_tolerance_later_is_not_missed_by_the_scan(tmp_path):
    """The error leaves tolerance at block 2 and is gone by block 5. A bisect that starts at the last block would see nothing; the scan finds
    block 2 and the report says the two methods disagree."""
    ref, cand, _ = run(tmp_path, plant=(2, "attn"), dip_after=3)
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["first_block"]["name"] == "layer_02"
    assert rep["bisect_agrees_with_scan"] is False
    assert any("not monotone" in w for w in rep["warnings"])


def test_when_the_first_block_already_leaves_the_tolerance_it_is_the_first(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(0, "attn"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["first_block"]["name"] == "layer_00"


def test_a_block_with_no_recorded_operations_is_located_and_the_report_says_it_cannot_go_deeper(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(4, "ff"), with_ops=False)
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["first_block"]["name"] == "layer_04" and rep["first_op"] is None
    assert any("no operations recorded" in n for n in rep["notes"])


def test_operations_are_taken_in_the_order_they_were_recorded_unless_an_order_is_given(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(2, "cross"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert [o["name"] for o in rep["ops"]] == ["layer_02.attn", "layer_02.cross", "layer_02.ff"]
    # An explicit order that puts the cross operation first: the planted error lives in every later tensor, so cross is now the first that leaves.
    rep2 = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01), ops_order={"layer_02": ["cross", "ff", "attn"]})
    assert [o["name"] for o in rep2["ops"]][0] == "layer_02.cross"
    assert rep2["first_op"]["name"] == "layer_02.cross"


# ---- the numbers for each tensor -------------------------------------------------------------------------------------

def test_each_tensor_carries_pcc_relative_l2_slope_and_the_worst_rows_and_channels(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(5, "cross"), plant_rows=slice(20, 22))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.001))
    e = next(x for x in rep["blocks"] if x["name"] == "layer_05")
    for key in ("pcc", "rel_l2", "slope", "max_abs_error", "worst_rows", "worst_channels"):
        assert key in e
    assert {r["row"] for r in e["worst_rows"][:2]} == {20, 21}
    assert len(e["worst_rows"]) <= 5 and all("err_norm" in r and "rel_err" in r and "ref_norm" in r for r in e["worst_rows"])
    assert all(0 <= c["channel"] < CH for c in e["worst_channels"])


def test_the_worst_channel_is_the_planted_one(tmp_path):
    ref, cand, _ = run(tmp_path)
    arr = ref.array("layer", "layer_03").copy()
    bad = arr.copy()
    bad[..., 9] += 5.0
    w = fd.tensor_report(arr, bad, top=3)
    assert w["worst_channels"][0]["channel"] == 9


def test_a_planted_error_on_the_high_norm_rows_is_reported_as_such(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(3, "attn"), plant_rows=slice(0, MEM))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.001))
    e = next(x for x in rep["blocks"] if x["name"] == "layer_03")
    hn = e["high_norm"]
    assert hn["rows"] == MEM and hn["of"] == ROWS
    assert hn["error_share"] > 0.9 and e["error_mostly_in_high_norm_rows"] is True
    assert hn["rel_l2"] > 10 * hn["rest_rel_l2"]
    assert any("high-norm rows" in line and "squared error" in line for line in fd.summarise(rep))


def test_an_error_on_the_ordinary_rows_is_not_blamed_on_the_high_norm_rows(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(3, "attn"), plant_rows=slice(MEM, ROWS))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.001))
    e = next(x for x in rep["blocks"] if x["name"] == "layer_03")
    assert e["high_norm"]["error_share"] < 0.2 and e["error_mostly_in_high_norm_rows"] is False


def test_an_array_whose_rows_have_equal_norms_has_no_high_norm_group(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, 30, 8))
    rep = fd.tensor_report(a, a * 1.01)
    assert rep["high_norm"] is None and rep["error_mostly_in_high_norm_rows"] is False


def test_padding_rows_of_zeros_do_not_make_every_row_look_high_norm():
    a = np.zeros((1, 20, 4))
    a[0, :6] = np.where(np.random.default_rng(1).normal(size=(6, 4)) > 0, 1.0, -1.0)
    rep = fd.tensor_report(a, a + 0.001)
    assert rep["high_norm"] is None


def test_a_one_dimensional_array_is_one_row():
    rep = fd.tensor_report(np.arange(1.0, 9.0), np.arange(1.0, 9.0) * 1.1)
    assert rep["comparable"] and rep["worst_rows"][0]["row"] == 0


def test_a_scaled_candidate_shows_in_the_slope(tmp_path):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(1, 20, 8))
    rep = fd.tensor_report(a, a * 1.2)
    assert rep["pcc"] == pytest.approx(1.0) and rep["slope"] == pytest.approx(1.2)


# ---- the environment floor next to each number -----------------------------------------------------------------------

def test_the_environment_floor_is_printed_next_to_each_number_and_sets_the_default_tolerance(tmp_path):
    ref, cand, env_b = run(tmp_path, plant=(5, "cross"), jitter=1e-3)
    rep = fd.find_first_divergence(ref, cand, floor=(ref, env_b), tolerance=fd.Tolerance(margin=3.0))
    e = next(x for x in rep["blocks"] if x["name"] == "layer_02")
    assert e["floor"]["rel_l2"] > 0 and e["tolerance_source"] == "3.0 times the environment floor"
    assert e["tolerance"] == pytest.approx(3.0 * e["floor"]["rel_l2"])
    assert e["ratio_to_floor"] == pytest.approx(e["rel_l2"] / e["floor"]["rel_l2"])
    assert rep["first_block"]["name"] == "layer_05"
    assert any("floor" in line for line in fd.summarise(rep))


def test_a_stated_tolerance_wins_over_the_floor(tmp_path):
    ref, cand, env_b = run(tmp_path, plant=(5, "cross"))
    rep = fd.find_first_divergence(ref, cand, floor=(ref, env_b), tolerance=fd.Tolerance(rel_l2=0.01))
    e = rep["blocks"][0]
    assert e["tolerance"] == 0.01 and e["tolerance_source"] == "stated" and e["floor"] is not None


def test_without_a_floor_or_a_stated_tolerance_the_fallback_is_named(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(5, "cross"))
    rep = fd.find_first_divergence(ref, cand)
    e = rep["blocks"][0]
    assert e["floor"] is None and e["tolerance_source"].startswith("fallback")
    assert any("no environment floor" in line for line in fd.summarise(rep))


def test_a_floor_pair_that_is_not_the_same_setup_is_refused(tmp_path):
    ref, cand, env_b = run(tmp_path)
    other = _write(tmp_path / "other", [(("layer", "layer_00"), np.zeros((1, 4, 4)))])
    # same settings, so this loads; change the seed to make it differ
    rec = ar.Recorder(tmp_path / "seed", model=MODEL, library=LIB, settings={**SETTINGS, "seed": 1}, seeds={})
    rec.record("layer", "layer_00", np.ones((1, 4, 4)))
    rec.finish()
    with pytest.raises(fd.DivergenceError, match="seed"):
        fd.find_first_divergence(ref, cand, floor=(ref, ar.load_run(tmp_path / "seed")))


# ---- refusals and awkward input ------------------------------------------------------------------------------------

def test_runs_with_different_settings_are_refused(tmp_path):
    ref, _, _ = run(tmp_path)
    rec = ar.Recorder(tmp_path / "x", model=MODEL, library=LIB, settings={**SETTINGS, "steps": 9}, seeds={})
    rec.record("layer", "layer_00", np.ones((1, 4, 4)))
    rec.finish()
    with pytest.raises(fd.DivergenceError, match="steps"):
        fd.find_first_divergence(ref, ar.load_run(tmp_path / "x"))


def test_a_block_the_candidate_never_recorded_is_listed_and_not_silently_passed(tmp_path):
    ref, _, _ = run(tmp_path)
    arrays = [((k, n), ref.array(k, n)) for (k, n) in ref.entries if n != "layer_03" and not n.startswith("layer_03.")]
    cand = ar.load_run(_write(tmp_path / "short", arrays, dtype="bfloat16"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert any("layer_03" in p and "candidate" in p for p in rep["problems"])
    assert rep["complete"] is False
    assert "layer_03" not in [b["name"] for b in rep["blocks"]]


def test_a_non_finite_candidate_block_leaves_the_tolerance(tmp_path):
    ref, _, _ = run(tmp_path)
    arrays = [((k, n), ref.array(k, n)) for (k, n) in ref.entries]
    arrays = [((k, n), (np.where(np.arange(a.size).reshape(a.shape) == 5, np.nan, a) if n == "layer_04" else a)) for (k, n), a in arrays]
    cand = ar.load_run(_write(tmp_path / "nan", arrays, dtype="bfloat16"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["first_block"]["name"] == "layer_04" and rep["first_block"]["comparable"] is False
    assert "not finite" in rep["first_block"]["reason"]
    json.dumps(rep, allow_nan=False)


def test_a_shape_mismatch_leaves_the_tolerance_with_the_reason(tmp_path):
    ref, _, _ = run(tmp_path)
    arrays = [((k, n), (np.zeros((1, 3, 3)) if n == "layer_02" else ref.array(k, n))) for (k, n) in ref.entries]
    cand = ar.load_run(_write(tmp_path / "shape", arrays, dtype="bfloat16"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["first_block"]["name"] == "layer_02" and "shapes differ" in rep["first_block"]["reason"]


def test_a_run_with_no_blocks_of_the_asked_kind_is_refused(tmp_path):
    ref, cand, _ = run(tmp_path)
    with pytest.raises(fd.DivergenceError, match="no block"):
        fd.find_first_divergence(ref, cand, stack=fd.Stack(kind="step_latent"))


def test_a_bad_tolerance_or_pattern_is_refused(tmp_path):
    for bad in (dict(rel_l2=-1.0), dict(rel_l2=float("nan")), dict(margin=0.5), dict(rel_l2=True)):
        with pytest.raises(fd.DivergenceError):
            fd.Tolerance(**bad)
    with pytest.raises(fd.DivergenceError):
        fd.Stack(block_re="(")


def test_blocks_are_ordered_by_number_not_by_text(tmp_path):
    rng = np.random.default_rng(0)
    names = [f"layer_{i}" for i in (2, 10, 1, 9)]
    ref = ar.load_run(_write(tmp_path / "r", [(("layer", n), rng.normal(size=(1, 4, 4))) for n in names]))
    cand = ar.load_run(_write(tmp_path / "c", [(("layer", n), ref.array("layer", n)) for n in names], dtype="bfloat16"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01), stack=fd.Stack(block_re=r"layer_(\d+)"))
    assert [b["name"] for b in rep["blocks"]] == ["layer_1", "layer_2", "layer_9", "layer_10"]


def test_arrays_held_in_memory_work_as_a_run(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(2, "ff"))
    mem_ref = fd.MemoryRun({k: ref.array(*k) for k in ref.entries})
    mem_cand = fd.MemoryRun({k: cand.array(*k) for k in cand.entries})
    rep = fd.find_first_divergence(mem_ref, mem_cand, tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["first_op"]["name"] == "layer_02.ff"


def test_other_tensors_are_reported_with_the_same_numbers(tmp_path):
    ref, cand, env_b = run(tmp_path, plant=(5, "cross"))
    rep = fd.find_first_divergence(ref, cand, floor=(ref, env_b), tolerance=fd.Tolerance(rel_l2=0.01))
    final = next(t for t in rep["tensors"] if t["name"] == "final")
    assert final["kind"] == "latent" and "rel_l2" in final and final["floor"] is not None


# ---- report and command line ------------------------------------------------------------------------------------

def test_the_report_is_plain_json(tmp_path):
    ref, cand, env_b = run(tmp_path, plant=(5, "cross"))
    rep = fd.find_first_divergence(ref, cand, floor=(ref, env_b), tolerance=fd.Tolerance(rel_l2=0.01))
    again = json.loads(json.dumps(rep, allow_nan=False))
    assert again["first_block"]["name"] == "layer_05"


def test_the_command_line_prints_a_summary_and_exits_1_when_a_divergence_is_located(tmp_path, capsys):
    r, c, f = build(tmp_path, plant=(5, "cross"))
    import io
    out = io.StringIO()
    code = fd.main([str(r), str(c), "--floor-ref", str(r), "--floor-other", str(f), "--rel-l2", "0.01"], out=out)
    text = out.getvalue()
    assert code == 1 and "layer_05.cross" in text


def test_the_command_line_writes_json_and_exits_0_for_a_clean_candidate(tmp_path):
    import io
    r, c, f = build(tmp_path)
    out = io.StringIO()
    target = tmp_path / "report.json"
    code = fd.main([str(r), str(c), "--rel-l2", "0.01", "--json", "--out", str(target)], out=out)
    assert code == 0
    assert json.loads(target.read_text())["located"] is False
    assert json.loads(out.getvalue())["located"] is False


def test_the_command_line_refuses_a_missing_run_with_exit_2(tmp_path):
    import io
    out = io.StringIO()
    assert fd.main([str(tmp_path / "nope"), str(tmp_path / "nada")], out=out) == 2
    assert out.getvalue().startswith("refused:")


def test_the_command_line_refuses_to_replace_an_existing_report_file(tmp_path):
    import io
    r, c, _ = build(tmp_path)
    target = tmp_path / "report.json"
    target.write_text("keep")
    assert fd.main([str(r), str(c), "--rel-l2", "0.01", "--out", str(target)], out=io.StringIO()) == 2
    assert target.read_text() == "keep"


BANNED = [r", not ", r"rather than", r"instead of", r"reads as", r"honest", r"\bthe one\b", r"which is why", r"This is what"]


def test_the_summary_text_avoids_the_phrases_the_project_does_not_use(tmp_path):
    ref, cand, env_b = run(tmp_path, plant=(5, "cross"), plant_rows=slice(0, MEM))
    for kw in (dict(floor=(ref, env_b)), dict()):
        rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01), **kw)
        text = "\n".join(fd.summarise(rep))
        for pat in BANNED:
            assert not re.search(pat, text), (pat, text)
    src = Path(fd.__file__).read_text()
    for pat in BANNED:
        assert not re.search(pat, src, flags=re.I), pat


def test_the_module_reads_recorded_runs_only_through_audio_reference():
    src = Path(fd.__file__).read_text()
    assert not re.search(r"\bjson\.loads?\(", src)
    assert "audio_reference" in src


# ---- named row groups, a zero floor, the length of the summary ----------------------------------------------------

def test_a_named_row_group_reports_its_own_error_and_share(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(3, "attn"), plant_rows=slice(0, MEM))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.001), row_groups={"memory": (0, MEM), "latent": (MEM, ROWS), "first": (0, 2)})
    e = next(x for x in rep["blocks"] if x["name"] == "layer_03")
    groups = {g["name"]: g for g in e["row_groups"]}
    assert groups["memory"]["energy_share"] > 0.9 and groups["latent"]["energy_share"] < 0.1
    assert groups["memory"]["error_share"] > 0.9 and groups["latent"]["error_share"] < 0.1
    assert groups["memory"]["rel_l2"] > 10 * groups["latent"]["rel_l2"]
    assert groups["memory"]["rows"] == [0, MEM]
    final = next(t for t in rep["tensors"] if t["name"] == "final")
    assert final["row_groups"] == []                       # row groups describe a block's layout and are not applied to other arrays
    assert any("rows memory (0 to 5)" in line for line in fd.summarise(rep))


def test_the_error_share_of_a_row_group_is_not_its_energy_share(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(3, "attn"), plant_rows=slice(MEM, ROWS))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.001), row_groups={"memory": (0, MEM), "latent": (MEM, ROWS)})
    e = next(x for x in rep["blocks"] if x["name"] == "layer_03")
    latent = next(g for g in e["row_groups"] if g["name"] == "latent")
    assert latent["error_share"] > 0.9 and latent["energy_share"] < 0.1


def test_a_row_group_that_does_not_fit_is_refused_or_skipped(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(3, "attn"))
    for bad in ({"x": (5, 5)}, {"x": (-1, 3)}, {"x": (0,)}, {1: (0, 2)}, {"x": (0, True)}):
        with pytest.raises(fd.DivergenceError, match="row group"):
            fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.001), row_groups=bad)
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.001), row_groups={"beyond": (ROWS, ROWS + 9)})
    assert rep["blocks"][0]["row_groups"] == []


def test_the_command_line_takes_row_groups(tmp_path):
    import io
    r, c, f = build(tmp_path, plant=(5, "cross"), plant_rows=slice(0, MEM))
    out = io.StringIO()
    assert fd.main([str(r), str(c), "--rel-l2", "0.01", "--rows", f"memory=0:{MEM}"], out=out) == 1
    assert "rows memory" in out.getvalue()
    out2 = io.StringIO()
    assert fd.main([str(r), str(c), "--rows", "memory"], out=out2) == 2 and "NAME=START:STOP" in out2.getvalue()


def test_two_environments_that_agree_exactly_give_a_tiny_tolerance_and_say_so(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(5, "cross"), jitter=1e-3)
    same = fd.MemoryRun({k: ref.array(*k) for k in ref.entries}, settings=ref.settings)
    rep = fd.find_first_divergence(ref, cand, floor=(ref, same), tolerance=fd.Tolerance(margin=3.0))
    e = rep["blocks"][0]
    assert e["floor"]["rel_l2"] == 0.0 and e["tolerance"] == pytest.approx(3.0 * fd.MIN_FLOOR)
    assert "agree exactly" in e["tolerance_source"] and e["leaves"] is True


def test_the_summary_lists_only_the_other_arrays_that_leave_and_counts_the_rest(tmp_path):
    ref, cand, env_b = run(tmp_path, plant=(5, "cross"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.01))
    text = "\n".join(fd.summarise(rep))
    assert "Other arrays: 1 compared, 1 leave their tolerance." in text
    rep2 = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=10.0))
    assert "Other arrays: 1 compared, 0 leave their tolerance." in "\n".join(fd.summarise(rep2))


# ---- review round 1: identity, noise draws, missing operations, exit status ---------------------------------------------

def _copy(tmp_path, name, run_, **kw):
    return ar.load_run(_write(tmp_path / name, [(k, run_.array(*k)) for k in run_.entries], **kw))


def test_floor_runs_of_another_model_revision_or_library_commit_are_refused(tmp_path):
    ref, cand, env_b = run(tmp_path)
    for name, kw, key in (("rev", dict(model={**MODEL, "revision": "c" * 40}), "model.revision"),
                          ("commit", dict(library={**LIB, "commit": "d" * 40}), "library.commit")):
        other = _copy(tmp_path, name, env_b, **kw)
        with pytest.raises(fd.DivergenceError, match=re.escape(key)):
            fd.find_first_divergence(ref, cand, floor=(ref, other))
        with pytest.raises(fd.DivergenceError, match=re.escape(key)):
            fd.find_first_divergence(ref, cand, floor=(other, env_b))


def test_a_candidate_of_another_model_revision_is_refused(tmp_path):
    ref, cand, _ = run(tmp_path)
    other = _copy(tmp_path, "oc", cand, dtype="bfloat16", model={**MODEL, "revision": "e" * 40})
    with pytest.raises(fd.DivergenceError, match="model.revision"):
        fd.find_first_divergence(ref, other)


def test_environment_fields_of_the_library_may_differ(tmp_path):
    ref, cand, env_b = run(tmp_path)
    other = _copy(tmp_path, "env", env_b, library={**LIB, "path_hash": "abc", "source_verified": False})
    rep = fd.find_first_divergence(ref, cand, floor=(ref, other), tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["environment_floor"] == "recorded"


def test_floor_runs_with_different_noise_draws_are_refused(tmp_path):
    ref, cand, env_b = run(tmp_path)
    arrays = [(k, env_b.array(*k)) for k in env_b.entries] + [(("noise", "initial"), np.ones((1, 3, 3)))]
    ref2 = ar.load_run(_write(tmp_path / "ref2", [(k, ref.array(*k)) for k in ref.entries] + [(("noise", "initial"), np.zeros((1, 3, 3)))]))
    env2 = ar.load_run(_write(tmp_path / "env2", arrays))
    with pytest.raises(fd.DivergenceError, match="noise"):
        fd.find_first_divergence(ref2, cand, floor=(ref2, env2))


def test_a_noise_recording_that_one_floor_run_lacks_is_refused(tmp_path):
    ref, cand, env_b = run(tmp_path)
    noise = [(("noise", "initial"), np.zeros((1, 3, 3)))]
    ref2 = ar.load_run(_write(tmp_path / "ref3", [(k, ref.array(*k)) for k in ref.entries] + noise))
    with pytest.raises(fd.DivergenceError, match="noise/initial is not in every run"):
        fd.find_first_divergence(ref2, cand, floor=(ref2, env_b))


def test_an_operation_the_candidate_never_recorded_is_listed_and_the_report_is_incomplete(tmp_path):
    ref, _, _ = run(tmp_path, plant=(3, "ff"))
    arrays = [((k, n), ref.array(k, n)) for (k, n) in ref.entries if n != "layer_03.cross"]
    cand_arrays = []
    for (k, n), a in arrays:
        cand_arrays.append(((k, n), a + (0.05 * np.abs(a).mean() if n in ("layer_03.ff", "layer_03") else 0)))
    cand = ar.load_run(_write(tmp_path / "noop", cand_arrays, dtype="bfloat16"))
    rep = fd.find_first_divergence(ref, cand, tolerance=fd.Tolerance(rel_l2=0.001))
    assert rep["first_op"]["name"] == "layer_03.ff"
    assert rep["complete"] is False and any("layer_03.cross" in p and "not in the candidate" in p for p in rep["problems"])


def test_the_command_line_exits_3_when_nothing_is_located_but_a_block_is_missing(tmp_path):
    import io
    r, _, _ = build(tmp_path)
    ref = ar.load_run(r)
    arrays = [((k, n), ref.array(k, n)) for (k, n) in ref.entries if n != "layer_03" and not n.startswith("layer_03.")]
    short = _write(tmp_path / "short", arrays, dtype="bfloat16")
    out = io.StringIO()
    assert fd.main([str(r), str(short), "--rel-l2", "0.01"], out=out) == 3
    assert "layer_03" in out.getvalue()


# ---- review round 2: candidate noise, operations missing anywhere, errors after the stack ------------------------------

def test_a_candidate_with_different_noise_draws_is_refused_before_any_attribution(tmp_path):
    ref, cand, _ = run(tmp_path, plant=(3, "attn"))
    noise = ("noise", "initial")
    ref2 = ar.load_run(_write(tmp_path / "rn", [(k, ref.array(*k)) for k in ref.entries] + [(noise, np.zeros((1, 3, 3)))]))
    cand2 = ar.load_run(_write(tmp_path / "cn", [(k, cand.array(*k)) for k in cand.entries] + [(noise, np.ones((1, 3, 3)))], dtype="bfloat16"))
    with pytest.raises(fd.DivergenceError, match="noise/initial differs"):
        fd.find_first_divergence(ref2, cand2, tolerance=fd.Tolerance(rel_l2=0.01))
    cand3 = ar.load_run(_write(tmp_path / "cn3", [(k, cand.array(*k)) for k in cand.entries], dtype="bfloat16"))
    with pytest.raises(fd.DivergenceError, match="noise/initial is not in every run"):
        fd.find_first_divergence(ref2, cand3, tolerance=fd.Tolerance(rel_l2=0.01))


def test_an_operation_missing_from_a_block_that_stays_within_tolerance_makes_the_report_incomplete(tmp_path):
    import io
    r, _, _ = build(tmp_path)
    ref = ar.load_run(r)
    arrays = [((k, n), ref.array(k, n)) for (k, n) in ref.entries if n != "layer_00.attn"]
    cand = _write(tmp_path / "noattn", arrays, dtype="bfloat16")
    rep = fd.find_first_divergence(ref, ar.load_run(cand), tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["located"] is False and rep["complete"] is False
    assert any("layer_00.attn" in p for p in rep["problems"])
    assert fd.main([str(r), str(cand), "--rel-l2", "0.01"], out=io.StringIO()) == 3


def test_an_error_that_enters_after_the_stack_is_a_divergence_and_exits_1(tmp_path):
    import io
    r, _, _ = build(tmp_path)
    ref = ar.load_run(r)
    arrays = [((k, n), ref.array(k, n) * (1.5 if (k, n) == ("latent", "final") else 1.0)) for (k, n) in ref.entries]
    cand = _write(tmp_path / "late", arrays, dtype="bfloat16")
    rep = fd.find_first_divergence(ref, ar.load_run(cand), tolerance=fd.Tolerance(rel_l2=0.01))
    assert rep["located"] is False and rep["diverged"] is True and rep["other_arrays_leaving"] == ["latent/final"]
    out = io.StringIO()
    assert fd.main([str(r), str(cand), "--rel-l2", "0.01"], out=out) == 1
    assert "enters after the stack" in out.getvalue() and "latent/final" in out.getvalue()
    clean = fd.find_first_divergence(ref, ref, tolerance=fd.Tolerance(rel_l2=0.01))
    assert clean["diverged"] is False and clean["other_arrays_leaving"] == []
