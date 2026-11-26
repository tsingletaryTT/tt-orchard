# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tt-orchard tier-plan --skew-file: the plan reads a measured expert-use skew from the file scripts/expert_skew.py or expert_skew_stream.py wrote
(issue 739 step 2). The file is checked against the model being planned before any figure is used."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import tier_plan

ROOT = Path(__file__).resolve().parent.parent
CFG = {"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
       "head_dim": 16, "vocab_size": 100, "intermediate_size": 128, "moe_intermediate_size": 32, "num_experts": 8,
       "num_experts_per_tok": 2, "tie_word_embeddings": False}


def result(**over):
    r = {"model": "tiny-moe", "tokens": 16384, "experts": 8, "top_k": 2, "moe_layers": 2, "text_note": "project docs",
         "mean": {"never_picked": 0.0, "top_10pct_share": 0.30, "top_25pct_share": 0.55, "top_50pct_share": 0.80, "entropy_vs_max": 0.9}}
    r.update(over)
    return r


def write(tmp_path, data, name="skew.json"):
    f = tmp_path / name
    f.write_text(json.dumps(data))
    return f


def load(f, **kw):
    kw.setdefault("experts", 8)
    kw.setdefault("top_k", 2)
    return tier_plan.load_skew_file(f, **kw)


def test_the_mean_shares_become_the_three_skew_points_and_the_source_names_the_measurement(tmp_path):
    points, source = load(write(tmp_path, result()))
    assert points == ((0.10, 0.30), (0.25, 0.55), (0.50, 0.80))
    assert "tiny-moe" in source and "16384 tokens" in source and "project docs" in source and "measured" in source


def test_the_committed_olmoe_measurement_loads_for_a_64_expert_top_8_model_and_matches_the_built_in_figures():
    f = ROOT / "docs/run-logs/2026-11-11-expert-skew-olmoe-1b-7b.json"
    points, _ = tier_plan.load_skew_file(f, experts=64, top_k=8)
    for (frac, share), (want_frac, want_share) in zip(points, tier_plan.OLMOE_SKEW):
        assert frac == want_frac and share == pytest.approx(want_share, abs=1e-3)


@pytest.mark.parametrize("over,what", [({"experts": 64}, "64 experts"), ({"top_k": 8}, "top 8")])
def test_a_file_measured_on_a_model_of_another_shape_is_refused_with_both_numbers(tmp_path, over, what):
    with pytest.raises(tier_plan.TierPlanError, match=what) as e:
        load(write(tmp_path, result(**over)))
    assert "8 experts" in str(e.value) or "top 2" in str(e.value)


def test_a_map_of_sources_needs_a_name_and_the_name_must_exist(tmp_path):
    f = write(tmp_path, {"docs": result(text_note="docs"), "code": result(text_note="code", mean={**result()["mean"], "top_10pct_share": 0.35})})
    with pytest.raises(tier_plan.TierPlanError, match="code, docs|docs, code"):
        load(f)
    with pytest.raises(tier_plan.TierPlanError, match="nope"):
        load(f, text="nope")
    points, source = load(f, text="code")
    assert points[0] == (0.10, 0.35) and "code" in source


def test_a_name_given_for_a_file_with_one_result_is_refused(tmp_path):
    with pytest.raises(tier_plan.TierPlanError, match="one text"):
        load(write(tmp_path, result()), text="docs")


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("mean"),
    lambda r: r["mean"].pop("top_25pct_share"),
    lambda r: r["mean"].__setitem__("top_25pct_share", "high"),
    lambda r: r["mean"].__setitem__("top_25pct_share", float("nan")),
    lambda r: r["mean"].__setitem__("top_25pct_share", True),
    lambda r: r["mean"].__setitem__("top_25pct_share", 10 ** 400),   # an integer too large for a float must give a refusal and never an OverflowError
    lambda r: r["mean"].__setitem__("top_10pct_share", 0.05),        # the busiest tenth cannot take less than a tenth
    lambda r: r["mean"].__setitem__("top_50pct_share", 0.40),        # a larger set cannot take less than a smaller one
    lambda r: r.__setitem__("tokens", 0),
    lambda r: r.__setitem__("tokens", True),
    lambda r: r.__setitem__("tokens", 1.5),
    lambda r: r.__setitem__("experts", "8"),
])
def test_a_result_with_a_missing_wrong_or_impossible_figure_is_refused(tmp_path, mutate):
    r = result()
    mutate(r)
    with pytest.raises(tier_plan.TierPlanError, match="skew file"):
        load(write(tmp_path, r))


def test_a_file_that_is_not_a_json_object_is_refused(tmp_path):
    f = tmp_path / "skew.json"
    for text in ("[1, 2]", "not json", ""):
        f.write_text(text)
        with pytest.raises(tier_plan.TierPlanError, match="skew file"):
            load(f)


def test_a_missing_file_a_link_and_an_oversized_file_are_refused(tmp_path):
    with pytest.raises(tier_plan.TierPlanError, match="skew file"):
        load(tmp_path / "absent.json")
    real = write(tmp_path, result())
    link = tmp_path / "link.json"
    os.symlink(real, link)
    with pytest.raises(tier_plan.TierPlanError, match="skew file"):
        load(link)
    big = tmp_path / "big.json"
    big.write_text(" " * (tier_plan.MAX_SKEW_FILE_BYTES + 1) + json.dumps(result()))
    with pytest.raises(tier_plan.TierPlanError, match="skew file"):
        load(big)


def test_a_named_pipe_is_refused_without_blocking(tmp_path):
    """Run in a child with a time limit: a reader that opens a pipe for reading waits for a writer for ever."""
    fifo = tmp_path / "skew.json"
    os.mkfifo(fifo)
    code = ("from orchard import tier_plan\n"
            "try:\n"
            f"    tier_plan.load_skew_file({str(fifo)!r}, experts=8, top_k=2)\n"
            "except tier_plan.TierPlanError as e:\n"
            "    print('refused:', e)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("refused: skew file"), r.stdout + r.stderr


def test_the_source_sentence_holds_only_printable_text_of_a_bounded_length(tmp_path):
    _, source = load(write(tmp_path, result(model="evil\x1b[31m" + "x" * 500, text_note="line\nbreak")))
    assert "\x1b" not in source and "\n" not in source and len(source) < 400


def test_the_command_uses_the_file_and_prints_where_the_hit_share_came_from(tmp_path, capsys):
    big = dict(CFG, hidden_size=4096, moe_intermediate_size=2048, intermediate_size=2048, num_experts=256, num_hidden_layers=40, vocab_size=150000,
               num_attention_heads=32, num_key_value_heads=8, head_dim=128)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(big))
    skew = write(tmp_path, result(experts=256))
    assert tier_plan.main([str(cfg), "--skew-file", str(skew), "--host-free-gib", "200", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inputs"]["expert_pick_skew"]["value"] == [[0.10, 0.30], [0.25, 0.55], [0.50, 0.80]]
    assert "tiny-moe" in out["inputs"]["expert_pick_skew"]["note"] and "16384 tokens" in out["inputs"]["expert_pick_skew"]["note"]
    labels = [r["hit_share_label"] for r in out["rows"] if r.get("hit_share_label", "") and r["hit_share_label"].startswith("estimate")]
    assert labels and all("tiny-moe" in label and "not measured for this model" not in label for label in labels)


def test_a_file_for_another_model_stops_the_command_with_exit_2_and_one_line(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG))
    assert tier_plan.main([str(cfg), "--skew-file", str(write(tmp_path, result(experts=64)))]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "64 experts" in err[0]


def test_skew_and_skew_file_together_are_refused(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG))
    assert tier_plan.main([str(cfg), "--skew", "0.25:0.6", "--skew-file", str(write(tmp_path, result()))]) == 2
    assert "only one" in capsys.readouterr().err


def test_a_name_without_a_file_is_refused(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG))
    assert tier_plan.main([str(cfg), "--skew-text", "docs"]) == 2
    assert "--skew-file" in capsys.readouterr().err
