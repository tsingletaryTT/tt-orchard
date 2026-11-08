# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/can_cad/gen_card.py (review of PR 629): the card states the measured per-check counts, counts distinct prompts from the saved rows,
and states the disk requirement as weights plus cache per profile. Synthetic result files only."""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "can_cad"
sys.path.insert(0, str(SCRIPTS))
import gen_card  # noqa: E402


def bench(prompts=8, repeats=3):
    d = [{"prompt_index": i, "repeat": r, "decode_tok_s": 14.0 + i * 0.1, "prompt_tokens": 133, "completion_tokens": 220, "ttft_s": 0.19, "total_s": 15.0}
         for r in range(repeats) for i in range(prompts)]
    return {"decode": d, "load_before": ["7.0", "7.1", "7.2"], "load_after": ["7.3", "7.4", "7.5"]}


def cq(prompts=8, repeats=3, **overrides):
    rows = [{"prompt_index": i, "repeat": r, "valid_solid": i != 7} for r in range(repeats) for i in range(prompts)]
    n = len(rows)
    summary = {"header_json": n, "one_python_block": n, "parses": n, "imports_ok": n, "ends_with_result": n, "executes": n - repeats, "valid_solid": n - repeats}
    summary.update(overrides)
    return {"n": n, "distinct_prompts": prompts, "summary": summary, "rows": rows, "sandbox": {"landlock": True, "tcp": True}}


PREFILL = [{"target": 2048, "prompt_tokens": 1975, "ttft_s": 0.6}] * 3 + [{"target": 7000, "prompt_tokens": 6719, "ttft_s": 2.9}] * 3


def build(b=None, c=None):
    return gen_card.build("owner/name", b or bench(), c or cq(), PREFILL)


def test_the_card_states_each_measured_format_count_and_makes_no_all_claim():
    _, card, readme = build(c=cq(header_json=20, imports_ok=22))
    lim = card["limitations"]
    assert "20 of 24 had a first line that parses as JSON" in lim and "22 of 24 had imports of only cadquery and math" in lim
    assert "24 of 24 had exactly one python block" in lim
    assert "all 24 replies" not in lim and "all 24 replies" not in readme


def test_prompts_are_counted_from_the_saved_rows_not_assumed_from_three_repeats():
    b, c = bench(prompts=8, repeats=2), cq(prompts=8, repeats=2)
    _, card, _ = build(b, c)
    assert "2 repeats of 8 CadQuery prompts" in card["performance"] and "checked on 8 prompts" in card["limitations"]
    assert "7 of 8 prompts" in card["limitations"]                      # valid solids per distinct prompt
    c2 = cq(prompts=5, repeats=3); c2.pop("distinct_prompts")
    _, card2, _ = build(bench(prompts=5, repeats=3), c2)
    assert "checked on 5 prompts" in card2["limitations"]


def test_the_disk_requirement_is_weights_plus_cache_per_profile():
    _, card, readme = build()
    for text in (card["prerequisites"], readme):
        assert "61 GB" in text and "35 to 45 GB of tensor cache for each profile" in text and "96 to 106 GB for one profile" in text
    assert "about 70 GB" not in readme


def test_the_card_says_what_the_sandbox_did_and_did_not_block():
    _, card, _ = build()
    assert "Only replies that passed the import and format checks were run" in card["limitations"]
    assert "No network namespace was used" in card["limitations"] and "Landlock" in card["limitations"]
    off = cq(); off["sandbox"] = {"landlock": False}
    assert "were not applied" in build(c=off)[1]["limitations"]


def test_the_performance_table_has_the_columns_the_catalog_requires():
    _, card, _ = build()
    header = next(line for line in card["performance"].splitlines() if line.startswith("| profile"))
    for col in ("profile", "isl", "osl", "concurrency", "n", "ttft", "prefill", "decode", "e2el"):
        assert col in header
