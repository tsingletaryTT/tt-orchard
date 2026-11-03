# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks on the recorded evidence of issue 301 (Granite chunked-prefill follow-ups).

These read files in docs/run-logs/evidence. They do not touch a chip. They keep the labels in the evidence honest: a pointer in a
conditions file must name a file that exists and holds the result it says it holds, and the agreement records past 1,024 tokens must
be records of that comparison.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "docs" / "run-logs" / "evidence"
S4 = EVID / "granite-package" / "results" / "s4"
FOLLOW = EVID / "granite-301-followups"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_pre_round2_smoke_label_points_at_the_file_that_holds_it():
    text = (S4 / "conditions.txt").read_text(encoding="utf-8")
    named = re.search(r"pre-round-2 smoke.*?is kept in ([\w./-]+\.json)", text, re.S)
    assert named, "conditions.txt must say where the pre-round-2 smoke is kept"
    pointed = S4 / named.group(1)
    assert pointed.is_file(), f"the label names {named.group(1)}, which does not exist"
    old = _load(pointed)
    # the pre-round-2 server gave 33.6 tok/s and 1.49 s, which the label states; the final-code smoke gave 34.6 and 1.24
    assert round(old["decode_run"]["timings"]["decode_tok_s"], 1) == 33.6
    assert round(old["long_prompt"]["timings"]["first_token_s"], 2) == 1.49
    final = _load(S4 / "smoke.json")
    assert final != old, "the final-code smoke and the pre-round-2 smoke must be different records"


def test_final_code_smoke_is_the_one_in_the_round3_evidence():
    assert (S4 / "smoke.json").read_bytes() == (EVID / "granite-chunked-trace-safe" / "round3" / "smoke.json").read_bytes()


def test_agreement_records_past_1024_tokens():
    seen = set()
    for path in sorted(FOLLOW.glob("agree*.json")):
        r = _load(path)
        assert r["mode"] == "agree" and r["trace"] is True
        assert r["tokens"] > 1024
        assert r["max_len"] >= r["tokens"] + 4          # room for the four greedy tokens read after the prompt
        assert r["top1_equal"] is True and r["next4_equal"] is True and r["top5_overlap"] == 5
        assert 0 <= r["max_state_rel_diff"] < 0.1
        seen.add(r["tokens"])
    assert len(seen) >= 2, "agreement was recorded at two lengths past 1,024 tokens"


def test_smoke_of_the_package_boot_passed_all_checks_and_records_its_load():
    r = _load(FOLLOW / "smoke-pkg030.json")
    assert all(c["ok"] is True for c in r["checks"])
    assert len(r["checks"]) == 9
    assert "load before" in (FOLLOW / "boot-conditions.txt").read_text(encoding="utf-8")
