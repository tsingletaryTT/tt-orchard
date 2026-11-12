# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 check asks a causal port to agree with itself at short lengths and around one tile of 32 tokens (a lesson from a skills-only port
of the tiny Qwen4, whose author tried prompts of 2, 3, 32 and 160 tokens): a prefill of seq[:L] and a prefill of seq[:L-1] followed by one decode step
must predict the same next token. The block is report-only; the gate checks that it agrees with itself."""
import json

import pytest

from orchard import full_port
from orchard.stages import gate_full_port
from port_check_fakes import HONEST, make_run, run_template, write_result
from test_port_check_fidelity import pc, record

# A port whose prefill follows the ids: the next id is the last id plus 1, except for the stage's own prompt, which answers the reference's first id.
CONSISTENT = (HONEST.replace("    return _peak(100)", "    return _peak(100 if list(ids) == [1, 2, 3] else ids[-1] + 1)")
              .replace("    return _peak(token_id + 1)", "    return _peak(100 if (token_id, position) == (3, 2) else token_id + 1)"))      # the same answer by either route
# The same port with a tile-boundary bug: a prompt of exactly 32 tokens is answered wrongly.
BUG_AT_32 = CONSISTENT.replace("def prefill(model, ids):", "def prefill(model, ids):\n    if len(ids) == 32:\n        return _peak(0)")
RAISES_SHORT = CONSISTENT.replace("def prefill(model, ids):", "def prefill(model, ids):\n    if len(ids) == 2:\n        raise ValueError('no short prompts')")


def lc(tmp_path, port):
    run, stage, _ = make_run(tmp_path, port)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    return run, stage, record(stage)["length_check"], done.stdout


def reasons_for(rec):
    out = []
    full_port._length_check_reasons(rec, out)
    return out


def test_a_consistent_port_agrees_at_every_length(tmp_path):
    run, stage, block, _ = lc(tmp_path, CONSISTENT)
    assert [r["length"] for r in block["lengths"]] == list(pc.LENGTH_CHECK_LENGTHS) and block["all_agree"] is True
    assert all(r["prefill_id"] == r["decode_id"] for r in block["lengths"])


def test_a_bug_at_one_tile_of_tokens_is_named_by_length_and_does_not_fail_the_run(tmp_path):
    run, stage, block, out = lc(tmp_path, BUG_AT_32)
    assert block["all_agree"] is False
    assert [r["length"] for r in block["lengths"] if not r["agree"]] == [32]
    assert "lengths [32]" in out and "reported, not gated" in out
    write_result(stage)
    assert gate_full_port(stage, run).ok


def test_a_port_that_raises_on_a_short_prompt_is_recorded_and_the_check_goes_on(tmp_path):
    run, stage, block, out = lc(tmp_path, RAISES_SHORT)
    assert "no short prompts" in block["error"] and "all_agree" not in block
    assert "length check: the port failed it" in out
    assert reasons_for({"length_check": block}) == []


def test_lengths_longer_than_the_sequence_are_skipped():
    class W:
        last = {}
        def ask(self, **kw):
            return 1
    block = pc.length_check(W(), [1, 2, 3, 4])             # a sequence of 4 ids leaves only the lengths 2 and 3
    assert [r["length"] for r in block["lengths"]] == [2, 3]


@pytest.mark.parametrize("port", [CONSISTENT, BUG_AT_32])
def test_the_gate_accepts_an_honest_block_and_a_block_with_a_disagreement(tmp_path, port):
    run, stage, block, _ = lc(tmp_path, port)
    assert reasons_for({"length_check": block}) == []


@pytest.mark.parametrize("edit", [
    lambda b: b["lengths"][0].__setitem__("agree", not b["lengths"][0]["agree"]),     # a row whose flag contradicts its ids
    lambda b: b.__setitem__("all_agree", not b["all_agree"]),                          # a summary that contradicts its rows
    lambda b: b["lengths"].reverse(),                                                  # lengths out of order
    lambda b: b["lengths"][1].__setitem__("decode_id", "x"),                           # an id that is not a whole number
    lambda b: b.__setitem__("lengths", "none"),                                        # not a list
])
def test_the_gate_refuses_a_block_that_contradicts_itself(tmp_path, edit):
    run, stage, block, _ = lc(tmp_path, BUG_AT_32)
    edit(block)
    assert reasons_for({"length_check": block})


def test_the_gate_reaches_the_block_through_the_whole_stage_gate(tmp_path):
    run, stage, block, _ = lc(tmp_path, CONSISTENT)
    write_result(stage)
    rec = record(stage)
    rec["length_check"]["all_agree"] = False
    (stage / "evidence" / "port-check.json").write_text(json.dumps(rec))
    assert any("length_check" in r for r in gate_full_port(stage, run).reasons)
