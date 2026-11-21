# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A generated port has no sampler on a die, so the vocabulary-per-die check is not applicable to it (issue 183, PR 787).

PR 787 added a check that asks an optional `sample` hook of the port for the id of a planted column. A port with no hook was recorded as
`skipped`. On the second Aion run the generated port had no hook, so the check was skipped although the hazard scan had flagged the vocabulary
(200,029 columns over two dies).

A generated port returns the whole vocabulary row to the host (`orchard/port_template.py`, the `_forward` function ends in
`out.reshape(-1, out.shape[-1])[n - 1, :VOCAB]`), its weights are replicated over the mesh, and the check's worker and the stage 5 server both choose
the next token by `argmax` over that row on the host. No per-die column index exists on that path, so a sampler cannot wrap one.

The generator now writes the module constant `TOKEN_CHOICE = "host-row"`. The check records `not-applicable` for a flagged vocabulary only when
the port defines no `sample`, declares that constant, and the row it returned had exactly the vocabulary's width. A port with no constant (a
hand-written port) keeps `skipped`, and a port that declares the constant but returned a narrower row keeps `skipped` too, with the width in the
reason. A port that defines `sample` is asked, whatever else it declares. The gate recomputes the same conditions from the record and the snapshot."""
import json

import pytest

from orchard import full_port, hazards
from orchard.stages import gate_full_port
from port_check_fakes import CHIPS, HONEST, make_run, write_result
from test_port_check_fidelity import pc, record
from test_port_check_vocab_check import PER_DIE, SAMPLE_OK, VOCAB, good_block, reasons_for, vc

MARKER = '\nTOKEN_CHOICE = "host-row"\n'                    # the literal: a test below keeps it equal to the template's and the gate's constant
# HONEST with a row as wide as the vocabulary of the Aion run, and the generator's constant: what a generated port is, as far as the check can see.
WIDE_NO_MARKER = HONEST.replace("VOCAB = 200", f"VOCAB = {VOCAB}")
WIDE = WIDE_NO_MARKER + MARKER
NARROW = HONEST + MARKER                                   # the constant, but a row of 200 entries
WITH_HOOK = WIDE + SAMPLE_OK[len(HONEST):]                 # the constant and a `sample` function


def na_block(**over):
    block = {"flagged": True, "vocab_size": VOCAB, "dies": 2, "per_die": PER_DIE, "limit": 65_536, "status": "not-applicable",
             "reason": hazards.VOCAB_NOT_APPLICABLE_REASON, "token_choice": hazards.TOKEN_CHOICE_HOST, "row_length": VOCAB, "cases": []}
    block.update(over)
    return block


# ---- the constants the three places share ---------------------------------------------------------------------------------------------

def test_the_template_and_the_gate_share_the_constant_and_the_sentence():
    assert pc.TOKEN_CHOICE_HOST == hazards.TOKEN_CHOICE_HOST == "host-row"
    assert pc.NOT_APPLICABLE_REASON == hazards.VOCAB_NOT_APPLICABLE_REASON
    assert 0 < len(pc.NOT_APPLICABLE_REASON) <= 300         # the gate's bound on a reason


def test_the_row_length_reads_a_list_a_numpy_array_and_a_torch_tensor_alike():
    import numpy as np
    import torch
    assert pc.row_length([0.0] * 7) == 7
    assert pc.row_length(np.zeros((1, 9))) == 9
    assert pc.row_length(torch.zeros(2, 5)) == 10


def test_the_checks_argmax_reaches_a_column_above_65535_on_the_host():
    """The claim behind `not-applicable`: the token is the host's argmax over the whole row, so a peak at a high column is found."""
    import torch
    row = torch.zeros(VOCAB)
    row[150_000] = 1.0
    assert pc.argmax(row) == 150_000 and pc.argmax(row.tolist()) == 150_000


# ---- through the template, as the supervisor runs it ----------------------------------------------------------------------------------

def test_a_port_that_declares_host_token_choice_and_returns_the_whole_row_is_not_applicable(tmp_path):
    run, stage, snap, done = vc(tmp_path, WIDE)
    assert done.returncode == 0, done.stdout + done.stderr
    block = record(stage)["vocab_check"]
    assert block["status"] == "not-applicable" and block["flagged"] is True and block["cases"] == []
    assert block["token_choice"] == "host-row" and block["row_length"] == VOCAB
    assert block["reason"] == pc.NOT_APPLICABLE_REASON
    assert (block["vocab_size"], block["dies"], block["per_die"], block["limit"]) == (VOCAB, 2, PER_DIE, 65_536)
    assert "vocab check: not-applicable" in done.stdout
    write_result(stage)
    assert gate_full_port(stage, run).ok


def test_a_port_with_no_declaration_keeps_the_skip_even_with_a_whole_row(tmp_path):
    """A hand-written port that says nothing about how its token is chosen is still skipped: the check cannot tell."""
    run, stage, snap, done = vc(tmp_path, WIDE_NO_MARKER)
    assert done.returncode == 0, done.stdout + done.stderr
    block = record(stage)["vocab_check"]
    assert block["status"] == "skipped" and "no sample" in block["reason"] and "token_choice" not in block
    write_result(stage)
    assert gate_full_port(stage, run).ok


def test_a_port_that_declares_it_but_returned_a_narrower_row_is_skipped_and_the_reason_gives_the_width(tmp_path):
    run, stage, snap, done = vc(tmp_path, NARROW)
    assert done.returncode == 0, done.stdout + done.stderr
    block = record(stage)["vocab_check"]
    assert block["status"] == "skipped" and block["cases"] == [] and len(block["reason"]) <= 300
    assert "200 entries" in block["reason"] and str(VOCAB) in block["reason"]
    write_result(stage)
    assert gate_full_port(stage, run).ok


@pytest.mark.parametrize("value", ['"device"', '""', '"host-row-"', "7", "None", "['host-row']", '"x" * 10_000', '"x" * 5_000_000', 'object()', 'b"host-row"'])
def test_any_other_declaration_keeps_the_skip_and_never_breaks_the_record(tmp_path, value):
    run, stage, snap, done = vc(tmp_path, WIDE_NO_MARKER + f"\nTOKEN_CHOICE = {value}\n")
    assert done.returncode == 0, done.stdout + done.stderr
    block = record(stage)["vocab_check"]
    assert block["status"] == "skipped" and len(block["reason"]) <= 300 and "token_choice" not in block


def test_a_port_that_defines_sample_is_asked_even_when_it_declares_host_token_choice(tmp_path):
    run, stage, snap, done = vc(tmp_path, WITH_HOOK)
    assert done.returncode == 0, done.stdout + done.stderr
    block = record(stage)["vocab_check"]
    assert block["status"] == "pass" and len(block["cases"]) == 8 and "token_choice" not in block
    assert len((run / "stages" / "2" / "sample-calls.txt").read_text().splitlines()) == 8      # the hook's own function ran at every planted column


def test_the_width_comes_from_the_ports_last_answer_not_from_its_declaration(tmp_path):
    """A port whose rows are narrow only on decode: the last row the worker saw before the question decides."""
    narrow_decode = WIDE.replace("def decode_step(model, token_id, position):\n    return _peak(token_id + 1)",
                                 "def decode_step(model, token_id, position):\n    return _peak(token_id + 1)[:200]")
    assert narrow_decode != WIDE
    run, stage, snap, done = vc(tmp_path, narrow_decode)
    assert done.returncode == 0, done.stdout + done.stderr
    assert record(stage)["vocab_check"]["status"] == "skipped"


def test_a_vocabulary_that_fits_leaves_no_block_even_for_a_declaring_port(tmp_path):
    run, stage, snap, done = vc(tmp_path, WIDE, vocab=100_000)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "vocab_check" not in record(stage)


# ---- the gate -------------------------------------------------------------------------------------------------------------------------

def gate_setup(tmp_path, vocab=VOCAB):
    run, stage, snap = make_run(tmp_path, WIDE)
    (snap / "config.json").write_text(json.dumps({"vocab_size": vocab}))
    return run, stage


def test_the_gate_accepts_a_well_formed_not_applicable_block(tmp_path):
    run, stage = gate_setup(tmp_path)
    assert reasons_for({"chips": CHIPS, "vocab_check": na_block()}, run, stage) == []


@pytest.mark.parametrize("over,needle", [
    ({"row_length": VOCAB - 1}, "row_length"),
    ({"row_length": VOCAB + 1}, "row_length"),
    ({"row_length": True}, "row_length"),
    ({"row_length": str(VOCAB)}, "row_length"),
    ({"row_length": None}, "row_length"),
    ({"token_choice": "device"}, "token_choice"),
    ({"token_choice": None}, "token_choice"),
    ({"reason": "no reason"}, "reason"),
    ({"reason": ""}, "reason"),
    ({"cases": good_block()["cases"]}, "cases"),
])
def test_the_gate_refuses_a_not_applicable_block_whose_evidence_does_not_hold(tmp_path, over, needle):
    run, stage = gate_setup(tmp_path)
    reasons = reasons_for({"chips": CHIPS, "vocab_check": na_block(**over)}, run, stage)
    assert reasons and any(needle in r for r in reasons), reasons


def test_the_gate_names_the_three_statuses_it_accepts(tmp_path):
    run, stage = gate_setup(tmp_path)
    reasons = reasons_for({"chips": CHIPS, "vocab_check": {**na_block(), "status": "n/a"}}, run, stage)
    assert any("not-applicable" in r for r in reasons), reasons


def test_the_gate_reads_the_row_length_against_its_own_reading_of_the_snapshot(tmp_path):
    """The block's own vocab_size can be right while the row was narrower: the gate compares the row with the snapshot's vocabulary."""
    run, stage = gate_setup(tmp_path)
    block = na_block(row_length=200)
    assert any("200" in r and "row_length" in r for r in reasons_for({"chips": CHIPS, "vocab_check": block}, run, stage))
