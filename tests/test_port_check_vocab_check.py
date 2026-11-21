# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 check runs the vocabulary-per-die hazard's targeted test (issue 183, hazard `vocab_per_die_above_65536`).

Stage 0's hazard scan predicts that a sampler index wraps at 65,536 columns per die. Its `stage2_check` said "sample with the greedy and
the sampled path on logits whose maximum sits in the last column of each die", but nothing ran it: the check's worker takes the logits a port
returns and runs argmax over the whole vocabulary on the host, and the 32 reference ids of a plain text seldom reach a column above 65,535 of a
die. Found on the Aion-1.0-Instruct-2.3B run (vocabulary 200,029, two dies, 100,015 columns each).

The template now calls an optional `sample(model, logits, temperature, top_k)` hook of the port, only when the checkpoint's vocabulary over the
chips of the mesh is above the limit. A wrong id fails the run with a named reason. A port with no hook is recorded as skipped, with the reason.
The gate recomputes whether the check applies from the snapshot's own config.json and the record's chips, and checks each case in the block."""
import json
from pathlib import Path

import pytest

from orchard import full_port, hazards
from orchard.stages import gate_full_port
from port_check_fakes import CHIPS, HONEST, make_run, run_template, write_result
from test_port_check_fidelity import pc, record

VOCAB = 200_029                       # Aion-1.0-Instruct-2.3B
PER_DIE = 100_015                     # ceil(200029 / 2)
CASES = [(0, 65_536), (0, 100_014), (1, 165_551), (1, 200_028)]      # (die, column): the first column past the limit and the last column of each die

# A port with a correct host sampler: the id of the largest logit. The call log lets a test see what the check asked for.
SAMPLE_OK = HONEST + '''
def sample(model, logits, temperature, top_k):
    with open("stages/2/sample-calls.txt", "a") as fh:
        fh.write(f"{max(range(len(logits)), key=lambda i: logits[i])} {temperature} {top_k}\\n")
    return max(range(len(logits)), key=lambda i: logits[i])
'''
# The hazard: the index within a die wraps at 65,536.
SAMPLE_WRAPS = HONEST + '''
def sample(model, logits, temperature, top_k):
    i = max(range(len(logits)), key=lambda j: logits[j])
    per_die = -(-len(logits) // 2)
    die = i // per_die
    return die * per_die + (i - die * per_die) % 65536
'''
SAMPLE_RAISES = HONEST + '''
def sample(model, logits, temperature, top_k):
    i = max(range(len(logits)), key=lambda j: logits[j])
    if i >= 65536:
        raise IndexError("index above 65535")
    return i
'''
SAMPLE_ONLY_GREEDY_WRAPS = HONEST + '''
def sample(model, logits, temperature, top_k):
    i = max(range(len(logits)), key=lambda j: logits[j])
    return i % 65536 if temperature == 0.0 else i
'''


def vc(tmp_path, port, *, vocab=VOCAB, chips=CHIPS, config=True):
    run, stage, snap = make_run(tmp_path, port)
    if config:
        (snap / "config.json").write_text(json.dumps({"vocab_size": vocab}))
    done = run_template(tmp_path, run, stage, chips=chips)
    return run, stage, snap, done


def reasons_for(rec, run, stage):
    out = []
    full_port._vocab_check_reasons(rec, run, out, stage)
    return out


# ---- the pure parts -------------------------------------------------------------------------------------------------------------------

def test_the_limit_is_the_hazard_scans_limit():
    assert pc.VOCAB_PER_DIE_LIMIT == hazards.VOCAB_PER_DIE_LIMIT


def test_the_probe_plants_the_first_column_past_the_limit_and_the_last_column_of_each_die():
    got = [(c["die"], c["planted"]) for c in pc.vocab_probe_cases(VOCAB, 2)]
    assert got == CASES


def test_a_vocabulary_that_fits_the_limit_has_no_probe_and_is_not_flagged():
    assert pc.vocab_per_die(131_072, 2) == 65_536 and pc.vocab_flagged(131_072, 2) is False
    assert pc.vocab_flagged(131_073, 2) is True
    assert pc.vocab_flagged(VOCAB, 4) is False                  # 50,008 columns per die


@pytest.mark.parametrize("vocab,dies", [(VOCAB, 2), (131_073, 2), (170_000, 2), (70_000, 1), (VOCAB, 1), (300_000, 3), (65_537, 1)])
def test_the_gates_expected_rows_equal_the_templates_probe_for_any_vocabulary_and_chip_count(vocab, dies):
    template = [(c["die"], c["column"], c["planted"], m) for c in pc.vocab_probe_cases(vocab, dies) for m, _, _ in pc.VOCAB_MODES]
    assert full_port._vocab_cases(vocab, dies) == template
    assert all(0 <= planted < vocab for _, _, planted, _ in template)


@pytest.mark.parametrize("config", [{}, {"vocab_size": True}, {"vocab_size": "200029"}, {"vocab_size": -1}, {"vocab_size": 10 ** 12}])
def test_a_config_without_a_usable_vocabulary_is_not_flagged(config):
    assert pc.vocab_size_of(config) is None


def test_the_vocabulary_is_read_from_the_text_config_too():
    assert pc.vocab_size_of({"text_config": {"vocab_size": VOCAB}}) == VOCAB


# ---- through the template, as the supervisor runs it ----------------------------------------------------------------------------------

def test_a_correct_sampler_passes_and_the_port_is_asked_on_both_paths_at_every_planted_column(tmp_path):
    run, stage, snap, done = vc(tmp_path, SAMPLE_OK)
    assert done.returncode == 0, done.stdout + done.stderr
    block = record(stage)["vocab_check"]
    assert block["status"] == "pass" and block["flagged"] is True
    assert (block["vocab_size"], block["dies"], block["per_die"], block["limit"]) == (VOCAB, 2, PER_DIE, 65_536)
    assert [(c["die"], c["planted"], c["mode"]) for c in block["cases"]] == [(d, p, m) for d, p in CASES for m in ("greedy", "sampled")]
    assert all(c["got"] == c["planted"] and c["ok"] is True for c in block["cases"])
    # Wiring: the port's own function was called, with the planted column, greedy (temperature 0) and sampled (above 0), one candidate each.
    calls = [line.split() for line in (run / "stages" / "2" / "sample-calls.txt").read_text().splitlines()]
    assert [(int(c[0]), float(c[1]) > 0, int(c[2])) for c in calls] == [(p, m, 1) for d, p in CASES for m in (False, True)]
    write_result(stage)
    assert gate_full_port(stage, run).ok


def test_a_sampler_that_wraps_at_65536_fails_the_run_and_names_the_die_the_column_and_the_wrong_id(tmp_path):
    run, stage, snap, done = vc(tmp_path, SAMPLE_WRAPS)
    assert done.returncode == 2
    assert "vocab check" in done.stdout and "die 0" in done.stdout and "column 65536" in done.stdout
    assert "PORT_CHECK_FAILURE" in done.stdout and '"op": "sample"' in done.stdout
    assert not (stage / "evidence" / "port-check.json").exists()


def test_a_wrong_id_on_the_sampled_path_alone_fails_too(tmp_path):
    run, stage, snap, done = vc(tmp_path, SAMPLE_ONLY_GREEDY_WRAPS)
    assert done.returncode == 2 and "greedy" in done.stdout


def test_a_sampler_that_raises_above_65535_fails_the_run_with_its_error(tmp_path):
    run, stage, snap, done = vc(tmp_path, SAMPLE_RAISES)
    assert done.returncode == 2 and "index above 65535" in done.stdout and '"op": "sample"' in done.stdout


def test_a_port_with_no_sample_function_is_recorded_as_skipped_with_the_reason_and_still_passes(tmp_path):
    run, stage, snap, done = vc(tmp_path, HONEST)
    assert done.returncode == 0, done.stdout + done.stderr
    block = record(stage)["vocab_check"]
    assert block["status"] == "skipped" and block["flagged"] is True and "no sample" in block["reason"] and block["cases"] == []
    assert "vocab check: skipped" in done.stdout
    write_result(stage)
    assert gate_full_port(stage, run).ok


def test_a_vocabulary_that_fits_leaves_the_record_without_a_block_and_never_calls_the_hook(tmp_path):
    run, stage, snap, done = vc(tmp_path, SAMPLE_WRAPS, vocab=100_000)       # 50,000 columns per die on two chips
    assert done.returncode == 0, done.stdout + done.stderr
    assert "vocab_check" not in record(stage)


def test_four_chips_bring_the_same_vocabulary_under_the_limit(tmp_path):
    four = CHIPS + ["0000:03:00.0", "0000:04:00.0"]
    run, stage, snap, done = vc(tmp_path, SAMPLE_WRAPS, chips=four)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "vocab_check" not in record(stage)


def test_a_snapshot_with_no_config_is_not_flagged(tmp_path):
    run, stage, snap, done = vc(tmp_path, SAMPLE_WRAPS, config=False)
    assert done.returncode == 0 and "vocab_check" not in record(stage)


# ---- the gate -------------------------------------------------------------------------------------------------------------------------

def good_block():
    return {"flagged": True, "vocab_size": VOCAB, "dies": 2, "per_die": PER_DIE, "limit": 65_536, "status": "pass", "reason": "",
            "cases": [{"die": d, "column": c, "planted": p, "mode": m, "got": p, "ok": True}
                      for d, p in CASES for m in ("greedy", "sampled") for c in [p - d * PER_DIE]]}


def gate_setup(tmp_path, vocab=VOCAB):
    run, stage, snap = make_run(tmp_path, SAMPLE_OK)
    (snap / "config.json").write_text(json.dumps({"vocab_size": vocab}))
    return run, stage


def test_the_gate_accepts_a_well_formed_pass_block(tmp_path):
    run, stage = gate_setup(tmp_path)
    assert reasons_for({"chips": CHIPS, "vocab_check": good_block()}, run, stage) == []


def test_the_gate_wants_the_block_when_its_own_reading_of_the_config_says_the_check_applies(tmp_path):
    run, stage = gate_setup(tmp_path)
    reasons = reasons_for({"chips": CHIPS}, run, stage)
    assert len(reasons) == 1 and "vocab_check" in reasons[0] and "200029" in reasons[0]


def test_the_gate_does_not_want_the_block_when_the_check_does_not_apply(tmp_path):
    run, stage = gate_setup(tmp_path, vocab=100_000)
    assert reasons_for({"chips": CHIPS}, run, stage) == []
    run, stage = gate_setup(tmp_path / "again")
    assert reasons_for({"chips": CHIPS + ["a", "b"]}, run, stage) == []


def test_the_gate_recomputes_each_case_and_refuses_a_pass_that_its_own_cases_contradict(tmp_path):
    run, stage = gate_setup(tmp_path)
    block = good_block()
    block["cases"][2]["got"] = 3                                  # the row still says ok: true
    reasons = reasons_for({"chips": CHIPS, "vocab_check": block}, run, stage)
    assert any("got 3" in r and "planted" in r for r in reasons)


def test_the_gate_refuses_a_pass_that_left_a_die_out(tmp_path):
    run, stage = gate_setup(tmp_path)
    block = good_block()
    block["cases"] = [c for c in block["cases"] if c["die"] == 0]
    assert any("die 1" in r for r in reasons_for({"chips": CHIPS, "vocab_check": block}, run, stage))


def test_the_gate_refuses_a_block_that_names_another_vocabulary_than_the_snapshots(tmp_path):
    run, stage = gate_setup(tmp_path)
    block = good_block()
    block["vocab_size"] = 100_000
    assert any("vocab_size" in r for r in reasons_for({"chips": CHIPS, "vocab_check": block}, run, stage))


def test_the_gate_accepts_a_skip_that_gives_a_reason_and_refuses_one_that_does_not(tmp_path):
    run, stage = gate_setup(tmp_path)
    skip = {"flagged": True, "vocab_size": VOCAB, "dies": 2, "per_die": PER_DIE, "limit": 65_536, "status": "skipped",
            "reason": "port_model defines no sample(model, logits, temperature, top_k)", "cases": []}
    assert reasons_for({"chips": CHIPS, "vocab_check": skip}, run, stage) == []
    assert reasons_for({"chips": CHIPS, "vocab_check": {**skip, "reason": ""}}, run, stage) != []


@pytest.mark.parametrize("status", ["fail", "error", True, None, "PASS"])
def test_the_gate_refuses_any_other_status(tmp_path, status):
    run, stage = gate_setup(tmp_path)
    assert reasons_for({"chips": CHIPS, "vocab_check": {**good_block(), "status": status}}, run, stage) != []


def test_the_gate_refuses_a_block_that_is_not_an_object(tmp_path):
    run, stage = gate_setup(tmp_path)
    assert any("must be an object" in r for r in reasons_for({"chips": CHIPS, "vocab_check": [1]}, run, stage))


def test_the_gate_stops_at_a_block_with_too_many_cases_before_it_reads_them(tmp_path):
    run, stage = gate_setup(tmp_path)
    many = {**good_block(), "cases": good_block()["cases"] * 20}
    reasons = reasons_for({"chips": CHIPS, "vocab_check": many}, run, stage)
    assert len(reasons) == 1 and "at most 64" in reasons[0]


def test_the_gate_refuses_cases_that_are_not_the_rows_it_expects_even_when_every_die_and_every_row_looks_fine(tmp_path):
    run, stage = gate_setup(tmp_path)
    greedy_only = {**good_block(), "cases": [c for c in good_block()["cases"] if c["mode"] == "greedy"]}
    low_column = good_block()
    low_column["cases"][0].update(column=5, planted=5, got=5)           # a planted column far below the limit proves nothing about the wrap
    for block in (greedy_only, low_column):
        assert any("not the 8 rows the gate expects" in r for r in reasons_for({"chips": CHIPS, "vocab_check": block}, run, stage))


def test_the_gate_refuses_a_row_that_says_not_ok_when_its_id_is_right(tmp_path):
    run, stage = gate_setup(tmp_path)
    block = good_block()
    block["cases"][1]["ok"] = False
    assert any("says ok=false" in r for r in reasons_for({"chips": CHIPS, "vocab_check": block}, run, stage))


def test_a_record_whose_chips_cannot_be_counted_is_refused_for_a_checkpoint_that_needs_the_check(tmp_path):
    run, stage = gate_setup(tmp_path, vocab=10_000_000)           # 65 chips would still leave 153,847 columns per die
    for chips in ([f"chip{i}" for i in range(65)], [], "0000:01:00.0", None):
        assert any("chips must be a list" in r for r in reasons_for({"chips": chips}, run, stage))
    run2, stage2 = gate_setup(tmp_path / "small", vocab=60_000)
    assert reasons_for({"chips": None}, run2, stage2) == []         # a vocabulary under the limit never needs the count


def test_the_gate_runs_this_check_as_part_of_the_stage_gate(tmp_path):
    # Wiring: delete the block from the record of a passing, flagged run and the stage gate, not only the helper, names it.
    run, stage, snap, done = vc(tmp_path, SAMPLE_OK)
    assert done.returncode == 0
    path = stage / "evidence" / "port-check.json"
    rec = json.loads(path.read_text())
    del rec["vocab_check"]
    path.write_text(json.dumps(rec, indent=2) + "\n")
    write_result(stage)
    verdict = gate_full_port(stage, run)
    assert not verdict.ok and "vocab_check" in " ".join(verdict.reasons)


# ---- a snapshot's config.json that something else controls ----------------------------------------------------------------------------

HOSTILE_CONFIG_SCRIPT = """
import sys
from pathlib import Path
from orchard import full_port
print(full_port._snapshot_vocab(Path(sys.argv[1])))
"""


@pytest.mark.parametrize("kind", ["fifo", "oversize", "link_to_a_directory", "not_json", "list"])
def test_a_config_json_that_is_not_a_small_regular_json_object_gives_no_vocabulary_and_no_hang(tmp_path, kind):
    """Run in a child with a timeout: a reader that opened a FIFO in blocking mode would hang here and the timeout names it."""
    import os
    import subprocess
    import sys
    run, stage, snap = make_run(tmp_path, SAMPLE_OK)
    cfg = snap / "config.json"
    if kind == "fifo":
        os.mkfifo(cfg)
    elif kind == "oversize":
        cfg.write_text('{"vocab_size": 200029, "pad": "' + " " * (2 << 20) + '"}')          # above the 1 MiB read cap
    elif kind == "link_to_a_directory":
        cfg.symlink_to(tmp_path)
    elif kind == "not_json":
        cfg.write_text("{vocab_size: 200029")
    else:
        cfg.write_text("[200029]")
    done = subprocess.run([sys.executable, "-c", HOSTILE_CONFIG_SCRIPT, str(stage)], capture_output=True, text=True, timeout=30,
                          cwd=str(Path(full_port.__file__).resolve().parent.parent))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "None"


@pytest.mark.parametrize("value", [True, False, "200029", 200029.0, -5, 0, 10 ** 12, None, [200029], {"n": 1}])
def test_the_gate_takes_only_a_whole_number_in_range_as_the_vocabulary(tmp_path, value):
    run, stage, snap = make_run(tmp_path, SAMPLE_OK)
    (snap / "config.json").write_text(json.dumps({"vocab_size": value}))
    assert full_port._snapshot_vocab(stage) is None


def test_the_gate_reads_the_vocabulary_from_the_text_config_when_there_is_one(tmp_path):
    run, stage, snap = make_run(tmp_path, SAMPLE_OK)
    (snap / "config.json").write_text(json.dumps({"vocab_size": 5, "text_config": {"vocab_size": VOCAB}}))
    assert full_port._snapshot_vocab(stage) == VOCAB
