# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 406: serve_and_compare.py reads the token the server generated back from the text the server returns. That fails for a token
that is half of a character, for a token that does not re-encode to itself, and for a stop token the server leaves out of the text. A
correct chip token was recorded as a mismatch. The reference token is known, so the template marks such positions `skipped` and the
scoring leaves them out of both the matches and the count. A mismatch on any other position still counts."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weights_swap_templates import GENERATED, NEAREST, TEMPLATES, VOCAB, swap, report  # noqa: E402,F401  (swap is a fixture)
from test_agreement_set_template import IDS, gate_reasons, install, make_sets, run, server_sets  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402

from orchard import agreement_core as core  # noqa: E402

UNK, PARTIAL, SPLIT, EOS = len(VOCAB), len(VOCAB) + 1, len(VOCAB) + 2, len(VOCAB) + 3
FULL_VOCAB = VOCAB + ["[UNK]", "�", "a b", "<eos>"]
SILENT = [PARTIAL, EOS]            # the server's text for these is empty: half a character, and a stop token left out of the response


def rich_tokenizer(model_dir: Path, special_eos: bool = True) -> None:
    """The fixture's word-level tokenizer plus: a token that decodes to a replacement character (half of a character), a token whose
    text re-encodes to other ids (it holds a space, and the pre-tokenizer splits there), and a special stop token."""
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(FULL_VOCAB)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    if special_eos:
        tok.add_special_tokens(["<eos>"])
    tok.save(str(model_dir / "tokenizer.json"))


def with_positions(ids: list[int], at: dict[int, int]) -> list[int]:
    out = list(ids)
    for k, v in at.items():
        out[k] = v
    return out


TRICKY = {5: PARTIAL, 11: SPLIT, 20: EOS}          # three positions of 32 that the text cannot give back


def start_tricky(swap, *, positions=None, special_eos=True, silent=None, **overrides):
    """Stage 1's reference for the main prompt holds the tricky ids; the server answers each from the vocabulary and says nothing for
    the silent ones."""
    gen = with_positions(GENERATED, positions or TRICKY)
    rich_tokenizer(swap["stage"] / "model-dir", special_eos)
    (swap["run"] / "stages" / "1" / "evidence" / "reference" / "generated-ids.json").write_text(
        json.dumps({"generated_ids": gen, "generated_text": " ".join(FULL_VOCAB[i] for i in gen)}))
    return gen, swap["start"]("perfect", server_extra={"vocab": FULL_VOCAB, "generated_ids": gen, "silent_ids": SILENT if silent is None else silent}, **overrides)


# ---- the 32-token check -----------------------------------------------------------------------------------------------------------------

def test_positions_the_text_cannot_give_back_are_skipped_not_counted_as_mismatches(swap):
    gen, r = start_tricky(swap)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    assert rep["positions_skipped"] == [5, 11, 20]
    assert rep["mismatches"] == [] and rep["matches"] == 29 and rep["n_tokens"] == 29 and rep["top1_agreement"] == 1.0
    draft = rep["result_draft"]
    assert draft["n_tokens"] == 29 and draft["top1_agreement"] == 1.0
    assert [f.get("skipped") for f in rep["teacher_forced"]].count(True) == 3


def test_a_wrong_token_on_a_readable_position_still_counts_as_a_mismatch(swap):
    gen, r = start_tricky(swap, positions={5: PARTIAL})
    r = swap["start"]("every4", server_extra={"vocab": FULL_VOCAB, "generated_ids": gen, "silent_ids": SILENT})
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    assert rep["positions_skipped"] == [5] and len(rep["mismatches"]) == 8 and rep["n_tokens"] == 31
    assert rep["matches"] == 23 and rep["top1_agreement"] == pytest.approx(23 / 31)


def test_a_run_with_nothing_unreadable_records_no_skipped_positions_and_the_old_count(swap):
    r = swap["start"]("perfect")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    assert rep["positions_skipped"] == [] and rep["n_tokens"] == 32 and rep["top1_agreement"] == 1.0


def test_a_token_that_re_encodes_to_other_ids_is_skipped_even_when_it_is_not_special(swap):
    gen, r = start_tricky(swap, positions={11: SPLIT}, special_eos=False)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    assert rep["positions_skipped"] == [11] and rep["mismatches"] == []


def test_a_stop_id_from_generation_config_is_skipped_even_when_it_reads_back_fine(swap):
    word = 1                                         # an ordinary word (the fixture reference never holds id 1): it re-encodes to itself, so only the eos rule can skip it
    (swap["stage"] / "model-dir" / "generation_config.json").write_text(json.dumps({"eos_token_id": [word]}))
    gen, r = start_tricky(swap, positions={20: word}, special_eos=False, silent=[word])
    assert r.returncode == 0, r.stdout + r.stderr
    assert report(swap)["positions_skipped"] == [20]


def test_the_same_ordinary_word_is_a_mismatch_when_nothing_marks_it_as_a_stop_id(swap):
    word = 1
    gen, r = start_tricky(swap, positions={20: word}, special_eos=False, silent=[word])
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    assert rep["positions_skipped"] == [] and [m["position"] for m in rep["mismatches"]] == [20]


def test_too_few_readable_positions_leave_a_count_the_gate_refuses(swap):
    from orchard import stages
    gen, r = start_tricky(swap, positions={k: PARTIAL for k in range(0, 24)})
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    assert rep["n_tokens"] == 8 and len(rep["positions_skipped"]) == 24
    reasons = stages._swap_reasons(rep["result_draft"] | {"serves": True, "coherent": True, "server_ready_s": 1.0})
    assert any("n_tokens must be a whole number of at least" in x for x in reasons), reasons


# ---- the agreement set ------------------------------------------------------------------------------------------------------------------

def tricky_sets(n=6):
    sets = make_sets(n)
    sets[0]["generated_ids"] = with_positions(sets[0]["generated_ids"], TRICKY)
    return sets


def start_set(swap, sets):
    rich_tokenizer(swap["stage"] / "model-dir")
    install(swap, sets)
    extra = {"vocab": FULL_VOCAB, "silent_ids": SILENT, "agreement_sets": server_sets(sets)}
    return swap["start"]("perfect", server_extra=extra, agreement_set=True)


def test_the_agreement_set_skips_the_unreadable_positions_and_the_gate_scores_the_rest(swap):
    sets = tricky_sets()
    r = start_set(swap, sets)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["agreement_set"]
    first = rec["entries"][0]
    assert first["skipped"] == [5, 11, 20] and all(e["skipped"] == [] for e in rec["entries"][1:])
    assert gate_reasons(swap) == []
    summary = core.judge(rec)["summary"]
    assert summary["per_prompt"][0]["n"] == 29 and summary["per_prompt"][0]["top1"] == 1.0 and summary["per_prompt"][0]["skipped"] == 3
    assert summary["pooled_top1"] == 1.0


def test_an_edited_record_that_skips_too_many_positions_is_refused(swap):
    sets = tricky_sets()
    assert start_set(swap, sets).returncode == 0
    path = swap["stage"] / "evidence" / "swap-check.json"
    rep = json.loads(path.read_text())
    rep["agreement_set"]["entries"][1]["skipped"] = list(range(0, 32, 2)) + [1]        # 17 of 32: a record that hides its mismatches
    path.write_text(json.dumps(rep))
    assert any(IDS[1] in x and "not recoverable" in x for x in gate_reasons(swap)), gate_reasons(swap)


# ---- the arithmetic ---------------------------------------------------------------------------------------------------------------------

def test_score_prompt_leaves_skipped_positions_out_of_both_sides():
    ref = list(range(10))
    chip = list(ref)
    chip[3] = None                                  # the false mismatch
    assert core.score_prompt(ref, chip)["top1"] == 0.9
    s = core.score_prompt(ref, chip, skipped=[3])
    assert (s["n"], s["matches"], s["top1"], s["skipped"], s["first_mismatch"]) == (9, 9, 1.0, 1, None)


def test_a_mismatch_on_an_unskipped_position_is_still_found():
    ref = list(range(10))
    chip = list(ref)
    chip[3], chip[7] = None, 99
    s = core.score_prompt(ref, chip, skipped=[3])
    assert (s["n"], s["matches"], s["first_mismatch"]) == (9, 8, 7)


@pytest.mark.parametrize("skipped", [[10], [-1], [3, 3], [True], ["3"], 3, [3.0], list(range(0, 10, 2)) + [1]])
def test_a_malformed_or_oversize_skipped_list_is_refused(skipped):
    ref = list(range(10))
    with pytest.raises(ValueError):
        core.score_prompt(ref, list(ref), skipped=skipped)


def test_summarise_and_the_damage_control_keep_the_skipped_positions():
    ref = list(range(40))
    chip = list(ref)
    entries = [{"id": f"p{i}", "kind": "chat", "reference_ids": ref, "chip_ids": list(chip), "skipped": [3, 4, 5]} for i in range(5)]
    assert core.summarise(entries)["per_prompt"][0]["n"] == 37
    assert core.calibrate(entries)["usable"] is True
    j = core.judge({"ran": True, "entries": entries, "control": None})
    assert j["summary"]["pooled_top1"] == 1.0


def test_the_damage_control_damages_compared_positions_even_when_skipped_ones_line_up_with_the_damage():
    ref = list(range(40))
    skipped = list(range(3, 40, 4))                 # exactly the positions a plain every-fourth damage would hit: 10 of 40 is the cap
    entries = [{"id": f"p{i}", "kind": "chat", "reference_ids": ref, "chip_ids": list(ref), "skipped": skipped} for i in range(5)]
    cal = core.calibrate(entries)
    assert cal["usable"] is True
    assert cal["damaged_pooled_top1"] == pytest.approx(23 / 30)         # every 4th of the 30 compared positions is wrong: 7 of 30


def test_a_control_that_skipped_other_positions_cannot_be_used():
    ref = list(range(40))
    mine = [{"id": "p0", "kind": "chat", "reference_ids": ref, "chip_ids": list(ref), "skipped": [3]}]
    same = [{"id": "p0", "reference_ids": ref, "chip_ids": list(ref), "skipped": [3]}]
    other = [{"id": "p0", "reference_ids": ref, "chip_ids": list(ref), "skipped": [4]}]
    assert core.control_problems(same, mine) == []
    assert any("skipped" in x for x in core.control_problems(other, mine))


def test_the_template_copy_of_the_core_is_still_identical():
    assert (TEMPLATES / "agreement_core.py").read_bytes() == (Path(core.__file__)).read_bytes()


# ---- Copilot round 1 on PR 422 ----------------------------------------------------------------------------------------------------------

def edit(swap, change):
    path = swap["stage"] / "evidence" / "swap-check.json"
    rep = json.loads(path.read_text())
    change(rep)
    path.write_text(json.dumps(rep))


def test_an_edited_record_that_marks_readable_mismatches_as_skipped_is_refused(swap):
    """Every fourth token wrong is 8 of 32 per prompt. Marking those eight positions skipped satisfies the quarter cap and lifts the
    agreement from 0.75 to 1.0, so the gate has to recompute the mask from the reference and the tokenizer."""
    sets = make_sets(wrong_every=4)
    rich_tokenizer(swap["stage"] / "model-dir")
    install(swap, sets)
    extra = {"vocab": FULL_VOCAB, "silent_ids": SILENT, "agreement_sets": server_sets(sets)}
    assert swap["start"]("perfect", server_extra=extra, agreement_set=True).returncode == 0
    assert any("pooled top-1" in x for x in gate_reasons(swap))
    wrong = list(range(3, 32, 4))

    def hide(rep):
        for e in rep["agreement_set"]["entries"]:
            e["skipped"] = wrong
    edit(swap, hide)
    reasons = gate_reasons(swap)
    assert any("skipped positions" in x and "tokenizer" in x for x in reasons), reasons


def test_an_edited_record_that_clears_the_mask_of_an_unreadable_token_is_refused_too(swap):
    sets = tricky_sets()
    assert start_set(swap, sets).returncode == 0
    edit(swap, lambda rep: rep["agreement_set"]["entries"][0].update(skipped=[]))
    assert any("skipped positions" in x for x in gate_reasons(swap))


def test_a_mask_the_gate_cannot_check_because_the_tokenizer_is_missing_is_refused(swap):
    sets = tricky_sets()
    assert start_set(swap, sets).returncode == 0
    (swap["stage"] / "model-dir" / "tokenizer.json").unlink()
    assert any("skipped positions" in x and "cannot be checked" in x for x in gate_reasons(swap)), gate_reasons(swap)


def start_set_with_control(swap, sets, control):
    extra = {"vocab": FULL_VOCAB, "silent_ids": SILENT, "agreement_sets": server_sets(sets)}
    return swap["start"]("perfect", server_extra=extra, agreement_set=True, agreement_control=str(control))


def test_the_control_mask_survives_the_reload_and_is_checked_like_the_runs(swap, tmp_path):
    sets = tricky_sets()
    assert start_set(swap, sets).returncode == 0
    rep = json.loads((swap["stage"] / "evidence" / "swap-check.json").read_text())
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"agreement_set": {**rep["agreement_set"], "model_id": NEAREST}}))
    assert start_set_with_control(swap, sets, path).returncode == 0
    got = report(swap)["agreement_set"]["control"]["entries"]
    assert got[0]["skipped"] == [5, 11, 20], got[0]
    assert not any("skipped" in x for x in gate_reasons(swap)), gate_reasons(swap)
    edit(swap, lambda r: r["agreement_set"]["control"]["entries"][0].update(skipped=[]))
    assert any("control" in x and "skipped positions" in x for x in gate_reasons(swap)), gate_reasons(swap)


def test_a_control_that_has_no_chip_id_at_a_skipped_position_can_be_used():
    ref = list(range(40))
    chip = list(ref)
    chip[3] = None                                  # a stop token answered with empty text leaves no id
    mine = [{"id": "p0", "reference_ids": ref, "chip_ids": list(ref), "skipped": [3]}]
    ctl = [{"id": "p0", "reference_ids": ref, "chip_ids": chip, "skipped": [3], "error": None}]
    assert core.control_problems(ctl, mine) == []
    chip[4] = None
    assert any("missing chip ids" in x for x in core.control_problems(ctl, mine))


def test_a_control_with_a_failed_request_is_still_refused_when_the_mask_is_nonempty():
    ref = list(range(40))
    mine = [{"id": "p0", "reference_ids": ref, "chip_ids": list(ref), "skipped": [3]}]
    ctl = [{"id": "p0", "reference_ids": ref, "chip_ids": list(ref), "skipped": [3], "error": "HTTP 500"}]
    assert any("failed request" in x for x in core.control_problems(ctl, mine))


@pytest.mark.parametrize("skipped", [[40], [-1], [3, 3], "3", [True]])
def test_a_control_with_a_malformed_mask_is_refused(skipped):
    ref = list(range(40))
    mine = [{"id": "p0", "reference_ids": ref, "chip_ids": list(ref)}]
    ctl = [{"id": "p0", "reference_ids": ref, "chip_ids": list(ref), "skipped": skipped, "error": None}]
    assert any("malformed skipped list" in x for x in core.control_problems(ctl, mine))


def write_config(swap, **doc):
    (swap["stage"] / "model-dir" / "config.json").write_text(json.dumps(doc))


def test_a_stop_id_from_config_json_is_skipped_in_the_32_token_check(swap):
    write_config(swap, eos_token_id=1)
    gen, r = start_tricky(swap, positions={20: 1}, special_eos=False, silent=[1])
    assert r.returncode == 0, r.stdout + r.stderr
    assert report(swap)["positions_skipped"] == [20]


def test_config_json_with_a_list_a_bool_or_text_gives_only_its_whole_number_ids(swap):
    write_config(swap, eos_token_id=[1, True, "2", 3.5, None])
    gen, r = start_tricky(swap, positions={20: 1, 21: 2}, special_eos=False, silent=[1, 2])
    assert r.returncode == 0, r.stdout + r.stderr
    assert report(swap)["positions_skipped"] == [20]


def test_a_stop_id_from_config_json_is_skipped_in_the_agreement_set_and_the_gate_agrees(swap):
    write_config(swap, eos_token_id=1)
    sets = make_sets()
    sets[0]["generated_ids"] = with_positions(sets[0]["generated_ids"], {9: 1})
    rich_tokenizer(swap["stage"] / "model-dir", special_eos=False)
    install(swap, sets)
    extra = {"vocab": FULL_VOCAB, "silent_ids": [1], "agreement_sets": server_sets(sets)}
    assert swap["start"]("perfect", server_extra=extra, agreement_set=True).returncode == 0
    assert report(swap)["agreement_set"]["entries"][0]["skipped"] == [0, 9]     # id 1 is also the reference token at position 0
    assert gate_reasons(swap) == []


def test_the_gates_mask_and_the_templates_mask_are_the_same_function_of_the_tokenizer(tmp_path):
    spec = importlib.util.spec_from_file_location("sac_parity", TEMPLATES / "serve_and_compare.py")
    tpl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpl)
    rich_tokenizer(tmp_path)
    tok = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    ids = list(range(len(FULL_VOCAB)))
    for stops in (set(), {1}, {3, 4}):
        (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": sorted(stops)}))
        want = tpl.unrecoverable_ids(tok, tmp_path, ids)
        assert want == core.unrecoverable_ids(tok, core.stop_ids_from({"eos_token_id": sorted(stops)}), ids)


def test_a_record_whose_mask_mixes_numbers_and_text_is_a_reason_and_not_a_crash(swap):
    sets = tricky_sets()
    assert start_set(swap, sets).returncode == 0
    edit(swap, lambda rep: rep["agreement_set"]["entries"][0].update(skipped=[5, "a", 20]))
    reasons = gate_reasons(swap)
    assert any("skipped" in x for x in reasons), reasons


def test_stop_ids_from_takes_whole_numbers_of_the_top_level_only():
    assert core.stop_ids_from({"eos_token_id": 2}, {"eos_token_id": [3, 4]}) == {2, 3, 4}
    assert core.stop_ids_from({"eos_token_id": True}, {"eos_token_id": [True, 5, "6", 7.5, None]}, {"text_config": {"eos_token_id": 9}}) == {5}
    assert core.stop_ids_from(None, [], "x", {"eos_token_id": {"a": 1}}) == set()
