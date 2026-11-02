# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The embedding kind (issue 376), the retrieval rows (issue 379) and media in a pair file (issue 377) of the scoring check. The template
runs a fake port on the CPU with a fake chip; the gate recomputes every number from the record and the stage 1 files."""
import importlib.util
import json
import os
import random

import pytest

from orchard import scoring_check as sc
from scoring_check_fakes import HONEST, make_run, pair_rows, run_template
from scoring_embed_fakes import (EMBED_DIM, HONEST_EMBED, MEDIA_EMBED, MEDIA_SCORE, NOISY_EMBED, N_EMBED, embed_rows, fake_vector, make_embed_run,
                                 media_entry)

spec = importlib.util.spec_from_file_location("scoring_check_template", sc.TEMPLATE)
tpl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpl)

PNG = b"\x89PNG\r\n\x1a\n" + b"fake image bytes" * 20


def gate(run, stage, kind="embedding"):
    return sc.check(stage, run, kind)[0]


def record(stage):
    return json.loads((stage / "evidence" / "scoring-check.json").read_text())


def write_record(stage, rec):
    (stage / "evidence" / "scoring-check.json").write_text(json.dumps(rec))


def mentions(reasons, word):
    assert reasons, "the gate passed a result it must refuse"
    assert [r for r in reasons if word in r], (word, reasons)


def embed_run(tmp_path, port=HONEST_EMBED, **kw):
    run, stage, _ = make_embed_run(tmp_path, port, **kw)
    return run, stage, run_template(tmp_path, run, stage)


# ---- the honest run ---------------------------------------------------------------------------------------------------------

def test_an_honest_embedding_port_gets_a_record_with_its_vectors_and_the_gate_passes_it(tmp_path):
    run, stage, done = embed_run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    rec = record(stage)
    assert rec["kind"] == "embedding" and rec["dim"] == EMBED_DIM and len(rec["chip_vectors"]) == N_EMBED
    assert rec["chip_vectors"][3] == fake_vector(3) and "chip_scores" not in rec
    m = rec["metrics"]
    assert m["count"] == N_EMBED and m["cosine_mean"] == pytest.approx(1.0) and m["cosine_min"] == pytest.approx(1.0)
    assert set(m["by_kind"]) == {"text", "multilingual", "code", "long", "document"} and m["by_kind"]["code"]["count"] == 6
    assert gate(run, stage) == []
    assert sc.check(stage, run)[0] == []                 # the kind is the config's when the caller names none


def test_the_gate_is_given_the_kind_and_a_record_of_the_other_kind_is_refused(tmp_path):
    run, stage, done = embed_run(tmp_path)
    assert done.returncode == 0
    mentions(gate(run, stage, "reranker"), "kind must be 'reranker'")
    mentions(sc.check(stage, run, "generative")[0], "model kind must be one of")


# ---- bounds -----------------------------------------------------------------------------------------------------------------

def test_a_port_whose_vectors_point_elsewhere_is_refused_on_the_mean_and_the_minimum(tmp_path):
    run, stage, done = embed_run(tmp_path, NOISY_EMBED)
    assert done.returncode == 0
    reasons = gate(run, stage)
    mentions(reasons, "mean cosine to the reference vector for all inputs")
    mentions(reasons, "minimum cosine to the reference vector for all inputs")


def test_one_input_kind_that_is_wrong_is_refused_by_name_even_when_the_others_are_exact(tmp_path):
    # kind "code" is every fifth input starting at 2; the port returns a different vector for those only.
    port = HONEST_EMBED.replace("    r = random.Random(ids[0] - 1)\n", "    r = random.Random(ids[0] - 1 + (500 if (ids[0] - 1) % 5 == 2 else 0))\n")
    run, stage, done = embed_run(tmp_path, port)
    assert done.returncode == 0
    reasons = gate(run, stage)
    mentions(reasons, "for kind code")
    assert not [r for r in reasons if "for kind text" in r or "for kind long" in r]


def test_a_set_with_too_few_inputs_or_kinds_is_refused(tmp_path):
    rows = embed_rows()[:10]                              # ten inputs of five kinds
    run, stage, _ = make_embed_run(tmp_path, rows=rows)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0
    mentions(gate(run, stage), "inputs; at least 24")
    rows = [dict(r, kind="text" if r["kind"] != "code" else "code") for r in embed_rows()]       # thirty inputs of two kinds
    run, stage, _ = make_embed_run(tmp_path / "b", rows=rows)
    assert run_template(tmp_path / "b", run, stage).returncode == 0
    mentions(gate(run, stage), "input kinds; at least 3")


def test_a_config_may_tighten_an_embedding_bound_and_never_loosen_one(tmp_path):
    cfg = sc.draft_config("/x", "embedding")
    cfg["bounds"]["cosine_min_min"] = 0.999999
    run, stage, _ = make_embed_run(tmp_path, NOISY_EMBED.replace("0.5", "0.01"), config=dict(cfg, model_snapshot=str(tmp_path / "snapshot")))
    assert run_template(tmp_path, run, stage).returncode == 0
    mentions(gate(run, stage), "below the minimum of 0.999999")
    loose = sc.draft_config(str(tmp_path / "snapshot"), "embedding")
    loose["bounds"]["cosine_mean_min"] = 0.5
    run, stage, _ = make_embed_run(tmp_path / "b", config=loose)
    assert run_template(tmp_path / "b", run, stage).returncode == 0
    mentions(gate(run, stage), "looser than the shipped 0.995")


def test_a_reranker_bound_is_an_unknown_key_in_an_embedding_config(tmp_path):
    cfg = sc.draft_config(str(tmp_path / "snapshot"), "embedding")
    cfg["bounds"]["pearson_min"] = 0.999
    run, stage, _ = make_embed_run(tmp_path, config=cfg)
    assert run_template(tmp_path, run, stage).returncode == 0
    mentions(gate(run, stage), "unknown key 'pearson_min'")


# ---- the port's answer -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("body,word", [
    ("[1.0] * 7", "length 7"),
    ("[float('nan')] * 8", "not a finite number"),
    ("[float('inf')] * 8", "not a finite number"),
    ("[0.0] * 8", "all zeros"),
    ("['a'] * 8", "not a finite number"),
    ("[True] * 8", "not a finite number"),
    ("None", "the port failed at 'embed'"),
    ("5", "the port failed at 'embed'"),
    ("[10 ** 400] * 8", "not a finite number"),
    ("[1e9] * 8", "not a finite number"),
])
def test_a_port_that_returns_a_bad_vector_fails_the_run_and_leaves_no_record(tmp_path, body, word):
    port = HONEST_EMBED.replace("return [r.uniform(-1, 1) for _ in range(8)]", f"return {body}")
    run, stage, done = embed_run(tmp_path, port)
    assert done.returncode == 2 and word in done.stdout, done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


def test_a_port_with_no_embed_function_fails_the_run(tmp_path):
    run, stage, done = embed_run(tmp_path, HONEST_EMBED.replace("def embed(", "def other("))
    assert done.returncode == 2 and "embed" in done.stdout


def test_a_port_may_return_a_numpy_style_array_with_tolist(tmp_path):
    port = HONEST_EMBED.replace("return [r.uniform(-1, 1) for _ in range(8)]",
                                "class A(list):\n        def tolist(self):\n            return list(self)\n    return A(r.uniform(-1, 1) for _ in range(8))")
    run, stage, done = embed_run(tmp_path, port)
    assert done.returncode == 0, done.stdout
    assert gate(run, stage) == []


# ---- the gate recomputes ----------------------------------------------------------------------------------------------------

def test_a_chip_vector_changed_without_its_metrics_is_refused(tmp_path):
    run, stage, _ = embed_run(tmp_path)
    rec = record(stage)
    rec["chip_vectors"][0] = [-x for x in rec["chip_vectors"][0]]
    write_record(stage, rec)
    reasons = gate(run, stage)
    mentions(reasons, "differs from the")
    mentions(reasons, "cosine_mean")


def test_metrics_changed_to_pass_a_failing_run_are_refused(tmp_path):
    run, stage, _ = embed_run(tmp_path, NOISY_EMBED)
    rec = record(stage)
    rec["metrics"]["cosine_mean"] = rec["metrics"]["cosine_min"] = 1.0
    for v in rec["metrics"]["by_kind"].values():
        v["mean"] = v["min"] = 1.0
    write_record(stage, rec)
    reasons = gate(run, stage)
    mentions(reasons, "metrics.cosine_mean")
    mentions(reasons, "metrics.by_kind")
    mentions(reasons, "below the minimum")


@pytest.mark.parametrize("field,value", [
    ("chip_vectors", None), ("chip_vectors", "x"), ("chip_vectors", [None] * N_EMBED), ("chip_vectors", [[1.0]] * N_EMBED),
    ("chip_vectors", [[0.0] * EMBED_DIM] * N_EMBED), ("chip_vectors", [[10 ** 400] * EMBED_DIM] * N_EMBED), ("chip_vectors", [[1.0] * EMBED_DIM] * 3),
    ("metrics", None), ("metrics", {"by_kind": None}), ("metrics", {"by_kind": {"text": 3}, "count": 30}), ("input_kinds", None), ("input_kinds", [1] * N_EMBED),
])
def test_a_record_field_of_the_wrong_type_is_a_reason_and_never_an_exception(tmp_path, field, value):
    run, stage, _ = embed_run(tmp_path)
    rec = record(stage)
    rec[field] = value
    write_record(stage, rec)
    assert gate(run, stage)


def test_the_gate_judges_the_reference_on_disk_and_not_the_one_in_the_record(tmp_path):
    run, stage, _ = embed_run(tmp_path)
    ref = run / "stages" / "1" / "evidence" / "reference" / "scoring-reference.json"
    doc = json.loads(ref.read_text())
    doc["vectors"][0] = fake_vector(99)
    ref.write_text(json.dumps(doc))
    mentions(gate(run, stage), "reference.sha256 does not match")


def test_a_record_that_regroups_the_input_kinds_is_refused(tmp_path):
    run, stage, _ = embed_run(tmp_path)
    rec = record(stage)
    rec["input_kinds"] = ["text"] * N_EMBED
    write_record(stage, rec)
    mentions(gate(run, stage), "input_kinds are not the input kinds")


@pytest.mark.parametrize("vectors", [[], [[]], [[0.0] * EMBED_DIM] * N_EMBED, [[1.0] * EMBED_DIM] * 3, [[1.0] * EMBED_DIM] * (N_EMBED - 1) + [[float("nan")] * EMBED_DIM]])
def test_a_reference_of_the_wrong_shape_fails_the_run_with_a_message(tmp_path, vectors):
    run, stage, _ = make_embed_run(tmp_path, vectors=vectors)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "FAILED" in done.stdout and "Traceback" not in done.stdout + done.stderr
    assert not (stage / "evidence" / "scoring-check.json").exists()


def test_the_templates_numbers_and_the_gates_numbers_are_the_same_for_vectors():
    rnd = random.Random(11)
    kinds = [["a", "b", "c"][i % 3] for i in range(12)]
    ref = [[rnd.uniform(-1, 1) for _ in range(16)] for _ in range(12)]
    for chip in ([[x + rnd.uniform(-0.1, 0.1) for x in v] for v in ref], [list(v) for v in ref], [[-x for x in v] for v in ref], [[1e-200] * 16 for _ in ref]):
        assert tpl.embed_numbers(ref, chip, kinds) == sc.embed_numbers(ref, chip, kinds)
    assert tpl.cosine([0.0, 0.0], [1.0, 2.0]) == sc.cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert tpl.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0 and tpl.cosine([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)
    assert tpl.cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert tpl.MAX_DIM == sc.MAX_DIM and tpl.MAX_VECTOR_NUMEL == sc.MAX_VECTOR_NUMEL and tpl.RETRIEVAL_K == sc.RETRIEVAL_K
    assert (tpl.MEDIA_DIR, tpl.MEDIA_KINDS, tpl.MEDIA_PER_ROW, tpl.MEDIA_FILES_MAX) == (sc.MEDIA_DIR, sc.MEDIA_KINDS, sc.MEDIA_PER_ROW, sc.MEDIA_FILES_MAX)


def test_an_embedding_bound_set_is_floors_only_and_matches_the_cited_numbers():
    assert set(sc.EMBEDDING_BOUNDS) == set(sc._EMBEDDING_FLOORS)
    assert sc.EMBEDDING_BOUNDS["cosine_mean_min"] < 0.9960 and sc.EMBEDDING_BOUNDS["cosine_min_min"] < 0.9926   # below the public bundle's numbers


# ---- retrieval rows (issue 379) --------------------------------------------------------------------------------------------

def test_ndcg_and_hit_at_one_on_a_small_case_worked_by_hand():
    groups = ["a", "a", "a", "b", "b", "c"]
    relevant = [True, False, False, False, True, False]
    ref = [3.0, 2.0, 1.0, 3.0, 1.0, 5.0]               # a: relevant first; b: relevant second; c: no relevant document, left out
    chip = [1.0, 2.0, 3.0, 1.0, 3.0, 5.0]              # a: relevant last (rank 3); b: relevant first
    r = sc.retrieval(ref, chip, groups, relevant)
    assert r["queries"] == 2 and r["k"] == 10
    assert r["ndcg_reference"] == pytest.approx((1.0 + 1 / 1.5849625007211563) / 2)
    assert r["ndcg_chip"] == pytest.approx((0.5 + 1.0) / 2)
    assert r["hit1_reference"] == 0.5 and r["hit1_chip"] == 0.5 and r["hit1_delta"] == 0.0
    assert r["ndcg_delta"] == pytest.approx(r["ndcg_chip"] - r["ndcg_reference"])
    assert tpl.retrieval(ref, chip, groups, relevant) == r
    assert sc.retrieval(ref, chip, ["a"] * 6, [False] * 6) is None


def test_a_tie_keeps_the_earlier_document_first_in_the_ranking_and_at_the_top():
    assert sc.ndcg_at_k([1.0, 1.0], [False, True], [0, 1]) == pytest.approx(1 / 1.5849625007211563)
    assert tpl.ndcg_at_k([1.0, 1.0], [False, True], [0, 1]) == sc.ndcg_at_k([1.0, 1.0], [False, True], [0, 1])


def labelled_rows():
    return [dict(r, relevant=(i % 4 == 0)) for i, r in enumerate(pair_rows())]       # the first document of each group scores highest


def test_a_labelled_pair_file_gives_a_record_with_retrieval_rows_and_the_gate_passes_it(tmp_path):
    run, stage, _ = make_run(tmp_path, HONEST, rows=labelled_rows())
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout
    rec = record(stage)
    assert rec["retrieval"]["queries"] == 8 and rec["retrieval"]["ndcg_chip"] == 1.0 and rec["retrieval"]["hit1_chip"] == 1.0
    assert gate(run, stage, "reranker") == []


def test_a_pair_file_without_labels_has_no_retrieval_rows_and_none_are_required(tmp_path):
    run, stage, _ = make_run(tmp_path, HONEST)
    assert run_template(tmp_path, run, stage).returncode == 0
    assert "retrieval" not in record(stage) and gate(run, stage, "reranker") == []


def test_a_port_that_ranks_a_relevant_document_second_in_one_group_loses_ndcg_and_is_refused(tmp_path):
    # group g0's first document gets a score below the second's: the relevant one falls to rank 2.
    port = HONEST.replace("[(ids[-1] - 100) / 10 for ids in pairs]",
                          "[(ids[-1] - 100) / 10 - (4.0 if ids[0] == 1 and ids[1] == 1 else 0.0) for ids in pairs]")
    run, stage, _ = make_run(tmp_path, port, rows=labelled_rows())
    assert run_template(tmp_path, run, stage).returncode == 0
    reasons = gate(run, stage, "reranker")
    mentions(reasons, "NDCG@10 of the chip")
    assert not [r for r in reasons if "Hit@1" in r]          # one group in eight is 0.125, inside the 0.15 allowed


def test_retrieval_rows_changed_in_the_record_or_removed_are_refused(tmp_path):
    run, stage, _ = make_run(tmp_path, HONEST, rows=labelled_rows())
    assert run_template(tmp_path, run, stage).returncode == 0
    rec = record(stage)
    good = json.loads(json.dumps(rec))
    rec["retrieval"]["ndcg_chip"] = 0.5
    write_record(stage, rec)
    mentions(gate(run, stage, "reranker"), "retrieval.ndcg_chip")
    rec = json.loads(json.dumps(good))
    del rec["retrieval"]
    write_record(stage, rec)
    mentions(gate(run, stage, "reranker"), "reports no retrieval object")
    rec = json.loads(json.dumps(good))
    rec["retrieval"] = "x"
    write_record(stage, rec)
    mentions(gate(run, stage, "reranker"), "reports no retrieval object")


def test_a_relevant_label_of_the_wrong_type_fails_the_run(tmp_path):
    rows = pair_rows()
    rows[0]["relevant"] = "yes"
    run, stage, _ = make_run(tmp_path, HONEST, rows=rows)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "relevant must be true or false" in done.stdout


def test_a_retrieval_bound_may_be_tightened_and_not_loosened(tmp_path):
    cfg = sc.draft_config(str(tmp_path / "snapshot"))
    cfg["bounds"]["ndcg_drop_max"] = 0.5
    run, stage, _ = make_run(tmp_path, HONEST, rows=labelled_rows(), config=cfg)
    assert run_template(tmp_path, run, stage).returncode == 0
    mentions(gate(run, stage, "reranker"), "ndcg_drop_max 0.5 is looser")


# ---- media (issue 377) ------------------------------------------------------------------------------------------------------

def media_rows(name="a.png", data=PNG, every=5):
    return embed_rows({n: [media_entry(name, data)] for n in range(0, N_EMBED, every)})


def test_a_port_is_given_the_bytes_of_each_media_file_for_the_inputs_that_name_one(tmp_path):
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=media_rows(), media_files={"a.png": PNG})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout
    assert gate(run, stage) == []


def test_a_port_that_does_not_take_a_media_argument_is_still_called_when_no_row_has_media(tmp_path):
    run, stage, done = embed_run(tmp_path)         # HONEST_EMBED has embed(model, ids) and nothing else
    assert done.returncode == 0 and gate(run, stage) == []


def test_a_text_only_port_given_media_fails_the_run_with_the_ports_message(tmp_path):
    run, stage, _ = make_embed_run(tmp_path, HONEST_EMBED, rows=media_rows(), media_files={"a.png": PNG})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "media" in done.stdout


def test_a_reranker_port_is_given_one_media_list_for_each_prompt(tmp_path):
    rows = pair_rows()
    rows = [dict(r, media=[media_entry("a.png", PNG)]) if i % 4 == 1 else r for i, r in enumerate(rows)]
    run, stage, _ = make_run(tmp_path, MEDIA_SCORE, rows=rows)
    ev = run / "stages" / "1" / "evidence" / "reference" / "media"
    ev.mkdir()
    (ev / "a.png").write_bytes(PNG)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout
    assert gate(run, stage, "reranker") == []


@pytest.mark.parametrize("mutate,word", [
    (lambda e: e.update(sha256="0" * 64), "does not match the sha256 the pair file names"),          # the parent's check, before the port runs
    (lambda e: e.update(path="stages/1/evidence/reference/media/../scoring-pairs.json"), "cannot be read"),
    (lambda e: e.update(path="/etc/passwd"), "cannot be read"),
    (lambda e: e.update(path="stages/1/evidence/reference/media/missing.png"), "cannot be read"),
    (lambda e: e.update(kind="audio"), "needs a kind"),
    (lambda e: e.pop("sha256"), "needs a kind"),
    (lambda e: e.update(path=None), "needs a kind"),
])
def test_a_media_entry_that_is_wrong_fails_the_run_with_a_message(tmp_path, mutate, word):
    rows = media_rows()
    mutate(rows[0]["media"][0])
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=rows, media_files={"a.png": PNG})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and word in done.stdout and "Traceback" not in done.stdout + done.stderr, done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


@pytest.mark.parametrize("media", ["x", {"a": 1}, [1], [[]], [None]])
def test_a_media_field_of_the_wrong_shape_fails_the_run_with_a_message(tmp_path, media):
    rows = embed_rows({0: media})
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=rows)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "media" in done.stdout and "Traceback" not in done.stdout + done.stderr


def test_more_media_entries_than_a_row_may_have_fail_the_run(tmp_path):
    rows = embed_rows({0: [media_entry("a.png", PNG)] * (tpl.MEDIA_PER_ROW + 1)})
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=rows, media_files={"a.png": PNG})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "at most 8 entries" in done.stdout


def test_more_media_files_than_the_set_may_have_fail_the_run(tmp_path):
    rows = embed_rows({n: [media_entry("a.png", PNG)] * 3 for n in range(N_EMBED)})          # 90 entries, three in each row
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=rows, media_files={"a.png": PNG})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "more than 64 media files" in done.stdout


def test_a_media_file_over_the_size_limit_fails_the_run_without_reading_it_all(tmp_path):
    big = b"x" * (tpl.MEDIA_MAX + 1)
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=media_rows("big.png", big), media_files={"big.png": big})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "larger than" in done.stdout


def test_a_media_file_that_is_a_link_or_a_fifo_fails_the_run_without_hanging(tmp_path):
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=media_rows(), media_files={"real.png": PNG})
    media = run / "stages" / "1" / "evidence" / "reference" / "media"
    os.symlink(media / "real.png", media / "a.png")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "cannot be read" in done.stdout
    os.unlink(media / "a.png")
    os.mkfifo(media / "a.png")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "cannot be read" in done.stdout


def test_a_media_directory_that_is_a_link_out_of_the_run_fails_the_run(tmp_path):
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=media_rows())
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.png").write_bytes(PNG)
    os.symlink(outside, run / "stages" / "1" / "evidence" / "reference" / "media")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "cannot be read" in done.stdout


def test_a_media_file_changed_after_the_run_is_refused_by_the_gate(tmp_path):
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=media_rows(), media_files={"a.png": PNG})
    assert run_template(tmp_path, run, stage).returncode == 0
    (run / "stages" / "1" / "evidence" / "reference" / "media" / "a.png").write_bytes(PNG + b"!")
    mentions(gate(run, stage), "does not match its sha256 as it is on disk now")


def test_a_media_file_that_is_removed_or_replaced_by_a_link_after_the_run_is_refused_by_the_gate(tmp_path):
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=media_rows(), media_files={"a.png": PNG, "b.png": PNG})
    assert run_template(tmp_path, run, stage).returncode == 0
    path = run / "stages" / "1" / "evidence" / "reference" / "media" / "a.png"
    path.unlink()
    mentions(gate(run, stage), "not a regular file directly under")
    os.symlink(tmp_path / "elsewhere", path)
    mentions(gate(run, stage), "not a regular file directly under")


def test_a_media_entry_that_leaves_the_media_directory_in_the_pair_file_is_refused_by_the_gate(tmp_path):
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=media_rows(), media_files={"a.png": PNG})
    assert run_template(tmp_path, run, stage).returncode == 0
    ev = run / "stages" / "1" / "evidence" / "reference"
    doc = json.loads((ev / "scoring-pairs.json").read_text())
    doc["pairs"][0]["media"][0]["path"] = "stages/1/evidence/reference/scoring-reference.json"
    (ev / "scoring-pairs.json").write_text(json.dumps(doc))
    reasons = gate(run, stage)
    mentions(reasons, "needs a kind")
    mentions(reasons, "pairs.sha256 does not match")
