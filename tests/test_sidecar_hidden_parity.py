"""The device-free parts of hidden_parity.py (orchard/skills/sidecar-parity-templates/).

The script runs on a leased board inside a tt-model bundle's venv. These tests cover everything that
does not need the board: the fixed records, the code hash check, the comparison math, the result
schema, the cache guard, and the lookup of the output-embedding weight. Nothing here imports ttnn,
opens a device or loads a model."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "orchard" / "skills" / "sidecar-parity-templates" / "hidden_parity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hidden_parity_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hp = load_module()


# ---- the fixed records -------------------------------------------------------------------------

def test_there_are_six_to_eight_text_only_records_with_at_least_ten_questions():
    records = hp.build_records()
    assert 6 <= len(records) <= 8
    assert sum(len(r["questions"]) for r in records) >= 10
    for r in records:
        assert "images" not in r and "videos" not in r and "media_kwargs" not in r


def test_every_question_type_the_head_knows_is_used():
    types = {q["type"] for r in hp.build_records() for q in r["questions"].values()}
    assert types == {"noul", "choice", "score"}


def test_the_records_are_valid_for_the_sidecars_own_checks():
    for r in hp.build_records():
        assert hp.validate_record(r) == []


def test_the_record_validator_catches_what_the_sidecar_would_refuse():
    base = {"id": "x", "state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "A"}}}}
    assert hp.validate_record(base) == []
    assert hp.validate_record({**base, "questions": {}})                                   # no questions
    assert any("type must be" in p for p in
               hp.validate_record({**base, "questions": {"q": {"type": "essay", "criteria": {"a": "A"}}}}))
    assert hp.validate_record({**base, "questions": {"q": {"type": "choice"}}})             # no criteria
    assert hp.validate_record({**base, "questions": {"q": {"type": "score", "criteria": []}}})
    assert hp.validate_record({k: v for k, v in base.items() if k != "state"})              # no state
    assert hp.validate_record({**base, "images": [object()]})                               # media


def test_the_records_are_fixed_and_json_serialisable():
    a, b = hp.build_records(), hp.build_records()
    assert a == b and json.loads(json.dumps(a)) == a
    assert len({r["id"] for r in a}) == len(a)


def test_the_records_digest_changes_with_the_records_and_the_layer_limit():
    records = hp.build_records()
    d = hp.records_digest(records, None)
    assert d == hp.records_digest(hp.build_records(), None)
    assert d != hp.records_digest(records[:-1], None) and d != hp.records_digest(records, 4)


# ---- the code hash -----------------------------------------------------------------------------

def test_the_code_hash_check_passes_on_a_match_and_names_both_hashes_on_a_mismatch(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("print('hi')\n")
    good = hp.sha256_file(f)
    assert len(good) == 64 and hp.verify_code_hash(f, good) is None
    with pytest.raises(hp.CodeHashMismatch) as e:
        hp.verify_code_hash(f, "0" * 64)
    assert good in str(e.value) and "0" * 64 in str(e.value)


def test_a_missing_code_file_is_a_mismatch_not_a_crash(tmp_path):
    with pytest.raises(hp.CodeHashMismatch):
        hp.verify_code_hash(tmp_path / "nope.py", "0" * 64)


def test_a_blank_expected_hash_is_refused(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x")
    with pytest.raises(hp.CodeHashMismatch, match="empty"):
        hp.verify_code_hash(f, "")


# ---- the math ----------------------------------------------------------------------------------

def test_pcc_is_one_for_identical_and_minus_one_for_negated_and_ignores_scale_and_shift():
    a = torch.randn(40, 8)
    assert hp.pcc(a, a) == pytest.approx(1.0)
    assert hp.pcc(a, -a) == pytest.approx(-1.0)
    assert hp.pcc(a, 3 * a + 5) == pytest.approx(1.0)


def test_pcc_drops_when_noise_is_added_and_works_on_bfloat16():
    a = torch.randn(64, 32)
    noisy = a + 0.5 * torch.randn_like(a)
    assert 0.5 < hp.pcc(a, noisy) < 0.99
    assert hp.pcc(a.bfloat16(), a.bfloat16()) == pytest.approx(1.0)


def test_pcc_of_constants_is_one_when_equal_and_zero_when_not():
    assert hp.pcc(torch.ones(10), torch.ones(10)) == 1.0
    assert hp.pcc(torch.ones(10), torch.zeros(10)) == 0.0


def test_pcc_refuses_different_shapes():
    with pytest.raises(ValueError):
        hp.pcc(torch.ones(3, 4), torch.ones(4, 3))


def test_cosine_is_scale_free():
    a = torch.randn(100)
    assert hp.cosine(a, 7 * a) == pytest.approx(1.0) and hp.cosine(a, -a) == pytest.approx(-1.0)


def test_probabilities_are_a_softmax_per_question():
    logits = [[torch.tensor([0.0, 0.0]), torch.tensor([10.0, 0.0, 0.0])]]
    probs = hp.probs_from_logits(logits)
    assert probs[0][0] == pytest.approx([0.5, 0.5])
    assert sum(probs[0][1]) == pytest.approx(1.0) and probs[0][1][0] > 0.99


def test_max_abs_prob_diff_looks_at_every_option_of_every_question():
    a = [[[0.5, 0.5], [0.2, 0.8]]]
    b = [[[0.5, 0.5], [0.3, 0.7]]]
    assert hp.max_abs_prob_diff(a, b) == pytest.approx(0.1)
    assert hp.max_abs_prob_diff(a, a) == 0.0


def test_record_metrics_counts_agreeing_top_choices():
    ref_h, dev_h = torch.randn(10, 4), torch.randn(10, 4)
    ref_p = [[[0.9, 0.1], [0.4, 0.6], [0.7, 0.3]]]
    dev_p = [[[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]]]
    m = hp.record_metrics("r1", ref_h, dev_h, ref_p[0], dev_p[0])
    assert m["id"] == "r1" and m["n_questions"] == 3 and m["top1_agree"] == 2 and m["tokens"] == 10
    assert m["prob_max_abs_diff"] == pytest.approx(0.2)
    assert -1.0 <= m["hidden_pcc"] <= 1.0


# ---- the result --------------------------------------------------------------------------------

def per_record(pcc=0.99, agree=3, n=3, diff=0.02):
    return {"id": "r", "tokens": 100, "hidden_pcc": pcc, "prob_max_abs_diff": diff, "top1_agree": agree,
            "n_questions": n}


META = {"tt_metal_sha": "ttnn-0.1", "code_sha256": "a" * 64, "head_sha256": "b" * 64, "mesh_shape": [1, 2]}
WIRING = {"cosine": 0.9995, "passed": True}


def result(**over):
    d = hp.aggregate([per_record(0.99), per_record(0.97, agree=2, diff=0.05)], {"prob_max_abs_diff": 0.001},
                     WIRING, META)
    d.update(over)
    return d


def test_the_aggregate_has_the_documented_fields_and_values():
    d = result()
    assert d["measured"] is True and d["n_records"] == 2 and d["n_questions"] == 6
    assert d["hidden_pcc_min"] == pytest.approx(0.97) and d["hidden_pcc_mean"] == pytest.approx(0.98)
    assert d["prob_max_abs_diff"] == pytest.approx(0.05)
    assert d["top1_agree_fraction"] == pytest.approx(5 / 6)
    assert d["noise_floor"] == {"prob_max_abs_diff": 0.001} and d["wiring_check"] == WIRING
    for k, v in META.items():
        assert d[k] == v


def test_every_number_is_labelled_measured():
    d = result()
    for key in ("hidden_pcc_min", "hidden_pcc_mean", "prob_max_abs_diff", "top1_agree_fraction"):
        assert d["label"][key] == "measured"


def test_the_aggregate_keeps_the_per_record_numbers_for_diagnosis():
    assert [r["id"] for r in result()["per_record"]] == ["r", "r"]


def test_a_good_result_has_no_schema_problems():
    assert hp.validate_result(result()) == []


@pytest.mark.parametrize("change", [
    {"measured": False}, {"n_questions": 0}, {"hidden_pcc_min": "high"}, {"top1_agree_fraction": 1.5},
    {"hidden_pcc_min": 1.5}, {"hidden_pcc_min": True}, {"n_records": True}, {"prob_max_abs_diff": -0.1},
    {"wiring_check": {"cosine": 0.9}}, {"noise_floor": {}}, {"mesh_shape": "1x2"}, {"code_sha256": ""},
])
def test_a_malformed_result_is_reported(change):
    assert hp.validate_result(result(**change))


def test_a_result_with_a_missing_key_is_reported():
    d = result()
    del d["hidden_pcc_min"]
    assert any("hidden_pcc_min" in p for p in hp.validate_result(d))


def test_an_empty_record_list_is_refused():
    with pytest.raises(ValueError, match="no records"):
        hp.aggregate([], {"prob_max_abs_diff": 0.0}, WIRING, META)


def test_evidence_is_written_as_json_and_read_back_equal(tmp_path):
    d = result()
    path = hp.write_evidence(tmp_path / "evidence", d)
    assert path.name == "sidecar-parity.json" and json.loads(path.read_text()) == json.loads(json.dumps(d))


def test_evidence_that_fails_its_own_schema_is_not_written(tmp_path):
    with pytest.raises(ValueError):
        hp.write_evidence(tmp_path / "evidence", result(n_questions=0))
    assert not (tmp_path / "evidence" / "sidecar-parity.json").exists()


# ---- the wiring check and the hidden-state read-back --------------------------------------------

def test_the_wiring_check_passes_when_the_hosts_logits_match_the_devices():
    weight = torch.randn(50, 8)
    row = torch.randn(8)
    tt_logits = weight @ row + 0.001 * torch.randn(50)
    w = hp.wiring_check(row, weight, tt_logits, minimum=0.99)
    assert w["passed"] is True and w["cosine"] > 0.99 and w["minimum"] == 0.99


def test_the_wiring_check_fails_for_a_different_row():
    weight = torch.randn(50, 8)
    w = hp.wiring_check(torch.randn(8), weight, weight @ torch.randn(8), minimum=0.99)
    assert w["passed"] is False


def test_the_wiring_check_ignores_padding_in_the_device_logits():
    weight = torch.randn(50, 8)
    row = torch.randn(8)
    padded = torch.cat([weight @ row, torch.zeros(14)])
    assert hp.wiring_check(row, weight, padded, minimum=0.99)["passed"] is True


def test_a_replicated_read_back_takes_one_replica_and_drops_the_padding():
    t = torch.arange(2 * 1 * 6 * 4, dtype=torch.float32).reshape(2, 1, 6, 4)
    t[1] = -1                                                       # the second replica must be ignored
    out = hp.assemble_hidden(t, n_devices=2, hidden_size=4, valid_len=5)
    assert out.shape == (5, 4) and torch.equal(out, t[0, 0, :5])


def test_a_sharded_read_back_is_joined_along_the_hidden_dimension():
    full = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
    shards = torch.stack([full[:, :4], full[:, 4:]]).reshape(2, 1, 6, 4)
    out = hp.assemble_hidden(shards, n_devices=2, hidden_size=8, valid_len=6)
    assert torch.equal(out, full)


def test_a_read_back_of_an_unexpected_width_is_refused():
    with pytest.raises(ValueError):
        hp.assemble_hidden(torch.zeros(2, 1, 6, 3), n_devices=2, hidden_size=8, valid_len=6)


def test_padding_goes_up_to_a_multiple_of_128():
    assert [hp.pad_length(n) for n in (1, 127, 128, 129, 300)] == [128, 128, 128, 256, 384]


# ---- the output-embedding weight ----------------------------------------------------------------

def test_the_untied_lm_head_is_preferred_and_the_tied_embedding_is_the_fallback():
    wm = {"lm_head.weight": "s1", "model.language_model.embed_tokens.weight": "s2"}
    assert hp.output_embedding_name(wm, tie=False) == "lm_head.weight"
    assert hp.output_embedding_name(wm, tie=True) == "model.language_model.embed_tokens.weight"
    assert hp.output_embedding_name({"model.language_model.embed_tokens.weight": "s"}, tie=False) == \
        "model.language_model.embed_tokens.weight"
    assert hp.output_embedding_name({"model.embed_tokens.weight": "s"}, tie=True) == "model.embed_tokens.weight"


def test_no_output_embedding_in_the_index_is_an_error():
    with pytest.raises(KeyError):
        hp.output_embedding_name({"model.layers.0.mlp.weight": "s"}, tie=False)


def test_a_tensor_is_read_from_the_right_shard(tmp_path):
    w1, w2 = torch.randn(4, 3), torch.randn(5, 3)
    safetensors_torch.save_file({"a.weight": w1}, str(tmp_path / "model-1.safetensors"))
    safetensors_torch.save_file({"lm_head.weight": w2}, str(tmp_path / "model-2.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a.weight": "model-1.safetensors", "lm_head.weight": "model-2.safetensors"}}))
    assert torch.equal(hp.load_tensor(tmp_path, "lm_head.weight"), w2)
    assert torch.equal(hp.load_tensor(tmp_path, "a.weight"), w1)


def test_a_tensor_missing_from_the_index_is_a_key_error(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(KeyError, match="weight index"):
        hp.load_tensor(tmp_path, "lm_head.weight")


# ---- the cached CPU reference --------------------------------------------------------------------

def test_a_saved_reference_is_reused_only_for_the_same_records_and_layer_limit(tmp_path):
    ref = {"hidden": [torch.randn(5, 4)], "logits": [[torch.randn(2)]], "noise_logits": [[torch.randn(2)]],
           "embedding_name": "lm_head.weight"}
    path = tmp_path / "cpu-reference.pt"
    hp.save_reference(path, "digest-1", ref)
    got = hp.load_reference(path, "digest-1")
    assert torch.equal(got["hidden"][0], ref["hidden"][0]) and got["embedding_name"] == "lm_head.weight"
    assert hp.load_reference(path, "digest-2") is None
    assert hp.load_reference(tmp_path / "missing.pt", "digest-1") is None


def test_a_corrupt_reference_file_is_ignored_not_trusted(tmp_path):
    path = tmp_path / "cpu-reference.pt"
    path.write_bytes(b"not a torch file")
    assert hp.load_reference(path, "digest-1") is None


# ---- the cache guard ---------------------------------------------------------------------------

def test_an_empty_or_new_cache_is_marked_for_the_model(tmp_path):
    hp.guard_cache(tmp_path / "new", "Cloudflare/clef@abc")
    assert (tmp_path / "new" / ".orchard-model").read_text() == "Cloudflare/clef@abc"
    (tmp_path / "empty").mkdir()
    hp.guard_cache(tmp_path / "empty", "x")
    assert (tmp_path / "empty" / ".orchard-model").read_text() == "x"


def test_a_cache_marked_for_this_model_is_accepted(tmp_path):
    hp.guard_cache(tmp_path / "c", "m")
    (tmp_path / "c" / "layer0.bin").write_text("w")
    hp.guard_cache(tmp_path / "c", "m")


@pytest.mark.parametrize("marker", [None, "other/model"])
def test_a_non_empty_cache_with_no_marker_or_another_models_exits_3(tmp_path, marker):
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / "layer0.bin").write_text("w")
    if marker:
        (cache / ".orchard-model").write_text(marker)
    with pytest.raises(SystemExit) as e:
        hp.guard_cache(cache, "Cloudflare/clef@abc")
    assert e.value.code == hp.EXIT_CACHE == 3


def test_the_exit_codes_are_the_documented_ones():
    assert (hp.EXIT_OK, hp.EXIT_CACHE, hp.EXIT_DEVICE, hp.EXIT_HASH, hp.EXIT_WIRING) == (0, 3, 4, 5, 6)


# ---- the configuration -------------------------------------------------------------------------

def test_the_config_reader_names_a_missing_key(tmp_path, capsys):
    p = tmp_path / "parity_config.json"
    p.write_text(json.dumps({"run_dir": "x"}))
    with pytest.raises(SystemExit) as e:
        hp.read_config(p)
    assert e.value.code == 2 and "model_snapshot" in capsys.readouterr().err


PLAN_CONFIG = {"run_dir": "r", "model_snapshot": "/hf/hub/models--Cloudflare--clef/snapshots/abc123",
               "base_snapshot": "b", "head_file": "joint_head.safetensors", "head_config": "joint_head_config.json",
               "code_file": "joint_schema_model.py", "code_sha256": "a" * 64, "records": "records.json",
               "tt_cache": "t", "mesh_shape": [1, 2], "n_layers": None}


def test_the_config_reader_accepts_exactly_the_keys_of_the_plan_and_adds_defaults(tmp_path):
    p = tmp_path / "parity_config.json"
    p.write_text(json.dumps(PLAN_CONFIG))
    cfg = hp.read_config(p)
    assert cfg["mesh_shape"] == [1, 2] and cfg["n_layers"] is None and cfg["wiring_cosine_min"] == hp.WIRING_COSINE_MIN
    assert cfg["cpu_threads"] == hp.CPU_THREADS and cfg["noise_threads"] == hp.NOISE_THREADS
    assert math.isfinite(cfg["wiring_cosine_min"])


def test_optional_keys_may_be_left_out(tmp_path):
    p = tmp_path / "parity_config.json"
    short = {k: v for k, v in PLAN_CONFIG.items() if k not in ("base_snapshot", "records", "mesh_shape", "n_layers")}
    p.write_text(json.dumps(short))
    assert hp.read_config(p)["mesh_shape"] == [1, 2]


@pytest.mark.parametrize("bad", [{"mesh_shape": [2]}, {"mesh_shape": "1x2"}, {"n_layers": 0}, {"n_layers": "8"},
                                 {"code_sha256": "abc"}, {"run_dir": 5}])
def test_a_wrong_value_is_named_and_exits_2(tmp_path, bad, capsys):
    p = tmp_path / "parity_config.json"
    p.write_text(json.dumps({**PLAN_CONFIG, **bad}))
    with pytest.raises(SystemExit) as e:
        hp.read_config(p)
    assert e.value.code == 2 and next(iter(bad)) in capsys.readouterr().err


def test_the_cache_marker_id_is_the_repo_at_the_revision_taken_from_the_snapshot_path():
    assert hp.model_label("/x/hub/models--Cloudflare--clef/snapshots/abc123") == "Cloudflare/clef@abc123"
    assert hp.model_label("/some/plain/dir") == "/some/plain/dir"


def test_an_explicit_model_id_in_the_config_wins(tmp_path):
    p = tmp_path / "parity_config.json"
    p.write_text(json.dumps({**PLAN_CONFIG, "model_id": "me/mine@1"}))
    assert hp.read_config(p)["model_id"] == "me/mine@1"
    p.write_text(json.dumps(PLAN_CONFIG))
    assert hp.read_config(p)["model_id"] == "Cloudflare/clef@abc123"
