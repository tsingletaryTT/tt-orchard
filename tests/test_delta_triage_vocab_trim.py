# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 recognises a vocabulary-trimmed copy of the nearest model (issue 56).

ProprietaryLegal/Qwen3.8-27B-CJK-Trim removes 66,533 vocabulary entries from Qwen3.8-27B. Its embedding and lm_head have 181,760 rows
against 248,320, `vocab_size` says the same, and every other tensor is the base's. Before this class, stage 0 called it a full port
because of those three differences. The class needs the kept rows to be rows of the nearest model's matrices, checked by value: a
smaller embedding that holds other numbers is a different model. On 2026-10-24 the nearest model's bundle served the real model with
the model's own vocab_size and the drafter off (tests/test_prepare_swap_vocab.py)."""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
H = 4
BASE_VOCAB, MODEL_VOCAB = 16, 10


def template():
    spec = importlib.util.spec_from_file_location("delta_triage_vocab_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_st(path: Path, tensors: dict, dtype="BF16") -> None:
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        arr = np.asarray(arr, dtype="<f4")
        data = ((arr.view("<u4") >> 16).astype("<u2").tobytes() if dtype == "BF16"
                else (arr.astype("i1").tobytes() if dtype == "I8" else arr.tobytes()))
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


class World:
    """A base model and a model that keeps `keep` of its vocabulary rows, as two directories of real safetensors files."""

    def __init__(self, tmp, keep=None, prefix=("model.", "model."), tied=False, extra_model=None, extra_base=None, model_dtype="BF16"):
        rng = np.random.default_rng(7)
        self.embed = rng.normal(size=(BASE_VOCAB, H)).astype("float32")
        self.head = rng.normal(size=(BASE_VOCAB, H)).astype("float32")
        self.keep = list(range(MODEL_VOCAB)) if keep is None else keep
        self.model, self.base = tmp / "model", tmp / "base"
        self.model.mkdir()
        self.base.mkdir()
        mp, bp = prefix
        norm = rng.normal(size=(H,)).astype("float32")
        base_t = {f"{bp}embed_tokens.weight": self.embed, f"{bp}norm.weight": norm, **(extra_base or {})}
        model_t = {f"{mp}embed_tokens.weight": self.embed[self.keep], f"{mp}norm.weight": norm, **(extra_model or {})}
        if not tied:
            base_t["lm_head.weight"] = self.head
            model_t["lm_head.weight"] = self.head[self.keep]
        write_st(self.base / "model.safetensors", base_t)
        write_st(self.model / "model.safetensors", model_t, dtype="BF16" if model_dtype == "BF16" else model_dtype)
        self.model_t, self.base_t = model_t, base_t

    def maps(self):
        t = template()
        return (t.read_tensors(self.model)[1], t.read_tensors(self.base)[1])


def cfg(model_vocab=MODEL_VOCAB, base_vocab=BASE_VOCAB, other=()):
    diff = ["vocab_size", *other]
    return {"decisive_keys_differing": diff, "errors": [],
            "text_key_values": {"vocab_size": {"model": model_vocab, "base": base_vocab},
                                **{k: {"model": 1, "base": 2} for k in other}}}


def analyse(w, c=None, sc=None):
    t = template()
    mt, bt = w.maps()
    tens = t.compare_tensor_maps(mt, bt, mshards=["m"], bshards=["b"], merr=[], berr=[], model_path="m", base_path="b")
    sc = sc or {"sidecars": [], "problems": []}
    c = c or cfg()
    vt = t.detect_vocab_trim(c, tens, sc, w.model, w.base, t.raw_pair_map(mt, bt))
    return t, c, tens, sc, vt


def test_rows_that_are_rows_of_the_nearest_models_matrices_make_a_vocabulary_trim(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path))
    assert vt["trimmed"] is True and vt["kept_subset"] is True
    assert (vt["model_vocab"], vt["base_vocab"]) == (MODEL_VOCAB, BASE_VOCAB)
    assert {x["name"]: (x["rows"], x["base_rows"], x["rows_found_in_base"]) for x in vt["tensors"]} == {
        "model.embed_tokens.weight": (MODEL_VOCAB, BASE_VOCAB, MODEL_VOCAB), "lm_head.weight": (MODEL_VOCAB, BASE_VOCAB, MODEL_VOCAB)}


def test_the_kept_rows_need_not_be_the_first_rows_or_in_order(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path, keep=[15, 3, 4, 0, 9, 8, 7, 12, 1, 2]))
    assert vt["kept_subset"] is True


def test_one_row_that_is_not_in_the_nearest_model_means_it_is_not_a_trim(tmp_path):
    w = World(tmp_path)
    write_st(w.model / "model.safetensors", {**w.model_t, "model.embed_tokens.weight": np.vstack([w.embed[:9], np.full((1, H), 3.25)]),
                                            })
    t, c, tens, sc, vt = analyse(w)
    assert vt["trimmed"] is True and vt["kept_subset"] is False
    row = next(x for x in vt["tensors"] if x["name"] == "model.embed_tokens.weight")
    assert row["rows_found_in_base"] == 9


def test_the_lm_head_is_checked_as_well_as_the_embedding(tmp_path):
    w = World(tmp_path)
    write_st(w.model / "model.safetensors", {**w.model_t, "lm_head.weight": np.vstack([w.head[:9], np.full((1, H), 3.25)])})
    assert analyse(w)[4]["kept_subset"] is False


def test_a_tied_model_with_no_lm_head_is_judged_on_its_embedding(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path, tied=True))
    assert vt["kept_subset"] is True and [x["name"] for x in vt["tensors"]] == ["model.embed_tokens.weight"]


def test_names_paired_by_prefix_are_found(tmp_path):
    layers = {f"{{}}layers.{i}.{n}.weight": np.ones((8, H)) for i in range(3) for n in ("mlp.up_proj", "mlp.down_proj")}
    w = World(tmp_path, prefix=("language_model.", "model.language_model."),
              extra_model={k.format("language_model."): v for k, v in layers.items()},
              extra_base={k.format("model.language_model."): v for k, v in layers.items()})
    t, c, tens, sc, vt = analyse(w)
    assert tens["normalized_name_sets"]["aligned"]["pairs"] > 0           # the names really were paired by prefix
    assert vt["trimmed"] is True and vt["kept_subset"] is True
    assert {x["name"] for x in vt["tensors"]} == {"language_model.embed_tokens.weight", "lm_head.weight"}


def test_a_difference_in_another_config_key_means_it_is_not_a_trim(tmp_path):
    assert analyse(World(tmp_path), c=cfg(other=("hidden_size",)))[4]["trimmed"] is False


def test_a_difference_in_another_tensors_shape_means_it_is_not_a_trim(tmp_path):
    w = World(tmp_path, extra_model={"model.layers.0.mlp.up_proj.weight": np.ones((6, H))},
              extra_base={"model.layers.0.mlp.up_proj.weight": np.ones((8, H))})
    assert analyse(w)[4]["trimmed"] is False


def test_a_tensor_only_one_model_has_means_it_is_not_a_trim(tmp_path):
    w = World(tmp_path, extra_model={"model.layers.0.extra.weight": np.ones((2, H))})
    assert analyse(w)[4]["trimmed"] is False


def test_a_changed_row_width_means_it_is_not_a_trim(tmp_path):
    w = World(tmp_path)
    write_st(w.model / "model.safetensors", {**w.model_t, "model.embed_tokens.weight": np.ones((MODEL_VOCAB, H + 1)),
                                            "lm_head.weight": np.ones((MODEL_VOCAB, H + 1))})
    assert analyse(w)[4]["trimmed"] is False


def test_a_larger_vocabulary_is_not_a_trim(tmp_path):
    w = World(tmp_path)
    big = np.vstack([w.embed, w.embed[:2]])
    write_st(w.model / "model.safetensors", {**w.model_t, "model.embed_tokens.weight": big, "lm_head.weight": np.vstack([w.head, w.head[:2]])})
    assert analyse(w, c=cfg(model_vocab=BASE_VOCAB + 2))[4]["trimmed"] is False


def test_a_config_that_says_the_vocabulary_changed_when_no_tensor_did_is_not_a_trim(tmp_path):
    w = World(tmp_path, keep=list(range(BASE_VOCAB)))
    vt = analyse(w)[4]                      # the config says 10 against 16; every matrix still has 16 rows
    assert vt["trimmed"] is False and vt["kept_subset"] is None and vt["tensors"] == []


def test_a_vocab_size_that_is_not_smaller_is_not_a_trim_even_when_the_rows_are(tmp_path):
    for mv in (BASE_VOCAB, BASE_VOCAB + 1):
        d = tmp_path / str(mv)
        d.mkdir()
        assert analyse(World(d), c=cfg(model_vocab=mv))[4]["trimmed"] is False, mv


def test_a_vocab_size_that_is_a_bool_or_text_is_not_a_trim(tmp_path):
    for i, mv in enumerate((True, "10", 1.5, None)):
        d = tmp_path / str(i)
        d.mkdir()
        assert analyse(World(d), c=cfg(model_vocab=mv))[4]["trimmed"] is False, mv


def test_a_matrix_with_more_rows_than_the_nearest_model_is_not_a_trim(tmp_path):
    w = World(tmp_path)
    write_st(w.model / "model.safetensors", {**w.model_t, "model.embed_tokens.weight": np.vstack([w.embed, w.embed[:3]])})
    vt = analyse(w)[4]
    assert vt["trimmed"] is False


def test_tensors_that_cannot_be_matched_to_the_nearest_models_leave_the_answer_unknown(tmp_path):
    w = World(tmp_path)
    t = template()
    mt, bt = w.maps()
    tens = t.compare_tensor_maps(mt, bt, mshards=["m"], bshards=["b"], merr=[], berr=[], model_path="m", base_path="b")
    vt = t.detect_vocab_trim(cfg(), tens, {"sidecars": [], "problems": []}, w.model, w.base, {})
    assert vt["trimmed"] is True and vt["kept_subset"] is None and vt["tensors"] == [] and "found" in vt["note"]


def test_reading_in_small_chunks_gives_the_same_answer(tmp_path):
    w = World(tmp_path, keep=[15, 3, 4, 0, 9, 8, 7, 12, 1, 2])
    t = template()
    mt, bt = w.maps()
    tens = t.compare_tensor_maps(mt, bt, mshards=["m"], bshards=["b"], merr=[], berr=[], model_path="m", base_path="b")
    for chunk in (1, 8, 20, 10 ** 9):
        t.VOCAB_CHUNK_BYTES = chunk
        vt = t.detect_vocab_trim(cfg(), tens, {"sidecars": [], "problems": []}, w.model, w.base, t.raw_pair_map(mt, bt))
        assert vt["kept_subset"] is True and all(x["rows_found_in_base"] == MODEL_VOCAB for x in vt["tensors"]), chunk


def test_tensors_that_differ_in_shape_with_no_config_difference_are_not_a_trim(tmp_path):
    c = {"decisive_keys_differing": [], "errors": [], "text_key_values": {}}
    assert analyse(World(tmp_path), c=c)[4]["trimmed"] is False


def test_a_dtype_the_check_does_not_read_leaves_the_answer_unknown(tmp_path):
    w = World(tmp_path, model_dtype="I8")
    write_st(w.base / "model.safetensors", w.base_t, dtype="I8")
    t, c, tens, sc, vt = analyse(w)
    assert vt["trimmed"] is True and vt["kept_subset"] is None and "dtype" in vt["note"]
    assert t.decide_class(c, tens, t.decide_path(c, tens), sc, vt) == "full-port"


def test_the_explained_reasons_are_exactly_the_vocabulary_ones(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path))
    reasons = t.decide_path(c, tens)
    assert reasons and sorted(vt["explained_reasons"]) == sorted(reasons)


def test_a_trim_gets_its_own_class(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path))
    assert t.decide_class(c, tens, t.decide_path(c, tens) + sc["problems"], sc, vt) == "vocab-trim"


def test_without_the_check_or_with_a_failed_check_the_class_stays_full_port(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path))
    reasons = t.decide_path(c, tens)
    assert t.decide_class(c, tens, reasons, sc) == "full-port"
    assert t.decide_class(c, tens, reasons, sc, dict(vt, kept_subset=False)) == "full-port"
    assert t.decide_class(c, tens, reasons, sc, dict(vt, kept_subset=None)) == "full-port"
    assert t.decide_class(c, tens, reasons, sc, dict(vt, trimmed=False)) == "full-port"


def test_a_sidecar_problem_beside_a_trim_keeps_the_class_from_being_a_trim(tmp_path):
    sc = {"sidecars": [{"file": "x.pt"}], "problems": ["sidecar x.pt is not a safetensors file"]}
    t, c, tens, sc, vt = analyse(World(tmp_path), sc=sc)
    assert vt["trimmed"] is False
    assert t.decide_class(c, tens, t.decide_path(c, tens) + sc["problems"], sc, vt) == "full-port"


def test_a_reason_the_trim_does_not_explain_keeps_the_class_from_being_a_trim(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path))
    reasons = t.decide_path(c, tens) + ["tensor model.layers.0.x has dtype I8 in the new model and BF16 in the nearest model"]
    assert t.decide_class(c, tens, reasons, sc, vt) == "full-port"


def test_the_sizes_are_read_from_the_config_comparison_and_not_guessed(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path), c=cfg(model_vocab=11, base_vocab=17))
    assert (vt["model_vocab"], vt["base_vocab"]) == (11, 17)
