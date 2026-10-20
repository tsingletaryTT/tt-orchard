# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 finds the analogues between a model and its nearest supported model the way a person would: by the shape of the
names, then the shapes of the tensors, then the values.

The case that showed the gap was perplexity-ai/pplx-decider-v1.2 against Qwen3.8-27B (2026-10-19). The two models have the same
layers. The decider names its text tensors `language_model.layers.5...` and its vision tensors `visual.blocks.0...`; the base
says `model.language_model.layers.5...` and `model.visual.blocks.0...`. The template's single rewrite rule matched none of them,
the draft said "share no tensor name", and the model was filed as a full port. The decider also has no `lm_head`: its
`readout.safetensors` holds 255 rows that are copies of the base head's rows for the answer-code tokens (A, B, ... AA, AB ...).
These tests build small models with that layout and real values.
"""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"


def template():
    spec = importlib.util.spec_from_file_location("delta_triage_alignment_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bf16(arr) -> bytes:
    return (np.asarray(arr, dtype="<f4").view("<u4") >> 16).astype("<u2").tobytes()


def write_st(path: Path, tensors: dict) -> None:
    """tensors: {name: float array}, written as BF16 safetensors with real data."""
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        arr = np.asarray(arr, dtype="<f4")
        data = bf16(arr)
        header[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def info(shape, dtype="BF16", shard="m.safetensors"):
    return {"dtype": dtype, "shape": list(shape), "shard": shard}


def layers(prefix: str, n=3):
    out = {}
    for i in range(n):
        out[f"{prefix}layers.{i}.mlp.gate_proj.weight"] = info((8, 4))
        out[f"{prefix}layers.{i}.input_layernorm.weight"] = info((4,))
    return out


def base_tensors(n=3):
    t = layers("model.language_model.", n)
    t["model.language_model.embed_tokens.weight"] = info((16, 4))
    t["model.language_model.norm.weight"] = info((4,))
    t["model.visual.blocks.0.attn.qkv.bias"] = info((12,))
    t["model.visual.pos_embed.weight"] = info((6, 3))
    t["lm_head.weight"] = info((16, 4))
    for k in ("mtp.fc.weight", "mtp.norm.weight"):
        t[k] = info((4, 4))
    return t


def decider_tensors(n=3):
    t = layers("language_model.", n)
    t["language_model.embed_tokens.weight"] = info((16, 4))
    t["language_model.norm.weight"] = info((4,))
    t["visual.blocks.0.attn.qkv.bias"] = info((12,))
    t["visual.pos_embed.weight"] = info((6, 3))
    return t


def compare(mt, bt):
    return template().compare_tensor_maps(mt, bt, mshards=["m"], bshards=["b"], merr=[], berr=[], model_path="m", base_path="b")


# ---- names ----------------------------------------------------------------------------------------------------------------------

def test_names_that_differ_only_by_a_prefix_are_matched_and_the_rewrites_are_reported():
    t = compare(decider_tensors(), base_tensors())
    ns = t["normalized_name_sets"]
    assert ns["common"] == len(decider_tensors())                         # every decider tensor has its analogue
    assert ns["only_in_model"] == []
    assert sorted(ns["only_in_base"]) == sorted(["lm_head.weight", "mtp.fc.weight", "mtp.norm.weight"])
    assert ns["aligned"]["pairs"] == len(decider_tensors())
    rules = {(r["model_prefix"], r["base_prefix"]): r for r in ns["aligned"]["rules"]}
    assert ("language_model", "model") in rules and rules[("language_model", "model")]["tensors"] == 3 * 2 + 2
    assert ("", "model") in rules                                          # the vision tensors gain `model.`
    assert all(r["same_shape"] == r["tensors"] for r in rules.values())


def test_a_layer_is_never_matched_to_another_layer():
    t = compare(decider_tensors(5), base_tensors(5))
    pairs = {m: b for m, b in t["normalized_name_sets"]["aligned"]["examples"]}
    for m, b in t["normalized_name_sets"]["aligned"]["all_pairs"].items():
        if ".layers." in m:
            assert m.split(".layers.")[1].split(".")[0] == b.split(".layers.")[1].split(".")[0], (m, b)
    assert pairs is not None


def test_an_aligned_pair_with_a_different_shape_is_still_a_shape_difference():
    mt = decider_tensors()
    mt["language_model.layers.1.mlp.gate_proj.weight"] = info((9, 4))
    t = compare(mt, base_tensors())
    assert [x["name"] for x in t["shape_diffs_normalized"]] == ["model.layers.1.mlp.gate_proj.weight"]


def test_a_name_with_two_equally_good_analogues_is_left_unmatched():
    """Both base names share three trailing parts with the model name: a tie, so neither is chosen."""
    bt = {"model.a.mlp.proj.weight": info((2, 2)), "model.b.mlp.proj.weight": info((2, 2))}
    mt = {"c.mlp.proj.weight": info((2, 2))}
    ns = compare(mt, bt)["normalized_name_sets"]
    assert ns["common"] == 0 and ns["only_in_model"] == ["c.mlp.proj.weight"]


def test_two_names_that_want_the_same_analogue_are_both_left_unmatched():
    bt = {"model.mlp.proj.weight": info((2, 2))}
    mt = {"language_model.mlp.proj.weight": info((2, 2)), "text.mlp.proj.weight": info((2, 2))}
    ns = compare(mt, bt)["normalized_name_sets"]
    assert ns["common"] == 0 and len(ns["only_in_model"]) == 2


def test_two_trailing_parts_are_not_enough_unless_one_name_is_the_end_of_the_other():
    """Same last two parts, but neither name contains the other and the shared run is short: not an analogue."""
    ns = compare({"x.proj.weight": info((2,))}, {"model.y.proj.weight": info((2,))})["normalized_name_sets"]
    assert ns["common"] == 0
    # the containment case is accepted: `visual.pos_embed.weight` is wholly the end of `model.visual.pos_embed.weight`
    ok = compare({"visual.pos_embed.weight": info((2,))}, {"model.visual.pos_embed.weight": info((2,))})["normalized_name_sets"]
    assert ok["common"] == 1


def test_a_prefix_rewrite_seen_once_does_not_reach_short_names():
    """One pair is not a rule: `a.layers.0...` pairing does not license pairing the unrelated short name `a.norm.weight`."""
    mt = {"a.layers.0.mlp.gate.weight": info((2, 2)), "a.norm.weight": info((2,))}
    bt = {"model.layers.0.mlp.gate.weight": info((2, 2)), "model.norm.weight": info((2,))}
    ns = compare(mt, bt)["normalized_name_sets"]
    assert ns["common"] == 1 and ns["only_in_model"] == ["a.norm.weight"]


def test_a_layout_the_old_rule_already_handled_needs_no_alignment():
    t = compare(layers("model.", 2), layers("model.language_model.", 2))
    assert t["normalized_name_sets"]["common"] == len(layers("model.", 2))
    assert t["normalized_name_sets"]["aligned"]["pairs"] == 0


def test_the_raw_rules_say_how_to_rename_the_models_keys_into_the_nearest_models_names():
    """Rules a loader can use: names that start with `from.` are called `to.` in the nearest model."""
    t = compare(decider_tensors(), base_tensors())
    rules = t["normalized_name_sets"]["aligned"]["raw_rules"]
    assert rules == [{"from": "language_model", "to": "model.language_model", "tensors": 3 * 2 + 2},
                     {"from": "visual", "to": "model.visual", "tensors": 2}]


def test_names_that_already_match_make_no_raw_rule_and_a_dropped_prefix_gives_an_empty_target():
    same = compare(layers("model.", 2), layers("model.", 2))["normalized_name_sets"]["aligned"]["raw_rules"]
    assert same == []
    dropped = compare({"text.layers.0.mlp.gate_proj.weight": info((8, 4)), "text.layers.1.mlp.gate_proj.weight": info((8, 4)),
                       "text.layers.2.mlp.gate_proj.weight": info((8, 4))},
                      {f"layers.{i}.mlp.gate_proj.weight": info((8, 4)) for i in range(3)})
    assert dropped["normalized_name_sets"]["aligned"]["raw_rules"] == [{"from": "text", "to": "", "tensors": 3}]


# ---- values ---------------------------------------------------------------------------------------------------------------------

def two_models(tmp_path, new_vals, base_vals):
    new, base = tmp_path / "new", tmp_path / "base"
    new.mkdir(), base.mkdir()
    write_st(new / "model.safetensors", new_vals)
    write_st(base / "model.safetensors", base_vals)
    return new, base


def test_sampled_values_say_identical_near_changed_or_unrelated(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 16)).astype("float32")
    new, base = two_models(tmp_path,
                           {"x": a, "y": a + 1e-3 * rng.normal(size=a.shape), "z": a + 0.7 * rng.normal(size=a.shape), "w": rng.normal(size=a.shape)},
                           {"m.x": a, "m.y": a, "m.z": a, "m.w": a})
    t = template()
    pairs = {"x": "m.x", "y": "m.y", "z": "m.z", "w": "m.w"}
    out = t.sample_value_diffs(new, base, pairs)
    cls = {r["model_tensor"]: r["relation"] for r in out["samples"]}
    assert cls == {"x": "identical", "y": "near", "z": "changed", "w": "unrelated"}
    assert out["summary"]["identical"] == 1 and out["summary"]["unrelated"] == 1


def test_value_sampling_reads_only_a_prefix_of_each_tensor(tmp_path):
    big = np.ones((4096, 64), dtype="float32")
    new, base = two_models(tmp_path, {"x": big}, {"m.x": big})
    out = template().sample_value_diffs(new, base, {"x": "m.x"})
    assert out["samples"][0]["elements"] <= template().SAMPLE_ELEMENTS


def test_a_dtype_that_cannot_be_decoded_is_skipped_with_a_note(tmp_path):
    new, base = tmp_path / "new", tmp_path / "base"
    new.mkdir(), base.mkdir()
    for d in (new, base):
        raw = json.dumps({"x": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}}).encode()
        (d / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x01\x02\x03\x04")
    out = template().sample_value_diffs(new, base, {"x": "x"})
    assert out["samples"] == [] and "I8" in " ".join(out["skipped"])


# ---- a restricted head ----------------------------------------------------------------------------------------------------------

def head_case(tmp_path, *, rows_from_head=True, vocab=64, dim=16, k=6, noise=0.0):
    rng = np.random.default_rng(1)
    head = rng.normal(size=(vocab, dim)).astype("float32")
    new, base = tmp_path / "new", tmp_path / "base"
    new.mkdir(), base.mkdir()
    write_st(base / "model.safetensors", {"lm_head.weight": head, "model.norm.weight": np.ones(dim, dtype="float32")})
    ids = [3, 7, 11, 40, 41, 62][:k]
    rows = head[ids] + noise * rng.normal(size=(k, dim)) if rows_from_head else rng.normal(size=(k, dim))
    write_st(new / "model.safetensors", {"model.norm.weight": np.ones(dim, dtype="float32")})
    write_st(new / "readout.safetensors", {"weight": rows})
    return new, base, ids


def sidecars_of(new):
    return {"sidecars": [{"file": "readout.safetensors", "num_tensors": 1}], "problems": []}


def test_a_sidecar_whose_rows_are_copies_of_the_base_head_rows_is_a_restricted_head(tmp_path):
    new, base, ids = head_case(tmp_path)
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is True and out["rows"] == 6 and out["base_rows"] == 64
    assert out["token_ids"] == ids and out["matched_rows"] == 6
    assert out["sidecar"] == "readout.safetensors" and out["min_cosine"] > 0.999


def test_rows_that_are_slightly_trained_still_count(tmp_path):
    new, base, ids = head_case(tmp_path, noise=1e-3)
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is True and out["token_ids"] == ids


def test_a_sidecar_of_unrelated_rows_is_not_a_restricted_head(tmp_path):
    new, base, _ = head_case(tmp_path, rows_from_head=False)
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is False and out["matched_rows"] < out["rows"]


def test_one_matching_row_among_unrelated_ones_is_not_a_restricted_head(tmp_path):
    new, base, ids = head_case(tmp_path, rows_from_head=False)
    head = np.random.default_rng(1).normal(size=(64, 16)).astype("float32")          # head_case's own head
    rows = np.random.default_rng(5).normal(size=(3, 16)).astype("float32")
    rows[1] = head[11]
    write_st(new / "readout.safetensors", {"weight": rows})
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["matched_rows"] == 1 and out["rows"] == 3 and out["restricted"] is False


def test_a_sidecar_as_tall_as_the_head_is_a_copy_not_a_restriction(tmp_path):
    new, base, ids = head_case(tmp_path)
    head = np.random.default_rng(1).normal(size=(64, 16)).astype("float32")
    write_st(new / "readout.safetensors", {"weight": head})                          # all 64 rows
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is False and "fewer rows" in out["note"]


def test_a_truncated_sidecar_gives_unknown_instead_of_crashing(tmp_path):
    """The header promises more bytes than the file holds (a download cut short). The reshape used to raise ValueError and end stage 0."""
    new, base, ids = head_case(tmp_path)
    side = new / "readout.safetensors"
    side.write_bytes(side.read_bytes()[:-20])
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is None and "readout.safetensors" in out["note"] and "could not be read" in out["note"]


def test_a_truncated_base_head_gives_unknown_instead_of_crashing(tmp_path):
    new, base, ids = head_case(tmp_path)
    shard = base / "model.safetensors"
    shard.write_bytes(shard.read_bytes()[:-500])
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is None and "could not be read" in out["note"]


def test_a_sidecar_header_with_no_offsets_gives_unknown_instead_of_crashing(tmp_path):
    new, base, ids = head_case(tmp_path)
    raw = json.dumps({"weight": {"dtype": "BF16", "shape": [6, 16]}}).encode()
    (new / "readout.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is None and "could not be read" in out["note"]


def test_a_model_with_its_own_lm_head_is_not_a_restricted_head_even_if_a_sidecar_copies_head_rows(tmp_path):
    """Restricted means the sidecar STANDS IN for the lm_head. A model that kept a full lm_head and ships a file of copied rows has not cut anything."""
    new, base, ids = head_case(tmp_path)
    rng = np.random.default_rng(1)
    head = rng.normal(size=(64, 16)).astype("float32")
    write_st(new / "model.safetensors", {"model.norm.weight": np.ones(16, dtype="float32"), "lm_head.weight": head})
    out = template().detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is False and "its own lm_head" in out["note"]


def test_without_numpy_the_head_check_says_it_was_skipped(tmp_path, monkeypatch):
    new, base, _ = head_case(tmp_path)
    t = template()
    monkeypatch.setattr(t, "np", None)
    out = t.detect_restricted_head(new, base, sidecars_of(new))
    assert out["restricted"] is None and "numpy" in out["note"]


def test_no_sidecar_or_no_base_head_means_no_restricted_head(tmp_path):
    new, base, _ = head_case(tmp_path)
    assert template().detect_restricted_head(new, base, {"sidecars": [], "problems": []})["restricted"] is False
    write_st(base / "model.safetensors", {"model.norm.weight": np.ones(16, dtype="float32")})      # a base without lm_head
    assert template().detect_restricted_head(new, base, sidecars_of(new))["restricted"] is False


def test_a_definite_backbone_difference_stands_whatever_a_sidecars_problems_are():
    """summerMC/Diffusion-Qwen3.8-27B adds 48 text tensors the nearest model lacks, and ships dllm_extras.pt. The sidecar cannot be read, but the backbone
    already needs new model code: that is full-port, and `unknown` is for what could not be decided."""
    t = template()
    tens = compare({**decider_tensors(), "language_model.layers.0.linear_attn.rev_gate": info((4,))}, base_tensors())
    cfg = {"decisive_keys_differing": [], "text_key_values": {}, "errors": []}
    sc = {"sidecars": [{"file": "dllm_extras.pt"}], "problems": ["sidecar dllm_extras.pt is not a safetensors file, so its tensors cannot be read without loading it"]}
    reasons = t.decide_path(cfg, tens) + sc["problems"]
    assert any("only in the new model" in r for r in reasons)
    assert t.decide_class(cfg, tens, reasons, sc) == "full-port"


def test_a_sidecar_problem_alone_still_makes_the_class_unknown_and_unreadable_backbone_headers_too():
    t = template()
    clean = compare(layers("model.", 2), layers("model.", 2))
    cfg = {"decisive_keys_differing": [], "text_key_values": {}, "errors": []}
    sc = {"sidecars": [{"file": "x.pt"}], "problems": ["sidecar x.pt is not a safetensors file"]}
    assert t.decide_class(cfg, clean, t.decide_path(cfg, clean) + sc["problems"], sc) == "unknown"
    broken = {**clean, "errors": ["model-00001.safetensors: header is cut short"]}
    assert t.decide_class(cfg, broken, ["something"], {"sidecars": [], "problems": []}) == "unknown"


def test_the_class_is_restricted_head_and_the_missing_lm_head_is_explained(tmp_path):
    t = template()
    tens = compare(decider_tensors(), base_tensors())
    tens["head"] = {"restricted": True, "rows": 6, "base_rows": 64, "sidecar": "readout.safetensors"}
    cfg = {"decisive_keys_differing": [], "text_key_values": {}, "errors": []}
    reasons = t.decide_path(cfg, tens)
    assert not any("lm_head" in r for r in reasons) and reasons == []
    assert t.decide_class(cfg, tens, reasons, {"sidecars": [{"file": "readout.safetensors"}], "problems": []}) == "restricted-head"
    tens["head"] = {"restricted": False}
    assert any("lm_head" in r for r in t.decide_path(cfg, tens))


# ---- code in subdirectories -----------------------------------------------------------------------------------------------------

def test_code_in_nested_directories_is_hashed_and_never_imported(tmp_path):
    (tmp_path / "source" / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "top.py").write_text("x = 1\n")
    (tmp_path / "source" / "src" / "pkg" / "model.py").write_text("raise SystemExit('must not run')\n")
    out = template().code_files(tmp_path)
    assert [c["file"] for c in out] == ["source/src/pkg/model.py", "top.py"]
    assert all(len(c["sha256"]) == 64 for c in out)


def test_the_code_listing_skips_unlikely_directories_and_stops_at_a_cap(tmp_path):
    t = template()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hook.py").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "m.py").write_text("x")
    deep = tmp_path
    for part in "abcdefgh":
        deep = deep / part
    deep.mkdir(parents=True)
    (deep / "too_deep.py").write_text("x")
    for i in range(t.MAX_CODE_FILES + 5):
        (tmp_path / f"f{i:03d}.py").write_text("x")
    names = [c["file"] for c in t.code_files(tmp_path)]
    assert len(names) == t.MAX_CODE_FILES
    assert not any(n.startswith((".git", "node_modules")) or "too_deep" in n for n in names)


def test_code_files_are_found_when_the_snapshot_links_each_file_to_a_blob(tmp_path):
    """A Hugging Face cache snapshot is a directory of symlinks into blobs/. The decider's code was missed because links were skipped."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_text("class Decider: ...\n")
    snap = tmp_path / "snap"
    (snap / "source" / "src" / "pkg").mkdir(parents=True)
    (snap / "source" / "src" / "pkg" / "model.py").symlink_to(blobs / "abc")
    (snap / "dangling.py").symlink_to(blobs / "missing")
    assert [c["file"] for c in template().code_files(snap)] == ["source/src/pkg/model.py"]


# ---- the whole script, on a decider-shaped model ----------------------------------------------------------------------------------

import shutil  # noqa: E402
import sys  # noqa: E402

pytest.importorskip("tokenizers")
from test_delta_triage_template import (BASE_REV, MODEL_REV, delta, make_snapshot, run_triage, stage_for,  # noqa: E402,F401
                                        text_tensors, tokenizer_json)
from orchard.stages import gate_delta  # noqa: E402


def poke(snap: Path, name: str, arr) -> None:
    """Write real values into one tensor of a zero-filled shard."""
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    shard = snap / index["weight_map"][name]
    raw = shard.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    lo, hi = json.loads(raw[8:8 + n])[name]["data_offsets"]
    data = bf16(arr)
    assert len(data) == hi - lo
    buf = bytearray(raw)
    buf[8 + n + lo:8 + n + hi] = data
    shard.write_bytes(bytes(buf))


@pytest.fixture
def decider_case(tmp_path, tokenizer_json):
    """The pplx-decider layout in miniature: text tensors named `language_model.*`, no lm_head, no mtp, and a readout sidecar whose rows
    are copies of three of the nearest model's lm_head rows."""
    n_vocab = len(json.loads(tokenizer_json)["model"]["vocab"]) + 3
    rng = np.random.default_rng(3)
    head = rng.normal(size=(n_vocab, 8)).astype("float32")
    base = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    poke(base, "lm_head.weight", head)
    change = {}
    for k, v in text_tensors(n_vocab).items():
        if k.startswith("model."):
            change[k] = None
            change["language_model." + k[len("model."):]] = v
        else:
            change[k] = None                                    # lm_head and mtp are gone
    new = make_snapshot(tmp_path / "new", MODEL_REV, nested=False, vision=False, tokenizer=tokenizer_json, tensor_change=change)
    ids = [5, 9, 20]

    def with_rows(rows):
        write_st(new / "readout.safetensors", {"weight": rows})
        return new
    return tmp_path, base, new, head, ids, with_rows


def run(tmp_path, new, base):
    stage = stage_for(tmp_path, new, base, model_id="perplexity-ai/pplx-decider-v1.2")
    r = run_triage(stage)
    assert r.returncode == 0, r.stderr[-800:]
    return stage, delta(stage)


def test_the_script_files_a_decider_shaped_model_as_a_restricted_head(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    stage, d = run(tmp_path, with_rows(head[ids]), base)
    assert d["class"] == "restricted-head" and d["path"] == "full-port"
    assert d["head"]["restricted"] is True and d["head"]["token_ids"] == ids and d["head"]["matched_rows"] == 3
    assert d["alignment"]["aligned_by_prefix"] > 0 and d["alignment"]["only_in_model"] == []
    assert len(d["path_reasons"]) == 1 and "lm_head.weight is replaced by 3 sidecar rows" in d["path_reasons"][0]
    assert d["values"]["summary"] is not None
    assert any(x["area"] == "other" and "lm_head" in x["finding"] for x in d["differences"])
    assert gate_delta(stage, tmp_path / "run").ok


def test_a_sidecar_that_does_not_come_from_the_head_leaves_the_missing_lm_head_unexplained(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    rng = np.random.default_rng(9)
    stage, d = run(tmp_path, with_rows(rng.normal(size=(3, 8)).astype("float32")), base)
    assert d["class"] == "full-port" and d["head"]["restricted"] is False
    assert any("lm_head" in r for r in d["path_reasons"])


def test_without_a_sidecar_a_missing_lm_head_is_a_full_port(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    stage, d = run(tmp_path, new, base)
    assert d["class"] == "full-port" and d["head"]["restricted"] is False


# ---- names paired by prefix are classed weights-only, but no swap or serve template applies the name map -------------------------------

def prefix_case(tmp_path, tokenizer_json, *, rename):
    """The nearest model's layout, with the text tensors renamed by prefix (or left alone). lm_head and mtp are kept, so only the names differ."""
    n_vocab = len(json.loads(tokenizer_json)["model"]["vocab"]) + 3
    base = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    change = {}
    if rename:
        for k, v in text_tensors(n_vocab).items():
            if k.startswith("model."):
                change[k] = None
                change["language_model." + k[len("model."):]] = v
    new = make_snapshot(tmp_path / "new", MODEL_REV, nested=False, vision=False, tokenizer=tokenizer_json, tensor_change=change)
    return base, new


def test_a_model_paired_with_the_nearest_by_prefix_gets_a_hazard_that_the_swap_does_not_apply_the_map(tmp_path, tokenizer_json):
    base, new = prefix_case(tmp_path, tokenizer_json, rename=True)
    stage, d = run(tmp_path, new, base)
    assert d["alignment"]["aligned_by_prefix"] > 0
    assert d["path"] == "weights-only", d["path_reasons"]
    [h] = [x for x in d["hazards"] if "paired by prefix" in x["finding"]]
    assert h["area"] == "other" and h["evidence"]
    assert "stage 1 CPU reference" in h["finding"] and "stages 2 and 4" in h["finding"]
    assert str(d["alignment"]["aligned_by_prefix"]) in h["finding"]
    assert gate_delta(stage, tmp_path / "run").ok


def test_a_model_whose_names_already_match_gets_no_name_map_hazard(tmp_path, tokenizer_json):
    base, new = prefix_case(tmp_path, tokenizer_json, rename=False)
    stage, d = run(tmp_path, new, base)
    assert d["alignment"]["aligned_by_prefix"] == 0
    assert not [x for x in d["hazards"] if "paired by prefix" in x["finding"]]


# ---- how a restricted head is read: facts from the model's own files and code ----------------------------------------------------

def test_a_json_file_that_states_an_attention_mode_is_reported(tmp_path):
    (tmp_path / "decision_config.json").write_text(json.dumps(
        {"attention_mode": "noncausal_full_attention", "pooling": "last", "temperature": 1.38, "codes": ["A", "B", "C"], "token_ids": [1, 2, 3]}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))              # no attention_mode: not reported
    big = {"attention_mode": "x", "pad": "y" * (300 * 1024)}
    (tmp_path / "huge.json").write_text(json.dumps(big))                                         # over the size cap: not read
    out = template().readout_facts(tmp_path, [])
    assert out["declared"] == [{"file": "decision_config.json", "attention_mode": "noncausal_full_attention", "pooling": "last",
                                "temperature": 1.38, "num_codes": 3}]


def test_code_hints_name_the_lines_that_say_how_attention_is_masked_and_where_the_head_is_read(tmp_path):
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "model.py").write_text(
        "import torch\n"
        "def enable_noncausal_full_attention(m):\n"
        "    m.register_forward_pre_hook(hook, with_kwargs=True)\n"
        "    kwargs['attention_mask'] = create_recurrent_attention_mask(config=c)\n"
        "readout = Linear(4, 255)\n")
    code = template().code_files(tmp_path)
    out = template().readout_facts(tmp_path, code)
    hints = {(h["file"], h["hint"]): h["lines"] for h in out["code_hints"]}
    assert hints[("source/model.py", "noncausal")] == [2]
    assert hints[("source/model.py", "forward hook")] == [3]
    assert hints[("source/model.py", "attention mask")] == [4]
    assert hints[("source/model.py", "readout")] == [5]


def test_code_that_is_too_large_or_unreadable_is_not_scanned(tmp_path):
    (tmp_path / "big.py").write_text("noncausal\n" + "x = 1\n" * 100_000)
    code = template().code_files(tmp_path)
    out = template().readout_facts(tmp_path, code)
    assert out["code_hints"] == [] and any("big.py" in s for s in out["skipped"])


def test_the_script_records_the_decision_configs_attention_mode_for_a_restricted_head(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    (new / "decision_config.json").write_text(json.dumps({"attention_mode": "noncausal_full_attention", "pooling": "last", "temperature": 1.4,
                                                          "codes": ["A", "B", "C"]}))
    stage, d = run(tmp_path, with_rows(head[ids]), base)
    assert d["class"] == "restricted-head"
    assert d["head"]["readout"]["declared"][0]["attention_mode"] == "noncausal_full_attention"
    hazard = next(h["finding"] for h in d["hazards"] if "restricted head" in h["finding"])
    assert "decision_config.json" in hazard and "noncausal_full_attention" in hazard
