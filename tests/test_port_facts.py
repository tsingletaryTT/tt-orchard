# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 of a full port starts the coder with a facts file read from the checkpoint (orchard/port_facts.py) and a port skeleton.

In three unattended rehearsal runs the coder spent its first 20 turns reading the stage 1 reference in 12,000-character pages and
probing for a `model.safetensors.index.json` that a single-file checkpoint does not have, wrote nothing, and was blocked. These tests
cover the facts the file holds for a single-file and a sharded checkpoint, the size bound that keeps it to one `read_file` page, the
reference's index conventions, and the refusals for hostile inputs. Everything is built in temporary directories; no model, no device."""
import ast
import json
import os
import struct
from pathlib import Path

import pytest

from orchard import port_facts, residency, safeio

SIZES = {"BF16": 2, "F32": 4}


def write_safetensors(path, tensors):
    """A valid safetensors file with zero data: `tensors` maps a name to (dtype, shape)."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = SIZES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


CONFIG = {"model_type": "toy_exp", "architectures": ["ToyExpForCausalLM"], "num_hidden_layers": 4, "hidden_size": 64,
          "vocab_size": 384, "layer_types": ["linear_attention"] * 3 + ["full_attention"], "ple_layer_ids": [2],
          "rope_scaling": {"type": "x"}, "tie_word_embeddings": False}

TENSORS = {"lm_head.weight": ("BF16", (384, 64)), "model.embed_tokens.weight": ("BF16", (384, 64)),
           **{f"model.layers.{i}.linear_attn.in_proj_qkv.weight": ("BF16", (320, 64)) for i in (0, 1, 2)},
           "model.layers.3.self_attn.q_proj.weight": ("BF16", (64, 64)),
           "model.layers.1.ple.conv1d.weight": ("BF16", (64, 1, 4))}

REFERENCE_SOURCE = '''
class ToyExpConfig:
    r"""
    Args:
        hidden_size (`int`, *optional*):
            Width of the residual stream.
        ple_layer_ids (`list[int]`, *optional*):
            One-indexed layer ids that carry the per-layer module. Must be in [1, num_hidden_layers].
        other_ids (`list[int]`, *optional*):
            A zero-based list that nothing here uses.
        rope_theta (`float`, *optional*):
            Base of the rotary embedding.
    """
'''


@pytest.fixture
def single(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps(CONFIG))
    write_safetensors(snap / "model.safetensors", TENSORS)
    return snap


def sharded(tmp_path):
    snap = tmp_path / "shards"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps(CONFIG))
    names = list(TENSORS)
    halves = {"model-00001-of-00002.safetensors": names[:3], "model-00002-of-00002.safetensors": names[3:]}
    for f, ns in halves.items():
        write_safetensors(snap / f, {n: TENSORS[n] for n in ns})
    weight_map = {n: f for f, ns in halves.items() for n in ns}
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return snap


def lines(facts):
    return facts["tensors"]


# ---- what the file says about the weights ------------------------------------------------------------------------------------

def test_a_single_file_checkpoint_is_described_and_the_missing_index_is_said(single):
    facts = port_facts.build(single)
    w = facts["weights"]
    assert w["layout"] == "single file" and w["index_file"] is False and w["files"] == ["model.safetensors"]
    assert "no model.safetensors.index.json" in w["note"] and "every tensor is in model.safetensors" in w["note"]
    assert w["tensor_count"] == len(TENSORS) and w["dtypes"] == {"BF16": len(TENSORS)}


def test_a_sharded_checkpoint_lists_its_shards_and_its_index(tmp_path):
    facts = port_facts.build(sharded(tmp_path))
    w = facts["weights"]
    assert w["layout"] == "sharded" and w["index_file"] is True and w["file_count"] == 2
    assert w["files"] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert "model.safetensors.index.json" in w["note"] and "weight_map" in w["note"]


def test_layer_numbers_collapse_and_each_group_says_which_layers_have_it(single):
    text = "\n".join(port_facts.build(single)["tensors"])
    assert "model.layers.{i}.linear_attn.in_proj_qkv.weight  BF16 [320, 64]  x3  layers 0-2" in text
    assert "model.layers.{i}.ple.conv1d.weight  BF16 [64, 1, 4]  x1  layers 1" in text      # layer 1, where the config's id 2 points
    assert "lm_head.weight  BF16 [384, 64]  x1" in text and "layers" not in [ln for ln in text.splitlines() if ln.startswith("lm_head")][0]


def test_the_layer_schedule_is_grouped_by_type(single):
    assert port_facts.build(single)["layer_schedule"] == {"linear_attention": "0-2", "full_attention": "3"}


def test_the_config_keeps_small_scalars_and_lists_and_names_what_it_left_out(single):
    cfg = port_facts.build(single)["config"]
    assert cfg["hidden_size"] == 64 and cfg["ple_layer_ids"] == [2] and cfg["tie_word_embeddings"] is False
    assert "rope_scaling" not in cfg and "rope_scaling" in port_facts.build(single)["config_left_out"]


def test_a_nested_text_config_is_summarised_too(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "outer", "text_config": {"num_hidden_layers": 2, "ple_layer_ids": [1]}}))
    write_safetensors(snap / "model.safetensors", {"model.layers.0.x": ("F32", (2,))})
    facts = port_facts.build(snap)
    assert facts["text_config"]["num_hidden_layers"] == 2 and facts["index_fields"][0]["field"] == "ple_layer_ids"


# ---- the reference's index conventions ---------------------------------------------------------------------------------------

def test_an_id_list_field_carries_the_references_index_convention(single):
    facts = port_facts.build(single, reference_source=REFERENCE_SOURCE)
    f = facts["index_fields"][0]
    assert f["field"] == "ple_layer_ids" and f["values"] == [2]
    assert "one-indexed" in f["convention"].lower() and "Must be in [1, num_hidden_layers]" in f["convention"]
    assert [c["field"] for c in facts["conventions"]] == ["ple_layer_ids", "other_ids"]


def test_only_fields_that_hold_layer_or_position_numbers_are_listed_as_index_fields(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    cfg = {"eos_token_id": [1, 2], "ple_layer_ids": [2], "mlp_only_layers": [0, 3], "full_attention_layers": [3], "vocab_size": 384,
           "bos_token_id": 1, "layer_types": ["a"], "hidden_sizes": [64, 128],
           "skip_layers": ["x", "y"], "block_ids": [True, False]}          # names that fit, values that are not whole numbers
    (snap / "config.json").write_text(json.dumps(cfg))
    write_safetensors(snap / "model.safetensors", {"w": ("F32", (2,))})
    fields = sorted(f["field"] for f in port_facts.build(snap)["index_fields"])
    assert fields == ["full_attention_layers", "mlp_only_layers", "ple_layer_ids"]


@pytest.mark.parametrize("value", [None, [], {}, {"a": 1}, 5, 10**20, "", "a\x00b", [None], [[1]], ["x" * 500], True])
def test_a_config_field_of_any_type_never_raises_and_a_bad_id_list_is_not_trusted(single, value):
    cfg = {**CONFIG, "layer_types": value, "ple_layer_ids": value, "model_type": value, "text_config": value, "hidden_size": value}
    (single / "config.json").write_text(json.dumps(cfg))
    facts = port_facts.build(single)
    assert facts["schema"] == port_facts.SCHEMA
    listed = [f["values"] for f in facts["index_fields"]]
    assert all(isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool) for x in v) for v in listed)


def test_without_the_references_source_the_convention_is_said_to_be_unknown(single):
    f = port_facts.build(single, reference_source=None)["index_fields"][0]
    assert "not stated" in f["convention"] and "layer numbers of the tensors" in f["convention"]


def test_conventions_come_from_the_docstring_of_the_field_they_describe():
    notes = port_facts.convention_notes(REFERENCE_SOURCE)
    assert [n["field"] for n in notes] == ["ple_layer_ids", "other_ids"]
    assert all(n["field"] != "hidden_size" and n["field"] != "rope_theta" for n in notes)


def test_a_model_type_that_could_leave_the_transformers_directory_finds_no_source(tmp_path, monkeypatch):
    root = tmp_path / "transformers"
    (root / "models" / "toy_exp").mkdir(parents=True)
    (root / "models" / "toy_exp" / "configuration_toy_exp.py").write_text(REFERENCE_SOURCE)
    assert port_facts.reference_source(root, "toy_exp") == REFERENCE_SOURCE
    asked = []
    real = safeio.read_bytes
    monkeypatch.setattr(safeio, "read_bytes", lambda path, **kw: asked.append(str(path)) or real(path, **kw))
    for bad in ("../toy_exp", "toy_exp/..", "", "a" * 100, "Toy", None, 5):
        assert port_facts.reference_source(root, bad) is None
    assert asked == [], "a model type that is not lowercase letters, digits and underscores must not reach a file read"
    assert port_facts.reference_source(tmp_path / "missing", "toy_exp") is None


def test_the_text_part_of_a_family_is_found_in_the_familys_own_module(tmp_path):
    root = tmp_path / "transformers"
    (root / "models" / "toy_exp").mkdir(parents=True)
    (root / "models" / "toy_exp" / "configuration_toy_exp.py").write_text(REFERENCE_SOURCE)
    assert port_facts.reference_source(root, "toy_exp_text") == REFERENCE_SOURCE
    assert port_facts.reference_source(root, "toy_exp_vision") == REFERENCE_SOURCE
    assert port_facts.reference_source(root, "toy_exp_other") is None              # only the known sub-config suffixes are stripped


def test_a_field_described_twice_in_one_module_is_noted_once():
    doubled = REFERENCE_SOURCE + REFERENCE_SOURCE
    assert [n["field"] for n in port_facts.convention_notes(doubled)] == ["ple_layer_ids", "other_ids"]


def test_the_reference_directory_is_read_without_importing_transformers():
    seen = []

    def run(argv, timeout):
        seen.append(argv)
        return 0, "/venv/site/transformers\n", ""
    assert port_facts.reference_root(run, "/py") == Path("/venv/site/transformers")
    code = seen[0][-1]
    assert "find_spec" in code and "import transformers" not in code and "-I" in seen[0]
    assert port_facts.reference_root(lambda a, t: (1, "", "x"), "/py") is None and port_facts.reference_root(run, None) is None
    assert port_facts.reference_root(lambda a, t: (0, "relative/path\n", ""), "/py") is None


# ---- the size bound ----------------------------------------------------------------------------------------------------------

def test_a_big_checkpoint_still_fits_one_read_file_page_and_says_what_was_left_out(tmp_path):
    snap = tmp_path / "big"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps(CONFIG))
    write_safetensors(snap / "model.safetensors", {f"block{k}.layers.{i}.w{k}": ("BF16", (k + 2, 3)) for k in range(400) for i in range(2)})
    facts = port_facts.build(snap)
    size = len(json.dumps(facts, indent=2))
    assert size <= port_facts.MAX_CHARS < 12000
    assert facts["tensors_left_out"] > 0 and len(facts["tensors"]) + facts["tensors_left_out"] == facts["tensor_groups"]


def test_config_fields_are_dropped_when_the_tensors_alone_do_not_make_the_file_fit(single, monkeypatch):
    whole = port_facts.build(single)
    bare = len(json.dumps(dict(whole, tensors=[]), indent=2)) + 1          # the file with no tensor lines at all
    monkeypatch.setattr(port_facts, "MAX_CHARS", bare - 60)
    facts = port_facts.build(single)
    assert len(json.dumps(facts, indent=2)) + 1 <= bare - 60 and facts["tensors_left_out"] == facts["tensor_groups"]
    assert len(facts["config"]) < len(whole["config"])
    monkeypatch.setattr(port_facts, "MAX_CHARS", 50)
    with pytest.raises(port_facts.PortFactsError, match="one read_file page"):
        port_facts.build(single)


def test_the_facts_are_written_into_the_stage_directory_as_json(single, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    path = port_facts.write(stage, port_facts.build(single))
    assert path == stage / port_facts.FACTS_NAME and json.loads(path.read_text())["schema"] == port_facts.SCHEMA


# ---- hostile and broken inputs -----------------------------------------------------------------------------------------------

@pytest.mark.timeout(30)
def test_a_config_that_is_a_fifo_is_refused_not_waited_on(single):
    (single / "config.json").unlink()
    os.mkfifo(single / "config.json")
    with pytest.raises(port_facts.PortFactsError, match="config.json"):
        port_facts.build(single)


def test_a_header_longer_than_the_cap_is_refused_before_it_is_read(single, monkeypatch):
    monkeypatch.setattr(residency, "MAX_HEADER_BYTES", 16)
    with pytest.raises(port_facts.PortFactsError, match="header"):
        port_facts.build(single)


@pytest.mark.parametrize("bad", [b"[1, 2]", b"not json", b'{"model_type": NaN}'])
def test_a_config_that_is_not_an_object_of_json_is_refused(single, bad):
    (single / "config.json").write_bytes(bad)
    with pytest.raises(port_facts.PortFactsError):
        port_facts.build(single)


def test_a_directory_with_no_weights_or_no_snapshot_is_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    with pytest.raises(port_facts.PortFactsError, match="safetensors"):
        port_facts.build(empty)
    with pytest.raises(port_facts.PortFactsError):
        port_facts.build(tmp_path / "missing")


def test_a_config_over_its_cap_is_refused(single, monkeypatch):
    monkeypatch.setattr(port_facts, "CONFIG_MAX", 10)
    with pytest.raises(port_facts.PortFactsError, match="config.json"):
        port_facts.build(single)


# ---- the skeleton ------------------------------------------------------------------------------------------------------------

def skeleton():
    return port_facts.SKELETON.read_text(encoding="utf-8")


def test_the_skeleton_defines_the_four_functions_and_each_one_refuses_until_written():
    tree = ast.parse(skeleton())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"load", "prefill", "decode_step", "close"} <= set(fns)
    for name in ("load", "prefill", "decode_step", "close"):
        raises = [n for n in ast.walk(fns[name]) if isinstance(n, ast.Raise)]
        assert raises and "NotImplementedError" in ast.dump(raises[0]), name


def test_the_skeleton_neither_names_the_reference_files_nor_hides_ttnn():
    text = skeleton()
    assert "generated-ids" not in text and "prompt-ids" not in text and "top5-logits" not in text and "stages/1" not in text
    assert "import ttnn" in text


def test_the_skeleton_reads_single_and_sharded_checkpoints_the_same_way():
    assert "*.safetensors" in skeleton() and "index.json" not in skeleton()
