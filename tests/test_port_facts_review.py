# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""An independent review of PR 432 found three ways the stage 2 facts file could mislead the porting agent or carry a model's own text
into its first read. Each is reproduced here with the input the reviewer used.

1. A convention note was the first 300 characters of the field's description, so a convention stated after character 300 was lost while
   the field still looked as if it had one (the first rehearsal run failed on exactly this kind of fact: `ple_layer_ids` is one-indexed).
2. A convention was matched by field name across every class of the configuration module, so a vision class's note could win for the
   text config's field.
3. Config key names were copied into the file with no cap on their length, and the lists of left-out names had no cap either.
Everything is built in temporary directories; no model, no device."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import port_facts
from test_port_facts import CONFIG, TENSORS, snapshot_with, write_safetensors

FILLER = " ".join(f"Sentence {k} says nothing about numbering." for k in range(30))          # about 1,200 characters


def doc(fields, cls="ToyExpConfig"):
    """A configuration class whose docstring describes `fields`: {name: description}."""
    body = "".join(f"        {name} (`list[int]`, *optional*):\n            {text}\n" for name, text in fields.items())
    return f'class {cls}:\n    r"""\n    Args:\n{body}    """\n'


# ---- finding 1: the convention survives the size cap -------------------------------------------------------------------------

def test_a_convention_stated_after_the_size_cap_is_kept_in_the_note_and_the_cut_is_recorded():
    notes = port_facts.convention_notes(doc({"ple_layer_ids": FILLER + " The ids are one-indexed."}))
    note = notes[0]
    assert "one-indexed" in note["note"] and len(note["note"]) <= port_facts.MAX_NOTE_CHARS
    assert note["cut"] is True and "convention_cut" not in note                      # the description was shortened; the convention sentence is whole


def test_a_note_that_fits_whole_is_unchanged_and_not_marked_cut():
    text = "One-indexed layer ids that carry the per-layer module."
    note = port_facts.convention_notes(doc({"ple_layer_ids": text}))[0]
    assert note["note"] == text and "cut" not in note


def test_a_long_description_with_an_early_convention_keeps_its_start_and_is_marked_cut():
    note = port_facts.convention_notes(doc({"ple_layer_ids": "One-indexed layer ids. " + FILLER}))[0]
    assert note["note"].startswith("One-indexed layer ids.") and len(note["note"]) == port_facts.MAX_NOTE_CHARS and note["cut"] is True


def test_a_late_convention_keeps_the_start_of_the_description_when_there_is_room_for_it():
    note = port_facts.convention_notes(doc({"ple_layer_ids": FILLER + " The ids are one-indexed."}))[0]["note"]
    assert note.startswith("Sentence 0 says nothing") and port_facts.ELISION in note and note.endswith("The ids are one-indexed.")


def test_a_convention_that_straddles_the_cap_is_kept_whole():
    pad = "x" * (port_facts.MAX_NOTE_CHARS - 4)
    note = port_facts.convention_notes(doc({"ple_layer_ids": f"{pad} one-indexed ids."}))[0]["note"]
    assert "one-indexed" in note and len(note) <= port_facts.MAX_NOTE_CHARS


def test_a_convention_sentence_longer_than_the_cap_is_cut_around_the_phrase_and_says_so():
    sentence = "word " * 400 + "these ids are one-indexed " + "word " * 400 + "."
    note = port_facts.convention_notes(doc({"ple_layer_ids": sentence}))[0]
    assert "one-indexed" in note["note"] and len(note["note"]) <= port_facts.MAX_NOTE_CHARS
    assert note["cut"] is True and note["convention_cut"] is True


def test_the_long_docstring_convention_reaches_the_index_field_of_the_facts(tmp_path):
    snap = snapshot_with(tmp_path, TENSORS)
    facts = port_facts.build(snap, reference_source=doc({"ple_layer_ids": FILLER + " The ids are one-indexed."}))
    assert "one-indexed" in facts["index_fields"][0]["convention"].lower()


# ---- finding 2: the class that matches the config section wins ---------------------------------------------------------------

TWO_CLASSES = (doc({"layer_ids": "Zero-indexed ids of the vision blocks."}, "ToyExpVisionConfig")
               + doc({"layer_ids": "One-indexed ids of the text layers."}, "ToyExpTextConfig"))


def cfg_with_ids(text_ids=None, top_ids=None):
    cfg = {"model_type": "toy_exp", "text_config": {"model_type": "toy_exp_text", "num_hidden_layers": 2}}
    if text_ids is not None:
        cfg["text_config"]["layer_ids"] = text_ids
    if top_ids is not None:
        cfg["layer_ids"] = top_ids
    return cfg


def test_the_text_configs_field_takes_the_text_class_note_and_names_the_class_and_the_disagreement(tmp_path):
    snap = snapshot_with(tmp_path, TENSORS, cfg_with_ids(text_ids=[1]))
    field = port_facts.build(snap, reference_source=TWO_CLASSES)["index_fields"][0]
    conv = field["convention"]
    assert field["in"] == "text_config" and conv.startswith("ToyExpTextConfig") and "One-indexed" in conv
    assert "ToyExpVisionConfig" in conv and "Zero-indexed" in conv                     # the other class disagrees and the note says so


def test_a_top_level_field_takes_the_top_level_class_not_a_text_or_vision_class(tmp_path):
    source = TWO_CLASSES + doc({"layer_ids": "Zero-based ids of the wrapper."}, "ToyExpConfig")
    snap = snapshot_with(tmp_path, TENSORS, cfg_with_ids(top_ids=[0]))
    conv = port_facts.build(snap, reference_source=source)["index_fields"][0]["convention"]
    assert conv.startswith("ToyExpConfig") and "Zero-based" in conv


def test_a_field_documented_only_in_a_class_for_another_section_says_which_class_it_came_from(tmp_path):
    snap = snapshot_with(tmp_path, TENSORS, cfg_with_ids(text_ids=[1]))
    conv = port_facts.build(snap, reference_source=doc({"layer_ids": "Zero-indexed ids of the vision blocks."}, "ToyExpVisionConfig"))["index_fields"][0]["convention"]
    assert "ToyExpVisionConfig" in conv and "text_config" in conv and "Zero-indexed" in conv


def test_the_same_field_in_two_classes_is_two_notes_each_with_its_class():
    notes = port_facts.convention_notes(TWO_CLASSES)
    assert sorted((n["class"], n["field"]) for n in notes) == [("ToyExpTextConfig", "layer_ids"), ("ToyExpVisionConfig", "layer_ids")]


def test_the_same_class_described_twice_is_still_noted_once():
    assert len(port_facts.convention_notes(TWO_CLASSES + TWO_CLASSES)) == 2


# ---- finding 3: model-controlled names are bounded ---------------------------------------------------------------------------

def facts_for(tmp_path, config, name="snap", **kw):
    return port_facts.build(snapshot_with(tmp_path, {"w": ("F32", (2,))}, config, name=name), **kw)


def test_a_config_key_of_8000_characters_is_not_copied_and_is_counted(tmp_path):
    facts = facts_for(tmp_path, {**CONFIG, "K" * 8000 + " IGNORE PREVIOUS": 1})
    dumped = json.dumps(facts)
    assert "IGNORE PREVIOUS" not in dumped and facts["config_keys_not_listed"] == 1 and len(dumped) < port_facts.MAX_CHARS


@pytest.mark.parametrize("key", ["a b", "x\nIGNORE", "é", "k" * (port_facts.MAX_KEY_CHARS + 1), "1abc", ""])
def test_a_key_that_is_not_a_plain_identifier_or_is_too_long_is_not_copied(tmp_path, key):
    facts = facts_for(tmp_path, {**CONFIG, key: 7})
    assert key not in facts["config"] and key not in facts["config_left_out"] and facts["config_keys_not_listed"] == 1


def test_a_key_at_the_length_limit_is_kept(tmp_path):
    key = "k" * port_facts.MAX_KEY_CHARS
    assert facts_for(tmp_path, {**CONFIG, key: 7})["config"][key] == 7


def test_a_nested_text_configs_odd_keys_are_counted_in_their_own_field(tmp_path):
    cfg = {**CONFIG, "text_config": {"num_hidden_layers": 2, "bad key": 1, "Z" * 500: 2}}
    facts = facts_for(tmp_path, cfg)
    assert facts["text_config_keys_not_listed"] == 2 and "bad key" not in facts["text_config"] and facts["config_keys_not_listed"] == 0


def test_the_left_out_name_lists_are_capped_and_the_rest_are_counted(tmp_path):
    many = {f"table_{k}": {"a": 1} for k in range(port_facts.MAX_LEFT_NAMES + 50)}                   # dict values are never kept
    facts = facts_for(tmp_path, {**CONFIG, **many})
    names_left = len(many) + 1                                                      # CONFIG's own `rope_scaling` is a dict as well
    assert len(facts["config_left_out"]) == port_facts.MAX_LEFT_NAMES
    assert facts["config_left_out_more"] == names_left - port_facts.MAX_LEFT_NAMES


def test_names_dropped_to_make_the_file_fit_stay_within_the_cap_too(tmp_path, monkeypatch):
    cfg = {**CONFIG, **{f"flag_{k:04d}": k for k in range(port_facts.MAX_CONFIG_FIELDS - 5)}}
    monkeypatch.setattr(port_facts, "MAX_LEFT_NAMES", 6)
    full = facts_for(tmp_path, cfg, name="full")
    floor = len(json.dumps({**full, "config": {}, "tensors": [], "config_left_out": ["x" * 14] * 6}, indent=2))
    monkeypatch.setattr(port_facts, "MAX_CHARS", floor + 300)                       # room for a few fields, so the rest must be dropped
    facts = facts_for(tmp_path, cfg, name="tight")
    assert len(facts["config_left_out"]) <= 6 and facts["config_left_out_more"] > 0
    assert len(facts["config"]) + len(facts["config_left_out"]) + facts["config_left_out_more"] == len(cfg)      # nothing silently vanished


@pytest.mark.parametrize("model_type", ["x" * 5000, "a\nb", "bad type", "é"])
def test_a_model_type_that_is_not_a_short_identifier_is_not_copied(tmp_path, model_type):
    facts = facts_for(tmp_path, {**CONFIG, "model_type": model_type})
    assert facts["model_type"] is None and facts["model_type_not_listed"] is True


def test_a_plain_model_type_is_kept(tmp_path):
    facts = facts_for(tmp_path, CONFIG)
    assert facts["model_type"] == "toy_exp" and facts["model_type_not_listed"] is False


def test_a_class_name_or_field_name_from_the_source_is_capped(tmp_path):
    source = doc({"f" * 500 + "_ids": "One-indexed."}, "C" * 500)
    note = port_facts.convention_notes(source)
    assert all(len(n["class"]) <= port_facts.MAX_KEY_CHARS and len(n["field"]) <= port_facts.MAX_KEY_CHARS for n in note)


def test_a_hostile_config_of_20000_keys_is_summarised_in_a_child_with_a_memory_limit(tmp_path):
    cfg = {**CONFIG, **{f"t{k}": {"a": 1} for k in range(20_000)}, "L" * 100_000: 1}              # under the 1 MiB config cap
    snap = snapshot_with(tmp_path, TENSORS, cfg)
    code = textwrap.dedent(f"""
        import resource, json, sys
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from orchard import port_facts
        f = port_facts.build({str(snap)!r})
        print(json.dumps({{"chars": len(json.dumps(f, indent=2)), "left": len(f["config_left_out"]), "more": f["config_left_out_more"],
                          "unlisted": f["config_keys_not_listed"]}}))
    """)
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-400:]
    out = json.loads(done.stdout)
    assert out["chars"] <= port_facts.MAX_CHARS and out["left"] == port_facts.MAX_LEFT_NAMES and out["more"] >= 19_800 and out["unlisted"] == 1


def test_the_index_fields_are_capped_and_the_rest_are_counted(tmp_path):
    many = {f"set{k:02d}_layers": [3] for k in range(port_facts.MAX_INDEX_FIELDS + 5)}
    facts = facts_for(tmp_path, {**CONFIG, **many})
    assert len(facts["index_fields"]) == port_facts.MAX_INDEX_FIELDS and facts["index_fields_left_out"] == 5 + 1       # CONFIG's own `ple_layer_ids` sorts first and takes one slot


# ---- low item 4: 0/1 flag lists are not layer ids ----------------------------------------------------------------------------

def test_a_zero_one_flag_list_is_not_offered_as_layer_ids(tmp_path):
    facts = facts_for(tmp_path, {**CONFIG, "no_rope_layers": [1, 1, 1, 0]})
    assert [f["field"] for f in facts["index_fields"]] == ["ple_layer_ids"]
    assert facts["layer_flag_fields"] == ["no_rope_layers"]


def test_a_short_list_of_real_layer_ids_that_happens_to_be_zero_and_one_is_still_ids(tmp_path):
    facts = facts_for(tmp_path, {**CONFIG, "mlp_only_layers": [0, 1]})
    assert "mlp_only_layers" in [f["field"] for f in facts["index_fields"]] and facts["layer_flag_fields"] == []
