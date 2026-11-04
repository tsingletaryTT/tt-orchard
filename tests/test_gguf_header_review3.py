# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 441, the third Copilot review (the round limit): three more ways a repository file could stop stage 0 or label evidence wrongly.

1. A lone surrogate (`"\\ud800"` in a JSON string is legal JSON and gives a Python string that cannot be encoded as UTF-8) in config.json or
   in a repository chat template must not stop stage 0, in the evidence writer, in a hash of the template, or in a copied row.
2. A model with no vocabulary head (a bare backbone such as `Qwen3Model`) has no `output.weight` because it never had one. That is not weight
   tying, so the GGUF must not be labelled tied, and the agreement row must not become a false `differ`.
3. An empty `chat_template.jinja` still takes precedence over the template in tokenizer_config.json: transformers 5.19.0 reads the file whenever
   it exists ("If independent chat template file(s) exist, they take priority over template entries in the tokenizer config") and assigns its text
   even when that text is empty. The repository's effective template is then empty, and a GGUF that carries the inline template differs from it."""
from __future__ import annotations

import json

import pytest

from test_gguf_header import TEMPLATE_TEXT, entry_of, hybrid_config, hybrid_kvs, module


def strict_loads(text: str):
    def refuse(name):
        raise ValueError(name)
    return json.loads(text, parse_constant=refuse)


def written(m, tmp_path, evidence):
    """Write `evidence` the way stage 0 does and read it back with a strict reader."""
    m.EVIDENCE = tmp_path / "evidence"
    path = m.write_json("gguf-header.json", evidence)
    return strict_loads(path.read_bytes().decode("utf-8"))


# ---- finding 1: lone surrogates --------------------------------------------------------------------------------------------------


def test_brief_replaces_a_lone_surrogate_in_a_scalar_a_key_and_a_value():
    m = module()
    assert m._brief("a\ud800b") == "a�b"
    out = m._brief({"k\udc00": ["v\ud800", {"n\ud801": 1}]})
    json.dumps(out, ensure_ascii=False).encode("utf-8")                       # must not raise
    assert out == {"k�": ["v�", {"n�": 1}]}


def test_a_lone_surrogate_in_a_copied_vision_config_value_does_not_stop_the_evidence_from_being_written(tmp_path):
    from test_gguf_header import T_STR, T_U32, gguf, kv, write
    kvs = [kv("general.architecture", T_STR, "clip"), kv("general.type", T_STR, "mmproj"), kv("clip.vision.block_count", T_U32, 24)]
    m = module()
    e = m.gguf_file_evidence(write(tmp_path, gguf(kvs)), {"vision_config": {"depth": "\ud800"}}, None)
    back = written(m, tmp_path, e)                                           # the writer used to raise UnicodeEncodeError here
    row = next(r for r in back["agreement"] if r["item"] == "vision block count")
    assert row["verdict"] == "not comparable" and row["config"] == "\ufffd"


def test_a_lone_surrogate_in_the_repository_template_does_not_raise_and_the_template_differs(tmp_path):
    m, e = entry_of(tmp_path, config=hybrid_config(), template="x\ud800y")
    row = next(r for r in e["agreement"] if r["item"] == "chat template")
    assert row["verdict"] == "differ"
    written(m, tmp_path, e)


def test_the_evidence_writer_never_fails_on_a_lone_surrogate_and_the_file_reads_back(tmp_path):
    m = module()
    back = written(m, tmp_path, {"a": "x\ud800", "b\udc00": 1})
    assert set(back) == {"a", "b\udc00"} and isinstance(back["a"], str)


def test_the_evidence_writer_still_writes_ordinary_text_as_utf8(tmp_path):
    m = module()
    m.EVIDENCE = tmp_path / "evidence"
    path = m.write_json("x.json", {"name": "café"})
    assert "café" in path.read_text(encoding="utf-8")                      # not escaped: the fallback is only for text that cannot be encoded


# ---- finding 2: a model with no vocabulary head is not tied ----------------------------------------------------------------------


def headless_config(arch="Qwen3_5Model", tie=False):
    cfg = hybrid_config()
    cfg["architectures"] = [arch]
    cfg["tie_word_embeddings"] = tie
    cfg["text_config"]["tie_word_embeddings"] = tie
    return cfg


def tied_row(e):
    return next(r for r in e["agreement"] if r["item"] == "tied embeddings")


def test_a_bare_backbone_without_output_weight_is_not_reported_tied(tmp_path):
    _, e = entry_of(tmp_path, config=headless_config(), template=TEMPLATE_TEXT)
    assert e["tied_embeddings"] is None
    row = tied_row(e)
    assert row["verdict"] == "not comparable" and "vocabulary head" in row["note"]


def test_a_model_with_a_head_architecture_and_no_output_weight_is_still_tied(tmp_path):
    _, e = entry_of(tmp_path, config=headless_config("Qwen3_5ForCausalLM", tie=True), template=TEMPLATE_TEXT)
    assert e["tied_embeddings"] is True and tied_row(e)["verdict"] == "agree"


def test_one_head_architecture_among_several_makes_the_model_headed(tmp_path):
    cfg = headless_config()
    cfg["architectures"] = ["Qwen3_5Model", "Qwen3_5ForCausalLM"]
    cfg["tie_word_embeddings"] = cfg["text_config"]["tie_word_embeddings"] = True
    _, e = entry_of(tmp_path, config=cfg, template=TEMPLATE_TEXT)
    assert e["tied_embeddings"] is True


def test_a_config_that_lists_no_architecture_keeps_the_old_inference(tmp_path):
    cfg = headless_config()
    del cfg["architectures"]
    cfg["tie_word_embeddings"] = cfg["text_config"]["tie_word_embeddings"] = True
    _, e = entry_of(tmp_path, config=cfg, template=TEMPLATE_TEXT)
    assert e["tied_embeddings"] is True


def test_a_present_output_weight_is_untied_even_for_a_bare_backbone(tmp_path):
    from test_gguf_header import F32, hybrid_tensors
    tensors = hybrid_tensors() + [("output.weight", [64, 300], F32)]
    _, e = entry_of(tmp_path, config=headless_config(), template=TEMPLATE_TEXT, tensors=tensors)
    assert e["tied_embeddings"] is False


# ---- finding 3: an empty chat_template.jinja still takes precedence --------------------------------------------------------------


def test_an_empty_jinja_file_is_the_effective_template_and_wins_over_tokenizer_config(tmp_path):
    m = module()
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "from-config"}))
    (tmp_path / "chat_template.jinja").write_text("")
    assert m.repo_chat_template(tmp_path) == ""


def test_an_unreadable_jinja_file_is_unknown_and_never_falls_back_to_tokenizer_config(tmp_path):
    m = module()
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "from-config"}))
    (tmp_path / "chat_template.jinja").write_bytes(b"x" * (m.REPORT_MAX_BYTES + 1))
    assert m.repo_chat_template(tmp_path) is None


def test_a_gguf_matching_the_unused_inline_template_differs_from_an_empty_effective_template(tmp_path):
    _, e = entry_of(tmp_path, config=hybrid_config(), template="")
    row = next(r for r in e["agreement"] if r["item"] == "chat template")
    assert row["verdict"] == "differ"
    assert "empty" in row["note"] and "chat_template.jinja" in row["note"]


def test_no_template_row_is_made_when_neither_the_gguf_nor_the_repository_has_one(tmp_path):
    kvs = hybrid_kvs(**{"tokenizer.chat_template": (None, None)})
    _, e = entry_of(tmp_path, kvs=kvs, config=hybrid_config(), template="")
    assert not [r for r in e["agreement"] if r["item"] == "chat template"]


def test_delta_json_is_written_with_the_same_safe_writer_and_reads_back(tmp_path):
    m = module()
    m.HERE_DIR = tmp_path
    m.write_delta({"ignored_files": [{"file": "bad\udcffname.gguf"}], "name": "café"})
    text = (tmp_path / "delta.json").read_bytes().decode("utf-8")
    assert strict_loads(text)["ignored_files"][0]["file"] == "bad\udcffname.gguf"


def test_every_delta_json_write_goes_through_write_delta():
    from test_gguf_header import TEMPLATE
    src = TEMPLATE.read_text(encoding="utf-8")
    assert src.count("\n    write_delta(delta)\n") == 2 and 'HERE_DIR / "delta.json").write_text' not in src
