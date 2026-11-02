# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The model kind (orchard/model_kind.py, issues 378 and 380): how stage 0 tells a reranker or an embedding model from a text model, what the
hazard scan leaves out for one, and which stage specs a run of each kind gets."""
import json
import os
import re
from pathlib import Path

import pytest

from orchard import blocked, hazards as hz, model_kind as mk, narrate, stage0_scan, status
from orchard import stages
from orchard.stages import (SCORING_STAGE_1, SCORING_STAGE_2, STAGES, WEIGHTS_ONLY_STAGE_2, FULL_PORT_STAGE_2, run_model_kind, scoring_routed, spec_for)

CFG = {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
       "head_dim": 32, "vocab_size": 151_936, "tie_word_embeddings": True}       # 151,936 over 2 dies is 75,968 per die, above the 65,536 limit


def snapshot(tmp_path, *, archs=("Qwen3ForCausalLM",), tag=None, card=True, **over):
    d = tmp_path / "snap"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**CFG, **({"architectures": list(archs)} if archs is not None else {}), **over}))
    if card:
        front = "---\nlicense: apache-2.0\n" + (f"pipeline_tag: {tag}\n" if tag else "") + "---\n\n# A model\n"
        (d / "README.md").write_text(front)
    return d


# ---- the kind --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tag,kind", [("text-ranking", "reranker"), ("feature-extraction", "embedding"), ("sentence-similarity", "embedding"),
                                      ("text-generation", "generative"), ("image-text-to-text", "generative"), (None, "generative")])
def test_the_model_cards_pipeline_tag_decides_the_kind(tmp_path, tag, kind):
    got = mk.classify(snapshot(tmp_path, tag=tag))
    assert got["kind"] == kind and got["pipeline_tag"] == tag and got["source"] == ("model card" if tag in mk.PIPELINE_TAGS else "default")


def test_qwen3_reranker_has_a_causal_architecture_so_only_the_card_or_the_operator_can_say_it_is_a_reranker(tmp_path):
    d = snapshot(tmp_path, archs=("Qwen3ForCausalLM",))
    assert mk.classify(d)["kind"] == "generative"
    assert mk.classify(d, "reranker")["kind"] == "reranker"


@pytest.mark.parametrize("arch,kind", [("Qwen3VLForEmbedding", "embedding"), ("Qwen3EmbeddingModel", "embedding"), ("Qwen3Model", "generative"),
                                       ("Qwen3ForCausalLM", "generative")])
def test_an_architecture_name_with_embedding_in_it_is_an_embedding_model_and_a_bare_backbone_alone_is_not_enough(tmp_path, arch, kind):
    got = mk.classify(snapshot(tmp_path, archs=(arch,), card=False))
    assert got["kind"] == kind and got["source"] == ("architecture" if kind == "embedding" else "default")


def test_the_operator_wins_over_the_card_and_the_record_says_so(tmp_path):
    got = mk.classify(snapshot(tmp_path, tag="text-ranking"), "generative")
    assert got["kind"] == "generative" and got["source"] == "operator" and got["pipeline_tag"] == "text-ranking"


def test_the_card_wins_over_the_architecture_name(tmp_path):
    assert mk.classify(snapshot(tmp_path, archs=("Qwen3VLForEmbedding",), tag="text-ranking"))["kind"] == "reranker"


@pytest.mark.parametrize("bad", ["", "rerank", "EMBEDDING", None.__class__, 5])
def test_an_override_that_is_not_a_kind_is_refused(tmp_path, bad):
    with pytest.raises(mk.ModelKindError):
        mk.classify(snapshot(tmp_path), bad)


def test_no_model_directory_no_config_and_no_card_give_the_default_and_never_raise(tmp_path):
    assert mk.classify(None)["kind"] == "generative" and mk.classify(None, "embedding")["kind"] == "embedding"
    assert mk.classify(tmp_path / "missing")["kind"] == "generative"
    (tmp_path / "config.json").write_text("{not json")
    assert mk.classify(tmp_path)["kind"] == "generative"
    assert mk.classify("a\0b")["kind"] == "generative"


@pytest.mark.parametrize("readme", [
    "no front matter\npipeline_tag: text-ranking\n",                    # the tag is outside the front matter
    "---\npipeline_tag: text-ranking\n",                                 # no closing line: still read, the tag is inside
    "---\n---\npipeline_tag: text-ranking\n",                            # the front matter ended first
    "---\npipeline_tag:\n---\n", "---\npipeline_tag: [a, b]\n---\n", "---\npipeline_tag: text ranking\n---\n",
])
def test_the_front_matter_is_read_strictly(tmp_path, readme):
    (tmp_path / "README.md").write_text(readme)
    got = mk.pipeline_tag(tmp_path)
    assert got == ("text-ranking" if readme == "---\npipeline_tag: text-ranking\n" else None)


def test_a_quoted_tag_and_other_keys_before_it_are_read(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: mit\ntags:\n- a\npipeline_tag: \"feature-extraction\"\nlanguage: en\n---\n")
    assert mk.pipeline_tag(tmp_path) == "feature-extraction"


def test_a_card_that_is_a_link_into_a_cache_blobs_directory_is_read_and_a_fifo_or_a_huge_file_is_not(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_text("---\npipeline_tag: text-ranking\n---\n")
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "README.md").symlink_to(blobs / "abc")
    assert mk.pipeline_tag(snap) == "text-ranking"
    fifo = tmp_path / "fifo"
    fifo.mkdir()
    os.mkfifo(fifo / "README.md")
    assert mk.pipeline_tag(fifo) is None
    big = tmp_path / "big"
    big.mkdir()
    (big / "README.md").write_text("---\n" + "x: y\n" * (mk.README_MAX_BYTES // 5 + 10) + "pipeline_tag: text-ranking\n---\n")
    assert mk.pipeline_tag(big) is None


def test_has_output_head_follows_the_architecture_names():
    assert mk.has_output_head({"architectures": ["Qwen3ForCausalLM"]}) is True
    assert mk.has_output_head({"architectures": ["Qwen3Model"]}) is False
    assert mk.has_output_head({"architectures": ["Qwen3Model", "Qwen3ForCausalLM"]}) is True
    for bad in ({}, {"architectures": []}, {"architectures": "x"}, {"architectures": [1]}, None, [1]):
        assert mk.has_output_head(bad) is None


def test_the_template_and_this_module_use_the_same_head_architecture_rule():
    text = (Path(stages.__file__).parent / "skills" / "delta-triage-templates" / "delta_triage.py").read_text()
    pattern = re.search(r'^HEAD_ARCH = re.compile\(r"(.*)"\)$', text, re.M).group(1)
    assert pattern == mk.HEAD_ARCH.pattern


# ---- the hazard scan (issue 380) -------------------------------------------------------------------------------------------

def names(scan):
    return {h.name for h in scan.hazards}


def test_a_text_model_with_a_large_vocabulary_still_hits_the_vocabulary_per_die_hazard():
    assert "vocab_per_die_above_65536" in names(hz.scan_model_hazards(CFG, None))


@pytest.mark.parametrize("kind", ["reranker", "embedding"])
def test_a_scoring_model_does_not_hit_the_hazards_of_a_full_vocabulary_projection(kind):
    scan = hz.scan_model_hazards(CFG, None, kind=kind)
    assert not names(scan) & set(hz.FULL_HEAD_HAZARDS)
    assert scan.facts["model_kind"] == kind
    note = [n for n in scan.notes if "vocabulary hazards" in n]
    assert len(note) == 1 and "vocab_per_die_above_65536" in note[0] and kind in note[0]
    assert ("two rows" in note[0]) == (kind == "reranker") and ("no output head" in note[0]) == (kind == "embedding")


def test_the_other_hazards_of_the_same_backbone_still_fire_for_a_scoring_model():
    cfg = {**CFG, "hidden_size": 96 + 16}         # 112 is not a whole number of 32-wide tiles
    assert "hidden_size_not_whole_tiles" in names(hz.scan_model_hazards(cfg, None, kind="embedding"))
    cfg = {**CFG, "num_experts": 300, "num_experts_per_tok": 4, "moe_intermediate_size": 128}
    assert "routed_experts_above_256" in names(hz.scan_model_hazards(cfg, None, kind="reranker"))


def test_a_vocabulary_that_is_not_whole_tiles_is_left_out_too_and_a_clean_vocabulary_has_no_note_for_a_text_model():
    odd = {**CFG, "vocab_size": 151_937}
    assert "vocab_not_whole_tiles" in names(hz.scan_model_hazards(odd, None))
    assert "vocab_not_whole_tiles" not in names(hz.scan_model_hazards(odd, None, kind="embedding"))
    assert not [n for n in hz.scan_model_hazards(CFG, None).notes if "vocabulary hazards" in n]


def test_a_bare_backbone_leaves_the_vocabulary_hazards_out_even_when_the_kind_is_generative():
    scan = hz.scan_model_hazards(CFG, None, has_output_head=False)
    assert not names(scan) & set(hz.FULL_HEAD_HAZARDS) and any("bare backbone" in n for n in scan.notes)
    assert "vocab_per_die_above_65536" in names(hz.scan_model_hazards(CFG, None, has_output_head=True))
    assert "vocab_per_die_above_65536" in names(hz.scan_model_hazards(CFG, None, has_output_head=None))


def test_the_nearest_bundles_head_hazards_are_not_compared_for_a_scoring_model():
    scan = hz.scan_model_hazards(CFG, CFG, kind="reranker")
    assert not names(scan) & set(hz.FULL_HEAD_HAZARDS)


@pytest.mark.parametrize("kind", ["", "rerank", None, 3, "Embedding"])
def test_an_unknown_kind_is_refused_by_the_scan(kind):
    with pytest.raises(hz.HazardInputError, match="kind must be"):
        hz.scan_model_hazards(CFG, None, kind=kind)


def test_stage_0s_record_carries_the_kind_and_the_scan_for_a_bare_backbone_skips_the_head_hazards(tmp_path):
    snap = snapshot(tmp_path, archs=("Qwen3Model",), tag="feature-extraction")
    rec = stage0_scan.build_record(tmp_path / "stage", model_dir=snap, kind="embedding")
    assert rec["scan"] == "ran" and rec["facts"]["model_kind"] == "embedding"
    assert not [h for h in rec["hazards"] if h["name"] in hz.FULL_HEAD_HAZARDS]
    plain = stage0_scan.build_record(tmp_path / "stage", model_dir=snapshot(tmp_path / "t", archs=("Qwen3ForCausalLM",)))
    assert [h["name"] for h in plain["hazards"] if h["name"] in hz.FULL_HEAD_HAZARDS] == ["vocab_per_die_above_65536"]


# ---- the stage specs -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["reranker", "embedding"])
def test_a_scoring_run_of_class_weights_only_gets_the_scoring_stage_1_and_2(kind):
    assert spec_for(1, "weights-only", None, "weights-only", kind=kind) is SCORING_STAGE_1
    assert spec_for(2, "weights-only", None, "weights-only", kind=kind) is SCORING_STAGE_2
    assert spec_for(3, "weights-only", None, "weights-only", kind=kind).number == 3


def test_a_generative_run_gets_the_stages_it_always_got():
    assert spec_for(1, "weights-only", None, "weights-only") is STAGES[1]
    assert spec_for(2, "weights-only", None, "weights-only") is WEIGHTS_ONLY_STAGE_2
    assert spec_for(2, "weights-only", None, "weights-only", kind="generative") is WEIGHTS_ONLY_STAGE_2
    assert spec_for(2, "full-port", None, "full-port", full_port=True) is FULL_PORT_STAGE_2


def test_a_scoring_run_of_class_full_port_takes_the_route_only_with_the_flag():
    assert spec_for(2, "full-port", None, "full-port", full_port=True, kind="reranker") is SCORING_STAGE_2
    assert spec_for(1, "full-port", None, "full-port", full_port=True, kind="reranker") is SCORING_STAGE_1
    assert spec_for(2, "full-port", None, "full-port", kind="reranker") is STAGES[2]
    assert spec_for(2, "full-port", None, "unknown", full_port=True, kind="reranker") is STAGES[2]


@pytest.mark.parametrize("path,cls", [("weights-only", "weights+sidecar"), ("weights-only", "vocab-trim"), ("stock-runtime", "stock-runtime"),
                                      ("full-port", "restricted-head"), ("full-port", "unknown"), (None, None), ("weights-only", None)])
def test_a_scoring_run_of_any_other_class_is_not_routed(path, cls):
    assert not scoring_routed("reranker", path, cls, True)
    assert not scoring_routed("embedding", path, cls, False)


def test_the_scoring_specs_name_their_skill_gate_file_and_draft_and_the_scoring_stage_2_has_one_board():
    assert (SCORING_STAGE_1.skill, SCORING_STAGE_1.gate_file, SCORING_STAGE_1.boards) == ("scoring-reference", "evidence/reference/scoring-pairs.json", 0)
    assert (SCORING_STAGE_2.skill, SCORING_STAGE_2.gate_file, SCORING_STAGE_2.draft, SCORING_STAGE_2.boards) == (
        "scoring-check", "evidence/scoring-check.json", "scoring_check", 1)
    assert SCORING_STAGE_1.number == 1 and SCORING_STAGE_2.number == 2


def entries_with(model_kind):
    end = {"result": "pass"}
    if model_kind is not None:
        end["model_kind"] = model_kind
    return [{"event": "stage_end", "stage": 0, "data": end}]


@pytest.mark.parametrize("rec,kind", [({"kind": "reranker", "source": "model card"}, "reranker"), ({"kind": "embedding"}, "embedding"),
                                      ({"kind": "generative"}, "generative"), ({"kind": "bogus"}, "generative"), ({"kind": ["x"]}, "generative"),
                                      ("reranker", "generative"), (None, "generative"), (5, "generative")])
def test_the_run_kind_is_read_from_stage_0s_passing_entry_and_defaults_to_generative(rec, kind):
    assert run_model_kind(entries_with(rec)) == kind


def test_no_stage_0_entry_and_a_failed_one_give_generative():
    assert run_model_kind([]) == "generative"
    assert run_model_kind([{"event": "stage_end", "stage": 0, "data": {"result": "fail", "model_kind": {"kind": "reranker"}}}]) == "generative"


# ---- the block code and its words ------------------------------------------------------------------------------------------

def test_the_scoring_pauses_are_classified_and_have_a_block_code_and_an_unblock_text():
    from orchard import supervisor
    for reason in (supervisor.SCORING_STAGES_UNBUILT, supervisor.SCORING_NOT_ROUTED):
        assert status.classify_pause({"reason": reason})[0] == "scoring"
        assert blocked.block_code({"reason": reason}) == "scoring-route"
    assert "scoring-route" in blocked.REASONS and narrate.UNBLOCK["scoring-route"] and "scoring-route" not in blocked.AUTO_RETRY
    for reason in (supervisor.FULL_PORT_STAGES_UNBUILT, supervisor.FULL_PORT, supervisor.RESTRICTED_HEAD, supervisor.AUDIO_UNBUILT):
        assert blocked.block_code({"reason": reason}) != "scoring-route"
