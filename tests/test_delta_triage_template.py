# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 0 template the delta-triage skill copies: delta_triage.py.

Everything here runs on fake model snapshots in tmp directories. A fake snapshot holds a config,
a generation config, a model card, a LICENSE, a chat template, a tiny byte-level BPE tokenizer
trained with the `tokenizers` package, and safetensors files whose headers are written by hand
(the data bytes are zeros, so no weights exist to load). The script is run as a subprocess, the
way the agent runs it, and its delta.json is checked with the real stage 0 gate and the
compare-delta check against the Hemmingway-1 reference answer.

Two layouts are built:
- "text": a text-only fine-tune with a flat config.json and `model.*` tensor names, against a
  vision-language nearest model with a nested `text_config`, `model.language_model.*` names and
  a vision tower (the Hemmingway-1 case);
- "vl": a vision-language fine-tune that keeps its vision tower (nested config,
  `model.language_model.*` names, extra visual tensors), against a text-only nearest model with a
  flat config (the reverse direction), and against a vision-language nearest model (the OpenThai
  case, where both keep the vision tower).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("tokenizers") is None:
    pytest.skip("SKIPPED: the `tokenizers` package is not importable, so the delta-triage template "
                "tests did not run. delta_triage.py uses it for the encode test. The template is "
                "untested on this interpreter.", allow_module_level=True)

from orchard.stages import compare_delta, gate_delta  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
REFERENCE = Path(__file__).parent / "fixtures" / "hemmingway_stage0_reference.md"
MODEL_REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf"
BASE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
AREAS = ("config", "architecture", "tensors", "tensor_names", "tokenizer", "chat_template", "files",
         "generation_config", "license")


def template_module():
    """The template, imported by path (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("delta_triage_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- a tiny tokenizer -----------------------------------------------------------------------------

# Qwen's pre-tokenizer pattern, with combining marks (\p{M}) inside the letter classes.
WITH_MARKS = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+"
              r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# The same pattern with \p{M} dropped, as Hemmingway-1's tokenizer has it.
WITHOUT_MARKS = WITH_MARKS.replace(r"[\p{L}\p{M}]+", r"\p{L}+").replace(r"[^\s\p{L}\p{M}\p{N}]", r"[^\s\p{L}\p{N}]")
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


@pytest.fixture(scope="module")
def tokenizer_json():
    """A byte-level BPE tokenizer trained on the template's own fixed strings (so its merges span
    Thai and Devanagari combining marks), as tokenizer.json text."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    texts = [t for _, t in template_module().FIXED_STRINGS]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(WITH_MARKS), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=900, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(texts * 20, trainer=trainer)
    return tok.to_str()


def drop_marks(text: str) -> str:
    """The tokenizer with \\p{M} dropped from its pre-tokenizer pattern."""
    data = json.loads(text)
    split = data["pre_tokenizer"]["pretokenizers"][0]
    assert split["pattern"]["Regex"] == WITH_MARKS
    split["pattern"]["Regex"] = WITHOUT_MARKS
    return json.dumps(data)


def merges_as_strings(text: str) -> str:
    """The same tokenizer with each merge written as one space-joined string (the older form)."""
    data = json.loads(text)
    data["model"]["merges"] = [" ".join(m) if isinstance(m, list) else m for m in data["model"]["merges"]]
    return json.dumps(data)


# ---- fake snapshots -------------------------------------------------------------------------------

H, KV, D, INTER = 8, 1, 4, 16
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F16": 2}


def text_tensors(vocab: int) -> dict:
    """The text model's tensors under the `model.` prefix, plus lm_head and two mtp tensors."""
    t = {"model.embed_tokens.weight": ("BF16", [vocab, H]), "model.norm.weight": ("BF16", [H]),
         "lm_head.weight": ("BF16", [vocab, H]), "mtp.fc.weight": ("BF16", [H, 2 * H]),
         "mtp.norm.weight": ("BF16", [H])}
    for i in range(2):
        p = f"model.layers.{i}."
        t.update({p + "input_layernorm.weight": ("BF16", [H]),
                  p + "self_attn.q_proj.weight": ("BF16", [2 * D, H]),
                  p + "self_attn.k_proj.weight": ("BF16", [KV * D, H]),
                  p + "self_attn.v_proj.weight": ("BF16", [KV * D, H]),
                  p + "self_attn.o_proj.weight": ("BF16", [H, 2 * D]),
                  p + "mlp.up_proj.weight": ("BF16", [INTER, H]),
                  p + "mlp.down_proj.weight": ("BF16", [H, INTER])})
    return t


VISION = {"model.visual.patch_embed.proj.weight": ("BF16", [H, 3]),
          "model.visual.blocks.0.attn.qkv.weight": ("BF16", [3 * H, H]),
          "model.visual.blocks.0.attn.qkv.bias": ("BF16", [3 * H]),
          "model.visual.merger.fc.weight": ("BF16", [H, H])}


def write_safetensors(path: Path, tensors: dict) -> None:
    """A safetensors file: an 8-byte little-endian header length, the JSON header, zero data."""
    header, offset = {}, 0
    for name, (dtype, shape) in sorted(tensors.items()):
        size = DTYPE_BYTES[dtype]
        for s in shape:
            size *= s
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def text_config(vocab: int, **change) -> dict:
    cfg = {"model_type": "qwen3_5_text", "num_hidden_layers": 2, "hidden_size": H,
           "num_attention_heads": 2, "num_key_value_heads": KV, "head_dim": D,
           "layer_types": ["linear_attention", "full_attention"], "vocab_size": vocab,
           "max_position_embeddings": 1024, "intermediate_size": INTER, "rms_norm_eps": 1e-6}
    cfg.update(change)
    return cfg


def make_snapshot(root: Path, rev: str, *, nested: bool, vision: bool, tokenizer: str,
                  license_id: str = "apache-2.0", config_change=None, tensor_change=None,
                  genconfig_change=None, template: str = "{% for m in messages %}{{ m.content }}{% endfor %}",
                  extra_files=None) -> Path:
    """An HF-cache-shaped snapshot directory (blobs are plain files here; the script follows links
    the same way either way)."""
    snap = root / "snapshots" / rev
    snap.mkdir(parents=True)
    vocab = json.loads(tokenizer)["model"]["vocab"]
    n_vocab = len(vocab) + len(SPECIALS)
    text = text_config(n_vocab, **(config_change or {}))
    if nested:
        cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
               "text_config": text, "vision_config": {"depth": 1, "hidden_size": H},
               "image_token_id": 5, "transformers_version": "5.8.0.dev0"}
    else:
        cfg = {**text, "architectures": ["Qwen3_5ForCausalLM"], "transformers_version": "5.17.0"}
    (snap / "config.json").write_text(json.dumps(cfg))
    gen = {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "eos_token_id": [2, 0]}
    gen.update(genconfig_change or {})
    (snap / "generation_config.json").write_text(json.dumps(gen))
    (snap / "tokenizer.json").write_text(tokenizer)
    (snap / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 1024}))
    (snap / "chat_template.jinja").write_text(template)
    (snap / "README.md").write_text(f"---\nlicense: {license_id}\nlibrary_name: transformers\n---\n\n"
                                    "# A model\nIt writes text.\n")
    (snap / "LICENSE").write_text({"apache-2.0": "\n   Apache License\n   Version 2.0, January 2004\n",
                                   "cc-by-nc-4.0": "Creative Commons Attribution-NonCommercial 4.0 "
                                                   "International (CC BY-NC 4.0)\n"}[license_id])
    tensors = text_tensors(n_vocab)
    if nested:
        tensors = {("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v
                   for k, v in tensors.items()}
    if vision:
        tensors.update(VISION)
    for name, value in (tensor_change or {}).items():
        if value is None:
            tensors.pop(name)
        else:
            tensors[name] = value
    names = sorted(tensors)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        write_safetensors(snap / shard, {k: tensors[k] for k in keys})
        weight_map.update({k: shard for k in keys})
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    for name, content in (extra_files or {}).items():
        (snap / name).write_text(content)
    return snap


def stage_for(tmp_path: Path, model: Path, nearest: Path, *, model_id="Altworld/Hemmingway-1",
              nearest_id="Qwen/Qwen3.8-27B") -> Path:
    stage = tmp_path / "run" / "stages" / "0"
    stage.mkdir(parents=True)
    shutil.copy(TEMPLATE, stage)
    (stage / "triage_config.json").write_text(json.dumps({
        "run_dir": str(tmp_path / "run"), "model_id": model_id, "model_snapshot": str(model),
        "nearest_model_id": nearest_id, "nearest_snapshot": str(nearest)}))
    return stage


def run_triage(stage: Path, *args):
    return subprocess.run([sys.executable, *args, str(stage / "delta_triage.py")], capture_output=True,
                          text=True, timeout=300)


def evidence(stage: Path, name: str) -> dict:
    return json.loads((stage / "evidence" / name).read_text())


def delta(stage: Path) -> dict:
    return json.loads((stage / "delta.json").read_text())


def area(d: dict, name: str) -> dict:
    [item] = [i for i in d["differences"] if i["area"] == name]
    return item


@pytest.fixture
def text_case(tmp_path, tokenizer_json):
    """Hemmingway-1's shape: a text-only fine-tune against a vision-language nearest model."""
    def build(model_tokenizer=None, **model_kw):
        model = make_snapshot(tmp_path / "new", MODEL_REV, nested=False, vision=False,
                              tokenizer=model_tokenizer or tokenizer_json, **model_kw)
        nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True,
                                tokenizer=tokenizer_json)
        return stage_for(tmp_path, model, nearest)
    return build


# ---- the weights-only case ------------------------------------------------------------------------

def test_a_text_only_fine_tune_of_a_vision_language_model_is_weights_only(text_case):
    stage = text_case()
    r = run_triage(stage)
    assert r.returncode == 0, r.stdout + r.stderr
    d = delta(stage)
    assert d["path"] == "weights-only", d.get("path_reasons")
    # A model id without a revision gets the snapshot's revision.
    assert d["model"] == f"Altworld/Hemmingway-1@{MODEL_REV}"
    assert d["nearest_model"] == f"Qwen/Qwen3.8-27B@{BASE_REV}"
    for name in AREAS:
        item = area(d, name)
        assert item["finding"].strip() and item["evidence"], name
        for rel in item["evidence"]:
            assert (stage.parent.parent / rel).is_file(), rel


def test_the_draft_passes_the_stage_0_gate_and_the_hemmingway_reference(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    gate = gate_delta(stage, stage.parent.parent)
    assert gate.ok, gate.reasons
    out = compare_delta(delta(stage), REFERENCE.read_text())
    assert out["ok"], out
    hazards = {h["area"] for h in delta(stage)["hazards"]}
    assert {"tensor_cache", "drafter", "disk"} <= hazards
    assert "license" not in hazards                      # both are apache-2.0 here


def test_the_draft_carries_the_sha256_of_the_script_that_wrote_it(text_case):
    import hashlib
    stage = text_case()
    assert run_triage(stage).returncode == 0
    assert delta(stage)["template_sha256"] == hashlib.sha256((stage / "delta_triage.py").read_bytes()).hexdigest()


def test_the_tensor_comparison_normalizes_the_language_model_prefix(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    t = evidence(stage, "tensor-compare.json")
    n_text = len(text_tensors(10))
    assert t["normalized_name_sets"]["common"] == n_text
    assert t["normalized_name_sets"]["only_in_model"] == []
    assert sorted(t["normalized_name_sets"]["only_in_base"]) == sorted(VISION)
    assert t["shape_diffs_normalized"] == [] and t["dtype_diffs_normalized"] == []
    # As written, only lm_head and the mtp tensors share a name.
    assert t["raw_name_sets"]["common"] == 3
    names = area(delta(stage), "tensor_names")["finding"]
    assert "model.language_model." in names
    tensors = area(delta(stage), "tensors")["finding"]
    assert str(n_text) in tensors and str(len(VISION)) in tensors


def test_the_config_comparison_reads_a_nested_text_config(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    c = evidence(stage, "config-compare.json")
    assert c["decisive_keys_equal"] is True
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
                "head_dim", "layer_types", "vocab_size"):
        assert c["text_key_values"][key]["equal"] is True, key
    assert c["model_architectures"] == ["Qwen3_5ForCausalLM"]
    assert c["base_architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert set(c["text_config_diffs"]) == {"architectures", "transformers_version"}
    assert c["text_config_diffs_bookkeeping_only"] is True
    arch = area(delta(stage), "architecture")["finding"]
    assert "Qwen3_5ForCausalLM" in arch and "Qwen3_5ForConditionalGeneration" in arch
    assert "vision" in arch


def test_files_genconfig_and_license_are_compared(text_case):
    stage = text_case(license_id="cc-by-nc-4.0", genconfig_change={"temperature": 0.7},
                      extra_files={"notes.txt": "only in the new model"})
    assert run_triage(stage).returncode == 0
    f = evidence(stage, "files-compare.json")
    assert f["files_only_in_model"] == ["notes.txt"]
    assert f["model"]["num_weight_shards"] == 2 and f["model"]["weight_shard_total_bytes"] > 0
    g = evidence(stage, "genconfig-license.json")
    assert g["generation_config_diffs"] == {"temperature": {"model": 0.7, "base": 1.0}}
    assert g["model_license"]["readme_license_field"] == "cc-by-nc-4.0"
    assert g["base_license"]["readme_license_field"] == "apache-2.0"
    assert g["license_changed"] is True
    d = delta(stage)
    assert "0.7" in area(d, "generation_config")["finding"]
    assert "cc-by-nc-4.0" in area(d, "license")["finding"]
    [lic] = [h for h in d["hazards"] if h["area"] == "license"]
    assert "cc-by-nc-4.0" in lic["finding"] and "apache-2.0" in lic["finding"]


def test_the_disk_hazard_states_the_measured_free_space(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    disk = evidence(stage, "disk-free.json")
    st = os.statvfs(stage.parent.parent)
    assert disk["free_bytes"] > 0 and abs(disk["free_bytes"] - st.f_bavail * st.f_frsize) < 2 ** 30
    [h] = [h for h in delta(stage)["hazards"] if h["area"] == "disk"]
    assert "GiB free" in h["finding"] and "os.statvfs" in h["finding"]


# ---- the tokenizer --------------------------------------------------------------------------------

def test_identical_tokenizers_encode_every_test_string_the_same(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    e = evidence(stage, "tokenizer-encode.json")
    assert e["ran"] is True and e["fixed"] >= 100 and e["random"] == 100
    assert e["differing"] == 0 and e["fraction_differing"] == 0.0
    tc = evidence(stage, "tokenizer-compare.json")
    assert tc["chat_template_equal"] is True and tc["pre_tokenizer_equal"] is True
    assert area(delta(stage), "chat_template")["finding"].startswith("Identical")


def test_dropping_combining_marks_from_the_pre_tokenizer_is_detected(text_case, tokenizer_json):
    stage = text_case(model_tokenizer=drop_marks(tokenizer_json))
    assert run_triage(stage).returncode == 0
    tc = evidence(stage, "tokenizer-compare.json")
    assert tc["pre_tokenizer_equal"] is False and tc["vocab_equal"] is True and tc["merges_equal"] is True
    assert any(r"\p{M}" in p for p in tc["pre_tokenizer_regex_base"])
    assert not any(r"\p{M}" in p for p in tc["pre_tokenizer_regex_model"])
    e = evidence(stage, "tokenizer-encode.json")
    assert e["differing"] > 0 and e["by_category"].get("thai", 0) > 0
    assert any(x["category"] == "thai" for x in e["examples"])
    finding = area(delta(stage), "tokenizer")["finding"]
    assert f"{e['differing']} of {e['total']}" in finding and "thai" in finding.lower()
    # A tokenizer difference alone does not change the path: the runtime loads the new tokenizer.
    assert delta(stage)["path"] == "weights-only"


def test_merges_written_as_strings_or_lists_compare_equal(text_case, tokenizer_json):
    stage = text_case(model_tokenizer=merges_as_strings(tokenizer_json))
    assert run_triage(stage).returncode == 0
    tc = evidence(stage, "tokenizer-compare.json")
    assert tc["merges_equal_raw"] is False and tc["merges_equal"] is True
    assert evidence(stage, "tokenizer-encode.json")["differing"] == 0


def test_a_different_chat_template_is_reported_by_sha256(text_case):
    stage = text_case(template="{{ messages[0].content }} changed")
    assert run_triage(stage).returncode == 0
    tc = evidence(stage, "tokenizer-compare.json")
    assert tc["chat_template_equal"] is False
    assert tc["model"]["chat_template_sha256"] != tc["base"]["chat_template_sha256"]
    assert tc["model"]["chat_template_sha256"][:12] in area(delta(stage), "chat_template")["finding"]


def test_the_fixed_strings_cover_the_scripts_and_the_random_strings_are_fixed():
    mod = template_module()
    cats = {c for c, _ in mod.FIXED_STRINGS}
    assert {"latin", "accented", "cjk", "devanagari", "thai", "korean"} <= cats
    assert len(mod.FIXED_STRINGS) >= 100
    assert mod.random_strings() == mod.random_strings() and len(mod.random_strings()) == 100


# ---- the full-port decisions ----------------------------------------------------------------------

def test_a_changed_tensor_shape_makes_the_path_full_port(text_case):
    stage = text_case(tensor_change={"model.layers.1.mlp.up_proj.weight": ("BF16", [2 * INTER, H])})
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["path"] == "full-port"
    assert any("model.layers.1.mlp.up_proj.weight" in r for r in d["path_reasons"])
    t = evidence(stage, "tensor-compare.json")
    assert [x["name"] for x in t["shape_diffs_normalized"]] == ["model.layers.1.mlp.up_proj.weight"]
    assert gate_delta(stage, stage.parent.parent).ok


def test_a_changed_dtype_makes_the_path_full_port(text_case):
    stage = text_case(tensor_change={"model.norm.weight": ("F32", [H])})
    assert run_triage(stage).returncode == 0
    assert delta(stage)["path"] == "full-port"
    assert [x["name"] for x in evidence(stage, "tensor-compare.json")["dtype_diffs_normalized"]] == \
        ["model.norm.weight"]


def test_a_changed_decisive_config_value_makes_the_path_full_port(text_case):
    stage = text_case(config_change={"num_key_value_heads": 2})
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["path"] == "full-port"
    assert any("num_key_value_heads" in r for r in d["path_reasons"])
    assert "num_key_value_heads" in area(d, "config")["finding"]


def test_an_unexplained_extra_text_tensor_makes_the_path_full_port(text_case):
    stage = text_case(tensor_change={"model.layers.0.self_attn.q_norm.weight": ("BF16", [D])})
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["path"] == "full-port"
    assert any("q_norm" in r for r in d["path_reasons"])


# ---- the vision-language fine-tune ----------------------------------------------------------------

@pytest.mark.parametrize("nearest_nested", [False, True], ids=["text-only-nearest", "vl-nearest"])
def test_a_vision_language_fine_tune_keeps_its_vision_tower(tmp_path, tokenizer_json, nearest_nested):
    model = make_snapshot(tmp_path / "new", MODEL_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=nearest_nested, vision=nearest_nested,
                            tokenizer=tokenizer_json)
    stage = stage_for(tmp_path, model, nearest, model_id=f"iapp/openthai@{MODEL_REV}")
    r = run_triage(stage)
    assert r.returncode == 0, r.stdout + r.stderr
    d = delta(stage)
    assert d["model"] == f"iapp/openthai@{MODEL_REV}"
    assert d["path"] == "weights-only", d.get("path_reasons")
    t = evidence(stage, "tensor-compare.json")
    assert t["normalized_name_sets"]["common"] == len(text_tensors(10)) + (len(VISION) if nearest_nested else 0)
    assert sorted(t["normalized_name_sets"]["only_in_model"]) == ([] if nearest_nested else sorted(VISION))
    assert t["normalized_name_sets"]["only_in_base"] == []
    arch = area(d, "architecture")["finding"]
    assert "vision" in arch
    c = evidence(stage, "config-compare.json")
    assert c["decisive_keys_equal"] is True
    assert gate_delta(stage, stage.parent.parent).ok


# ---- what the script may and may not touch ---------------------------------------------------------

def tree(root: Path) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_the_script_writes_only_inside_its_stage_directory_and_never_opens_a_socket(text_case, tmp_path):
    stage = text_case()
    before = {k: v for k, v in tree(tmp_path).items() if not k.startswith(str(stage) + os.sep)}
    # Any socket the script tries to open raises, so a network call fails the run.
    guard = ("import runpy, socket, sys\n"
             "def refuse(*a, **k):\n    raise RuntimeError('network use is not allowed')\n"
             "socket.socket = refuse\nsocket.create_connection = refuse\n"
             f"sys.argv = [{str(stage / 'delta_triage.py')!r}]\n"
             f"runpy.run_path({str(stage / 'delta_triage.py')!r}, run_name='__main__')\n")
    r = subprocess.run([sys.executable, "-c", guard], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    after = {k: v for k, v in tree(tmp_path).items() if not k.startswith(str(stage) + os.sep)}
    assert after == before
    assert (stage / "delta.json").is_file()


def test_a_missing_config_key_exits_2_and_writes_no_delta(text_case):
    stage = text_case()
    cfg = json.loads((stage / "triage_config.json").read_text())
    del cfg["nearest_snapshot"]
    (stage / "triage_config.json").write_text(json.dumps(cfg))
    r = run_triage(stage)
    assert r.returncode == 2 and "nearest_snapshot" in r.stderr
    assert not (stage / "delta.json").exists()


def test_the_script_prints_progress_and_names_the_path(text_case):
    stage = text_case()
    r = run_triage(stage)
    assert r.returncode == 0
    assert "delta.json" in r.stdout and "weights-only" in r.stdout


# ---- sidecars (a weights file the nearest model's runtime does not load) --------------------------

import hashlib  # noqa: E402

HEAD_TENSORS = {"evidence_layers.0.attention.in_proj_weight": ("BF16", [3 * H, H]),
                "evidence_layers.0.attention.in_proj_bias": ("BF16", [3 * H]),
                "hidden_norm.weight": ("BF16", [H])}


def add_sidecar(stage: Path, tensors=None, name="joint_head.safetensors") -> Path:
    model = Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])
    write_safetensors(model / name, tensors or HEAD_TENSORS)
    return model / name


def test_a_weight_file_outside_the_index_is_a_sidecar_and_the_class_says_so(text_case):
    stage = text_case()
    head = add_sidecar(stage)
    r = run_triage(stage)
    assert r.returncode == 0, r.stdout + r.stderr
    d = delta(stage)
    assert d["class"] == "weights+sidecar" and d["path"] == "weights-only", d.get("path_reasons")
    [side] = d["sidecars"]
    assert side["file"] == "joint_head.safetensors" and side["num_tensors"] == 3
    assert side["size"] == head.stat().st_size and side["dtypes"] == ["BF16"]
    assert side["sha256"] == hashlib.sha256(head.read_bytes()).hexdigest()
    assert side["tensor_names"] == sorted(HEAD_TENSORS)
    assert evidence(stage, "sidecar-compare.json")["sidecars"][0]["file"] == "joint_head.safetensors"


def test_the_sidecar_is_not_counted_as_extra_text_tensors_of_the_backbone(text_case):
    stage = text_case()
    add_sidecar(stage)
    assert run_triage(stage).returncode == 0
    t = evidence(stage, "tensor-compare.json")
    assert not any(n.startswith("evidence_layers") for n in t["normalized_name_sets"]["only_in_model"])
    assert t["normalized_name_sets"]["unexplained_only_in_model"] == []


def test_the_draft_with_a_sidecar_passes_the_stage_0_gate(text_case):
    stage = text_case()
    add_sidecar(stage)
    assert run_triage(stage).returncode == 0
    gate = gate_delta(stage, stage.parent.parent)
    assert gate.ok, gate.reasons


def test_a_sidecar_whose_tensor_names_overlap_the_backbone_is_unknown_and_full_port(text_case):
    stage = text_case()
    add_sidecar(stage, {"model.norm.weight": ("BF16", [H]), **HEAD_TENSORS})
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["class"] == "unknown" and d["path"] == "full-port"
    assert any("overlap" in r and "model.norm.weight" in r for r in d["path_reasons"])


def test_a_sidecar_that_is_not_safetensors_is_unknown_because_its_tensors_cannot_be_read(text_case):
    stage = text_case()
    model = Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])
    (model / "head.bin").write_bytes(b"pickle")
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["class"] == "unknown" and any("head.bin" in r for r in d["path_reasons"])


def test_a_plain_weights_only_model_has_no_sidecars(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["class"] == "weights-only" and d["sidecars"] == []


def test_a_changed_shape_is_full_port_even_when_a_sidecar_is_present(text_case):
    stage = text_case(config_change={"num_key_value_heads": 2})
    add_sidecar(stage)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["class"] == "full-port" and d["path"] == "full-port"


def test_a_code_file_in_the_repo_is_recorded_with_its_hash_and_raises_a_hazard(text_case):
    stage = text_case(extra_files={"joint_schema_model.py": "import torch\n"})
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["code_files"] == [{"file": "joint_schema_model.py",
                                "sha256": hashlib.sha256(b"import torch\n").hexdigest()}]
    assert any(h["area"] == "other" and "joint_schema_model.py" in h["finding"] for h in d["hazards"])
    assert d["class"] == "weights-only"                    # code alone does not change the class


def test_a_sidecar_shows_in_the_differences_and_the_hazards(text_case):
    stage = text_case()
    add_sidecar(stage)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert any(i["area"] == "other" and "joint_head.safetensors" in i["finding"] for i in d["differences"])
    assert any(h["area"] == "other" and "sidecar" in h["finding"] for h in d["hazards"])


def test_a_model_with_a_single_unindexed_weight_file_is_not_mistaken_for_a_sidecar(tmp_path, tokenizer_json):
    """No index file: model.safetensors is the backbone itself."""
    model = make_snapshot(tmp_path / "new", MODEL_REV, nested=False, vision=False, tokenizer=tokenizer_json)
    (model / "model.safetensors.index.json").unlink()
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    stage = stage_for(tmp_path, model, nearest)
    assert run_triage(stage).returncode == 0
    assert delta(stage)["sidecars"] == [] and delta(stage)["class"] == "weights-only"


# ---- gaps the mutation run found ---------------------------------------------------------------

def model_dir_of(stage: Path) -> Path:
    return Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])


def test_backbone_shards_with_unusual_names_are_found_through_the_index(text_case):
    stage = text_case()
    model = model_dir_of(stage)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    renames = {"model-00001-of-00002.safetensors": "weights-a.safetensors",
               "model-00002-of-00002.safetensors": "weights-b.safetensors"}
    for old, new in renames.items():
        (model / old).rename(model / new)
    index["weight_map"] = {k: renames[v] for k, v in index["weight_map"].items()}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    add_sidecar(stage)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["class"] == "weights+sidecar"
    assert evidence(stage, "tensor-compare.json")["model"]["shards"] == ["weights-a.safetensors", "weights-b.safetensors"]


def test_a_single_unindexed_model_file_and_a_sidecar_are_told_apart_by_name(tmp_path, tokenizer_json):
    model = make_snapshot(tmp_path / "new", MODEL_REV, nested=False, vision=False, tokenizer=tokenizer_json)
    vocab = len(json.loads(tokenizer_json)["model"]["vocab"]) + len(SPECIALS)
    for shard in model.glob("model-0000*-of-00002.safetensors"):
        shard.unlink()
    (model / "model.safetensors.index.json").unlink()
    write_safetensors(model / "model.safetensors", text_tensors(vocab))
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    stage = stage_for(tmp_path, model, nearest)
    add_sidecar(stage)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert [s["file"] for s in d["sidecars"]] == ["joint_head.safetensors"] and d["class"] == "weights+sidecar"


def test_a_file_the_nearest_model_also_has_is_not_a_sidecar_of_the_new_one(text_case, tmp_path):
    stage = text_case()
    add_sidecar(stage)
    nearest = Path(json.loads((stage / "triage_config.json").read_text())["nearest_snapshot"])
    write_safetensors(nearest / "joint_head.safetensors", HEAD_TENSORS)
    assert run_triage(stage).returncode == 0
    assert delta(stage)["sidecars"] == [] and delta(stage)["class"] == "weights-only"


def test_a_sidecar_overlap_is_found_after_the_language_model_prefix_is_normalized(text_case):
    stage = text_case()
    add_sidecar(stage, {"model.language_model.norm.weight": ("BF16", [H]), **HEAD_TENSORS})
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["class"] == "unknown" and any("overlap" in r for r in d["path_reasons"])


def test_a_corrupt_safetensors_sidecar_is_unknown(text_case):
    stage = text_case()
    (model_dir_of(stage) / "broken.safetensors").write_bytes(b"not a header at all")
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["class"] == "unknown" and any("broken.safetensors cannot be read" in r for r in d["path_reasons"])


def test_a_pytorch_bin_file_next_to_the_shards_is_not_a_sidecar(text_case):
    stage = text_case()
    (model_dir_of(stage) / "pytorch_model-00001-of-00002.bin").write_bytes(b"pickle")
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["sidecars"] == [] and d["class"] == "weights-only"


# ---- the MTP head -------------------------------------------------------------------------------
# The nearest bundle's speculative drafter needs mtp.* tensors. delta.json says how many each model has, so
# the swap checks, stage 7 and the bundle can act on it without reading weights again.

NO_MTP = {"mtp.fc.weight": None, "mtp.norm.weight": None}


def test_the_counts_of_mtp_tensors_are_recorded(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    assert delta(stage)["mtp"] == {"nearest_tensors": 2, "new_tensors": 2}


def test_a_model_without_mtp_tensors_is_still_weights_only_and_gets_a_drafter_hazard(text_case):
    stage = text_case(tensor_change=NO_MTP)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["path"] == "weights-only" and d["mtp"] == {"nearest_tensors": 2, "new_tensors": 0}
    drafter = [h for h in d["hazards"] if h["area"] == "drafter"]
    assert len(drafter) == 2 and any("no mtp" in h["finding"].lower() and "MTP head" in h["finding"] for h in drafter)


def test_a_model_that_keeps_its_mtp_tensors_gets_only_the_standing_drafter_hazard(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    assert len([h for h in delta(stage)["hazards"] if h["area"] == "drafter"]) == 1


def test_the_draft_with_an_mtp_count_still_passes_the_stage_0_gate(text_case):
    stage = text_case(tensor_change=NO_MTP)
    assert run_triage(stage).returncode == 0
    gate = gate_delta(stage, stage.parent.parent)
    assert gate.ok, gate.reasons


def test_the_mtp_tensors_are_counted_after_the_language_model_prefix_is_normalized(tmp_path, tokenizer_json):
    model = make_snapshot(tmp_path / "new", MODEL_REV, nested=True, vision=True, tokenizer=tokenizer_json,
                          tensor_change={"model.language_model.mtp.extra.weight": ("BF16", [H])})
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    stage = stage_for(tmp_path, model, nearest)
    assert run_triage(stage).returncode == 0
    assert delta(stage)["mtp"] == {"nearest_tensors": 2, "new_tensors": 3}


def test_a_tensor_that_only_contains_the_letters_mtp_is_not_counted(text_case):
    stage = text_case(tensor_change={"model.layers.0.attn_mtpx.weight": ("BF16", [H])})
    assert run_triage(stage).returncode == 0
    assert delta(stage)["mtp"] == {"nearest_tensors": 2, "new_tensors": 2}


def test_no_drafter_hazard_is_added_when_neither_model_has_mtp_tensors(tmp_path, tokenizer_json):
    model = make_snapshot(tmp_path / "new", MODEL_REV, nested=False, vision=False, tokenizer=tokenizer_json,
                          tensor_change=NO_MTP)
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json,
                            tensor_change={"mtp.fc.weight": None, "mtp.norm.weight": None})
    stage = stage_for(tmp_path, model, nearest)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["mtp"] == {"nearest_tensors": 0, "new_tensors": 0}
    assert len([h for h in d["hazards"] if h["area"] == "drafter"]) == 1


def test_a_cut_short_shard_is_reported_by_the_directory_wrapper(text_case):
    """compare_tensors reads the directories and hands the read errors to compare_tensor_maps; a shard
    that cannot be read in the nearest model must still show up as an error."""
    stage = text_case()
    nearest = Path(json.loads((stage / "triage_config.json").read_text())["nearest_snapshot"])
    shard = sorted(nearest.glob("model*.safetensors"))[0]
    shard.write_bytes(shard.read_bytes()[:5])
    mod = template_module()
    out = mod.compare_tensors(model_dir_of(stage), nearest)
    assert any(shard.name in e for e in out["errors"]), out


def test_the_directory_wrappers_report_their_directories_as_paths(text_case):
    stage = text_case()
    model, nearest = model_dir_of(stage), Path(json.loads((stage / "triage_config.json").read_text())["nearest_snapshot"])
    mod = template_module()
    t = mod.compare_tensors(model, nearest)
    assert t["model"]["path"] == str(model) and t["base"]["path"] == str(nearest)
    c = mod.compare_configs(model, nearest)
    assert c["model_config_path"] == str(model / "config.json")
    assert c["base_config_path"] == str(nearest / "config.json")


# ---- the research report as an optional input -------------------------------------------------------

def write_candidates(stage, rows):
    path = stage.parent.parent / "research" / "base-candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidates": rows}))
    cfg = json.loads((stage / "triage_config.json").read_text())
    cfg["base_candidates"] = str(path)
    (stage / "triage_config.json").write_text(json.dumps(cfg))
    return path


ROW = {"repo": "Qwen/Qwen3.8-27B", "confidence": "stated", "class_estimate": {"class": "weights-only"}}


def baseline_decision(stage):
    """(class, path) of a run with no base_candidates input, taken before the key is added to the config."""
    cfg_path = stage / "triage_config.json"
    saved = cfg_path.read_text()
    cfg = json.loads(saved)
    cfg.pop("base_candidates", None)
    cfg_path.write_text(json.dumps(cfg))
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    cfg_path.write_text(saved)
    return d["class"], d["path"]


def test_the_draft_records_how_the_research_estimate_compared_with_the_measurement(text_case):
    stage = text_case()
    write_candidates(stage, [ROW])
    assert run_triage(stage).returncode == 0
    br = delta(stage)["base_research"]
    assert br["candidate"] == {"repo": "Qwen/Qwen3.8-27B", "confidence": "stated"}
    assert (br["estimated_class"], br["measured_class"], br["agreed"]) == ("weights-only", "weights-only", True)


def test_a_disagreement_is_recorded_and_changes_nothing_else(text_case):
    stage = text_case()
    write_candidates(stage, [{**ROW, "class_estimate": {"class": "full-port"}}])
    run_triage(stage)
    d = delta(stage)
    assert d["base_research"]["agreed"] is False and d["path"] == "weights-only" and d["class"] == "weights-only"


@pytest.mark.parametrize("content", ["not json", "[]", '{"candidates": "x"}', '{"candidates": []}',
                                     '{"candidates": [7, null]}'])
def test_an_unreadable_or_empty_report_gives_a_null_candidate_and_no_failure(text_case, content):
    stage = text_case()
    write_candidates(stage, []).write_text(content)
    base = baseline_decision(stage)
    r = run_triage(stage)
    assert r.returncode == 0, r.stdout + r.stderr
    d = delta(stage)
    br = d["base_research"]
    assert br["candidate"] is None and br["agreed"] is None and br["estimated_class"] is None
    assert (d["class"], d["path"]) == base


@pytest.mark.parametrize("estimate", [None, "weights-only", ["weights-only"], {"class": None}, {}])
def test_a_missing_or_malformed_estimate_is_ignored(text_case, estimate):
    stage = text_case()
    write_candidates(stage, [{"repo": "Qwen/Qwen3.8-27B", "confidence": "stated", "class_estimate": estimate}])
    base = baseline_decision(stage)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    br = d["base_research"]
    assert br["candidate"]["repo"] == "Qwen/Qwen3.8-27B" and br["estimated_class"] is None and br["agreed"] is None
    assert (d["class"], d["path"]) == base


def test_without_the_key_the_draft_has_no_base_research(text_case):
    stage = text_case()
    run_triage(stage)
    assert "base_research" not in delta(stage)


def test_a_candidate_for_another_repo_is_not_matched(text_case):
    stage = text_case()
    write_candidates(stage, [{**ROW, "repo": "other/model"}])
    run_triage(stage)
    assert delta(stage)["base_research"]["candidate"] is None


def test_the_draft_with_base_research_still_passes_the_stage_0_gate(text_case):
    stage = text_case()
    write_candidates(stage, [ROW])
    run_triage(stage)
    assert "base_research" in delta(stage)
    assert gate_delta(stage, stage.parent.parent).ok


def test_report_text_is_cleaned_before_it_is_recorded(text_case):
    """The report is fetched text. Control characters and newlines are removed, the length is capped, and
    a value that is not a string is recorded as null."""
    stage = text_case()
    write_candidates(stage, [{**ROW, "confidence": "sta\nted\x1b[31m" + "x" * 400,
                              "class_estimate": {"class": ["weights-only"]}}])
    assert run_triage(stage).returncode == 0
    br = delta(stage)["base_research"]
    conf = br["candidate"]["confidence"]
    assert "\n" not in conf and "\x1b" not in conf and len(conf) <= 300
    assert br["estimated_class"] is None and br["agreed"] is None


# ---- a measured sidecar model against the estimate (headers cannot see a sidecar) -------------------

@pytest.mark.parametrize("estimate, agreed", [("weights-only", True), ("full-port", False), ("unknown", False)])
def test_a_measured_sidecar_model_agrees_with_a_weights_only_estimate_only(text_case, estimate, agreed):
    stage = text_case()
    add_sidecar(stage)
    write_candidates(stage, [{**ROW, "class_estimate": {"class": estimate}}])
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    br = d["base_research"]
    assert d["class"] == "weights+sidecar" and br["measured_class"] == "weights+sidecar"
    assert br["estimated_class"] == estimate and br["agreed"] is agreed
    assert set(br) == {"candidate", "estimated_class", "measured_class", "agreed"}


def test_a_measured_weights_only_model_still_agrees_with_a_weights_only_estimate(text_case):
    stage = text_case()
    write_candidates(stage, [ROW])
    run_triage(stage)
    assert delta(stage)["base_research"]["agreed"] is True


def test_a_report_over_the_size_cap_is_treated_as_unreadable(text_case):
    stage = text_case()
    base = baseline_decision(stage)
    path = write_candidates(stage, [ROW])
    path.write_text(json.dumps({"candidates": [ROW], "pad": "x" * (2 * 1024 * 1024 + 10)}))
    r = run_triage(stage)
    assert r.returncode == 0, r.stdout + r.stderr
    d = delta(stage)
    assert d["base_research"]["candidate"] is None and d["base_research"]["agreed"] is None
    assert (d["class"], d["path"]) == base


def test_a_measured_full_port_does_not_agree_with_a_weights_only_estimate(text_case):
    stage = text_case(config_change={"num_key_value_heads": 2})
    write_candidates(stage, [ROW])
    assert run_triage(stage).returncode == 0
    br = delta(stage)["base_research"]
    assert br["measured_class"] == "full-port" and br["agreed"] is False


def write_view(stage, rows, **top):
    """The file `bringup` hands stage 0 (research/base-candidates.stage0.json): a class string per candidate."""
    path = stage.parent.parent / "research" / "base-candidates.stage0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": "h/new", "selected": None, "candidates": rows, **top}))
    cfg = json.loads((stage / "triage_config.json").read_text())
    cfg["base_candidates"] = str(path)
    (stage / "triage_config.json").write_text(json.dumps(cfg))
    return path


VIEW_ROW = {"repo": "Qwen/Qwen3.8-27B", "confidence": "stated", "group": 0, "class_estimate": "weights-only",
            "bundles": ["tt/q"]}


def test_the_stage_0_view_shape_is_read_too(text_case):
    stage = text_case()
    write_view(stage, [VIEW_ROW])
    assert run_triage(stage).returncode == 0
    br = delta(stage)["base_research"]
    assert br["candidate"] == {"repo": "Qwen/Qwen3.8-27B", "confidence": "stated"}
    assert (br["estimated_class"], br["measured_class"], br["agreed"]) == ("weights-only", "weights-only", True)


def test_a_view_with_no_estimate_or_a_disagreeing_one_is_recorded_without_changing_the_decision(text_case):
    stage = text_case()
    write_view(stage, [{**VIEW_ROW, "class_estimate": "full-port"}])
    run_triage(stage)
    d = delta(stage)
    assert d["base_research"]["agreed"] is False and d["path"] == "weights-only"
    write_view(stage, [{**VIEW_ROW, "class_estimate": None}])
    run_triage(stage)
    assert delta(stage)["base_research"]["estimated_class"] is None


# ---- the generation config: stop tokens and sampling defaults the served directory will use -----------------
# BAAI/AREX-2 ships no generation_config.json. Staging copies the nearest model's, so the server stops on the
# nearest model's eos_token_id and applies its sampling defaults; with neither file the server stops after
# loading the weights. Stage 0 says so before any hardware is used.

def drop_model_file(stage: Path, name: str) -> None:
    cfg = json.loads((stage / "triage_config.json").read_text())
    (Path(cfg["model_snapshot"]) / name).unlink()


def drop_base_file(stage: Path, name: str) -> None:
    cfg = json.loads((stage / "triage_config.json").read_text())
    (Path(cfg["nearest_snapshot"]) / name).unlink()


def gen_hazards(stage: Path) -> list[dict]:
    return [h for h in delta(stage)["hazards"] if h["area"] == "generation_config"]


def test_a_model_with_no_generation_config_gets_a_hazard_that_names_the_borrowed_settings(text_case):
    stage = text_case()
    drop_model_file(stage, "generation_config.json")
    assert run_triage(stage).returncode == 0
    [h] = gen_hazards(stage)
    assert "ships no generation_config.json" in h["finding"]
    assert "[2, 0]" in h["finding"]                                  # the nearest model's stop tokens
    assert "temperature 1.0" in h["finding"] and "top_k 20" in h["finding"] and "top_p 0.95" in h["finding"]
    assert h["evidence"] == ["stages/0/evidence/genconfig-license.json"]
    assert delta(stage)["path"] == "weights-only"                    # a flag, not a reason to stop


def test_the_evidence_records_each_models_stop_tokens(text_case):
    stage = text_case(genconfig_change={"eos_token_id": 7})
    assert run_triage(stage).returncode == 0
    g = evidence(stage, "genconfig-license.json")
    assert g["model_eos_token_ids"] == [7] and g["base_eos_token_ids"] == [2, 0]


def test_a_model_with_its_own_generation_config_and_stop_tokens_gets_no_hazard(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    assert gen_hazards(stage) == []


def test_a_generation_config_without_eos_token_id_gets_a_hazard(text_case):
    stage = text_case()
    path = json.loads((stage / "triage_config.json").read_text())["model_snapshot"]
    gen = json.loads((Path(path) / "generation_config.json").read_text())
    del gen["eos_token_id"]
    (Path(path) / "generation_config.json").write_text(json.dumps(gen))
    assert run_triage(stage).returncode == 0
    [h] = gen_hazards(stage)
    assert "has no eos_token_id" in h["finding"] and "stops after loading the weights" in h["finding"]
    assert evidence(stage, "genconfig-license.json")["model_eos_token_ids"] == []


def test_a_generation_config_that_is_not_json_counts_as_having_no_stop_tokens(text_case):
    stage = text_case()
    path = json.loads((stage / "triage_config.json").read_text())["model_snapshot"]
    (Path(path) / "generation_config.json").write_text("{not json")
    assert run_triage(stage).returncode == 0
    [h] = gen_hazards(stage)
    assert "has no eos_token_id" in h["finding"]


def test_neither_model_shipping_one_says_the_server_has_no_stop_token(text_case):
    stage = text_case()
    drop_model_file(stage, "generation_config.json")
    drop_base_file(stage, "generation_config.json")
    assert run_triage(stage).returncode == 0
    [h] = gen_hazards(stage)
    assert "Neither model ships a generation_config.json" in h["finding"]
    assert "stops after loading the weights" in h["finding"]


@pytest.mark.parametrize("value, ids", [(7, [7]), ([3, 4], [3, 4]), ("x", []), ([1, "x"], []), ([True, 3], []), (True, []), (None, [])])
def test_stop_tokens_are_read_only_when_they_are_integers(text_case, value, ids):
    stage = text_case(genconfig_change={"eos_token_id": value})
    assert run_triage(stage).returncode == 0
    assert evidence(stage, "genconfig-license.json")["model_eos_token_ids"] == ids


# ---- the stop token can also come from the staged config.json ---------------------------------------------------
# Staging copies the nearest model's config.json, and the bundle reads a top-level eos_token_id from it when
# generation_config.json has none (prepare_swap.stop_token_ids does the same). A missing or incomplete
# generation_config.json therefore predicts a failed start only when that config.json has none either.

NEAREST_CONFIG_PHRASE = "the nearest model's config.json, [7777], which staging copies"


def set_top_level_eos(snapshot_key: str, stage: Path, value) -> None:
    cfg = json.loads((stage / "triage_config.json").read_text())
    path = Path(cfg[snapshot_key]) / "config.json"
    data = json.loads(path.read_text())
    data["eos_token_id"] = value
    path.write_text(json.dumps(data))


def drop_eos(snapshot_key: str, stage: Path) -> None:
    cfg = json.loads((stage / "triage_config.json").read_text())
    path = Path(cfg[snapshot_key]) / "generation_config.json"
    gen = json.loads(path.read_text())
    del gen["eos_token_id"]
    path.write_text(json.dumps(gen))


def test_the_evidence_records_the_nearest_models_config_stop_tokens(text_case):
    stage = text_case()
    set_top_level_eos("nearest_snapshot", stage, [5, 6])
    assert run_triage(stage).returncode == 0
    assert evidence(stage, "genconfig-license.json")["base_config_eos_token_ids"] == [5, 6]


def test_a_nested_or_absent_config_stop_token_is_recorded_as_none(text_case):
    stage = text_case()                                   # the nearest model nests its config under text_config
    assert run_triage(stage).returncode == 0
    assert evidence(stage, "genconfig-license.json")["base_config_eos_token_ids"] == []


def test_a_missing_model_file_with_a_stop_token_in_the_staged_config_does_not_predict_a_failed_start(text_case):
    stage = text_case()
    drop_model_file(stage, "generation_config.json")
    drop_eos("nearest_snapshot", stage)                    # the borrowed generation_config.json has no eos_token_id
    set_top_level_eos("nearest_snapshot", stage, 7777)        # but the config.json staging copies has one
    assert run_triage(stage).returncode == 0
    [h] = gen_hazards(stage)
    assert "stops after loading the weights" not in h["finding"] and "will stop" in h["finding"]
    assert NEAREST_CONFIG_PHRASE in h["finding"]                       # says where the stop token comes from


def test_neither_file_with_a_stop_token_in_the_staged_config_does_not_predict_a_failed_start(text_case):
    stage = text_case()
    drop_model_file(stage, "generation_config.json")
    drop_base_file(stage, "generation_config.json")
    set_top_level_eos("nearest_snapshot", stage, 7777)
    assert run_triage(stage).returncode == 0
    [h] = gen_hazards(stage)
    assert "stops after loading the weights" not in h["finding"] and "no sampling defaults" in h["finding"]
    assert NEAREST_CONFIG_PHRASE in h["finding"]


def test_an_incomplete_model_file_with_a_stop_token_in_the_staged_config_does_not_predict_a_failed_start(text_case):
    stage = text_case()
    drop_eos("model_snapshot", stage)
    set_top_level_eos("nearest_snapshot", stage, 7777)
    assert run_triage(stage).returncode == 0
    [h] = gen_hazards(stage)
    assert "has no eos_token_id" in h["finding"] and "stops after loading the weights" not in h["finding"]
    assert NEAREST_CONFIG_PHRASE in h["finding"]


def test_the_new_models_own_config_stop_token_does_not_count_because_staging_replaces_its_config(text_case):
    stage = text_case()
    drop_model_file(stage, "generation_config.json")
    drop_base_file(stage, "generation_config.json")
    set_top_level_eos("model_snapshot", stage, 9)          # staging does not use the model's config.json
    assert run_triage(stage).returncode == 0
    [h] = gen_hazards(stage)
    assert "stops after loading the weights" in h["finding"]


def test_a_nearest_config_that_is_not_a_json_object_records_no_stop_token(text_case):
    stage = text_case()
    cfg = json.loads((stage / "triage_config.json").read_text())
    base_config = Path(cfg["nearest_snapshot"]) / "config.json"
    base_config.write_text("[1, 2]")
    run_triage(stage)                                       # other checks may refuse such a config; the record must still be sane
    g = evidence(stage, "genconfig-license.json")
    assert g["base_config_eos_token_ids"] == []
