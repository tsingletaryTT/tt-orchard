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
