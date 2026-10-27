# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The audio branch of the stage 0 template (plan docs/superpowers/plans/2026-10-28-audio-bringup.md, task A2).

A model in the `stable-audio-tools` layout has no top-level config.json: its architecture is in `model_config.json`, its weights are one
`model.safetensors` that holds several components under name prefixes, and its text encoder is a directory of its own. The branch
compares the two models' configs and tensor headers and decides the class. Everything runs on tiny fake snapshots whose safetensors files
carry real (small) values, so the value check has something to read. The script runs as a subprocess, the way the agent runs it."""
from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import pytest

from orchard.stages import gate_delta
from test_delta_triage_template import TEMPLATE, run_triage, stage_for, template_module, tokenizer_json  # noqa: F401

MODEL_REV = "8bdfb53c6b0fbc62113cd571fd37373efe15fa14"
BASE_REV = "5a6a5a6a5a6a5a6a5a6a5a6a5a6a5a6a5a6a5a6a"
MODEL_ID, BASE_ID = "stabilityai/stable-audio-3-small-music", "stabilityai/stable-audio-3-small-music-base"

CONFIG = {
    "model_type": "diffusion_cond_inpaint", "sample_size": 5292032, "sample_rate": 44100, "audio_channels": 2,
    "model": {
        "pretransform": {"type": "autoencoder", "config": {"latent_dim": 256, "downsampling_ratio": 4096}},
        "conditioning": {"cond_dim": 768, "configs": [
            {"id": "prompt", "type": "t5gemma", "config": {"max_length": 256, "repo_id": MODEL_ID, "subfolder": "t5gemma-b-b-ul2"}},
            {"id": "seconds_total", "type": "number", "config": {"min_val": 0, "max_val": 384}}]},
        "diffusion": {"type": "dit", "config": {"embed_dim": 1024, "depth": 20, "num_heads": 16}},
        "io_channels": 256},
    "training": {"use_ema": True, "arc": {"disc_update_interval": 2}},
}
TEXT_ENCODER_CONFIG = {"architectures": ["T5GemmaForConditionalGeneration"], "encoder": {"hidden_size": 768}}

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8}
TENSORS = {"model.model.layers.0.attn.weight": ("F32", [8, 8]), "model.model.layers.0.ff.weight": ("F32", [16, 8]),
           "pretransform.model.encoder.weight": ("F32", [8, 4]), "pretransform.model.decoder.weight": ("F32", [4, 8]),
           "conditioner.conditioners.seconds_total.embedder.weight": ("F32", [8, 2])}
TEXT_TENSORS = {"model.encoder.layers.0.weight": ("F32", [8, 8]), "model.decoder.layers.0.weight": ("F32", [8, 8])}


def write_safetensors(path: Path, tensors: dict, seed: float = 1.0, seeds: dict | None = None, metadata: dict | None = None) -> None:
    """A safetensors file with small real values (a ramp scaled by `seed`, or by `seeds[name]`, and shifted by the tensor's index)."""
    header, offset, blobs = {}, 0, []
    if metadata:
        header["__metadata__"] = metadata
    for i, (name, (dtype, shape)) in enumerate(sorted(tensors.items())):
        n = 1
        for s in shape:
            n *= s
        if dtype == "F32":
            data = struct.pack(f"<{n}f", *[(seeds or {}).get(name, seed) * (k + 1 + i) for k in range(n)])
        else:
            data = b"\0" * (n * DTYPE_BYTES[dtype])
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


def make_audio(root: Path, rev: str, *, repo: str, config=None, tensors=None, seed=1.0, text_config=None, text_tensors=None,
               extra=None, license_id="other") -> Path:
    snap = root / "snapshots" / rev
    (snap / "t5gemma-b-b-ul2").mkdir(parents=True)
    cfg = copy.deepcopy(CONFIG) if config is None else config
    if config is None:
        cfg["model"]["conditioning"]["configs"][0]["config"]["repo_id"] = repo
    (snap / "model_config.json").write_text(json.dumps(cfg))
    write_safetensors(snap / "model.safetensors", TENSORS if tensors is None else tensors, seed)
    (snap / "t5gemma-b-b-ul2" / "config.json").write_text(json.dumps(TEXT_ENCODER_CONFIG if text_config is None else text_config))
    write_safetensors(snap / "t5gemma-b-b-ul2" / "model.safetensors", TEXT_TENSORS if text_tensors is None else text_tensors, 3.0)
    (snap / "README.md").write_text(f"---\nlicense: {license_id}\nlicense_name: stable-audio-community\nlibrary_name: stable-audio-3\n"
                                    "pipeline_tag: text-to-audio\n---\n\n# Audio\nIt makes music.\n")
    (snap / "LICENSE.md").write_text("Stability AI Community License\n")
    for name, content in (extra or {}).items():
        (snap / name).write_bytes(content if isinstance(content, bytes) else content.encode())
    return snap


@pytest.fixture
def audio_case(tmp_path):
    def build(model_kw=None, base_kw=None):
        model = make_audio(tmp_path / "new", MODEL_REV, repo=MODEL_ID, seed=2.0, **(model_kw or {}))
        base = make_audio(tmp_path / "base", BASE_REV, repo=BASE_ID, seed=1.0, **(base_kw or {}))
        return stage_for(tmp_path, model, base, model_id=MODEL_ID, nearest_id=BASE_ID)
    return build


def run_ok(stage):
    r = run_triage(stage)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((stage / "delta.json").read_text())


def gate(stage):
    return gate_delta(stage, stage.parent.parent)


def reasons(d):
    return " | ".join(d["path_reasons"])


# ---- the layout check ---------------------------------------------------------------------------

def test_a_stable_audio_tools_snapshot_is_audio(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    assert template_module().detect_audio(snap) is True


def test_a_text_model_with_a_stray_model_config_is_not_audio(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID, extra={"config.json": "{}"})
    assert template_module().detect_audio(snap) is False


@pytest.mark.parametrize("config", ["[]", "{}", '{"model_type": "x"}', '{"model": {}}', '{"model_type": 3, "model": {}}',
                                    "not json"])
def test_a_model_config_that_does_not_look_like_one_is_not_audio(tmp_path, config):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    (snap / "model_config.json").write_text(config)
    assert template_module().detect_audio(snap) is False


def test_a_snapshot_without_weights_is_not_audio(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    (snap / "model.safetensors").unlink()
    assert template_module().detect_audio(snap) is False


def test_an_oversized_model_config_is_not_read(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    (snap / "model_config.json").write_text(json.dumps({**CONFIG, "pad": "x" * (2 << 20)}))
    assert template_module().detect_audio(snap) is False


# ---- a fine-tune of the same architecture -----------------------------------------------------

def test_a_fine_tune_with_the_same_layout_is_weights_only(audio_case):
    stage = audio_case()
    d = run_ok(stage)
    assert d["class"] == "weights-only" and d["path"] == "weights-only" and d["path_reasons"] == []
    assert d["facts"] == {"form": "full", "modality": "audio"}
    assert gate(stage).ok, gate(stage).reasons


def test_the_delta_describes_the_components(audio_case):
    d = run_ok(audio_case())
    comp = {c["prefix"]: c for c in d["audio"]["components"]}
    assert set(comp) == {"model.model", "pretransform.model", "conditioner.conditioners"}
    assert comp["model.model"]["tensors"] == 2 and comp["model.model"]["parameters"] == 64 + 128
    assert comp["model.model"]["role"] == "diffusion transformer" and comp["pretransform.model"]["role"] == "autoencoder"
    assert d["audio"]["sample_rate"] == 44100 and d["audio"]["audio_channels"] == 2


def test_the_values_are_sampled_and_a_fine_tune_is_not_identical(audio_case):
    d = run_ok(audio_case())
    assert d["values"]["summary"]["identical"] == 0 and sum(d["values"]["summary"].values()) > 0


def test_identical_weights_are_flagged_as_a_copy_of_the_base(tmp_path):
    model = make_audio(tmp_path / "new", MODEL_REV, repo=MODEL_ID, seed=1.0)
    base = make_audio(tmp_path / "base", BASE_REV, repo=BASE_ID, seed=1.0)
    d = run_ok(stage_for(tmp_path, model, base, model_id=MODEL_ID, nearest_id=BASE_ID))
    assert any("identical" in h["finding"] and "copy" in h["finding"] for h in d["hazards"])


def test_a_differing_repo_id_is_a_pointer_and_does_not_change_the_class(audio_case):
    d = run_ok(audio_case())
    assert d["class"] == "weights-only"
    pointer = d["audio"]["config"]["pointer_differences"]
    assert any(k.endswith("config.repo_id") for k in pointer)


def test_a_training_section_difference_is_listed_and_allowed(audio_case):
    cfg = copy.deepcopy(CONFIG)
    cfg["model"]["conditioning"]["configs"][0]["config"]["repo_id"] = MODEL_ID
    cfg["training"]["arc"]["disc_update_interval"] = 4
    cfg["training"]["demo"] = {"x": 1}
    d = run_ok(audio_case(base_kw={"config": cfg}))
    assert d["class"] == "weights-only"
    assert "training.arc.disc_update_interval" in d["audio"]["config"]["training_differences"]
    assert "training.demo.x" in d["audio"]["config"]["training_differences"]


def test_files_only_in_the_base_are_named(audio_case):
    d = run_ok(audio_case(base_kw={"extra": {"svd_bases.pt": b"x" * 10}}))
    files = next(x for x in d["differences"] if x["area"] == "files")
    assert "svd_bases.pt" in files["finding"]
    assert d["audio"]["files_only_in_base"] == ["svd_bases.pt"]


def test_the_delta_names_the_license_and_its_difference_areas(audio_case):
    d = run_ok(audio_case())
    areas = {x["area"] for x in d["differences"]}
    assert {"config", "architecture", "tensors", "tensor_names", "files", "license"} <= areas
    lic = next(x for x in d["differences"] if x["area"] == "license")
    assert "other" in lic["finding"]


def test_a_hazard_says_no_stage_judges_the_sound(audio_case):
    d = run_ok(audio_case())
    assert any("No stage listens to the output" in h["finding"] and "sound" in h["finding"] for h in d["hazards"])


# ---- what makes it a full port ------------------------------------------------------------------

def changed_model_config(**sections):
    cfg = copy.deepcopy(CONFIG)
    cfg["model"]["conditioning"]["configs"][0]["config"]["repo_id"] = MODEL_ID
    for dotted, value in sections.items():
        node, keys = cfg, dotted.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return cfg


def test_a_different_diffusion_depth_is_a_full_port(audio_case):
    d = run_ok(audio_case(model_kw={"config": changed_model_config(model__diffusion__config__depth=24)}))
    assert d["class"] == "full-port" and d["path"] == "full-port"
    assert "model.diffusion.config.depth" in reasons(d) and "24" in reasons(d)


def test_a_different_sample_rate_is_a_full_port(audio_case):
    d = run_ok(audio_case(model_kw={"config": changed_model_config(sample_rate=48000)}))
    assert d["class"] == "full-port" and "sample_rate" in reasons(d)


def test_a_different_model_type_is_a_full_port(audio_case):
    d = run_ok(audio_case(model_kw={"config": changed_model_config(model_type="diffusion_cond")}))
    assert d["class"] == "full-port" and "model_type" in reasons(d)


def test_a_new_key_in_the_model_section_is_a_full_port(audio_case):
    d = run_ok(audio_case(model_kw={"config": changed_model_config(model__extra={"a": 1})}))
    assert d["class"] == "full-port" and "model.extra.a" in reasons(d)


def test_a_shape_difference_is_a_full_port(audio_case):
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("F32", [8, 9])}
    d = run_ok(audio_case(model_kw={"tensors": t}))
    assert d["class"] == "full-port" and "model.model.layers.0.attn.weight" in reasons(d)


def test_a_tensor_only_in_the_model_is_a_full_port(audio_case):
    t = {**TENSORS, "model.model.layers.1.attn.weight": ("F32", [8, 8])}
    d = run_ok(audio_case(model_kw={"tensors": t}))
    assert d["class"] == "full-port" and "only in the new model" in reasons(d)


def test_a_tensor_only_in_the_base_is_a_full_port(audio_case):
    t = {**TENSORS, "model.model.layers.1.attn.weight": ("F32", [8, 8])}
    d = run_ok(audio_case(base_kw={"tensors": t}))
    assert d["class"] == "full-port" and "only in the nearest model" in reasons(d)


def test_a_small_float_dtype_difference_is_tolerated_as_in_the_text_path(audio_case):
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("BF16", [8, 8])}
    d = run_ok(audio_case(model_kw={"tensors": t}))
    assert d["class"] == "weights-only"


def test_an_integer_dtype_difference_is_a_full_port(audio_case):
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("I64", [8, 8])}
    d = run_ok(audio_case(model_kw={"tensors": t}))
    assert d["class"] == "full-port" and "dtype" in reasons(d)


def test_a_different_text_encoder_config_is_a_full_port(audio_case):
    other = {**TEXT_ENCODER_CONFIG, "encoder": {"hidden_size": 1024}}
    d = run_ok(audio_case(model_kw={"text_config": other}))
    assert d["class"] == "full-port" and "t5gemma-b-b-ul2" in reasons(d)


def test_a_different_text_encoder_tensor_shape_is_a_full_port(audio_case):
    t = {**TEXT_TENSORS, "model.encoder.layers.0.weight": ("F32", [8, 9])}
    d = run_ok(audio_case(model_kw={"text_tensors": t}))
    assert d["class"] == "full-port" and "t5gemma-b-b-ul2" in reasons(d)


def test_text_encoder_weights_that_differ_in_value_are_fine(audio_case):
    # make_audio writes the text encoder with the same values for both models; change only the values on one side
    stage = audio_case()
    model = Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])
    write_safetensors(model / "t5gemma-b-b-ul2" / "model.safetensors", TEXT_TENSORS, 9.0)
    assert run_ok(stage)["class"] == "weights-only"


def test_a_text_encoder_from_another_repository_is_not_compared(audio_case):
    cfg = changed_model_config()
    cfg["model"]["conditioning"]["configs"][0]["config"] = {"max_length": 256, "repo_id": "google/t5gemma-b-b-ul2", "subfolder": "t5gemma-b-b-ul2"}
    base_cfg = copy.deepcopy(cfg)
    d = run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": base_cfg}))
    assert d["class"] == "weights-only"
    assert any("another repository" in h["finding"] for h in d["hazards"])


# ---- what leaves the class unknown ----------------------------------------------------------------

def test_an_unreadable_weight_header_is_unknown(audio_case):
    stage = audio_case()
    model = Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])
    (model / "model.safetensors").write_bytes(b"\xff" * 16)
    d = run_ok(stage)
    assert d["class"] == "unknown" and d["path"] == "full-port"
    assert gate(stage).ok


def test_a_header_larger_than_the_limit_is_unknown(audio_case):
    stage = audio_case()
    model = Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])
    (model / "model.safetensors").write_bytes(struct.pack("<Q", 9 << 20) + b"{" + b" " * 64)
    assert run_ok(stage)["class"] == "unknown"


def test_a_nearest_model_that_is_not_an_audio_layout_is_unknown(tmp_path):
    model = make_audio(tmp_path / "new", MODEL_REV, repo=MODEL_ID)
    base = make_audio(tmp_path / "base", BASE_REV, repo=BASE_ID, extra={"config.json": "{}"})
    d = run_ok(stage_for(tmp_path, model, base, model_id=MODEL_ID, nearest_id=BASE_ID))
    assert d["class"] == "unknown" and "not in the stable-audio-tools layout" in reasons(d)


def test_a_missing_conditioner_subfolder_is_unknown(audio_case):
    stage = audio_case()
    model = Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])
    (model / "t5gemma-b-b-ul2" / "model.safetensors").unlink()
    d = run_ok(stage)
    assert d["class"] == "unknown" and "t5gemma-b-b-ul2" in reasons(d)


# ---- the record ------------------------------------------------------------------------------------

def test_the_script_names_the_class_and_the_audio_branch(audio_case):
    r = run_triage(audio_case())
    assert r.returncode == 0 and "audio" in r.stdout and "weights-only" in r.stdout


def test_the_template_hash_is_stamped_so_the_gate_accepts_it(audio_case):
    stage = audio_case()
    d = run_ok(stage)
    import hashlib
    assert d["template_sha256"] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()


def test_a_text_run_writes_no_facts(tmp_path, tokenizer_json):
    from test_delta_triage_template import BASE_REV as TB, MODEL_REV as TM, make_snapshot
    model = make_snapshot(tmp_path / "new", TM, nested=False, vision=False, tokenizer=tokenizer_json)
    base = make_snapshot(tmp_path / "base", TB, nested=True, vision=True, tokenizer=tokenizer_json)
    stage = stage_for(tmp_path, model, base)
    r = run_triage(stage)
    assert r.returncode == 0, r.stderr
    assert "facts" not in json.loads((stage / "delta.json").read_text())



def test_a_base_config_that_names_the_new_models_repository_still_compares_the_subfolder(audio_case):
    cfg = changed_model_config()
    d = run_ok(audio_case(base_kw={"config": cfg}))
    assert d["audio"]["text_encoders"][0]["compared"] is True and d["class"] == "weights-only"


def test_a_subfolder_that_escapes_the_snapshot_is_not_followed(audio_case):
    cfg = changed_model_config()
    cfg["model"]["conditioning"]["configs"][0]["config"]["subfolder"] = "../outside"
    d = run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": copy.deepcopy(cfg)}))
    assert d["audio"]["text_encoders"] == []


# ---- guards the first mutation table found unguarded -------------------------------------------------

def model_snapshot(stage):
    return Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])


def test_a_header_over_the_limit_that_is_valid_json_is_refused(audio_case):
    stage = audio_case()
    pad = json.dumps({"__metadata__": {"x": "y" * (9 << 20)}}).encode()
    (model_snapshot(stage) / "model.safetensors").write_bytes(struct.pack("<Q", len(pad)) + pad)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "larger than" in reasons(d)


def test_a_file_shorter_than_a_length_prefix_is_unknown_and_does_not_crash(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "model.safetensors").write_bytes(b"\x01\x02\x03\x04")
    assert run_ok(stage)["class"] == "unknown"


@pytest.mark.parametrize("entry", [{"dtype": "F32", "shape": "8x8", "data_offsets": [0, 4]},
                                   {"dtype": "F32", "shape": [8, -1], "data_offsets": [0, 4]},
                                   {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]},
                                   {"dtype": 32, "shape": [1], "data_offsets": [0, 4]},
                                   {"shape": [1], "data_offsets": [0, 4]},
                                   "not an object"])
def test_a_malformed_tensor_entry_is_unknown_and_does_not_crash(audio_case, entry):
    stage = audio_case()
    blob = json.dumps({"model.model.layers.0.attn.weight": entry}).encode()
    blob += b" " * (-len(blob) % 8)
    (model_snapshot(stage) / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 8)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "malformed entry" in reasons(d)


def test_a_metadata_entry_in_the_header_is_not_a_tensor(audio_case):
    stage = audio_case()
    model = model_snapshot(stage)
    write_safetensors(model / "model.safetensors", TENSORS, 2.0, metadata={"format": "pt"})
    assert run_ok(stage)["class"] == "weights-only"


def test_a_directory_named_like_a_weight_file_is_not_read(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "extra.safetensors").mkdir()
    assert run_ok(stage)["class"] == "weights-only"


def test_a_subfolder_only_the_model_names_is_a_full_port(audio_case):
    cfg = changed_model_config()
    cfg["model"]["conditioning"]["configs"][0]["config"]["subfolder"] = "t5gemma-b-b-ul2"
    other = copy.deepcopy(cfg)
    del other["model"]["conditioning"]["configs"][0]["config"]["subfolder"]
    d = run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": other}))
    assert d["class"] == "full-port" and "named by the new model's config only" in reasons(d)


def test_an_extra_text_encoder_tensor_is_a_full_port(audio_case):
    t = {**TEXT_TENSORS, "model.encoder.layers.1.weight": ("F32", [8, 8])}
    d = run_ok(audio_case(model_kw={"text_tensors": t}))
    assert d["class"] == "full-port" and "t5gemma-b-b-ul2 tensors exist only in the new model" in reasons(d)


def test_a_missing_text_encoder_directory_is_unknown_and_does_not_crash(audio_case):
    stage = audio_case()
    import shutil
    shutil.rmtree(model_snapshot(stage) / "t5gemma-b-b-ul2")
    d = run_ok(stage)
    assert d["class"] == "unknown" and "not a directory in the new model" in reasons(d)


def test_weights_that_are_identical_only_in_part_are_not_called_a_copy(tmp_path):
    model = make_audio(tmp_path / "new", MODEL_REV, repo=MODEL_ID, seed=1.0)
    base = make_audio(tmp_path / "base", BASE_REV, repo=BASE_ID, seed=1.0)
    write_safetensors(model / "model.safetensors", TENSORS, 1.0, seeds={"model.model.layers.0.ff.weight": 5.0})
    d = run_ok(stage_for(tmp_path, model, base, model_id=MODEL_ID, nearest_id=BASE_ID))
    assert d["values"]["summary"]["identical"] > 0 and d["values"]["summary"]["near"] + d["values"]["summary"]["changed"] > 0
    assert not any("copy of it" in h["finding"] for h in d["hazards"])


def test_a_file_only_in_the_base_gets_a_hazard(audio_case):
    d = run_ok(audio_case(base_kw={"extra": {"svd_bases.pt": b"x"}}))
    assert any("svd_bases.pt" in h["finding"] and "lacks" in h["finding"] for h in d["hazards"])


def test_a_boolean_sample_rate_is_not_a_number(audio_case):
    cfg = changed_model_config()
    cfg["sample_rate"] = True
    base = changed_model_config()
    base["sample_rate"] = True
    d = run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": base}))
    assert d["audio"]["sample_rate"] is None


def test_the_recorded_differences_are_capped_but_the_class_counts_all(audio_case):
    cfg = copy.deepcopy(changed_model_config())
    cfg["training"]["many"] = {f"k{i}": i for i in range(80)}
    d = run_ok(audio_case(model_kw={"config": cfg}))
    assert len(d["audio"]["config"]["training_differences"]) == 50
    config = next(x for x in d["differences"] if x["area"] == "config")
    assert "80 in the training section" in config["finding"] or "81 in the training section" in config["finding"]


# ---- Copilot's review of PR 123: empty maps, offset pairs, the conditioning section --------------------

def write_header_only(path, header):
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


@pytest.mark.parametrize("header", [{}, {"__metadata__": {"format": "pt"}}])
def test_an_empty_tensor_map_in_the_new_model_is_unknown(audio_case, header):
    stage = audio_case()
    write_header_only(model_snapshot(stage) / "model.safetensors", header)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "new model holds no tensors" in reasons(d)
    assert gate(stage).ok


@pytest.mark.parametrize("header", [{}, {"__metadata__": {"format": "pt"}}])
def test_an_empty_tensor_map_in_the_nearest_model_is_unknown(audio_case, header):
    stage = audio_case()
    cfg = json.loads((stage / "triage_config.json").read_text())
    write_header_only(Path(cfg["nearest_snapshot"]) / "model.safetensors", header)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "nearest model holds no tensors" in reasons(d)


def test_two_empty_tensor_maps_are_not_a_weights_only_pair(audio_case):
    stage = audio_case()
    cfg = json.loads((stage / "triage_config.json").read_text())
    write_header_only(Path(cfg["nearest_snapshot"]) / "model.safetensors", {})
    write_header_only(model_snapshot(stage) / "model.safetensors", {})
    assert run_ok(stage)["class"] == "unknown"


@pytest.mark.parametrize("offsets", [[0], [], [0, 4, 8], [4, 0], [-1, 4], ["a", 4], [0, True], "0-4", None, [0, 10 ** 9]])
def test_a_tensor_offset_pair_that_is_not_a_valid_range_is_a_header_problem(audio_case, offsets):
    stage = audio_case()
    entry = {"dtype": "F32", "shape": [1], "data_offsets": offsets}
    blob = json.dumps({"model.model.layers.0.attn.weight": entry}).encode()
    blob += b" " * (-len(blob) % 8)
    (model_snapshot(stage) / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 4)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "malformed entry" in reasons(d)
    assert gate(stage).ok


def test_a_missing_offset_pair_is_a_header_problem(audio_case):
    stage = audio_case()
    blob = json.dumps({"model.model.layers.0.attn.weight": {"dtype": "F32", "shape": [1]}}).encode()
    blob += b" " * (-len(blob) % 8)
    (model_snapshot(stage) / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 4)
    assert run_ok(stage)["class"] == "unknown"


def test_an_offset_pair_at_the_end_of_the_data_is_valid(audio_case):
    stage = audio_case()
    write_safetensors(model_snapshot(stage) / "model.safetensors", TENSORS, 2.0)
    assert run_ok(stage)["class"] == "weights-only"


@pytest.mark.parametrize("conditioning", [[1], ["x"], "text", 3, [{"id": "prompt"}], {"configs": "x"}, {"configs": [1, "x", None]},
                                          {"configs": [{"config": "x"}]}, {"configs": [{"config": {"subfolder": 5}}]}])
def test_a_malformed_conditioning_section_is_unknown_and_does_not_crash(audio_case, conditioning):
    cfg = changed_model_config()
    cfg["model"]["conditioning"] = conditioning
    base = changed_model_config()
    stage = audio_case(model_kw={"config": cfg}, base_kw={"config": base})
    d = run_ok(stage)
    assert gate(stage).ok
    assert d["class"] == "unknown" and "new model" in reasons(d) and "model.conditioning" in reasons(d)


def test_a_malformed_conditioning_section_in_the_nearest_model_is_unknown(audio_case):
    base = changed_model_config()
    base["model"]["conditioning"] = [{"id": "prompt"}]
    d = run_ok(audio_case(base_kw={"config": base}))
    assert d["class"] == "unknown" and "nearest model" in reasons(d) and "conditioning" in reasons(d)


def test_a_model_without_a_conditioning_section_has_no_text_encoder_and_is_not_a_problem(audio_case):
    cfg = changed_model_config()
    del cfg["model"]["conditioning"]
    base = copy.deepcopy(cfg)
    d = run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": base}))
    assert d["class"] == "weights-only" and d["audio"]["text_encoders"] == []


def test_a_conditioning_entry_without_a_config_is_not_a_problem(audio_case):
    cfg = changed_model_config()
    cfg["model"]["conditioning"]["configs"].append({"id": "extra", "type": "number"})
    base = copy.deepcopy(cfg)
    assert run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": base}))["class"] == "weights-only"


def test_an_unreadable_header_is_not_also_reported_as_an_empty_map(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "model.safetensors").write_bytes(b"\xff" * 16)
    assert "holds no tensors" not in reasons(run_ok(stage))


def test_a_nearest_model_that_is_not_an_audio_layout_is_not_also_reported_as_an_empty_map(tmp_path):
    model = make_audio(tmp_path / "new", MODEL_REV, repo=MODEL_ID)
    base = make_audio(tmp_path / "base", BASE_REV, repo=BASE_ID, extra={"config.json": "{}"})
    d = run_ok(stage_for(tmp_path, model, base, model_id=MODEL_ID, nearest_id=BASE_ID))
    assert "holds no tensors" not in reasons(d)


def test_a_conditioning_section_without_configs_is_not_a_problem(audio_case):
    cfg = changed_model_config()
    cfg["model"]["conditioning"] = {"cond_dim": 768}
    base = copy.deepcopy(cfg)
    d = run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": base}))
    assert d["class"] == "weights-only" and d["audio"]["text_encoders"] == []


def test_a_configs_value_that_is_not_a_list_is_named(audio_case):
    cfg = changed_model_config()
    cfg["model"]["conditioning"]["configs"] = {"a": 1}
    d = run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": changed_model_config()}))
    assert d["class"] == "unknown" and "configs is not a list" in reasons(d)


# ---- round three: a tensor's byte span must match its dtype and shape --------------------------------

def one_tensor_file(path, dtype, shape, span, payload=None):
    entry = {"dtype": dtype, "shape": shape, "data_offsets": [0, span]}
    blob = json.dumps({"model.model.layers.0.attn.weight": entry}).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * (span if payload is None else payload))


@pytest.mark.parametrize("dtype,shape,span", [("F32", [8, 8], 4), ("F32", [8, 8], 255), ("F32", [8, 8], 257), ("BF16", [4], 7), ("F16", [3], 5),
                                              ("F64", [2], 8), ("I64", [2], 8 + 1), ("I32", [5], 4), ("U8", [3], 4), ("BOOL", [3], 2), ("I8", [3], 4), ("F8_E4M3", [3], 2),
                                              ("F32", [0], 4), ("F32", [2, 0], 4), ("F32", [], 8)])
def test_a_tensor_whose_byte_span_does_not_fit_its_dtype_and_shape_is_malformed(audio_case, dtype, shape, span):
    stage = audio_case()
    one_tensor_file(model_snapshot(stage) / "model.safetensors", dtype, shape, span)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "malformed entry" in reasons(d) and "byte span" in reasons(d)
    assert gate(stage).ok


@pytest.mark.parametrize("dtype,shape,span", [("F32", [8, 8], 256), ("BF16", [4], 8), ("F16", [3], 6), ("F64", [2], 16), ("I64", [2], 16),
                                              ("I32", [5], 20), ("U8", [3], 3), ("BOOL", [3], 3), ("F32", [0], 0), ("F32", [], 4),
                                              ("F8_E4M3", [5], 5), ("I8", [3], 3), ("F8_E5M2", [4], 4), ("I16", [3], 6), ("U16", [3], 6), ("U32", [2], 8), ("U64", [2], 16)])
def test_a_tensor_whose_span_is_exactly_its_size_is_accepted(audio_case, dtype, shape, span):
    stage = audio_case()
    model = model_snapshot(stage)
    one_tensor_file(model / "model.safetensors", dtype, shape, span)
    cfg = json.loads((stage / "triage_config.json").read_text())
    one_tensor_file(Path(cfg["nearest_snapshot"]) / "model.safetensors", dtype, shape, span)
    d = run_ok(stage)
    assert "byte span" not in reasons(d) and "malformed" not in reasons(d)


def test_a_dtype_this_script_does_not_know_is_accepted_with_a_valid_range(audio_case):
    stage = audio_case()
    cfg = json.loads((stage / "triage_config.json").read_text())
    for snap in (model_snapshot(stage), Path(cfg["nearest_snapshot"])):
        one_tensor_file(snap / "model.safetensors", "F6_E2M3", [5], 4)
    assert "malformed" not in reasons(run_ok(stage))


def test_the_span_check_names_the_expected_size(audio_case):
    stage = audio_case()
    one_tensor_file(model_snapshot(stage) / "model.safetensors", "F32", [8, 8], 4)
    assert "256" in reasons(run_ok(stage))
