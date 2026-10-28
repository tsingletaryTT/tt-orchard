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
import subprocess
import sys
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
        "diffusion": {"type": "dit", "diffusion_objective": "rf_denoiser", "config": {"embed_dim": 1024, "depth": 20, "num_heads": 16}},
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


# ---- round four: overlapping and gapped ranges, duplicate names, empty containers, strict JSON ------------------

def raw_file(path, entries, payload_size, metadata=None, extra_header=None):
    header = dict(entries)
    if metadata is not None:
        header["__metadata__"] = metadata
    if extra_header:
        header.update(extra_header)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * payload_size)


def entry(offsets, dtype="F32", shape=(1,), **extra):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets), **extra}


def head_text(stage):
    return reasons(run_ok(stage))


NAME_A, NAME_B = "model.model.layers.0.attn.weight", "model.model.layers.0.ff.weight"


@pytest.mark.parametrize("a,b,size,why", [
    ((0, 4), (0, 4), 4, "overlap"), ((0, 8), (4, 8), 8, "overlap"), ((0, 8), (4, 12), 12, "overlap"),
    ((0, 4), (8, 12), 12, "gap"), ((4, 8), (8, 12), 12, "gap"), ((0, 4), (4, 8), 12, "gap"), ((0, 4), (4, 8), 6, "past"),
])
def test_tensor_ranges_must_tile_the_data_section_without_overlap_or_gap(audio_case, a, b, size, why):
    stage = audio_case()
    raw_file(model_snapshot(stage) / "model.safetensors", {NAME_A: entry(a, shape=(((a[1] - a[0]) // 4,))), NAME_B: entry(b, shape=(((b[1] - b[0]) // 4,)))}, size)
    text = head_text(stage)
    assert {"overlap": "overlaps", "gap": "no tensor claims", "past": "malformed"}[why] in text
    assert run_ok(stage)["class"] == "unknown"


def test_contiguous_ranges_in_any_header_order_are_accepted(audio_case):
    stage = audio_case()
    raw_file(model_snapshot(stage) / "model.safetensors", {NAME_B: entry((4, 8)), NAME_A: entry((0, 4))}, 8)
    cfg = json.loads((stage / "triage_config.json").read_text())
    raw_file(Path(cfg["nearest_snapshot"]) / "model.safetensors", {NAME_A: entry((0, 4)), NAME_B: entry((4, 8))}, 8)
    assert run_ok(stage)["class"] == "weights-only"


def test_empty_ranges_may_sit_anywhere_but_the_data_must_still_be_covered(audio_case):
    stage = audio_case()
    raw_file(model_snapshot(stage) / "model.safetensors", {NAME_A: entry((0, 4)), NAME_B: entry((4, 4), shape=(0,)), "x.empty": entry((0, 0), shape=(0,))}, 4)
    cfg = json.loads((stage / "triage_config.json").read_text())
    raw_file(Path(cfg["nearest_snapshot"]) / "model.safetensors", {NAME_A: entry((0, 4)), NAME_B: entry((4, 4), shape=(0,)), "x.empty": entry((0, 0), shape=(0,))}, 4)
    assert run_ok(stage)["class"] == "weights-only"


def test_trailing_data_no_tensor_claims_is_hidden_data_and_malformed(audio_case):
    stage = audio_case()
    raw_file(model_snapshot(stage) / "model.safetensors", {NAME_A: entry((0, 4))}, 64)
    assert "malformed" in head_text(stage) or "hid" in head_text(stage)


def test_data_with_no_tensor_at_all_is_not_an_empty_checkpoint_either(audio_case):
    stage = audio_case()
    raw_file(model_snapshot(stage) / "model.safetensors", {}, 16)
    assert run_ok(stage)["class"] == "unknown"


@pytest.mark.parametrize("extra", [{"surprise": 1}, {"strides": [1]}])
def test_a_header_entry_with_an_unknown_key_is_malformed(audio_case, extra):
    stage = audio_case()
    raw_file(model_snapshot(stage) / "model.safetensors", {NAME_A: entry((0, 4), **extra)}, 4)
    assert "malformed entry" in head_text(stage)


@pytest.mark.parametrize("drop", ["dtype", "shape", "data_offsets"])
def test_a_header_entry_missing_a_key_is_malformed(audio_case, drop):
    stage = audio_case()
    e = entry((0, 4))
    del e[drop]
    raw_file(model_snapshot(stage) / "model.safetensors", {NAME_A: e}, 4)
    assert "malformed entry" in head_text(stage)


@pytest.mark.parametrize("meta", ["text", 3, ["a"], None])
def test_a_metadata_entry_that_is_not_an_object_is_malformed(audio_case, meta):
    stage = audio_case()
    raw_file(model_snapshot(stage) / "model.safetensors", {NAME_A: entry((0, 4))}, 4, extra_header={"__metadata__": meta})
    assert "__metadata__" in head_text(stage)


def test_metadata_values_must_be_text(audio_case):
    stage = audio_case()
    raw_file(model_snapshot(stage) / "model.safetensors", {NAME_A: entry((0, 4))}, 4, metadata={"format": 3})
    assert "__metadata__" in head_text(stage)


def test_the_same_tensor_name_in_two_files_is_a_problem_not_a_silent_replacement(audio_case):
    stage = audio_case()
    model = model_snapshot(stage)
    write_safetensors(model / "z-extra.safetensors", {"model.model.layers.0.attn.weight": ("F32", [8, 8])}, 5.0)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "appears in both" in reasons(d) and "model.model.layers.0.attn.weight" in reasons(d)
    assert gate(stage).ok


def test_the_same_name_in_two_files_of_the_nearest_model_is_a_problem(audio_case):
    stage = audio_case()
    cfg = json.loads((stage / "triage_config.json").read_text())
    write_safetensors(Path(cfg["nearest_snapshot"]) / "z-extra.safetensors", {"model.model.layers.0.attn.weight": ("F32", [8, 8])}, 5.0)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "nearest model" in reasons(d) and "appears in both" in reasons(d)


def test_two_files_with_distinct_names_are_both_read(audio_case):
    stage = audio_case()
    model = model_snapshot(stage)
    write_safetensors(model / "z-extra.safetensors", {"model.model.extra.weight": ("F32", [2])}, 5.0)
    d = run_ok(stage)
    assert d["class"] == "full-port" and "only in the new model" in reasons(d)


@pytest.mark.parametrize("empty", [{}, []])
def test_an_empty_container_added_to_the_model_section_is_a_difference(audio_case, empty):
    cfg = changed_model_config(model__diffusion__sampling_distribution_shift_options=empty)
    d = run_ok(audio_case(model_kw={"config": cfg}))
    assert d["class"] == "full-port" and "sampling_distribution_shift_options" in reasons(d)


def test_an_empty_container_in_both_configs_is_not_a_difference(audio_case):
    cfg = changed_model_config(model__diffusion__sampling_distribution_shift_options={})
    assert run_ok(audio_case(model_kw={"config": cfg}, base_kw={"config": copy.deepcopy(cfg)}))["class"] == "weights-only"


def test_an_empty_list_against_an_empty_object_is_a_difference(audio_case):
    a = changed_model_config(model__x={})
    b = changed_model_config(model__x=[])
    assert run_ok(audio_case(model_kw={"config": a}, base_kw={"config": b}))["class"] == "full-port"


def test_an_empty_container_at_the_top_level_is_a_difference(audio_case):
    cfg = changed_model_config()
    cfg["extras"] = {}
    assert run_ok(audio_case(model_kw={"config": cfg}))["class"] == "full-port"


def test_a_list_that_is_longer_by_empty_members_is_a_difference(audio_case):
    a = changed_model_config(model__x=[1])
    b = changed_model_config(model__x=[1, {}])
    assert run_ok(audio_case(model_kw={"config": a}, base_kw={"config": b}))["class"] == "full-port"


@pytest.mark.parametrize("a,b", [(1, True), (0, False), (1, 1.0), (2, "2")])
def test_values_that_python_calls_equal_but_json_types_differ_are_differences(audio_case, a, b):
    ca, cb = changed_model_config(model__x=a), changed_model_config(model__x=b)
    d = run_ok(audio_case(model_kw={"config": ca}, base_kw={"config": cb}))
    assert d["class"] == "full-port" and "model.x" in reasons(d)


def test_a_boolean_and_an_integer_in_a_text_encoder_config_are_different(audio_case):
    d = run_ok(audio_case(model_kw={"text_config": {**TEXT_ENCODER_CONFIG, "flag": 1}}, base_kw={"text_config": {**TEXT_ENCODER_CONFIG, "flag": True}}))
    assert d["class"] == "full-port" and "config.json differs" in reasons(d)


def test_a_text_encoder_config_with_an_empty_container_added_differs(audio_case):
    d = run_ok(audio_case(model_kw={"text_config": {**TEXT_ENCODER_CONFIG, "extra": {}}}))
    assert d["class"] == "full-port"


@pytest.mark.parametrize("text", ['{"model_type": "x", "model": {"a": 1, "a": 2}}', '{"model_type": "x", "model": {}, "model_type": "y"}',
                                  '{"model_type": "x", "model": {"a": NaN}}', '{"model_type": "x", "model": {"a": Infinity}}',
                                  '{"model_type": "x", "model": {"a": -Infinity}}', '{"model_type": "x", "model": {"a": 1e999}}',
                                  '{"model_type": "x", "model": {"a": ' + "9" * 40 + "}}"])
def test_a_config_with_duplicate_keys_non_finite_numbers_or_huge_numbers_is_not_read(tmp_path, text):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    (snap / "model_config.json").write_text(text)
    assert template_module().detect_audio(snap) is False


def test_a_config_nested_deeper_than_the_limit_is_not_read(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    deep = "[" * 200 + "1" + "]" * 200
    (snap / "model_config.json").write_text('{"model_type": "x", "model": {"a": ' + deep + "}}")
    assert template_module().detect_audio(snap) is False


def test_a_config_just_inside_the_depth_limit_is_read(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    deep = "[" * 20 + "1" + "]" * 20
    (snap / "model_config.json").write_text('{"model_type": "x", "model": {"a": ' + deep + "}}")
    assert template_module().detect_audio(snap) is True


def test_a_pathologically_deep_config_does_not_crash_the_reader(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    (snap / "model_config.json").write_text('{"model_type": "x", "model": {"a": ' + "[" * 100000 + "}}")
    assert template_module().detect_audio(snap) is False


def test_a_header_with_a_duplicate_tensor_name_is_malformed(audio_case):
    stage = audio_case()
    e = json.dumps(entry((0, 4)))
    blob = ('{"%s": %s, "%s": %s}' % (NAME_A, e, NAME_A, e)).encode()
    blob += b" " * (-len(blob) % 8)
    (model_snapshot(stage) / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 4)
    assert "duplicate" in head_text(stage)


def test_a_header_with_a_non_finite_number_is_malformed(audio_case):
    stage = audio_case()
    blob = ('{"%s": {"dtype": "F32", "shape": [NaN], "data_offsets": [0, 4]}}' % NAME_A).encode()
    blob += b" " * (-len(blob) % 8)
    (model_snapshot(stage) / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 4)
    assert run_ok(stage)["class"] == "unknown"


# ---- boundaries the first mutation table found unguarded --------------------------------------------------

@pytest.mark.parametrize("digits,minus,ok", [(30, "", True), (31, "", False), (30, "-", True), (31, "-", False)])
def test_an_integer_of_more_than_thirty_digits_is_refused_with_or_without_a_sign(tmp_path, digits, minus, ok):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    (snap / "model_config.json").write_text('{"model_type": "x", "model": {"a": ' + minus + "9" * digits + "}}")
    assert template_module().detect_audio(snap) is ok


@pytest.mark.parametrize("levels,ok", [(32, True), (33, False)])
def test_nesting_is_limited_to_thirty_two_containers(tmp_path, levels, ok):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    inner = levels - 2                               # the top object and "model" are the first two levels
    deep = "[" * inner + "1" + "]" * inner
    (snap / "model_config.json").write_text('{"model_type": "x", "model": {"a": ' + deep + "}}")
    assert template_module().detect_audio(snap) is ok


def test_a_header_that_claims_more_bytes_than_the_file_holds_is_malformed(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "model.safetensors").write_bytes(struct.pack("<Q", 100) + b"{}" + b" " * 8)
    assert "cut short" in head_text(stage)


def test_an_empty_range_inside_another_tensors_bytes_is_still_an_empty_range(audio_case):
    stage = audio_case()
    entries = {NAME_A: entry((0, 8), shape=(2,)), "x.empty": entry((2, 2), shape=(0,))}
    raw_file(model_snapshot(stage) / "model.safetensors", entries, 8)
    cfg = json.loads((stage / "triage_config.json").read_text())
    raw_file(Path(cfg["nearest_snapshot"]) / "model.safetensors", entries, 8)
    assert run_ok(stage)["class"] == "weights-only"


@pytest.mark.parametrize("text", ["[]", "3", '"x"', "null"])
def test_a_header_that_is_not_an_object_is_unknown_and_does_not_crash(audio_case, text):
    stage = audio_case()
    blob = text.encode()
    blob += b" " * (-len(blob) % 8)
    (model_snapshot(stage) / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "not a JSON object" in reasons(d)


# ---- round five: an audio candidate with an unreadable config stays audio; one descriptor per file ------------------

UNREADABLE_CONFIGS = ["not json", "[]", '{"model_type": "x", "model": {"a": 1, "a": 2}}', '{"model_type": "x"}', '{"model": {}}', "",
                      '{"model_type": "x", "model": {"a": NaN}}']


@pytest.mark.parametrize("text", UNREADABLE_CONFIGS)
def test_a_model_config_that_is_present_but_unreadable_is_still_an_audio_run_with_class_unknown(audio_case, text):
    stage = audio_case()
    (model_snapshot(stage) / "model_config.json").write_text(text)
    d = run_ok(stage)
    assert d["class"] == "unknown" and d["path"] == "full-port" and d["facts"] == {"form": "full", "modality": "audio"}
    assert "new model" in reasons(d) and "model_config.json" in reasons(d)
    assert gate(stage).ok, gate(stage).reasons


def test_an_oversized_model_config_is_still_an_audio_run_with_class_unknown(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "model_config.json").write_text(json.dumps({**CONFIG, "pad": "x" * (2 << 20)}))
    d = run_ok(stage)
    assert d["class"] == "unknown" and d["facts"]["modality"] == "audio"


def test_a_model_config_that_is_a_directory_or_a_dangling_link_is_still_audio(audio_case, tmp_path):
    import shutil
    stage = audio_case()
    cfg = model_snapshot(stage) / "model_config.json"
    cfg.unlink()
    cfg.mkdir()
    assert run_ok(stage)["facts"]["modality"] == "audio"
    shutil.rmtree(cfg)
    cfg.symlink_to(tmp_path / "nowhere.json")
    d = run_ok(stage)
    assert d["class"] == "unknown" and d["facts"]["modality"] == "audio"


def test_a_snapshot_with_a_model_config_and_no_weights_is_audio_with_class_unknown(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "model.safetensors").unlink()
    d = run_ok(stage)
    assert d["class"] == "unknown" and d["facts"]["modality"] == "audio"


def test_an_unreadable_nearest_model_config_is_unknown_and_names_the_nearest_model(audio_case):
    stage = audio_case()
    cfg = json.loads((stage / "triage_config.json").read_text())
    (Path(cfg["nearest_snapshot"]) / "model_config.json").write_text("not json")
    d = run_ok(stage)
    assert d["class"] == "unknown" and "nearest model" in reasons(d) and d["facts"]["modality"] == "audio"


def test_a_snapshot_with_a_config_json_is_never_an_audio_candidate(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID, extra={"config.json": "{}"})
    (snap / "model_config.json").write_text("not json")
    assert template_module().looks_like_audio(snap) is False


@pytest.mark.parametrize("make,expected", [("none", False), ("file", True), ("dir", True)])
def test_looks_like_audio_needs_a_model_config_entry_and_no_config_json(tmp_path, make, expected):
    snap = tmp_path / "s"
    snap.mkdir()
    if make == "file":
        (snap / "model_config.json").write_text("not json")
    elif make == "dir":
        (snap / "model_config.json").mkdir()
    assert template_module().looks_like_audio(snap) is expected


def test_the_unknown_audio_result_stops_the_supervisor_run_because_it_records_the_facts(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "model_config.json").write_text("not json")
    run_ok(stage)
    d = json.loads((stage / "delta.json").read_text())
    from orchard.stages import delta_facts
    assert delta_facts(stage.parent.parent) == {"form": "full", "modality": "audio"} and d["class"] == "unknown"


# ---- one descriptor per file: header and size from the same open ---------------------------------------------

def test_a_header_and_its_size_come_from_one_open_file(tmp_path, monkeypatch):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    real_stat = Path.stat

    def stat_guard(self, *a, **k):
        if self.name.endswith((".safetensors", ".json")):
            raise AssertionError(f"{self.name} was looked at by name")
        return real_stat(self, *a, **k)
    monkeypatch.setattr(Path, "stat", stat_guard)
    assert mod.audio_header(snap / "model.safetensors")
    assert mod._audio_json(snap / "model_config.json")[1] is None


def test_a_config_is_read_with_one_open_and_no_second_look_at_the_name(tmp_path, monkeypatch):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_text by name")))
    assert mod._audio_json(snap / "model_config.json")[1] is None


def test_a_fifo_is_not_a_regular_file_and_is_not_opened_for_reading(tmp_path):
    import os
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    (snap / "model_config.json").unlink()
    os.mkfifo(snap / "model_config.json")
    data, err = template_module()._audio_json(snap / "model_config.json")
    assert data is None and "not a regular file" in err


def test_a_file_that_changes_after_it_was_read_is_reported(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod.audio_identity_reset()
    mod.audio_tensors(snap)
    mod._audio_json(snap / "model_config.json")
    assert mod.audio_identity_problems() == []
    write_safetensors(snap / "model.safetensors", TENSORS, 9.0)          # same size, new inode and time
    problems = mod.audio_identity_problems()
    assert problems and "model.safetensors" in problems[0] and "changed while" in problems[0]


def test_a_file_that_was_removed_after_it_was_read_is_reported(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod.audio_identity_reset()
    mod._audio_json(snap / "model_config.json")
    (snap / "model_config.json").unlink()
    assert "model_config.json" in mod.audio_identity_problems()[0]


def test_a_new_model_with_a_bad_layout_is_reported_once_whatever_its_weights_look_like(audio_case):
    """The layout problem is the finding. The weights of a directory already refused as a layout are not read, so they add no second reason."""
    stage = audio_case()
    snap = model_snapshot(stage)
    (snap / "model_config.json").write_text("{}")
    write_header_only(snap / "model.safetensors", {})
    d = run_ok(stage)
    text = reasons(d)
    assert d["class"] == "unknown" and "not a usable stable-audio-tools layout" in text and "holds no tensors" not in text
    (snap / "model.safetensors").write_bytes(b"\xff" * 16)
    text = reasons(run_ok(stage))
    assert "not a usable stable-audio-tools layout" in text and "new model model.safetensors" not in text


def test_a_new_model_with_a_bad_layout_and_no_weights_file_is_reported_once(audio_case):
    stage = audio_case()
    snap = model_snapshot(stage)
    (snap / "model_config.json").write_text("{}")
    (snap / "model.safetensors").unlink()
    text = reasons(run_ok(stage))
    assert "not a usable stable-audio-tools layout" in text and "holds no tensors" not in text


def test_a_dangling_config_json_link_keeps_a_snapshot_out_of_the_audio_branch(tmp_path):
    """A top-level config.json that exists as a name (here a dangling link) marks a text model; looks_like_audio must not use a following check."""
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    (snap / "config.json").symlink_to(tmp_path / "nowhere")
    assert template_module().looks_like_audio(snap) is False


# ---- the identity check and the descriptor's own size, each seen to matter ---------------------------------

def _lying_fstat(monkeypatch, mod, extra):
    """os.fstat as the template sees it, reporting `extra` more bytes than the file has (a file that grew after the open)."""
    import os
    from types import SimpleNamespace
    real = os.fstat

    def fake(fd):
        r = real(fd)
        return SimpleNamespace(st_mode=r.st_mode, st_size=r.st_size + extra, st_dev=r.st_dev, st_ino=r.st_ino, st_mtime_ns=r.st_mtime_ns)
    monkeypatch.setattr(mod.os, "fstat", fake)


def test_resetting_the_identity_record_forgets_earlier_files(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod._audio_json(snap / "model_config.json")
    assert mod.AUDIO_IDENTITY
    mod.audio_identity_reset()
    assert mod.AUDIO_IDENTITY == {}


def test_a_weight_file_replaced_during_the_value_sample_makes_the_real_run_unknown(audio_case, tmp_path):
    """End to end: the stage script runs, and the weight file is swapped (same bytes, new inode) while the value sample reads it."""
    stage = audio_case()
    model = model_snapshot(stage) / "model.safetensors"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('dt', {str(stage / 'delta_triage.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "orig = mod.sample_value_diffs\n"
        "def patched(*a, **k):\n"
        "    out = orig(*a, **k)\n"
        f"    p = {str(model)!r}\n"
        "    data = open(p, 'rb').read()\n"
        "    open(p + '.new', 'wb').write(data)\n"
        "    os.replace(p + '.new', p)\n"
        "    return out\n"
        "mod.sample_value_diffs = patched\n"
        "sys.exit(mod.main())\n")
    r = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((stage / "delta.json").read_text())
    assert d["class"] == "unknown" and "model.safetensors changed while stage 0 was reading it" in reasons(d)


def test_a_config_that_grows_past_the_cap_after_it_was_opened_is_refused(tmp_path, monkeypatch):
    mod = template_module()
    path = tmp_path / "model_config.json"
    path.write_text('{"a": "' + "x" * (mod.AUDIO_JSON_MAX + 10) + '"}')
    _lying_fstat(monkeypatch, mod, -(mod.AUDIO_JSON_MAX + 100))          # the size taken at the open said it fits
    data, err = mod._audio_json(path)
    assert data is None and "larger than" in err


def test_the_data_section_size_is_the_size_of_the_opened_file(tmp_path, monkeypatch):
    mod = template_module()
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    assert mod.audio_header(snap / "model.safetensors")
    _lying_fstat(monkeypatch, mod, 16)
    with pytest.raises(ValueError, match="no tensor claims"):
        mod.audio_header(snap / "model.safetensors")


# ---- review of PR 142: the first identity of a file is kept across later opens -------------------------

def _replace_same_bytes(path):
    import os
    data = path.read_bytes()
    tmp = path.with_name(path.name + ".new")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def test_a_config_replaced_between_two_opens_is_reported_even_when_the_second_open_is_the_last(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod.audio_identity_reset()
    assert mod._audio_json(snap / "model_config.json")[1] is None
    _replace_same_bytes(snap / "model_config.json")            # same bytes, new inode
    mod._audio_json(snap / "model_config.json")
    problems = mod.audio_identity_problems()
    assert problems and "model_config.json changed while stage 0 was reading it" in problems[0]


def test_a_config_with_new_content_at_the_second_open_is_refused_by_hash(tmp_path):
    import os
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod.audio_identity_reset()
    path = snap / "model_config.json"
    first = path.read_bytes()
    assert mod._audio_json(path)[1] is None
    st = os.stat(path)
    i = first.index(b'"') + 1
    path.write_bytes(first[:i] + (b"Z" if first[i:i + 1] != b"Z" else b"Y") + first[i + 1:])        # one byte changed in place: same size
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))          # even with the old time and size kept in step, the hash differs
    data, err = mod._audio_json(path)
    assert data is None and "changed between" in err


def test_a_config_that_is_the_same_file_at_both_opens_is_fine(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod.audio_identity_reset()
    assert mod._audio_json(snap / "model_config.json")[1] is None
    assert mod._audio_json(snap / "model_config.json")[1] is None
    assert mod.audio_identity_problems() == []


def test_a_config_replaced_after_the_layout_check_makes_the_real_run_unknown(audio_case, tmp_path):
    """End to end: audio_main reads the config twice (layout check, then comparison). The swap happens between them."""
    stage = audio_case()
    cfgpath = model_snapshot(stage) / "model_config.json"
    driver = tmp_path / "driver2.py"
    driver.write_text(
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('dt', {str(stage / 'delta_triage.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "orig = mod.audio_layout_problem\n"
        "state = []\n"
        "def patched(snapshot):\n"
        "    out = orig(snapshot)\n"
        f"    if str(snapshot) == {str(model_snapshot(stage))!r} and not state:\n"
        "        state.append(1)\n"
        f"        p = {str(cfgpath)!r}\n"
        "        data = open(p, 'rb').read()\n"
        "        open(p + '.new', 'wb').write(data)\n"
        "        os.replace(p + '.new', p)\n"
        "    return out\n"
        "mod.audio_layout_problem = patched\n"
        "sys.exit(mod.main())\n")
    r = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((stage / "delta.json").read_text())
    assert d["class"] == "unknown" and "model_config.json changed while stage 0 was reading it" in reasons(d)


def test_resetting_forgets_the_first_hashes_and_the_noted_mismatches(tmp_path):
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod.audio_identity_reset()
    mod._audio_json(snap / "model_config.json")
    _replace_same_bytes(snap / "model_config.json")
    mod._audio_json(snap / "model_config.json")
    assert mod.AUDIO_JSON_SHA and mod.AUDIO_REOPENED_DIFFERENT
    mod.audio_identity_reset()
    assert mod.AUDIO_JSON_SHA == {} and mod.AUDIO_REOPENED_DIFFERENT == set() and mod.audio_identity_problems() == []


def test_the_hash_covers_the_whole_file_and_a_later_read_still_compares_with_the_first(tmp_path):
    import os
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod.audio_identity_reset()
    path = snap / "model_config.json"
    first = path.read_bytes()
    assert mod._audio_json(path)[1] is None
    st = os.stat(path)
    i = first.rindex(b'"') - 1                          # the last character of the last string, far from the start
    path.write_bytes(first[:i] + (b"Z" if first[i:i + 1] != b"Z" else b"Y") + first[i + 1:])
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert "changed between" in mod._audio_json(path)[1]
    assert "changed between" in mod._audio_json(path)[1]          # a third read is still compared with the first, not with the second


def test_a_file_swapped_in_for_one_open_and_swapped_back_is_still_reported(tmp_path):
    import os
    snap = make_audio(tmp_path, MODEL_REV, repo=MODEL_ID)
    mod = template_module()
    mod.audio_identity_reset()
    path = snap / "model_config.json"
    assert mod._audio_json(path)[1] is None
    original = path.with_name("original-config")
    os.rename(path, original)
    path.write_bytes(original.read_bytes())               # a different file with the same bytes
    mod._audio_json(path)
    path.unlink()
    os.rename(original, path)                             # the first file is back: the final stat alone would find nothing
    problems = mod.audio_identity_problems()
    assert problems and "different file at a later open" in problems[0]
