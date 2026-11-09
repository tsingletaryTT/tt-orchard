# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/tiny_qwen4/make_franken.py on a small synthetic Qwen3.5 base (no real weights, CPU only).

The base is a random 8-layer Qwen3.5 text model with the Qwen3.8 file layout (`model.language_model.*`, a nested `text_config`, a
visual tensor and an MTP tensor that the franken must leave out). The tests pin the claims of the script's docstring: with the
hyper-connection weights at zero the franken computes the base's logits (to float32 rounding; bf16 rounds the stored norm weight), a
seeded perturbation moves them, a short prompt gives the same output whatever the indexer weights are and a small budget does not,
the real indexer tensors are mapped by name, and every refusal fires before anything is written. They need transformers with
`qwen4_exp` (the reference interpreter) and skip otherwise."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import os

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
if importlib.util.find_spec("transformers.models.qwen4_exp") is None:
    pytest.skip("SKIPPED: this transformers has no qwen4_exp", allow_module_level=True)

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "tiny_qwen4" / "make_franken.py"
spec = importlib.util.spec_from_file_location("make_franken_under_test", SCRIPT)
mf = importlib.util.module_from_spec(spec)
sys.modules["make_franken_under_test"] = mf
spec.loader.exec_module(mf)

LAYERS = 8
H = 64
TEXT = dict(
    model_type="qwen3_5_text", hidden_size=H, num_attention_heads=4, num_key_value_heads=2, head_dim=32, intermediate_size=128,
    vocab_size=300, num_hidden_layers=LAYERS, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
    linear_value_head_dim=32, linear_conv_kernel_dim=4, full_attention_interval=4, rms_norm_eps=1e-6, max_position_embeddings=512,
    tie_word_embeddings=False, output_gate_type="swish", bos_token_id=1, eos_token_id=2,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [1, 1, 2], "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                     "rope_type": "default"})


@pytest.fixture(scope="module")
def base_snapshot(tmp_path_factory):
    """A random Qwen3.5 text model saved the way the Qwen3.8 snapshot is: prefixed names, a nested config, an index file."""
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    keep = {k: v for k, v in TEXT.items() if k != "model_type"}
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**keep)).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.normal_(0, 0.2 if ("norm" in name and "linear_attn.norm" not in name) else 0.05)
    tensors = {}
    for name, value in model.state_dict().items():
        tensors[name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name] = value.clone().bfloat16()
    tensors["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 4)
    tensors["mtp.fc.weight"] = torch.zeros(4, 4)
    root = tmp_path_factory.mktemp("base") / "models--Qwen--Qwen3.8-27B" / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    save_file(tensors, str(root / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": TEXT}))
    (root / "tokenizer.json").write_text("{}")
    return root


def _reference_python():
    """An interpreter with qwen4_exp, pytest not required, and a torch that accepts the indexer's int32 scatter index."""
    candidates = [os.environ.get("ORCHARD_REFERENCE_PYTHON"), os.path.expanduser("~/orchard-venvs/reference/bin/python"), sys.executable]
    code = ("import torch, transformers.models.qwen4_exp; "
            "torch.zeros(2, 3, dtype=torch.bool).scatter(-1, torch.zeros(2, 1, dtype=torch.int32), True)")
    for c in candidates:
        if c and os.path.exists(c) and subprocess.run([c, "-c", code], capture_output=True).returncode == 0:
            return c
    return None


PY = _reference_python()
needs_real_forward = pytest.mark.skipif(PY is None, reason=(
    "SKIPPED: no interpreter with qwen4_exp and a torch that accepts an int32 scatter index (set ORCHARD_REFERENCE_PYTHON), "
    "so the franken proofs were not run"))
LAST = {"err": ""}


def run(base, out, *extra, layers=4, dtype="float32", decode=False):
    """The script. A forward pass (the proofs, the reference decode) runs in the reference interpreter (torch 2.14), because the indexer's
    int32 scatter needs a torch newer than the test runner's; generation alone runs in the current interpreter with the decode skipped."""
    forward = "--prove" in extra or decode
    interpreter = PY if forward else sys.executable
    extra = extra if forward else (*extra, "--skip-decode-check")
    r = subprocess.run([interpreter, str(SCRIPT), "--out", str(out), "--base-snapshot", str(base), "--layers", str(layers), "--dtype", dtype,
                        "--max-gb", "1", *extra], capture_output=True, text=True)
    LAST["err"] = r.stderr
    return r.returncode


def proofs_of(out):
    return json.loads((out / "proofs.json").read_text())


# ---- the equality with the base ---------------------------------------------------------------------------------------------------

@needs_real_forward
def test_identity_hyper_connections_reproduce_the_base_logits_in_float32(base_snapshot, tmp_path):
    out = tmp_path / "f32"
    assert run(base_snapshot, out, "--prove") == 0
    p = proofs_of(out)
    assert p["identity_holds"] is True and p["against_base"]["max_abs_diff"] <= 5e-5
    assert p["against_base"]["top1_equal_all"] is True and p["deterministic"] is True


@needs_real_forward
def test_the_same_holds_for_eight_layers(base_snapshot, tmp_path):
    out = tmp_path / "f32k8"
    assert run(base_snapshot, out, "--prove", layers=8) == 0
    assert proofs_of(out)["against_base"]["max_abs_diff"] <= 5e-5


@needs_real_forward
def test_bf16_is_close_and_says_how_close(base_snapshot, tmp_path):
    out = tmp_path / "bf16"
    assert run(base_snapshot, out, "--prove", dtype="bfloat16") == 0
    p = proofs_of(out)["against_base"]
    assert p["max_abs_diff"] <= p["tolerance"] and p["prompts"][0]["max_abs_logit"] > 0 and len(p["prompts"]) == 3


def test_a_wrong_output_gate_activation_would_break_the_equality(base_snapshot):
    """The delta net's gated norm reads output_gate_type; sigmoid is the class default and is not the base's activation."""
    text = mf.read_base_text_config(base_snapshot)
    assert mf.franken_config(text, 4, dtype="float32")["output_gate_type"] == "silu"


def test_hc_norm_weight_is_one_plus_twice_the_base_norm_weight_for_every_stream(base_snapshot, tmp_path):
    from safetensors import safe_open
    out = tmp_path / "norms"
    assert run(base_snapshot, out) == 0
    with safe_open(str(base_snapshot / "model-00001-of-00001.safetensors"), framework="pt") as base, \
            safe_open(str(out / "model.safetensors"), framework="pt") as made:
        w = base.get_tensor("model.language_model.layers.2.input_layernorm.weight").float()
        got = made.get_tensor("model.layers.2.attn_hyper_connection.hc_norm.weight").float()
        assert torch.allclose(got, (1 + 2 * w).repeat(4))
        last = base.get_tensor("model.language_model.norm.weight").float()
        assert torch.allclose(made.get_tensor("model.hyper_connection_mixer.hc_norm.weight").float(), (1 + 2 * last).repeat(4))
        assert float(made.get_tensor("model.layers.0.mlp_hyper_connection.block_inject_weight.weight").abs().max()) == 0.0


# ---- the perturbation and the indexer ---------------------------------------------------------------------------------------------

@needs_real_forward
def test_a_seeded_perturbation_moves_the_logits_and_is_deterministic(base_snapshot, tmp_path):
    out = tmp_path / "pert"
    assert run(base_snapshot, out, "--prove", "--hc-sigma", "0.05") == 0
    p = proofs_of(out)
    assert p["perturbed_differs"] is True and p["deterministic"] is True and p["against_base"]["max_abs_diff"] > 5e-4
    again = tmp_path / "pert2"
    assert run(base_snapshot, again, "--hc-sigma", "0.05") == 0
    from safetensors import safe_open
    key = "model.layers.1.attn_hyper_connection.input_mix_weight_up.weight"
    with safe_open(str(out / "model.safetensors"), framework="pt") as a, safe_open(str(again / "model.safetensors"), framework="pt") as b:
        assert torch.equal(a.get_tensor(key), b.get_tensor(key)) and float(a.get_tensor(key).abs().max()) > 0


@needs_real_forward
def test_a_short_prompt_ignores_the_indexer_and_a_small_budget_does_not(base_snapshot, tmp_path):
    out = tmp_path / "idx"
    assert run(base_snapshot, out, "--prove", "--set", "indexer_budget=64") == 0
    idx = proofs_of(out)["indexer"]
    assert idx["all_selected_equals_base"] is True and idx["narrow_differs"] is True and idx["narrow_vs_all_selected_max_abs_diff"] > 5e-5


def test_the_real_indexer_tensors_are_mapped_by_name(base_snapshot, tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.manual_seed(3)
    rel = tmp_path / "models--higa2515--Qwen3.8-27B-QSA-BF16" / "snapshots" / ("b" * 40)
    rel.mkdir(parents=True)
    tensors = {}
    for layer in (3, 7):
        stem = f"model.language_model.layers.{layer}.self_attn._qsa_indexer."
        tensors[stem + "q_proj.weight"] = torch.randn(4 * 32, H)
        tensors[stem + "k_proj.weight"] = torch.randn(32, H)
        tensors[stem + "q_norm"] = torch.randn(32)
        tensors[stem + "k_norm"] = torch.randn(32)
        tensors[stem + "k_block_norm"] = torch.randn(32)
    save_file(tensors, str(rel / "qsa-indexers.safetensors"))
    out = tmp_path / "real"
    assert run(base_snapshot, out, "--indexers", str(rel), layers=8) == 0
    stem = "model.language_model.layers.7.self_attn._qsa_indexer."
    with safe_open(str(out / "model.safetensors"), framework="pt") as made:
        got = made.get_tensor("model.layers.7.self_attn.indexer.index_qk_proj.weight")
        want = torch.cat([tensors[stem + "q_proj.weight"], tensors[stem + "k_proj.weight"]], 0)
        assert torch.allclose(got, want)
        assert torch.allclose(made.get_tensor("model.layers.7.self_attn.indexer.k_layernorm.weight"), tensors[stem + "k_block_norm"])
        assert torch.allclose(made.get_tensor("model.layers.7.self_attn.indexer.q_layernorm.weight"), tensors[stem + "q_norm"])
    manifest = json.loads((out / "franken-manifest.json").read_text())
    assert manifest["tensor_counts"]["indexer_real"] == 6
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "from `higa2515/Qwen3.8-27B-QSA-BF16`, revision `" + "b" * 40 + "`" in card
    assert "that release's terms also apply to them" in card


def test_an_indexer_file_with_the_wrong_shape_is_refused(base_snapshot, tmp_path, capsys):
    from safetensors.torch import save_file
    rel = tmp_path / "rel" / "snapshots" / ("c" * 40)
    rel.mkdir(parents=True)
    t = {}
    for layer in (3,):
        stem = f"model.language_model.layers.{layer}.self_attn._qsa_indexer."
        t.update({stem + "q_proj.weight": torch.zeros(10, H), stem + "k_proj.weight": torch.zeros(32, H),
                  stem + "q_norm": torch.zeros(32), stem + "k_norm": torch.zeros(32), stem + "k_block_norm": torch.zeros(32)})
    save_file(t, str(rel / "qsa-indexers.safetensors"))
    out = tmp_path / "bad"
    assert run(base_snapshot, out, "--indexers", str(rel), layers=4) == 2
    assert not out.exists() or not (out / "model.safetensors").exists()
    assert "do not fit" in LAST["err"]


# ---- the manifest -----------------------------------------------------------------------------------------------------------------

def test_the_manifest_lists_every_file_with_its_hash_and_says_what_is_real(base_snapshot, tmp_path):
    import hashlib
    out = tmp_path / "man"
    assert run(base_snapshot, out) == 0
    m = json.loads((out / "franken-manifest.json").read_text())
    assert set(m["files"]) >= {"config.json", "model.safetensors", "generation_config.json", "tokenizer.json"}
    for name, rec in m["files"].items():
        assert rec["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert "function equals the base" in m["identity_initialised"] and "visual tower" in " ".join(m["not_carried"])
    assert m["layers"] == 4 and m["sources"]["base_snapshot"] == "a" * 40 and m["tensor_counts"]["real"] > 0


def test_the_checkpoint_loads_with_no_missing_or_unexpected_key(base_snapshot, tmp_path):
    out = tmp_path / "load"
    assert run(base_snapshot, out) == 0
    qd = mf.helpers()[1]
    model = mf.load_franken(torch, transformers, qd, out)
    assert sum(p.numel() for p in model.parameters()) > 0
    meta = mf.build_meta_model(torch, transformers, qd, json.loads((out / "config.json").read_text()) | {"model_type": "qwen4_exp_text"})
    assert set(meta.state_dict()) == set(model.state_dict())


def test_check_keys_refuses_a_tensor_the_class_does_not_have_and_one_it_lacks():
    with pytest.raises(mf.FrankenRefused, match="missing"):
        mf.check_keys(["a", "b"], ["a"])
    with pytest.raises(mf.FrankenRefused, match="unexpected"):
        mf.check_keys(["a"], ["a", "c"])


# ---- refusals, each before anything is written -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("extra,layers,why", [
    ([], 6, "multiple of 4"), ([], 12, "more than the base"), ([], 0, "multiple of 4"),
    (["--dtype", "float16"], 4, "dtype"), (["--hc-sigma", "-1"], 4, "hc-sigma"), (["--hc-sigma", "2"], 4, "hc-sigma"),
    (["--set", "bogus=3"], 4, "only indexer_budget"), (["--set", "indexer_budget=abc"], 4, "whole number"),
    (["--set", "indexer_budget=6"], 4, "multiple of 4"), (["--set", "indexer_budget=0"], 4, "between 1"),
    (["--max-gb", "0.0000001"], 4, "estimated"),
])
def test_bad_requests_are_refused_and_write_nothing(base_snapshot, tmp_path, capsys, extra, layers, why):
    out = tmp_path / "refused"
    code = mf.main(["--out", str(out), "--base-snapshot", str(base_snapshot), "--layers", str(layers), "--max-gb", "1", *extra])
    err = capsys.readouterr().err
    assert code == 2 and not out.exists() and why in err


def test_an_output_directory_inside_the_repository_is_refused(base_snapshot, capsys):
    inside = REPO / "franken-should-not-exist"
    assert mf.main(["--out", str(inside), "--base-snapshot", str(base_snapshot), "--layers", "4"]) == 2
    assert not inside.exists()


def test_a_base_with_a_different_layer_schedule_is_refused(tmp_path, base_snapshot, capsys):
    snap = tmp_path / "odd" / "snapshots" / ("d" * 40)
    snap.mkdir(parents=True)
    text = dict(TEXT, layer_types=["full_attention"] * LAYERS)
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": text}))
    assert mf.main(["--out", str(tmp_path / "o"), "--base-snapshot", str(snap), "--layers", "4"]) == 2
    assert "3 linear-attention" in capsys.readouterr().err


def test_a_base_that_is_not_a_qwen3_5_text_config_is_refused(tmp_path, capsys):
    snap = tmp_path / "x" / "snapshots" / ("e" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert mf.main(["--out", str(tmp_path / "o"), "--base-snapshot", str(snap)]) == 2
    assert "qwen3_5" in capsys.readouterr().err


def test_a_snapshot_that_is_missing_is_refused(tmp_path, capsys):
    assert mf.main(["--out", str(tmp_path / "o"), "--base-snapshot", str(tmp_path / "nope")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_the_default_snapshot_is_the_one_that_holds_weights(tmp_path, monkeypatch):
    root = tmp_path / "hub" / mf.BASE_REPO_DIR / "snapshots"
    full, empty = root / ("1" * 40), root / ("2" * 40)
    full.mkdir(parents=True)
    empty.mkdir()
    (full / "model.safetensors.index.json").write_text("{}")
    (full / "model-00001-of-00001.safetensors").write_bytes(b"x")
    (empty / "config.json").write_text("{}")
    monkeypatch.setattr(mf, "HF_HUB", tmp_path / "hub")
    assert mf.default_snapshot() == full
    other = root / ("3" * 40)
    other.mkdir()
    (other / "model.safetensors.index.json").write_text("{}")
    (other / "model-00001-of-00001.safetensors").write_bytes(b"x")
    with pytest.raises(mf.FrankenRefused, match="exactly one snapshot"):
        mf.default_snapshot()


def _linked_copy(base_snapshot, tmp_path):
    """The same base laid out like a Hugging Face cache: every snapshot file is a link into ../../blobs/."""
    import os
    import shutil
    model_dir = tmp_path / "hub" / "models--Qwen--Qwen3.8-27B"
    snap = model_dir / "snapshots" / ("9" * 40)
    blobs = model_dir / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    for f in base_snapshot.iterdir():
        shutil.copy(f, blobs / f.name)
        os.symlink(f"../../blobs/{f.name}", snap / f.name)
    return snap, blobs


def test_a_hub_cache_layout_with_every_file_linked_into_blobs_is_read(base_snapshot, tmp_path):
    snap, _ = _linked_copy(base_snapshot, tmp_path)
    out = tmp_path / "linked"
    assert run(snap, out) == 0 and (out / "model.safetensors").is_file() and (out / "tokenizer.json").is_file()


@pytest.mark.parametrize("victim", ["config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json"])
def test_a_snapshot_file_that_links_outside_the_cache_is_refused(base_snapshot, tmp_path, victim):
    import os
    snap, blobs = _linked_copy(base_snapshot, tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / victim).write_bytes((blobs / victim).read_bytes())
    (snap / victim).unlink()
    os.symlink(str(elsewhere / victim), snap / victim)
    out = tmp_path / "evil"
    assert run(snap, out) == 2 and not (out / "model.safetensors").exists()
    assert "links outside" in LAST["err"]


def test_a_missing_indexer_file_is_refused(base_snapshot, tmp_path):
    empty = tmp_path / "rel" / "snapshots" / ("f" * 40)
    empty.mkdir(parents=True)
    out = tmp_path / "noidx"
    assert run(base_snapshot, out, "--indexers", str(empty)) == 2 and "missing" in LAST["err"]


def test_a_two_hop_hub_cache_link_into_the_hub_wide_store_is_read(base_snapshot, tmp_path, monkeypatch):
    """Newer huggingface_hub: snapshot file -> ../../blobs/<hash> -> <hub>/blobs/<xx>/<hash>. Accepted only inside the hub cache."""
    import os
    hub = tmp_path / "hub"
    snap, blobs = _linked_copy(base_snapshot, tmp_path)
    store = hub / "blobs" / "ab"
    store.mkdir(parents=True)
    for f in list(blobs.iterdir()):
        (store / f.name).write_bytes(f.read_bytes())
        f.unlink()
        os.symlink(str(store / f.name), blobs / f.name)
    monkeypatch.setattr(mf, "HF_HUB", hub)
    assert mf.snap_file(snap, "config.json").samefile(store / "config.json")
    # PR 644 made the same layout acceptable in safeio itself (the snapshot sits at <hub>/models--*/snapshots/<rev> and <hub>/blobs is a real
    # directory), so pointing HF_HUB somewhere else no longer changes the answer. The hostile cases (a linked hub store, a target outside, a
    # pipe) are covered by tests/test_hub_wide_blob_store.py.
    monkeypatch.setattr(mf, "HF_HUB", tmp_path / "some-other-hub")
    assert mf.snap_file(snap, "config.json").samefile(store / "config.json")


# ---- review of PR 639 -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,want", [("bfloat16", "bfloat16"), ("float32", "float32")])
def test_the_proof_loader_keeps_the_checkpoints_dtype(base_snapshot, tmp_path, dtype, want):
    """transformers turns config.dtype into a torch dtype, and the loader once treated that as 'not a string' and used float32, so the
    bf16 proofs ran the franken in float32 against a bf16 base."""
    out = tmp_path / dtype
    assert run(base_snapshot, out, dtype=dtype) == 0
    qd = mf.helpers()[1]
    model = mf.load_franken(torch, transformers, qd, out)
    assert {p.dtype for p in model.parameters()} == {getattr(torch, want)}


def test_the_proof_record_names_the_dtypes_both_models_ran_in():
    rec = mf.execution_dtypes(torch, type("M", (), {"parameters": lambda self: iter([torch.zeros(1, dtype=torch.bfloat16)])})(),
                              type("M", (), {"parameters": lambda self: iter([torch.zeros(1, dtype=torch.float32)])})())
    assert rec == {"franken": "torch.bfloat16", "base": "torch.float32"}


def test_a_proof_that_was_not_run_or_that_failed_is_not_a_pass():
    ok = {"deterministic": True, "identity_holds": True, "indexer": {"narrow_differs": True, "all_selected_equals_base": True}}
    assert mf.proofs_pass(ok) is True
    assert mf.proofs_pass({**ok, "indexer": {}}) is False                                  # an omitted indexer proof is a failure
    assert mf.proofs_pass({**ok, "indexer": {"narrow_differs": False}}) is False
    assert mf.proofs_pass({**ok, "indexer": {"all_selected_equals_base": True}}) is False   # the narrow-budget comparison is missing
    assert mf.proofs_pass({**ok, "indexer": {"all_selected_equals_base": True, "narrow_differs": False}}) is False
    assert mf.proofs_pass({"deterministic": True, "perturbed_differs": True, "indexer": {}}) is False
    assert mf.proofs_pass({**ok, "deterministic": False}) is False
    assert mf.proofs_pass({"deterministic": True, "perturbed_differs": True, "indexer": ok["indexer"]}) is True
    assert mf.proofs_pass({"deterministic": True, "perturbed_differs": False, "indexer": ok["indexer"]}) is False
    assert mf.proofs_pass({"deterministic": True, "indexer": ok["indexer"]}) is False       # neither the identity nor the perturbation claim was made


@needs_real_forward
def test_a_perturbed_run_does_not_claim_the_all_selected_model_equals_the_base(base_snapshot, tmp_path):
    out = tmp_path / "pertflag"
    assert run(base_snapshot, out, "--prove", "--hc-sigma", "0.05") == 0
    idx = proofs_of(out)["indexer"]
    assert idx["all_selected_equals_base"] is False and idx["narrow_differs"] is True


@needs_real_forward
def test_a_small_budget_still_gets_an_indexer_proof_and_the_identity_is_judged_with_every_block_selected(base_snapshot, tmp_path):
    """With indexer_budget=8 the configured model is sparse and differs from the base on purpose. The identity claim and the perturbation
    claim are judged on the model with every block selected, and the wide/narrow comparison always runs."""
    out = tmp_path / "small"
    assert run(base_snapshot, out, "--prove", "--set", "indexer_budget=8") == 0
    p = proofs_of(out)
    assert p["identity_holds"] is True and p["indexer"]["narrow_differs"] is True and p["indexer"]["all_selected_equals_base"] is True
    assert p["against_base"]["max_abs_diff"] > 5e-5            # the configured sparse model really differs, and that is reported
    out2 = tmp_path / "small-pert"
    assert run(base_snapshot, out2, "--prove", "--set", "indexer_budget=8", "--hc-sigma", "0.001") == 0
    assert proofs_of(out2)["perturbed_differs"] in (True, False)       # judged on the all-selected model, not on the sparse attention


@pytest.mark.parametrize("edit,why", [
    ({"hidden_size": "5120"}, "hidden_size"), ({"linear_key_head_dim": -1}, "linear_key_head_dim"),
    ({"linear_conv_kernel_dim": 0}, "linear_conv_kernel_dim"), ({"num_key_value_heads": True}, "num_key_value_heads"),
    ({"rope_parameters": [1, 2]}, "rope_parameters"), ({"linear_num_value_heads": None}, "linear_num_value_heads"),
])
def test_a_malformed_base_config_is_refused_not_a_traceback(tmp_path, capsys, edit, why):
    snap = tmp_path / "bad" / "snapshots" / ("a1" * 20)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {**TEXT, **edit}}))
    assert mf.main(["--out", str(tmp_path / "o"), "--base-snapshot", str(snap), "--layers", "4"]) == 2
    assert why in capsys.readouterr().err and not (tmp_path / "o").exists()


@pytest.mark.parametrize("payload", ["not json", "[1, 2]", '{"weight_map": [1]}', '{"weight_map": {"a": 3}}', '{"other": {}}', "{}"])
def test_a_bad_weight_index_is_refused_not_a_traceback(base_snapshot, tmp_path, capsys, payload):
    import shutil
    snap = tmp_path / "idx" / "snapshots" / ("b1" * 20)
    shutil.copytree(base_snapshot, snap)
    (snap / "model.safetensors.index.json").write_text(payload)
    assert mf.main(["--out", str(tmp_path / "o"), "--base-snapshot", str(snap), "--layers", "4", "--max-gb", "1"]) == 2
    assert "index" in capsys.readouterr().err and not (tmp_path / "o").exists()


@pytest.mark.parametrize("sigma", ["nan", "inf", "-inf", "1.5"])
def test_a_hc_sigma_that_is_not_in_zero_to_one_is_refused(base_snapshot, tmp_path, capsys, sigma):
    code = mf.main(["--out", str(tmp_path / "o"), "--base-snapshot", str(base_snapshot), "--layers", "4", "--max-gb", "1", f"--hc-sigma={sigma}"])
    assert code == 2 and "hc-sigma" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_a_refused_tokenizer_file_leaves_no_output_directory(base_snapshot, tmp_path):
    import os
    snap, blobs = _linked_copy(base_snapshot, tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "tokenizer.json").write_text("{}")
    (snap / "tokenizer.json").unlink()
    os.symlink(str(elsewhere / "tokenizer.json"), snap / "tokenizer.json")
    out = tmp_path / "tokrefuse"
    assert run(snap, out) == 2 and not out.exists() and "links outside" in LAST["err"]


def test_a_needed_tensor_with_a_non_string_shard_name_is_refused_not_a_traceback(base_snapshot, tmp_path, capsys):
    import shutil
    snap = tmp_path / "idx2" / "snapshots" / ("c1" * 20)
    shutil.copytree(base_snapshot, snap)
    table = json.loads((snap / "model.safetensors.index.json").read_text())
    table["weight_map"]["model.language_model.embed_tokens.weight"] = 3
    (snap / "model.safetensors.index.json").write_text(json.dumps(table))
    assert mf.main(["--out", str(tmp_path / "o"), "--base-snapshot", str(snap), "--layers", "4", "--max-gb", "1"]) == 2
    assert "weight index" in capsys.readouterr().err and not (tmp_path / "o").exists()


# ---- the degenerate-reference guard (coordinator finding on the first hardware attempt) --------------------------------------------

def _decode_record(out):
    return json.loads((out / "franken-manifest.json").read_text())["reference_decode"]


@needs_real_forward
def test_the_manifest_records_a_stage_1_style_greedy_decode(base_snapshot, tmp_path):
    out = tmp_path / "dec"
    code = run(base_snapshot, out, "--allow-degenerate", decode=True)
    assert code == 0
    rec = _decode_record(out)
    assert len(rec["generated_ids"]) == 32 and rec["distinct_ids"] == len(set(rec["generated_ids"]))
    assert rec["prompt_source"].startswith("seeded ids") and rec["degenerate"] == (rec["distinct_ids"] < 4)


def _constant_output_base(base_snapshot, tmp_path):
    """The base with an lm_head whose largest row wins whatever the input: every greedy token is the same."""
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    snap = tmp_path / "const" / "snapshots" / ("d1" * 20)
    shutil.copytree(base_snapshot, snap)
    shard = snap / "model-00001-of-00001.safetensors"
    with safe_open(str(shard), framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    head = tensors["lm_head.weight"].float()
    head[:] = 0
    head[7] = 1.0
    tensors["lm_head.weight"] = head.bfloat16()
    tensors["model.language_model.embed_tokens.weight"] = torch.ones_like(tensors["model.language_model.embed_tokens.weight"])
    save_file(tensors, str(shard), metadata={"format": "pt"})
    return snap


@needs_real_forward
def test_a_checkpoint_whose_reference_decode_is_constant_is_refused_with_exit_3(base_snapshot, tmp_path):
    snap = _constant_output_base(base_snapshot, tmp_path)
    out = tmp_path / "degenerate"
    assert run(snap, out, decode=True) == 3
    assert "distinct" in LAST["err"]
    rec = _decode_record(out)
    assert rec["degenerate"] is True and rec["distinct_ids"] < 4        # the checkpoint and its manifest stay, flagged


@needs_real_forward
def test_allow_degenerate_turns_the_refusal_into_a_recorded_warning(base_snapshot, tmp_path):
    snap = _constant_output_base(base_snapshot, tmp_path)
    out = tmp_path / "allowed"
    assert run(snap, out, "--allow-degenerate", decode=True) == 0
    assert _decode_record(out)["degenerate"] is True


def test_the_distinct_id_rule_is_four():
    assert mf.MIN_DISTINCT_IDS == 4
    assert mf.is_degenerate([5] * 32) and mf.is_degenerate([1, 2, 3] * 11) and not mf.is_degenerate([1, 2, 3, 4] * 8)


@needs_real_forward
def test_the_proofs_compare_the_decode_with_the_bases(base_snapshot, tmp_path):
    out = tmp_path / "decproof"
    assert run(base_snapshot, out, "--prove", "--allow-degenerate") == 0
    d = proofs_of(out)["decode"]
    assert d["franken_ids"] == d["base_ids"] and d["same_as_base"] is True and len(d["base_ids"]) == 32


def test_skipping_the_decode_check_is_recorded_not_silent(base_snapshot, tmp_path):
    out = tmp_path / "skipped"
    assert run(base_snapshot, out) == 0
    rec = _decode_record(out)
    assert rec["not_run"] is True and rec["reason"] == "--skip-decode-check"


# ---- teacher-forced reference: many positions whose answers vary even when the greedy decode does not ---------------------------------

def test_the_teacher_forced_passage_is_long_enough_and_fixed():
    assert len(mf.TEACHER_FORCED_TEXT.split()) >= 80 and mf.TEACHER_FORCED_MIN_POSITIONS >= 64


@needs_real_forward
def test_the_checkpoint_carries_a_teacher_forced_reference_and_the_proofs_compare_it_with_the_base(base_snapshot, tmp_path):
    out = tmp_path / "tf"
    assert run(base_snapshot, out, "--prove", "--allow-degenerate") == 0
    ref = json.loads((out / "reference" / "teacher-forced.json").read_text())
    n = len(ref["prompt_ids"])
    assert n >= mf.TEACHER_FORCED_MIN_POSITIONS and len(ref["top1_ids"]) == n and len(ref["top5_last"]) == mf.TEACHER_FORCED_TOP5_POSITIONS
    p = proofs_of(out)["teacher_forced"]
    assert p["positions"] == n and p["same_top1_as_base"] is True and p["agree_fraction"] == 1.0
    assert p["distinct_top1"] == len(set(ref["top1_ids"]))
    m = json.loads((out / "franken-manifest.json").read_text())
    assert m["teacher_forced"]["positions"] == n and "reference/teacher-forced.json" in m["files"]


def test_agreement_is_the_share_of_equal_positions_and_unequal_lengths_never_agree():
    assert mf.agreement([1, 2, 3, 4], [1, 2, 3, 4]) == (1.0, True)
    assert mf.agreement([1, 2, 3, 4], [1, 2, 9, 4]) == (0.75, False)
    assert mf.agreement([1, 2], [1, 2, 3]) == (0.0, False) and mf.agreement([], []) == (0.0, False)


def test_a_teacher_forced_passage_that_tokenizes_short_is_refused():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(10))}
    with pytest.raises(mf.FrankenRefused, match="fewer than"):
        mf.teacher_forced_prompt(None, None, 300, Tok())
    ids, source = mf.teacher_forced_prompt(None, None, 300, None)
    assert len(ids) >= mf.TEACHER_FORCED_MIN_POSITIONS and "seeded" in source


# ---- what the config says, and the model card (Track C, item C4) -----------------------------------------------------------------

# Keys the franken's config.json may carry that the Qwen4Exp text config does not declare: `intermediate_size` is the dense feed-forward
# width the dense shim reads, and `full_attention_interval` is a key of the Qwen3.8 config that the class reads when no layer_types is given.
EXTRA_CONFIG_KEYS = {"intermediate_size", "full_attention_interval"}


def _config_keys_the_class_declares() -> set:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    keys = set()
    for cls in Qwen4ExpTextConfig.__mro__:
        keys |= set(getattr(cls, "__annotations__", {}))
    return keys


@pytest.mark.parametrize("layers", [4, 8])
def test_the_config_lists_one_layer_type_per_layer_in_the_base_schedule(base_snapshot, tmp_path, layers):
    """A real Qwen4 config is expected to carry layer_types (the class maps its full_attention entries to indexed_attention), and stage 0
    compares layer_types as a deciding key, so the franken writes the base's own entries for the layers it keeps."""
    out = tmp_path / f"types{layers}"
    assert run(base_snapshot, out, layers=layers) == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["layer_types"] == (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)
    assert len(cfg["layer_types"]) == cfg["num_hidden_layers"] == layers
    m = json.loads((out / "franken-manifest.json").read_text())
    assert m["config"]["layer_types"] == cfg["layer_types"] == m["layer_types"]


def test_the_written_layer_types_load_as_the_classes_indexed_attention_layers(base_snapshot, tmp_path):
    out = tmp_path / "typesload"
    assert run(base_snapshot, out) == 0
    qd = mf.helpers()[1]
    config = qd.prepare_config(transformers, json.loads((out / "config.json").read_text()))
    assert config.layer_types == ["linear_attention"] * 3 + ["indexed_attention"]


def test_the_config_has_no_seed_key_and_the_manifest_keeps_the_seed(base_snapshot, tmp_path):
    """`seed` in a Qwen4Exp config is the n-gram hash seed, a different setting from the generator's seed. The generator's seed is
    recorded in the manifest."""
    out = tmp_path / "seed"
    assert run(base_snapshot, out, "--seed", "99", "--hc-sigma", "0.01") == 0
    cfg = json.loads((out / "config.json").read_text())
    assert "seed" not in cfg
    assert json.loads((out / "franken-manifest.json").read_text())["seed"] == 99


def test_every_config_key_is_one_the_class_declares_or_a_named_extra(base_snapshot, tmp_path):
    out = tmp_path / "keys"
    assert run(base_snapshot, out) == 0
    cfg = json.loads((out / "config.json").read_text())
    assert set(cfg) - _config_keys_the_class_declares() - EXTRA_CONFIG_KEYS == set()
    assert "seed" not in cfg


def _card(out):
    return (out / "README.md").read_text(encoding="utf-8")


def test_the_output_has_a_model_card_whose_license_stage_0_and_stage_7_read_as_apache(base_snapshot, tmp_path):
    from orchard.package_card import read_license
    out = tmp_path / "card"
    assert run(base_snapshot, out) == 0
    assert read_license(out) == "apache-2.0"
    triage = importlib.util.spec_from_file_location("delta_triage_for_card",
                                                    REPO / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py")
    dt = importlib.util.module_from_spec(triage)
    triage.loader.exec_module(dt)
    side = dt.license_side(out)
    assert side["license"] == "apache-2.0" and side["source"] == "model card front matter"


def test_the_card_names_the_base_revision_and_the_options_and_says_not_to_upload(base_snapshot, tmp_path):
    snap, _ = _linked_copy(base_snapshot, tmp_path)
    out = tmp_path / "cardopts"
    assert run(snap, out, "--seed", "77", "--hc-sigma", "0.02", "--set", "indexer_budget=64", layers=8, dtype="bfloat16") == 0
    card = _card(out)
    for want in ("9" * 40, "Qwen/Qwen3.8-27B", "first 8 layers", "bfloat16", "0.02", "77", "indexer_budget=64", "perturbed",
                 "not a Qwen4 release", "Do not upload", "tt-model bundles"):
        assert want in card, want


def test_an_identity_checkpoint_says_it_computes_the_bases_function(base_snapshot, tmp_path):
    out = tmp_path / "cardid"
    assert run(base_snapshot, out) == 0
    card = _card(out)
    assert "same function as the first 4 layers" in card and "perturbed" not in card


def test_the_card_holds_no_machine_path(base_snapshot, tmp_path):
    out = tmp_path / "cardpaths"
    assert run(base_snapshot, out) == 0
    card = _card(out)
    assert str(tmp_path) not in card and str(base_snapshot) not in card and "/home/" not in card and "/tmp" not in card


@pytest.mark.parametrize("where", [("someone",), ("models--someone--x", "elsewhere")])
def test_a_base_outside_the_cache_layout_is_named_only_by_its_revision(base_snapshot, tmp_path, where):
    """A plain directory gives no repository name to show, and the folder above it may be a user's name. A `models--` folder counts only
    with a `snapshots` folder between it and the revision."""
    import shutil
    plain = tmp_path.joinpath(*where) / ("c" * 40)
    shutil.copytree(base_snapshot, plain)
    out = tmp_path / "cardplain"
    assert run(plain, out) == 0
    card = _card(out)
    assert "c" * 40 in card and "someone" not in card and "not in the Hugging Face cache layout" in card
    assert card.count("Hugging Face cache layout") >= 1 and "the indexer tensors come from another release" not in card.lower()
    assert card.count("`") % 2 == 0


def test_a_base_directory_not_named_by_a_commit_gives_no_revision(base_snapshot, tmp_path):
    import shutil
    plain = tmp_path / "models--Qwen--Qwen3.8-27B" / "snapshots" / "main-copy"
    shutil.copytree(base_snapshot, plain)
    out = tmp_path / "cardnorev"
    assert run(plain, out) == 0
    card = _card(out)
    assert "main-copy" not in card and "the snapshot directory is not named by a commit" in card


def test_the_manifest_lists_the_card_with_its_hash(base_snapshot, tmp_path):
    import hashlib
    out = tmp_path / "cardman"
    assert run(base_snapshot, out) == 0
    m = json.loads((out / "franken-manifest.json").read_text())
    assert m["files"]["README.md"]["sha256"] == hashlib.sha256((out / "README.md").read_bytes()).hexdigest()


def _base_with_card(base_snapshot, tmp_path, text):
    import shutil
    snap = tmp_path / "carded" / "models--Qwen--Qwen3.8-27B" / "snapshots" / ("b" * 40)
    shutil.copytree(base_snapshot, snap)
    (snap / "README.md").write_text(text, encoding="utf-8")
    return snap


def test_a_base_card_that_names_apache_is_said_so(base_snapshot, tmp_path):
    snap = _base_with_card(base_snapshot, tmp_path, "---\nlicense: apache-2.0\n---\n# Qwen3.8-27B\n")
    out = tmp_path / "apachebase"
    assert run(snap, out) == 0
    assert "The base model's card names apache-2.0" in _card(out)


def test_a_base_without_a_card_license_is_said_so(base_snapshot, tmp_path):
    out = tmp_path / "nolicense"
    assert run(base_snapshot, out) == 0
    assert "names no license, so the license above was not checked against it" in _card(out)


@pytest.mark.parametrize("license_line", ["license: cc-by-nc-4.0", "license: other\nlicense_name: qwen-research"])
def test_a_base_whose_card_names_another_license_is_refused_and_writes_nothing(base_snapshot, tmp_path, license_line):
    snap = _base_with_card(base_snapshot, tmp_path, f"---\n{license_line}\n---\n# A base\n")
    out = tmp_path / "otherlicense"
    assert run(snap, out) == 2 and not out.exists() and "Apache-2.0" in LAST["err"]


def test_a_base_card_that_links_outside_the_cache_is_refused_and_writes_nothing(base_snapshot, tmp_path):
    import os
    snap, _ = _linked_copy(base_snapshot, tmp_path)
    elsewhere = tmp_path / "elsewhere-card"
    elsewhere.mkdir()
    (elsewhere / "README.md").write_text("---\nlicense: apache-2.0\n---\n")
    os.symlink(str(elsewhere / "README.md"), snap / "README.md")
    out = tmp_path / "cardlink"
    assert run(snap, out) == 2 and not out.exists() and "links outside" in LAST["err"]


def test_an_oversized_base_card_is_refused_not_a_traceback(base_snapshot, tmp_path):
    snap = _base_with_card(base_snapshot, tmp_path, "---\nlicense: apache-2.0\n---\n" + "x" * (mf.MAX_CARD_BYTES + 1))
    out = tmp_path / "bigcard"
    assert run(snap, out) == 2 and not out.exists() and "README.md" in LAST["err"] and "Traceback" not in LAST["err"]


@pytest.mark.parametrize("text, want", [
    ("---\nlicense: apache-2.0\n---\n", "apache-2.0"),
    ("---\nlicense: 'Apache-2.0'\n---\n", "apache-2.0"),
    ("---\r\nlicense: apache-2.0\r\n---\r\n", "apache-2.0"),
    ("---\ntags: [a]\n---\nlicense: mit\n", None),
    ("license: mit\n", None),
    ("", None),
    ("---\nlicense:\n---\n", None),
])
def test_the_base_card_license_is_read_from_the_front_matter_only(text, want):
    assert mf.card_license(text) == want
