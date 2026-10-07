"""Fakes for the stage 7 tests: an installed source bundle, a finished run and its HF caches.

`make_source` writes a v6 thin bundle of the nearest model in the layout `tt-model` installs, with a
run.sh whose command carries fixed extra vLLM arguments, as the operator's hand-made bundles do.
`make_run` writes a run directory whose stages 0, 1, 2, 4 and 6 passed on the weights-only path,
plus two Hugging Face caches: the run's (holding the new model) and the operator's (holding the
drafter and the base model). `fake_bin` puts tests/fake_package_thin.py on PATH as `tt-model`.
Nothing here opens a device or reaches the network.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import fake_package_thin

NEW, BASE = "Altworld/Hemmingway-1", "Qwen/Qwen3.8-27B"
NEW_REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf"
BASE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFTER, DRAFTER_REV = "incoai/Qwen3.8-27B-DFlash2", "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
ENTRY = "models.demos.blackhole.qwen36.tt.qwen36_vllm_dflash:Qwen36DFlashForCausalLM"
SOURCE_EXTRA = ("--additional-config '{\"tt\": {\"l1_small_size\": 24576}}' "
                "--max-num-batched-tokens 65536 --no-async-scheduling")
SOURCE_ENV = {"ARCH_NAME": "blackhole", "DFLASH_WEIGHTS": f"{DRAFTER}@{DRAFTER_REV}",
              "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES": "0"}
VOCAB = [a + b for a in ("ba", "de", "ki", "lo", "mu", "ra", "so", "tu") for b in ("n", "l", "r", "s", "t")]
PROMPT_IDS = [1, 2, 3, 4, 5]
GENERATED = list(range(2, 34))
HAVE_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


def tokenizer_json() -> str:
    """A WordLevel tokenizer over VOCAB when `tokenizers` is importable, else a stub."""
    if not HAVE_TOKENIZERS:
        return "{}"
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)} | {"[UNK]": len(VOCAB)},
                              unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return tok.to_str()


def make_source(root: Path, *, name="qwen3.8-27b-dflash2-p300", chips=2, mesh="P150x2",
                weights=BASE, env=None) -> Path:
    """An installed v6 bundle at root/<org>/<name>, as `tt-model` lays it out."""
    b = root / "episod" / name
    (b / "wheels").mkdir(parents=True)
    (b / "wheels" / "vllm_tt_plugin-0.1.0-py3-none-any.whl").write_bytes(b"PK plugin wheel")
    (b / "wheels" / "ttnn-0.79.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"PK ttnn wheel")
    (b / "model.py").write_text(f"from {ENTRY.split(':')[0]} import {ENTRY.split(':')[1]}  # noqa\n")
    (b / "requirements.txt").write_text("ttnn==0.79.0\n")
    (b / "vllm_models" / name).mkdir(parents=True)
    (b / "vllm_models" / name / "vllm_metadata.json").write_text(json.dumps(
        {"arch": "Qwen3_5ForConditionalGeneration", "main_class": ENTRY}))
    env = dict(SOURCE_ENV if env is None else env)
    manifest = {
        "schema_version": "6", "name": name, "arch": "blackhole", "device_count": chips,
        "producer": {"hostname": "redacted"},
        "weights": {"repo_id": weights, "revision": BASE_REV},
        "mesh": {"devices": chips, "topology": mesh, "fabric": None},
        "entrypoint": {"cls": ENTRY, "arch_name": "Qwen3_5ForConditionalGeneration"},
        "resources": {"max_model_len": 262144, "max_num_seqs": 4, "block_size": 64, "extra_args": []},
        "env": env,
        "deps": {"python": "3.12", "requirements": "requirements.txt",
                 "wheels": ["wheels/vllm_tt_plugin-0.1.0-py3-none-any.whl"], "wheels_dir": "wheels",
                 "models_wheels": ["wheels/ttnn-0.79.0-cp312-cp312-linux_x86_64.whl"],
                 "vllm": {"version": "0.26.0", "target_device": "empty"}, "kind": "vllm"}}
    (b / "tt_kernel_manifest.json").write_text(json.dumps(manifest, indent=2))
    run_sh = fake_package_thin.RUN_SH.format(
        weights=weights, rev=BASE_REV, seqs=4, block=64, ctx=262144,
        exports="\n".join(f'export {k}="{v}"' for k, v in env.items()))
    (b / "run.sh").write_text(run_sh.replace(' "$@")', f' {SOURCE_EXTRA} "$@")'))
    return b


def hf_snapshot(hf_home: Path, repo: str, rev: str, files: dict) -> Path:
    """An HF-cache-shaped snapshot under hf_home/hub: files are relative links into blobs/.

    Blobs are named by a hash of their content, as in the real cache, so two revisions of one repo
    share a blob only when the file's bytes are the same."""
    org, name = repo.split("/")
    root = hf_home / "hub" / f"models--{org}--{name}"
    blobs, snap = root / "blobs", root / "snapshots" / rev
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True)
    for fname, content in files.items():
        blob = blobs / hashlib.sha256(content.encode()).hexdigest()
        blob.write_text(content)
        (snap / fname).symlink_to(os.path.relpath(blob, snap))
    return snap


def write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, indent=2))
    return path


def make_run(tmp: Path, source: Path, *, license_id="cc-by-nc-4.0") -> dict:
    """A run whose stages 0, 1, 2, 4 and 6 passed on the weights-only path, and its two HF caches."""
    run, hf_run, hf_op = tmp / "run", tmp / "hf-run", tmp / "hf-operator"
    front = f"---\nlicense: {license_id}\nbase_model:\n- {BASE}\n---\n" if license_id else ""
    snap = hf_snapshot(hf_run, NEW, NEW_REV, {
        "README.md": front + "# Hemmingway-1\n", "config.json": '{"architectures": ["Qwen3_5ForCausalLM"]}',
        "tokenizer.json": tokenizer_json(), "model-00001-of-00001.safetensors": "new weights"})
    hf_snapshot(hf_op, DRAFTER, DRAFTER_REV, {"model.safetensors": "drafter"})
    hf_snapshot(hf_op, BASE, BASE_REV, {"model-00001-of-00001.safetensors": "base weights"})
    # Stage 0 names both models as <repo>@<revision>, as the delta-triage skill asks.
    write(run / "stages/0/delta.json", {"model": f"{NEW}@{NEW_REV}", "nearest_model": f"{BASE}@{BASE_REV}",
                                        "path": "weights-only"})
    ref = run / "stages/1/evidence/reference"
    write(ref / "prompt-ids.json", {"prompt_ids": PROMPT_IDS})
    write(ref / "generated-ids.json", {"generated_ids": GENERATED,
                                       "generated_text": " ".join(VOCAB[i] for i in GENERATED)})
    s2 = run / "stages/2"
    # The label is the short repo id, as run 3's stage agent typed it. model_dir is what was served.
    write(s2 / "evidence/swap-check.json", {"top1_agreement": 0.94, "new_model_id": NEW,
                                            "model_dir": str(s2 / "model-dir")})
    write(s2 / "evidence/server.log", "ready\n")
    write(s2 / "result.json", {"serves": True, "server_ready_s": 280.5, "coherent": True,
                               "free_run_text": "x", "top1_agreement": 0.94, "n_tokens": 32,
                               "cache_dir": "c", "evidence": ["stages/2/evidence/swap-check.json",
                                                              "stages/2/evidence/server.log"]})
    write(s2 / "swap_config.json", {"run_dir": str(run), "bundle_dir": str(source),
                                    "nearest_model_id": BASE, "new_snapshot": str(snap),
                                    "new_model_id": NEW, "hf_home": str(hf_op), "port": 8100})
    write(s2 / "model-dir/config.json", '{"architectures": ["Qwen3_5ForConditionalGeneration"]}')
    write(s2 / "model-dir/preprocessor_config.json", "{}")
    (s2 / "model-dir/tokenizer.json").symlink_to(snap / "tokenizer.json")
    # prepare_swap.py links each weight file to its blob (the realpath of the snapshot file).
    weights = "model-00001-of-00001.safetensors"
    (s2 / "model-dir" / weights).symlink_to(os.path.realpath(snap / weights))
    write(run / "stages/4/evidence/hw-test-output.txt", "ok\n")
    write(run / "stages/4/result.json", {"configs": [
        {"chips": 2, "pass": True, "evidence": ["stages/4/evidence/hw-test-output.txt"]},
        {"chips": 1, "pass": True, "evidence": ["stages/4/evidence/hw-test-output.txt"]},
        {"chips": 4, "pass": True, "evidence": ["stages/4/evidence/hw-test-output.txt"]}]})
    write(run / "stages/6/evidence/bench.txt", "decode 80.0 tok/s/user\n")
    write(run / "stages/6/result.json", {"numbers": [
        {"name": "decode", "value": 80.0, "unit": "tok/s/user", "label": "measured",
         "evidence": ["stages/6/evidence/bench.txt"]},
        {"name": "ttft", "value": None, "unit": "ms", "label": "TODO"}],
        "qualitative": {"evidence": ["stages/6/evidence/bench.txt"]}})
    return {"run": run, "hf_run": hf_run, "hf_op": hf_op, "snapshot": snap}


def fake_bin(tmp: Path) -> tuple[Path, Path]:
    """A directory holding `tt-model` (tests/fake_package_thin.py) and the log it writes."""
    d = tmp / "fake-bin"
    d.mkdir()
    log = tmp / "tt-model-calls.jsonl"
    (d / "tt-model").write_text(
        f'#!/bin/sh\nFAKE_TT_MODEL_LOG="{log}" exec "{sys.executable}" "{fake_package_thin.__file__}" "$@"\n')
    (d / "tt-model").chmod(0o755)
    return d, log


def calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def fake_boot_result(stage: Path, *, returncode=0, top1=0.94) -> None:
    """What the supervisor and verify_bundle.py leave in stages/7 after the boot check: the test's
    output and test-result.json, and with exit 0 the verify.json evidence."""
    write(stage / "evidence/hw-test-output.txt", "verify_bundle: done\n")
    write(stage / "test-result.json", {"returncode": returncode, "timed_out": False,
                                       "output": {"path": "stages/7/evidence/hw-test-output.txt",
                                                  "sha256": "0" * 64}})
    if returncode == 0:
        md = stage / "verify/bundle/model-dir"
        write(stage / "verify/evidence/server.log", "ready\n")
        write(stage / "verify/evidence/verify.json", {
            "label": "measured", "top1_agreement": top1, "coherent": True, "n_tokens": 32,
            "server_ready_s": 301.2, "served_model": str(md),
            "server_weights_env": {"MODEL_WEIGHTS_DIR": str(md), "HF_MODEL": str(md)},
            "evidence": ["stages/7/verify/evidence/verify.json",
                         "stages/7/verify/evidence/server.log"]})


SIDECAR = {"file": "joint_head.safetensors", "size": 256125024, "sha256": "a" * 64, "num_tensors": 122}


def make_clef_like(run: Path, source: Path) -> None:
    """Turn make_run's run into Clef's shape: stage 2 served with the speculative drafter off and host
    sampling (stages/2/run.sh is the edited copy prepare_swap.py leaves), and stage 0 found a sidecar."""
    text = (source / "run.sh").read_text()
    text = re.sub(r"^export QWEN36_DRAFTER=.*\n", "", text, flags=re.M)
    text = text.replace("CMD=(", 'export QWEN36_DRAFTER=""\nCMD=(', 1)
    write(run / "stages/2/run.sh", text)
    d = json.loads((run / "stages/0/delta.json").read_text())
    write(run / "stages/0/delta.json", {**d, "class": "weights+sidecar", "sidecars": [SIDECAR]})
