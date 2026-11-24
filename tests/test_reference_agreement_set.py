# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 15: stage 1 writes the CPU reference of the agreement set.

The stage 2 and 4 check (`serve_and_compare.py`, step j) teacher-forces the six prompts of `agreement_prompts.json` against the server and
scores the chip's ids against `stages/1/evidence/reference/agreement-reference.json`. Until this change nothing wrote that file, so the set
stayed off. With `"agreement_set": true` in `reference_config.json`, the reference template (`reference_gate.py`) now renders each prompt
through the chat template, decodes 32 greedy tokens on the CPU and writes the file, with the model's log-probability of each token.

The tests run the template for real on a tiny random Qwen2 (CPU), the way test_reference_fidelity_tokens does, with a short prompt set,
because the tiny model has 256 positions and the shipped long prompt has about 2,000 tokens."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from test_reference_gate_template import STACK, TEMPLATE, build_model, ev, needs_stack, template_module

REPO = Path(__file__).resolve().parent.parent
SHIPPED_SET = REPO / "orchard" / "skills" / "weights-swap-templates" / "agreement_prompts.json"
N = 32


def small_set(count=5, **change):
    """A valid prompt set of `count` short prompts for the tiny model."""
    kinds = ["chat", "code", "tool-call", "non-english", "summary", "list"]
    prompts = [{"id": f"p{i}-{kinds[i % len(kinds)]}", "kind": kinds[i % len(kinds)],
                "messages": [{"role": "user", "content": f"Tell me item {i} about boilers and old pipes."}]} for i in range(count)]
    return {"version": 1, "note": "test set", "prompts": prompts, **change}


def write_set(path: Path, data) -> bytes:
    raw = (json.dumps(data, indent=1) + "\n").encode()
    path.write_bytes(raw)
    return raw


def run_with(tmp_path, snap, prompt_set="default", **config):
    stage = tmp_path / "run" / "stages" / "1"
    stage.mkdir(parents=True)
    (stage / "reference_gate.py").write_text(TEMPLATE.read_text())
    if prompt_set == "default":
        write_set(stage / "agreement_prompts.json", small_set())
    elif prompt_set is not None:
        (stage / "agreement_prompts.json").write_bytes(prompt_set if isinstance(prompt_set, bytes) else json.dumps(prompt_set).encode())
    (stage / "reference_config.json").write_text(json.dumps(
        {"run_dir": str(tmp_path / "run"), "model_snapshot": str(snap), "python": sys.executable, **config}))
    r = subprocess.run([sys.executable, str(stage / "reference_gate.py")], capture_output=True, text=True, timeout=600)
    return stage, r


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    if not STACK:
        pytest.skip("SKIPPED: torch, transformers or tokenizers is not importable")
    tmp = tmp_path_factory.mktemp("agreeref")
    snap = build_model(tmp / "model", thai=False)
    stage, r = run_with(tmp / "a", snap, agreement_set=True)
    stage2, r2 = run_with(tmp / "b", snap, agreement_set=True)
    plain, rp = run_with(tmp / "c", snap)
    return {"snap": snap, "stage": stage, "r": r, "stage2": stage2, "r2": r2, "plain": plain, "rp": rp}


def ref_path(stage):
    return ev(stage, "reference", "agreement-reference.json")


def ref(stage):
    return json.loads(ref_path(stage).read_text())


@needs_stack
def test_the_option_writes_a_reference_entry_for_every_prompt_of_the_set(made):
    assert made["r"].returncode == 0, made["r"].stdout + made["r"].stderr
    doc = ref(made["stage"])
    expected = small_set()["prompts"]
    assert [e["id"] for e in doc["entries"]] == [p["id"] for p in expected]
    assert [e["kind"] for e in doc["entries"]] == [p["kind"] for p in expected]
    for e in doc["entries"]:
        assert len(e["generated_ids"]) == N and len(e["prompt_ids"]) > 5
        assert all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in e["prompt_ids"] + e["generated_ids"])


@needs_stack
def test_the_record_names_the_prompt_set_by_the_sha256_of_the_file_it_read(made):
    raw = (made["stage"] / "agreement_prompts.json").read_bytes()
    assert ref(made["stage"])["prompt_set_sha256"] == hashlib.sha256(raw).hexdigest()


@needs_stack
def test_the_prompt_ids_are_the_chat_template_rendering_of_the_messages(made):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(made["snap"]))
    doc = ref(made["stage"])
    for e, p in zip(doc["entries"], small_set()["prompts"]):
        text = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
        assert e["prompt_ids"] == tok(text, add_special_tokens=False)["input_ids"]
        assert e["chat_template_used"] is True


@needs_stack
def test_the_generated_ids_are_the_greedy_argmax_of_the_whole_sequence_at_every_step(made):
    """Every step's token is the argmax of a fresh forward pass over the prompt and the tokens before it, with no cache: the chip check
    teacher-forces the same way (one request per position), so this is the reference it is scored against."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(made["snap"]), torch_dtype=torch.bfloat16).eval()
    entry = ref(made["stage"])["entries"][0]
    seq = list(entry["prompt_ids"])
    with torch.inference_mode():
        for k in range(N):
            logits = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0, -1].float()
            assert int(torch.argmax(logits)) == entry["generated_ids"][k], f"step {k}"
            seq.append(entry["generated_ids"][k])


@needs_stack
def test_each_logprob_is_the_models_log_probability_of_its_token_and_is_finite(made):
    entry = ref(made["stage"])["entries"][1]
    assert len(entry["logprobs"]) == N
    assert all(isinstance(v, float) and math.isfinite(v) and v <= 0 for v in entry["logprobs"])
    # greedy picks the most likely token, so its log-probability cannot be below the uniform choice over the vocabulary
    assert all(v > -math.log(10_000) for v in entry["logprobs"])


@needs_stack
def test_a_rerun_writes_the_same_bytes(made):
    assert ref_path(made["stage"]).read_bytes() == ref_path(made["stage2"]).read_bytes()


@needs_stack
def test_without_the_option_no_reference_file_is_written_and_the_summary_says_nothing_about_it(made):
    assert made["rp"].returncode == 0
    assert not ref_path(made["plain"]).exists()
    assert "agreement_reference" not in json.loads(ev(made["plain"], "summary.json").read_text())


@needs_stack
def test_the_summary_and_the_artifact_list_name_the_reference(made):
    summary = json.loads(ev(made["stage"], "summary.json").read_text())["agreement_reference"]
    assert summary["prompts"] == 5 and summary["tokens_each"] == N and summary["complete"] is True
    art = json.loads((made["stage"] / "reference.json").read_text())["reference_artifacts"]
    assert art["agreement_reference"] == "stages/1/evidence/reference/agreement-reference.json"


@needs_stack
def test_the_stage_still_passes_its_gate_with_the_option_on(made):
    from orchard.stages import gate_reference
    gate = gate_reference(made["stage"], made["stage"].parent.parent)
    assert gate.ok, gate.reasons


@needs_stack
def test_the_stage_2_template_accepts_the_file_as_its_reference(made, tmp_path, monkeypatch):
    """The reader that exists today: serve_and_compare.agreement_preflight must take the written file without exiting."""
    import importlib.util
    templates = REPO / "orchard" / "skills" / "weights-swap-templates"
    sys.path.insert(0, str(templates))
    spec = importlib.util.spec_from_file_location("serve_and_compare_under_test", templates / "serve_and_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stage2 = tmp_path / "s2"
    stage2.mkdir()
    (stage2 / "agreement_prompts.json").write_bytes((made["stage"] / "agreement_prompts.json").read_bytes())
    monkeypatch.setattr(mod, "STAGE_DIR", stage2)
    run_dir = made["stage"].parent.parent
    try:
        out = mod.agreement_preflight({"agreement_set": True, "run_dir": str(run_dir)})
    finally:
        sys.path.remove(str(templates))
    assert [e["id"] for e in out["entries"]] == [p["id"] for p in small_set()["prompts"]]


# ---- a bad request stops the script before the model loads ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["yes", 1, 0, [], {}, None])
def test_an_agreement_set_value_that_is_not_true_or_false_exits_2_before_the_model_loads(tmp_path, bad):
    stage, r = run_with(tmp_path, tmp_path / "no-model", agreement_set=bad)
    # the message is the key's own; the path of the missing model must not be what stopped the script (the test's tmp_path holds the test's name)
    assert r.returncode == 2 and "agreement_set must be true or false" in r.stderr and "model_snapshot" not in r.stderr
    assert not (stage / "evidence").exists() or not list((stage / "evidence").glob("*.json"))


def test_false_and_a_missing_key_both_mean_off(tmp_path):
    mod = template_module()
    assert mod.agreement_setting({}) is False and mod.agreement_setting({"agreement_set": False}) is False
    assert mod.agreement_setting({"agreement_set": True}) is True


def test_the_option_without_the_prompt_file_exits_2_and_says_where_to_copy_it_from(tmp_path):
    stage, r = run_with(tmp_path, tmp_path / "no-model", prompt_set=None, agreement_set=True)
    assert r.returncode == 2
    assert "agreement_prompts.json" in (r.stdout + r.stderr) and "weights-swap-templates" in (r.stdout + r.stderr)


@pytest.mark.parametrize("change,word", [
    (lambda d: d["prompts"].__delitem__(slice(4, None)), "at least 5"),
    (lambda d: d["prompts"][1].update(id=d["prompts"][0]["id"]), "duplicate"),
    (lambda d: d["prompts"][2].update(messages=[]), "messages"),
    (lambda d: d["prompts"][2].update(messages=[{"role": "user", "content": 7}]), "messages"),
    (lambda d: d["prompts"][3].update(kind=None), "kind"),
    (lambda d: d["prompts"][0].update(id=""), "id"),
])
def test_a_prompt_set_the_gate_would_refuse_exits_2_before_the_model_loads(tmp_path, change, word):
    data = small_set(6)
    change(data)
    stage, r = run_with(tmp_path, tmp_path / "no-model", prompt_set=data, agreement_set=True)
    assert r.returncode == 2 and word in (r.stdout + r.stderr)


def test_a_prompt_file_that_is_not_json_or_not_an_object_exits_2(tmp_path):
    for raw in (b"not json", b"[1, 2]", b"\xff\xfe"):
        sub = tmp_path / str(len(raw)) / raw.hex()[:6]
        sub.mkdir(parents=True)
        stage, r = run_with(sub, tmp_path / "no-model", prompt_set=raw, agreement_set=True)
        assert r.returncode == 2, raw


def test_the_template_check_of_the_prompt_set_agrees_with_the_repository_check():
    """The template cannot import orchard, so it has its own copy of the rule. It must give the same verdict as agreement_core on the shipped
    set and on sets that break each rule."""
    from orchard import agreement_core as core
    mod = template_module()
    shipped = json.loads(SHIPPED_SET.read_text())
    assert mod.agreement_prompt_problems(shipped) == [] == core.prompt_set_problems(shipped)
    for data in (small_set(4), small_set(6, prompts=[]), [], {"prompts": "x"}):
        assert bool(mod.agreement_prompt_problems(data)) == bool(core.prompt_set_problems(data)), data
    dup = small_set(6)
    dup["prompts"][1]["id"] = dup["prompts"][0]["id"]
    assert bool(mod.agreement_prompt_problems(dup)) and bool(core.prompt_set_problems(dup))


# ---- the decode itself, with a fake model ---------------------------------------------------------------------------------------

class FakeTok:
    """Renders messages as one id per character of the last message and decodes ids back to letters."""
    chat_template = "x"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


def test_a_prompt_that_fails_leaves_no_reference_file_and_names_the_prompt(tmp_path, monkeypatch):
    """A file with fewer entries than the set would be taken for a complete reference by the supervisor, so a failure writes none."""
    mod = template_module()
    mod.REFERENCE = tmp_path / "evidence" / "reference"
    mod.EVIDENCE = tmp_path / "evidence"
    calls = []

    def fake_decode(torch, model, tokenizer, prompt_ids, n, recompute_max):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("out of memory")
        return {"generated_ids": list(range(n)), "logprobs": [-1.0] * n, "mode": "recompute"}

    monkeypatch.setattr(mod, "agreement_greedy", fake_decode)
    path = tmp_path / "agreement_prompts.json"
    write_set(path, small_set())
    doc = mod.agreement_reference(None, None, FakeTok(), path, N)
    assert doc["complete"] is False and doc["error"].startswith("prompt 'p2-tool-call'") and "out of memory" in doc["error"]
    assert not (mod.REFERENCE / "agreement-reference.json").exists()
    assert (mod.REFERENCE / "agreement-reference-error.json").is_file()


# ---- the two decode paths and the atomic write -----------------------------------------------------------------------------------

@needs_stack
def test_short_prompts_are_decoded_without_the_cache_and_the_entry_says_so(made):
    assert {e["decode_mode"] for e in ref(made["stage"])["entries"]} == {"recompute"}


@needs_stack
def test_a_prompt_above_the_limit_uses_the_cache_and_starts_with_the_same_token(made):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mod = template_module()
    tok = AutoTokenizer.from_pretrained(str(made["snap"]))
    model = AutoModelForCausalLM.from_pretrained(str(made["snap"]), torch_dtype=torch.bfloat16).eval()
    ids = ref(made["stage"])["entries"][0]["prompt_ids"]
    cached = mod.agreement_greedy(torch, model, tok, ids, 8, recompute_max=0)
    whole = mod.agreement_greedy(torch, model, tok, ids, 8, recompute_max=10_000)
    assert cached["mode"] == "key-value cache" and whole["mode"] == "recompute"
    assert len(cached["generated_ids"]) == len(cached["logprobs"]) == 8
    assert cached["generated_ids"][0] == whole["generated_ids"][0]
    assert all(math.isfinite(v) and v <= 0 for v in cached["logprobs"])


def test_a_model_that_returns_no_cache_raises_rather_than_decoding_without_one():
    mod = template_module()
    torch = pytest.importorskip("torch")

    class NoCache:
        def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
            return type("Out", (), {"logits": torch.zeros(1, input_ids.shape[1], 8), "past_key_values": None})()

    with pytest.raises(RuntimeError, match="key-value cache"):
        mod.agreement_greedy(torch, NoCache(), None, [1, 2, 3], 2, recompute_max=0)


def test_the_write_is_atomic_and_leaves_no_temporary_file(tmp_path):
    mod = template_module()
    target = tmp_path / "d" / "agreement-reference.json"
    mod.write_json_atomic(target, {"a": 1})
    mod.write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert [p.name for p in target.parent.iterdir()] == ["agreement-reference.json"]


def test_a_link_or_a_file_left_at_the_temporary_name_is_not_written_through(tmp_path):
    mod = template_module()
    target = tmp_path / "agreement-reference.json"
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("keep me")
    (tmp_path / "agreement-reference.json.tmp").symlink_to(elsewhere)
    mod.write_json_atomic(target, {"a": 1})
    assert elsewhere.read_text() == "keep me" and json.loads(target.read_text()) == {"a": 1}


def test_a_fifo_named_like_the_prompt_set_is_refused_and_does_not_block(tmp_path):
    import os
    mod = template_module()
    fifo = tmp_path / "agreement_prompts.json"
    os.mkfifo(fifo)
    with pytest.raises(OSError, match="not a regular file"):
        mod.read_small_regular(fifo, 1000)


def test_a_link_or_an_oversized_file_is_refused_by_the_small_reader(tmp_path):
    mod = template_module()
    real = tmp_path / "real.json"
    real.write_text("{}" * 600)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(OSError):
        mod.read_small_regular(link, 5000)
    with pytest.raises(OSError, match="larger"):
        mod.read_small_regular(real, 100)
    assert mod.read_small_regular(real, 5000) == real.read_bytes()


def test_a_file_that_grows_after_its_size_was_read_is_refused(tmp_path, monkeypatch):
    """The size check by name is not enough: the read itself stops at the limit."""
    import os
    import stat as stat_mod
    mod = template_module()
    path = tmp_path / "agreement_prompts.json"
    path.write_bytes(b"x" * 5000)
    real = os.lstat(path)
    fake = os.stat_result((stat_mod.S_IFREG | 0o644, real.st_ino, real.st_dev, 1, real.st_uid, real.st_gid, 10, 0, 0, 0))
    monkeypatch.setattr(mod.os, "lstat", lambda p: fake)          # the size the first look saw
    with pytest.raises(OSError, match="grew past"):
        mod.read_small_regular(path, 100)
