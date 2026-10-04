"""The stage 1 template the reference-gate skill copies: reference_gate.py.

The end-to-end tests build a TINY random-weight model with transformers (a 2-layer Qwen2 with a
hidden size of 32) and a byte-level BPE tokenizer trained with the `tokenizers` package, save both
with save_pretrained into a tmp directory, and run the script as a subprocess, the way the agent
runs it. The helper tests import the script by path and use fakes. Nothing here opens a device:
torch runs on CPU.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "reference-gate-templates" / "reference_gate.py"
STACK = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "tokenizers"))
needs_stack = pytest.mark.skipif(not STACK, reason=(
    "SKIPPED: torch, transformers or tokenizers is not importable, so the reference-gate template "
    "was not run end to end on this interpreter. Its helper tests still ran."))


def template_module():
    spec = importlib.util.spec_from_file_location("reference_gate_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- helpers, with fakes --------------------------------------------------------------------------

def steps_from(prompt, tokens):
    """Loop records as the script keeps them: each step's sequence before and after its token."""
    out, seq = [], list(prompt)
    for t in tokens:
        out.append({"chosen": t, "before": list(seq), "after": seq + [t]})
        seq = seq + [t]
    return out


def test_tokens_appended_at_the_end_pass_the_append_check():
    mod = template_module()
    ok, why = mod.appended_at_end([1, 2, 3], steps_from([1, 2, 3], [7, 8, 9]))
    assert ok, why


@pytest.mark.parametrize("bad", ["prepended", "dropped", "two", "wrong-token"])
def test_a_token_not_appended_at_the_end_fails_the_append_check(bad):
    mod = template_module()
    steps = steps_from([1, 2, 3], [7, 8, 9])
    if bad == "prepended":           # the new token went to the front
        steps[1]["after"] = [8] + steps[1]["before"]
    elif bad == "dropped":           # the sequence lost its last token
        steps[1]["after"] = steps[1]["before"][:-1] + [8]
    elif bad == "two":               # two tokens were added in one step
        steps[1]["after"] = steps[1]["before"] + [8, 8]
    else:                            # the token appended is not the one chosen
        steps[1]["after"] = steps[1]["before"] + [5]
    ok, why = mod.appended_at_end([1, 2, 3], steps)
    assert not ok and "step 1" in why


def test_a_step_that_does_not_continue_from_the_previous_one_fails():
    mod = template_module()
    steps = steps_from([1, 2, 3], [7, 8, 9])
    steps[2]["before"] = [1, 2, 3, 7]          # step 1's token was lost between steps
    steps[2]["after"] = [1, 2, 3, 7, 9]
    ok, why = mod.appended_at_end([1, 2, 3], steps)
    assert not ok and "step 2" in why


@pytest.mark.parametrize("text,start", [("Let me write this.", "Let me"), ("We need to be brief.", "We need"),
                                        ("<think>\nok", "<think"), ("  Let me see", "Let me"),
                                        ("Hi Sam, the boiler is broken.", None)])
def test_reasoning_starts_are_recognised(text, start):
    assert template_module().reasoning_start(text) == start


def test_ignored_keys_follow_the_class_patterns():
    mod = template_module()
    keys = ["model.embed_tokens.weight", "mtp.fc.weight", "mtp.layers.0.norm.weight", "lm_head.weight"]
    ignored, kept = mod.split_ignored(keys, [r"^mtp.*"])
    assert ignored == ["mtp.fc.weight", "mtp.layers.0.norm.weight"]
    assert kept == ["model.embed_tokens.weight", "lm_head.weight"]
    assert mod.split_ignored(keys, None) == ([], keys)


def test_the_round_trip_strings_cover_the_scripts():
    mod = template_module()
    assert len(mod.ROUNDTRIP_STRINGS) == 14
    joined = "".join(mod.ROUNDTRIP_STRINGS)
    for ch in ("é", "😀", "你", "न", "ส", "!", "\t"):
        assert ch in joined, ch
    assert any(s.startswith(" ") and s.endswith(" ") for s in mod.ROUNDTRIP_STRINGS)
    assert mod.PROMPT == "Write the text I send my landlord about the broken boiler."


def test_a_missing_package_exits_4_and_names_it(tmp_path):
    stage = tmp_path / "run" / "stages" / "1"
    stage.mkdir(parents=True)
    script = stage / "reference_gate.py"
    script.write_text(TEMPLATE.read_text())
    (stage / "reference_config.json").write_text(json.dumps(
        {"run_dir": str(tmp_path / "run"), "model_snapshot": str(tmp_path), "python": sys.executable}))
    guard = ("import runpy, sys\nsys.modules['torch'] = None\n"
             f"sys.argv = [{str(script)!r}]\nrunpy.run_path({str(script)!r}, run_name='__main__')\n")
    r = subprocess.run([sys.executable, "-c", guard], capture_output=True, text=True, timeout=120)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "torch" in r.stderr and "not importable" in r.stderr
    assert not (stage / "reference.json").exists()


def test_a_missing_config_key_exits_2(tmp_path):
    stage = tmp_path / "run" / "stages" / "1"
    stage.mkdir(parents=True)
    (stage / "reference_gate.py").write_text(TEMPLATE.read_text())
    (stage / "reference_config.json").write_text(json.dumps({"run_dir": str(tmp_path / "run")}))
    r = subprocess.run([sys.executable, str(stage / "reference_gate.py")], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 2 and "model_snapshot" in r.stderr


# ---- end to end with a tiny random model ----------------------------------------------------------

THAI_CORPUS = ["สวัสดีครับ", "เขียนข้อความถึงเจ้าของบ้านเรื่องหม้อต้มน้ำร้อนเสีย", "ภาษาไทย"]
LATIN_CORPUS = ["Write the text I send my landlord about the broken boiler.", "Hello, world.",
                "The quick brown fox jumps over the lazy dog.", "user assistant system"]
TEMPLATE_JINJA = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                  "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def build_model(root: Path, *, thai: bool, drop_key: str | None = None, extra_key: str | None = None) -> Path:
    """A 2-layer random Qwen2 and its tokenizer, saved like a Hugging Face snapshot."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator((LATIN_CORPUS + (THAI_CORPUS if thai else [])) * 20, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = TEMPLATE_JINJA
    snap = root / "snapshots" / ("0" * 40)
    snap.mkdir(parents=True)
    fast.save_pretrained(snap)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(snap, safe_serialization=True)
    if drop_key or extra_key:
        from safetensors.torch import load_file, save_file
        path = snap / "model.safetensors"
        state = load_file(str(path))
        if drop_key:
            del state[drop_key]
        if extra_key:
            state[extra_key] = torch.zeros(4, 4, dtype=torch.bfloat16)
        save_file(state, str(path), metadata={"format": "pt"})
    return snap


def run_gate(tmp_path: Path, snap: Path):
    stage = tmp_path / "run" / "stages" / "1"
    stage.mkdir(parents=True)
    (stage / "reference_gate.py").write_text(TEMPLATE.read_text())
    (stage / "reference_config.json").write_text(json.dumps(
        {"run_dir": str(tmp_path / "run"), "model_snapshot": str(snap), "python": sys.executable}))
    r = subprocess.run([sys.executable, str(stage / "reference_gate.py")], capture_output=True, text=True,
                       timeout=600)
    return stage, r


@pytest.fixture(scope="module")
def good_run(tmp_path_factory):
    if not STACK:
        pytest.skip("SKIPPED: torch, transformers or tokenizers is not importable")
    tmp = tmp_path_factory.mktemp("refgate")
    snap = build_model(tmp / "model", thai=True)
    stage, r = run_gate(tmp, snap)
    return stage, r


def ev(stage: Path, *parts) -> Path:
    return stage.joinpath("evidence", *parts)


@needs_stack
def test_the_script_writes_every_evidence_file_and_a_reference_that_passes_the_gate(good_run):
    from orchard.stages import gate_reference
    stage, r = good_run
    assert r.returncode == 0, r.stdout + r.stderr
    for name in ("load-report.json", "tokenizer-roundtrip.json", "decode.txt", "card-check.txt",
                 "summary.json", "reference/prompt-ids.json", "reference/generated-ids.json",
                 "reference/top5-logits.json"):
        assert ev(stage, *name.split("/")).is_file(), name
    ref = json.loads((stage / "reference.json").read_text())
    assert [c["name"] for c in ref["checks"]] == ["loads", "tokenizer round trip", "decodes forward",
                                                  "matches the card"]
    gate = gate_reference(stage, stage.parent.parent)
    assert gate.ok, (gate.reasons, ref)
    assert ref["environment"]["torch"] and ref["environment"]["transformers"]


@needs_stack
def test_the_reference_ids_have_the_shape_stage_2_reads(good_run):
    stage, _ = good_run
    p = json.loads(ev(stage, "reference", "prompt-ids.json").read_text())
    g = json.loads(ev(stage, "reference", "generated-ids.json").read_text())
    assert isinstance(p["prompt_ids"], list) and p["prompt_ids"] and "boiler" in p["prompt_text"]
    assert len(g["generated_ids"]) == 32 and isinstance(g["generated_text"], str) and g["complete"] is True
    top = json.loads(ev(stage, "reference", "top5-logits.json").read_text())
    assert len(top["steps"]) == 32 and all(len(s["top5"]) == 5 for s in top["steps"])
    assert top["prompt_last_position"]["argmax"] == g["generated_ids"][0]


@needs_stack
def test_the_decode_checks_are_recorded(good_run):
    stage, _ = good_run
    ref = json.loads((stage / "reference.json").read_text())
    decode = next(c for c in ref["checks"] if c["name"] == "decodes forward")
    assert decode["pass"] is True
    assert "appended at the end" in decode["note"] and "first generated token" in decode["note"]
    text = ev(stage, "decode.txt").read_text()
    assert "append check: pass" in text and "first-token check: pass" in text


@needs_stack
def test_the_load_report_counts_keys(good_run):
    stage, _ = good_run
    rep = json.loads(ev(stage, "load-report.json").read_text())
    assert rep["missing_keys"] == [] and rep["unexpected_keys"] == [] and rep["mismatched_keys"] == []
    assert rep["pass"] is True and rep["dtype"] == "torch.bfloat16" and rep["low_cpu_mem_usage"] is True
    assert rep["checkpoint_key_count"] > 0


@needs_stack
def test_the_round_trip_and_the_card_check(good_run):
    stage, _ = good_run
    rt = json.loads(ev(stage, "tokenizer-roundtrip.json").read_text())
    assert rt["total"] == 14 and rt["ok"] == 14
    ref = json.loads((stage / "reference.json").read_text())
    card = next(c for c in ref["checks"] if c["name"] == "matches the card")
    assert "form check only" in card["note"] and "reasoning" in card["note"]
    assert "form check only" in ev(stage, "card-check.txt").read_text()


@needs_stack
def test_a_tokenizer_with_thai_tokens_gets_a_thai_decode_too(good_run):
    stage, r = good_run
    assert ev(stage, "reference", "thai-prompt-ids.json").is_file()
    assert ev(stage, "reference", "thai-generated-ids.json").is_file()
    summary = json.loads(ev(stage, "summary.json").read_text())
    assert summary["thai_tokens"] is True


@needs_stack
def test_the_script_streams_progress(good_run):
    _, r = good_run
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("reference_gate:")]
    assert any("step 32/32" in ln for ln in lines)
    assert any("load-report.json" in ln for ln in lines)
    assert len(lines) >= 40


@needs_stack
def test_a_tokenizer_without_thai_tokens_gets_no_thai_decode(tmp_path):
    snap = build_model(tmp_path / "model", thai=False)
    stage, r = run_gate(tmp_path, snap)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not ev(stage, "reference", "thai-prompt-ids.json").exists()
    assert json.loads(ev(stage, "summary.json").read_text())["thai_tokens"] is False


@needs_stack
def test_a_missing_weight_is_reported_and_fails_the_load_check(tmp_path):
    snap = build_model(tmp_path / "model", thai=False, drop_key="model.layers.1.mlp.up_proj.weight")
    stage, r = run_gate(tmp_path, snap)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(ev(stage, "load-report.json").read_text())
    assert rep["missing_keys"] == ["model.layers.1.mlp.up_proj.weight"] and rep["pass"] is False
    ref = json.loads((stage / "reference.json").read_text())
    loads = next(c for c in ref["checks"] if c["name"] == "loads")
    assert loads["pass"] is False and "model.layers.1.mlp.up_proj.weight" in loads["note"]
    assert ref["verdict"] == "fail"


@needs_stack
def test_an_unexpected_weight_is_reported(tmp_path):
    snap = build_model(tmp_path / "model", thai=False, extra_key="extra.proj.weight")
    stage, r = run_gate(tmp_path, snap)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(ev(stage, "load-report.json").read_text())
    assert rep["unexpected_keys"] == ["extra.proj.weight"] and rep["pass"] is False
