# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks for the four review findings on scripts/minicpm3 (issue 219), with no chip: ttnn and the chip models are replaced by stand-ins.

1. run_chip passes --layers to both model constructors.
2. make_reference builds the native model from the model directory and the weights file (no sibling `<dir>_w` directory).
3. serve reports finish_reason from the token that ended the loop.
4. summarize's usage text names the arguments it reads.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "minicpm3"


def load_script(monkeypatch, name, fakes):
    """Import scripts/minicpm3/<name>.py fresh with the given stand-in modules in sys.modules."""
    for mod_name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, mod_name, mod)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"_t_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_ttnn():
    return types.ModuleType("ttnn")


class Recorder:
    """A model class that remembers its constructor arguments."""
    calls = []

    def __init__(self, *args, **kwargs):
        Recorder.calls.append(kwargs)


# 1. --layers -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tp", [1, 2])
def test_layers_flag_reaches_the_model_constructor(monkeypatch, tp):
    Recorder.calls = []
    chip = types.ModuleType("chip_model")
    chip.ChipMiniCPM3 = Recorder
    chip_tp = types.ModuleType("chip_model_tp")
    chip_tp.ChipMiniCPM3TP = Recorder
    run_chip = load_script(monkeypatch, "run_chip", {"ttnn": fake_ttnn(), "chip_model": chip, "chip_model_tp": chip_tp})
    args = types.SimpleNamespace(tp=tp, layers=3, model_dir="d", weights="w", max_len=64, mlp_dtype="bf16", mlp_fidelity="hifi2", attn="mla",
                                 fidelity="default", upd="paged", opt=3, shard=False)
    run_chip.build_model(args, dev=object())
    assert Recorder.calls[-1]["n_layers"] == 3


# 3. finish_reason --------------------------------------------------------------------------------------------------------------------------

class FakeTok:
    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids):
        return "".join(chr(97 + i) for i in ids)


class ScriptedModel:
    """Predicts the scripted tokens in order; every logits vector is one-hot on the next scripted token."""
    def __init__(self, script):
        self.script, self.i = script, 0

    def _next(self):
        v = torch.zeros(10)
        v[self.script[min(self.i, len(self.script) - 1)]] = 1.0
        self.i += 1
        return v

    def prefill(self, ids):
        return self._next()

    def traced_step(self, t, p):
        return self._next()


def run_generate(monkeypatch, script, max_tokens):
    serve = load_script(monkeypatch, "serve", {"ttnn": fake_ttnn()})
    monkeypatch.setattr(serve, "EOS", (9,))
    serve.STATE.update(tok=FakeTok(), model=ScriptedModel(script), max_len=128)
    text = "".join(serve.generate([{"role": "user", "content": "hi"}], max_tokens))
    return text, serve.STATE["finish"], serve.STATE["last_counts"]


def test_finish_is_length_when_max_tokens_ends_the_loop_even_if_eos_is_next(monkeypatch):
    text, finish, counts = run_generate(monkeypatch, [4, 5, 9], max_tokens=2)
    assert text == "ef" and counts == (3, 2)
    assert finish == "length"


def test_finish_is_stop_when_eos_ends_the_loop(monkeypatch):
    text, finish, counts = run_generate(monkeypatch, [4, 5, 9], max_tokens=10)
    assert text == "ef" and counts == (3, 2)
    assert finish == "stop"


# 4. summarize usage ------------------------------------------------------------------------------------------------------------------------

def test_summarize_usage_names_only_the_arguments_it_reads(monkeypatch):
    summarize = load_script(monkeypatch, "summarize", {})
    assert "FLOOR" not in summarize.USAGE
    with pytest.raises(SystemExit) as stop:
        summarize.main(["summarize.py"])
    assert "REF_DIR" in str(stop.value.code)


# 2. native model without a sibling directory ------------------------------------------------------------------------------------------------

def test_native_model_is_built_from_the_weights_file(tmp_path, monkeypatch):
    import importlib.util as iu
    if iu.find_spec("transformers.models.minicpm3") is None:
        pytest.skip("this transformers has no minicpm3 model")
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, MiniCPM3Config
    dims = dict(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32,
                kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, scale_emb=4.0, scale_depth=1.4, dim_model_base=16, rms_norm_eps=1e-5,
                max_position_embeddings=256)
    factors = [1.1, 1.4, 2.0, 3.5]
    cfg = MiniCPM3Config(**dims, rope_parameters={"rope_type": "longrope", "factor": 1.0, "rope_theta": 10000.0, "short_factor": factors, "long_factor": factors,
                                                  "original_max_position_embeddings": 256})
    torch.manual_seed(0)
    want_model = AutoModelForCausalLM.from_config(cfg).float().eval()
    state = {k: v.detach().clone().contiguous() for k, v in want_model.state_dict().items() if k.startswith("model.") and "rotary" not in k}
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    save_file(state, str(tmp_path / "w.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(dict(dims, rope_scaling={"type": "longrope", "short_factor": factors, "long_factor": factors,
                                                                              "original_max_position_embeddings": 256})))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    native = importlib.import_module("native_model")
    got_model = native.build_native_model(str(model_dir), str(tmp_path / "w.safetensors"), dtype=torch.float32)
    ids = torch.tensor([[1, 5, 9, 33, 70, 2]])
    with torch.no_grad():
        assert torch.allclose(got_model(ids).logits, want_model(ids).logits, atol=1e-5)
    assert not (tmp_path / "m_w").exists()
    # a weights file with a missing tensor is refused
    state.pop(next(k for k in state if "layers.1." in k))
    save_file(state, str(tmp_path / "short.safetensors"))
    with pytest.raises(ValueError, match="do not match"):
        native.build_native_model(str(model_dir), str(tmp_path / "short.safetensors"), dtype=torch.float32)


def test_make_reference_has_no_sibling_directory_path():
    src = (SCRIPTS / "make_reference.py").read_text()
    assert '"_w"' not in src and "'_w'" not in src
