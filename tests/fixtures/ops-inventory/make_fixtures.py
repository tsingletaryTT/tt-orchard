"""Regenerate the raw-ops fixtures with the reference interpreter. Usage:
    ~/orchard-venvs/reference/bin/python tests/fixtures/ops-inventory/make_fixtures.py
The fixtures are real traces of the architectures in the installed transformers, so the suite needs no torch."""
import json, subprocess, sys, tempfile
from pathlib import Path

import transformers

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent.parent.parent / "orchard" / "skills" / "ops-inventory-templates" / "ops_inventory.py"
CASES = {   # fixture name -> (config class, architectures entry, --class override or None, config overrides)
    "qwen3-tiny": ("Qwen3Config", "Qwen3ForCausalLM", None,
                   dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=2, num_key_value_heads=2, head_dim=16)),
    "rwkv": ("RwkvConfig", "RwkvForCausalLM", None, {}),
    "mamba2": ("Mamba2Config", "Mamba2ForCausalLM", None, {}),
    "moshi-lm": ("MoshiConfig", "MoshiForConditionalGeneration", "MoshiForCausalLM", {}),
}
for name, (cfg_cls, arch, override, kw) in CASES.items():
    with tempfile.TemporaryDirectory() as d:
        getattr(transformers, cfg_cls)(**kw).save_pretrained(d)
        p = Path(d) / "config.json"
        cfg = json.loads(p.read_text())
        cfg["architectures"] = [arch]
        p.write_text(json.dumps(cfg))
        out = HERE / f"raw-{name}.json"
        cmd = [sys.executable, str(TEMPLATE), "--model-dir", d, "--out", str(out)] + (["--class", override] if override else [])
        print(name, subprocess.run(cmd).returncode)
        raw = json.loads(out.read_text())
        raw["model_dir"] = f"<fixture {name}>"          # the temp directory is gone and must not leak into a fixture
        out.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
