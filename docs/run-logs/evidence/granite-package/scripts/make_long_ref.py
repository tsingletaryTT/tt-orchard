"""Reference logits for a long natural-language sequence (CPU, no chip): the model's own card text, first N tokens, one forward pass in float32
and one in bfloat16 with transformers. Used to check the chip's attention layers far beyond the 40 positions of the earlier prompts.
usage: python make_long_ref.py SNAPSHOT_DIR OUT.npz N [EXTRA_TEXT_FILE]"""
import sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

snap, out, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(str(snap))
extra = Path(sys.argv[4]).read_text() if len(sys.argv) > 4 else ""
text = (snap / "README.md").read_text() + "\n\n" + extra if extra else (snap / "README.md").read_text() * 4
ids = tok(text, add_special_tokens=False)["input_ids"][:n]
assert len(ids) == n, len(ids)
torch.set_num_threads(6)
res = {"tokens": np.array(ids)}
for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    m = AutoModelForCausalLM.from_pretrained(str(snap), dtype=dt).eval()
    with torch.no_grad():
        lg = m(torch.tensor([ids])).logits[0, :-1].float().numpy()
    res[name] = lg
    print(name, lg.shape, flush=True)
    del m
np.savez(out, **res)
