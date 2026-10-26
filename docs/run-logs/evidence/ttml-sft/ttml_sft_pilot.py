"""tt-train SFTTrainer pilot: fine-tune Llama-3.2-1B-Instruct on one Blackhole chip.

Run it under a gozer lease on one chip. It needs the host tt-metal tree's ttml
(shared venv) and numpy 1.x on PYTHONPATH (ttnn 0.80.0 wants numpy < 2).

Data is read from a parquet file or a JSONL file with a `messages` column
(chat turns). The last assistant turn is the completion; everything before it
is the prompt, rendered with the model's chat template and masked out of the
loss. `datasets` is not used (the shared venv's datasets does not import with
its huggingface_hub).

The script records per-step loss and wall time in --out (JSON), and the device
memory tracker's report for the first step.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import ttml
from functools import partial
from transformers import AutoTokenizer

from ttml.common.utils import set_seed
from ttml.datasets import InMemoryDataloader, sft_collate_fn
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from ttml.modules import LoraConfig
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback

MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


def load_rows(path, n, skip=0):
    if path.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(path)]
    else:
        rows = pd.read_parquet(path).to_dict("records")
    return rows[skip : skip + n]


def encode(rows, tok, max_len, pad_id):
    """Tokenise prompt and completion; drop rows that do not fit in max_len."""
    out, dropped = [], 0
    for r in rows:
        msgs = [dict(role=m["role"], content=m["content"]) for m in r["messages"]]
        if msgs[-1]["role"] != "assistant":
            dropped += 1
            continue
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        p = tok.encode(prompt, add_special_tokens=False)
        c = tok.encode(msgs[-1]["content"] + "<|eot_id|>", add_special_tokens=False)
        if len(p) + len(c) > max_len:
            dropped += 1
            continue
        # The SDPA kernel needs the sequence length to be a multiple of 32, and
        # sft_collate_fn pads only to the longest example, so pad every example to max_len here.
        pad_n = max_len - len(p) - len(c)
        ids = p + c + [pad_id] * pad_n
        # Labels must be shifted by one: position t predicts token t+1 (ttml does not shift for us).
        # Only completion tokens count: label t is kept when token t+1 belongs to the completion.
        labels = [-100] * (len(p) - 1) + c + [-100] * (pad_n + 1)
        out.append({"input_ids": ids, "labels": labels})
    return out, dropped


class Recorder(TrainerCallback):
    def __init__(self, path, mem):
        self.path, self.mem, self.recs, self.t = path, mem, [], time.perf_counter()

    def on_step_end(self, trainer, step, loss, lr):
        now = time.perf_counter()
        self.recs.append({"step": step, "loss": loss, "lr": lr, "step_s": now - self.t})
        print(f"step {step} loss={loss:.4f} step_s={now - self.t:.2f}", flush=True)
        if step == 1 and self.mem:
            MemoryUsageTracker.end_capture("FIRST_STEP_DONE")
            MemoryUsageTracker.print_memory_usage()
            MemoryUsageTracker.clear()
        self.t = time.perf_counter()

    def on_eval_end(self, trainer, step, eval_loss):
        print(f"eval step {step} loss={eval_loss:.4f}", flush=True)
        self.recs.append({"step": step, "eval_loss": eval_loss})
        self.t = time.perf_counter()

    def on_train_end(self, trainer):
        json.dump(self.recs, open(self.path, "w"), indent=1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True, help="local dir with model.safetensors")
    a.add_argument("--data", required=True)
    a.add_argument("--n", type=int, default=200)
    a.add_argument("--eval-data", default=None)
    a.add_argument("--eval-n", type=int, default=20)
    a.add_argument("--steps", type=int, default=20)
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--max-len", type=int, default=512)
    a.add_argument("--lr", type=float, default=1e-4)
    a.add_argument("--mode", choices=["lora", "full"], default="lora")
    a.add_argument("--eval-interval", type=int, default=0)
    a.add_argument("--memory", action="store_true")
    a.add_argument("--out", default="metrics.json")
    a.add_argument("--save-dir", default=None)
    args = a.parse_args()

    set_seed(42)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train, dropped = encode(load_rows(args.data, args.n), tok, args.max_len, pad)
    print(f"train examples {len(train)} (dropped {dropped} over {args.max_len} tokens)")
    evald = None
    if args.eval_data:
        evald, d2 = encode(load_rows(args.eval_data, args.eval_n), tok, args.max_len, pad)
        print(f"eval examples {len(evald)} (dropped {d2})")

    if args.memory:
        guard = MemoryUsageTracker.begin_capture()  # keep the guard alive; dropping it ends the capture
    ttml.open_device_mesh(ttml.Mesh((1, 1), ("_0", "_1")))
    ctx = ttml.autograd.AutoContext.get_instance()

    # Llama-3.2-1B: 16 layers, 32 heads, 8 kv heads, hidden 2048, mlp 8192, tied embeddings.
    cfg = LlamaConfig(
        hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, vocab_size=len(tok),
        max_position_embeddings=2048, rope_theta=500000.0,
        runner_type=RunnerType.MemoryEfficient, weight_tying=WeightTyingType.Enabled,
        rope_scaling=LlamaRopeScalingConfig(scaling_factor=32.0, high_freq_factor=4.0,
                                            low_freq_factor=1.0, original_context_length=8192),
    )
    t0 = time.perf_counter()
    model = Llama(cfg)
    load_from_safetensors(model, args.model, cfg)
    print(f"model built and loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    if args.memory:
        MemoryUsageTracker.snapshot("MODEL_LOADED")

    collate = partial(sft_collate_fn, max_seq_len=args.max_len, pad_token_id=pad)
    tl = InMemoryDataloader(train, collate, batch_size=args.batch, shuffle=True)
    el = InMemoryDataloader(evald, collate, batch_size=args.batch, shuffle=False) if evald else None
    peft = LoraConfig(rank=16, alpha=32.0, target_modules=["q_linear", "kv_linear", "out_linear"]) \
        if args.mode == "lora" else None
    sc = SFTConfig(max_steps=args.steps, learning_rate=args.lr, max_seq_len=args.max_len,
                   eval_interval=args.eval_interval, save_interval=0, seed=42)
    rec = Recorder(args.out, args.memory)
    tr = SFTTrainer(model=model, train_dataloader=tl, eval_dataloader=el, config=sc, peft_config=peft,
                    optimizer={"type": "AdamW", "lr": args.lr, "beta1": 0.9, "beta2": 0.999,
                               "epsilon": 1e-8, "weight_decay": 0.01},
                    callbacks=[rec])
    if el is not None:
        rec.recs.append({"step": 0, "eval_loss": tr._eval()})
        print("initial eval loss", rec.recs[-1]["eval_loss"], flush=True)
    tr.train()
    if el is not None:
        rec.recs.append({"step": args.steps, "final_eval_loss": tr._eval()})
        print("final eval loss", rec.recs[-1]["final_eval_loss"], flush=True)
        json.dump(rec.recs, open(args.out, "w"), indent=1)
    ctx.close_device()


if __name__ == "__main__":
    main()
