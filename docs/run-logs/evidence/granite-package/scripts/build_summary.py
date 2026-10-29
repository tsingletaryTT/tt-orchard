"""Build evidence/summary.json for the Granite package from the raw result files.
usage: python build_summary.py F1_DIR F2_DIR BOOT_JSON_OR_NONE FLOOR_JSON OUT.json [KERNEL_DIR_OR_NONE]
F1_DIR: agree-fp32.json, bench-fp32.json (agreement, decode and prefill at the default limit); F2_DIR: bench-L*.json (context sweep, untraced);
BOOT_JSON: the smoke test's output from a server run (or 'none'); FLOOR_JSON: the bf16 CPU control of the four short prompts."""
import json
import sys
from pathlib import Path

f1, f2, boot, floor, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5]
kdir = sys.argv[6] if len(sys.argv) > 6 else "none"
J = lambda p: json.load(open(p))  # noqa: E731
LAB = "5 repeats after 1 warm-up, one process"
ag, b1 = J(f1 / "agree-fp32.json"), J(f1 / "bench-fp32.json")
fl = J(floor)
dec = lambda d: {k: d["decode"][k] for k in ("mean", "min", "max")}  # noqa: E731
b2048, bun = J(f2 / "bench-L2048.json"), J(f2 / "bench-L2048-untraced.json")
sweep = [{"max_len": L, "mean": J(f2 / f"bench-L{L}.json")["decode"]["mean"], "label": LAB} for L in (512, 1024, 4096)]
boot_d = {"first_token_s": None, "decode_tok_s": None, "label": "not measured yet"}
if boot != "none":
    b = J(boot)
    t = b["decode_run"]["timings"]
    boot_d = {"first_token_s": b["stream"]["first_content_s"], "decode_tok_s": t["decode_tok_s"], "label": "one request each (decode: a 64-token completion), after the server's load"}
long_ = ag["long"]
s = {
    "floor_tok_s": 175.2,
    "decode": dec(b2048), "decode_label": LAB, "decode_untraced": dec(bun), "decode_untraced_label": LAB,
    "context": ag["max_len"], "first_token_s_12": b1["first_token_s_12_token_prompt"]["median"],
    "prefill": [{"n": p["n"], "tok_s": p["loop_read_last"]["median_tok_s"], "label": "median of 3 runs"} for p in b1["prefill"]],
    "prefill_every_step": [{"n": p["n"], "tok_s": p["loop_read_every_step"]["median_tok_s"]} for p in b1["prefill"]],
    "context_sweep": sweep, "boot": boot_d,
    "decode_repeat_in_other_session": dec(b1),
    "agreement": {"floor": {"rel_l2": [x["teacher_forced_rel_l2_mean"] for x in fl], "top1": [x["top1_agree"] for x in fl], "prefix": [x["free_run_match_prefix"] for x in fl]},
                  "chip": {"rel_l2": [p["rel_l2_mean"] for p in ag["prompts"]], "top1": [p["top1"] for p in ag["prompts"]], "prefix": [p["free_run_prefix"] for p in ag["prompts"]]}},
    "long": {"tokens": long_["tokens"], "chip": long_["chip_vs_fp32"], "floor": long_["bf16_cpu_vs_fp32"], "chip_top1_vs_bf16_cpu": long_["chip_vs_bf16_cpu_top1"],
             "rel_l2_by_128_positions": long_["rel_l2_by_128_positions"]},
}
d, u = s["decode"], s["decode_untraced"]
pf = {p["n"]: p["tok_s"] for p in s["prefill"]}
pe = {p["n"]: p["tok_s"] for p in s["prefill_every_step"]}
a = s["agreement"]
ratio = [c / f for c, f in zip(a["chip"]["rel_l2"], a["floor"]["rel_l2"])]
lg = s["long"]
sw = {c["max_len"]: c["mean"] for c in s["context_sweep"]}
s["numbers_note"] = "Load averages of the host during each session are in the evidence files."
s["agreement_note"] = (f"The relative L2 of this package is {min(ratio):.1f} to {max(ratio):.1f} times that of the bf16 CPU run on the four prompts. Top-1 agreement equals "
                       "the bf16 CPU run's on prompts 0, 1 and 3 and is higher on prompt 2 (one position of 24), and the first difference from the reference comes at the "
                       "same token on prompts 0, 1 and 3 and later on prompt 2. Four prompts of 24 tokens do not support a claim about other text.")
s["long_note"] = (f"The sequence is {lg['tokens']} tokens: the model's own card followed by an English project log. Mean relative L2 of this package is {lg['chip']['rel_l2_mean'] / lg['floor']['rel_l2_mean']:.1f} "
                  f"times the bf16 CPU run's, the largest single-position relative L2 is {lg['chip']['rel_l2_max']:.2f} (bf16 CPU run: {lg['floor']['rel_l2_max']:.2f}), and top-1 agreement with fp32 is "
                  f"{lg['chip']['top1']:.3f} against {lg['floor']['top1']:.3f}. One sequence of one kind of text.")
s["prefill_note"] = (f"Prefill is a loop of decode steps, one prompt token per step, with the logits read back after the last prompt token only: {pf[128]:.1f} tok/s for a 128-token prompt, {pf[512]:.1f} for 512 and "
                     f"{pf[1024]:.1f} for 1,024 ({pe[1024]:.1f} when the logits are read at every step). A 1,000-token prompt takes about {1000 / pf[1024]:.0f} seconds before the first token. "
                     "A chunked scan for the prompt is not built.")
if boot != "none":
    lp = b["long_prompt"]
    s["prefill_note"] += f" Through the server, a {lp['prompt_tokens']:,}-token prompt took {lp['timings']['prefill_s']:.1f} seconds to the first token."
s["limitations"] = [
    "One chip with bf16 weights. One request runs at a time; there is no batching and no multi-chip mode.",
    f"The context limit is {s['context']} tokens for prompt plus output. The model supports 131,072.",
    f"Prompts are processed one token per step ({pf[1024]:.0f} tokens per second), so a long prompt takes tens of seconds before the first token.",
    "The Mamba-2 state, the step size and the decay are kept in float32 on the chip. `--ssd-precision bf16` is about 8% faster and less accurate on long sequences "
    "(see evidence/long1024-agree-bf16.json: mean relative L2 0.041 on a 1,024-token sequence against 0.013 with float32).",
    "Tool calls and document inputs of the chat template are refused. Temperature above 0 samples on the host; sampled output was not evaluated.",
    "The server has no sign-in and binds 127.0.0.1 only.",
    f"Decode speed is {d['mean'] / 175.2 * 100:.0f}% of the one-die bf16 floor.",
]
s["not_tested"] = [
    "Answer quality on any task or benchmark. Agreement was measured on four short prompts and one long sequence only.",
    "Several requests arriving together, long runs, and restarts.",
    "Any hardware other than one chip of a Blackhole p300c board, and any tt-metal build other than v0.80.0.",
    "A tt-model build of `tt-model.yaml` (it is a draft that was not built) and the tt-model catalog.",
    "Contexts above the limit.",
]
kt = "The state update and the readout can run as tt-lang kernels written by a local coder model (`--ssd state_full`, `state_full_ro`, `bcast`, `bcast_ro`; they need tt-lang importable). They keep the state in bf16."
if kdir != "none":
    kc, kr = J(Path(kdir) / "kernel-bench-composed.json")["decode"]["mean"], J(Path(kdir) / "kernel-bench-state_full_ro.json")["decode"]["mean"]
    s["kernel_modes"] = {"composed_bf16": kc, "state_full_ro_bf16": kr}
    kt += f" Measured with this package's runtime at a {s['context']}-position context, traced, in one session (one run each): composed with a bf16 state {kc:.1f} tok/s, `state_full_ro` {kr:.1f} tok/s. They are not the default, and their agreement was not re-checked with this runtime."
else:
    kt += " They were not measured with this package's runtime."
s["kernel_modes"] = {**s.get("kernel_modes", {}), "text": kt}
json.dump(s, open(out, "w"), indent=1)
print(json.dumps({k: s[k] for k in ("decode", "decode_untraced", "context", "first_token_s_12", "context_sweep")}, indent=1)[:900])
