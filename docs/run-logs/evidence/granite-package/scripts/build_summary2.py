"""Build evidence/summary.json (version 0.2.0) from the raw result files.
usage: python build_summary2.py OUT.json [S3_DIR]
Each session is reported on its own; nothing is averaged across sessions. S3_DIR (optional) holds bench-fast.json, bench-untraced.json, smoke.json."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
J = lambda p: json.load(open(R / p))  # noqa: E731
dec = lambda d: {k: d["decode"][k] for k in ("mean", "min", "max")}  # noqa: E731
LAB = "5 repeats after 1 warm-up, one process"


def load_at_start(p):
    lines = [ln for ln in open(R / p) if ln.startswith("bench start ") or ln.startswith("start ")]
    first = (lines[-1] if lines[-1].startswith("bench") else lines[0]).replace("bench ", "").split()      # the bench's own start line when there is one
    return float(first[3]), first[1][11:16]


def srv(b):
    return {"first_token_s": b["stream"]["first_content_s"], "decode_tok_s": b["decode_run"]["timings"]["decode_tok_s"],
            "long_prompt_tokens": b["long_prompt"]["prompt_tokens"], "long_prompt_first_token_s": b["long_prompt"]["timings"]["prefill_s"]}


def pf(b):
    return [(p["n"], p["loop_read_last"]["median_tok_s"]) for p in b["prefill"]]


sessions = []
l1, t1 = load_at_start("f1/conditions.txt")
sessions.append({"name": "session 1", "when": f"{t1} Pacific, host load {l1:.1f} (0.1.0 prefill)", "load": l1, "prompt_tokens": 12,
                 "decode": dec(J("f2/bench-L2048.json")), "decode_untraced": dec(J("f2/bench-L2048-untraced.json")),
                 "ttft_s": J("f1/bench-fp32.json")["first_token_s_12_token_prompt"]["median"], "prefill": pf(J("f1/bench-fp32.json")),
                 "server": srv(J("bootA2/smoke.json")), "evidence": "`evidence/bench-L2048.json`, `evidence/bench-L2048-untraced.json`, `evidence/bench-fp32-L2048-prefill.json`",
                 "server_evidence": "`evidence/server-smoke.json`"})
l2, t2 = load_at_start("s1/conditions.txt")
sessions.append({"name": "session 2", "when": f"{t2} Pacific, host load {l2:.1f}", "load": l2, "prompt_tokens": 12,
                 "decode": dec(J("s1/bench-fast.json")), "decode_untraced": dec(J("s1/bench-untraced.json")),
                 "ttft_s": J("s1/bench-fast.json")["first_token_s_12_token_prompt"]["median"], "prefill": pf(J("s1/bench-fast.json")), "server": None,
                 "evidence": "`evidence/s2-bench-fast.json`, `evidence/s2-bench-untraced.json`", "server_evidence": ""})
if len(sys.argv) > 2:
    d = sys.argv[2]
    l3, t3 = load_at_start(f"{d}/conditions.txt")
    sessions.append({"name": "session 3", "when": f"{t3} Pacific, host load {l3:.1f}", "load": l3, "prompt_tokens": 12,
                     "decode": dec(J(f"{d}/bench-fast.json")), "decode_untraced": None,       # the untraced run of this session ran after the lease was reaped and is discarded
                     "ttft_s": J(f"{d}/bench-fast.json")["first_token_s_12_token_prompt"]["median"], "prefill": pf(J(f"{d}/bench-fast.json")),
                     "server": srv(J(f"{d}/smoke.json")) if (R / d / "smoke.json").exists() else None,
                     "evidence": "`evidence/s3-bench-fast.json`", "server_evidence": "`evidence/s3-server-smoke.json`"})
floor = J("../ibm-granite--granite-4.0-h-1b/reference/bf16_cpu_control.json")
prose = J("s1/agree-prose.json")
long_ = []
for kind, f in (("prose (the model card and an English project log)", "s1/agree-prose.json"), ("code (Python source files)", "s1/agree-code.json"),
                ("non-English (French, Spanish, German, Italian)", "s1/agree-multi.json")):
    l = J(f)["long"]
    long_.append({"kind": kind, "tokens": l["tokens"], "chip": l["chip_vs_fp32"], "floor": l["bf16_cpu_vs_fp32"], "first_1024": l.get("first_1024_positions")})
nohead = J("s1/bench-nofast.json")["prefill"][0]["loop_read_last"]["median_tok_s"]
fast512 = [v for n, v in sessions[1]["prefill"] if n == 512][0]
best = min(sessions, key=lambda x: x["load"])
h = best
pf_best = dict(h["prefill"])
s = {"version": "0.2.0", "floor_tok_s": 175.2, "context": 2048, "sessions": sessions,
     "agreement_short": {"floor": {"rel_l2": [x["teacher_forced_rel_l2_mean"] for x in floor], "top1": [x["top1_agree"] for x in floor], "prefix": [x["free_run_match_prefix"] for x in floor]},
                         "chip": {"rel_l2": [p["rel_l2_mean"] for p in prose["prompts"]], "top1": [p["top1"] for p in prose["prompts"]], "prefix": [p["free_run_prefix"] for p in prose["prompts"]]}},
     "agreement_long": long_, "context_sweep": [{"max_len": L, "mean": J(f"f2/bench-L{L}.json")["decode"]["mean"]} for L in (512, 1024, 2048, 4096)]}
a = s["agreement_short"]
ratio = [c / f for c, f in zip(a["chip"]["rel_l2"], a["floor"]["rel_l2"])]
s["numbers_note"] = "Load averages of the host during each session are in the evidence files."
spread = [x["decode"]["mean"] for x in sessions]
untraced_n = [x for x in sessions if x["decode_untraced"]]
s["sessions_note"] = (f"The same traced decode measured {min(spread):.1f} to {max(spread):.1f} tok/s across the sessions. The host runs other jobs, and the traced step still depends on "
                      "the host for the per-token input writes and the logits read, so a loaded host lowers the number. Prefill in session 1 ran the 0.1.0 code (the full program for every prompt token); "
                      f"sessions 2 and 3 run the 0.2.0 prefill, which replays a second trace that stops after the last layer (same logits; 512-token prompt: {fast512:.1f} tok/s with it against {nohead:.1f} without it, same session).")
s["agreement_note"] = (f"The relative L2 of this package is {min(ratio):.1f} to {max(ratio):.1f} times that of the bf16 CPU run on the four prompts. Top-1 agreement equals "
                       "the bf16 CPU run's on prompts 0, 1 and 3 and is higher on prompt 2 (one position of 24), and the first difference from the reference comes at the "
                       "same token on prompts 0, 1 and 3 and later on prompt 2. Four prompts of 24 tokens do not support a claim about other text.")
rs = [t["chip"]["rel_l2_mean"] / t["floor"]["rel_l2_mean"] for t in long_]
s["long_note"] = (f"Mean relative L2 of this package is {min(rs):.1f} to {max(rs):.1f} times the bf16 CPU run's on these sequences, and top-1 agreement with fp32 is at or above the bf16 CPU run's on all three. "
                  "The largest single-position relative L2 is higher than the bf16 CPU run's in all three (see the table). The code sequence has one position with a relative L2 of 0.62. "
                  "The non-English passage is 653 tokens (a short passage). One sequence of each kind, one run: this does not describe other text.")
s["prefill_note"] = (f"Prefill is a loop of decode steps, one prompt token per step, with the logits read back after the last prompt token only. It runs at {pf_best[128]:.1f} tok/s for a 128-token prompt, "
                     f"{pf_best[512]:.1f} for 512 and {pf_best[1024]:.1f} for 1,024 in {h['name']}. A 1,500-token prompt takes about {1500 / pf_best[1024]:.0f} seconds before the first token. "
                     "A chunked scan for the prompt is not built.")
s["limits"] = [
    f"Prefill is a loop of decode steps: about {pf_best[1024]:.0f} tok/s, so a 1,500-token prompt takes about {1500 / pf_best[1024]:.0f} s before the first token.",
    "One request at a time (a second one gets 503). No batching and no multi-chip mode.",
    "Greedy decoding is what was measured. A temperature above 0 samples on the host; sampled output was not evaluated.",
    "Weights are bf16 on one chip. The Mamba-2 state, the step size and the decay are kept in float32 (`--ssd-precision bf16` is about 8% faster and less accurate on long sequences: "
    "mean relative L2 0.041 on a 1,024-token sequence against 0.013 with float32, `evidence/long1024-agree-bf16.json`).",
    f"The context limit is {s['context']} tokens for prompt plus output (default). The model supports 131,072.",
    "No tool calls and no document inputs (the server refuses them with 400).",
    "The kernel modes are optional, need a bf16 state and the tt-lang pair (tt-lang-light 1.1.10.dev20261026 with tt-metal 0e9d200), and are not the default.",
    "Tested only on a Blackhole p300c board with tt-metal v0.80.0.",
    f"Traced decode is {h['decode']['mean'] / 175.2 * 100:.0f}% of the one-die bf16 floor in {h['name']}.",
    "The server has no sign-in and binds 127.0.0.1 by default.",
]
s["not_tested"] = [
    "Answer quality on any task or benchmark. Agreement was measured on four short prompts and three long sequences only.",
    "Several requests arriving together beyond the 503 behaviour, long runs, and restarts.",
    "Any hardware other than one chip of a Blackhole p300c board, and any tt-metal build other than v0.80.0.",
    "A tt-model build of `tt-model.yaml` (it is a draft that was not built) and the tt-model catalog.",
    "Contexts above the limit, and a CPU-only torch wheel in place of the default one.",
]
k = {}
try:
    k = {"composed": J("kb/kernel-bench-composed.json")["decode"]["mean"], "state_full_ro": J("kb/kernel-bench-state_full_ro.json")["decode"]["mean"]}
except FileNotFoundError:
    pass
kt = ("The state update and the readout can run as tt-lang kernels written by a local coder model (`--ssd state_full`, `state_full_ro`, `bcast`, `bcast_ro`; they need tt-lang importable). "
      "They keep the state in bf16.")
if k:
    kt += (f" Measured with this package's runtime at a {s['context']}-position context, traced, in one session (one run each): composed with a bf16 state {k['composed']:.1f} tok/s, "
           f"`state_full_ro` {k['state_full_ro']:.1f} tok/s. They are not the default.")
try:
    ka, kb = J("final/agree-kernel.json")["long"], J("final/agree-composed-bf16.json")["long"]
    kt += (f" Agreement over the prose sequence (teacher forced against fp32, mean relative L2, top-1): `state_full_ro` with a bf16 state {ka['chip_vs_fp32']['rel_l2_mean']:.3f}, {ka['chip_vs_fp32']['top1']:.3f} "
           f"over 2,048 tokens and {ka['first_1024_positions']['chip_vs_fp32']['rel_l2_mean']:.3f}, {ka['first_1024_positions']['chip_vs_fp32']['top1']:.3f} over the first 1,024; composed with a bf16 state {kb['chip_vs_fp32']['rel_l2_mean']:.3f}, "
           f"{kb['chip_vs_fp32']['top1']:.3f} and {kb['first_1024_positions']['chip_vs_fp32']['rel_l2_mean']:.3f}, {kb['first_1024_positions']['chip_vs_fp32']['top1']:.3f}; the default (float32 state) 0.017, 0.984 and 0.013, 0.978. "
           "The kernel mode has the same error as the bf16 composed path and three to four times the default's.")
    k["agreement_prose_2048"] = {"state_full_ro_bf16": ka["chip_vs_fp32"], "composed_bf16": kb["chip_vs_fp32"]}
except FileNotFoundError:
    pass
s["kernel_modes"] = {**k, "text": kt}
s["changelog"] = [
    "Server: a stop string as long as the output so far no longer leaks part of itself (a negative slice).",
    "Server: a streamed request with a prompt longer than the context gets a JSON 400 (it got no response before).",
    "Server: streamed responses are written as tokens are produced (they were written after the whole generation).",
    "Server: a second request while one runs gets 503 with Retry-After at once; a client that disconnects mid-stream frees the slot; idle connections time out after 30 seconds; `HOST` is documented (no sign-in).",
    "Server: the ready line and `tools/evaluate.py` record the precision the model actually runs with (kernel modes force bf16).",
    "Runtime: prefill replays a second trace that stops after the last layer (no final norm, no lm_head); same logits, small gain.",
    "`bash run.sh --self-test`: one line per check (python, packages, `import ttnn`, chips, pinned weights, disk, port) without opening the device.",
    "Float32 state stays the default (as in 0.1.0). Requirements are pinned to the tested ranges, including ttnn's own dependencies.",
    "Package: `tt-model.yaml` checks (non-mapping root, ready marker), `SHA256SUMS`, `SECURITY.md`, this changelog, machine-readable `measured` rows, a Limits section, two or more measurement sessions, agreement on three kinds of text.",
]
json.dump(s, open(sys.argv[1], "w"), indent=1)
print([(x["name"], x["when"], round(x["decode"]["mean"], 1), (round(x["decode_untraced"]["mean"], 1) if x["decode_untraced"] else None), round(x["ttft_s"], 3)) for x in sessions])
