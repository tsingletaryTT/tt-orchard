"""Update the card for the 32768-token re-run. Reads the saved pre-edit card, so it can be run again; every replacement must match exactly once."""
import json, shutil
from pathlib import Path
RUN = Path.home() / "orchard-runs/bytkim--qwen3.8-27b-pi"
card_path = RUN / "stages/7/package/qwen3.8-27b-pi-p300/README.md"
SP = Path("/tmp/claude-1000/-home-ttuser-code-tt-orchard/2af874b9-69cb-4ce3-bb93-e834216cb0cf/scratchpad")
if not (SP / "README.before-rerun.md").exists():
    shutil.copy(card_path, SP / "README.before-rerun.md")
card = (SP / "README.before-rerun.md").read_text()
R = json.load(open(RUN / "packaging/benchmark.json"))["rerun_32k"]["tasks"]

def once(old, new):
    global card
    assert card.count(old) == 1, old[:70]
    card = card.replace(old, new)

# 1. table rows, after the IFEval base row
LABEL = {"humaneval": "HumanEval pass@1, all 164 problems", "mbpp": "MBPP pass@1, all 500 problems", "ifeval": "IFEval prompt-level strict, all 541 prompts"}
rows = []
for k in ("humaneval", "mbpp", "ifeval"):
    for who, nm in (("pi", "this model"), ("base", "base model Qwen/Qwen3.8-27B")):
        t = R[k][who]
        rows.append(f"| {LABEL[k]}, empty replies asked again with a limit of 32768 generated tokens: {nm} (standard error {t['stderr']}) | {t['score']} fraction | measured | `packaging/benchmark.json` |")
anchor = "| IFEval prompt-level strict, all 541 prompts: base model Qwen/Qwen3.8-27B (standard error 0.015) | 0.85 fraction | measured | `packaging/benchmark.json` |\n"
once(anchor, anchor + "\n".join(rows) + "\n")

# 2. the benchmark section
p = {k: R[k]["paired"] for k in R}
def pv(d): return "below 0.001" if d["sign_test_p"] < 0.001 else str(d["sign_test_p"])
def line(k, label): d = p[k]; return f"- {label}: this model alone was right on {d['pi_only_right']} items, the base alone on {d['base_only_right']} (sign test p {pv(d)}, {d['items']} items)."
once("The fine-tune scores at or above the base on every set. The differences are small. Paired by problem, where the harness kept each answer:",
     "The fine-tune scores at or above the base on every set at this limit. Paired by problem, where the harness kept each answer, the GSM8K difference is clear and the others are smaller:")
once("A higher limit would raise both and might change the gap.",
     "A higher limit raises both, and it changes the gap on MBPP.")
once("\n## Limitations",
f"""
### Asking the cut-off problems again with a higher limit

Every problem whose reply came back empty at the 8192-token limit was asked again with a limit of 32768 generated tokens, under the same settings, for both models. These were {R['humaneval']['pi']['rerun_items']} and {R['humaneval']['base']['rerun_items']} problems of HumanEval, {R['mbpp']['pi']['rerun_items']} and {R['mbpp']['base']['rerun_items']} of MBPP and {R['ifeval']['pi']['rerun_items']} and {R['ifeval']['base']['rerun_items']} of IFEval, for this model and the base. A problem whose re-run did not help keeps its first result, and no problem got worse. GSM8K and BFCL had no cut-off replies and were not asked again. The Expected performance table lists the new scores, and `packaging/benchmark.json` holds the counts.

- HumanEval: this model {R['humaneval']['pi']['score_before']} to {R['humaneval']['pi']['score']}, the base {R['humaneval']['base']['score_before']} to {R['humaneval']['base']['score']}.
- MBPP: this model {R['mbpp']['pi']['score_before']} to {R['mbpp']['pi']['score']}, the base {R['mbpp']['base']['score_before']} to {R['mbpp']['base']['score']}.
- IFEval: this model {R['ifeval']['pi']['score_before']} to {R['ifeval']['pi']['score']}, the base {R['ifeval']['base']['score_before']} to {R['ifeval']['base']['score']}.

With the higher limit the two models are close on all three sets, and the paired tests do not separate them:
{line('humaneval', 'HumanEval')}
{line('mbpp', 'MBPP')}
{line('ifeval', 'IFEval')}
The MBPP difference was clear at 8192 tokens and is not clear at 32768 tokens. Some replies still ran out at 32768 tokens: this model {R['humaneval']['pi']['still_cut_off_at_32768']}, {R['mbpp']['pi']['still_cut_off_at_32768']} and {R['ifeval']['pi']['still_cut_off_at_32768']} problems on the three sets, the base {R['humaneval']['base']['still_cut_off_at_32768']}, {R['mbpp']['base']['still_cut_off_at_32768']} and {R['ifeval']['base']['still_cut_off_at_32768']}. The list of problems to ask again was the empty replies. A reply cut off after its answer had begun has text, so it was not in the list. The harness's request log has no problem ids, so those could not be found. The count differs from the 8192-token cut-off counts by one for three of the six model and set pairs.

## Limitations""")

# 3. limitations and not measured
once("a limit of 8192 generated tokens). It differs from the model author's setup",
     "a limit of 8192 generated tokens, and 32768 for the problems asked again). It differs from the model author's setup")
once("- scores with thinking off, with a higher generation limit, or with a temperature above zero",
     "- scores with thinking off, with a limit above 32768 generated tokens, or with a temperature above zero")

# 4. intended use
once("Text generation with bytkim/Qwen3.8-27B-pi through the OpenAI-compatible server that `tt-model serve` starts.",
"""Text generation with bytkim/Qwen3.8-27B-pi through the OpenAI-compatible server that `tt-model serve` starts.

The benchmark below supports these uses on a 2-chip board: writing and fixing short Python functions, step-by-step arithmetic and word problems, following written instructions, and tool calls with structured arguments. It is a fine-tune of Qwen/Qwen3.8-27B that runs on the same bundle at the same cost. It scores at or above the base on every set measured. The one clear difference is on GSM8K. On code, instruction following and tool calls the two models are close once replies are given enough room to finish. Choose it over the base when you want the same setup with the small gain on arithmetic and word problems. Nothing here compares it with other models, and nothing covers work on real repositories or agent sessions of many turns.""")
card_path.write_text(card)
print(len(card))
