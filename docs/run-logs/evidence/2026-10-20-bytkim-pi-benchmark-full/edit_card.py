import json, re, shutil
from pathlib import Path
RUN = Path.home()/"orchard-runs/bytkim--qwen3.8-27b-pi"
card_path = RUN/"stages/7/package/qwen3.8-27b-pi-p300/README.md"
SP = Path("/tmp/claude-1000/-home-ttuser-code-tt-orchard/2af874b9-69cb-4ce3-bb93-e834216cb0cf/scratchpad")
if not (SP/"README.before-benchmark.md").exists():
    shutil.copy(card_path, SP/"README.before-benchmark.md")
card = (SP/"README.before-benchmark.md").read_text()
B = json.load(open(RUN/"packaging/benchmark.json"))["tasks"]
EV = "`packaging/benchmark.json`"
def v(x): return str(x)
rows = []
def add(name, val, unit="fraction"): rows.append(f"| {name} | {val} {unit} | measured | {EV} |")
spec = [("humaneval", "HumanEval pass@1, all 164 problems, zero-shot, code in a python block"),
        ("mbpp", "MBPP pass@1, all 500 problems, zero-shot"),
        ("gsm8k", "GSM8K, all 1,319 problems, 5-shot chat turns, flexible-extract answer match"),
        ("ifeval", "IFEval prompt-level strict, all 541 prompts"),
        ("bfcl_simple", "BFCL v3 `simple` tool calls, all 400 items, scored by function name and argument values"),
        ("bfcl_multiple", "BFCL v3 `multiple` tool calls, all 200 items")]
for key, label in spec:
    for who, nm in (("pi", "this model"), ("base", "base model Qwen/Qwen3.8-27B")):
        t = B[key][who]
        add(f"{label}: {nm} (standard error {v(t['stderr'])})", v(t["score"]))
for who, nm in (("pi", "this model"), ("base", "base model")):
    t = B["gsm8k_strict_format"][who]
    add(f"GSM8K strict-match, which also needs the `####` answer format: {nm} (standard error {v(t['stderr'])})", v(t["score"]))
anchor = "| decode speed over the base's, `chat` prompt"
i = card.index(anchor); j = card.index("\n", i)
card = card[:j+1] + "\n".join(rows) + "\n" + card[j+1:]

def sect(name, body):
    global card
    m = re.search(rf"^## {re.escape(name)}\n.*?(?=^## |\Z)", card, re.S | re.M)
    if m is None:
        k = card.index("## Limitations"); card = card[:k] + f"## {name}\n\n{body.strip()}\n\n" + card[k:]; return
    card = card[:m.start()] + f"## {name}\n\n{body.strip()}\n\n" + card[m.end():]

p = {k: B[k]["paired"] for k in ("humaneval", "mbpp", "gsm8k", "ifeval")}
def pair(k, label):
    d = p[k]; pv = "below 0.001" if d["sign_test_p"] < 0.001 else str(d["sign_test_p"])
    return f"- {label}: this model alone was right on {d['pi_only_right']} items, the base alone on {d['base_only_right']} (sign test p {pv}, {d['items']} items)."
cut = {k: (B[k]["pi"]["cut_off"], B[k]["base"]["cut_off"]) for k in ("humaneval", "mbpp", "ifeval")}
sect("Benchmark against the base model", f"""
The two models were served one after the other in one session on one 2-chip board, through the same bundle (bfp8 weights, the DFlash2 drafter on, sampling on the device), with the same `run.sh` apart from the model path. Every problem of each set was asked once at temperature 0, with the chat template's default thinking mode on and a limit of 8192 generated tokens per reply, four requests at a time. The harness is lm-evaluation-harness 0.4.13 for HumanEval, MBPP, GSM8K and IFEval (the HumanEval and MBPP tasks are chat versions written in the tt-orchard repository, because the harness's own instruct versions score a chat reply as empty). The BFCL scorer is also in tt-orchard and uses the Apache-2.0 BFCL v3 data at revision `61fc0608cfd8`. The Expected performance table lists the scores and `packaging/benchmark.json` holds all numbers, counts and conditions.

The fine-tune scores at or above the base on every set. The differences are small. Paired by problem, where the harness kept each answer:
{pair('humaneval', 'HumanEval')}
{pair('mbpp', 'MBPP')}
{pair('gsm8k', 'GSM8K')}
{pair('ifeval', 'IFEval')}
The BFCL runs saved counts only, so there is no paired test for them. Their scores differ by less than one standard error on `simple` and by about one on `multiple`.

Replies that ran out of the 8192-token limit while thinking are counted as wrong. For this model and the base: HumanEval {cut['humaneval'][0]} and {cut['humaneval'][1]} of 164, MBPP {cut['mbpp'][0]} and {cut['mbpp'][1]} of 500, IFEval {cut['ifeval'][0]} and {cut['ifeval'][1]} of 541, and none on GSM8K or BFCL. These caps lower both models' scores, and most of the failures on HumanEval and MBPP are such replies. A higher limit would raise both and might change the gap.
""")
sect("Limitations", """
- Tested on 2 chips (mesh P150x2) only; other configurations are not claimed by this bundle.
- Accuracy was compared with the CPU reference on one short fixed prompt only (32 tokens).
- The benchmark uses this package's settings (bfp8 weights, drafter on, sampling on the device, thinking on, a limit of 8192 generated tokens). It differs from the model author's setup, so its scores need not match theirs.
- HumanEval, MBPP, GSM8K and IFEval are public sets and may be in either model's training data.
- Speed was measured on one board in one session, one request at a time, greedy decoding, three repetitions per prompt. No concurrent load, long-context prefill or sampling with a temperature was timed. The benchmark ran four requests at once, and its wall times depend on the text each model wrote, so they are not a speed comparison.
- The first start of this package on an empty tensor cache (the Expected performance table) took much longer than stage 2's start of the same weights. The two starts differ in more than the package, so the table does not say why. A start on a warm tensor cache is faster; its time was measured once and is not listed here.
""")
sect("Risks and safety considerations", """
The checks above cover code, tool calls, arithmetic and instruction following on fixed sets. They do not cover long outputs beyond the length limit, safety behaviour or agent sessions of many turns. Output can differ from bytkim/Qwen3.8-27B-pi run on a CPU or GPU in ways they do not show.
""")
sect("Not measured", """
- safety behaviour and open-ended output quality of this fine-tune
- the benchmark claims on the model author's card (charts of Pi-harness completion by reasoning effort): they were not read and are not verified here
- scores with thinking off, with a higher generation limit, or with a temperature above zero
- MMLU-Pro, long-context retrieval, and agent tasks of many turns
- decode speed at several input lengths and several concurrent requests
- a download of the weights through `tt-model pull`, and a boot of the package from the Hub
""")
card = card.replace("Its acceptance rate on this model is listed under Not measured.", "Its acceptance rate during the benchmark is in `packaging/benchmark.json`: between 0.46 and 0.60 of drafted tokens, by task, for this model, against 0.46 to 0.56 for the base.")
card_path.write_text(card)
print(len(card))
