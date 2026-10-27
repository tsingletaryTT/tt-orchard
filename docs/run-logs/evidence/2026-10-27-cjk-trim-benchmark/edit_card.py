"""Adds the benchmark text to the package card that `tt-orchard package-card` wrote. Every number comes from benchmark.json and the CJK files.
usage: edit_card.py CARD_README BENCH_DIR SCRATCH_DIR   (the card as package-card wrote it is kept in SCRATCH_DIR so the script can run again)"""
import json, re, shutil, sys
from pathlib import Path
card_path, B, SP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
keep = SP / "README.before-edit.md"
if not keep.exists():
    shutil.copy(card_path, keep)
card = keep.read_text()
bm = json.load(open(B / "benchmark.json"))
T, C, S = bm["tasks"], bm["cjk_check"], bm["tokenizer"]["sets"]
trim = json.load(open(B / "full-trim/cjk.json"))["prompts"]
base = json.load(open(B / "full-base/cjk.json"))["prompts"]
n = len(trim)
script_ok = lambda rs, langs=("zh", "ja", "ko", "mx"): sum(1 for r in rs if r["lang"] in langs and r["expected_script_ok"])
formed = lambda rs: sum(1 for r in rs if r["well_formed"])
cut = lambda rs: sum(1 for r in rs if r.get("finish") == "length")
cjk_chars_trim = sum(1 for r in trim for ch in (r.get("reply") or "") if 0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF or 0xAC00 <= ord(ch) <= 0xD7AF)
sp = bm["speed"]
dec = lambda m, k: sp[m][k]["decode_tok_s"]


def sect(name, body, before="## Limitations"):
    global card
    m = re.search(rf"^## {re.escape(name)}\n.*?(?=^## |\Z)", card, re.S | re.M)
    text = f"## {name}\n\n{body.strip()}\n\n"
    if m is None:
        k = card.index(before)
        card = card[:k] + text + card[k:]
    else:
        card = card[:m.start()] + text + card[m.end():]


def add_rows():
    global card
    ev = "`stages/7/benchmark/benchmark.json`"
    rows = []
    for key, label in (("chat", "short chat prompt"), ("length-12000w", "long prompt")):
        for m, nm in (("trim", "this model"), ("base", "nearest model, same conditions")):
            rows.append(f"| decode speed, {label} ({sp[m][key]['prompt_tokens']} prompt tokens), one request, {nm} | {sp[m][key]['decode_tok_s']} tokens/s | measured | {ev} |")
            rows.append(f"| time to first token, {label} ({sp[m][key]['prompt_tokens']} prompt tokens), one request, {nm} | {sp[m][key]['ttft_s']} s | measured | {ev} |")
    anchor = "| bfcl_multiple_c4 paired sign test p, this model against the nearest model"
    i = card.index(anchor); j = card.index("\n", i)
    card = card[:j + 1] + "\n".join(rows) + "\n" + card[j + 1:]


def t(task, who):
    return T[task][who]


add_rows()
sect("Benchmark against the base model, and what the vocabulary trim changes", f"""
This card has a small benchmark. It was run to find out what removing 66,560 vocabulary entries changes, and its sets are small. Not run: IFEval,
HumanEval, MBPP and any long-context or agent task. Both models were served at the same time on the two boards of one machine, through the same
2-chip bundle with the drafter off and sampling on the host (the base model's `run.sh` differs from this model's only in the model directory). Thinking
was on as the chat template sets it, temperature 0, one run per item.

**English and code, small sets: no difference was found.** GSM8K, the first 100 items: this model {t('gsm8k','trim')['passed']} right, the base {t('gsm8k','base')['passed']}
(one item only this model got right). BFCL tool calls, at one request at a time: `simple`, first 50 items, {t('bfcl_simple_c1','trim')['passed']} against
{t('bfcl_simple_c1','base')['passed']}; `multiple`, first 50 items, {t('bfcl_multiple_c1','trim')['passed']} against {t('bfcl_multiple_c1','base')['passed']}. With four requests at a time
the BFCL scores of both models drop: `simple`, first 100 items, {t('bfcl_simple_c4','trim')['passed']} against {t('bfcl_simple_c4','base')['passed']}; `multiple`, first 100 items,
{t('bfcl_multiple_c4','trim')['passed']} against {t('bfcl_multiple_c4','base')['passed']} (paired sign test p {t('bfcl_multiple_c4','paired')['sign_test_p']}, not a clear difference). This server configuration
(drafter off, sampling on the host) loses tool-call accuracy when four requests run at once, for the base model too; use one request at a time for tool calling.
With 50 to 100 items per set a gap of a few items is inside the noise.

**Chinese, Japanese and Korean input does not work.** The tokenizer of this model has no entry for most CJK characters, so it splits them into short byte pieces,
and the model does not read those pieces as the characters they stand for (what the check below shows). On {S['zh']['strings']} Chinese, {S['ja']['strings']} Japanese and {S['ko']['strings']} Korean paragraphs from Wikipedia this model's tokenizer produced
{S['zh']['ratio_total']} times, {S['ja']['ratio_total']} times and {S['ko']['ratio_total']} times as many tokens as the base model's. On English and code strings (500 each) the pieces were identical.
In a check of {n} prompts (10 each in Chinese, Japanese and Korean, and 10 mixed with English), thinking on, 8192 tokens, the base model replied in the asked script on
{script_ok(base)} of {n} prompts. This model did on {script_ok(trim)} of {n}: it wrote {cjk_chars_trim} CJK characters in total. For most prompts it answered in English or Russian that the message looked
like garbled characters; {cut(trim)} of {n} replies ran into the 8192-token limit with no answer (the base model: {cut(base)}). Only {formed(trim)} of {n} replies were well formed (finished,
not empty, no repeated characters), against {formed(base)} for the base model. Do not use this model for CJK text. Mixed prompts that need a CJK word in the answer fail the same way.

**Speed.** Decode speed with one request at a time, thinking on, median of two runs, is in the Expected performance table for a short chat prompt and for a prompt of
about {sp['trim']['length-12000w']['prompt_tokens']} tokens, for both models. The two models ran at the same time on the two boards, so both include the same neighbour load. The sample is small:
decode speed differs by a few percent on the short prompts, and the long prompt is the one row with a larger gap.
""")

card = card.replace("- Accuracy was compared with the CPU reference on one short fixed prompt only.",
                    "- Accuracy was compared with the CPU reference on one short fixed prompt, and against the base model on small English and tool-call sets (see the benchmark section). IFEval, HumanEval, MBPP and long-context tasks were not run.")
card = card.replace("- Sampling runs on the host.",
                    "- Sampling runs on the host.\n- Chinese, Japanese and Korean text does not work: the model cannot read it and does not write it (see the benchmark section). Text in those languages costs about four times as many tokens as with the base model's tokenizer.\n- With four requests at a time the BFCL tool-call scores of this configuration drop for both models; one request at a time is advised for tool calling.")
if "## Not measured" in card:
    card = card.replace("## Not measured\n\n", "## Not measured\n\n- IFEval, HumanEval, MBPP, long-context behaviour and agent tasks, and any set larger than the sizes named in the benchmark section\n- accuracy on English and code beyond those small sets, so a loss of a few points would not show\n", 1)
card_path.write_text(card)
print("edited", card_path)
