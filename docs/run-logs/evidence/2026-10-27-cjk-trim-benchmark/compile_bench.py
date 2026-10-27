"""Compile the CJK-Trim benchmark into one path-free benchmark.json. usage: compile_bench.py BENCH_DIR OUT.json CHECKOUT"""
import glob, json, math, sys
from pathlib import Path
B, OUT, CHECKOUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, CHECKOUT)
from orchard.bench_rerun import read_samples  # newline-only split

KEYS = {"trim": "full-trim", "base": "full-base"}


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def paired(a, b):
    ids = sorted(set(a) & set(b))
    w = sum(1 for i in ids if a[i] > b[i]); l = sum(1 for i in ids if a[i] < b[i])
    return {"items": len(ids), "trim_only_right": w, "base_only_right": l, "sign_test_p": float(f"{sign_p(w, l):.4g}")}


tasks, cases = {}, {}
for m, d in KEYS.items():
    # GSM8K: flexible-extract, first 100 documents
    rows = {r["doc_id"]: float(r["exact_match"]) for r in read_samples(str(B / d / "gsm8k"), "gsm8k") if r.get("filter") == "flexible-extract"}
    cases[("gsm8k", m)] = rows
    s = json.load(open(B / d / "gsm8k" / "summary.json"))["phases"][0]
    n = len(rows); p = int(sum(rows.values()))
    tasks.setdefault("gsm8k", {})[m] = {"n": n, "passed": p, "score": round(p / n, 4), "stderr": round(math.sqrt(p / n * (1 - p / n) / n), 4),
                                         "cut_off": s["proxy"]["cut_off"], "requests": s["proxy"]["requests"], "errors": s["proxy"]["errors"], "wall_s": round(s["wall_s"])}
    for sess, tag in (("bfcl-c1", "c1"), ("bfcl-c4", "c4")):
        ph = json.load(open(B / d / sess / "summary.json"))["phases"][0]
        for kind in ("simple", "multiple"):
            v = json.load(open(B / d / sess / "bfcl" / f"bfcl-{kind}.json"))
            cases[(f"bfcl_{kind}_{tag}", m)] = {i["id"]: 1.0 if i["passed"] else 0.0 for i in v["items"]}
            tasks.setdefault(f"bfcl_{kind}_{tag}", {})[m] = {"n": v["n"], "passed": v["passed"], "score": round(v["score"], 4), "stderr": round(v["stderr"], 4),
                                                              "request_errors": v["errors"], "cut_off": v["cut_off"], "wall_s": round(ph["wall_s"])}
for t in tasks:
    tasks[t]["paired"] = paired(cases[(t, "trim")], cases[(t, "base")])
cjk = json.load(open(B / "cjk-compare.json"))
tok = json.load(open(B / "tok" / "tokenizer-check.json"))
speed = {}
for m, d in KEYS.items():
    sp = json.load(open(B / d / "speed.json"))
    speed[m] = {k: {"decode_tok_s": round(v["summary"]["decode_tok_s"]["median"], 1), "ttft_s": round(v["summary"]["ttft_s"]["median"], 2),
                    "completion_tokens": v["summary"]["completion_tokens"]["median"], "prompt_tokens": v["runs"][0]["prompt_tokens"]} for k, v in sp["prompts"].items()}
out = {"models": {"trim": "ProprietaryLegal/Qwen3.8-27B-CJK-Trim@e5476cfa7df694f10fa9d00494f08e51010327b5",
                  "base": "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
       "conditions": ("small sets: GSM8K first 100 items, BFCL simple and multiple first 100 items at 4 requests at a time and first 50 at 1 at a time; "
                      "IFEval, HumanEval and MBPP were not run; 2-chip bundle qwen3.8-27b-dflash2-p300 with the drafter off and sampling on the host for both models; "
                      "thinking as the chat template defaults it, temperature 0, 32768 tokens, stop string '</s>' for GSM8K; lm-evaluation-harness 0.4.13; "
                      "BFCL v3 revision 61fc0608cfd831fcfbbaa676ebdfef0ed963eeda; the two models ran at the same time on the two boards"),
       "tasks": tasks, "cjk_check": cjk, "tokenizer": tok, "speed": speed}
json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
for t, v in tasks.items():
    print(t, {m: (x["passed"], x["n"], x["score"]) for m, x in v.items() if m != "paired"}, v["paired"])
