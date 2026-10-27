"""Token counts of the trim's tokenizer against the base's, on about 500 strings each of English, code, Chinese, Japanese and Korean.
Text: the first rows of the 20231101 Wikipedia dump in each language (read through the datasets-server API, so nothing large is downloaded)
cut into paragraphs of 40 to 400 characters, and Python source files from this repository cut into chunks of 4 to 25 lines. Fixed order, no sampling.
usage: tokenizer_check.py OUT.json CHECKOUT_DIR"""
import json, sys, glob, urllib.request, statistics
from tokenizers import Tokenizer
TRIM = "<HOME>/.cache/huggingface/hub/models--ProprietaryLegal--Qwen3.8-27B-CJK-Trim/snapshots/e5476cfa7df694f10fa9d00494f08e51010327b5/tokenizer.json"
BASE = "<HOME>/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json"
OUT = sys.argv[1]


def wiki(lang, n=500):
    out = []
    off = 0
    while len(out) < n and off < 600:
        u = f"https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config=20231101.{lang}&split=train&offset={off}&length=100"
        rows = json.load(urllib.request.urlopen(u, timeout=60))["rows"]
        for r in rows:
            for p in r["row"]["text"].split("\n"):
                if 40 <= len(p) <= 400:
                    out.append(p)
        off += 100
    return out[:n]


def code(n=500):
    out = []
    for f in sorted(glob.glob(sys.argv[2] + "/orchard/**/*.py", recursive=True)):
        L = open(f, encoding="utf-8").read().split("\n")
        for i in range(0, len(L) - 25, 25):
            out.append("\n".join(L[i:i + 4 + (i // 25) % 22]))
        if len(out) >= n:
            break
    return out[:n]


sets = {"en": wiki("en"), "code": code(), "zh": wiki("zh"), "ja": wiki("ja"), "ko": wiki("ko")}
t, b = Tokenizer.from_file(TRIM), Tokenizer.from_file(BASE)
res = {}
for k, ss in sets.items():
    et = [t.encode(s) for s in ss]
    eb = [b.encode(s) for s in ss]
    nt = [len(e.ids) for e in et]
    nb = [len(e.ids) for e in eb]
    same = sum(1 for x, y in zip(et, eb) if x.tokens == y.tokens)
    per = [x / y for x, y in zip(nt, nb)]
    chars = sum(map(len, ss))
    res[k] = {"strings": len(ss), "chars": chars, "tokens_trim": sum(nt), "tokens_base": sum(nb), "ratio_total": round(sum(nt) / sum(nb), 4),
              "ratio_median_per_string": round(statistics.median(per), 4), "ratio_max_per_string": round(max(per), 3),
              "strings_with_more_tokens_in_trim": sum(1 for x, y in zip(nt, nb) if x > y), "strings_with_identical_pieces": same,
              "chars_per_token_trim": round(chars / sum(nt), 3), "chars_per_token_base": round(chars / sum(nb), 3)}
    print(k, res[k], flush=True)
json.dump({"trim_vocab": t.get_vocab_size(), "base_vocab": b.get_vocab_size(), "sets": res}, open(OUT, "w"), indent=1)
