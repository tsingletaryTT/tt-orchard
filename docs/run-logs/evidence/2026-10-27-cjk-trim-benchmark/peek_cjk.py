"""Prints the trim's CJK replies (first 110 characters), finish reason, tokens and script shares beside the base's. usage: peek_cjk.py BENCH_DIR"""
import json, sys
t = json.load(open(sys.argv[1] + "/full-trim/cjk.json"))["prompts"]
b = json.load(open(sys.argv[1] + "/full-base/cjk.json"))["prompts"]
for x, y in zip(t, b):
    sh = x.get("script_share", {})
    print(x["i"], x["lang"], "| trim:", x.get("finish"), x.get("completion_tokens"), "cjk-share", round(sh.get("han", 0) + sh.get("kana", 0) + sh.get("hangul", 0), 2),
          "other", sh.get("other"), repr((x.get("reply") or "")[:110]), "| base:", y.get("finish"), y.get("completion_tokens"), repr((y.get("reply") or "")[:40]))
