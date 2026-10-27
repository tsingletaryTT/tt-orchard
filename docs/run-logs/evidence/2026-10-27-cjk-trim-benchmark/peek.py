"""Prints the scores, proxy counts and CJK rows of one session directory. usage: peek.py SESSION_DIR"""
import json, sys
d = sys.argv[1]
for n in ("bfcl-c1", "bfcl-c4", "gsm8k"):
    try:
        s = json.load(open(f"{d}/{n}/summary.json"))
    except OSError:
        continue
    for p in s["phases"]:
        sc = p.get("scores")
        if p["name"] == "bfcl":
            sc = {k: {x: v[x] for x in ("n", "passed", "score", "errors", "cut_off")} for k, v in sc.items()}
        else:
            sc = {k: [(m["filter"], m["metric"], m["value"]) for m in v["metrics"]] for k, v in sc["tasks"].items()}
        print(n, p["name"], "ok", p["ok"], "wall", round(p["wall_s"]), sc, p["proxy"])
try:
    for r in json.load(open(f"{d}/cjk.json"))["prompts"]:
        print(r["i"], r["lang"], r.get("finish"), r.get("completion_tokens"), r["well_formed"], r["expected_script_ok"], r["wall_s"], (r.get("reply") or r.get("error") or "")[:50].replace("\n", " "))
except OSError:
    pass
