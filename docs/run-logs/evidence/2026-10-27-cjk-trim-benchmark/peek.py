"""Prints the scores, proxy counts and CJK rows of one session directory. usage: peek.py SESSION_DIR
A phase with no scores (lm-eval failed, timed out or wrote no results) is printed with its `ok` and `problem` and the run goes on."""
import json, sys


def scores_text(p):
    sc = p.get("scores")
    if not isinstance(sc, dict):
        return None
    try:
        if p.get("name") == "bfcl":
            return {k: {x: v.get(x) for x in ("n", "passed", "score", "errors", "cut_off")} for k, v in sc.items()}
        return {k: [(m.get("filter"), m.get("metric"), m.get("value")) for m in v.get("metrics", [])] for k, v in sc.get("tasks", {}).items()}
    except (AttributeError, TypeError):
        return None


d = sys.argv[1]
for n in ("bfcl-c1", "bfcl-c4", "gsm8k"):
    try:
        s = json.load(open(f"{d}/{n}/summary.json"))
    except (OSError, ValueError) as e:
        print(n, "summary unreadable:", repr(e)[:100])
        continue
    for p in s.get("phases", []):
        wall = p.get("wall_s")
        sc = scores_text(p)
        print(n, p.get("name"), "ok", p.get("ok"), "wall", round(wall) if isinstance(wall, (int, float)) else None,
              "scores", sc if sc is not None else "none", "problem", p.get("problem"), "proxy", p.get("proxy"))
try:
    for r in json.load(open(f"{d}/cjk.json"))["prompts"]:
        print(r["i"], r["lang"], r.get("finish"), r.get("completion_tokens"), r.get("well_formed"), r.get("expected_script_ok"), r.get("wall_s"), (r.get("reply") or r.get("error") or "")[:50].replace("\n", " "))
except (OSError, ValueError, KeyError):
    pass
