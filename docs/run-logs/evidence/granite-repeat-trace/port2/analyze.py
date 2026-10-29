"""Summarise the result JSON files of one session directory.
usage: python analyze.py SESSION_DIR [speed|agree]
speed: per configuration, the mean tok/s of each run (rounds r1, r2, ...), their median, and the share of the 175.2 tok/s floor.
agree: per configuration, relative L2 mean, top-1 and free-run prefix per prompt.
"""
import glob, json, re, statistics, sys
from pathlib import Path

d = Path(sys.argv[1])
what = sys.argv[2] if len(sys.argv) > 2 else "speed"
runs = {}
for f in sorted(d.glob("*.json")):
    j = json.load(open(f))
    if "mean" not in j and what == "speed":
        continue
    m = re.match(r"(.+?)-(un|tr)(?:-r?(\d+))?$", f.stem)
    if not m:
        continue
    runs.setdefault((m.group(1), m.group(2)), []).append(j)
for (cfg, tr), js in sorted(runs.items()):
    if what == "speed":
        means = [j["mean"] for j in js]
        lo, hi = min(j["min"] for j in js), max(j["max"] for j in js)
        med = statistics.median(means)
        print(f"{cfg:12} {'traced' if tr == 'tr' else 'untraced':9} runs {', '.join(f'{x:.1f}' for x in means):28} median {med:5.1f} ({lo:.1f} to {hi:.1f})  {med / 175.2 * 100:4.1f}% of floor")
    else:
        j = js[0]
        if "prompts" not in j:
            continue
        P = j["prompts"]
        print(cfg, tr, "rel", [round(p["rel_l2_mean"], 4) for p in P], "top1", [round(p["top1"], 3) for p in P], "prefix", [p["free_run_prefix"] for p in P])
