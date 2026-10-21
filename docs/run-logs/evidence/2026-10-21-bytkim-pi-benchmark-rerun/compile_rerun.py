"""Add the 32768-token re-run of the cut-off items to the compiled benchmark json (a `rerun_32k` block). Reads the full run and the re-run; changes nothing else."""
import glob, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from orchard import bench_rerun as br
ROOT = Path.home() / "orchard-runs/bytkim--qwen3.8-27b-pi"
FULL, AGAIN = ROOT / "benchmark-full", ROOT / "benchmark-rerun"
METRIC = {"humaneval": "pass@1", "mbpp": "pass_at_1", "ifeval": "prompt_level_strict_acc"}

def sign_test(w, l):
    n = w + l
    if n == 0: return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def merged(model, phase):
    first = {s["doc_id"]: s for s in br.read_samples(FULL / f"{model}-default", phase)}
    again = {s["doc_id"]: s for s in br.read_samples(AGAIN / f"{model}-32k", phase)}
    m = METRIC[phase]
    ok = lambda s: 1.0 if s.get(m) in (True, 1, 1.0) else 0.0
    return {i: ok(again.get(i, s)) for i, s in first.items()}, {i: ok(s) for i, s in first.items()}, len(again)

out = {"conditions": "the items whose reply was empty at the 8192-token limit were asked again with a limit of 32768 generated tokens; every other setting as in the full run; "
                     "all other items keep their first result", "tasks": {}}
for phase in METRIC:
    entry, after, before = {}, {}, {}
    for model in ("pi", "base"):
        after[model], before[model], rn = merged(model, phase)
        summ = json.load(open(AGAIN / f"{model}-32k/summary.json"))
        px = next(p for p in summ["phases"] if p["name"] == phase)["proxy"]
        n = len(after[model]); p = sum(after[model].values()) / n
        entry[model] = {"n": n, "rerun_items": rn, "still_cut_off_at_32768": px["cut_off"], "errors": px["errors"],
                        "score_before": round(sum(before[model].values()) / n, 3), "score": round(p, 3), "passed": round(p * n), "stderr": round(math.sqrt(p * (1 - p) / (n - 1)), 3)}
    ids = sorted(set(after["pi"]) & set(after["base"]))
    w = sum(after["pi"][i] > after["base"][i] for i in ids); l = sum(after["pi"][i] < after["base"][i] for i in ids)
    entry["paired"] = {"items": len(ids), "pi_only_right": w, "base_only_right": l, "sign_test_p": round(sign_test(w, l), 4)}
    out["tasks"][phase] = entry
data = json.load(open(sys.argv[1]))
data["rerun_32k"] = out
json.dump(data, open(sys.argv[2], "w"), indent=2)
for t, v in out["tasks"].items(): print(t, json.dumps(v))
