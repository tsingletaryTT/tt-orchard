"""Compile the full-size benchmark of bytkim/Qwen3.8-27B-pi against Qwen/Qwen3.8-27B into one path-free json file."""
import json, glob, math, sys
from pathlib import Path
RUN = Path.home()/"orchard-runs/bytkim--qwen3.8-27b-pi/benchmark-full"
LM = {"humaneval":("humaneval_chat","create_test","pass@1"), "mbpp":("mbpp_chat","extract_code","pass_at_1"),
      "gsm8k":("gsm8k","flexible-extract","exact_match"), "ifeval":("ifeval","none","prompt_level_strict_acc")}

def sign_test(w, l):
    n = w + l
    if n == 0: return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def per_item(model, phase):
    name, filt, metric = LM[phase]
    f = glob.glob(str(RUN/model/phase/"results/*/samples_*.jsonl"))[0]
    out = {}
    for line in open(f):
        r = json.loads(line)
        if r.get("filter") != filt: continue
        v = r.get(metric)
        out[r["doc_id"]] = 1.0 if v in (True, 1, 1.0) else 0.0
    return out

res = {"models": {"pi": "bytkim/Qwen3.8-27B-pi@34d3f21045f7b41453e66c39e137cfc818a603ce", "base": "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
       "conditions": "same bundle (qwen3.8-27b-dflash2-p300, 2 chips, bfp8, DFlash2 drafter on, sampling on the device), same run.sh apart from the model path, thinking as the chat template defaults it, temperature 0, max_gen_toks 8192, concurrency 4, every item of each set, lm-evaluation-harness 0.4.13, BFCL v3 revision 61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
       "tasks": {}}
S = {m: json.load(open(RUN/f"{m}-default/summary.json")) for m in ("pi", "base")}
for m in S:
    for p in S[m]["phases"]:
        px, dr = p["proxy"], p.get("drafter") or {}
        base = {"cut_off": px["cut_off"], "requests": px["requests"], "errors": px["errors"], "wall_s": round(p["wall_s"]),
                "drafter_tokens_per_iteration": round(dr["tokens_per_iter"], 2) if dr.get("tokens_per_iter") else None,
                "drafter_accept_rate": round(dr["accept_rate"], 3) if dr.get("accept_rate") else None}
        if p["name"] == "bfcl":
            for sub, v in p["scores"].items():
                res["tasks"].setdefault("bfcl_" + sub, {})[m] = {"n": v["n"], "passed": v["passed"], "score": round(v["score"], 3), "stderr": round(v["stderr"], 3), **{k: base[k] for k in ("drafter_tokens_per_iteration", "drafter_accept_rate")}}
            continue
        name, filt, metric = LM[p["name"]]
        for mm in p["scores"]["tasks"][name]["metrics"]:
            if mm["filter"] == filt and mm["metric"] == metric:
                n = p["scores"]["tasks"][name]["n_samples"]["effective"]
                res["tasks"].setdefault(p["name"], {})[m] = {"n": n, "passed": round(mm["value"] * n), "score": round(mm["value"], 3), "stderr": round(mm["stderr"], 3), **base}
        if p["name"] == "gsm8k":
            for mm in p["scores"]["tasks"][name]["metrics"]:
                if mm["filter"] == "strict-match":
                    res["tasks"].setdefault("gsm8k_strict_format", {})[m] = {"n": n, "score": round(mm["value"], 3), "stderr": round(mm["stderr"], 3)}
for phase in LM:
    a, b = per_item("pi-default", phase), per_item("base-default", phase)
    ids = sorted(set(a) & set(b))
    w = sum(1 for i in ids if a[i] > b[i]); l = sum(1 for i in ids if a[i] < b[i])
    res["tasks"][phase]["paired"] = {"items": len(ids), "pi_only_right": w, "base_only_right": l, "sign_test_p": round(sign_test(w, l), 4)}
json.dump(res, open(sys.argv[1], "w"), indent=2)
for t, v in res["tasks"].items():
    print(t, {m: (x.get("passed"), x["score"], x.get("cut_off")) for m, x in v.items() if m in ("pi", "base")}, v.get("paired"))
