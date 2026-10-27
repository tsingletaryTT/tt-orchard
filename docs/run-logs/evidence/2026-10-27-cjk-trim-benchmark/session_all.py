"""One boot per model for the CJK-Trim benchmark (2026-10-27). Run under `gozer run`, which holds the lease.

Starts the model's run.sh the way orchard.measure_serving does, then, on that one server and in this order:
  1. speed: measure_serving's four prompts (2 reps) and a prompt-length sweep (about 1k, 4k, 16k prompt tokens, 160 tokens generated, 2 reps)
  2. cjk: cjk_check.py's 40 prompts, 4 at a time
  3. bfcl at one request at a time (the issue 83 control), 50 items per set
  4. bfcl at 4 at a time, 100 items per set, then gsm8k, 100 items (`--limit`: the first N documents, the same for both models), 4 at a time,
     32768 tokens, thinking as the template defaults it, stop string '</s>'
The base control's run.sh is checked against its control record (`bench_session.control_summary`) before the server starts.
--scale pilot cuts every set to a few items to measure time and cache size.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.environ["ORCHARD_CHECKOUT"])
from orchard import measure_serving as ms, bench_session as bs  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjk_check  # noqa: E402

PARA = ("The committee met on Tuesday to review the quarterly budget, the staffing plan for the new office, and the schedule for the autumn "
        "product launch. Several members raised concerns about the cost of the lease and the timing of the hiring. ")


def long_prompt(words_target):
    body = PARA * max(1, words_target // 35)
    return [{"role": "user", "content": body + "\n\nSummarize the text above in about 150 words."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True); ap.add_argument("--model-id", required=True)
    ap.add_argument("--run-sh", required=True); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--control-record"); ap.add_argument("--tt-cache", required=True); ap.add_argument("--hf-home", required=True)
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--bfcl-dir", required=True)
    ap.add_argument("--scale", choices=["pilot", "full"], default="full")
    ap.add_argument("--gsm8k-items", type=int, default=100)
    ap.add_argument("--only", default="speed,cjk,bfcl1,main", help="comma list of steps to run")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pilot = a.scale == "pilot"
    steps = set(a.only.split(","))
    sc = ms.load_swap_helpers()
    if a.control_record:
        ctl = bs.control_summary(a.run_sh, json.loads(Path(a.control_record).read_text()), model_dir=a.model_dir)
        (out / "control.json").write_text(json.dumps(ctl, indent=1))
    cache = Path(a.tt_cache); sc.guard_cache(cache, a.model_id)
    log_path = out / "server.log"
    env = dict(os.environ, TT_CACHE_PATH=str(cache), TT_CACHE_HOME=str(cache), HF_HOME=a.hf_home, HF_HUB_OFFLINE="1",
               MODEL_WEIGHTS_DIR=a.model_dir, HF_MODEL=a.model_dir)
    log = open(log_path, "wb")
    proc = subprocess.Popen(["bash", a.run_sh, "--port", str(a.port)], env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    summary = {"label": a.label, "model_id": a.model_id, "scale": a.scale}
    try:
        summary["server_ready_s"] = round(sc.wait_healthy(proc, a.port, 3600.0, log_path), 1)
        print("READY", summary["server_ready_s"], flush=True)
        url = f"http://127.0.0.1:{a.port}"
        served = (json.loads(ms.get_text(a.port, "/v1/models"))["data"][0]["id"])
        ms.run_prompt(a.port, served, ms.warmup_prompt())
        if "speed" in steps:
            t = time.time()
            prompts = ms.PROMPTS + [{"id": f"length-{n}w", "temperature": 0, "max_tokens": 160, "messages": long_prompt(n)} for n in ((700, 3000, 12000) if not pilot else (700,))]
            res = ms.measure(a.port, served, prompts, 1 if pilot else 2)
            json.dump({"label": a.label, "served_name": served, "reps": 1 if pilot else 2, "prompt_set_sha256": ms.prompt_set_hash(prompts), "prompts": res},
                      open(out / "speed.json", "w"), indent=1)
            for k, v in res.items():
                s = v["summary"]; print("SPEED", k, s["decode_tok_s"] and round(s["decode_tok_s"]["median"], 1), s["ttft_s"]["median"], s["completion_tokens"]["median"], flush=True)
            print("STEP speed done", round(time.time() - t), flush=True)
        if "cjk" in steps:
            t = time.time()
            items = cjk_check.PROMPTS[::5] if pilot else cjk_check.PROMPTS
            def one(ip):
                i, (lang, prompt) = ip
                s0 = time.time()
                try:
                    r = cjk_check.post(url, {"model": served, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 8192})
                    ch = r["choices"][0]; text = ch["message"].get("content") or ""
                    rec = {"i": i, "lang": lang, "prompt": prompt, "reply": text, "finish": ch.get("finish_reason"),
                           "completion_tokens": r["usage"]["completion_tokens"], "prompt_tokens": r["usage"]["prompt_tokens"]}
                    rec.update(cjk_check.measures(lang, text, ch.get("finish_reason")))
                except Exception as e:
                    rec = {"i": i, "lang": lang, "prompt": prompt, "error": repr(e)[:300], "well_formed": False, "expected_script_ok": False}
                rec["wall_s"] = round(time.time() - s0, 1)
                return rec
            # index by the prompt's place in the full list so both models' files line up
            idx = [(cjk_check.PROMPTS.index(p), p) for p in items]
            with ThreadPoolExecutor(4) as ex:
                recs = list(ex.map(one, idx))
            json.dump({"prompts": recs, "concurrency": 4, "min_script_share": cjk_check.MIN_SCRIPT_SHARE}, open(out / "cjk.json", "w"), ensure_ascii=False, indent=1)
            print("STEP cjk done", round(time.time() - t), flush=True)
        common = ["--url", url, "--bfcl-dir", a.bfcl_dir, "--thinking", "default", "--max-gen-toks", "32768", "--stop", "</s>",
                  "--quiet-s", "1800", "--phase-backstop-s", "14400"]
        py = sys.executable
        def bench(name, phases, limit, conc):
            t = time.time()
            r = subprocess.run([py, "-m", "orchard.bench_session", "--label", f"{a.label}-{name}", "--out-dir", str(out / name), "--phases", phases,
                                "--limit", str(limit), "--concurrency", str(conc), *common], cwd=os.environ["ORCHARD_CHECKOUT"])
            print("STEP", name, "exit", r.returncode, round(time.time() - t), flush=True)
        if "bfcl1" in steps:
            bench("bfcl-c1", "bfcl", 10 if pilot else 50, 1)
        if "main" in steps:
            bench("bfcl-c4", "bfcl", 10 if pilot else 100, 4)
            bench("gsm8k", "gsm8k", 10 if pilot else a.gsm8k_items, 4)
    finally:
        sc.stop_server(proc); log.close()
        summary["wall_s"] = round(time.time() - t0)
        (out / "session-summary.json").write_text(json.dumps(summary, indent=1))
        print("SESSION_ALL end", summary, flush=True)


main()
