#!/usr/bin/env python3
"""Agent-loop capability check through the Telegraph helper (docs/run-logs/2026-10-28-staged-construction.md).

Usage: TELEGRAPH_CONFIG=<bringup.toml> 2026-10-28-staged-construction-agent-loop.py EXPORT_DIR OUT_DIR ATTEMPTS [BUDGET_USD]
EXPORT_DIR is a copy of the orchard code (put on sys.path). For each attempt k and each op in OPS (interleaved: center 1,
linear_scan 1, center 2, ...) it runs a loop of at most MAX_TURNS helper calls. Each call is one Telegraph `ask` (the helper has no
tools), so the harness plays the tool part: the reply's first python block is written to kernel.py with the probe's own
write tool, which runs `python3 kernel_check.py` and returns its output, exactly the feedback the local coder gets after a write.
The next call's request holds the prompt, the files, and every earlier kernel with its check output. The loop ends at the
first PASS, at MAX_TURNS, when the helper refuses (cap) or when the run's spend reaches BUDGET_USD. Spend counts every request that
was sent, answered or not: a failed answer that reports tokens (an empty answer cut at the output limit) is priced from them, and a
request that was sent but reports no usage stops the run, because its cost is unknown. A request that would be cut by the
helper's prompt limit stops the run before it is sent. Each run has an id (in every problem id and in the results), and each call
records whether the answer came from the cache. Strays a generated kernel started are swept before every verdict and on exit. The final kernel.py is judged
by the unedited authoritative check (kernelprobe.authoritative_check). One problem id per attempt, so the per-problem caps apply to
one attempt each. Prints one JSON line per attempt and writes OUT_DIR/summary.json. The key is read by the telegraph module only.
"""
import argparse
import dataclasses
import json
import os
import re
import secrets
import sys
from pathlib import Path

export, out_dir, attempts = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])   # absolute: the checks run elsewhere
budget = float(sys.argv[4]) if len(sys.argv) > 4 else 5.5
sys.path.insert(0, str(export))

from orchard import kernelprobe as kp, telegraph as tg, telegraph_cmd as tc, ttlang_env  # noqa: E402

OPS = ("center", "linear_scan")
MODEL = "anthropic/claude-sonnet-5-5"
MAX_TURNS = min(8, kp.BUDGET["max_turns"])          # the cap of 6 calls per problem in the config may stop an attempt sooner
SKIP_REFERENCES = ("guide.md", "examples.md")        # left out to hold the cost; the skill says they are for an older tt-lang

env = kp.default_env(argparse.Namespace(src_dir=None, venv_dir=None))
references, problems = ttlang_env.read_references(env, ttlang_env.skill_dir_from(argparse.Namespace()))
assert not problems, problems
facts = kp.tt_lang_facts(env)
cfg = dataclasses.replace(tc.load_config(Path(os.environ["TELEGRAPH_CONFIG"])), models=(MODEL,))
home = Path.home()
tel = tg.Telegraph(cfg, tc.resolve_directory(None, cfg, os.environ, home), environ=dict(os.environ), home=str(home))
BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
FAILURE = ("Reply with the complete kernel.py as one ```python code block. It is written to kernel.py and the check runs; its output "
           "is shown to you in the next request. If the check prints PASS you are done.")

RUN_ID = secrets.token_hex(4)            # in every problem id, so a reused ledger neither replays cached answers nor inherits caps
PRICE = cfg.prices.get(MODEL)           # dollars per million tokens, (input, output)
LOCAL_REFUSAL = ("cap", "budget", "not enabled", "key")    # words in the reason of an answer that was refused before any request
spent, rows, stop = 0.0, [], None


def priced(got):
    """(cost in dollars or None when unknown, whether a request was sent). A failed answer with token counts was sent and is priced
    from them; one with none and a local-refusal reason was not sent; any other failure was sent with unknown cost."""
    if got.cost_usd is not None:
        return got.cost_usd, True
    if (got.tokens_in or got.tokens_out) and PRICE:
        return round(got.tokens_in * PRICE[0] / 1e6 + got.tokens_out * PRICE[1] / 1e6, 9), True
    if any(w in (got.reason or "").lower() for w in LOCAL_REFUSAL):
        return 0.0, False
    return None, True

for k in range(1, attempts + 1):
    for op in OPS:
        if stop or spent >= budget:
            rows.append({"op": op, "attempt": k, "skipped": stop or f"spend {spent:.3f} reached the budget {budget}"})
            print(json.dumps(rows[-1]), flush=True)
            continue
        adir = out_dir / op / f"attempt-{k}"
        kp.prepare_attempt(adir, op=op, env=env, tt_lang=facts, references=references)
        files = [f"=== {rel} ===\n{(adir / rel).read_text(encoding='utf-8', errors='replace')}"
                 for rel in ["kernel_spec.json"] + sorted(str(p.relative_to(adir)) for p in (adir / "reference").iterdir()
                                                          if p.name not in SKIP_REFERENCES)]
        user = (f"Write kernel.py for the op `{op}` in kernel_spec.json. Start by reading kernel_spec.json and "
                "reference/eltwise_add.py, then the reference files the skill names for this kind of op.")
        base = ("Agent-loop capability check. This is what a smaller coder model gets for one kernel-writing task. It has tools to "
                "read files and a write tool that runs the check after every write. You have no tools: the files it can read are "
                "printed here, and each kernel you write is checked for you.\n\n--- system prompt ---\n"
                + kp.SKILL.read_text(encoding="utf-8") + f"\n\n--- user message ---\n{user}\n\n--- files ---\n" + "\n\n".join(files))
        tools = kp.attempt_tools(adir)
        history, row = [], {"run_id": RUN_ID, "op": op, "attempt": k, "turns": 0, "kernels_written": 0, "calls": [],
                            "cost_usd": 0.0, "ended": None}
        for turn in range(1, MAX_TURNS + 1):
            if spent >= budget:
                row["ended"] = f"budget: spend {spent:.3f}"
                break
            problem = f"staged-176-agent-{RUN_ID}-{op}-{k}"
            task = f"Attempt {k}, turn {turn} of at most {MAX_TURNS}.\n\n" + base + "".join(history)
            request_chars = len(f"Problem: {problem}\n\nTask:\n{task}\n\nWhat failed:\n{FAILURE}") + len(tg.SYSTEM)
            if request_chars > cfg.max_prompt_chars:
                stop = (f"the request ({request_chars} characters) is over the helper's max_prompt_chars "
                        f"({cfg.max_prompt_chars}) and would be cut in the middle; raise it in the telegraph config or shorten the prompt")
                row["ended"] = stop
                break
            got = tel.ask(problem, task, FAILURE)
            cost, sent = priced(got)
            row["calls"].append({"ok": got.ok, "reason": got.reason, "sent": sent, "cached": got.cached, "tokens_in": got.tokens_in,
                                 "tokens_out": got.tokens_out, "cost_usd": cost, "latency_s": round(got.latency_s, 1),
                                 "call_id": got.call_id})
            if cost is None:
                stop = f"a request was sent and its cost is unknown ({got.reason[:120]}); no further calls"
            else:
                spent += cost
                row["cost_usd"] += cost
            if not got.ok:
                row["ended"] = f"no answer: {got.reason}"[:200]
                break
            row["turns"] = turn
            (adir / f"reply-{turn}.txt").write_text(got.text, encoding="utf-8")
            blocks = BLOCK.findall(got.text)
            if not blocks:
                history.append(f"\n\n--- turn {turn}: your reply held no python code block ---")
                continue
            shown = tools.write_file("kernel.py", blocks[0])
            row["kernels_written"] += 1
            # The whole tool response: the probe's write tool already clips the check's output, as it does for the coder.
            history.append(f"\n\n--- turn {turn}: your kernel.py ---\n```python\n{blocks[0]}```\n--- result of the write ---\n{shown}")
            if re.search(r"^PASS$", shown, re.M):
                row["ended"] = "agent check PASS"
                break
        else:
            row["ended"] = row["ended"] or f"{MAX_TURNS} turns"
        row["ended"] = row["ended"] or "turn cap"
        swept = kp.sweep_strays(adir)          # as run_attempt does: nothing a kernel started may run on into the verdict
        try:
            check = kp.authoritative_check(adir, op=op, env=env, tt_lang=facts)
        finally:
            swept += kp.sweep_strays(adir)
        row.update(passed=check.get("passed") is True, category=check.get("category"), first_line=check.get("first_line") or "",
                   cost_usd=round(row["cost_usd"], 6), swept=swept)
        rows.append(row)
        print(json.dumps({k2: v for k2, v in row.items() if k2 != "calls"}), flush=True)
(out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")      # written once, after every attempt ended
print(f"total spend ${spent:.4f}")
