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
OUT_DIR must not exist or must be empty (a reused directory would lose earlier kernels and the summary). Every call is appended to
OUT_DIR/calls.jsonl as soon as it returns, and summary.json is rewritten after every attempt, so an interrupted paid run keeps its
evidence; an attempt cut short by an exception is swept, written as a row ("interrupted: ...") and the exception is raised again.
A successful answer without provider usage has an unknown cost (its token counts are estimates): it stops the run at once, inside the
attempt. An attempt with an unknown cost has cost_known false, and the printed total is then "at least" the known spend.
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
if out_dir.exists() and any(out_dir.iterdir()):
    sys.exit(f"refused: {out_dir} already holds files; give a fresh output directory (a reused one loses earlier kernels and the summary)")

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
# Reasons of an answer that was refused before any request, as whole words. A reason that starts with an HTTP status or an exception
# name comes from a request that was sent ("HTTP 500: capacity exhausted" is not a cap), so it never counts as a local refusal.
LOCAL_REFUSAL = re.compile(r"\b(cap|budget|not enabled|key|No outside call was made)\b", re.I)
SENT_ERROR = re.compile(r"^(HTTP \d{3}\b|\w+(Error|Exception|Timeout)\b)")
spent, rows, stop = 0.0, [], None
out_dir.mkdir(parents=True, exist_ok=True)
CALLS = out_dir / "calls.jsonl"


def save_summary():
    """Rewrite summary.json through a temporary file after every attempt, so a reader never sees half of it."""
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp, out_dir / "summary.json")


def fenced_kernel(text):
    """The first complete fenced block tagged python or py, else the first untagged one; a block with any other tag (json, bash)
    is never the kernel. Fences are read line by line, so the closing fence of one block cannot open another."""
    blocks, tag, body = [], None, []
    for line in text.splitlines(keepends=True):
        mark = line.strip()
        if tag is None:
            if mark.startswith("```"):
                tag, body = mark[3:].strip().lower(), []
        elif mark == "```":
            blocks.append((tag, "".join(body)))
            tag = None
        else:
            body.append(line)
    for wanted in (("python", "py"), ("",)):
        for t, b in blocks:
            if t in wanted:
                return b
    return None


def priced(got):
    """(cost in dollars or None when unknown, whether a request was sent). A failed answer with token counts was sent and is priced
    from them; one with none and a local-refusal reason was not sent; any other failure was sent with unknown cost."""
    if got.cost_usd is not None:
        return got.cost_usd, True
    if got.ok:
        return None, True          # an answer without reported usage: the token counts are estimates, so the bill is unknown
    if (got.tokens_in or got.tokens_out) and PRICE:
        return round(got.tokens_in * PRICE[0] / 1e6 + got.tokens_out * PRICE[1] / 1e6, 9), True
    reason = got.reason or ""
    if not SENT_ERROR.search(reason) and LOCAL_REFUSAL.search(reason):
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
        row.update(cost_known=True, unchanged_writes=0)
        try:
            for turn in range(1, MAX_TURNS + 1):
                if stop:
                    row["ended"] = row["ended"] or f"stopped: {stop}"[:200]
                    break
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
                call = {"ok": got.ok, "reason": got.reason, "sent": sent, "cached": got.cached, "tokens_in": got.tokens_in,
                        "tokens_out": got.tokens_out, "cost_usd": cost, "latency_s": round(got.latency_s, 1), "call_id": got.call_id}
                row["calls"].append(call)
                with CALLS.open("a", encoding="utf-8") as f:       # on disk before anything else can fail
                    f.write(json.dumps({"run_id": RUN_ID, "op": op, "attempt": k, "turn": turn, **call}) + "\n")
                if cost is None:
                    stop = f"a request was sent and its cost is unknown ({(got.reason or 'answer without usage')[:120]}); no further calls"
                    row["cost_known"] = False
                    row["ended"] = stop
                    break
                spent += cost
                row["cost_usd"] += cost
                if not got.ok:
                    row["ended"] = f"no answer: {got.reason}"[:200]
                    break
                row["turns"] = turn
                (adir / f"reply-{turn}.txt").write_text(got.text, encoding="utf-8")
                kernel = fenced_kernel(got.text)
                if kernel is None:
                    history.append(f"\n\n--- turn {turn}: your reply held no python code block ---")
                    continue
                shown = tools.write_file("kernel.py", kernel)
                # write_file answers "unchanged:" for a repeat of the file's content and does not run the check: no new kernel.
                if shown.startswith("unchanged:"):
                    row["unchanged_writes"] += 1
                else:
                    row["kernels_written"] += 1
                # The whole tool response: the probe's write tool already clips the check's output, as it does for the coder.
                history.append(f"\n\n--- turn {turn}: your kernel.py ---\n```python\n{kernel}```\n--- result of the write ---\n{shown}")
                if re.search(r"^PASS$", shown, re.M):
                    row["ended"] = "agent check PASS"
                    break
            else:
                row["ended"] = row["ended"] or f"{MAX_TURNS} turns"
        except BaseException as exc:
            # An interruption or a failure in a request or a write: sweep what the kernel started, keep what was spent and written,
            # then raise again. No verdict is run on a half-finished attempt.
            kp.sweep_strays(adir)
            row.update(ended=f"interrupted: {type(exc).__name__}: {exc}"[:200], passed=False, category=None, first_line="",
                       cost_usd=round(row["cost_usd"], 6), swept=0)
            rows.append(row)
            save_summary()
            raise
        row["ended"] = row["ended"] or "turn cap"
        swept = kp.sweep_strays(adir)          # as run_attempt does: nothing a kernel started may run on into the verdict
        try:
            check = kp.authoritative_check(adir, op=op, env=env, tt_lang=facts)
        finally:
            swept += kp.sweep_strays(adir)
        row.update(passed=check.get("passed") is True, category=check.get("category"), first_line=check.get("first_line") or "",
                   cost_usd=round(row["cost_usd"], 6), swept=swept)
        rows.append(row)
        save_summary()
        print(json.dumps({k2: v for k2, v in row.items() if k2 != "calls"}), flush=True)
save_summary()
unknown = [r for r in rows if r.get("cost_known") is False]
print(f"total spend at least ${spent:.4f}: the cost of {len(unknown)} attempt(s) is unknown" if unknown else f"total spend ${spent:.4f}")
