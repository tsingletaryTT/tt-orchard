#!/usr/bin/env python3
"""Agent-loop capability check through the Telegraph helper (docs/run-logs/2026-10-28-staged-construction.md).

Usage: TELEGRAPH_CONFIG=<bringup.toml> 2026-10-28-staged-construction-agent-loop.py EXPORT_DIR OUT_DIR ATTEMPTS [BUDGET_USD]
EXPORT_DIR is a copy of the orchard code (put on sys.path). For each attempt k and each op in OPS (interleaved: center 1,
linear_scan 1, center 2, ...) it runs a loop of at most MAX_TURNS helper calls. Each call is one Telegraph `ask` (the helper has no
tools), so the harness plays the tool part: the reply's first python block is written to kernel.py with the probe's own
write tool, which runs `python3 kernel_check.py` and returns its output, exactly the feedback the local coder gets after a write.
The next call's request holds the prompt, the files, and every earlier kernel with its check output. The loop ends at the
first PASS, at MAX_TURNS, when the helper refuses (cap) or when the run's spend reaches BUDGET_USD. The final kernel.py is judged
by the unedited authoritative check (kernelprobe.authoritative_check). One problem id per attempt, so the per-problem caps apply to
one attempt each. Prints one JSON line per attempt and writes OUT_DIR/summary.json. The key is read by the telegraph module only.
"""
import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

export, out_dir, attempts = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
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

spent, rows = 0.0, []
for k in range(1, attempts + 1):
    for op in OPS:
        if spent >= budget:
            rows.append({"op": op, "attempt": k, "skipped": f"spend {spent:.3f} reached the budget {budget}"})
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
        history, row = [], {"op": op, "attempt": k, "turns": 0, "calls": [], "cost_usd": 0.0, "ended": None}
        for turn in range(1, MAX_TURNS + 1):
            if spent >= budget:
                row["ended"] = f"budget: spend {spent:.3f}"
                break
            task = f"Attempt {k}, turn {turn} of at most {MAX_TURNS}.\n\n" + base + "".join(history)
            got = tel.ask(f"staged-176-agent-{op}-{k}", task, FAILURE)
            row["calls"].append({"ok": got.ok, "reason": got.reason, "tokens_in": got.tokens_in, "tokens_out": got.tokens_out,
                                 "cost_usd": got.cost_usd, "latency_s": round(got.latency_s, 1), "call_id": got.call_id})
            if not got.ok:
                row["ended"] = f"no answer: {got.reason}"[:200]
                break
            spent += got.cost_usd or 0.0
            row["cost_usd"] += got.cost_usd or 0.0
            row["turns"] = turn
            (adir / f"reply-{turn}.txt").write_text(got.text, encoding="utf-8")
            blocks = BLOCK.findall(got.text)
            if not blocks:
                history.append(f"\n\n--- turn {turn}: your reply held no python code block ---")
                continue
            shown = tools.write_file("kernel.py", blocks[0])
            history.append(f"\n\n--- turn {turn}: your kernel.py ---\n```python\n{blocks[0]}```\n--- result of the write ---\n{shown[-1500:]}")
            if re.search(r"^PASS$", shown, re.M):
                row["ended"] = "agent check PASS"
                break
        else:
            row["ended"] = row["ended"] or f"{MAX_TURNS} turns"
        row["ended"] = row["ended"] or "turn cap"
        check = kp.authoritative_check(adir, op=op, env=env, tt_lang=facts)
        row.update(passed=check.get("passed") is True, category=check.get("category"), first_line=check.get("first_line") or "",
                   cost_usd=round(row["cost_usd"], 6))
        rows.append(row)
        print(json.dumps({k2: v for k2, v in row.items() if k2 != "calls"}), flush=True)
(out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"total spend ${spent:.4f}")
