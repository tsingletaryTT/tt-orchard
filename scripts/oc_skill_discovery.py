#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Does OpenCode v2 find and use SKILL.md skills by itself? A capability test with counts, not a performance measurement.

    python3 scripts/oc_skill_discovery.py [--root DIR]

For each layout OpenCode documents or accepts, a project directory gets copies (not links) of three skills: the installed tt-lang skill
(ttm-tt-lang), and two tenstorrent/skills (tt-device-usage, functional-decoder). OpenCode runs in the usual sandbox with the `skill` tool
switched on and four questions, one fresh run each:

  list    "Which skills do you have? Reply with their names."
  ttlang  a fact only the installed tt-lang skill states (the keyword that sets how many blocks a dataflow buffer holds, `buffer_factor`)
  device  two facts only tt-device-usage states (a reset is the operator's action; device jobs run under a bounded `timeout`)
  decoder the output path functional-decoder names (models/autoports/<model>/tt/functional_decoder.py)

Per run it records whether a `skill` tool call happened (and which skill), whether the answer holds the expected fact, steps and tokens. The
answer check is a substring match on fixed strings, listed in EXPECT. Layouts: claude (.claude/skills), opencode-skills (.opencode/skills),
opencode-skill (.opencode/skill), config-path (the config's `skills` list names a folder), and none (no skills anywhere: the control, which
shows what the model answers from its own knowledge).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import opencode_probe as op  # noqa: E402

SKILLS = {
    "ttm-tt-lang": Path.home() / ".claude" / "skills" / "ttm-tt-lang",
    "tt-device-usage": Path.home() / "code" / "skills" / "plugins" / "tt-model-bringup" / "skills" / "tt-device-usage",
    "functional-decoder": Path.home() / "code" / "skills" / "plugins" / "tt-model-bringup" / "skills" / "functional-decoder",
}
QUESTIONS = {
    "list": "Which skills do you have available? Reply with only their names, one per line.",
    "ttlang": ("In TT-Lang as the installed tt-lang skill documents it, which keyword argument of `ttl.make_dataflow_buffer_like` sets how many "
               "blocks the buffer holds, and what value does the skill's guide example give it? Answer from the skill, in one sentence."),
    "device": ("According to the tt-device-usage skill: may a bringup agent reset the chips on its own, and under what must every device job "
               "run? Answer from the skill, in two short sentences."),
    "decoder": ("According to the functional-decoder skill, at what path under the repository does the implementation file go? "
                "Give the path pattern only."),
}
EXPECT = {"list": ["ttm-tt-lang", "tt-device-usage", "functional-decoder"], "ttlang": ["buffer_factor"],
          "device": ["operator", "timeout"], "decoder": ["models/autoports/", "functional_decoder.py"]}
LAYOUTS = {"claude": ".claude/skills", "opencode-skills": ".opencode/skills", "opencode-skill": ".opencode/skill", "config-path": "skills-folder",
           "none": None}


def build_project(base: Path, layout: str) -> Path:
    adir = base / "adir"
    shutil.rmtree(base, ignore_errors=True)
    adir.mkdir(parents=True)
    sub = LAYOUTS[layout]
    if sub:
        for name, src in SKILLS.items():
            shutil.copytree(src, adir / sub / name, symlinks=False)
    return adir


def answer_text(events: list) -> str:
    return "\n".join(str((e.get("part") or {}).get("text") or "") for e in events if e.get("type") == "text")


def skill_calls(events: list) -> list:
    out = []
    for e in events:
        part = e.get("part") or {}
        if e.get("type") == "tool_use" and part.get("tool") == "skill":
            out.append(json.dumps((part.get("state") or {}).get("input"))[:120])
    return out


def main(argv: list) -> int:
    root = Path(argv[argv.index("--root") + 1]) if "--root" in argv else Path.home() / "orchard-runs" / "ocprobe" / "skilltest"
    results = []
    out_file = root / "results.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        for qname, question in QUESTIONS.items():
            base = root / layout / qname
            adir = build_project(base, layout)
            paths = [str(adir / "skills-folder")] if layout == "config-path" else None
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            r = op.run_opencode(adir, base / "meta", system="You are a careful assistant. Use the skills you are given when they apply.",
                                user=question, endpoint="http://127.0.0.1:8011/v1", model="Qwen/Qwen3-Coder-Next", temperature=0.7, top_p=None,
                                seed=2026112200, steps=12, max_tokens=4096, wall_deadline_s=600, claude_config=(layout == "claude"),
                                mapping=False, skill_tool=True, skill_paths=paths, title=f"{layout}-{qname}")
            ev = op.read_events(r["events"])
            ans = answer_text(ev)
            calls = skill_calls(ev)
            s = op.summarize(ev)
            rec = {"layout": layout, "question": qname, "started_utc": started, "ended_utc": r["ended_utc"], "wall_s": r["wall_s"],
                   "returncode": r["returncode"], "why": r["why"], "skill_tool_calls": calls, "tools": s["tool_calls"], "steps": s["steps"],
                   "tokens": s["tokens"], "foreign_models": s["foreign_models"],
                   "expected_found": {k: (k.lower() in ans.lower()) for k in EXPECT[qname]}, "answer": ans[:700]}
            results.append(rec)
            with open(out_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(layout, qname, "| skill calls", len(calls), "| found", rec["expected_found"], "| steps", s["steps"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
