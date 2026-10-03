"""The fresh, short context each agent step starts from (spec section 5).

A step gets only: the stage skill, the rules of the run, a ledger excerpt (how earlier stages
ended and what happened to this stage so far), the result files of earlier stages, and, for a
resumed or finishing step, the files this stage already wrote. Nothing from an earlier step's
conversation is carried over, so a long run never pays for a long context.
"""
from __future__ import annotations

from pathlib import Path

from orchard.defaults import CONTEXT_FILE_CHARS, SKILL_CHARS
from orchard.stages import STAGES, StageSpec, evidence_record

PHASE_TASKS = {
    "run": ("Do this stage's work as the skill describes. Write {gate} in your stage directory. "
            "Then reply with a short summary and no tool call."),
    "prepare": ("Prepare this stage's hardware test. Do not run it yourself. Write hw_test.json in "
                "your stage directory as {{\"command\": \"<one shell command, run from the run "
                "directory>\", \"deadline_s\": <seconds>}} and handoff.json with the keys goal, "
                "stage, evidence, next_action and check_on_return. The command cannot be a shell "
                "script path; write a Python script and give `python3 stages/{n}/<script>.py`. The "
                "supervisor runs the command on a leased board with TT_VISIBLE_DEVICES set, saves "
                "its output to evidence/hw-test-output.txt and writes test-result.json. Then reply "
                "with a short summary and no tool call."),
    "finish": ("The supervisor ran your hardware test; its record and output are below and in "
               "your stage directory. Write {gate} from that evidence. If the test failed, write "
               "what failed; do not invent a result. Then reply with a short summary and no tool "
               "call."),
}

RULES = """Rules of this run:
- Your tools are shell and write_file. shell starts in the run directory {run_dir}. write_file writes
  only inside your stage directory, stages/{n}.
- Put every file that backs a claim under stages/{n}/evidence/. The supervisor records each one in the
  ledger with its sha256. Evidence paths you cite are relative to the run directory.
- Label every number "measured" (you measured it and the evidence file shows it) or "TODO".
- Never publish, push or upload anything. Never reset chips, never take, release or reset a lease,
  and never stop or start a container, a model server or another process you did not start. The
  supervisor refuses the common spellings of these commands and says how to rewrite them. It
  cannot catch every spelling, so these rules apply to everything you run.
  There are no GitHub or Hugging Face credentials in your environment.
- Do not open Tenstorrent devices from your shell, and do not run code that does (ttnn, tt-metal,
  vLLM, a test that opens a mesh). Your shell's TT_VISIBLE_DEVICES names no chip. Nothing stops a
  device open from your shell, so this rule is yours to keep. Hardware work happens only in the
  hardware test that the supervisor runs for you under a lease.
- When you are finished, reply with a short summary and no tool call."""


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[cut at {limit} characters]"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def ledger_excerpt(entries: list[dict], stage: int) -> list[str]:
    lines = []
    for e in entries:
        d = e["data"]
        if e["event"] == "stage_end":
            why = "; ".join(d.get("reasons") or []) if d.get("result") != "pass" else ""
            lines.append(f"stage {e['stage']}: {d.get('result')}" + (f" ({why})" if why else ""))
        elif e["stage"] == stage and e["event"] in ("escalate", "notice"):
            lines.append(f"stage {stage} {e['event']}: " + str(d.get("reason") or d.get("what")
                                                              or d.get("findings") or "")[:300])
    return lines[-30:]


def build_messages(*, spec: StageSpec, phase: str, run_dir, stage_dir, skill_path: Path | None,
                   refs: dict, facts: dict, entries: list[dict], resumed: bool) -> tuple[str, str]:
    """(system, user) for one agent step."""
    run_dir, stage_dir = Path(run_dir), Path(stage_dir)
    n = spec.number
    skill = _read(skill_path) if skill_path else ""
    system = [f"You are the agent for stage {n} ({spec.name}) of a model bring-up run by tt-orchard.",
              f"Phase: {phase}", "", RULES.format(run_dir=run_dir, n=n), "",
              f"## Skill: {spec.skill} ({skill_path})", clip(skill, SKILL_CHARS)]
    if refs:
        system += ["", "## Related skills (read one with shell if you need it)"]
        system += [f"- {name}: {path or 'not installed on this machine'}" for name, path in refs.items()]

    user = ["## Task", PHASE_TASKS[phase].format(gate=spec.gate_file, n=n), "", "## Run"]
    user += [f"- {k}: {v}" for k, v in facts.items()]
    user += [f"- stage directory: stages/{n}", "", "## Ledger so far"]
    user += [f"- {line}" for line in ledger_excerpt(entries, n)] or ["- nothing yet"]
    user += ["", "## Results of earlier stages"]
    shown = False
    for s in STAGES:
        if s.number >= n or not s.gate_file:
            continue
        f = run_dir / "stages" / str(s.number) / s.gate_file
        if f.is_file():
            rec = evidence_record(run_dir, f)
            user += [f"### {rec['path']} (sha256 {rec['sha256'][:12]})",
                     clip(_read(f), CONTEXT_FILE_CHARS)]
            shown = True
    if not shown:
        user.append("none")
    own = []
    if resumed and spec.marker and (stage_dir / spec.marker).is_file():
        own.append(spec.marker)
    if phase == "finish":
        own += ["hw_test.json", "test-result.json"]
    for name in own:
        f = stage_dir / name
        if f.is_file():
            user += ["", f"## stages/{n}/{name}", clip(_read(f), CONTEXT_FILE_CHARS)]
    return "\n".join(system), "\n".join(user)


def facts_from(run_start: dict, run_dir) -> dict:
    """The run facts every context lists, from the run_start entry."""
    facts = {"model": run_start.get("model"), "run directory": str(run_dir)}
    for name, path in (run_start.get("inputs") or {}).items():
        facts[f"input {name}"] = path
    return facts
