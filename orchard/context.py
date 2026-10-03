"""The fresh, short context each agent step starts from (spec section 5).

A step gets only: the stage skill, the rules of the run, a ledger excerpt (how earlier stages
ended and what happened to this stage so far), the result files of earlier stages, and, for a
resumed or finishing step, the files this stage already wrote. Nothing from an earlier step's
conversation is carried over, so a long run never pays for a long context. The supervisor's
gate-feedback continuation is an exception: it adds a message to the step that just ended
(orchard/supervisor.py, `gate_feedback_text`).
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
    # A stage whose spec has `tests` (stage 4 on the weights-only path) prepares a list of tests.
    "prepare-tests": ("Prepare this stage's hardware tests, one per chip configuration. Do not run "
                      "them yourself. Put each configuration's files in configs/<chips>/ of your "
                      "stage directory as the skill describes. Then write hw_tests.json in your "
                      "stage directory as {{\"tests\": [{{\"chips\": <n>, \"script\": "
                      "\"serve_and_compare.py\" or \"serve_and_compare_container.py\", "
                      "\"deadline_s\": <seconds>}}, ...]}} and handoff.json with the keys goal, "
                      "stage, evidence, next_action and check_on_return. The supervisor runs each "
                      "test as `python3 stages/{n}/configs/<chips>/<script>`, in order of chip "
                      "count, on leased chips with TT_VISIBLE_DEVICES, ORCHARD_DEVICE_IDS and "
                      "ORCHARD_TEST_LABEL set. It saves each test's output to "
                      "tests/<chips>/output.txt, writes tests/<chips>/test-result.json, and writes "
                      "test-result.json for the whole list. Then reply with a short summary and no "
                      "tool call."),
    "finish": ("The supervisor ran your hardware test; its record and output are below and in "
               "your stage directory. Write {gate} from that evidence. Do not invent a result. "
               "If the test failed (returncode not 0, or timed_out true), write {gate} with the "
               "failure recorded honestly: copy the failure text from "
               "stages/{n}/evidence/hw-test-output.txt into a `failure` field, mark what did not "
               "happen as false (for the weights swap, `serves` false), and set every number you "
               "did not measure to null. Then stop. Do not investigate the failure and do not try "
               "to make the result pass. The next attempt runs a fresh test. Reply with a short "
               "summary and no tool call."),
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
  Your environment holds no tokens. Do not read credential files anywhere on the machine, such
  as SSH keys or token files.
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

    task = PHASE_TASKS["prepare-tests" if phase == "prepare" and spec.tests else phase]
    user = ["## Task", task.format(gate=spec.gate_file, n=n), "", "## Run"]
    user += [f"- {k}: {v}" for k, v in facts.items()]
    user += [f"- stage directory: stages/{n}"]
    if n == 4 and facts.get("required chip configurations"):
        user += ["", "## Chip configurations", MESH_NOTE.format(counts=facts["required chip configurations"])]
    user += ["", "## Ledger so far"]
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
        own += ["hw_tests.json" if spec.tests else "hw_test.json", "test-result.json"]
    for name in own:
        f = stage_dir / name
        if f.is_file():
            user += ["", f"## stages/{n}/{name}", clip(_read(f), CONTEXT_FILE_CHARS)]
    return "\n".join(system), "\n".join(user)


MESH_NOTE = ("Required chip counts for this run: {counts}. Each one needs an entry in configs with "
             "pass true and evidence. Any other chip count is optional: try it, and if it does not "
             "work, record it with pass false and a reason. Do not leave it out and do not mark it "
             "as passing.")


def facts_from(run_start: dict, run_dir) -> dict:
    """The run facts every context lists, from the run_start entry."""
    facts = {"model": run_start.get("model"), "run directory": str(run_dir)}
    for name, path in (run_start.get("inputs") or {}).items():
        facts[f"input {name}"] = path
    if run_start.get("required_chips"):
        facts["required chip configurations"] = ", ".join(str(c) for c in run_start["required_chips"])
    return facts
