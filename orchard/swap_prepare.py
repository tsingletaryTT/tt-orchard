# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2's prepare step, run by the supervisor when it already holds every fact.

On the weights-only path the supervisor drafts `stages/2/swap_config.json` and copies the tested template `prepare_swap.py` next to it
(orchard/swap_draft.py). Everything the prepare step then does is mechanical: run the script, check that it built `model-dir/` (and
`run.sh` for a bundle), and write `hw_test.json` and `handoff.json`. An agent given those steps failed three times in one run
(it repeated an `ls`, it wrote a result file from the skill's example values, it explored a cache directory for 20 turns), so the
supervisor does them itself. The agent is called only when the script exits nonzero, and it is given the script's message. Without a
config or a script (the facts could not be drafted), nothing here runs and the agent prepares as its skill describes.

The second function closes a hole next to this one. The prepare phase must not leave a hardware result behind: `result.json` comes
from the finish step, which reads the real test output, and the gate reads that file. `set_aside_results` moves any such file out of the
way before the hardware test starts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from orchard import safeio
from orchard.agent import spawn_checked
from orchard.runner import Denied

PREPARE_TIMEOUT_S = 900          # choice: prepare_swap.py links and copies files and takes seconds; this only catches a hang
TEST_DEADLINE_S = 3600           # the deadline the skill gives hw_test.json
OUTPUT_NAME = "prepare-swap-output.txt"           # in the stage's evidence directory
MAX_CONFIG_BYTES = 1024 * 1024    # the same bound the supervisor puts on swap_config.json
NOTE_TAIL_CHARS = 1500           # how much of the script's output the agent is told

# Files only the hardware test or the finish step may write. A copy written during prepare is set aside under this suffix.
RESULT_FILES = ("result.json", "test-result.json", "evidence/swap-check.json", "evidence/hw-test-output.txt",
                "evidence/port-check.json")       # the last is the full-port stage 2's record (orchard/full_port.py)
REJECTED_SUFFIX = ".rejected-from-prepare"


@dataclass(frozen=True)
class PrepareOutcome:
    """done: hw_test.json and handoff.json are written and the agent has nothing to do. note: what to tell the agent when it is called
    (None when the script was not run at all, so the agent works from its skill alone). returncode: the script's, None when it did
    not run or did not finish."""
    done: bool
    note: str | None = None
    returncode: int | None = None
    output: str | None = None            # the output file, relative to the stage directory


def _tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text[-NOTE_TAIL_CHARS:]


def _agent_note(code: int | None, tail: str, *, timed_out: bool = False, refused: str | None = None) -> str:
    if refused:
        what = f"The supervisor ran prepare_swap.py and the command runner refused it: {refused}."
    elif timed_out:
        what = f"The supervisor ran prepare_swap.py and it did not finish within {PREPARE_TIMEOUT_S} seconds."
    else:
        what = f"The supervisor ran prepare_swap.py and it exited {code}."
    out = f" Its output ends:\n{tail}\n" if tail else " It printed nothing.\n"
    if code == 2:
        act = ("Change the value the message names in stages/2/swap_config.json, run `python3 stages/2/prepare_swap.py` again, "
               "then write hw_test.json and handoff.json as the skill describes.")
    else:
        act = ("Read the output above. Change swap_config.json only if it shows a wrong value, run the script again, "
               "then write hw_test.json and handoff.json as the skill describes. If the failure is not a config value, say so and stop.")
    return f"{what}{out}{act}"


def prepare(*, run_dir, stage_dir, env: dict, stage: int = 2) -> PrepareOutcome:
    """Run prepare_swap.py in `stage_dir` and, when it succeeds, write hw_test.json and handoff.json."""
    run_dir, stage_dir = Path(run_dir), Path(stage_dir)
    config, script = stage_dir / "swap_config.json", stage_dir / "prepare_swap.py"
    if not config.is_file() or not script.is_file():
        return PrepareOutcome(False)
    out_path = stage_dir / "evidence" / OUTPUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(out_path, stage_dir)
    command = f"python3 stages/{stage}/prepare_swap.py"
    code, timed_out, refused = None, False, None
    # A FIFO or link an agent left at the output path would block the open or send the output elsewhere: the name is removed first.
    with safeio.create_for_write(out_path) as out:
        try:
            code, timed_out = spawn_checked(command, run_dir, env, PREPARE_TIMEOUT_S, out)
        except Denied as exc:
            refused = str(exc)
    tail = _tail(out_path)
    if refused or timed_out or code != 0:
        return PrepareOutcome(False, _agent_note(code, tail, timed_out=timed_out, refused=refused), code, rel)
    # The script exited 0. Check what it was supposed to build before the test is written.
    package = _is_container(config)
    missing = [name for name in (("model-dir",) if package else ("model-dir", "run.sh")) if not (stage_dir / name).exists()]
    if missing:
        note = (f"The supervisor ran prepare_swap.py and it exited 0, but {', '.join(missing)} is missing from stages/{stage}. "
                f"Its output ends:\n{tail}\nRun it again and read the output; then write hw_test.json and handoff.json as the skill describes.")
        return PrepareOutcome(False, note, code, rel)
    script_name = "serve_and_compare_container.py" if package else "serve_and_compare.py"
    test = {"command": f"python3 stages/{stage}/{script_name}", "deadline_s": TEST_DEADLINE_S}
    handoff = {
        "goal": "Serve the new model's weights with the existing TT implementation on one leased board and compare its greedy tokens "
                "with the stage 1 CPU reference.",
        "stage": stage,
        "evidence": [f"stages/{stage}/evidence/hw-test-output.txt", f"stages/{stage}/evidence/swap-check.json"],
        "next_action": f"The supervisor runs {script_name} on a leased board. The finish step then copies result_draft from "
                       "swap-check.json into result.json.",
        "check_on_return": f"stages/{stage}/test-result.json has returncode 0 and swap-check.json has serves, coherent, n_tokens "
                           "of at least 16 and top1_agreement of at least 0.85.",
    }
    for name, body in (("hw_test.json", test), ("handoff.json", handoff)):
        safeio.atomic_write(stage_dir / name, json.dumps(body, indent=2) + "\n")
    return PrepareOutcome(True, None, code, rel)


def _is_container(config: Path) -> bool:
    """A swap_config.json that names a container package (and no bundle) is served by the container script."""
    try:
        cfg = safeio.read_json(config, max_bytes=MAX_CONFIG_BYTES, expect=dict)
    except safeio.SafeReadError:
        return False
    return bool(cfg.get("package")) and not cfg.get("bundle_dir")


def set_aside_results(stage_dir) -> list[str]:
    """Move every hardware-result file found in `stage_dir` to `<name>.rejected-from-prepare`. Returns the names moved (relative to
    the stage directory). Call it after the prepare phase and before the hardware test: nothing in the stage may claim a result
    before the test has run."""
    stage_dir = Path(stage_dir)
    moved = []
    for name in RESULT_FILES:
        f = stage_dir / name
        if f.is_file() or f.is_symlink():
            os.replace(f, f.with_name(f.name + REJECTED_SUFFIX))
            moved.append(name)
    return moved
