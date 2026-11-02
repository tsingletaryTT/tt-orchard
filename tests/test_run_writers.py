# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Writers into a stage or run directory replace the path they write (issue 355).

A stage directory is writable by the agent. `Path.write_text` on a FIFO blocks forever and on a link writes through to the link's target. Each
writer here runs in a child process with a timeout (a blocked write must fail the test, not hang the suite). The child leaves a FIFO, then a
link to a file outside the run, at every name the writer could touch (its target and the old fixed `.tmp` name), calls the writer, and the test
checks that the call returned, the target is a regular file with the new content and the outside file is unchanged.
"""
import subprocess
import sys
import textwrap

import pytest

CHILD = textwrap.dedent('''
    import json, os, sys, tempfile
    from pathlib import Path

    base = Path(tempfile.mkdtemp(prefix="run-writers-"))
    outside = base / "outside.txt"
    outside.write_text("keep")
    mode = sys.argv[2]
    stage = base / "run" / "stages" / "2"
    stage.mkdir(parents=True)

    def plant(*names):
        for name in names:
            p = Path(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.is_symlink() or p.exists():
                p.unlink()
            if mode == "fifo":
                os.mkfifo(p)
            else:
                os.symlink(outside, p)

    {body}

    assert outside.read_text() == "keep", "the file outside the run was written through"
    for t in TARGETS:
        assert Path(t).is_file() and not Path(t).is_symlink(), f"{{t}} is not a regular file"
    print("ok")
''')

BODIES = {
    "graft_trial.write_decision": '''
        from orchard import graft_trial
        TARGETS = [stage / graft_trial.DECISION_FILE]
        plant(*TARGETS, stage / (graft_trial.DECISION_FILE + ".tmp"))
        graft_trial.write_decision(stage, {"choice": "safe"}, candidate=None, trial=False)
    ''',
    "sampling_choice.write_decision": '''
        from orchard import sampling_choice
        TARGETS = [stage / sampling_choice.DECISION_FILE]
        plant(*TARGETS, stage / (sampling_choice.DECISION_FILE + ".tmp"))
        sampling_choice.write_decision(stage, {"choice": "keep"}, trial=False)
    ''',
    "text_runtime.write_json": '''
        from orchard import text_runtime
        TARGETS = [stage / "text_chip_config.json"]
        plant(*TARGETS, stage / "text_chip_config.json.tmp")
        text_runtime.write_json(TARGETS[0], {"a": 1})
    ''',
    "audio_plan.write_json": '''
        from orchard import audio_plan
        TARGETS = [stage / "audio_chip_config.json"]
        plant(*TARGETS, stage / "audio_chip_config.json.tmp")
        audio_plan.write_json(TARGETS[0], {"a": 1})
    ''',
    "package._write_json": '''
        from orchard import package
        TARGETS = [stage / "package.json"]
        plant(*TARGETS, stage / "package.json.tmp")
        package._write_json(TARGETS[0], {"a": 1})
    ''',
    "package_cmd._write_handoff": '''
        from orchard import package_cmd
        TARGETS = [stage / package_cmd.HANDOFF_FILE]
        plant(*TARGETS, stage / (package_cmd.HANDOFF_FILE + ".tmp"))
        package_cmd._write_handoff(stage)
    ''',
    "package_cmd._replace_file": '''
        from orchard import package_cmd
        TARGETS = [stage / "benchmark" / "x.txt"]
        plant(*TARGETS, stage / "benchmark" / "x.txt.tmp")
        package_cmd._replace_file(TARGETS[0], "text")
    ''',
    "hwtests.write_plan": '''
        from orchard import hwtests
        TARGETS = [stage / hwtests.PLAN]
        plant(*TARGETS)
        hwtests.write_plan(stage, [])
    ''',
    "kernelstaged.write_staged_report": '''
        from orchard import kernelstaged
        TARGETS = [stage / "kernel-probe-staged-report.md"]
        plant(*TARGETS)
        kernelstaged.write_staged_report(stage, [], "center")
    ''',
    "blocked.write_bundle": '''
        from orchard import blocked
        run = base / "run"
        TARGETS = [run / "blocked.json", run / "BLOCKED.md"]
        plant(*TARGETS)
        blocked.write_bundle(run, [], "blocked", "a reason", now=0.0)
    ''',
    "stock_runtime.evidence": '''
        from orchard import safeio
        TARGETS = [stage / "evidence" / "stock-layout.json"]
        plant(*TARGETS)
        safeio.atomic_write(TARGETS[0], "{}")
    ''',
    "safeio.create_for_write": '''
        from orchard import safeio
        TARGETS = [stage / "log" / "step.log"]
        plant(*TARGETS)
        with safeio.create_for_write(TARGETS[0]) as f:
            f.write(b"hello")
        assert TARGETS[0].read_bytes() == b"hello"
    ''',
}


@pytest.mark.parametrize("mode", ["fifo", "link"])
@pytest.mark.parametrize("name", sorted(BODIES))
def test_a_writer_replaces_a_fifo_or_link_at_its_path(name, mode):
    code = CHILD.format(body=textwrap.dedent(BODIES[name]).strip())
    done = subprocess.run([sys.executable, "-c", code, "x", mode], capture_output=True, text=True, timeout=60,
                          env={"PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parent.parent), "PATH": "/usr/bin:/bin"})
    assert done.returncode == 0 and done.stdout.strip() == "ok", (done.stdout, done.stderr[-1500:])
