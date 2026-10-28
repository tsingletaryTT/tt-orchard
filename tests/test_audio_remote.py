# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The hop from the dependency-free supervisor to the interpreter that has numpy (orchard/audio_remote.py, orchard/audio_cli.py). The interpreter used
here is the test's own, which has numpy; the point under test is the protocol: one JSON request in, one JSON answer out, failures reported and never a pass."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import audio_cli, audio_remote, audio_runtime
from orchard import stages
from orchard.audio_plan import AudioStageError
from test_audio_stages import stage1, make_run

PY = sys.executable
ENV = {"PATH": os.environ.get("PATH", "")}


def test_a_gate_runs_in_another_interpreter_and_gives_the_same_answer_as_in_process(tmp_path):
    run_dir, stage_dir, _, _ = stage1(tmp_path)
    args = {"stage_dir": str(stage_dir), "run_dir": str(run_dir)}
    assert audio_remote.call(PY, "gate_reference", args, timeout_s=120, env=ENV) == audio_cli.dispatch("gate_reference", args)["result"] == []


def test_a_gate_that_finds_a_problem_returns_it_across_the_hop(tmp_path):
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    f = run_dir / rec["runs"]["reference"]["dir"] / "arrays" / "latent__final.npy"
    f.write_bytes(f.read_bytes()[:-4] + b"\x01\x02\x03\x04")
    problems = audio_remote.call(PY, "gate_reference", {"stage_dir": str(stage_dir), "run_dir": str(run_dir)}, timeout_s=120, env=ENV)
    assert problems and any("reference" in p for p in problems)


def test_a_verb_error_is_raised_with_its_own_message(tmp_path):
    with pytest.raises(AudioStageError) as e:
        audio_remote.call(PY, "finish", {"stage_dir": str(tmp_path), "run_dir": str(tmp_path), "ref_stage_dir": str(tmp_path), "plan": [], "roles": {}},
                          timeout_s=120, env=ENV)
    assert "cannot be read" in str(e.value) and "no answer" not in str(e.value)       # the verb answered; it did not crash


def test_the_interpreter_is_started_isolated_so_a_module_in_the_environment_cannot_shadow_numpy(tmp_path):
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "numpy.py").write_text("raise RuntimeError('shadowed numpy')\n")
    env = {**ENV, "PYTHONPATH": str(poison)}
    run_dir = tmp_path / "run"
    assert audio_remote.call(PY, "gate_reference", {"stage_dir": str(run_dir / "s"), "run_dir": str(run_dir)}, timeout_s=120, env=env)


def test_an_unknown_verb_and_a_malformed_request_are_errors():
    with pytest.raises(AudioStageError) as e:
        audio_remote.call(PY, "nonsense", {}, timeout_s=60, env=ENV)
    assert "unknown verb" in str(e.value)
    with pytest.raises(AudioStageError) as e:
        audio_remote.call(PY, "gate_chip", {"stage_dir": "x"}, timeout_s=60, env=ENV)
    assert "bad request" in str(e.value)


def test_the_cli_refuses_unreadable_input():
    r = subprocess.run([PY, "-I", str(audio_remote.CLI)], input="not json", capture_output=True, text=True)
    assert r.returncode == 1 and "unreadable request" in json.loads(r.stdout.splitlines()[-1])["error"]


def test_an_interpreter_that_dies_without_an_answer_is_reported_with_its_error_output(tmp_path):
    script = tmp_path / "fake_python"
    script.write_text("#!/bin/sh\necho 'boom: no module named numpy' >&2\nexit 3\n")
    script.chmod(0o755)
    with pytest.raises(AudioStageError) as e:
        audio_remote.call(str(script), "gate_chip", {}, timeout_s=60, env=ENV)
    assert "no answer" in str(e.value) and "no module named numpy" in str(e.value)


def test_a_missing_interpreter_is_reported(tmp_path):
    with pytest.raises(AudioStageError) as e:
        audio_remote.call(str(tmp_path / "nope"), "gate_chip", {}, timeout_s=60, env=ENV)
    assert "cannot start" in str(e.value)


def test_an_answer_that_is_not_an_object_or_has_no_result_is_an_error(tmp_path):
    for body, word in (("[1]", "not an object"), ('{"x": 1}', "no result")):
        script = tmp_path / f"p{len(word)}"
        script.write_text(f"#!/bin/sh\ncat >/dev/null\necho '{body}'\n")
        script.chmod(0o755)
        with pytest.raises(AudioStageError) as e:
            audio_remote.call(str(script), "gate_chip", {}, timeout_s=60, env=ENV)
        assert word in str(e.value)


def test_a_verb_past_its_deadline_is_killed_with_everything_it_started(tmp_path):
    marker = tmp_path / "child-alive"
    script = tmp_path / "slow_python"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        (sleep 2; touch {marker}) &
        sleep 30
        """))
    script.chmod(0o755)
    with pytest.raises(AudioStageError) as e:
        audio_remote.call(str(script), "gate_chip", {}, timeout_s=1.0, env=ENV)
    assert "ran past" in str(e.value)
    import time
    time.sleep(3.5)                      # the child would have written its file after 2 s had the session not been killed
    assert not marker.exists()


# ---- the gates in stages.py use the hop and fail closed ---------------------------------------------------------------------------------

def test_a_gate_with_no_selection_in_the_ledger_fails(tmp_path):
    (tmp_path / "ledger.jsonl").write_text("")
    r = stages.gate_audio_reference(tmp_path / "stages" / "1", tmp_path)
    assert not r.ok and "no audio runtime selection" in r.reasons[0]
    assert not stages.gate_audio_chip(tmp_path / "stages" / "2", tmp_path).ok


def test_a_gate_whose_interpreter_cannot_run_fails_and_never_passes(tmp_path):
    entry = {"event": "decision", "data": {"decision": audio_runtime.AUDIO_RUNTIME_SELECTED,
                                          "paths": audio_runtime.AudioPaths(str(tmp_path / "nope"), "/c", "/s").record()}}
    (tmp_path / "ledger.jsonl").write_text(json.dumps(entry) + "\n")
    r = stages.gate_audio_chip(tmp_path / "stages" / "2", tmp_path)
    assert not r.ok and "cannot start" in r.reasons[0]


def test_a_gate_whose_verb_answers_with_something_that_is_not_a_list_fails(tmp_path):
    script = tmp_path / "py"
    script.write_text("#!/bin/sh\ncat >/dev/null\necho '{\"result\": \"abc\"}'\n")
    script.chmod(0o755)
    entry = {"event": "decision", "data": {"decision": audio_runtime.AUDIO_RUNTIME_SELECTED, "paths": audio_runtime.AudioPaths(str(script), "/c", "/s").record()}}
    (tmp_path / "ledger.jsonl").write_text(json.dumps(entry) + "\n")
    r = stages.gate_audio_chip(tmp_path / "stages" / "2", tmp_path)
    assert not r.ok and list(r.reasons) == ["the audio gate gave an answer that is not a list"]


def test_a_gate_whose_verb_answers_with_problems_passes_only_when_the_list_is_empty(tmp_path):
    run_dir, stage_dir, _, _ = stage1(tmp_path)
    entry = {"event": "decision", "data": {"decision": audio_runtime.AUDIO_RUNTIME_SELECTED, "paths": audio_runtime.AudioPaths(PY, "/c", "/s").record()}}
    (run_dir / "ledger.jsonl").write_text(json.dumps(entry) + "\n")
    assert stages.gate_audio_reference(stage_dir, run_dir).ok
    (stage_dir / "audio_reference.json").write_text("{}")
    r = stages.gate_audio_reference(stage_dir, run_dir)
    assert not r.ok and r.reasons


def test_the_selection_is_read_from_the_ledger_file_and_a_bad_line_is_skipped(tmp_path):
    good = {"event": "decision", "data": {"decision": audio_runtime.AUDIO_RUNTIME_SELECTED, "paths": audio_runtime.AudioPaths("/r", "/c", "/s").record()}}
    (tmp_path / "ledger.jsonl").write_text("{broken " + audio_runtime.AUDIO_RUNTIME_SELECTED + "\n" + json.dumps(good) + "\n")
    assert audio_runtime.selection_from_run_dir(tmp_path) == audio_runtime.AudioPaths("/r", "/c", "/s")
    assert audio_runtime.selection_from_run_dir(tmp_path / "nowhere") is None
