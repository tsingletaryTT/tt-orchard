# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 389, items 1 and 2: the stack freshness probe asks about the interpreter and the inputs the run itself uses.

1. A text-runtime run's reference is the text interpreter (`text_runtime.text_paths`, or the one stage 0 recorded), not `ORCHARD_REFERENCE_PYTHON`.
2. A resumed run is judged on the inputs in its `run_start`, which are the inputs the agent works with, not on the resuming command's inputs.
Items 3 to 5 of the issue (import settings, filtered PYTHONPATH entries, the text runtime's model directory) need a decision about which settings
to trust and are not changed here.
"""
from types import SimpleNamespace

import pytest

from orchard import stack_pins as sp, supervisor, text_runtime

NAME = next(iter(text_runtime.REGISTRY))


class Led:
    def __init__(self, entries=None, fail=False):
        self.entries, self.fail = entries or [], fail

    def read(self):
        if self.fail:
            raise OSError("unreadable")
        return self.entries


def run_start(**inputs):
    return {"event": "run_start", "stage": None, "data": {"inputs": inputs}}


def selected(python):
    return {"event": "decision", "stage": 0, "data": {"decision": text_runtime.TEXT_RUNTIME_SELECTED, "paths": {"python": python}}}


def sources(monkeypatch, tmp_path, *, inputs, ledger, environ):
    got = []
    monkeypatch.setattr(sp, "make_check", lambda s, **kw: got.append(s))
    sup = SimpleNamespace(paths=SimpleNamespace(operator_home=str(tmp_path / "ops")), home=tmp_path / "home", inputs=inputs, run_dir=tmp_path / "run",
                          extra_env={}, ledger=ledger)
    monkeypatch.delenv("ORCHARD_TT_METAL_DIR", raising=False)
    supervisor.stack_check_for(sup, SimpleNamespace(skills_dir=[]), environ=environ)
    return got[0]


# ---- item 1: the interpreter stage 1 uses -----------------------------------------------------------------------------------

def test_a_new_text_runtime_run_probes_the_text_interpreter_not_the_reference_variable(monkeypatch, tmp_path):
    s = sources(monkeypatch, tmp_path, inputs={"model": "/m", text_runtime.RUNTIME_INPUT: NAME}, ledger=Led(),
                environ={"ORCHARD_REFERENCE_PYTHON": "/env/ref-python", "ORCHARD_TEXT_PYTHON": "/env/text-python"})
    assert s.ref_python == "/env/text-python"


def test_a_text_runtime_run_without_the_text_variable_uses_the_default_text_interpreter(monkeypatch, tmp_path):
    s = sources(monkeypatch, tmp_path, inputs={"model": "/m", text_runtime.RUNTIME_INPUT: NAME}, ledger=Led(), environ={"ORCHARD_REFERENCE_PYTHON": "/env/ref"})
    assert s.ref_python == str(tmp_path / "ops" / ".tenstorrent-venv" / "bin" / "python")


def test_a_resumed_text_runtime_run_probes_the_interpreter_stage_0_recorded(monkeypatch, tmp_path):
    led = Led([run_start(model="/m", **{text_runtime.RUNTIME_INPUT: NAME}), selected("/recorded/python")])
    s = sources(monkeypatch, tmp_path, inputs={"model": "/m"}, ledger=led, environ={"ORCHARD_TEXT_PYTHON": "/env/text-python"})
    assert s.ref_python == "/recorded/python"


def test_a_run_that_is_not_on_the_text_route_keeps_the_reference_interpreter(monkeypatch, tmp_path):
    s = sources(monkeypatch, tmp_path, inputs={"model": "/m"}, ledger=Led(), environ={"ORCHARD_REFERENCE_PYTHON": "/env/ref", "ORCHARD_TEXT_PYTHON": "/env/text"})
    assert s.ref_python == "/env/ref"


def test_an_unregistered_runtime_name_is_not_the_text_route(monkeypatch, tmp_path):
    s = sources(monkeypatch, tmp_path, inputs={"model": "/m", text_runtime.RUNTIME_INPUT: "no-such-runtime"}, ledger=Led(),
                environ={"ORCHARD_REFERENCE_PYTHON": "/env/ref", "ORCHARD_TEXT_PYTHON": "/env/text"})
    assert s.ref_python == "/env/ref"


# ---- item 2: a resume is judged on the inputs the run started with ---------------------------------------------------------

def test_a_resume_compares_the_checkpoint_and_interpreter_the_run_started_with(monkeypatch, tmp_path):
    led = Led([run_start(model="/snapshot-A", reference_python="/ref-A/python")])
    s = sources(monkeypatch, tmp_path, inputs={"model": "/snapshot-B", "reference_python": "/ref-B/python"}, ledger=led, environ={})
    assert s.model_dir == "/snapshot-A" and s.ref_python == "/ref-A/python"


def test_a_new_run_with_no_run_start_uses_the_command_s_inputs(monkeypatch, tmp_path):
    s = sources(monkeypatch, tmp_path, inputs={"model": "/snapshot-B", "reference_python": "/ref-B/python"}, ledger=Led(), environ={})
    assert s.model_dir == "/snapshot-B" and s.ref_python == "/ref-B/python"


def test_an_unreadable_ledger_falls_back_to_the_command_s_inputs_and_never_raises(monkeypatch, tmp_path):
    s = sources(monkeypatch, tmp_path, inputs={"model": "/snapshot-B"}, ledger=Led(fail=True), environ={})
    assert s.model_dir == "/snapshot-B"


@pytest.mark.parametrize("bad", [None, "text", ["a"], 7])
def test_a_run_start_whose_inputs_are_not_a_mapping_is_ignored(monkeypatch, tmp_path, bad):
    led = Led([{"event": "run_start", "stage": None, "data": {"inputs": bad}}])
    s = sources(monkeypatch, tmp_path, inputs={"model": "/snapshot-B"}, ledger=led, environ={})
    assert s.model_dir == "/snapshot-B"
